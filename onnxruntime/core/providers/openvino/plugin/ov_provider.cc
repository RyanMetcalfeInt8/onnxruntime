// Copyright (C) Intel Corporation
// Licensed under the MIT License

#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <functional>

#include "ov_provider.h"
#include "ov_compute.h"
#include "ov_ep_context.h"
#include "../common/ov_supported_ops.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

#define ORT_EP_UTILS_ORT_GRAPH_TO_PROTO_IMPL
#include "core/providers/utils/ort_graph_to_proto.h"

using namespace onnxruntime::openvino_ep;

template <typename T>
static OrtStatus* GetSessionConfigEntryOrDefault(const OrtApi& ort_api, const OrtSessionOptions& session_options,
                                                 const char* config_key, T default_val, T& config_val) {
  int has_config = 0;
  OVEP_RETURN_IF_ERROR(ort_api.HasSessionConfigEntry(&session_options, config_key, &has_config));

  if (has_config != 1) {
    config_val = default_val;
    return nullptr;
  }

  size_t size = 0;
  OVEP_RETURN_IF_ERROR(ort_api.GetSessionConfigEntry(&session_options, config_key, nullptr, &size));

  std::string string_config_val(size, '\0');
  OVEP_RETURN_IF_ERROR(ort_api.GetSessionConfigEntry(&session_options, config_key, string_config_val.data(), &size));
  string_config_val.resize(size - 1);

  try {
    if constexpr (std::is_same_v<T, std::string>) {
      config_val = string_config_val;
    } else if constexpr (std::is_same_v<T, int>) {
      config_val = std::stoi(string_config_val);
    } else if constexpr (std::is_same_v<T, bool>) {
      config_val = string_config_val == "1";
    } else if constexpr (std::is_same_v<T, std::filesystem::path>) {
      config_val = std::filesystem::path(string_config_val);
    } else {
      static_assert(false, "unsupported type");
    }
  } catch (const std::exception& e) {
    return ort_api.CreateStatus(ORT_INVALID_ARGUMENT, e.what());
  }

  return nullptr;
}

OrtStatus* OpenVINOEpPluginOptions::Init(const OrtApi& ort_api, const OrtSessionOptions& session_options) {
  OVEP_RETURN_IF_ERROR(GetSessionConfigEntryOrDefault(ort_api, session_options, kOrtSessionOptionEpContextEnable, false, ep_ctx_.enable_));
  OVEP_RETURN_IF_ERROR(GetSessionConfigEntryOrDefault(ort_api, session_options, kOrtSessionOptionEpContextEmbedMode, false, ep_ctx_.embed_));
  OVEP_RETURN_IF_ERROR(GetSessionConfigEntryOrDefault(ort_api, session_options, kOrtSessionOptionShareEpContexts, false, ep_ctx_.share_));
  OVEP_RETURN_IF_ERROR(GetSessionConfigEntryOrDefault(ort_api, session_options, kOrtSessionOptionEpContextFilePath, std::filesystem::path(), ep_ctx_.path_));

  return nullptr;
}

// Implementation class definition
OpenVINOEpPlugin::OpenVINOEpPlugin(ApiPtrs apis, const std::string& name,
                                   const OpenVINOEpPluginOptions& options,
                                   const OrtLogger& logger,
                                   const std::string ov_device_type,
                                   std::shared_ptr<ov::Core> ov_core)
    : ApiPtrs(apis),
      name_(name),
      logger_(&logger),
      ov_device_type_(ov_device_type),
      ov_core_(ov_core),
      options_(options) {
  ort_version_supported = ORT_API_VERSION;  // set to the ORT version we were compiled with.

  OrtEp::GetName = GetNameImpl;
  OrtEp::GetCapability = GetCapabilityImpl;
  OrtEp::Compile = CompileImpl;
  OrtEp::ReleaseNodeComputeInfos = ReleaseNodeComputeInfosImpl;
}

OpenVINOEpPlugin::~OpenVINOEpPlugin() = default;

struct NodeView {
  std::string_view domain;
  std::string_view op_type;
  bool is_ep_context = false;

  OrtStatus* Init(const OrtApi& ort_api, const OrtNode* node_ptr) {
    const char* op_type_char = nullptr;
    OVEP_RETURN_IF_ERROR(ort_api.Node_GetOperatorType(node_ptr, &op_type_char));
    op_type = op_type_char;

    const char* domain_char = nullptr;
    OVEP_RETURN_IF_ERROR(ort_api.Node_GetDomain(node_ptr, &domain_char));

    // Use empty string if domain is null (default domain)
    domain = domain_char ? domain_char : "";

    is_ep_context = SupportedOps::Get().IsEpContextNode(op_type, domain);

    return nullptr;
  }
};

OrtStatus* OpenVINOEpPlugin::GetCapability(const OrtGraph* graph, OrtEpGraphSupportInfo* graph_support_info) {
  size_t num_nodes = 0;
  OVEP_RETURN_IF_ERROR(ort_api.Graph_GetNumNodes(graph, &num_nodes));

  std::vector<const OrtNode*> nodes(num_nodes);
  OVEP_RETURN_IF_ERROR(ort_api.Graph_GetNodes(graph, nodes.data(), nodes.size()));

  auto add_nodes_to_fuse = [this, graph_support_info](std::vector<const OrtNode*> nodes) -> OrtStatus* {
    OrtNodeFusionOptions options{};
    options.ort_version_supported = ORT_API_VERSION;
    options.drop_constant_initializers = true;
    return ep_api.EpGraphSupportInfo_AddNodesToFuse(graph_support_info, nodes.data(), nodes.size(), &options);
  };

  // Get supported operations instance
  const auto& supported_ops = SupportedOps::Get();

  // Check each node for support
  std::vector<const OrtNode*> supported_nodes;
  for (const auto& node_ptr : nodes) {
    NodeView node_info;
    OVEP_RETURN_IF_ERROR(node_info.Init(ort_api, node_ptr));

    // Check if this operation is supported
    const bool is_ep_context_node = node_info.is_ep_context;
    if (supported_ops.IsOpSupported(node_info.op_type, node_info.domain) && !is_ep_context_node) {
      supported_nodes.push_back(node_ptr);
    } else {
      // Fuse all the supported nodes collected so far
      if (!supported_nodes.empty()) {
        OVEP_RETURN_IF_ERROR(add_nodes_to_fuse(supported_nodes));
      }
      supported_nodes.clear();

      if (is_ep_context_node) {
        OVEP_RETURN_IF_ERROR(add_nodes_to_fuse(std::vector<const OrtNode*>{node_ptr}));
      }
    }
  }

  if (!supported_nodes.empty()) {
    OVEP_RETURN_IF_ERROR(add_nodes_to_fuse(supported_nodes));
  }

  return nullptr;
}

OrtStatus* OpenVINOEpPlugin::Compile(const OrtGraph** graphs, const OrtNode** fused_nodes,
                                     size_t count, OrtNodeComputeInfo** node_compute_infos, OrtNode** ep_context_nodes) {
  // Process all graphs
  for (size_t i = 0; i < count; ++i) {
    const OrtGraph* graph = graphs[i];
    const OrtNode* fused_node = fused_nodes[i];
    if (graph == nullptr) {
      return ort_api.CreateStatus(ORT_INVALID_ARGUMENT, "Graph is null");
    }

    size_t num_nodes = 0;
    OVEP_RETURN_IF_ERROR(ort_api.Graph_GetNumNodes(graph, &num_nodes));

    std::vector<const OrtNode*> nodes(num_nodes);
    OVEP_RETURN_IF_ERROR(ort_api.Graph_GetNodes(graph, nodes.data(), nodes.size()));

    auto ov_compute = std::make_unique<OvComputeInfo>(*this, *ov_core_);

    OnnxIOMapping io_mapping;
    OVEP_RETURN_IF_ERROR(io_mapping.Init(ort_api, *graph));

    auto try_export_ep_context = [&](OvComputeInfo& ov_compute, OnnxIOMapping io_mapping) -> OrtStatus* {
      if (!options_.ep_ctx_.enable_) {
        return nullptr;
      }

      const char* fused_node_name = nullptr;
      OVEP_RETURN_IF_ERROR(ort_api.Node_GetName(fused_node, &fused_node_name));

      EpContextNode node(*this);
      node.embed_mode = options_.ep_ctx_.embed_ ? 1u : 0u;
      node.partition_name = fused_node_name;
      node.ep_cache_context = std::string(fused_node_name) + ".blob";

      std::filesystem::path containing_dir{options_.ep_ctx_.path_.parent_path()};
      if (containing_dir.empty()) {
        const ORTCHAR_T* model_path = nullptr;
        OVEP_RETURN_IF_ERROR(ort_api.Graph_GetModelPath(graph, &model_path));
        containing_dir = model_path;
      }
      node.private_fields_.model_dir = containing_dir;

      node.private_fields_.node_name = fused_node_name;
      node.main_context = 1;  // We always return a single fused node.
      // TODO(ericcraw) Figure out what to put here.
      // node.ep_sdk_version =

      OVEP_RETURN_IF_ERROR(ov_compute.Export(node));
      OVEP_RETURN_IF_ERROR(node.CreateNode(io_mapping, ep_context_nodes[i]));
      return nullptr;
    };

    if (num_nodes == 1) {
      // Only supporting single ep context node per graph for now
      NodeView node_info;
      const OrtNode* node = nodes[0];
      OVEP_RETURN_IF_ERROR(node_info.Init(ort_api, node));
      if (node_info.is_ep_context) {
        EpContextNode ep_context_node(*this);

        const ORTCHAR_T* model_path = nullptr;
        OVEP_RETURN_IF_ERROR(ort_api.Graph_GetModelPath(graph, &model_path));
        OVEP_RETURN_IF_ERROR(ep_context_node.Init(node, model_path));

        OVEP_RETURN_IF_ERROR(ov_compute->Init(ov_device_type_, io_mapping, std::move(ep_context_node)));
        OVEP_RETURN_IF_ERROR(try_export_ep_context(*ov_compute, std::move(io_mapping)));
        node_compute_infos[i] = ov_compute.release();
        continue;
      }
    }

    std::unique_ptr<onnx::ModelProto> model_proto = std::make_unique<onnx::ModelProto>();
    OVEP_RETURN_IF_ERROR(OrtEpUtils::OrtGraphToProto(*graph, *model_proto));
    OVEP_RETURN_IF_ERROR(ov_compute->Init(ov_device_type_, io_mapping, std::move(model_proto)));
    OVEP_RETURN_IF_ERROR(try_export_ep_context(*ov_compute, std::move(io_mapping)));

    node_compute_infos[i] = ov_compute.release();
  }

  return nullptr;
}

void OpenVINOEpPlugin::ReleaseNodeComputeInfos(OrtNodeComputeInfo** node_compute_infos,
                                               size_t num_node_compute_infos) {
  // Clean up any compute info objects
  for (size_t i = 0; i < num_node_compute_infos; i++) {
    delete static_cast<OvComputeInfo*>(node_compute_infos[i]);
  }
}
