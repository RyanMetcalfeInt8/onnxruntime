// Copyright (C) Intel Corporation
// Licensed under the MIT License

#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <functional>
#include <shared_mutex>
#include <algorithm>
#include <cctype>
#include <optional>
#include <sstream>

#include "ov_provider.h"
#include "ov_compute.h"
#include "ov_compute_stateful.h"
#include "ov_ep_context.h"
#include "common/weak_singleton.h"
#include "common/ov_supported_ops.h"
#include "ov_plugin_utils.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/common/common.h"
#include "transformations/ov_weights_as_input.h"

#define ORT_EP_UTILS_ORT_GRAPH_TO_PROTO_IMPL
#include "core/providers/utils/ort_graph_to_proto.h"

using namespace onnxruntime::openvino_ep_plugin;

using shared_ctx_singleton = onnxruntime::openvino_ep::WeakSingleton<SharedContext>;

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

OrtStatus* OpenVINOEpPluginOptions::Init(const OrtApi& ort_api, const Ort::Logger& logger, const OrtSessionOptions& session_options, const std::string& ep_name) {
  // Parse provider-specific options
  OVEP_RETURN_IF_ERROR(ParseProviderOptions(ort_api, logger, session_options, ep_name));

  // Initialize EP context configuration
  OVEP_RETURN_IF_ERROR(ParseEpContextOptions(ort_api, session_options));

  return nullptr;
}

OrtStatus* OpenVINOEpPluginOptions::ParseProviderOptions(const OrtApi& ort_api, const Ort::Logger& logger, const OrtSessionOptions& session_options, const std::string& ep_name) {
  std::string target_ep_name = ep_name;
  std::transform(target_ep_name.begin(), target_ep_name.end(), target_ep_name.begin(), [](unsigned char c) { return std::tolower(c); });
  std::string provider_prefix = "ep." + target_ep_name + ".";

  // Define valid keys for the OpenVINO EP provider options
  const auto& valid_keys = OpenVINOEpProviderOptions::GetValidProviderKeys();

  // Helper to get option value from session config
  auto get_option_value = [&](std::string_view option_name) -> std::optional<std::string> {
    const std::string config_key = std::string(provider_prefix) + std::string(option_name);

    std::string value;
    if (GetSessionConfigEntryOrDefault(ort_api, session_options, config_key.c_str(), std::string(), value) == nullptr && !value.empty()) {
      return value;
    }

    return std::nullopt;
  };

  // Helper to parse boolean values with case-insensitive comparison
  auto parse_bool = [](std::string_view value) -> std::optional<bool> {
    std::string lower_value;
    lower_value.resize(value.size());
    std::transform(value.begin(), value.end(), lower_value.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    if (lower_value == "true" || lower_value == "1") {
      return true;
    } else if (lower_value == "false" || lower_value == "0") {
      return false;
    }
    return std::nullopt;
  };

  // Process each valid option if provided
  for (const auto& key : valid_keys) {
    auto value_opt = get_option_value(key);
    if (!value_opt) continue;  // Option not provided

    const std::string& value = *value_opt;

    if (key == "enable_causallm") {
      if (auto bool_val = parse_bool(value)) {
        provider_options_.enable_causallm_ = *bool_val;
      } else {
        const std::string error_msg =
            "Invalid boolean value '" + value + "' for option '" + key +
            "'. Valid values are: 'true', 'false', '1', '0' (case insensitive)";
        return ort_api.CreateStatus(ORT_INVALID_ARGUMENT, error_msg.c_str());
      }
    } else if (key == "load_config") {
      // Parse load_config using the plugin-specific utility function
      OVEP_RETURN_IF_ERROR(ParsePluginLoadConfigOption(ort_api, logger, value, provider_options_.load_config_));
    } else if (key == "reshape_input") {
      // Parse reshape_input option using the plugin-specific utility function
      OVEP_RETURN_IF_ERROR(ParsePluginReshapeInputOption(ort_api, logger, value, provider_options_.reshape_input_));
    }
  }

  return nullptr;
}

OrtStatus* OpenVINOEpPluginOptions::ParseEpContextOptions(const OrtApi& ort_api, const OrtSessionOptions& session_options) {
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
      shared_context_(shared_ctx_singleton::Get()),
      options_(options) {
  ort_version_supported = ORT_API_VERSION;  // set to the ORT version we were compiled with.

  OrtEp::GetName = GetNameImpl;
  OrtEp::GetCapability = GetCapabilityImpl;
  OrtEp::Compile = CompileImpl;
  OrtEp::ReleaseNodeComputeInfos = ReleaseNodeComputeInfosImpl;
  OrtEp::SetDynamicOptions = SetDynamicOptionsImpl;
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

static OrtStatus* AddSharableWeights(const OrtGraph& graph_ref, SharedContext& shared_context) {
  const OrtApi& ort_api = Ort::GetApi();
  size_t num_initializers = 0;
  const OrtGraph* graph = &graph_ref;
  OVEP_RETURN_IF_ERROR(ort_api.Graph_GetNumInitializers(graph, &num_initializers));

  std::vector<const OrtValueInfo*> initializers(num_initializers);
  OVEP_RETURN_IF_ERROR(ort_api.Graph_GetInitializers(graph, initializers.data(), initializers.size()));

  for (const auto& initializer : initializers) {
    const char* initializer_name = nullptr;
    OVEP_RETURN_IF_ERROR(ort_api.GetValueInfoName(initializer, &initializer_name));

    // Get external initializer information if available
    OrtExternalInitializerInfo* ext_info = nullptr;
    DeferOrtRelease<OrtExternalInitializerInfo> release_ext(&ext_info, ort_api.ReleaseExternalInitializerInfo);
    OVEP_RETURN_IF_ERROR(ort_api.ValueInfo_GetExternalInitializerInfo(initializer, &ext_info));

    if (ext_info != nullptr) {
      // Extract external weight parameters
      const ORTCHAR_T* file_path = ort_api.ExternalInitializerInfo_GetFilePath(ext_info);
      int64_t offset = ort_api.ExternalInitializerInfo_GetFileOffset(ext_info);
      size_t size = ort_api.ExternalInitializerInfo_GetByteSize(ext_info);

      shared_context.shared_weights.AddExternalWeight(
          std::string(initializer_name),
          static_cast<size_t>(offset),
          size,
          file_path);
    }
  }

  return nullptr;
}

std::vector<ModelTransformation> OpenVINOEpPlugin::BuildTransformationPipeline(std::filesystem::path model_dir) {
  std::vector<ModelTransformation> transformations;

  if (options_.ep_ctx_.share_) {
    ov::RemoteContext remote_context;
    try {
      remote_context = ov_core_->get_default_context(ov_device_type_);
    } catch (const ov::Exception&) {
      remote_context = ov::RemoteContext();
    }

    transformations.emplace_back(
        /* transformation function */
        [&, remote_context = std::move(remote_context)](std::shared_ptr<ov::Model>& model) { return weights_as_inputs::TransformSharedWeightsToInpus(model, shared_context_->shared_weights); },
        /* optional initializer */
        [&, remote_context, model_dir](ov::InferRequest& ir) {
          shared_context_->shared_weights.SetSharedWeightsOnInferRequest(ir, remote_context, model_dir);
        });
  }

  return transformations;
}

OrtStatus* OpenVINOEpPlugin::SetDynamicOptions(const char* const* option_keys, const char* const* option_values, size_t num_options) {
  // Process all options
  for (size_t i = 0; i < num_options; i++) {
    const std::string key(option_keys[i]);
    const std::string value(option_values[i]);
    if (key == kOrtEpDynamicOptionsWorkloadType) {
      std::string workload_type ;
      if (value == "Efficient") {
        workload_type = "EFFICIENT";
      } else if (value == "Default") {
        workload_type = "DEFAULT";
      } else {
        ORT_CXX_LOGF(logger_, ORT_LOGGING_LEVEL_WARNING, "Unknown workload_type - ignoring %s / %s", key, value);
        ORT_CXX_LOGF(logger_, ORT_LOGGING_LEVEL_WARNING, "Supported types are 'Efficient' and 'Default'");
      }
      if (!workload_type.empty()) {
        for (auto& ov_compute : computes_) {
          OVEP_RETURN_IF_ERROR(ov_compute->SetWorkloadType(workload_type));
        }
      }
    } else if (key == "kvcache_rewind") {
      int64_t index;
      try {
        index = std::stoll(value);
      } catch (const std::exception& e) {
        ORT_CXX_LOGF(logger_, ORT_LOGGING_LEVEL_WARNING,
                     "Conversion for kvcache_rewind string value to int64_t index failed. Exception: %s", e.what());
        return nullptr;
      }
      for (auto& ov_compute : computes_) {
        OVEP_RETURN_IF_ERROR(ov_compute->KVCacheRewind(static_cast<size_t>(index)));
      }
    }
  }

  return nullptr;
}

OrtStatus* OpenVINOEpPlugin::Compile(const OrtGraph** graphs, const OrtNode** fused_nodes,
                                     size_t count, OrtNodeComputeInfo** node_compute_infos, OrtNode** ep_context_nodes) {
  // Clear our computes cache.
  computes_.clear();

  // Process all graphs
  for (size_t i = 0; i < count; ++i) {
    const OrtGraph* graph = graphs[i];
    const OrtNode* fused_node = fused_nodes[i];

    size_t num_nodes = 0;
    OVEP_RETURN_IF_ERROR(ort_api.Graph_GetNumNodes(graph, &num_nodes));

    std::vector<const OrtNode*> nodes(num_nodes);
    OVEP_RETURN_IF_ERROR(ort_api.Graph_GetNodes(graph, nodes.data(), nodes.size()));

    auto ov_compute = options_.provider_options_.enable_causallm_ ? std::make_unique<OvComputeInfoStateful>(*this, *ov_core_, logger_) :
                                                                    std::make_unique<OvComputeInfo>(*this, *ov_core_, logger_);

    OnnxIOMapping io_mapping;
    OVEP_RETURN_IF_ERROR(io_mapping.Init(ort_api, *graph));

    std::filesystem::path model_path;
    const ORTCHAR_T* char_model_path = nullptr;
    OVEP_RETURN_IF_ERROR(ort_api.Graph_GetModelPath(graph, &char_model_path));
    model_path = std::filesystem::path(char_model_path);
    std::filesystem::path model_dir = model_path.parent_path();

    // Output path is the EP context path if specified -- otherwise the same as model path.
    std::filesystem::path output_directory{options_.ep_ctx_.path_.parent_path()};
    std::filesystem::path output_model_path{options_.ep_ctx_.path_};
    if (options_.ep_ctx_.path_.empty()) {
      output_directory = model_dir;
      output_model_path = model_path;
    }

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

      node.private_fields_.model_dir = output_directory;
      node.private_fields_.node_name = fused_node_name;

      node.main_context = 1;  // We always return a single fused node.
      // TODO(ericcraw) Figure out what to put here.
      // node.ep_sdk_version =
      node.onnx_model_filename = output_model_path.filename().string();

      OVEP_RETURN_IF_ERROR(ov_compute.Export(node));
      OVEP_RETURN_IF_ERROR(node.CreateNode(io_mapping, ep_context_nodes[i]));
      if (options_.ep_ctx_.share_) {
        OVEP_RETURN_IF_ERROR(shared_context_->shared_weights.SaveMetaData(node.private_fields_.model_dir, node.onnx_model_filename));
      }

      return nullptr;
    };

    if (num_nodes == 1) {
      // Only supporting single ep context node per graph for now
      NodeView node_info;
      const OrtNode* node = nodes[0];
      OVEP_RETURN_IF_ERROR(node_info.Init(ort_api, node));
      if (node_info.is_ep_context) {
        EpContextNode ep_context_node(*this);
        OVEP_RETURN_IF_ERROR(ep_context_node.Init(node, model_path));

        if (options_.ep_ctx_.share_) {
          OVEP_RETURN_IF_ERROR(shared_context_->shared_weights.LoadMetaData(ep_context_node.private_fields_.model_dir, ep_context_node.onnx_model_filename));
        }

        OVEP_RETURN_IF_ERROR(ov_compute->Init(ov_device_type_, options_.provider_options_.GetDeviceConfig(ov_device_type_), io_mapping, std::move(ep_context_node), BuildTransformationPipeline(model_dir)));
        OVEP_RETURN_IF_ERROR(try_export_ep_context(*ov_compute, std::move(io_mapping)));
        computes_.push_back(ov_compute.release());
        node_compute_infos[i] = computes_.back();
        continue;
      }
    }

    if (options_.ep_ctx_.share_) {
      AddSharableWeights(*graph, *shared_context_);
    }

    std::unique_ptr<onnx::ModelProto> model_proto = std::make_unique<onnx::ModelProto>();
    OVEP_RETURN_IF_ERROR(OrtEpUtils::OrtGraphToProto(*graph, *model_proto));
    OVEP_RETURN_IF_ERROR(ov_compute->Init(ov_device_type_, options_.provider_options_.GetDeviceConfig(ov_device_type_), io_mapping, std::move(model_proto), BuildTransformationPipeline(model_dir)));
    OVEP_RETURN_IF_ERROR(try_export_ep_context(*ov_compute, std::move(io_mapping)));

    computes_.push_back(ov_compute.release());
    node_compute_infos[i] = computes_.back();
  }

  return nullptr;
}

void OpenVINOEpPlugin::ReleaseNodeComputeInfos(OrtNodeComputeInfo** node_compute_infos,
                                               size_t num_node_compute_infos) {
  // Clean up any compute info objects
  for (size_t i = 0; i < num_node_compute_infos; i++) {
    delete static_cast<OvComputeInfo*>(node_compute_infos[i]);
  }

  computes_.clear();
}
