// Copyright (C) Intel Corporation
// Licensed under the MIT License

#pragma once

#include <string>
#include <cstdint>

#include "ov_provider.h"

template <typename T>
struct DeferOrtRelease {
  DeferOrtRelease(T** object_ptr, std::function<void(T*)> release_func)
      : objects_(object_ptr), count_(1), release_func_(release_func) {}

  DeferOrtRelease(T** objects, size_t count, std::function<void(T*)> release_func)
      : objects_(objects), count_(count), release_func_(release_func) {}

  ~DeferOrtRelease() {
    if (objects_ != nullptr && count_ > 0) {
      for (size_t i = 0; i < count_; ++i) {
        if (objects_[i] != nullptr) {
          release_func_(objects_[i]);
          objects_[i] = nullptr;
        }
      }
    }
  }
  T** objects_ = nullptr;
  size_t count_ = 0;
  std::function<void(T*)> release_func_ = nullptr;
};

namespace onnxruntime {
namespace openvino_ep {

struct EpContextNode : ApiPtrs {
  size_t num_nodes{0};
  int64_t main_context{1};
  std::string ep_cache_context;
  int64_t embed_mode{1};
  std::string ep_sdk_version;
  std::string onnx_model_filename;
  std::string hardware_architecture;
  std::string partition_name;
  std::string source;
  std::string notes;
  int64_t max_size{0};

  enum class EpContextType {
    Native,
    OV_IR,
  };

  struct private_fields {
    EpContextType type{EpContextType::Native};
    std::string node_name{"OpenVINO_EP_Node"};
  } private_fields_;

  EpContextNode(ApiPtrs apis) : ApiPtrs(apis) {}
  EpContextNode(ApiPtrs apis, const OrtNode* node)
      : ApiPtrs(apis) {
    // Helper lambda to extract attribute values

    auto get_attr_int64 = [&](const char* name, int64_t default_val) -> int64_t {
      int64_t val = default_val;
      const OrtOpAttr* attr = nullptr;
      const OrtOpAttrType type = ORT_OP_ATTR_INT;

      OrtStatus* status = ort_api.Node_GetAttributeByName(node, name, &attr);
      if (status) {
        ort_api.ReleaseStatus(status);
        return val;
      }

      size_t size_read = 0;
      status = ort_api.ReadOpAttr(attr, type, &val, sizeof(val), &size_read);
      if (status || size_read != sizeof(val)) {
        ort_api.ReleaseStatus(status);
        return val;
      }

      return val;
    };

    auto get_attr_string = [&](const char* name) -> std::string {
      std::string val{};
      const OrtOpAttr* attr = nullptr;
      const OrtOpAttrType type = ORT_OP_ATTR_STRING;

      OrtStatus* status = ort_api.Node_GetAttributeByName(node, name, &attr);
      if (status) {
        ort_api.ReleaseStatus(status);
        return val;
      }
      size_t required_size = 0;
      status = ort_api.ReadOpAttr(attr, type, nullptr, 0, &required_size);
      if (status) {
        // expect it to fail
        ort_api.ReleaseStatus(status);
      }

      val.resize(required_size);
      status = ort_api.ReadOpAttr(attr, type, val.data(), val.size(), &required_size);
      if (status) {
        ort_api.ReleaseStatus(status);
        return val;
      }

      return val;
    };

    main_context = get_attr_int64("main_context", 1);
    ep_cache_context = get_attr_string("ep_cache_context");
    embed_mode = get_attr_int64("embed_mode", 1);
    ep_sdk_version = get_attr_string("ep_sdk_version");
    onnx_model_filename = get_attr_string("onnx_model_filename");
    hardware_architecture = get_attr_string("hardware_architecture");
    partition_name = get_attr_string("partition_name");
    source = get_attr_string("source");
    notes = get_attr_string("notes");
    max_size = get_attr_int64("max_size", 0);
    private_fields_.type = EpContextType::Native;
    private_fields_.node_name = "OpenVINO_EP_Node";
  }

  OrtStatus* CreateNode(const OnnxIOMapping& io_map, OrtNode*& node) {
    std::array<OrtOpAttr*, 6> attributes = {};
    DeferOrtRelease<OrtOpAttr> defer_release_attrs(attributes.data(), attributes.size(), ort_api.ReleaseOpAttr);

    RETURN_IF_ERROR(ort_api.CreateOpAttr("ep_cache_context", ep_cache_context.c_str(), 1, ORT_OP_ATTR_STRING, &attributes[0]));
    RETURN_IF_ERROR(ort_api.CreateOpAttr("main_context", &main_context, 1, ORT_OP_ATTR_INT, &attributes[1]));
    RETURN_IF_ERROR(ort_api.CreateOpAttr("embed_mode", &embed_mode, 1, ORT_OP_ATTR_INT, &attributes[2]));
    RETURN_IF_ERROR(ort_api.CreateOpAttr("ep_sdk_version", ep_sdk_version.c_str(), 1, ORT_OP_ATTR_STRING, &attributes[3]));
    RETURN_IF_ERROR(ort_api.CreateOpAttr("partition_name", partition_name.c_str(), 1, ORT_OP_ATTR_STRING, &attributes[4]));
    RETURN_IF_ERROR(ort_api.CreateOpAttr("source", source.c_str(), 1, ORT_OP_ATTR_STRING, &attributes[5]));

    // Prepare input and output names
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    for (const auto& input : io_map.input_names) {
      input_names.push_back(input.c_str());
    }
    for (const auto& output : io_map.output_names) {
      output_names.push_back(output.c_str());
    }

    // Create the node
    OrtStatus* status = model_editor_api.CreateNode(
        "EPContext",
        "com.microsoft",
        private_fields_.node_name.c_str(),
        input_names.data(),
        input_names.size(),
        output_names.data(),
        output_names.size(),
        attributes.data(),
        attributes.size(),
        &node);

    return status;
  }
};

}  // namespace openvino_ep
}  // namespace onnxruntime
