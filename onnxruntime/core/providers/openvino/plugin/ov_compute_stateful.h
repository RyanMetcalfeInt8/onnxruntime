// Copyright (C) Intel Corporation
// Licensed under the MIT License

# pragma once

#include "ov_compute.h"

namespace onnxruntime {
namespace openvino_ep_plugin {

struct OvComputeInfoStateful : OvComputeInfo {
  OvComputeInfoStateful(ApiPtrs apis, ov::Core& ov_core, const Ort::Logger& logger);

  OrtStatus* Init(const std::string& ov_device, const ov::AnyMap& configs, const OnnxIOMapping&, EpContextNode ep_context_node, std::vector<ModelTransformation> transformations = {}) override;
  OrtStatus* Init(const std::string& ov_device, const ov::AnyMap& configs, const OnnxIOMapping&, std::unique_ptr<onnx::ModelProto> model_proto, std::vector<ModelTransformation> transformations = {}) override;

  OrtStatus* Compute(void* compute_state,
                     OrtKernelContext* kernel_context) override;

  OrtStatus* KVCacheRewind(const size_t& index) override;

 private:
  ov::CompiledModel stateful_compile_ir_(std::shared_ptr<ov::Model> model, const ov::AnyMap& device_config);
  OrtStatus* pre_infer_();
  ov::InferRequest infer_request_;
  std::unique_ptr<WrappedInferRequest> _infer_request;
  std::string _ov_device;

  // If prefill_use_full_chat_history is true, cache the "input_ids" & "position_ids" tensors,
  // and ensure that full chat history is passed for each prefill call.
  bool prefill_use_full_chat_history_ = false;
  std::vector<int64_t> cached_input_ids_;
  std::vector<int64_t> cached_position_ids_;
};

}  // namespace openvino_ep
}  // namespace onnxruntime