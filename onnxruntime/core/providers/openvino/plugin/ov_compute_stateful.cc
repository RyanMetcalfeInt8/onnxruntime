// Copyright (C) Intel Corporation
// Licensed under the MIT License

#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>

#include "ov_compute_stateful.h"
#include "ov_ep_context.h"
#include "../ov_stateful_patch_utils.h"

using namespace onnxruntime::openvino_ep;

namespace onnxruntime {
namespace openvino_ep_plugin {

OvComputeInfoStateful::OvComputeInfoStateful(ApiPtrs apis, ov::Core& ov_core) : OvComputeInfo(apis, ov_core) {}

OrtStatus* OvComputeInfoStateful::Init(const std::string& ov_device, const ov::AnyMap& configs, const OnnxIOMapping& io_mapping, EpContextNode ep_context_node) {
  _ov_device = ov_device;

  switch (ep_context_node.private_fields_.type) {
    case EpContextNode::EpContextType::Native:
      if (ep_context_node.embed_mode != 0) {
        std::istringstream model_stream(std::move(ep_context_node.ep_cache_context));
        compiled_model_ = ov_core_.import_model(model_stream, ov_device, configs);
      } else {
        const std::filesystem::path ep_ctx_path = ep_context_node.private_fields_.model_dir / ep_context_node.ep_cache_context;
        std::ifstream model_stream(ep_ctx_path, std::ios_base::binary | std::ios_base::in);
        compiled_model_ = ov_core_.import_model(model_stream, ov_device, configs);
      }
      break;
    case EpContextNode::EpContextType::OV_IR:
      // Outstanding open to MSFT: How do I get the path to the graph? Graph API doesn't have the source model path.
      if (ep_context_node.embed_mode != 0) {
        // To support this we would need to save the weights location somewhere, or have a schema for embedding the weights along with the IR.
        return Ort::Status("Epctx with OVIR must use embed_mode == 0", ORT_INVALID_ARGUMENT).release();
      } else {
        const std::filesystem::path ep_ctx_path = ep_context_node.private_fields_.model_dir / ep_context_node.ep_cache_context;
        auto model = ov_core_.read_model(ep_ctx_path);
        compiled_model_ = stateful_compile_ir_(model, configs);
      }
      break;
    default:
      return Ort::Status("Unsupported EpContextType", ORT_INVALID_ARGUMENT).release();
  }

  _infer_request = std::make_unique<WrappedInferRequest>(std::move(compiled_model_.create_infer_request()));
  onnx_to_ov_bindings_ = std::make_unique<OnnxToOvNetworkBindings>(compiled_model_, io_mapping, SessionContext{true});

  bool gpu_or_npu = ((_ov_device.find("NPU") != std::string::npos) || (_ov_device.find("GPU") != std::string::npos));
  if (gpu_or_npu) {
    prefill_use_full_chat_history_ = true;
  }

  return nullptr;
}

OrtStatus* OvComputeInfoStateful::Init(const std::string& ov_device, const ov::AnyMap& configs, const OnnxIOMapping& io_mapping, std::unique_ptr<onnx::ModelProto> model_proto) {
  _ov_device = ov_device;

  std::string model = model_proto->SerializeAsString();
  model_proto.reset();

  auto ov_model = ov_core_.read_model(std::move(model), ov::Tensor{});
  compiled_model_ = stateful_compile_ir_(ov_model, configs);

  _infer_request = std::make_unique<WrappedInferRequest>(std::move(compiled_model_.create_infer_request()));
  onnx_to_ov_bindings_ = std::make_unique<OnnxToOvNetworkBindings>(compiled_model_, io_mapping, SessionContext{true});

  bool gpu_or_npu = ((_ov_device.find("NPU") != std::string::npos) || (_ov_device.find("GPU") != std::string::npos));
  if (gpu_or_npu) {
    prefill_use_full_chat_history_ = true;
  }

  return nullptr;
}

ov::CompiledModel OvComputeInfoStateful::stateful_compile_ir_(std::shared_ptr<ov::Model> model, const ov::AnyMap& device_config) {
  ov::CompiledModel compiled_model;
  ov::AnyMap config = device_config;

  std::cout << "Model input to stateful_compile_ir_:" << std::endl;
  LogBasicModelInfo(model);

  bool model_status = IsStateful(model);
  std::cout << "Model IsStateful() Status:\t" << (model_status ? "True" : "False") << std::endl;

  if (!model_status) {
    std::cout << "Converting from Stateless OV Model to Stateful OV Model" << std::endl;
    PatchStatefulDecoder(model);

    std::cout << "Model after stateful transformation:" << std::endl;
    LogBasicModelInfo(model);
  }

  auto kv_pos = GetKVAxesPos(model);
  if (_ov_device.find("NPU") != std::string::npos) {
    KVDesc kv_desc;
    auto parse_genai_config = [&](const std::string& key, unsigned int default_value) {
      return (config.count(key) && !config.at(key).empty() && config.at(key).as<std::string>() != "0") ? config.at(key).as<unsigned int>() : default_value;
    };

    kv_desc.max_prompt_len = parse_genai_config("MAX_PROMPT_LEN", CausalLMConfig().max_prompt_len);
    kv_desc.min_response_len = parse_genai_config("MIN_RESPONSE_LEN", CausalLMConfig().min_response_len);

    // For compilation, MAX_PROMPT_LEN & MIN_RESPONSE_LEN should not be 0
    if (kv_desc.max_prompt_len == 0 || kv_desc.min_response_len == 0) {
      throw std::runtime_error("MAX_PROMPT_LEN or MIN_RESPONSE_LEN cannot be 0");
    }

    std::cout << "kv_pos.batch = " << kv_pos.batch << std::endl;
    std::cout << "kv_pos.seq_len = " << kv_pos.seq_len << std::endl;
    std::cout << "kv_desc.max_prompt_len:\t" << kv_desc.max_prompt_len << std::endl;
    std::cout << "kv_desc.min_response_len:\t" << kv_desc.min_response_len << std::endl;

    UpdateNPUConfig(config, kv_pos, kv_desc);
  } else {
    // This patches the OV IR model so that it only produces the logits required for sampling.
    // It's not needed for NPU, as it is already internally applied inside NPU compilation.
    ApplySliceBeforeMatmulTransformation(model);
  }

  compiled_model = ov_core_.compile_model(model, _ov_device, config);

  return compiled_model;
}

static inline void FillTensor(ov::InferRequest infer_request, const std::string& tensor_name, const ov::element::Type& type,
                       const std::vector<size_t>& shape, int32_t fill_value) {
  ov::Tensor tensor = ov::Tensor(type, shape);
  std::fill_n(tensor.data<int32_t>(), tensor.get_size(), fill_value);
  infer_request.set_tensor(tensor_name, tensor);
}

static inline void CacheTensor(ov::InferRequest infer_request, const std::string& tensor_name, std::vector<int64_t>& cache) {
  auto tensor = infer_request.get_tensor(tensor_name);
  auto* pData = tensor.data<int64_t>();
  for (size_t i = 0; i < tensor.get_size(); i++) {
    cache.emplace_back(pData[i]);
  }
}

static inline void SetTensorFromCache(ov::InferRequest infer_request, const std::string& tensor_name,
                               const std::vector<int64_t>& cache_data) {
  auto tensor = infer_request.get_tensor(tensor_name);
  auto new_shape = tensor.get_shape();
  new_shape[1] = cache_data.size();

  auto new_tensor = ov::Tensor(tensor.get_element_type(), new_shape);
  auto* pNewData = new_tensor.data<int64_t>();
  std::memcpy(pNewData, cache_data.data(), cache_data.size() * sizeof(int64_t));

  infer_request.set_tensor(tensor_name, new_tensor);
}

static inline bool HasTensor(ov::InferRequest infer_request, const std::string& tensor_name) {
  // Check if tensor exists by examining input names in the compiled model
  const auto& model = infer_request.get_compiled_model();
  for (const auto& input : model.inputs()) {
    const auto& names = input.get_names();
    if (names.find(tensor_name) != names.end()) {
      return true;
    }
  }
  return false;
}

OrtStatus* OvComputeInfoStateful::pre_infer_()
{
  auto &infer_request = _infer_request->ov();

  // Workaround: Setting the value here as it cannot be set at the ORT GenAI layer currently.
  // TODO(ankit): Address this issue and implement the fix at the appropriate layer.
  FillTensor(infer_request , "beam_idx", ov::element::i32, {1}, 0);

  // If 'prefill use full chat history' mode is enabled, we need to cache input_ids and position_ids.
  if (prefill_use_full_chat_history_) {
    auto input_ids_tensor = infer_request.get_tensor("input_ids");
    CacheTensor(infer_request , "input_ids", cached_input_ids_);

    // "position_ids" (GQA with Rotary Embeddings doesnt have position_ids) - check if exists
    auto has_position_ids = HasTensor(infer_request, "position_ids");
    if (has_position_ids) {
      CacheTensor(infer_request , "position_ids", cached_position_ids_);
    }

    // If we're about to run the prefill model
    if (input_ids_tensor.get_size() > 1) {
      // Check if the size of the current "input_ids" tensor does not match the size of the cached "input_ids".
      // This indicates that we are running a subsequent prompt (not the initial prefill).
      if (input_ids_tensor.get_shape()[1] != cached_input_ids_.size()) {
        // Clear the internal KVCache state. For NPU device, this operation is a no-op.
        infer_request.reset_state();

        // Set tensors using cached values
        SetTensorFromCache(infer_request , "input_ids", cached_input_ids_);

        // Only override position_ids if it exists and we have cached values
        if (has_position_ids && !cached_position_ids_.empty()) {
          SetTensorFromCache(infer_request , "position_ids", cached_position_ids_);
        }
      }
    }
  }

  return nullptr;
}

OrtStatus* OvComputeInfoStateful::Compute(void* /*compute_state*/,
                                  OrtKernelContext* kernel_context) {
  Ort::KernelContext context(kernel_context);

  if (!onnx_to_ov_bindings_->has_dynamic_io_) {
    return Ort::Status("Stateful Compute requires dynamic IO", ORT_RUNTIME_EXCEPTION).release();
  }

  // Dynamic shape inference

  // We don't know the output shapes so we need to get the outputs from the infer request and copy them into the ort
  // tensors instead of binding them to the infer request directly.

  // Bind inputs
  for (const auto& input_info : onnx_to_ov_bindings_->network_inputs_) {
    // Set the input shape based on the input tensor from ort
    auto tensor = context.GetInput(input_info.onnx_index);
    auto ort_shape = tensor.GetTensorTypeAndShapeInfo().GetShape();
    auto input_shape = ParameterShape(ort_shape);

    _infer_request->SetTensor(input_info.name,
                              input_info.type,
                              input_shape,
                              const_cast<void*>(tensor.GetTensorRawData()));
  }

  // pre-inference
  OVEP_RETURN_IF_ERROR(pre_infer_());

  // Run Inference
  _infer_request->Infer();

  // Copy outputs
  for (const auto& output_info : onnx_to_ov_bindings_->network_outputs_) {
    auto ov_tensor = _infer_request->ov().get_tensor(output_info.name);
    auto output_shape = ParameterShape::ToOrtShape(ov_tensor.get_shape());
    auto ort_tensor = context.GetOutput(output_info.onnx_index, output_shape);
    OVEP_RETURN_IF(ov_tensor.get_byte_size() != ort_tensor.GetTensorSizeInBytes(),
                   ort_api,
                   std::format("Output tensor size mismatch for {}", output_info.name).c_str());

    std::memcpy(ort_tensor.GetTensorMutableRawData(),
                ov_tensor.data(),
                ov_tensor.get_byte_size());
  }

  return nullptr;
}

}  // namespace openvino_ep
}  // namespace onnxruntime