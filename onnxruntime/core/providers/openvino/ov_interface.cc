// Copyright (C) Intel Corporation
// Licensed under the MIT License

#include "core/providers/openvino/ov_interface.h"

#define ORT_API_MANUAL_INIT
#include "core/session/onnxruntime_cxx_api.h"
#include "core/providers/shared_library/provider_api.h"
#include "core/providers/openvino/backend_utils.h"

// for make stateful utility function(s)
#include "core/providers/openvino/ov_stateful_patch_utils.h"

using Exception = ov::Exception;

namespace onnxruntime {
namespace openvino_ep {

static const std::string log_tag = "[OpenVINO-EP] ";

#ifndef NDEBUG
void printDebugInfo(const ov::CompiledModel& obj) {
  if (onnxruntime::openvino_ep::backend_utils::IsDebugEnabled()) {
    // output of the actual settings that the device selected
    auto supported_properties = obj.get_property(ov::supported_properties);
    std::cout << "Model:" << std::endl;
    for (const auto& cfg : supported_properties) {
      if (cfg == ov::supported_properties)
        continue;
      auto prop = obj.get_property(cfg);
      if (cfg == ov::device::properties) {
        auto devices_properties = prop.as<ov::AnyMap>();
        for (auto& item : devices_properties) {
          std::cout << "  " << item.first << ": " << std::endl;
          for (auto& item2 : item.second.as<ov::AnyMap>()) {
            OPENVINO_SUPPRESS_DEPRECATED_START
            if (item2.first == ov::supported_properties || item2.first == "SUPPORTED_CONFIG_KEYS)" ||
                item2.first == "SUPPORTED_METRICS")
              continue;
            OPENVINO_SUPPRESS_DEPRECATED_END
            std::cout << "    " << item2.first << ": " << item2.second.as<std::string>() << std::endl;
          }
        }
      } else {
        std::cout << "  " << cfg << ": " << prop.as<std::string>() << std::endl;
      }
    }
  }
}
#endif

std::shared_ptr<OVNetwork> OVCore::ReadModel(const std::string& model, const std::string& model_path) {
  try {
    std::istringstream modelStringStream(model);
    std::istream& modelStream = modelStringStream;
    // Try to load with FrontEndManager
    ov::frontend::FrontEndManager manager;
    ov::frontend::FrontEnd::Ptr FE;
    ov::frontend::InputModel::Ptr inputModel;

    ov::AnyVector params{&modelStream, model_path};

    FE = manager.load_by_model(params);
    if (FE) {
      inputModel = FE->load(params);
      return FE->convert(inputModel);
    } else {
      ORT_THROW(log_tag + "[OpenVINO-EP] Unknown exception while Reading network");
    }
  } catch (const Exception& e) {
    ORT_THROW(log_tag + "[OpenVINO-EP] Exception while Reading network: " + std::string(e.what()));
  } catch (...) {
    ORT_THROW(log_tag + "[OpenVINO-EP] Unknown exception while Reading network");
  }
}

OVExeNetwork OVCore::CompileModel(std::shared_ptr<OVNetwork>& ie_cnn_network,
                                  std::string& hw_target,
                                  ov::AnyMap& device_config,
                                  const std::string& name) {
  ov::CompiledModel obj;
  try {
    if (true) {
      ov::AnyMap config;

      std::cout << "stateless model" << std::endl;
      logBasicModelInfo(ie_cnn_network);

      std::cout << "making stateful..." << std::endl;
      patch_stateful_decoder(ie_cnn_network);

      std::cout << "after stateful transition:" << std::endl;
      logBasicModelInfo(ie_cnn_network);

      // This patches the model so that it only produces the logits required for sampling.
      // Actually either way that happens within NPUW::LLMCompiledModel creation, but this is
      // here mostly to align this behavior for other devices (CPU, GPU).
      apply_slice_before_matmul_transformation(ie_cnn_network);

      auto kv_pos = get_kv_axes_pos(ie_cnn_network);
      std::cout << "kv_pos.batch = " << kv_pos.batch << std::endl;
      std::cout << "kv_pos.seq_len = " << kv_pos.seq_len << std::endl;

      if (hw_target.find("NPU") != std::string::npos) {
        KVDesc kv_desc;
        kv_desc.max_prompt_len = pop_int_and_cast(device_config, "MAX_PROMPT_LEN").value_or(1024u);
        kv_desc.min_response_len = pop_int_and_cast(device_config, "MIN_RESPONSE_LEN").value_or(128u);

        std::cout << "kv_desc.max_prompt_len = " << kv_desc.max_prompt_len << std::endl;
        std::cout << "kv_desc.min_response_len = " << kv_desc.min_response_len << std::endl;

        update_npu_config(config, ie_cnn_network, kv_pos, kv_desc);

        //"++NPUW_LLM_PREFILL_CONFIG" : {"NPUW_DEVICES" : "NPU,CPU", "NPUW_SUBMODEL_DEVICE": "0:CPU,last:CPU"}

        //update_config(config, {"++NPUW_LLM_PREFILL_CONFIG", ov::AnyMap{{"NPUW_DEVICES", "NPU,CPU"}, {"NPUW_SUBMODEL_DEVICE", "0:CPU,last:CPU"}}});

        // force NPUW to use CPU for prefill model. This is needed to obtain accurate first token result.
        update_config(config, {"++NPUW_LLM_PREFILL_CONFIG", ov::AnyMap{{"NPUW_DEVICES", "CPU"}}});
        //update_config(config, {"++NPUW_LLM_GENERATE_CONFIG", ov::AnyMap{{"NPUW_DEVICES", "NPU"}}});
      }

      std::cout << "calling compile on stateful model..." << std::endl;
      obj = core.compile_model(ie_cnn_network, hw_target, config);
      std::cout << "done calling compile on stateful model..." << std::endl;
    } else {
      obj = core.compile_model(ie_cnn_network, hw_target, device_config);
    }
#ifndef NDEBUG
    printDebugInfo(obj);
#endif
    OVExeNetwork exe(obj, hw_target);
    return exe;
  } catch (const Exception& e) {
    ORT_THROW(log_tag + " Exception while Loading Network for graph: " + name + e.what());
  } catch (...) {
    ORT_THROW(log_tag + " Exception while Loading Network for graph " + name);
  }
}

OVExeNetwork OVCore::CompileModel(const std::string& onnx_model,
                                  std::string& hw_target,
                                  ov::AnyMap& device_config,
                                  const std::string& name) {
  ov::CompiledModel obj;
  try {
    obj = core.compile_model(onnx_model, ov::Tensor(), hw_target, device_config);
#ifndef NDEBUG
    printDebugInfo(obj);
#endif
    OVExeNetwork exe(obj, hw_target);
    return exe;
  } catch (const Exception& e) {
    ORT_THROW(log_tag + " Exception while Loading Network for graph: " + name + e.what());
  } catch (...) {
    ORT_THROW(log_tag + " Exception while Loading Network for graph " + name);
  }
}

OVExeNetwork OVCore::ImportModel(std::istream& model_stream,
                                 std::string hw_target,
                                 const ov::AnyMap& device_config,
                                 std::string name) {
  try {
    ov::CompiledModel obj;
    obj = core.import_model(model_stream, hw_target, device_config);
#ifndef NDEBUG
    printDebugInfo(obj);
#endif
    OVExeNetwork exe(obj, hw_target);
    return exe;
  } catch (const Exception& e) {
    ORT_THROW(log_tag + " Exception while Loading Network for graph: " + name + e.what());
  } catch (...) {
    ORT_THROW(log_tag + " Exception while Loading Network for graph " + name);
  }
}

void OVCore::SetCache(const std::string& cache_dir_path) {
  core.set_property(ov::cache_dir(cache_dir_path));
}

#ifdef IO_BUFFER_ENABLED
OVExeNetwork OVCore::CompileModel(std::shared_ptr<const OVNetwork>& model,
                                  OVRemoteContextPtr context, std::string name) {
  try {
    auto obj = core.compile_model(model, *context);
#ifndef NDEBUG
    printDebugInfo(obj);
#endif
    return OVExeNetwork(obj);
  } catch (const Exception& e) {
    ORT_THROW(log_tag + " Exception while Loading Network for graph: " + name + e.what());
  } catch (...) {
    ORT_THROW(log_tag + " Exception while Loading Network for graph " + name);
  }
}
OVExeNetwork OVCore::ImportModel(std::shared_ptr<std::istringstream> model_stream,
                                 OVRemoteContextPtr context, std::string name) {
  try {
    auto obj = core.import_model(*model_stream, *context);
#ifndef NDEBUG
    printDebugInfo(obj);
#endif
    OVExeNetwork exe(obj);
    return exe;
  } catch (const Exception& e) {
    ORT_THROW(log_tag + " Exception while Loading Network for graph: " + name + e.what());
  } catch (...) {
    ORT_THROW(log_tag + " Exception while Loading Network for graph " + name);
  }
}
#endif

std::vector<std::string> OVCore::GetAvailableDevices() {
  auto available_devices = core.get_available_devices();
  return available_devices;
}

void OVCore::SetStreams(const std::string& device_type, int num_streams) {
  core.set_property(device_type, {ov::num_streams(num_streams)});
}

OVInferRequest OVExeNetwork::CreateInferRequest() {
  try {
    auto infReq = obj.create_infer_request();
    OVInferRequest inf_obj(std::move(infReq), device);
    return inf_obj;
  } catch (const Exception& e) {
    ORT_THROW(log_tag + "Exception while creating InferRequest object: " + e.what());
  } catch (...) {
    ORT_THROW(log_tag + "Exception while creating InferRequest object.");
  }
}

OVTensorPtr OVInferRequest::GetTensor(const std::string& input_name) {
  try {
    auto tobj = ovInfReq.get_tensor(input_name);
    OVTensorPtr blob = std::make_shared<OVTensor>(tobj);
    return blob;
  } catch (const Exception& e) {
    ORT_THROW(log_tag + " Cannot access IE Blob for input: " + input_name + e.what());
  } catch (...) {
    ORT_THROW(log_tag + " Cannot access IE Blob for input: " + input_name);
  }
}

std::string OVInferRequest::GetInputTensorName(uint32_t index) {
  try {
    const auto& model = ovInfReq.get_compiled_model();
    return *model.input(index).get_names().begin();
  } catch (const Exception& e) {
    ORT_THROW(log_tag + " Cannot access IE Blob for input number: ", index, e.what());
  } catch (...) {
    ORT_THROW(log_tag + " Cannot access IE Blob for input number: ", index);
  }
}

void OVInferRequest::SetTensor(const std::string& name, OVTensorPtr& blob) {
  try {
    ovInfReq.set_tensor(name, *(blob.get()));
  } catch (const Exception& e) {
    ORT_THROW(log_tag + " Cannot set Remote Blob for output: " + name + e.what());
  } catch (...) {
    ORT_THROW(log_tag + " Cannot set Remote Blob for output: " + name);
  }
}

uint32_t OVInferRequest::GetNumInputs() {
  return static_cast<uint32_t>(ovInfReq.get_compiled_model().inputs().size());
}

void OVInferRequest::RewindKVCache(size_t index)
{
  if (device == "NPU") {
    std::cout << "RewindKVCache on NPU: Trimming cached input_ids / position_ids to length "
        << index << std::endl;
    if (cached_input_ids.size() > index) {
      cached_input_ids.resize(index);
    }

    if (cached_position_ids.size() > index) {
      cached_position_ids.resize(index);
    }
  }
  else {
    std::cout << "OVInferRequest::RewindKVCache: Trimming internal states to length = "
        << index << std::endl;
    if (index == 0) {
      //in this case, since we're trimming *all* of the KVCache, just reset the state.
      ovInfReq.reset_state();
    } else {
      // retrieve kvcache states, and trim...
      // Most of this code was grabbed from here:
      // https://github.com/openvinotoolkit/openvino.genai/blob/releases/2025/1/src/cpp/src/utils.cpp#L329
      auto states = ovInfReq.query_state();
      for (auto& state : states) {
        ov::Tensor old_tensor = state.get_state();
        // [BATCH_SIZE, num_kv_heads, seq_len, head_size]
        auto shape = old_tensor.get_shape();

        if (shape[2] > index) {
          shape[2] = index;

          ov::Coordinate new_shape_begin{0, 0, 0, 0};
          ov::Coordinate new_shape_end{shape};

          auto trimmed_tensor = ov::Tensor(old_tensor, new_shape_begin, new_shape_end);

          ov::Tensor new_tensor(old_tensor.get_element_type(), shape);
          trimmed_tensor.copy_to(new_tensor);

          state.set_state(new_tensor);
        }
      }
    }
  }
}

void OVInferRequest::StartAsync() {
  try {
    // Since we can't seem to set at ORT GenAI layer right now, we just set it here
    // as a workaround.
    // TODO: Fix this.
    ov::Tensor beam_idx = ov::Tensor(ov::element::i32, {1});
    std::fill_n(beam_idx.data<int32_t>(), 1, 0);
    ovInfReq.set_tensor("beam_idx", beam_idx);

    if (device == "NPU") {
      auto input_ids_tensor = ovInfReq.get_tensor("input_ids");

      // add input_ids to our cache
      {
        auto* pData = input_ids_tensor.data<int64_t>();
        for (size_t i = 0; i < input_ids_tensor.get_size(); i++) {
          cached_input_ids.push_back(pData[i]);
        }
      }

      // add position_ids to our cache
      {
        auto position_ids = ovInfReq.get_tensor("position_ids");
        auto* pData = position_ids.data<int64_t>();
        for (size_t i = 0; i < position_ids.get_size(); i++) {
          cached_position_ids.push_back(pData[i]);
        }
      }

      // if we're about to run prefill model
      if (input_ids_tensor.get_size() > 1) {
        //if the input_ids size doesn't equal cached size of the input_ids
        // then it means that we're running 2nd (or later) prompt.
        if (input_ids_tensor.get_shape()[1] != cached_input_ids.size()) {
          // set a new input_ids tensor with the content of our cached input_ids
          {
            auto new_shape = input_ids_tensor.get_shape();
            new_shape[1] = cached_input_ids.size();
            auto new_input_ids = ov::Tensor(input_ids_tensor.get_element_type(), new_shape);
            auto* pNewInputIds = new_input_ids.data<int64_t>();
            std::memcpy(pNewInputIds, cached_input_ids.data(), cached_input_ids.size() * sizeof(int64_t));
            ovInfReq.set_tensor("input_ids", new_input_ids);
          }

          // set a new position_ids tensor with the content of our cached position_ids
          {
            auto position_ids_tensor = ovInfReq.get_tensor("position_ids");
            auto new_shape = position_ids_tensor.get_shape();
            new_shape[1] = cached_position_ids.size();
            auto new_position_ids = ov::Tensor(position_ids_tensor.get_element_type(), new_shape);
            auto* pNewPositionIds = new_position_ids.data<int64_t>();
            std::memcpy(pNewPositionIds, cached_position_ids.data(), cached_position_ids.size() * sizeof(int64_t));
            ovInfReq.set_tensor("position_ids", new_position_ids);
          }
        }
      }
    }

    ovInfReq.start_async();
  } catch (const Exception& e) {
    ORT_THROW(log_tag + " Couldn't start Inference: " + e.what());
  } catch (...) {
    ORT_THROW(log_tag + " In Error Couldn't start Inference");
  }
}

void OVInferRequest::Infer() {
  try {
    ovInfReq.infer();
  } catch (const Exception& e) {
    ORT_THROW(log_tag + " Couldn't start Inference: " + e.what());
  } catch (...) {
    ORT_THROW(log_tag + " In Error Couldn't start Inference");
  }
}

void OVInferRequest::WaitRequest() {
  try {
    ovInfReq.wait();
  } catch (const Exception& e) {
    ORT_THROW(log_tag + " Wait Model Failed: " + e.what());
  } catch (...) {
    ORT_THROW(log_tag + " Wait Mode Failed");
  }
}

void OVInferRequest::QueryStatus() {
  std::cout << "ovInfReq.query_state()"
            << " ";
}
}  // namespace openvino_ep
}  // namespace onnxruntime
