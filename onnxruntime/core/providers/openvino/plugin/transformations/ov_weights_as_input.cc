// Copyright (C) Intel Corporation
// Licensed under the MIT License

#include "plugin/transformations/ov_weights_as_input.h"
#include "plugin/ov_utils.h"
#include "openvino/op/constant.hpp"
#include "openvino/pass/pass.hpp"
#include "openvino/pass/manager.hpp"

namespace onnxruntime {
namespace openvino_ep_plugin {

std::shared_ptr<ov::Model> convertConstantsToInputs(std::shared_ptr<ov::Model> model, SharedContext::SharedWeights& shared_weights, const ov::RemoteContext& remote_context) {
  // Collect all constants in the model
  std::vector<std::shared_ptr<ov::op::v0::Constant>> constants_to_convert;

  for (auto&& node : model->get_ordered_ops()) {
    if (auto constant = std::dynamic_pointer_cast<ov::op::v0::Constant>(node)) {
      constants_to_convert.push_back(constant);
    }
  }

  // Convert each constant to a parameter
  std::vector<std::shared_ptr<ov::op::v0::Parameter>> new_parameters;

  for (size_t i = 0; i < constants_to_convert.size(); ++i) {
    auto constant = constants_to_convert[i];

    // Create parameter with same shape and type as constant
    auto param_name = constant->get_friendly_name();
    auto parameter = std::make_shared<ov::op::v0::Parameter>(
        constant->get_element_type(),
        constant->get_shape());
    parameter->set_friendly_name(param_name);

    // Replace all uses of constant with the new parameter
    constant->output(0).replace(parameter->output(0));

    new_parameters.push_back(parameter);

    shared_weights.AddWeight(param_name, remote_context, constant->get_tensor_view());
  }

  // Get existing parameters and add new ones
  auto original_params = model->get_parameters();
  original_params.insert(original_params.end(), new_parameters.begin(), new_parameters.end());

  // Create new model with additional parameters
  auto new_model = std::make_shared<ov::Model>(
      model->get_results(),
      original_params,
      model->get_friendly_name() + "_weights_as_inputs");

  return new_model;
}

OrtStatus* weights_as_inputs::InitSharedWeightsAndTransform(std::shared_ptr<ov::Model>& model, SharedContext::SharedWeights& shared_weights, const ov::RemoteContext& remote_context) {
  try {
    model = convertConstantsToInputs(model, shared_weights, remote_context);
    return nullptr;  // Success
  } catch (const std::exception& e) {
    return Ort::Status(OVEP_ERROR_STR("Failed to transform model weights: ", e.what()).c_str(),
                       ORT_RUNTIME_EXCEPTION);
  } catch (...) {
    return Ort::Status(OVEP_ERROR_STR("Failed to transform model weights: unknown error").c_str(), ORT_RUNTIME_EXCEPTION);
  }
}

}  // namespace openvino_ep_plugin
}  // namespace onnxruntime
