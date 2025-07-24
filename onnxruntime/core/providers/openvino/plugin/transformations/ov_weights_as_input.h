// Copyright (C) Intel Corporation
// Licensed under the MIT License

#pragma once

#include "plugin/ov_shared_context.h"

namespace onnxruntime {
namespace openvino_ep_plugin {

namespace weights_as_inputs {
OrtStatus* InitSharedWeightsAndTransform(std::shared_ptr<ov::Model>& model, SharedContext::SharedWeights& shared_weights, const ov::RemoteContext& remote_context);
}

}  // namespace openvino_ep_plugin
}  // namespace onnxruntime
