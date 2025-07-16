
// Copyright (C) Intel Corporation
// Licensed under the MIT License

#pragma once

#include <string_view>
#include <istream>

namespace onnxruntime {
namespace openvino_ep {
namespace utils {

bool IsModelStreamXML(std::istream& model_stream);
static inline bool IsXmlHeader(std::string_view header) {
  return header.rfind("<?xml", 0) == 0 && header.find("<net ") != std::string::npos;
}

}  // namespace utils
}  // namespace openvino_ep
}  // namespace onnxruntime
