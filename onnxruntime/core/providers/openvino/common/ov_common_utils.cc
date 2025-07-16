// Copyright (C) Intel Corporation
// Licensed under the MIT License

#include <istream>
#include <string>

#include "ov_common_utils.h"

namespace onnxruntime {
namespace openvino_ep {
namespace utils {

bool IsModelStreamXML(std::istream& model_stream) {
  std::streampos originalPos = model_stream.tellg();

  // first, get the total size of model_stream in bytes
  model_stream.seekg(0, std::ios::end);
  auto end_pos = model_stream.tellg();
  //  Restore the stream position
  model_stream.seekg(originalPos);
  auto total_size = end_pos - originalPos;

  // Choose 32 bytes to hold content of:
  // '<?xml version-"1.0"?> <net '
  const std::streamsize header_check_len = 32;
  if (total_size < header_check_len) {
    return false;
  }

  // read 32 bytes into header
  std::string header(header_check_len, '\0');
  model_stream.read(&header[0], header_check_len);
  // Clear any read errors
  model_stream.clear();
  // Restore the stream position
  model_stream.seekg(originalPos);

  return IsXmlHeader(header);
}

}  // namespace utils
}  // namespace openvino_ep
}  // namespace onnxruntime
