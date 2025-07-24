// Copyright (C) Intel Corporation
// Licensed under the MIT License

#include "ov_shared_context.h"

#include "openvino/runtime/intel_npu/level_zero/level_zero.hpp"

namespace onnxruntime {
namespace openvino_ep_plugin {

SharedContext::SharedWeights::WeightsFile::WeightsFile(std::filesystem::path filename) : file_(filename, std::ios::in | std::ios::binary) {
  try {
    file_.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    weights_size_ = file_.seekg(0, std::ios::end).tellg();
  } catch (std::ifstream::failure& e) {
    OVEP_THROW("Error: Failed to open weight file at ", filename.string(), " ", e.what());
  }
}

void SharedContext::SharedWeights::WeightsFile::load_weights(size_t file_offset, void* data, size_t size) {
  OVEP_ENFORCE(file_offset < weights_size_ && size <= weights_size_ && (file_offset <= weights_size_ - size), "Error: File offset is out of bounds.");
  file_.seekg(file_offset);
  file_.read(reinterpret_cast<char*>(data), size);
}

std::ostream& operator<<(std::ostream& stream, const SharedContext::SharedWeights::Metadata::Map& metadata) {
  try {
    stream << metadata.size();

    // Write each key-value pair
    // Put elements in separate lines to facilitate reading
    for (const auto& [key, value] : metadata) {
      stream << std::endl
             << key.name;
      stream << std::endl
             << value.location;
      stream << std::endl
             << value.data_offset;
      stream << std::endl
             << value.size;
      stream << std::endl
             << value.dimensions.size();
      for (const auto& dim : value.dimensions) {
        stream << std::endl
               << dim;
      }
      stream << std::endl
             << value.element_type;
    }
  } catch (const std::exception& e) {
    OVEP_THROW("Error: Failed to write map data.", e.what());
  } catch (...) {
    OVEP_THROW("Error: Failed to write map data.");
  }

  OVEP_ENFORCE(stream.good(), "Error: Failed to write map data.");
  return stream;
}

std::istream& operator>>(std::istream& stream, SharedContext::SharedWeights::Metadata::Map& metadata) {
  size_t map_size{0};
  try {
    stream >> map_size;

    while (!stream.eof()) {
      SharedContext::SharedWeights::Metadata::Key key;
      SharedContext::SharedWeights::Metadata::Value value;
      stream >> key.name;
      stream >> value.location;
      stream >> value.data_offset;
      stream >> value.size;
      size_t num_dimensions;
      stream >> num_dimensions;

      if (stream.fail()) {
        OVEP_THROW("Error: Failed to read num_dimensions from stream.");
      }

      constexpr size_t MAX_SAFE_DIMENSIONS = 1024;

      size_t safe_num_dimensions = num_dimensions;

      if (num_dimensions == 0 || safe_num_dimensions > MAX_SAFE_DIMENSIONS) {
        OVEP_THROW("Invalid number of dimensions provided.");
      }
      try {
        value.dimensions.resize(safe_num_dimensions);
      } catch (const std::bad_alloc&) {
        OVEP_THROW("Error: Memory allocation failed while resizing dimensions.");
      }

      for (auto& dim : value.dimensions) {
        stream >> dim;
      }
      stream >> value.element_type;
      metadata.emplace(key, value);
    }
  } catch (const std::exception& e) {
    OVEP_THROW("Error: Failed to read map data.", e.what());
  } catch (...) {
    OVEP_THROW("Error: Failed to read map data.");
  }

  OVEP_ENFORCE(metadata.size() == map_size, "Error: Inconsistent map data.");

  return stream;
}
const void* GetTensorData(const ov::Tensor& tensor) {
  // suppress deprecation warnings for ov::Tensor .data()
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#elif defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

  return static_cast<const void*>(tensor.data());

#if defined(_MSC_VER)
#pragma warning(pop)
#elif defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
}

void SharedContext::SharedWeights::AddWeight(const std::string& name, const ov::RemoteContext& remote_context, const ov::Tensor& tensor) {
  SharedContext::SharedWeights::Metadata::Key key{.name = name};

  std::unique_lock<std::shared_mutex> lg(mutex);
  auto& value = metadata[key];
  if (value.tensor) {
    // already created tensor
    return;
  }

  if (remote_context.is<ov::intel_npu::level_zero::ZeroContext>()) {
    auto npu_context = remote_context.as<ov::intel_npu::level_zero::ZeroContext>();
    auto&& remote_tensor = npu_context.create_l0_host_tensor(tensor.get_element_type(), tensor.get_shape(), ov::intel_npu::TensorType::INPUT);

    // Copy data to remote tensor
    std::memcpy(remote_tensor.get(), GetTensorData(tensor), remote_tensor.get_byte_size());
    value.tensor = std::make_shared<ov::Tensor>(remote_tensor);
  } else {
    // Use vanilla tensors
    value.tensor = std::make_shared<ov::Tensor>(tensor.get_element_type(), tensor.get_shape());
    std::memcpy(value.tensor->data(), GetTensorData(tensor), tensor.get_byte_size());
  }
}

}  // namespace openvino_ep_plugin
}  // namespace onnxruntime
