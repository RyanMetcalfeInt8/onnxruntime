// Copyright (C) Intel Corporation
// Licensed under the MIT License

#include "ov_shared_context.h"

#include "openvino/runtime/intel_npu/level_zero/level_zero.hpp"

namespace onnxruntime {
namespace openvino_ep_plugin {

constexpr uint32_t kMetaDataVersion = 1;

SharedContext::SharedWeights::WeightsFile::WeightsFile(std::filesystem::path filename) : file_(filename, std::ios::in | std::ios::binary) {
  try {
    file_.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    weights_size_ = file_.seekg(0, std::ios::end).tellg();
  } catch (std::ifstream::failure& e) {
    OVEP_THROW("Error: Failed to open weight file at ", filename.string(), " ", e.what());
  }
}

void SharedContext::SharedWeights::WeightsFile::LoadWeights(size_t file_offset, void* data, size_t size) {
  OVEP_ENFORCE(file_offset < weights_size_ && size <= weights_size_ && (file_offset <= weights_size_ - size), "Error: File offset is out of bounds.");
  file_.seekg(file_offset);
  file_.read(reinterpret_cast<char*>(data), size);
}

std::ostream& operator<<(std::ostream& stream, const SharedContext::SharedWeights::Metadata::Map& metadata) {
  stream << kMetaDataVersion << std::endl;
  stream << metadata.size();

  // Write each key-value pair
  // Put elements in separate lines to facilitate reading
  for (const auto& [key, value] : metadata) {
    stream << std::endl
           << key.name;
    stream << std::endl
           << value.serialized.location;
    stream << std::endl
           << value.serialized.data_offset;
    stream << std::endl
           << value.serialized.size;
  }

  OVEP_ENFORCE(stream.good(), "Error: Failed to write map data.");
  return stream;
}

std::istream& operator>>(std::istream& stream, SharedContext::SharedWeights::Metadata::Map& metadata) {
  size_t map_size{0};
  uint32_t version{0};
  stream >> version;
  OVEP_ENFORCE(version == kMetaDataVersion, "Unsupported metadata version: ", version);

  stream >> map_size;

  while (!stream.eof()) {
    SharedContext::SharedWeights::Metadata::Key key;
    SharedContext::SharedWeights::Metadata::Value value;
    stream >> key.name;
    stream >> value.serialized.location;
    stream >> value.serialized.data_offset;
    stream >> value.serialized.size;
    OVEP_ENFORCE(!stream.fail(), "Error: Failed to read metadata.");
    metadata.emplace(key, value);
  }

  OVEP_ENFORCE(metadata.size() == map_size, "Error: Inconsistent map data.");

  return stream;
}

static void LoadTensorFromFile(
    SharedContext::SharedWeights::Metadata::Value& value,
    std::filesystem::path model_dir,
    const ov::RemoteContext& remote_context,
    std::unordered_map<std::filesystem::path, std::unique_ptr<SharedContext::SharedWeights::WeightsFile>>& weight_files,
    const ov::element::Type& element_type,
    const ov::Shape& dimensions) {
  auto weights_location = model_dir / value.serialized.location;
  auto& weights_file = weight_files[weights_location];
  if (!weights_file) {
    weights_file = std::make_unique<SharedContext::SharedWeights::WeightsFile>(weights_location);
  }

  if (remote_context.is<ov::intel_npu::level_zero::ZeroContext>()) {
    auto npu_context = remote_context.as<ov::intel_npu::level_zero::ZeroContext>();
    auto&& remote_tensor = npu_context.create_l0_host_tensor(element_type, dimensions, ov::intel_npu::TensorType::INPUT);

    OVEP_ENFORCE(remote_tensor.get_byte_size() == value.serialized.size, "Remote tensor size mismatch");
    weights_file->LoadWeights(value.serialized.data_offset, remote_tensor.get(), value.serialized.size);
    value.tensor = std::make_shared<ov::Tensor>(remote_tensor);
  } else {
    // Use vanilla tensors
    value.tensor = std::make_shared<ov::Tensor>(element_type, dimensions);
    OVEP_ENFORCE(value.tensor->get_byte_size() == value.serialized.size, "Tensor size mismatch");
    weights_file->LoadWeights(value.serialized.data_offset, value.tensor->data(), value.serialized.size);
  }
}

OrtStatus* SharedContext::SharedWeights::LoadMetaData(std::filesystem::path model_dir, const std::string &model_name) {
  std::unique_lock<std::shared_mutex> lg(mutex);
  auto metadata_filepath = GetMetaDataFilePath(model_dir, model_name);
  std::ifstream metadata_file(metadata_filepath, std::ios::binary);
  if (metadata_file.fail()) {
    return Ort::Status(OVEP_ERROR_STR("Failed to read from shared context meta data file ", metadata_filepath.string()).c_str(), ORT_INVALID_ARGUMENT).release();
  }
  metadata_file >> metadata;
  return nullptr;
}

OrtStatus* SharedContext::SharedWeights::SaveMetaData(std::filesystem::path model_dir, const std::string& model_name) {
  std::unique_lock<std::shared_mutex> lg(mutex);
  auto metadata_filepath = GetMetaDataFilePath(model_dir, model_name);
  std::ofstream metadata_file(metadata_filepath, std::ios::binary);
  if (metadata_file.fail()) {
    return Ort::Status(OVEP_ERROR_STR("Failed to write to shared context meta data file ", metadata_filepath.string()).c_str(), ORT_INVALID_ARGUMENT).release();
  }
  metadata_file << metadata;
  return nullptr;
}

void SharedContext::SharedWeights::SetSharedWeightsOnInferRequest(ov::InferRequest& ir, const ov::RemoteContext& remote_context, std::filesystem::path model_dir) {
  std::unique_lock<std::shared_mutex> ul(mutex);
  auto&& compiled_model = ir.get_compiled_model();

  for (const auto& input : compiled_model.inputs()) {
    using Key = SharedContext::SharedWeights::Metadata::Key;
    const auto tensor_key = Key{*input.get_names().begin()};

    auto it = metadata.find(tensor_key);
    if (it == metadata.end()) continue;  // No shared weight for this tensor
    auto& value = it->second;

    if (!value.tensor) {
      LoadTensorFromFile(value, model_dir, remote_context, weight_files, input.get_element_type(), input.get_shape());
    }
    ir.set_tensor(tensor_key.name, *value.tensor);
  }
}

}  // namespace openvino_ep_plugin
}  // namespace onnxruntime
