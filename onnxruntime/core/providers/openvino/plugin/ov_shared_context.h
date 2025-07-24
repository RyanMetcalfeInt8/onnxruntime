// Copyright (C) Intel Corporation
// Licensed under the MIT License

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <filesystem>
#include <fstream>
#include <shared_mutex>

#include "openvino/openvino.hpp"
#include "ov_utils.h"

namespace onnxruntime {
namespace openvino_ep_plugin {

class SharedContext {
 public:
  SharedContext() {}
  struct SharedWeights {
    struct Metadata {
      struct Key {
        std::string name;
        bool operator==(const Key&) const = default;
      };
      struct Hash {
        std::size_t operator()(const Key& key) const noexcept {
          return std::hash<std::string>()(key.name);
        }
      };
      struct Value {
        std::string location;
        size_t data_offset;
        size_t size;
        ov::Shape dimensions;
        ov::element::Type element_type;
        std::shared_ptr<ov::Tensor> tensor;
      };
      using Map = std::unordered_map<Key, Value, Hash>;
      friend std::ostream& operator<<(std::ostream& right, const Metadata::Map& metadata);
      friend std::istream& operator>>(std::istream& right, Metadata::Map& metadata);
    };

    struct WeightsFile {
      OVEP_DISABLE_COPY_AND_MOVE(WeightsFile);
      WeightsFile() = delete;
      explicit WeightsFile(std::filesystem::path filename);

      void load_weights(size_t file_offset, void* data, size_t size);

     private:
      std::ifstream file_;
      size_t weights_size_;
    };

    std::shared_mutex mutex;
    std::filesystem::path external_weight_filename;
    std::unique_ptr<WeightsFile> mapped_weights;
    Metadata::Map metadata;
    std::filesystem::path metadata_filepath;

    void AddWeight(const std::string&, const ov::RemoteContext&, const ov::Tensor&);
  } shared_weights;
};

}  // namespace openvino_ep_plugin
}  // namespace onnxruntime
