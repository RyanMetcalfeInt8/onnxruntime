// Copyright (C) Intel Corporation
// Licensed under the MIT License

#include <sstream>
#include <set>
#include "nlohmann/json.hpp"
#include "ov_plugin_utils.h"

namespace onnxruntime {
namespace openvino_ep_plugin {

OrtStatus* ParsePluginLoadConfigOption(const OrtApi& ort_api, const Ort::Logger& logger, const std::string& config_str, ConfigMap& target_map) {
  if (config_str.empty()) {
    ORT_CXX_LOGF(logger, ORT_LOGGING_LEVEL_WARNING, "Empty OV Config Map passed. Skipping load_config option parsing.");
    target_map = {};
    return nullptr;
  }

  std::stringstream input_str_stream(config_str);
  try {
    nlohmann::json json_config = nlohmann::json::parse(input_str_stream);

    if (!json_config.is_object()) {
      return ort_api.CreateStatus(ORT_INVALID_ARGUMENT, "Invalid JSON structure: Expected an object at the root.");
    }

    for (const auto& [key, value] : json_config.items()) {
      ov::AnyMap inner_map;

      // Ensure that the value for each device is an object (PROPERTY -> VALUE)
      if (!value.is_object()) {
        return ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                    "Invalid JSON structure: Expected an object for device properties.");
      }

      for (auto& [inner_key, inner_value] : value.items()) {
        if (inner_value.is_string()) {
          inner_map[inner_key] = ov::Any(inner_value.get<std::string>());
        } else if (inner_value.is_number_integer()) {
          inner_map[inner_key] = ov::Any(inner_value.get<int64_t>());
        } else if (inner_value.is_number_float()) {
          inner_map[inner_key] = ov::Any(inner_value.get<double>());
        } else if (inner_value.is_boolean()) {
          inner_map[inner_key] = ov::Any(inner_value.get<bool>());
        } else {
          ORT_CXX_LOGF(logger, ORT_LOGGING_LEVEL_WARNING,
                       "Unsupported JSON value type for key: %s. Skipping key.", inner_key.c_str());
        }
      }
      target_map[key] = std::move(inner_map);
    }
  } catch (const nlohmann::json::parse_error& e) {
    // Handle syntax errors in JSON
    return ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                std::format("JSON parsing error: {}", e.what()).c_str());
  } catch (const nlohmann::json::type_error& e) {
    // Handle invalid type accesses
    return ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                std::format("JSON type error: {}", e.what()).c_str());
  } catch (const std::exception& e) {
    return ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                std::format("Error parsing load_config Map: {}", e.what()).c_str());
  }

  return nullptr;
}

OrtStatus* ParsePluginReshapeInputOption([[maybe_unused]] const OrtApi& ort_api, [[maybe_unused]] const Ort::Logger& logger, [[maybe_unused]] const std::string& config_str, [[maybe_unused]] ReshapeMap& reshape_map) {
  return nullptr;  // Placeholder for future implementation
}

}  // namespace openvino_ep_plugin
}  // namespace onnxruntime
