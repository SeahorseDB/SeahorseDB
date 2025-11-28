#pragma once

#include <functional>
#include <map>
#include <memory>
#include <string>

namespace vdb {

class MetadataChecker;  // Forward declaration

// Type alias for the factory function
using MetadataCheckerFactory =
    std::function<std::shared_ptr<MetadataChecker>()>;

// Function to get the registered checker factories
const std::map<std::string, MetadataCheckerFactory>&
GetRegisteredCheckerFactories();

// Forward declaration of the registry accessor function (defined in .cc)
std::map<std::string, MetadataCheckerFactory>& GetCheckerRegistry();

// It is used to register the checker to the registry with the given key and
// factory function during the program startup
struct MetadataCheckerRegistrar {
  MetadataCheckerRegistrar(const std::string& key,
                           MetadataCheckerFactory factory) {
    // Access the registry using the forward-declared function
    std::map<std::string, MetadataCheckerFactory>& registry =
        GetCheckerRegistry();
    // Register the factory function in the map during static initialization
    registry.emplace(key, std::move(factory));
  }
};

}  // namespace vdb