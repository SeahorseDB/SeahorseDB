#include "vdb/data/checker_registry.hh"

#include <map>
#include <string>

namespace vdb {

// Definition for the static registry accessor function (declared in .hh)
std::map<std::string, MetadataCheckerFactory>& GetCheckerRegistry() {
  static std::map<std::string, MetadataCheckerFactory> registry;
  return registry;
}

// Function definition for getting the registered factories
const std::map<std::string, MetadataCheckerFactory>&
GetRegisteredCheckerFactories() {
  return GetCheckerRegistry();
}

}  // namespace vdb