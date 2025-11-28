#pragma once
#include <unordered_map>
#include <fstream>
#include <mutex>
#include <algorithm>
#include <assert.h>
#include <fstream>

#include "vdb/index/hnsw/common/interface.hh"

#include "vdb/vector/distance/space_header.hh"
#include "vdb/vector/distance/space_ip_float32.hh"
#include "vdb/vector/distance/space_l2_float32.hh"

namespace vdb {
namespace hnsw {
template <typename dist_t>
class BruteforceSearch : public AlgorithmInterface<dist_t> {
 public:
  char *data_;
  size_t maxelements_;
  size_t cur_element_count;
  size_t size_per_element_;

  size_t data_size_;
  DISTFUNC_FLOAT32 fstdistfunc_;
  void *dist_func_param_;
  size_t dim_;
  std::mutex index_lock;

  std::unordered_map<labeltype, size_t> dict_external_to_internal;

  BruteforceSearch(vdb::DistanceSpace space, size_t dim,
                   const std::string &location)
      : data_(nullptr),
        maxelements_(0),
        cur_element_count(0),
        size_per_element_(0),
        data_size_(0),
        dist_func_param_(nullptr) {
    loadIndex(location, space, dim);
  }

  bool setDistanceSpace(vdb::DistanceSpace space, size_t dim) {
    dim_ = dim;
    dist_func_param_ = &dim_;
    switch (space) {
      case vdb::DistanceSpace::kIP: {
        vdb::InnerProductSpaceFloat32 space(dim);
        fstdistfunc_ = space.get_dist_func();
        data_size_ = space.get_data_size();
      } break;
      case vdb::DistanceSpace::kL2: {
        vdb::L2SpaceFloat32 space(dim);
        fstdistfunc_ = space.get_dist_func();
        data_size_ = space.get_data_size();
      } break;
      default:
        return false;
    }
    return true;
  }

  BruteforceSearch(vdb::DistanceSpace space, size_t dim, size_t maxElements) {
    maxelements_ = maxElements;
    setDistanceSpace(space, dim);
    size_per_element_ = data_size_ + sizeof(labeltype);
    data_ = (char *)malloc(maxElements * size_per_element_);
    if (data_ == nullptr)
      throw std::runtime_error(
          "Not enough memory: BruteforceSearch failed to allocate data");
    cur_element_count = 0;
  }

  ~BruteforceSearch() { free(data_); }

  void addPoint(const void *datapoint, labeltype label,
                [[maybe_unused]] bool replace_deleted = false) {
    int idx;
    {
      std::unique_lock<std::mutex> lock(index_lock);

      auto search = dict_external_to_internal.find(label);
      if (search != dict_external_to_internal.end()) {
        idx = search->second;
      } else {
        if (cur_element_count >= maxelements_) {
          throw std::runtime_error(
              "The number of elements exceeds the specified limit\n");
        }
        idx = cur_element_count;
        dict_external_to_internal[label] = idx;
        cur_element_count++;
      }
    }
    memcpy(data_ + size_per_element_ * idx + data_size_, &label,
           sizeof(labeltype));
    memcpy(data_ + size_per_element_ * idx, datapoint, data_size_);
  }

  void removePoint(labeltype cur_external) {
    std::unique_lock<std::mutex> lock(index_lock);

    auto found = dict_external_to_internal.find(cur_external);
    if (found == dict_external_to_internal.end()) {
      return;
    }

    dict_external_to_internal.erase(found);

    size_t cur_c = found->second;
    labeltype label =
        *((labeltype *)(data_ + size_per_element_ * (cur_element_count - 1) +
                        data_size_));
    dict_external_to_internal[label] = cur_c;
    memcpy(data_ + size_per_element_ * cur_c,
           data_ + size_per_element_ * (cur_element_count - 1),
           data_size_ + sizeof(labeltype));
    cur_element_count--;
  }

  std::priority_queue<std::pair<dist_t, labeltype>> searchKnn(
      const void *query_data, size_t k,
      BaseFilterFunctor *isIdAllowed = nullptr) const {
    assert(k <= cur_element_count);
    std::priority_queue<std::pair<dist_t, labeltype>> topResults;
    if (cur_element_count == 0) return topResults;
    for (int i = 0; i < k; i++) {
      dist_t dist = fstdistfunc_(query_data, data_ + size_per_element_ * i,
                                 dist_func_param_);
      labeltype label =
          *((labeltype *)(data_ + size_per_element_ * i + data_size_));
      if ((!isIdAllowed) || (*isIdAllowed)(label)) {
        topResults.emplace(dist, label);
      }
    }
    dist_t lastdist = topResults.empty() ? std::numeric_limits<dist_t>::max()
                                         : topResults.top().first;
    for (int i = k; i < cur_element_count; i++) {
      dist_t dist = fstdistfunc_(query_data, data_ + size_per_element_ * i,
                                 dist_func_param_);
      if (dist <= lastdist) {
        labeltype label =
            *((labeltype *)(data_ + size_per_element_ * i + data_size_));
        if ((!isIdAllowed) || (*isIdAllowed)(label)) {
          topResults.emplace(dist, label);
        }
        if (topResults.size() > k) topResults.pop();

        if (!topResults.empty()) {
          lastdist = topResults.top().first;
        }
      }
    }
    return topResults;
  }

  void saveIndex(const std::string &location) {
    std::ofstream output(location, std::ios::binary);
    std::streampos position;

    writeBinaryPOD(output, maxelements_);
    writeBinaryPOD(output, size_per_element_);
    writeBinaryPOD(output, cur_element_count);

    output.write(data_, maxelements_ * size_per_element_);

    output.close();
  }

  void loadIndex(const std::string &location, vdb::DistanceSpace space,
                 size_t dim) {
    std::ifstream input(location, std::ios::binary);
    std::streampos position;

    readBinaryPOD(input, maxelements_);
    readBinaryPOD(input, size_per_element_);
    readBinaryPOD(input, cur_element_count);

    setDistanceSpace(space, dim);
    size_per_element_ = data_size_ + sizeof(labeltype);
    data_ = (char *)malloc(maxelements_ * size_per_element_);
    if (data_ == nullptr)
      throw std::runtime_error(
          "Not enough memory: loadIndex failed to allocate data");

    input.read(data_, maxelements_ * size_per_element_);

    input.close();
  }
};
}  // namespace hnsw
}  // namespace vdb
