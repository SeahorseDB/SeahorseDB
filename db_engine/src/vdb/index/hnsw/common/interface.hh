#pragma once

#include <queue>
#include <vector>
#include <iostream>
#include <string.h>

#include "vdb/index/hnsw/common/filter.hh"

namespace vdb {
namespace hnsw {
typedef uint64_t labeltype;

template <typename dist_t>
class BaseSearchStopCondition {
 public:
  virtual void add_point_to_result(labeltype label, const void *datapoint,
                                   dist_t dist) = 0;

  virtual void remove_point_from_result(labeltype label, const void *datapoint,
                                        dist_t dist) = 0;

  virtual bool should_stop_search(dist_t candidate_dist, dist_t lowerBound) = 0;

  virtual bool should_consider_candidate(dist_t candidate_dist,
                                         dist_t lowerBound) = 0;

  virtual bool should_remove_extra() = 0;

  virtual void filter_results(
      std::vector<std::pair<dist_t, labeltype>> &candidates) = 0;

  virtual ~BaseSearchStopCondition() {}
};

template <typename T>
class pairGreater {
 public:
  bool operator()(const T &p1, const T &p2) { return p1.first > p2.first; }
};

template <typename T>
static void writeBinaryPOD(std::ostream &out, const T &podRef) {
  out.write((char *)&podRef, sizeof(T));
}

template <typename T>
static void readBinaryPOD(std::istream &in, T &podRef) {
  in.read((char *)&podRef, sizeof(T));
}
template <typename dist_t>
class AlgorithmInterface {
 public:
  virtual void addPoint(const void *datapoint, labeltype label,
                        bool replace_deleted = false) = 0;

  virtual std::priority_queue<std::pair<dist_t, labeltype>> searchKnn(
      const void *, size_t, BaseFilterFunctor *isIdAllowed = nullptr) = 0;

  // Return k nearest neighbor in the order of closer fist
  virtual std::vector<std::pair<dist_t, labeltype>> searchKnnCloserFirst(
      const void *query_data, size_t k,
      BaseFilterFunctor *isIdAllowed = nullptr);

  virtual void saveIndex(const std::string &location) = 0;
  virtual ~AlgorithmInterface() {}
};

template <typename dist_t>
std::vector<std::pair<dist_t, labeltype>>
AlgorithmInterface<dist_t>::searchKnnCloserFirst(
    const void *query_data, size_t k, BaseFilterFunctor *isIdAllowed) {
  std::vector<std::pair<dist_t, labeltype>> result;

  // here searchKnn returns the result in the order of further first
  auto ret = searchKnn(query_data, k, isIdAllowed);
  {
    size_t sz = ret.size();
    result.resize(sz);
    while (!ret.empty()) {
      result[--sz] = ret.top();
      ret.pop();
    }
  }

  return result;
}
}  // namespace hnsw
}  // namespace vdb