#include <cstddef>
#include <memory>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <iomanip>

#include "vdb/common/memory_allocator.hh"
#include "vdb/common/system_log.hh"

#ifdef __cplusplus
extern "C" {
#include "zmalloc.h"
#include "server.h"
}
#endif

namespace vdb {

BaseAllocator base_allocator;
ArrowMemoryPool arrow_pool;

std::atomic_uint64_t vdb_allocated_size = 0;

std::string BytesToHumanReadable(uint64_t bytes) {
  if (bytes < 1024) {
    return std::to_string(bytes) + " bytes";
  } else if (bytes < 1024 * 1024) {
    double kb = static_cast<double>(bytes) / 1024.0;
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3) << kb << " KB";
    return oss.str();
  } else if (bytes < 1024 * 1024 * 1024) {
    double mb = static_cast<double>(bytes) / (1024.0 * 1024.0);
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3) << mb << " MB";
    return oss.str();
  } else {
    double gb = static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3) << gb << " GB";
    return oss.str();
  }
}

uint64_t GetRedisResidentSize() {
  size_t resident, allocated, active;
  zmalloc_get_allocator_info(&allocated, &active, &resident);
  if (resident == 0) {
    size_t process_rss = zmalloc_get_rss();
    size_t lua_memory = evalMemory();
    resident = process_rss - lua_memory;
  }
  return resident;
}

uint64_t GetRedisAllocatedSize() { return zmalloc_used_memory(); }
uint64_t GetVdbAllocatedSize() { return vdb_allocated_size.load(); }
void ResetVdbAllocatedSize() {
  /* reset vdb_allocated_size only in debug mode */
#if !defined(NDEBUG)
  vdb_allocated_size.store(0);
#endif
}

/* if allocated size is managed by higher layer, don't need to allocate memory
 * with prefix(size) */
arrow::Result<void *> BaseAllocator::AllocateNoPrefixInternal(uint64_t n,
                                                              bool use_throw,
                                                              const char *file,
                                                              const int line) {
  if (server.maxmemory != 0) {
    ARROW_RETURN_NOT_OK(CheckOom(use_throw, n));
  }
  auto ptr = malloc(n);
  if (ptr != nullptr) {
    zmalloc_increase_used_memory(n);
    vdb_allocated_size += n;
  } else {
    return HandleOom(false, n);
  }
  SYSTEM_LOG_WITH_PATH(
      vdb::LogTopic::Unknown, LogLevel::kLogMemDebug, file, line,
      "AllocateNoPrefix Done: ptr=%p size=%ld (redis=%lu, vdb=%lu)", ptr, n,
      GetRedisAllocatedSize(), GetVdbAllocatedSize());
  return ptr;
}

void BaseAllocator::DeallocateNoPrefixInternal(void *p, uint64_t n,
                                               const char *file,
                                               const int line) {
  zmalloc_decrease_used_memory(n);
  vdb_allocated_size -= n;
  free(p);
  SYSTEM_LOG_WITH_PATH(
      vdb::LogTopic::Unknown, LogLevel::kLogMemDebug, file, line,
      "DeallocateNoPrefix Done: ptr=%p size=%ld (redis=%lu, vdb=%lu)", p, n,
      GetRedisAllocatedSize(), GetVdbAllocatedSize());
}

/* length of allocated size is stored in prefix portion */
arrow::Result<void *> BaseAllocator::AllocateWithPrefixInternal(
    uint64_t n, bool use_throw, const char *file, const int line) {
  if (server.maxmemory != 0) {
    ARROW_RETURN_NOT_OK(CheckOom(use_throw, n));
  }
  auto ptr = ztrymalloc(n);
  if (ptr != nullptr) {
    vdb_allocated_size += zmalloc_size(ptr);
  } else {
    return HandleOom(false, n);
  }
  SYSTEM_LOG_WITH_PATH(
      vdb::LogTopic::Unknown, LogLevel::kLogMemDebug, file, line,
      "AllocateWithPrefix Done: ptr=%p size=%ld (redis=%lu, vdb=%lu)", ptr,
      zmalloc_size(ptr), GetRedisAllocatedSize(), GetVdbAllocatedSize());
  return ptr;
}

void BaseAllocator::DeallocateWithPrefixInternal(void *p, const char *file,
                                                 const int line) {
  size_t chunk_size = zmalloc_size(p);
  vdb_allocated_size -= chunk_size;
  zfree(p);
  SYSTEM_LOG_WITH_PATH(
      vdb::LogTopic::Unknown, LogLevel::kLogMemDebug, file, line,
      "DeallocateWithPrefix Done: ptr=%p size=%ld (redis=%lu, vdb=%lu)", p,
      chunk_size, GetRedisAllocatedSize(), GetVdbAllocatedSize());
}

arrow::Result<void *> BaseAllocator::AllocateAlignedInternal(uint64_t alignment,
                                                             uint64_t n,
                                                             bool use_throw,
                                                             const char *file,
                                                             const int line) {
  if (server.maxmemory != 0) {
    ARROW_RETURN_NOT_OK(CheckOom(use_throw, n));
  }
  void *ptr = nullptr;
  if (posix_memalign(&ptr, alignment, n) == 0) {
    zmalloc_increase_used_memory(n);
    vdb_allocated_size += n;
  } else {
    return HandleOom(false, n);
  }
  SYSTEM_LOG_WITH_PATH(
      vdb::LogTopic::Unknown, LogLevel::kLogMemDebug, file, line,
      "AllocateAligned Done: ptr=%p align=%lu, size=%lu "
      "(redis=%lu, vdb=%lu)",
      ptr, alignment, n, GetRedisAllocatedSize(), GetVdbAllocatedSize());
  return ptr;
}

void BaseAllocator::DeallocateAlignedInternal(void *p, uint64_t n,
                                              const char *file,
                                              const int line) {
  zmalloc_decrease_used_memory(n);
  vdb_allocated_size -= n;
  free(p);
  SYSTEM_LOG_WITH_PATH(
      vdb::LogTopic::Unknown, LogLevel::kLogMemDebug, file, line,
      "DeallocateAligned Done: ptr=%p size=%ld (redis=%lu, vdb=%lu)", p, n,
      GetRedisAllocatedSize(), GetVdbAllocatedSize());
}

inline arrow::Status BaseAllocator::CheckOom(bool use_throw, uint64_t size) {
  auto used_memory = GetRedisAllocatedSize();
  if (server.maxmemory != 0 && (used_memory + size) >= server.maxmemory) {
    SYSTEM_LOG(vdb::LogTopic::Unknown, LogLevel::kLogAlways,
               "VdbBaseAllocator: Out of memory trying to allocate %lu bytes",
               size);

    bool should_handle_oom = server.hidden_enable_oom_testing;
#ifndef NDEBUG
    should_handle_oom = true;
#endif
    if (should_handle_oom) {
      server.temporary_oom_blocker = true;
      if (use_throw) {
        throw std::runtime_error("out of memory.");
      } else {
        return arrow::Status::OutOfMemory("out of memory. ", used_memory, " + ",
                                          size, " is over ", server.maxmemory);
      }
    }
  }
  return arrow::Status::OK();
}

inline arrow::Status BaseAllocator::HandleOom(bool use_throw, uint64_t size) {
  SYSTEM_LOG(vdb::LogTopic::Unknown, LogLevel::kLogAlways,
             "VdbBaseAllocator: Out of memory trying to allocate %lu bytes",
             size);
  server.temporary_oom_blocker = true;
  if (use_throw) {
    throw std::runtime_error("out of memory.");
  } else {
    return arrow::Status::OutOfMemory("out of memory. malloc size ", size,
                                      " is failed ");
  }
}

ArrowMemoryPool::ArrowMemoryPool() : pool_{arrow::default_memory_pool()} {}

arrow::Status ArrowMemoryPool::Allocate(int64_t size, int64_t alignment,
                                        uint8_t **out) {
  if (server.maxmemory != 0) {
    ARROW_RETURN_NOT_OK(CheckOom(size));
  }
  arrow::Status status = pool_->Allocate(size, alignment, out);
  ARROW_RETURN_NOT_OK(status);
  zmalloc_increase_used_memory(size);
  vdb_allocated_size += size;
  SYSTEM_LOG(
      vdb::LogTopic::Unknown, LogLevel::kLogMemDebug,
      "ArrowMemoryPool(Allocate) align=%ld size=%ld (redis=%lu, vdb=%lu)",
      alignment, size, GetRedisAllocatedSize(), GetVdbAllocatedSize());
  return status;
}

arrow::Status ArrowMemoryPool::Reallocate(int64_t old_size, int64_t new_size,
                                          int64_t alignment, uint8_t **ptr) {
  if (server.maxmemory != 0 && new_size > old_size) {
    ARROW_RETURN_NOT_OK(CheckOom(new_size - old_size));
  }
  arrow::Status status = pool_->Reallocate(old_size, new_size, alignment, ptr);
  ARROW_RETURN_NOT_OK(status);
  if (new_size > old_size) {
    size_t size_change = static_cast<size_t>(new_size - old_size);
    zmalloc_increase_used_memory(size_change);
    vdb_allocated_size += size_change;
  } else {
    size_t size_change = static_cast<size_t>(old_size - new_size);
    zmalloc_decrease_used_memory(size_change);
    vdb_allocated_size -= size_change;
  }
  SYSTEM_LOG(vdb::LogTopic::Unknown, LogLevel::kLogMemDebug,
             "ArrowMemoryPool(Reallocate) align=%ld size=(%ld -> %ld) "
             "(redis=%lu, vdb=%lu)",
             alignment, old_size, new_size, GetRedisAllocatedSize(),
             GetVdbAllocatedSize());
  return status;
}

void ArrowMemoryPool::Free(uint8_t *buffer, int64_t size, int64_t alignment) {
  pool_->Free(buffer, size, alignment);
  zmalloc_decrease_used_memory(size);
  vdb_allocated_size -= size;
  SYSTEM_LOG(vdb::LogTopic::Unknown, LogLevel::kLogMemDebug,
             "ArrowMemoryPool(Free) align=%ld size=%ld (redis=%lu, vdb=%lu)",
             alignment, size, GetRedisAllocatedSize(), GetVdbAllocatedSize());
}

inline arrow::Status ArrowMemoryPool::CheckOom(uint64_t size) {
  auto used_memory = GetRedisAllocatedSize();
  if (server.maxmemory != 0 && (used_memory + size) >= server.maxmemory) {
    SYSTEM_LOG(vdb::LogTopic::Unknown, LogLevel::kLogAlways,
               "ArrowMemoryPool: Out of memory trying to allocate %lu bytes",
               size);

    bool should_handle_oom = server.hidden_enable_oom_testing;
#ifndef NDEBUG
    should_handle_oom = true;
#endif
    if (should_handle_oom) {
      server.temporary_oom_blocker = true;
      return arrow::Status::OutOfMemory("VDB MemoryPool: out of memory. ",
                                        used_memory, " + ", size, " is over ",
                                        server.maxmemory);
    }
  }
  return arrow::Status::OK();
}

int64_t ArrowMemoryPool::bytes_allocated() const {
  int64_t nb_bytes = pool_->bytes_allocated();
  return nb_bytes;
}

int64_t ArrowMemoryPool::max_memory() const {
  int64_t mem = pool_->max_memory();
  return mem;
}

int64_t ArrowMemoryPool::total_bytes_allocated() const {
  int64_t mem = pool_->total_bytes_allocated();
  return mem;
}

int64_t ArrowMemoryPool::num_allocations() const {
  int64_t mem = pool_->num_allocations();
  return mem;
}

std::string ArrowMemoryPool::backend_name() const {
  return pool_->backend_name();
}

void ArrowMemoryPool::TrackMmapMemory(int64_t size) {
  mmap_bytes_allocated_.fetch_add(size, std::memory_order_relaxed);
  SYSTEM_LOG(vdb::LogTopic::Memory, LogLevel::kLogMemDebug,
             "ArrowMemoryPool(TrackMmap) size=%ld, total_mmap=%ld", size,
             mmap_bytes_allocated_.load());
}

void ArrowMemoryPool::UntrackMmapMemory(int64_t size) {
  mmap_bytes_allocated_.fetch_sub(size, std::memory_order_relaxed);
  SYSTEM_LOG(vdb::LogTopic::Memory, LogLevel::kLogMemDebug,
             "ArrowMemoryPool(UntrackMmap) size=%ld, total_mmap=%ld", size,
             mmap_bytes_allocated_.load());
}

int64_t ArrowMemoryPool::mmap_bytes_allocated() const {
  return mmap_bytes_allocated_.load(std::memory_order_relaxed);
}

}  // namespace vdb
