#pragma once

#include <chrono>
#include <mutex>
#include <string>
#include <unordered_map>

namespace vdb {

class FdManager {
 private:
  struct FdEntry {
    int fd;
    std::chrono::steady_clock::time_point last_used;
    size_t use_count;
    bool in_use;
  };

  struct HashByStringInPair {
    std::size_t operator()(const std::pair<std::string, int>& p) const {
      return std::hash<std::string>{}(p.first);
    }
  };
  // key: (file_path, flags)
  std::unordered_map<std::pair<std::string, int>, FdEntry, HashByStringInPair>
      fd_pool_;
  std::mutex fd_pool_mutex_;
  const size_t max_cache_size_ = 1024;
  const std::chrono::seconds max_idle_time_{300};

  FdManager() {}

 protected:
  friend class FdHandle;
  int GetFdOrOpen(const std::string& file_path, int flags);
  int GetFdOrCreate(const std::string& file_path, int flags, mode_t mode);

  int GetFd(const std::string& file_path, int flags);
  std::pair<std::string, int> GetFilePathAndFlags(int fd);

  int CreateFile(const std::string& file_path, int flags, mode_t mode);
  int OpenFile(const std::string& file_path, int flags);
  void AddFd(const std::string& file_path, int flags, int fd);
  void ReleaseFd(int fd);
  void ReleaseFile(const std::string& file_path, int flags);
  void RemoveFile(const std::string& file_path);

 public:
  static FdManager& GetInstance();
  ~FdManager();

  void RemoveOldest();
  void CleanupIdleFds();
  void CleanupFdsInDirectory(const std::string& directory_name,
                             bool force = false);
  void CleanupAllFds(bool force = false);

  int64_t GetFdCount() const;

  FdManager(const FdManager&) = delete;
  FdManager& operator=(const FdManager&) = delete;
  FdManager(FdManager&&) = delete;
  FdManager& operator=(FdManager&&) = delete;
};

/*
 * FdHandle is a helper class that implements the RAII pattern
 * to acquire a file descriptor and release it when going out of scope.
 */
class FdHandle {
 private:
  int fd_;
  bool released_;

 public:
  FdHandle() : fd_(-1), released_(true) {}

  FdHandle(const std::string& file_path, int flags)
      : fd_(FdManager::GetInstance().GetFdOrOpen(file_path, flags)),
        released_(false) {
    if (fd_ == -1) {
      released_ = true;
    }
  }

  FdHandle(const std::string& file_path, int flags, mode_t mode)
      : fd_(FdManager::GetInstance().GetFdOrCreate(file_path, flags, mode)),
        released_(false) {
    if (fd_ == -1) {
      released_ = true;
    }
  }

  ~FdHandle() {
    if (!released_) {
      FdManager::GetInstance().ReleaseFd(fd_);
    }
  }

  void ReInit(const std::string& file_path, int flags) {
    Release();
    fd_ = FdManager::GetInstance().GetFdOrOpen(file_path, flags);
    if (fd_ == -1) {
      released_ = true;
    } else {
      released_ = false;
    }
  }

  void ReInit(const std::string& file_path, int flags, mode_t mode) {
    Release();
    fd_ = FdManager::GetInstance().GetFdOrCreate(file_path, flags, mode);
    if (fd_ == -1) {
      released_ = true;
    } else {
      released_ = false;
    }
  }

  // Access the file descriptor
  int Get() const { return fd_; }

  std::pair<std::string, int> GetFilePathAndFlags() const {
    return FdManager::GetInstance().GetFilePathAndFlags(fd_);
  }

  // Explicitly release the file descriptor
  void Release() {
    if (!released_) {
      FdManager::GetInstance().ReleaseFd(fd_);
      fd_ = -1;
      released_ = true;
    }
  }

  FdHandle(const FdHandle&) = delete;
  FdHandle& operator=(const FdHandle&) = delete;

  FdHandle(FdHandle&& other) noexcept
      : fd_(other.fd_), released_(other.released_) {
    other.released_ = true;
  }

  FdHandle& operator=(FdHandle&& other) noexcept {
    if (this != &other) {
      if (!released_) {
        FdManager::GetInstance().ReleaseFd(fd_);
      }
      fd_ = other.fd_;
      released_ = other.released_;
      other.released_ = true;
    }
    return *this;
  }
};

}  // namespace vdb