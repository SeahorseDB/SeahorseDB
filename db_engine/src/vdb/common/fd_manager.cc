#include <iostream>
#include <stdexcept>
#include <fcntl.h>
#include <unistd.h>

#include "vdb/common/fd_manager.hh"

namespace vdb {

FdManager& FdManager::GetInstance() {
  static FdManager instance;
  return instance;
}

FdManager::~FdManager() { CleanupAllFds(true); }

int FdManager::GetFdOrOpen(const std::string& file_path, int flags) {
  auto fd = GetFd(file_path, flags);
  if (fd == -1) {
    fd = OpenFile(file_path, flags);
  }
  return fd;
}

int FdManager::GetFdOrCreate(const std::string& file_path, int flags,
                             mode_t mode) {
  auto fd = GetFd(file_path, flags);
  if (fd == -1) {
    fd = CreateFile(file_path, flags, mode);
  }
  return fd;
}

int FdManager::GetFd(const std::string& file_path, int flags) {
  std::lock_guard<std::mutex> lock(fd_pool_mutex_);
  auto it = fd_pool_.find(std::make_pair(file_path, flags));
  if (it != fd_pool_.end()) {
    it->second.last_used = std::chrono::steady_clock::now();
    it->second.use_count++;
    it->second.in_use = true;
    return it->second.fd;
  }
  return -1;
}

int FdManager::CreateFile(const std::string& file_path, int flags,
                          mode_t mode) {
  int fd = open(file_path.c_str(), flags, mode);
  if (fd != -1) {
    AddFd(file_path, flags, fd);
  }
  return fd;
}

int FdManager::OpenFile(const std::string& file_path, int flags) {
  int fd = open(file_path.c_str(), flags);
  if (fd != -1) {
    AddFd(file_path, flags, fd);
  }
  return fd;
}

void FdManager::AddFd(const std::string& file_path, int flags, int fd) {
  std::lock_guard<std::mutex> lock(fd_pool_mutex_);

  if (fd_pool_.size() >= max_cache_size_) {
    RemoveOldest();
  }

  fd_pool_[std::make_pair(file_path, flags)] = {
      fd, std::chrono::steady_clock::now(), 1, true};
}

std::pair<std::string, int> FdManager::GetFilePathAndFlags(int fd) {
  std::lock_guard<std::mutex> lock(fd_pool_mutex_);
  for (auto& [key, entry] : fd_pool_) {
    if (entry.fd == fd) {
      return key;
    }
  }
  return std::make_pair("", 0);
}

void FdManager::ReleaseFd(int fd) {
  std::lock_guard<std::mutex> lock(fd_pool_mutex_);
  for (auto& [_, entry] : fd_pool_) {
    if (entry.fd == fd) {
      entry.in_use = false;
      entry.last_used = std::chrono::steady_clock::now();
      return;
    }
  }
}

void FdManager::ReleaseFile(const std::string& file_path, int flags) {
  std::lock_guard<std::mutex> lock(fd_pool_mutex_);
  auto it = fd_pool_.find(std::make_pair(file_path, flags));
  if (it != fd_pool_.end()) {
    it->second.in_use = false;
    it->second.last_used = std::chrono::steady_clock::now();
  }
}

void FdManager::RemoveFile(const std::string& file_path) {
  std::lock_guard<std::mutex> lock(fd_pool_mutex_);
  for (auto& [key, entry] : fd_pool_) {
    if (key.first == file_path) {
      if (!entry.in_use) {
        close(entry.fd);
        fd_pool_.erase(key);
      }
    }
  }
}

void FdManager::RemoveOldest() {
  if (fd_pool_.empty()) return;

  auto oldest = fd_pool_.end();
  for (auto it = fd_pool_.begin(); it != fd_pool_.end(); ++it) {
    if (!it->second.in_use &&
        (oldest == fd_pool_.end() ||
         it->second.last_used < oldest->second.last_used)) {
      oldest = it;
    }
  }

  if (oldest != fd_pool_.end()) {
    close(oldest->second.fd);
    fd_pool_.erase(oldest);
  }
}

void FdManager::CleanupIdleFds() {
  std::lock_guard<std::mutex> lock(fd_pool_mutex_);
  auto now = std::chrono::steady_clock::now();

  for (auto it = fd_pool_.begin(); it != fd_pool_.end();) {
    if (!it->second.in_use && now - it->second.last_used > max_idle_time_) {
      close(it->second.fd);
      it = fd_pool_.erase(it);
    } else {
      ++it;
    }
  }
}

void FdManager::CleanupFdsInDirectory(const std::string& directory_name,
                                      bool force) {
  std::string directory_path = "/" + directory_name + "/";
  std::lock_guard<std::mutex> lock(fd_pool_mutex_);
  for (auto it = fd_pool_.begin(); it != fd_pool_.end();) {
    if (it->first.first.find(directory_path) != std::string::npos) {
      if (!it->second.in_use || force) {
        close(it->second.fd);
        it = fd_pool_.erase(it);
      } else {
        ++it;
      }
    } else {
      ++it;
    }
  }
}

void FdManager::CleanupAllFds(bool force) {
  std::lock_guard<std::mutex> lock(fd_pool_mutex_);
  for (auto it = fd_pool_.begin(); it != fd_pool_.end();) {
    if (!it->second.in_use || force) {
      close(it->second.fd);
      it = fd_pool_.erase(it);
    } else {
      ++it;
    }
  }
}

int64_t FdManager::GetFdCount() const { return fd_pool_.size(); }

}  // namespace vdb