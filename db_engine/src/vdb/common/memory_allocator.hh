#pragma once

#include <map>
#include <memory>

#include <arrow/memory_pool.h>

namespace vdb {

/* How To Use Allocator
 *
 * 1. Class
 * By default, use vdb::make_shared for class allocations. If it is necessary to
 * allocate a certain size first and then place the class and call the
 * constructor, use placement new.
 *
 * vdb::make_shared can be used in the same way as std::make_shared. A
 * difference is that it adds up the amount of memory allocated to allow
 * monitoring of memory usage within the server.
 *
 * Placement new allows a constructor to be called in an already allocated space
 * and can be used as follows:
 *
 * char *buffer = AllocateWithPrefix(sizeof(MyClass));
 * MyClass* pMyClass = new (buffer) MyClass(arguments);
 *
 *
 * 2. Space Allocation
 *
 * Space allocation refers to functions that replace C's malloc. There are three
 * types of allocators: general allocators, allocators that store allocated
 * size, and allocators that allocate with alignment.
 *
 * General allocator:
 * Use AllocateNoPrefix, DeallocateNoPrefix to allocate and deallocate.
 * Internally, these functions use malloc to allocate space and add the used
 * space to the memory usage. They are used when memory usage can be remembered
 * within a scope or calculated within a structure. Since the used amount must
 * be provided when deallocating, do not use this allocator if the usage amount
 * is unknown at the deallocate point.
 *
 * Allocator that stores size:
 * Use AllocateWithPrefix, DeallocateWithPrefix to allocate and deallocate. If
 * the operating system or allocation function provides a way to fetch the size
 * of the allocated memory space, only the requested size is allocated.
 * Otherwise, an additional 8 bytes are allocated to remember the allocated size
 * at the front of the allocated area. When using this, the allocated size is
 * stored, so it is not necessary to provide the size during deallocation. An
 * additional 8 bytes are allocated, so use it only when necessary to avoid
 * waste.
 *
 * Allocator with alignment:
 * Use AllocateAligned, DeallocateAligned to allocate and deallocate. It can
 * allocate aligned memory space based on posix_memaligned. Since alignment is
 * required, unlike other allocators, it takes an additional argument for
 * alignment. Like the general allocator, the allocated size must be provided at
 * deallocate.
 *
 * When allocating buffers for various purposes, choose one of the three above.
 * Use Allocate and Deallocate from the same allocator to ensure that the memory
 * usage does not break.
 *
 * 3. std Structure Allocation
 *
 * Commonly used std structures like vector, string, map are made to be used in
 * the same way.
 *
 * If std::vector is needed, use vdb::vector, for std::string use vdb::string,
 * for std::map use vdb::map, etc.
 *
 * However, since std:: structures and vdb:: structures are not compatible, a
 * deep copy is unavoidable when converting. Currently, the use of vdb::
 * structures is limited to main structures within vdb like Table, Segment,
 * IndexHandler, Index, Set (MutableArray), etc. Use std:: structures in other
 * parts.
 *
 *  If any additional std structures are needed besides std::vector, std::string
 * and std::map, refer to vdb::vector for guidance. You should define and use
 * them in the same way as std:: structures.
 *
 * std::vector -> vdb::vector
 * std::structure -> vdb::structure
 *
 *
 *
 *
 *
 * How To Use ArrowMemoryPool
 *
 * In Arrow, there are mainly two cases where a memory pool is required as an
 * argument:
 *
 * 1. When creating something based on a builder.
 *
 * For builders, the constructor is designed to take a memory pool as an
 * argument. Therefore, you should insert the globally created memory pool in
 * the order of the constructor's arguments. For example, in the case of the
 * BinaryBuilder, which is a superclass of StringBuilder, it is as follows:
 *
 * class BaseBinaryBuilder
 * : public ArrayBuilder,
 *   public internal::ArrayBuilderExtraOps<BaseBinaryBuilder<TYPE>,
 * std::string_view> { public: using TypeClass = TYPE; using offset_type =
 * typename TypeClass::offset_type;
 *
 *     explicit BaseBinaryBuilder(MemoryPool* pool = default_memory_pool(),
 *                                int64_t alignment = kDefaultBufferAlignment)
 *     : ArrayBuilder(pool, alignment),
 *       offsets_builder_(pool, alignment),
 *       value_data_builder_(pool, alignment) {}
 *
 *     BaseBinaryBuilder(const std::shared_ptr<DataType>& type, MemoryPool*
 * pool) : BaseBinaryBuilder(pool) {}
 * };
 *
 * Usage is as follows:
 * arrow::StringBuilder builder(&vdb::arrow_pool);
 *
 *
 * 2. When serializing an already created structure.
 *
 * For writing operations within Arrow, such as Serialize, the related
 * structures (like Writers) are created with an argument called
 * IpcWriteOptions.
 *
 * Example:
 * Result<std::shared_ptr<Buffer>> SerializeRecordBatch(const RecordBatch&
 * batch, const IpcWriteOptions& options);
 *
 * ARROW_EXPORT
 * Result<std::shared_ptr<RecordBatchWriter>> MakeStreamWriter(
 *     io::OutputStream* sink, const std::shared_ptr<Schema>& schema,
 *     const IpcWriteOptions& options = IpcWriteOptions::Defaults());
 *
 * The memory pool is included within these Options. After retrieving the
 * Options, you can switch the memory pool to vdb::arrow_pool as follows:
 *
 * arrow::ipc::IpcWriteOptions options =
 * arrow::ipc::IpcWriteOptions::Defaults();
 *
 * options.memory_pool = &vdb::arrow_pool;
 *
 * Usage is as follows:
 * SerializeRecordBatch(rb, options);
 */

class BaseAllocator;
class ArrowMemoryPool;

extern BaseAllocator base_allocator;
extern ArrowMemoryPool arrow_pool;

#define ARROW_CAST_OR_RAISE_IMPL(result_name, lhs, type, rexpr) \
  auto&& result_name = (rexpr);                                 \
  ARROW_RETURN_IF_(!(result_name).ok(), (result_name).status(), \
                   ARROW_STRINGIFY(rexpr));                     \
  lhs = static_cast<type>(std::move(result_name).ValueUnsafe());

#define ARROW_CAST_OR_RAISE(lhs, type, rexpr)                              \
  ARROW_CAST_OR_RAISE_IMPL(                                                \
      ARROW_ASSIGN_OR_RAISE_NAME(_error_or_value, __COUNTER__), lhs, type, \
      rexpr);

#define ARROW_CAST_OR_NULL(lhs, type, rexpr) \
  lhs = static_cast<type>((rexpr).ValueOr(nullptr));

/* if allocated size is managed by higher layer, don't need to allocate memory
 * with prefix(size) */
#define AllocateNoPrefix(n) \
  vdb::base_allocator.AllocateNoPrefixInternal(n, false, __FILE__, __LINE__)
#define AllocateNoPrefixUseThrow(n) \
  vdb::base_allocator.AllocateNoPrefixInternal(n, true, __FILE__, __LINE__)
#define DeallocateNoPrefix(p, n) \
  vdb::base_allocator.DeallocateNoPrefixInternal(p, n, __FILE__, __LINE__)

/* length of allocated size is stored in prefix portion */
#define AllocateWithPrefix(n) \
  vdb::base_allocator.AllocateWithPrefixInternal(n, false, __FILE__, __LINE__)
#define AllocateWithPrefixUseThrow(n) \
  vdb::base_allocator.AllocateWithPrefixInternal(n, true, __FILE__, __LINE__)
#define DeallocateWithPrefix(p) \
  vdb::base_allocator.DeallocateWithPrefixInternal(p, __FILE__, __LINE__)

/* aligned chunk is allocated. no prefix is stored. */
#define AllocateAligned(alignment, n)                                        \
  vdb::base_allocator.AllocateAlignedInternal(alignment, n, false, __FILE__, \
                                              __LINE__)
#define AllocateAlignedUseThrow(alignment, n)                               \
  vdb::base_allocator.AllocateAlignedInternal(alignment, n, true, __FILE__, \
                                              __LINE__)
#define DeallocateAligned(p, n) \
  vdb::base_allocator.DeallocateAlignedInternal(p, n, __FILE__, __LINE__)

std::string BytesToHumanReadable(uint64_t bytes);
uint64_t GetRedisResidentSize();
uint64_t GetRedisAllocatedSize();
uint64_t GetVdbAllocatedSize();
void ResetVdbAllocatedSize();

class BaseAllocator {
 public:
  BaseAllocator() noexcept {}

  /* if allocated size is managed by higher layer, don't need to allocate memory
   * with prefix(size) */
  arrow::Result<void*> AllocateNoPrefixInternal(uint64_t n, bool use_throw,
                                                const char* file = __FILE__,
                                                const int line = __LINE__);

  void DeallocateNoPrefixInternal(void* p, uint64_t n,
                                  const char* file = __FILE__,
                                  const int line = __LINE__);

  /* length of allocated size is stored in prefix portion */
  arrow::Result<void*> AllocateWithPrefixInternal(uint64_t n, bool use_throw,
                                                  const char* file = __FILE__,
                                                  const int line = __LINE__);

  void DeallocateWithPrefixInternal(void* p, const char* file = __FILE__,
                                    const int line = __LINE__);

  /* aligned chunk is allocated. no prefix is stored. */
  arrow::Result<void*> AllocateAlignedInternal(uint64_t alignment, uint64_t n,
                                               bool use_throw,
                                               const char* file = __FILE__,
                                               const int line = __LINE__);

  void DeallocateAlignedInternal(void* p, uint64_t n,
                                 const char* file = __FILE__,
                                 const int line = __LINE__);

  inline arrow::Status CheckOom(bool use_throw, uint64_t size);
  inline arrow::Status HandleOom(bool use_throw, uint64_t size);
};

template <typename T>
class Allocator : public std::allocator<T>, public BaseAllocator {
 public:
  using value_type = T;
  using pointer = T*;
  using const_pointer = const T*;
  using reference = T&;
  using const_reference = const T&;
  using size_type = std::size_t;
  using difference_type = ptrdiff_t;

  Allocator() noexcept : BaseAllocator() {}
  template <typename U>
  Allocator(const Allocator<U>& other) noexcept
      : BaseAllocator(static_cast<const BaseAllocator&>(other)) {}

  template <typename U>
  struct rebind {
    typedef Allocator<U> other;
  };

  static std::shared_ptr<Allocator<T>> Make() {
    return std::make_shared<Allocator<T>>();
  }

  pointer allocate(size_type n) {
    ARROW_CAST_OR_NULL(auto ptr, pointer,
                       BaseAllocator::AllocateNoPrefixInternal(
                           n * sizeof(T), true /* use_throw */));
    return ptr;
  }

  void deallocate(pointer p, size_type n) {
    BaseAllocator::DeallocateNoPrefixInternal(p, n * sizeof(T));
  }

  template <class U, class... Args>
  void construct(U* p, Args&&... args) {
    new ((void*)p) U(std::forward<Args>(args)...);
  }

  template <class U>
  void destroy(U* p) {
    p->~U();
  }

  size_type max_size() const noexcept {
    return std::numeric_limits<size_type>::max() / sizeof(T);
  }
};

template <typename T, typename U>
bool operator==(const Allocator<T>&, const Allocator<U>&) {
  return true;
}

template <typename T, typename U>
bool operator!=(const Allocator<T>&, const Allocator<U>&) {
  return false;
}

template <typename T>
class SizeAwareAllocator : public std::allocator<T>, public BaseAllocator {
 public:
  using value_type = T;
  using pointer = T*;
  using const_pointer = const T*;
  using reference = T&;
  using const_reference = const T&;
  using size_type = std::size_t;
  using difference_type = ptrdiff_t;

  SizeAwareAllocator() noexcept : BaseAllocator() {}
  template <typename U>
  SizeAwareAllocator(const SizeAwareAllocator<U>& other) noexcept
      : BaseAllocator(static_cast<const BaseAllocator&>(other)) {}

  template <typename U>
  struct rebind {
    typedef SizeAwareAllocator<U> other;
  };

  pointer allocate(size_type n) {
    ARROW_CAST_OR_NULL(auto ptr, pointer,
                       BaseAllocator::AllocateWithPrefixInternal(
                           n * sizeof(T), true /* use_throw */));
    return ptr;
  }

  void deallocate(pointer p, [[maybe_unused]] size_type n) {
    BaseAllocator::DeallocateWithPrefixInternal(p);
  }

  template <class U, class... Args>
  void construct(U* p, Args&&... args) {
    new ((void*)p) U(std::forward<Args>(args)...);
  }

  template <class U>
  void destroy(U* p) {
    p->~U();
  }

  size_type max_size() const noexcept {
    return std::numeric_limits<size_type>::max() / sizeof(T);
  }
};

template <typename T, typename U>
bool operator==(const SizeAwareAllocator<T>&, const SizeAwareAllocator<U>&) {
  return true;
}

template <typename T, typename U>
bool operator!=(const SizeAwareAllocator<T>&, const SizeAwareAllocator<U>&) {
  return false;
}

template <typename T>
class AlignedAllocator : public std::allocator<T>, public BaseAllocator {
 public:
  using value_type = T;
  using pointer = T*;
  using const_pointer = const T*;
  using reference = T&;
  using const_reference = const T&;
  using size_type = std::size_t;
  using difference_type = ptrdiff_t;

  AlignedAllocator(size_t alignment) noexcept
      : BaseAllocator(), alignment_{alignment} {}
  template <typename U>
  AlignedAllocator(const AlignedAllocator<U>& other) noexcept
      : BaseAllocator(static_cast<const BaseAllocator&>(other)),
        alignment_{other.alignment_} {}

  template <typename U>
  struct rebind {
    typedef AlignedAllocator<U> other;
  };
  pointer allocate(size_type n) {
    ARROW_CAST_OR_NULL(auto ptr, pointer,
                       BaseAllocator::AllocateAlignedInternal(
                           alignment_, n * sizeof(T), true /* use_throw */));
    return ptr;
  }

  void deallocate(pointer p, size_type n) {
    BaseAllocator::DeallocateAlignedInternal(p, n * sizeof(T));
  }

  template <class U, class... Args>
  void construct(U* p, Args&&... args) {
    new ((void*)p) U(std::forward<Args>(args)...);
  }

  template <class U>
  void destroy(U* p) {
    p->~U();
  }

  size_type max_size() const noexcept {
    return std::numeric_limits<size_type>::max() / sizeof(T);
  }
  size_t alignment_;
};

template <typename T, typename U>
bool operator==(const AlignedAllocator<T>&, const AlignedAllocator<U>&) {
  return true;
}

template <typename T, typename U>
bool operator!=(const AlignedAllocator<T>&, const AlignedAllocator<U>&) {
  return false;
}

class ArrowMemoryPool : public arrow::MemoryPool {
 public:
  explicit ArrowMemoryPool();
  ~ArrowMemoryPool() override = default;

  using MemoryPool::Allocate;
  using MemoryPool::Free;
  using MemoryPool::Reallocate;

  arrow::Status Allocate(int64_t size, int64_t alignment,
                         uint8_t** out) override;
  arrow::Status Reallocate(int64_t old_size, int64_t new_size,
                           int64_t alignment, uint8_t** ptr) override;
  void Free(uint8_t* buffer, int64_t size, int64_t alignment) override;

  inline arrow::Status CheckOom(uint64_t size);

  int64_t bytes_allocated() const override;

  int64_t max_memory() const override;

  int64_t total_bytes_allocated() const override;

  int64_t num_allocations() const override;

  std::string backend_name() const override;

  // Track mmap memory (actual allocation is managed by OS)
  void TrackMmapMemory(int64_t size);
  void UntrackMmapMemory(int64_t size);
  int64_t mmap_bytes_allocated() const;

 private:
  arrow::MemoryPool* pool_;
  std::atomic<int64_t> mmap_bytes_allocated_{0};
};

/* copy is required (std::string -> vdb::string, vdb::string -> std::string) */
using string = std::basic_string<char, std::char_traits<char>, Allocator<char>>;

template <typename T>
using vector = std::vector<T, Allocator<T>>;

template <typename Key, typename T, typename Compare = std::less<Key>>
using map = std::map<Key, T, Compare, Allocator<std::pair<const Key, T>>>;

template <class T, class... Args>
std::shared_ptr<T> make_shared(Args&&... args) {
  static Allocator<T> allocator;
  return std::allocate_shared<T, Allocator<T>>(allocator,
                                               std::forward<Args>(args)...);
}

template <class T, class... Args>
std::shared_ptr<T> make_size_aware_shared(Args&&... args) {
  static SizeAwareAllocator<T> allocator;
  return std::allocate_shared<T, SizeAwareAllocator<T>>(
      allocator, std::forward<Args>(args)...);
}

template <class T, class... Args>
std::shared_ptr<T> make_aligned_shared(size_t alignment, Args&&... args) {
  AlignedAllocator<T> allocator(alignment);
  return std::allocate_shared<T, AlignedAllocator<T>>(
      allocator, std::forward<Args>(args)...);
}
}  // namespace vdb
