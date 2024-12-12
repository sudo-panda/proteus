//===-- JitStorageCache.hpp -- JIT storage-based cache header impl. --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_JITSTOREDCACHE_HPP
#define PROTEUS_JITSTOREDCACHE_HPP

#include <cstdint>
#include <filesystem>
#include <llvm/Support/MemoryBufferRef.h>

#include "llvm/ADT/StringRef.h"

#include "Utils.h"

namespace proteus {

// NOTE: Storage cache assumes that stored code is re-usable across runs!
// TODO: Source code changes should invalidate the cache. Also, if storing
// assembly (PTX) or binary (ELF), then device globals may have different
// addresses that render it invalid. In this case, store LLVM IR to re-link
// globals.
template <typename Function_t> class JitStorageCache {
public:
  JitStorageCache() { std::filesystem::create_directory(StorageDirectory); }
  std::unique_ptr<llvm::MemoryBuffer> lookupObject(uint64_t HashValue,
                                                   llvm::StringRef Kernel) {
    TIMESCOPE("object lookup");
    Accesses++;

    std::string Filepath =
        StorageDirectory + "/cache-jit-" + std::to_string(HashValue) + ".o";

    auto MemBuffer = llvm::MemoryBuffer::getFile(Filepath);
    if (!MemBuffer)
      return nullptr;

    Hits++;
    return std::move(MemBuffer.get());
  }

  std::unique_ptr<llvm::MemoryBuffer> lookupBitcode(uint64_t HashValue,
                                                    llvm::StringRef Kernel) {
    TIMESCOPE("object lookup");
    Accesses++;

    std::string Filepath =
        StorageDirectory + "/cache-jit-" + std::to_string(HashValue) + ".bc";

    auto MemBuffer = llvm::MemoryBuffer::getFile(Filepath);
    if (!MemBuffer)
      return nullptr;

    Hits++;
    return std::move(MemBuffer.get());
  }

  void storeObject(uint64_t HashValue, llvm::MemoryBufferRef ObjBufRef) {
    TIMESCOPE("Store object");
    saveToFile(
        (StorageDirectory + "/cache-jit-" + std::to_string(HashValue) + ".o"),
        HashValue,
        llvm::StringRef{ObjBufRef.getBufferStart(), ObjBufRef.getBufferSize()});
  }

  void storeBitcode(uint64_t HashValue, llvm::StringRef IR) {
    saveToFile(
        (StorageDirectory + "/cache-jit-" + std::to_string(HashValue) + ".bc"),
        HashValue, IR);
  }

  void printStats() {
    // Use printf to avoid re-ordering outputs by outs() in HIP.
    printf("JitStorageCache hits %lu total %lu\n", Hits, Accesses);
  }

private:
  uint64_t Hits = 0;
  uint64_t Accesses = 0;
  const std::string StorageDirectory = ".proteus";

  void saveToFile(llvm::StringRef Filepath, uint64_t HashValue, llvm::StringRef Data) {
    std::error_code EC;
    llvm::raw_fd_ostream Out(Filepath, EC);
    if (EC)
      FATAL_ERROR("Cannot open file" + Filepath);
    Out << Data;
    Out.close();
  }
};

} // namespace proteus

#endif
