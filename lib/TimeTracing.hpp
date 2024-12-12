//===-- TimeTracing.hpp -- Time tracing helpers --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_TIME_TRACING_HPP
#define PROTEUS_TIME_TRACING_HPP

#include "llvm/Support/TimeProfiler.h"

namespace proteus {

struct TimeTracerRAII {
  TimeTracerRAII() { llvm::timeTraceProfilerInitialize(500 /* us */, "jit"); }

  ~TimeTracerRAII() {
    if (auto E = llvm::timeTraceProfilerWrite("", "-")) {
      handleAllErrors(std::move(E));
      return;
    }
    llvm::timeTraceProfilerCleanup();
  }
};

#if ENABLE_TIME_TRACING
#define TIMESCOPE(x) TimeTraceScope T(x);
#else
#define TIMESCOPE(x)
#endif

} // namespace proteus

#endif
