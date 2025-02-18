//===-- CompilerInterfaceDevice.h -- JIT entry point for GPU header --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_COMPILERINTERFACEDEVICE_H
#define PROTEUS_COMPILERINTERFACEDEVICE_H

#include "CompilerInterfaceTypes.h"

#if ENABLE_CUDA
#include "JitEngineDeviceCUDA.hpp"
using JitDeviceImplT = proteus::JitEngineDeviceCUDA;

extern "C" cudaError_t __jit_launch_kernel(const char *ModuleUniqueId,
                                           void *Kernel, dim3 GridDim,
                                           dim3 BlockDim, void **KernelArgs,
                                           uint64_t ShmemSize, void *Stream);

#elif ENABLE_HIP
#include "JitEngineDeviceHIP.hpp"
using JitDeviceImplT = proteus::JitEngineDeviceHIP;

extern "C" hipError_t __jit_launch_kernel(const char *ModuleUniqueId,
                                          void *Kernel, uint64_t GridDimXY,
                                          uint32_t GridDimZ,
                                          uint64_t BlockDim_XY,
                                          uint32_t BlockDimZ, void **KernelArgs,
                                          uint64_t ShmemSize, void *Stream);

#else
#error "CompilerInterfaceDevice requires ENABLE_CUDA or ENABLE_HIP"
#endif

#endif
