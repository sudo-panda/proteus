#include "CompilerInterfaceDevice.h"

// Return "auto" should resolve to cudaError_t or hipError_t.
static inline auto
__jit_launch_kernel_internal(const char *ModuleUniqueId, void *Kernel,
                             dim3 GridDim, dim3 BlockDim, void **KernelArgs,
                             uint64_t ShmemSize, void *Stream) {

  using namespace proteus;
  auto &Jit = JitDeviceImplT::instance();
  auto optionalKernelInfo = Jit.getJITKernelInfo(Kernel);
  if (!optionalKernelInfo) {
    return Jit.launchKernelDirect(
        Kernel, GridDim, BlockDim, KernelArgs, ShmemSize,
        static_cast<typename JitDeviceImplT::DeviceStream_t>(Stream));
  }

  const auto &KernelInfo = optionalKernelInfo.value();
  const char *KernelName = KernelInfo.getName();
  int32_t NumRuntimeConstants = KernelInfo.getNumRCs();
  auto RCIndices = KernelInfo.getRCIndices();

  auto printKernelLaunchInfo = [&]() {
    llvm::dbgs() << "JIT Launch Kernel\n";
    llvm::dbgs() << "=== Kernel Info\n";
    llvm::dbgs() << "KernelName " << KernelName << "\n";
    llvm::dbgs() << "Grid " << GridDim.x << ", " << GridDim.y << ", " << GridDim.z
           << "\n";
    llvm::dbgs() << "Block " << BlockDim.x << ", " << BlockDim.y << ", " << BlockDim.z
           << "\n";
    llvm::dbgs() << "KernelArgs " << KernelArgs << "\n";
    llvm::dbgs() << "ShmemSize " << ShmemSize << "\n";
    llvm::dbgs() << "Stream " << Stream << "\n";
    llvm::dbgs() << "=== End Kernel Info\n";
  };

  TIMESCOPE("__jit_launch_kernel");
  DBG(printKernelLaunchInfo());

  return Jit.compileAndRun(
      ModuleUniqueId, Kernel, KernelName, RCIndices, NumRuntimeConstants,
      GridDim, BlockDim, KernelArgs, ShmemSize,
      static_cast<typename JitDeviceImplT::DeviceStream_t>(Stream));
}
