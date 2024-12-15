//===-- JitEngineDevice.cpp -- Base JIT Engine Device header impl. --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_JITENGINEDEVICE_HPP
#define PROTEUS_JITENGINEDEVICE_HPP

#include "llvm/Linker/Linker.h"
#include <cstdint>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBufferRef.h>
#include <memory>

#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Target/TargetMachine.h"
#include <llvm/IR/Constants.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>
#include <optional>
#include <torch/script.h>

#include "CompilerInterfaceTypes.h"
#include "JitCache.hpp"
#include "JitEngine.hpp"
#include "JitStorageCache.hpp"
#include "TimeTracing.hpp"
#include "TransformArgumentSpecialization.hpp"
#include "Utils.h"

namespace proteus {

class JITKernelInfo {
  char const *Name;
  llvm::SmallVector<int32_t> RCIndices;
  int32_t NumRCs;

public:
  JITKernelInfo(char const *_Name, int32_t *_RCIndices, int32_t _NumRCs)
      : Name(_Name), NumRCs(_NumRCs) {
    for (int32_t I = 0; I < NumRCs; ++I) {
      RCIndices.push_back(_RCIndices[I]);
    }
  }
  JITKernelInfo() : Name(nullptr), NumRCs(0), RCIndices() {}
  auto getName() const { return Name; }
  auto getRCIndices() const { return RCIndices; }
  auto getNumRCs() const { return NumRCs; }
};

struct FatbinWrapper_t {
  int32_t Magic;
  int32_t Version;
  const char *Binary;
  void **PrelinkedFatbins;
};

template <typename ImplT> struct DeviceTraits;

template <typename ImplT> class JitEngineDevice : protected JitEngine {
public:
  using DeviceError_t = typename DeviceTraits<ImplT>::DeviceError_t;
  using DeviceStream_t = typename DeviceTraits<ImplT>::DeviceStream_t;
  using KernelFunction_t = typename DeviceTraits<ImplT>::KernelFunction_t;

  DeviceError_t
  compileAndRun(llvm::StringRef ModuleUniqueId, void *Kernel, llvm::StringRef KernelName,
                const llvm::SmallVector<int32_t> &RCIndices, int NumRuntimeConstants,
                dim3 GridDim, dim3 BlockDim, void **KernelArgs,
                uint64_t ShmemSize,
                typename DeviceTraits<ImplT>::DeviceStream_t Stream);

  void insertRegisterVar(const char *VarName, const void *Addr) {
    VarNameToDevPtr[VarName] = Addr;
  }
  void registerLinkedBinary(FatbinWrapper_t *FatbinWrapper,
                            const char *ModuleId);
  void registerFatBinary(void *Handle, FatbinWrapper_t *FatbinWrapper,
                         const char *ModuleId);
  void registerFatBinaryEnd();
  void registerFunction(void *Handle, void *Kernel, char *KernelName,
                        int32_t *RCIndices, int32_t NumRCs);

  struct BinaryInfo {
    FatbinWrapper_t *FatbinWrapper;
    llvm::SmallVector<std::string> LinkedModuleIds;
  };

  void *CurHandle = nullptr;
  std::unordered_map<std::string, FatbinWrapper_t *> ModuleIdToFatBinary;
  llvm::DenseMap<void *, BinaryInfo> HandleToBinaryInfo;
  llvm::DenseMap<void *, void *> KernelToHandleMap;
  llvm::SmallVector<std::string> GlobalLinkedModuleIds;
  llvm::SmallPtrSet<void *, 8> GlobalLinkedBinaries;

  bool containsJITKernelInfo(const void *Func) {
    return JITKernelInfoMap.contains(Func);
  }

  std::optional<JITKernelInfo> getJITKernelInfo(const void *Func) {
    if (!containsJITKernelInfo(Func)) {
      return std::nullopt;
    }
    return JITKernelInfoMap[Func];
  }

private:
  //------------------------------------------------------------------
  // Begin Methods implemented in the derived device engine class.
  //------------------------------------------------------------------
  void *resolveDeviceGlobalAddr(const void *Addr) {
    return static_cast<ImplT &>(*this).resolveDeviceGlobalAddr(Addr);
  }

  void setLaunchBoundsForKernel(llvm::Module &M, llvm::Function &F, size_t GridSize,
                                int BlockSize) {
    static_cast<ImplT &>(*this).setLaunchBoundsForKernel(M, F, GridSize,
                                                         BlockSize);
  }

  void setKernelDims(llvm::Module &M, dim3 &GridDim, dim3 &BlockDim) {
    auto ReplaceIntrinsicDim = [&](llvm::StringRef IntrinsicName, uint32_t DimValue) {
      auto CollectCallUsers = [](llvm::Function &F) {
        llvm::SmallVector<llvm::CallInst *> CallUsers;
        for (auto *User : F.users()) {
          auto *Call = llvm::dyn_cast<llvm::CallInst>(User);
          if (!Call)
            continue;
          CallUsers.push_back(Call);
        }

        return CallUsers;
      };
      llvm::Function *IntrinsicFunction = M.getFunction(IntrinsicName);
      if (!IntrinsicFunction)
        return;

      for (auto *Call : CollectCallUsers(*IntrinsicFunction)) {
        llvm::Value *ConstantValue =
            llvm::ConstantInt::get(llvm::Type::getInt32Ty(M.getContext()), DimValue);
        Call->replaceAllUsesWith(ConstantValue);
        Call->eraseFromParent();
      }
    };

    ReplaceIntrinsicDim(ImplT::gridDimXFnName(), GridDim.x);
    ReplaceIntrinsicDim(ImplT::gridDimYFnName(), GridDim.y);
    ReplaceIntrinsicDim(ImplT::gridDimZFnName(), GridDim.z);

    ReplaceIntrinsicDim(ImplT::blockDimXFnName(), BlockDim.x);
    ReplaceIntrinsicDim(ImplT::blockDimYFnName(), BlockDim.y);
    ReplaceIntrinsicDim(ImplT::blockDimZFnName(), BlockDim.z);

    auto InsertAssume = [&](llvm::StringRef IntrinsicName, int DimValue) {
      llvm::Function *IntrinsicFunction = M.getFunction(IntrinsicName);
      if (!IntrinsicFunction || IntrinsicFunction->use_empty())
        return;

      // Iterate over all uses of the intrinsic.
      for (auto U : IntrinsicFunction->users()) {
        auto *Call = llvm::dyn_cast<llvm::CallInst>(U);
        if (!Call)
          continue;

        // Insert the llvm.assume intrinsic.
        llvm::IRBuilder<> Builder(Call->getNextNode());
        llvm::Value *Bound = llvm::ConstantInt::get(Call->getType(), DimValue);
        llvm::Value *Cmp = Builder.CreateICmpULT(Call, Bound);

        llvm::Function *AssumeIntrinsic =
            llvm::Intrinsic::getDeclaration(&M, llvm::Intrinsic::assume);
        Builder.CreateCall(AssumeIntrinsic, Cmp);
      }
    };

    // Inform LLVM about the range of possible values of threadIdx.*.
    InsertAssume(ImplT::threadIdxXFnName(), BlockDim.x);
    InsertAssume(ImplT::threadIdxYFnName(), BlockDim.y);
    InsertAssume(ImplT::threadIdxZFnName(), BlockDim.z);

    // Inform LLVM about the range of possible values of blockIdx.*.
    InsertAssume(ImplT::blockIdxXFnName(), GridDim.x);
    InsertAssume(ImplT::blockIdxYFnName(), GridDim.y);
    InsertAssume(ImplT::blockIdxZFnName(), GridDim.z);
  }

  void getRuntimeConstantsFromModule(llvm::Module &M, void **KernelArgs,
                                     llvm::StringRef KernelName,
                                     const llvm::SmallVector<int32_t> &RCIndices,
                                     llvm::SmallVector<RuntimeConstant> &RCsVec) {
    llvm::Function *F = M.getFunction(KernelName);
    for (int I = 0; I < RCIndices.size(); ++I) {
      llvm::Value *Arg = F->getArg(RCIndices[I]);
      llvm::Type *ArgType = Arg->getType();
      llvm::Constant *C = nullptr;

      RuntimeConstant RC;
      if (ArgType->isIntegerTy(1)) {
        RC.Value.BoolVal = *(bool *)KernelArgs[RCIndices[I]];
      } else if (ArgType->isIntegerTy(8)) {
        RC.Value.Int8Val = *(int8_t *)KernelArgs[RCIndices[I]];
      } else if (ArgType->isIntegerTy(32)) {
        RC.Value.Int32Val = *(int32_t *)KernelArgs[RCIndices[I]];
      } else if (ArgType->isIntegerTy(64)) {
        RC.Value.Int64Val = *(int64_t *)KernelArgs[RCIndices[I]];
      } else if (ArgType->isFloatTy()) {
        RC.Value.FloatVal = *(float *)KernelArgs[RCIndices[I]];
      }
      // NOTE: long double on device should correspond to plain double.
      // XXX: CUDA with a long double SILENTLY fails to create a working
      // kernel in AOT compilation, with or without JIT.
      else if (ArgType->isDoubleTy()) {
        RC.Value.DoubleVal = *(double *)KernelArgs[RCIndices[I]];
      } else if (ArgType->isX86_FP80Ty() || ArgType->isPPC_FP128Ty() ||
                 ArgType->isFP128Ty()) {
        RC.Value.LongDoubleVal = *(long double *)KernelArgs[RCIndices[I]];
      } else if (ArgType->isPointerTy()) {
        RC.Value.PtrVal = (void *)KernelArgs[RCIndices[I]];
      } else {
        std::string TypeString;
        llvm::raw_string_ostream TypeOstream(TypeString);
        ArgType->print(TypeOstream);
        FATAL_ERROR("JIT Incompatible type in runtime constant: " +
                    TypeOstream.str());
      }

      RCsVec.push_back(RC);
    }
  }

  DeviceError_t launchKernelFunction(KernelFunction_t KernelFunc, dim3 GridDim,
                                     dim3 BlockDim, void **KernelArgs,
                                     uint64_t ShmemSize,
                                     DeviceStream_t Stream) {
    TIMESCOPE(__FUNCTION__);
    return static_cast<ImplT &>(*this).launchKernelFunction(
        KernelFunc, GridDim, BlockDim, KernelArgs, ShmemSize, Stream);
  }

  DeviceError_t launchKernelDirect(void *KernelFunc, dim3 GridDim,
                                   dim3 BlockDim, void **KernelArgs,
                                   uint64_t ShmemSize, DeviceStream_t Stream) {
    return static_cast<ImplT &>(*this).launchKernelDirect(
        KernelFunc, GridDim, BlockDim, KernelArgs, ShmemSize, Stream);
  }

  std::unique_ptr<llvm::MemoryBuffer> codegenObject(llvm::Module &M, llvm::StringRef DeviceArch) {
    return static_cast<ImplT &>(*this).codegenObject(M, DeviceArch);
  }

  KernelFunction_t getKernelFunctionFromImage(llvm::StringRef KernelName,
                                              const void *Image) {
    return static_cast<ImplT &>(*this).getKernelFunctionFromImage(KernelName,
                                                                  Image);
  }

  std::unique_ptr<llvm::MemoryBuffer> extractDeviceBitcode(llvm::StringRef KernelName,
                                                     void *Kernel) {
    TIMESCOPE(__FUNCTION__)
    return static_cast<ImplT &>(*this).extractDeviceBitcode(KernelName, Kernel);
  }
  //------------------------------------------------------------------
  // End Methods implemented in the derived device engine class.
  //------------------------------------------------------------------

  void specializeIR(llvm::Module &M, llvm::StringRef FnName, llvm::StringRef Suffix,
                    dim3 &BlockDim, dim3 &GridDim,
                    const llvm::SmallVector<int32_t> &RCIndices, RuntimeConstant *RC,
                    int NumRuntimeConstants);

  void
  relinkGlobals(llvm::Module &M,
                std::unordered_map<std::string, const void *> &VarNameToDevPtr);

protected:
  JitEngineDevice() {}
  ~JitEngineDevice() {
    CodeCache.printStats();
    StorageCache.printStats();
  }

  JitCache<KernelFunction_t> CodeCache;
  JitStorageCache<KernelFunction_t> StorageCache;
  std::string DeviceArch;
  std::unordered_map<std::string, const void *> VarNameToDevPtr;
  void linkJitModule(llvm::Module *M, llvm::LLVMContext *Ctx, llvm::StringRef KernelName,
                     llvm::SmallVector<std::unique_ptr<llvm::Module>> &LinkedModules);

private:
  // This map is private and only accessible via the API.
  llvm::DenseMap<const void *, JITKernelInfo> JITKernelInfoMap;
};

template <typename ImplT>
void JitEngineDevice<ImplT>::specializeIR(llvm::Module &M, llvm::StringRef FnName,
                                          llvm::StringRef Suffix, dim3 &BlockDim,
                                          dim3 &GridDim,
                                          const llvm::SmallVector<int32_t> &RCIndices,
                                          RuntimeConstant *RC,
                                          int NumRuntimeConstants) {

  TIMESCOPE("specializeIR");
  DBG(llvm::dbgs() << "=== Parsed Module\n" << M << "=== End of Parsed Module\n");
  llvm::Function *F = M.getFunction(FnName);
  assert(F && "Expected non-null function!");

  // Remove llvm.global.annotations now that we have read them.
  if (auto *GlobalAnnotations = M.getGlobalVariable("llvm.global.annotations"))
    M.eraseGlobalVariable(GlobalAnnotations);
  // Remove the __clang_gpu_used_external used in HIP RDC compilation and its
  // uses in llvm.used, llvm.compiler.used.
  if (auto *ClangGPUUsedExternal =
          M.getNamedGlobal("__clang_gpu_used_external")) {
    removeFromUsedLists(M, [&ClangGPUUsedExternal](llvm::Constant *C) {
      if (auto *GV = llvm::dyn_cast<llvm::GlobalVariable>(C))
        return GV == ClangGPUUsedExternal;
      return false;
    });
    M.eraseGlobalVariable(ClangGPUUsedExternal);
  }

  // Replace argument uses with runtime constants.
  if (Config.ENV_PROTEUS_SPECIALIZE_ARGS)
    // TODO: change NumRuntimeConstants to size_t at interface.
    TransformArgumentSpecialization::transform(
        M, *F, RCIndices,
        llvm::ArrayRef<RuntimeConstant>{RC,
                                  static_cast<size_t>(NumRuntimeConstants)});

  // Replace uses of blockDim.* and gridDim.* with constants.
  if (Config.ENV_PROTEUS_SPECIALIZE_DIMS) {
    setKernelDims(M, GridDim, BlockDim);
  }

  DBG(llvm::dbgs() << "=== JIT Module\n" << M << "=== End of JIT Module\n");

  F->setName(FnName + Suffix);

  if (Config.ENV_PROTEUS_SET_LAUNCH_BOUNDS)
    setLaunchBoundsForKernel(M, *F, GridDim.x * GridDim.y * GridDim.z,
                             BlockDim.x * BlockDim.y * BlockDim.z);

#if ENABLE_DEBUG
  llvm::dbgs() << "=== Final Module\n" << M << "=== End Final Module\n";
  if (verifyModule(M, &errs()))
    FATAL_ERROR("Broken module found, JIT compilation aborted!");
  else
    llvm::dbgs() << "Module verified!\n";
#endif
}

template <typename ImplT>
void JitEngineDevice<ImplT>::relinkGlobals(
    llvm::Module &M, std::unordered_map<std::string, const void *> &VarNameToDevPtr) {
  // Re-link globals to fixed addresses provided by registered
  // variables.
  for (auto RegisterVar : VarNameToDevPtr) {
    // For CUDA we must defer resolving the global symbol address here
    // instead when registering the variable in the constructor context.
    void *DevPtr = resolveDeviceGlobalAddr(RegisterVar.second);
    auto &VarName = RegisterVar.first;
    auto *GV = M.getNamedGlobal(VarName);
    // Skip linking if the GV does not exist in the module.
    if (!GV)
      continue;
    // Remove the re-linked global from llvm.compiler.used since that
    // use is not replaceable by the fixed addr constant expression.
    removeFromUsedLists(M, [&GV](llvm::Constant *C) {
      if (GV == C)
        return true;

      return false;
    });

    llvm::Constant *Addr =
        llvm::ConstantInt::get(llvm::Type::getInt64Ty(M.getContext()), (uint64_t)DevPtr);
    llvm::Value *CE = llvm::ConstantExpr::getIntToPtr(Addr, GV->getType());
    GV->replaceAllUsesWith(CE);
  }

#if ENABLE_DEBUG
  llvm::dbgs() << "=== Linked M\n" << M << "=== End of Linked M\n";
  if (verifyModule(M, &errs()))
    FATAL_ERROR("After linking, broken module found, JIT compilation aborted!");
  else
    llvm::dbgs() << "Module verified!\n";
#endif
}

template <typename ImplT>
typename DeviceTraits<ImplT>::DeviceError_t
JitEngineDevice<ImplT>::compileAndRun(
    llvm::StringRef ModuleUniqueId, void *Kernel, llvm::StringRef KernelName,
    const llvm::SmallVector<int32_t> &RCIndices, int NumRuntimeConstants,
    dim3 GridDim, dim3 BlockDim, void **KernelArgs, uint64_t ShmemSize,
    typename DeviceTraits<ImplT>::DeviceStream_t Stream) {
  TIMESCOPE("compileAndRun");

  llvm::SmallVector<RuntimeConstant> RCsVec;

  auto IRBuffer = extractDeviceBitcode(KernelName, Kernel);

  auto parseBitcode = [&]() -> llvm::Expected<llvm::orc::ThreadSafeModule> {
    auto Ctx = std::make_unique<llvm::LLVMContext>();
    llvm::SMDiagnostic Err;
    if (auto M = parseIR(IRBuffer->getMemBufferRef(), Err, *Ctx))
      return llvm::orc::ThreadSafeModule(std::move(M), std::move(Ctx));

    return createSMDiagnosticError(Err);
  };

  auto SafeModule = parseBitcode();
  if (auto E = SafeModule.takeError())
    FATAL_ERROR(toString(std::move(E)).c_str());

  auto *JitModule = SafeModule->getModuleUnlocked();
  getRuntimeConstantsFromModule(*JitModule, KernelArgs, KernelName, RCIndices,
                                RCsVec);

  uint64_t HashValue = CodeCache.hash(ModuleUniqueId, KernelName, RCsVec.data(),
                                      NumRuntimeConstants);
  typename DeviceTraits<ImplT>::KernelFunction_t KernelFunc =
      CodeCache.lookup(HashValue);
  if (KernelFunc)
    return launchKernelFunction(KernelFunc, GridDim, BlockDim, KernelArgs,
                                ShmemSize, Stream);

  // NOTE: we don't need a suffix to differentiate kernels, each specialization
  // will be in its own module uniquely identify by HashValue. It exists only
  // for debugging purposes to verify that the jitted kernel executes.
  std::string Suffix = mangleSuffix(HashValue);
  std::string KernelMangled = (KernelName + Suffix).str();

  torch::jit::script::Module module;
  try {
    torch::Tensor inp = torch::rand({20});
    module = torch::jit::load("/usr/WS2/kundu1/RT_Tuner/examples/ts-ex/my_module_model.pt");
    torch::Tensor out = module.forward({inp}).toTensor();
    std::cout << "Input Tensor:\n" << inp << std::endl;
    std::cout << "Output Tensor:\n" << out << std::endl;
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
  }

  if (Config.ENV_PROTEUS_USE_STORED_CACHE) {
    // If there device global variables, lookup the IR and codegen object
    // before launching. Else, if there aren't device global variables, lookup
    // the object and launch.

    // TODO: Check for globals is very conservative and always re-builds from
    // LLVM IR even if the Jit module does not use global variables.  A better
    // solution is to keep track of whether a kernel uses gvars (store a flag in
    // the cache file?) and load the object in case it does not use any.
    // TODO: Can we use RTC interfaces for fast linking on object files?
    bool HasDeviceGlobals = !VarNameToDevPtr.empty();
    if (auto CacheBuf =
            (HasDeviceGlobals
                 ? StorageCache.lookupBitcode(HashValue, KernelMangled)
                 : StorageCache.lookupObject(HashValue, KernelMangled))) {
      std::unique_ptr<llvm::MemoryBuffer> ObjBuf;
      if (HasDeviceGlobals) {
        auto Ctx = std::make_unique<llvm::LLVMContext>();
        llvm::SMDiagnostic Err;
        auto M = parseIR(CacheBuf->getMemBufferRef(), Err, *Ctx);
        relinkGlobals(*M, VarNameToDevPtr);
        ObjBuf = codegenObject(*M, DeviceArch);
      } else {
        ObjBuf = std::move(CacheBuf);
      }

      auto KernelFunc =
          getKernelFunctionFromImage(KernelMangled, ObjBuf->getBufferStart());

      CodeCache.insert(HashValue, KernelFunc, KernelName, RCsVec.data(),
                       NumRuntimeConstants);

      return launchKernelFunction(KernelFunc, GridDim, BlockDim, KernelArgs,
                                  ShmemSize, Stream);
    }
  }

  specializeIR(*JitModule, KernelName, Suffix, BlockDim, GridDim, RCIndices,
               RCsVec.data(), NumRuntimeConstants);

  // For CUDA, run the target-specific optimization pipeline to optimize the
  // LLVM IR before handing over to the CUDA driver PTX compiler.
  optimizeIR(*JitModule, DeviceArch);

  llvm::SmallString<4096> ModuleBuffer;
  llvm::raw_svector_ostream ModuleBufferOS(ModuleBuffer);
  WriteBitcodeToFile(*JitModule, ModuleBufferOS);
  StorageCache.storeBitcode(HashValue, ModuleBuffer);

  relinkGlobals(*JitModule, VarNameToDevPtr);

  auto ObjBuf = codegenObject(*JitModule, DeviceArch);
  if (Config.ENV_PROTEUS_USE_STORED_CACHE)
    StorageCache.storeObject(HashValue, ObjBuf->getMemBufferRef());

  KernelFunc =
      getKernelFunctionFromImage(KernelMangled, ObjBuf->getBufferStart());

  CodeCache.insert(HashValue, KernelFunc, KernelName, RCsVec.data(),
                   NumRuntimeConstants);

  return launchKernelFunction(KernelFunc, GridDim, BlockDim, KernelArgs,
                              ShmemSize, Stream);
}

template <typename ImplT>
void JitEngineDevice<ImplT>::registerFatBinary(void *Handle,
                                               FatbinWrapper_t *FatbinWrapper,
                                               const char *ModuleId) {
  CurHandle = Handle;
  DBG(llvm::dbgs() << "Register fatbinary Handle " << Handle << " FatbinWrapper "
             << FatbinWrapper << " Binary " << (void *)FatbinWrapper->Binary
             << " ModuleId " << ModuleId << "\n");
  if (FatbinWrapper->PrelinkedFatbins) {
    // This is RDC compilation, just insert the FatbinWrapper and ignore the
    // ModuleId coming from the link.stub.
    HandleToBinaryInfo[Handle] = {FatbinWrapper, {}};

    // Initialize GlobalLinkedBinaries with prelinked fatbins.
    void *Ptr = FatbinWrapper->PrelinkedFatbins[0];
    for (int I = 0; Ptr != nullptr;
         ++I, Ptr = FatbinWrapper->PrelinkedFatbins[I]) {
      DBG(llvm::dbgs() << "I " << I << " PrelinkedFatbin " << Ptr << "\n");
      GlobalLinkedBinaries.insert(Ptr);
    }
  } else {
    // This is non-RDC compilation, associate the ModuleId of the JIT bitcode in
    // the module with the FatbinWrapper.
    ModuleIdToFatBinary[ModuleId] = FatbinWrapper;
    HandleToBinaryInfo[Handle] = {FatbinWrapper, {ModuleId}};
  }
}

template <typename ImplT> void JitEngineDevice<ImplT>::registerFatBinaryEnd() {
  DBG(llvm::dbgs() << "Register fatbinary end\n");
  CurHandle = nullptr;
}

template <typename ImplT>
void JitEngineDevice<ImplT>::registerFunction(void *Handle, void *Kernel,
                                              char *KernelName,
                                              int32_t *RCIndices,
                                              int32_t NumRCs) {
  DBG(llvm::dbgs() << "Register function " << Kernel << " To Handle " << Handle
             << "\n");
  assert(!KernelToHandleMap.contains(Kernel) &&
         "Expected kernel inserted only once in the map");
  KernelToHandleMap[Kernel] = Handle;

  JITKernelInfoMap[Kernel] = JITKernelInfo(KernelName, RCIndices, NumRCs);
}

template <typename ImplT>
void JitEngineDevice<ImplT>::registerLinkedBinary(
    FatbinWrapper_t *FatbinWrapper, const char *ModuleId) {
  DBG(llvm::dbgs() << "Register linked binary FatBinary " << FatbinWrapper
             << " Binary " << (void *)FatbinWrapper->Binary << " ModuleId "
             << ModuleId << "\n");
  if (CurHandle) {
    if (!HandleToBinaryInfo.contains(CurHandle))
      FATAL_ERROR("Expected CurHandle in map");

    HandleToBinaryInfo[CurHandle].LinkedModuleIds.push_back(ModuleId);
  } else
    GlobalLinkedModuleIds.push_back(ModuleId);

  ModuleIdToFatBinary[ModuleId] = FatbinWrapper;
}

template <typename ImplT>
void JitEngineDevice<ImplT>::linkJitModule(
    llvm::Module *M, llvm::LLVMContext *Ctx, llvm::StringRef KernelName,
    llvm::SmallVector<std::unique_ptr<llvm::Module>> &LinkedModules) {
  if (LinkedModules.empty())
    FATAL_ERROR("Expected jit module");

  llvm::Linker IRLinker(*M);
  for (auto &LinkedM : LinkedModules) {
    // Returns true if linking failed.
    if (IRLinker.linkInModule(std::move(LinkedM), llvm::Linker::LinkOnlyNeeded,
                              [&KernelName](llvm::Module &M, const llvm::StringSet<> &GVS) {
                                for (auto &Symbol : GVS) {
                                  if (Symbol.getKey() == KernelName)
                                    continue;

                                  llvm::Function *F = M.getFunction(Symbol.getKey());
                                  if (!F)
                                    continue;

                                  // Internalize functions, the JIT module is
                                  // self-contained.
                                  F->setLinkage(llvm::GlobalValue::InternalLinkage);
                                }
                              }))
      FATAL_ERROR("Linking failed");
  }
}

} // namespace proteus

#endif
