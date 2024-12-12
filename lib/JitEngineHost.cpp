//===-- JitEngineHost.cpp -- JIT Engine for CPU using ORC ---===//
//
// Part of Proteus Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the JitEngine interface for dynamic compilation and optimization
// of CPU code.
//
//===----------------------------------------------------------------------===//

#include <memory>

#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/ExecutionEngine/JITEventListener.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Object/SymbolSize.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Passes/OptimizationLevel.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>

#include "CompilerInterfaceTypes.h"
#include "JitEngineHost.hpp"
#include "TransformArgumentSpecialization.hpp"
#include "Utils.h"

using namespace proteus;
using namespace llvm::orc;

#if ENABLE_HIP || ENABLE_CUDA
#include "CompilerInterfaceDevice.h"
#endif

// A function object that creates a simple pass pipeline to apply to each
// module as it passes through the IRTransformLayer.
class OptimizationTransform {
public:
  OptimizationTransform(JitEngineHost &JitEngineImpl)
      : JitEngineImpl(JitEngineImpl) {}

  llvm::Expected<ThreadSafeModule> operator()(ThreadSafeModule TSM,
                                        MaterializationResponsibility &R) {
    TSM.withModuleDo([this](llvm::Module &M) {
      DBG(llvm::dbgs() << "=== Begin Before Optimization\n"
                 << M << "=== End Before\n");
      TIMESCOPE("Run Optimization Transform");
      JitEngineImpl.optimizeIR(M, llvm::sys::getHostCPUName());
      DBG(llvm::dbgs() << "=== Begin After Optimization\n"
                 << M << "=== End After Optimization\n");
#if ENABLE_DEBUG
      if (verifyModule(M, &errs()))
        FATAL_ERROR(
            "Broken module found after optimization, JIT compilation aborted!");
      else
        llvm::dbgs() << "Module after optimization verified!\n";
#endif
    });
    return std::move(TSM);
  }

  llvm::Expected<ThreadSafeModule> operator()(ThreadSafeModule TSM) {
    TSM.withModuleDo([this](llvm::Module &M) {
      DBG(llvm::dbgs() << "=== Begin Before Optimization\n"
                 << M << "=== End Before\n");
      TIMESCOPE("Run Optimization Transform");
      JitEngineImpl.optimizeIR(M, llvm::sys::getHostCPUName());
      DBG(llvm::dbgs() << "=== Begin After Optimization\n"
                 << M << "=== End After Optimization\n");
#if ENABLE_DEBUG
      if (verifyModule(M, &errs()))
        FATAL_ERROR(
            "Broken module found after optimization, JIT compilation aborted!");
      else
        llvm::dbgs() << "Module after optimization verified!\n";
#endif
    });
    return std::move(TSM);
  }

private:
  JitEngineHost &JitEngineImpl;
};

JitEngineHost &JitEngineHost::instance() {
  static char *args[] = {nullptr};
  static JitEngineHost Jit(0, args);
  return Jit;
}

void JitEngineHost::addStaticLibrarySymbols() {
  // Create a SymbolMap for static symbols.
  SymbolMap SymbolMap;

#if ENABLE_CUDA
  // NOTE: For CUDA codes we link the CUDA runtime statically to access device
  // global variables. So, if the host JIT module uses CUDA functions, we
  // need to resolve them statically in the JIT module's linker.

  // TODO: Instead of manually adding symbols, can we handle
  // materialization errors through a notify callback and automatically add
  // those symbols?

  // Insert symbols in the SymbolMap, disambiguate if needed.
  using CudaLaunchKernelFn =
      cudaError_t (*)(const void *, dim3, dim3, void **, size_t, cudaStream_t);
  auto CudaLaunchKernel = static_cast<CudaLaunchKernelFn>(&cudaLaunchKernel);
  SymbolMap[LLJITPtr->mangleAndIntern("cudaLaunchKernel")] =
      llvm::orc::ExecutorSymbolDef(
          llvm::orc::ExecutorAddr{reinterpret_cast<uintptr_t>(CudaLaunchKernel)},
          llvm::JITSymbolFlags::Exported);

#endif
  // Register the symbol manually.
  cantFail(LLJITPtr->getMainJITDylib().define(absoluteSymbols(SymbolMap)));
}

void JitEngineHost::dumpSymbolInfo(
    const llvm::object::ObjectFile &loadedObj,
    const llvm::RuntimeDyld::LoadedObjectInfo &objInfo) {
  // Dump information about symbols.
  auto pid = llvm::sys::Process::getProcessId();
  std::error_code EC;
  llvm::raw_fd_ostream ofd("/tmp/perf-" + std::to_string(pid) + ".map", EC,
                     llvm::sys::fs::OF_Append);
  if (EC)
    FATAL_ERROR("Cannot open perf map file");
  for (auto symSizePair : llvm::object::computeSymbolSizes(loadedObj)) {
    auto sym = symSizePair.first;
    auto size = symSizePair.second;
    auto symName = sym.getName();
    // Skip any unnamed symbols.
    if (!symName || symName->empty())
      continue;
    // The relative address of the symbol inside its section.
    auto symAddr = sym.getAddress();
    if (!symAddr)
      continue;
    // The address the functions was loaded at.
    auto loadedSymAddress = *symAddr;
    auto symbolSection = sym.getSection();
    if (symbolSection) {
      // Compute the load address of the symbol by adding the section load
      // address.
      loadedSymAddress += objInfo.getSectionLoadAddress(*symbolSection.get());
    }
    llvm::outs() << llvm::format("Address range: [%12p, %12p]", loadedSymAddress,
                     loadedSymAddress + size)
           << "\tSymbol: " << *symName << "\n";

    if (size > 0)
      ofd << llvm::format("%lx %x)", loadedSymAddress, size) << " " << *symName
          << "\n";
  }

  ofd.close();
}

void JitEngineHost::notifyLoaded(MaterializationResponsibility &R,
                                 const llvm::object::ObjectFile &Obj,
                                 const llvm::RuntimeDyld::LoadedObjectInfo &LOI) {
  dumpSymbolInfo(Obj, LOI);
}

JitEngineHost::~JitEngineHost() { CodeCache.printStats(); }

llvm::Expected<llvm::orc::ThreadSafeModule>
JitEngineHost::specializeIR(llvm::StringRef FnName, llvm::StringRef Suffix, llvm::StringRef IR,
                            RuntimeConstant *RC, int NumRuntimeConstants) {
  TIMESCOPE("specializeIR");
  auto Ctx = std::make_unique<llvm::LLVMContext>();
  llvm::SMDiagnostic Err;
  if (auto M = parseIR(llvm::MemoryBufferRef(IR, ("Mod-" + FnName + Suffix).str()),
                       Err, *Ctx)) {
    // llvm::dbgs() << "=== Parsed Module\n" << *M << "=== End of Parsed Module\n ";
    llvm::Function *F = M->getFunction(FnName);
    assert(F && "Expected non-null function!");

    // Find GlobalValue declarations that are externally defined. Resolve them
    // statically as absolute symbols in the ORC linker. Required for resolving
    // __jit_launch_kernel for a host JIT function when libproteus is compiled
    // as a static library. For other non-resolved symbols, return a fatal error
    // to investigate.
    for (auto &GV : M->global_values()) {
      if (!GV.isDeclaration())
        continue;

      if (llvm::Function *F = llvm::dyn_cast<llvm::Function>(&GV))
        if (F->isIntrinsic())
          continue;

      auto ExecutorAddr = LLJITPtr->lookup(GV.getName());
      auto Error = ExecutorAddr.takeError();
      if (!Error)
        continue;
      // Consume the error and fix with static linking.
      consumeError(std::move(Error));

      DBG(llvm::dbgs() << "Resolve statically missing GV symbol " << GV.getName()
                 << "\n");

#if ENABLE_CUDA || ENABLE_HIP
      if (GV.getName() == "__jit_launch_kernel") {
        DBG(llvm::dbgs() << "Resolving via ORC jit_launch_kernel\n");
        SymbolMap SymbolMap;
        SymbolMap[LLJITPtr->mangleAndIntern("__jit_launch_kernel")] =
            llvm::orc::ExecutorSymbolDef(
                llvm::orc::ExecutorAddr{
                    reinterpret_cast<uintptr_t>(__jit_launch_kernel)},
                llvm::JITSymbolFlags::Exported);

        cantFail(
            LLJITPtr->getMainJITDylib().define(absoluteSymbols(SymbolMap)));

        continue;
      }
#endif

      FATAL_ERROR("Unknown global value" + GV.getName() + " to resolve");
    }
    // Replace argument uses with runtime constants.
    // TODO: change NumRuntimeConstants to size_t at interface.
    llvm::MDNode *Node = F->getMetadata("jit_arg_nos");
    assert(Node && "Expected metata for jit argument positions");
    DBG(llvm::dbgs() << "Metadata jit for F " << F->getName() << " = " << *Node
               << "\n");

    // Replace argument uses with runtime constants.
    llvm::SmallVector<int32_t> ArgPos;
    for (int I = 0; I < Node->getNumOperands(); ++I) {
      llvm::ConstantAsMetadata *CAM = llvm::cast<llvm::ConstantAsMetadata>(Node->getOperand(I));
      llvm::ConstantInt *ConstInt = llvm::cast<llvm::ConstantInt>(CAM->getValue());
      int ArgNo = ConstInt->getZExtValue();
      ArgPos.push_back(ArgNo);
    }

    TransformArgumentSpecialization::transform(
        *M, *F, ArgPos,
        llvm::ArrayRef<RuntimeConstant>{RC,
                                  static_cast<size_t>(NumRuntimeConstants)});

    // llvm::dbgs() << "=== JIT Module\n" << *M << "=== End of JIT Module\n";

    F->setName(FnName + Suffix);

#if ENABLE_DEBUG
    llvm::dbgs() << "=== Final Module\n" << *M << "=== End Final Module\n";
    if (verifyModule(*M, &errs()))
      FATAL_ERROR("Broken module found, JIT compilation aborted!");
    else
      llvm::dbgs() << "Module verified!\n";
#endif
    return ThreadSafeModule(std::move(M), std::move(Ctx));
  }

  return createSMDiagnosticError(Err);
}

void *JitEngineHost::compileAndLink(llvm::StringRef FnName, char *IR, int IRSize,
                                    RuntimeConstant *RC,
                                    int NumRuntimeConstants) {
  TIMESCOPE("compileAndLink");

  // TODO: implement ModuleUniqueId for host code.
  uint64_t HashValue = CodeCache.hash("", FnName, RC, NumRuntimeConstants);
  void *JitFnPtr = CodeCache.lookup(HashValue);
  if (JitFnPtr)
    return JitFnPtr;

  std::string Suffix = mangleSuffix(HashValue);
  std::string MangledFnName = FnName.str() + Suffix;

  llvm::StringRef StrIR(IR, IRSize);
  // (3) Add modules.
  ExitOnErr(LLJITPtr->addIRModule(
      ExitOnErr(specializeIR(FnName, Suffix, StrIR, RC, NumRuntimeConstants))));

  DBG(llvm::dbgs() << "===\n"
             << *LLJITPtr->getExecutionSession().getSymbolStringPool()
             << "===\n");

  // (4) Look up the JIT'd function.
  DBG(llvm::dbgs() << "Lookup FnName " << FnName << " mangled as " << MangledFnName
             << "\n");
  auto EntryAddr = ExitOnErr(LLJITPtr->lookup(MangledFnName));

  JitFnPtr = (void *)EntryAddr.getValue();
  DBG(llvm::dbgs() << "FnName " << FnName << " Mangled " << MangledFnName
             << " address " << JitFnPtr << "\n");
  assert(JitFnPtr && "Expected non-null JIT function pointer");
  CodeCache.insert(HashValue, JitFnPtr, FnName, RC, NumRuntimeConstants);

  llvm::dbgs() << "=== JIT compile: " << FnName << " Mangled " << MangledFnName
         << " RC HashValue " << HashValue << " Addr " << JitFnPtr << "\n";
  return JitFnPtr;
}

JitEngineHost::JitEngineHost(int argc, char *argv[]) {
  llvm::InitLLVM X(argc, argv);

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  ExitOnErr.setBanner("JIT: ");
  // Create the LLJIT instance.
  // TODO: Fix support for debugging jitted code. This appears to be
  // the correct interface (see orcv2 examples) but it does not work.
  // By dumpSymbolInfo() the debug sections are not populated. Why?
  LLJITPtr =
      ExitOnErr(LLJITBuilder()
                    .setObjectLinkingLayerCreator([&](ExecutionSession &ES,
                                                      const llvm::Triple &TT) {
                      auto GetMemMgr = []() {
                        return std::make_unique<llvm::SectionMemoryManager>();
                      };
                      auto ObjLinkingLayer =
                          std::make_unique<RTDyldObjectLinkingLayer>(
                              ES, std::move(GetMemMgr));

                      // Register the event listener.
                      ObjLinkingLayer->registerJITEventListener(
                          *llvm::JITEventListener::createGDBRegistrationListener());

                      // Make sure the debug info sections aren't stripped.
                      ObjLinkingLayer->setProcessAllSections(true);

#if defined(ENABLE_DEBUG) || defined(ENABLE_PERFMAP)
                      ObjLinkingLayer->setNotifyLoaded(notifyLoaded);
#endif
                      return ObjLinkingLayer;
                    })
                    .create());
  // (2) Resolve symbols in the main process.
  llvm::orc::MangleAndInterner Mangle(LLJITPtr->getExecutionSession(),
                                LLJITPtr->getDataLayout());
  LLJITPtr->getMainJITDylib().addGenerator(
      ExitOnErr(llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
          LLJITPtr->getDataLayout().getGlobalPrefix(),
          [MainName = Mangle("main")](const llvm::orc::SymbolStringPtr &Name) {
            // llvm::dbgs() << "Search name " << Name << "\n";
            return Name != MainName;
          })));

  // Add static library functions for JIT linking.
  addStaticLibrarySymbols();

  // (3) Install transform to optimize modules when they're materialized.
  LLJITPtr->getIRTransformLayer().setTransform(OptimizationTransform(*this));
}
