//===-- LLJITWithOptimizingIRTransform.cpp -- LLJIT with IR optimization --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// In this example we will use an IR transform to optimize a module as it
// passes through LLJIT's IRTransformLayer.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Analysis/CFLSteensAliasAnalysis.h"
#include "llvm/Analysis/TypeBasedAliasAnalysis.h"
#include "llvm/Analysis/ScopedNoAliasAA.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/FunctionAttrs.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/SimpleLoopUnswitch.h"
#include "llvm/Transforms/Vectorize.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/AggressiveInstCombine/AggressiveInstCombine.h"

#include "./ExampleModules.h"

#define USE_MAP

using namespace llvm;
using namespace llvm::orc;

// A function object that creates a simple pass pipeline to apply to each
// module as it passes through the IRTransformLayer.
class MyOptimizationTransform {
public:
  MyOptimizationTransform() : PM(std::make_unique<legacy::PassManager>()) {

    PM->add(createCFLSteensAAWrapperPass());
    PM->add(createTypeBasedAAWrapperPass());
    PM->add(createScopedNoAliasAAWrapperPass());
    PM->add(createIPSCCPPass());
    PM->add(createGlobalOptimizerPass());
    PM->add(createPromoteMemoryToRegisterPass());
    PM->add(createGlobalsAAWrapperPass());
    PM->add(createInstructionCombiningPass());
    PM->add(createCFGSimplificationPass());
    PM->add(createFunctionInliningPass());
    PM->add(createPostOrderFunctionAttrsLegacyPass());
    PM->add(createSROAPass());
    PM->add(createEarlyCSEPass(true));
    PM->add(createGVNHoistPass());
    PM->add(createGVNSinkPass());
    PM->add(createCFGSimplificationPass());
    PM->add(createConstraintEliminationPass());
    PM->add(createJumpThreadingPass());
    PM->add(createCorrelatedValuePropagationPass());
    PM->add(createCFGSimplificationPass());
    PM->add(createAggressiveInstCombinerPass());
    PM->add(createInstructionCombiningPass());
    PM->add(createTailCallEliminationPass());
    PM->add(createCFGSimplificationPass());
    PM->add(createReassociatePass());
    PM->add(createVectorCombinePass());
    PM->add(createLoopInstSimplifyPass());
    PM->add(createLoopSimplifyCFGPass());
    PM->add(createLICMPass());
    PM->add(createLoopRotatePass());
    PM->add(createLICMPass());
    PM->add(createSimpleLoopUnswitchLegacyPass());
    PM->add(createCFGSimplificationPass());
    PM->add(createLoopFlattenPass());
    PM->add(createLoopSimplifyCFGPass());
    PM->add(createLoopIdiomPass());
    PM->add(createIndVarSimplifyPass());
    PM->add(createLoopDeletionPass());
    PM->add(createLoopInterchangePass());
    PM->add(createSimpleLoopUnrollPass());
    PM->add(createSROAPass());
    PM->add(createNewGVNPass());
    PM->add(createSCCPPass());
    PM->add(createConstraintEliminationPass());
    PM->add(createBitTrackingDCEPass());
    PM->add(createInstructionCombiningPass());
    PM->add(createDFAJumpThreadingPass());
    PM->add(createJumpThreadingPass());
    PM->add(createCorrelatedValuePropagationPass());
    PM->add(createAggressiveDCEPass());
    PM->add(createMemCpyOptPass());
    PM->add(createDeadStoreEliminationPass());
    PM->add(createLICMPass());
    PM->add(createLoopRerollPass());
    PM->add(createCFGSimplificationPass());
    PM->add(createInstructionCombiningPass());
    PM->add(createPartialInliningPass());
    PM->add(createReversePostOrderFunctionAttrsPass());
    PM->add(createGlobalOptimizerPass());
    PM->add(createGlobalDCEPass());
    PM->add(createLoopVersioningLICMPass());
    PM->add(createLICMPass());
    PM->add(createGlobalsAAWrapperPass());
    PM->add(createFloat2IntPass());
    PM->add(createLowerConstantIntrinsicsPass());
    PM->add(createLowerMatrixIntrinsicsPass());
    PM->add(createEarlyCSEPass(false));
    PM->add(createLoopRotatePass());
    PM->add(createLoopDistributePass());
    PM->add(createLoopVectorizePass());
    PM->add(createLoopLoadEliminationPass());
    PM->add(createInstructionCombiningPass());
    PM->add(createEarlyCSEPass());
    PM->add(createCorrelatedValuePropagationPass());
    PM->add(createInstructionCombiningPass());
    PM->add(createLICMPass());
    PM->add(createSimpleLoopUnswitchLegacyPass());
    PM->add(createCFGSimplificationPass());
    PM->add(createInstructionCombiningPass());
    PM->add(createCFGSimplificationPass(SimplifyCFGOptions()
                                           .forwardSwitchCondToPhi(true)
                                           .convertSwitchRangeToICmp(true)
                                           .convertSwitchToLookupTable(true)
                                           .needCanonicalLoops(false)
                                           .hoistCommonInsts(true)
                                           .sinkCommonInsts(true)));
    PM->add(createSLPVectorizerPass());
    PM->add(createEarlyCSEPass());
    PM->add(createVectorCombinePass());
    PM->add(createInstructionCombiningPass());
    PM->add(createLoopUnrollAndJamPass());
    PM->add(createLoopUnrollPass());
    PM->add(createInstructionCombiningPass());
    PM->add(createLICMPass());
    PM->add(createGlobalDCEPass());
    PM->add(createConstantMergePass());
    PM->add(createLoopSinkPass());
    PM->add(createInstSimplifyLegacyPass());
    PM->add(createDivRemPairsPass());
    PM->add(createCFGSimplificationPass(
        SimplifyCFGOptions().convertSwitchRangeToICmp(true)));

#if 0
    PM->add(createFunctionInliningPass());
    PM->add(createSCCPPass());
    PM->add(createCFGSimplificationPass());
    PM->add(createLoopVectorizePass());
    PM->add(createSLPVectorizerPass());
    PM->add(createCFGSimplificationPass());
#endif

    //PM->add(createTailCallEliminationPass());
    //PM->add(createFunctionInliningPass());
    //PM->add(createIndVarSimplifyPass());
    //PM->add(createCFGSimplificationPass());

    //PM->add(createCFGSimplificationPass());
    //PM->add(createAggressiveDCEPass());
    //PM->add(createEarlyCSEPass());
    //PM->add(createSROAPass());
    //PM->add(createCFGSimplificationPass());
  }

  Expected<ThreadSafeModule> operator()(ThreadSafeModule TSM,
                                        MaterializationResponsibility &R) {
    TSM.withModuleDo([this](Module &M) {
      //dbgs() << "--- BEFORE OPTIMIZATION ---\n" << M << "\n";
      PM->run(M);
      //dbgs() << "--- AFTER OPTIMIZATION ---\n" << M << "\n";
    });
    return std::move(TSM);
  }

private:
  std::unique_ptr<legacy::PassManager> PM;
};

struct RuntimeConstant {
  int ArgNo;
  union {
    int32_t Int32Val;
    int64_t Int64Val;
    float FloatVal;
    double DoubleVal;
  };
};


std::unique_ptr<LLJIT> J;
// TODO: make it a singleton?
class JitEngine {
public:

  ExitOnError ExitOnErr;

  JitEngine(int argc, char **argv)
    {
    InitLLVM X(argc, argv);

    InitializeNativeTarget();
    InitializeNativeTargetAsmPrinter();

    ExitOnErr.setBanner("JIT: ");
#ifdef USE_LINKER
    J = ExitOnErr(LLJITBuilder().create());
    // (2) Resolve symbols in the main process.
    orc::MangleAndInterner Mangle(J->getExecutionSession(), J->getDataLayout());
    J->getMainJITDylib().addGenerator(
        ExitOnErr(orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
            J->getDataLayout().getGlobalPrefix(),
            [MainName = Mangle("main")](const orc::SymbolStringPtr &Name) {
              // dbgs() << "Search name " << Name << "\n";
              return Name != MainName;
            })));

    // (2) Install transform to optimize modules when they're materialized.
    J->getIRTransformLayer().setTransform(MyOptimizationTransform());
#endif

    //dbgs() << "JIT inited\n";
    //getchar();
  }

  JitEngine() {
    char *argv[0];
    JitEngine(0, argv);
  }

  Expected<llvm::orc::ThreadSafeModule>
#ifdef USE_LINKER
  parseSource(StringRef FnName, StringRef Suffix, StringRef IR,
#endif
  parseSource(StringRef FnName, StringRef IR,
              RuntimeConstant *RC, int NumRuntimeConstants) {
    auto Ctx = std::make_unique<LLVMContext>();
    SMDiagnostic Err;
    if (auto M = parseIR(MemoryBufferRef(IR, "Mod"), Err, *Ctx)) {
      // dbgs() << "parsed Module " << *M << "\n";
      Function *F = M->getFunction(FnName);

      // Clone the function and replace argument uses with runtime constants.
      ValueToValueMapTy VMap;
      Function *NewF = cloneFunctionDecl(*M, *F, &VMap);
      moveFunctionBody(*F, VMap, nullptr, NewF);
      for (int I = 0; I < NumRuntimeConstants; ++I) {
        int ArgNo = RC[I].ArgNo;
        Value *Arg = NewF->getArg(ArgNo);
        // TODO: add constant creation for FP types too.
        Type *ArgType = Arg->getType();
        Constant *C = nullptr;
        if (ArgType->isIntegerTy(32)) {
          // dbgs() << "RC is Int32\n";
          C = ConstantInt::get(ArgType, RC[I].Int32Val);
        } else if (ArgType->isIntegerTy(64)) {
          // dbgs() << "RC is Int64\n";
          C = ConstantInt::get(ArgType, RC[I].Int64Val);
        } else if (ArgType->isFloatTy()) {
          // dbgs() << "RC is Float\n";
          C = ConstantFP::get(ArgType, RC[I].FloatVal);
        } else if (ArgType->isDoubleTy()) {
          // dbgs() << "RC is Double\n";
          C = ConstantFP::get(ArgType, RC[I].DoubleVal);
        } else
          report_fatal_error("Incompatible type in runtime constant");

        Arg->replaceAllUsesWith(C);
      }
      F->replaceAllUsesWith(NewF);
      F->eraseFromParent();

#ifdef USE_MAP
#ifdef USE_LINKER
      NewF->setName(FnName + Suffix);
#endif
      NewF->setName(FnName);
#endif
#ifdef USE_LINKER
      NewF->setName(FnName);
      for (Function &F : *M) {
        if (F.isDeclaration())
          continue;
        //dbgs() << "Rename F " << F.getName() << " -> " << F.getName() + Suffix;
        F.setName(F.getName() + Suffix);
      }
#endif
      // dbgs() << "NewF " << *NewF << "\n";
      // getchar();
      // dbgs() << "=== Modified Module\n" << *M << "***\n";
      return ThreadSafeModule(std::move(M), std::move(Ctx));
    }

    return createSMDiagnosticError(Err);
  }

  void *compileAndLink(StringRef FnName, StringRef IR, RuntimeConstant *RC,
                      int NumRuntimeConstants) {

#ifdef USE_MAP
    // (1) Create LLJIT instance.
    auto J = ExitOnErr(LLJITBuilder().create());

    // (2) Resolve symbols in the main process.
    orc::MangleAndInterner Mangle(J->getExecutionSession(), J->getDataLayout());
    J->getMainJITDylib().addGenerator(
        ExitOnErr(orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
            J->getDataLayout().getGlobalPrefix(),
            [MainName = Mangle("main")](const orc::SymbolStringPtr &Name) {
              //dbgs() << "Search name " << Name << "\n";
              return Name != MainName;
            })));

    // (2) Install transform to optimize modules when they're materialized.
    J->getIRTransformLayer().setTransform(MyOptimizationTransform());
#endif

    dbgs() << "======= COMPILING " << FnName << "=====================\n";
#ifdef USE_LINKER
    std::string Suffix = mangleSuffix(FnName, RC, NumRuntimeConstants);
    std::string MangledFnName = FnName.str() + Suffix;
#endif
    // (3) Add modules.
    ExitOnErr(J->addIRModule(
#ifdef USE_LINKER
        ExitOnErr(parseSource(FnName, Suffix, IR, RC, NumRuntimeConstants))));
#endif
        ExitOnErr(parseSource(FnName, IR, RC, NumRuntimeConstants))));

    // (4) Look up the JIT'd function and call it.
    //dbgs() << "Lookup FnName " << FnName << "\n";
#ifdef USE_LINKER
    auto EntryAddr = ExitOnErr(J->lookup(MangledFnName));
#endif
    auto EntryAddr = ExitOnErr(J->lookup(FnName));

    return (void *)EntryAddr.getValue();
  }

#ifdef USE_LINKER
  std::string mangleSuffix(StringRef FnName, RuntimeConstant *RC,
                           int NumRuntimeConstants) {
    // Generate mangled name with runtime constants.
    return "";
    std::string Suffix = ".";
    for (int I = 0; I < NumRuntimeConstants; ++I)
      Suffix += std::to_string(RC[I].Int64Val);
    return Suffix;
  }

  void *lookup(StringRef FnName, RuntimeConstant *RC, int NumRuntimeConstants) {
    std::string Suffix = mangleSuffix(FnName, RC, NumRuntimeConstants);
    std::string MangledFnName = FnName.str() + Suffix;
    auto EntryAddr = J->lookup(MangledFnName);

    if (!EntryAddr)
      return nullptr;

    return (void *)EntryAddr->getValue();
  }
#endif
};

class JitFnKey {
  StringRef FnName;
  SmallVector<RuntimeConstant, 8> RuntimeConstants;

public:
  JitFnKey(char *Fn, RuntimeConstant *RC, int NumRuntimeConstants)
      : FnName(Fn), RuntimeConstants(RC, RC + NumRuntimeConstants) {}
  bool operator<(const JitFnKey &RHS) const {
    //dbgs() << "Compare " << this << " with " << &RHS << "\n";
    //dbgs() << "Compare FnNames " << FnName << " < " << RHS.FnName << "\n";
    if (FnName < RHS.FnName) {
      //dbgs() << "return true\n";
      return true;
    }

    if (FnName > RHS.FnName)
      return false;

    if (RuntimeConstants.size() < RHS.RuntimeConstants.size())
      return true;

    if (RuntimeConstants.size() > RHS.RuntimeConstants.size())
      return false;

    // Functions have the number of runtime constant arguments.
    for (int I = 0; I < RuntimeConstants.size(); ++I) {
      //dbgs() << "Compare I " << I << " " << RuntimeConstants[I].Int64Val << " < " << RHS.RuntimeConstants[I].Int64Val << "\n";
      if (RuntimeConstants[I].Int64Val < RHS.RuntimeConstants[I].Int64Val) {
      //if (RuntimeConstants[I].Int32Val < RHS.RuntimeConstants[I].Int32Val) {
        //dbgs() << "return true\n";
        return true;
      }

      if (RuntimeConstants[I].Int64Val > RHS.RuntimeConstants[I].Int64Val)
        return false;
    }

    //dbgs() << "return false\n";
    return false;
  }

  StringRef getFnName() { return FnName; }
  SmallVectorImpl<RuntimeConstant>& getRuntimeConstants() {
    return RuntimeConstants;
  }
};

raw_ostream& operator<<(raw_ostream& OS, JitFnKey& Key)
{
    OS <<  Key.getFnName() << " with ";
    for(auto &RC : Key.getRuntimeConstants())
      OS << "(" << RC.ArgNo << ", " << RC.Int64Val << ") ";
    OS << "\n";
    return OS;
}

#ifdef KEEP_STATS
int hits = 0;
int total = 0;
#endif

JitEngine *Jit;
__attribute__((constructor))
void InitJit() {
  Jit = new JitEngine();
}

__attribute__((destructor))
void DeleteJit() {
  delete Jit;

#ifdef KEEP_STATS
  printf("hits %d total %d\n", hits, total);
#endif
}

//std::unordered_map<Key, void *> JitCache;
#ifdef USE_MAP
std::map<JitFnKey, void *> JitCache;
#endif
extern "C" {
__attribute__((used,weak))
void *__jit_entry(char *FnName, char *IR, RuntimeConstant *RC,
                  int NumRuntimeConstants) {
#ifdef USE_LINKER
  dbgs() << "v2 NumRuntimeConstants " << NumRuntimeConstants << "\n";
  for (int I = 0; I < NumRuntimeConstants; ++I)
    dbgs() << "RC[" << I << "]: ArgNo=" << RC[I].ArgNo
           << " Value Int32=" << RC[I].Int32Val
           << " Value Int64=" << RC[I].Int64Val
           << " Value Float=" << RC[I].FloatVal
           << " Value Double=" << RC[I].DoubleVal
           << "\n";
#endif

  JitFnKey Key(FnName, RC, NumRuntimeConstants);

#ifdef KEEP_STATS
  total++;
#endif
#ifdef USE_LINKER
  void *JitFnPtr = Jit->lookup(FnName, RC, NumRuntimeConstants);
  if (!JitFnPtr)
    JitFnPtr = Jit->compileAndLink(FnName, IR, RC, NumRuntimeConstants);
#ifdef KEEP_STATS
  else
    hits++;
#endif
#endif

#ifdef USE_MAP
  void *JitFnPtr;
  auto It = JitCache.find(Key);
  if (It == JitCache.end()) {
    JitFnPtr = Jit->compileAndLink(FnName, IR, RC, NumRuntimeConstants);
    JitCache[Key] = JitFnPtr;
    //dbgs() << "New Key " << Key << " inserting...\n";
  }
  else {
    //dbgs() << "Key " << Key << " found! cache hit\n";
    JitFnPtr = It->second;
#ifdef KEEP_STATS
    hits++;
#endif
  }
  //getchar();
#endif

  return JitFnPtr;
}
}
