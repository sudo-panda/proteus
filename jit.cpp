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

#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/CFLSteensAliasAnalysis.h"
#include "llvm/Analysis/DemandedBits.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/ScopedNoAliasAA.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/TypeBasedAliasAnalysis.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/TimeProfiler.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/AggressiveInstCombine/AggressiveInstCombine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/FunctionAttrs.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/SimpleLoopUnswitch.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Vectorize.h"

//#include "./ExampleModules.h"

#define USE_MAP

using namespace llvm;
using namespace llvm::orc;

inline Error createSMDiagnosticError(llvm::SMDiagnostic &Diag) {
  std::string Msg;
  {
    raw_string_ostream OS(Msg);
    Diag.print("", OS);
  }
  return make_error<StringError>(std::move(Msg), inconvertibleErrorCode());
}
// Returns the TargetMachine instance or zero if no triple is provided.
static TargetMachine* GetTargetMachine(Triple TheTriple, StringRef CPUStr,
                                       StringRef FeaturesStr,
                                       const TargetOptions &Options) {
  std::string Error;
  const Target *TheTarget =
      TargetRegistry::lookupTarget(codegen::getMArch(), TheTriple, Error);
  // Some modules don't specify a triple, and this is okay.
  if (!TheTarget) {
    return nullptr;
  }

  return TheTarget->createTargetMachine(
      TheTriple.getTriple(), codegen::getCPUStr(), codegen::getFeaturesStr(),
      Options, codegen::getExplicitRelocModel(),
      codegen::getExplicitCodeModel(), CodeGenOpt::Aggressive);
}

// A function object that creates a simple pass pipeline to apply to each
// module as it passes through the IRTransformLayer.
class MyOptimizationTransform {
public:
  MyOptimizationTransform() : PM(std::make_unique<legacy::PassManager>()) {

#if 0
    PM->add(createTailCallEliminationPass());
    PM->add(createFunctionInliningPass());
    PM->add(createIndVarSimplifyPass());
    PM->add(createCFGSimplificationPass());
    PM->add(createLICMPass());
#endif
  }

  Expected<ThreadSafeModule> operator()(ThreadSafeModule TSM,
                                        MaterializationResponsibility &R) {
    TSM.withModuleDo([this](Module &M) {
      for (Function &F : M) {
        // Set linkonce_odr symbols to external to avoid removing them during
        // optimization.
        if (F.hasLinkOnceODRLinkage())
          F.setLinkage(GlobalValue::ExternalLinkage);
      }

#if 0
      dbgs() << "--- BEFORE OPTIMIZATION ---\n" << M << "\n";
#endif

      Triple ModuleTriple(M.getTargetTriple());
      std::string CPUStr, FeaturesStr;
      TargetMachine *Machine = nullptr;
      const TargetOptions Options =
          codegen::InitTargetOptionsFromCodeGenFlags(ModuleTriple);

      if (ModuleTriple.getArch()) {
        CPUStr = codegen::getCPUStr();
        FeaturesStr = codegen::getFeaturesStr();
        Machine = GetTargetMachine(ModuleTriple, CPUStr, FeaturesStr, Options);
      } else if (ModuleTriple.getArchName() != "unknown" &&
                 ModuleTriple.getArchName() != "") {
        errs() << "unrecognized architecture '"
               << ModuleTriple.getArchName() << "' provided.\n";
        abort();
      }
      std::unique_ptr<TargetMachine> TM(Machine);
      codegen::setFunctionAttributes(CPUStr, FeaturesStr, M);
      TargetLibraryInfoImpl TLII(ModuleTriple);
      legacy::PassManager MPasses;

      MPasses.add(new TargetLibraryInfoWrapperPass(TLII));
      MPasses.add(createTargetTransformInfoWrapperPass(
          TM ? TM->getTargetIRAnalysis() : TargetIRAnalysis()));

      std::unique_ptr<legacy::FunctionPassManager> FPasses;
      FPasses.reset(new legacy::FunctionPassManager(&M));
      FPasses->add(createTargetTransformInfoWrapperPass(
          TM ? TM->getTargetIRAnalysis() : TargetIRAnalysis()));

      if (TM) {
        // FIXME: We should dyn_cast this when supported.
        auto &LTM = static_cast<LLVMTargetMachine &>(*TM);
        Pass *TPC = LTM.createPassConfig(MPasses);
        MPasses.add(TPC);
      }

      unsigned int OptLevel = 3;

      {
        //TimeTraceScope T("Builder");
        PassManagerBuilder Builder;
        Builder.OptLevel = OptLevel;
        Builder.SizeLevel = 0;
        Builder.Inliner = createFunctionInliningPass(OptLevel, 0, false);
        Builder.DisableUnrollLoops = false;
        Builder.LoopVectorize = true;
        Builder.SLPVectorize = true;
        TM->adjustPassManager(Builder);
        Builder.populateFunctionPassManager(*FPasses);
        Builder.populateModulePassManager(MPasses);
      }

      {
        //TimeTraceScope T("RunPassPipeline");
        if (FPasses) {
          FPasses->doInitialization();
          for (Function &F : M)
            FPasses->run(F);
          FPasses->doFinalization();
        }
        MPasses.run(M);
      }
      //PM->run(M);
#if 0
      dbgs() << "--- AFTER OPTIMIZATION ---\n" << M << "\n";
#endif
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


static codegen::RegisterCodeGenFlags CFG;
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
#ifdef USE_MAP
  parseSource(StringRef FnName, StringRef IR,
#endif
              RuntimeConstant *RC, int NumRuntimeConstants) {

    //TimeTraceScope T("parseSource");
    auto Ctx = std::make_unique<LLVMContext>();
    SMDiagnostic Err;
    if (auto M = parseIR(MemoryBufferRef(IR, "Mod"), Err, *Ctx)) {
      //dbgs() << "=== Parsed Module\n" << *M << "=== End of Parsed Module\n";
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
      NewF->setName(FnName);
#endif
#ifdef USE_LINKER
      NewF->setName(FnName + Suffix);
      for (Function &F : *M) {
        if (F.isDeclaration())
          continue;

        if (&F == NewF)
          continue;

        //dbgs() << "Rename F " << F.getName() << " -> " << F.getName() + Suffix;
        F.setName(F.getName() + ".." + NewF->getName());
      }
#endif
      // dbgs() << "NewF " << *NewF << "\n";
      // getchar();
#if 0
      dbgs() << "=== Modified Module\n" << *M << "=== End of Modified Module\n";
      if (verifyModule(*M, &errs()))
        report_fatal_error("Broken module found, JIT compilation aborted!", false);
      else
        dbgs() << "Module verified!\n";
#endif
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

    dbgs() << "======= COMPILING " << FnName << " =====================\n";
    //getchar();
#ifdef USE_LINKER
    std::string Suffix = mangleSuffix(FnName, RC, NumRuntimeConstants);
    std::string MangledFnName = FnName.str() + Suffix;
#endif
    // (3) Add modules.
    ExitOnErr(J->addIRModule(
#ifdef USE_LINKER
        ExitOnErr(parseSource(FnName, Suffix, IR, RC, NumRuntimeConstants))));
#endif
#ifdef USE_MAP
        ExitOnErr(parseSource(FnName, IR, RC, NumRuntimeConstants))));
#endif

    // (4) Look up the JIT'd function and call it.
    //dbgs() << "Lookup FnName " << FnName << "\n";
#ifdef USE_LINKER
    auto EntryAddr = ExitOnErr(J->lookup(MangledFnName));
#endif
#ifdef USE_MAP
    auto EntryAddr = ExitOnErr(J->lookup(FnName));
#endif

    return (void *)EntryAddr.getValue();
  }

#ifdef USE_LINKER
  std::string mangleSuffix(StringRef FnName, RuntimeConstant *RC,
                           int NumRuntimeConstants) {
    // Generate mangled name with runtime constants.
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

    // Functions have the same number of runtime constant arguments.
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

int hits = 0;
int total = 0;

JitEngine *Jit;
__attribute__((constructor))
void InitJit() {
  Jit = new JitEngine();
  //timeTraceProfilerInitialize(500 /* us */, "jit");
}

__attribute__((destructor))
void DeleteJit() {
  //timeTraceProfilerWrite("", "-");
  delete Jit;

  printf("hits %d total %d\n", hits, total);
}

//std::unordered_map<Key, void *> JitCache;
#ifdef USE_MAP
std::map<JitFnKey, void *> JitCache;
#endif
extern "C" {
__attribute__((used))
void *__jit_entry(char *FnName, char *IR, RuntimeConstant *RC,
                  int NumRuntimeConstants) {
  //TimeTraceScope T("__jit_entry");
#if 0
  dbgs() << "FnName " << FnName << " NumRuntimeConstants " << NumRuntimeConstants << "\n";
  for (int I = 0; I < NumRuntimeConstants; ++I)
    dbgs() << "RC[" << I << "]: ArgNo=" << RC[I].ArgNo
           << " Value Int32=" << RC[I].Int32Val
           << " Value Int64=" << RC[I].Int64Val
           << " Value Float=" << RC[I].FloatVal
           << " Value Double=" << RC[I].DoubleVal
           << "\n";
#endif

  total++;
#ifdef USE_LINKER
  void *JitFnPtr;
  {
    //TimeTraceScope T("findLinker");
    JitFnPtr = Jit->lookup(FnName, RC, NumRuntimeConstants);
  }
  if (!JitFnPtr) {
    //TimeTraceScope T("compileAndLink");
    JitFnPtr = Jit->compileAndLink(FnName, IR, RC, NumRuntimeConstants);
  }
  else
    hits++;
#endif

#ifdef USE_MAP
  JitFnKey Key(FnName, RC, NumRuntimeConstants);
  void *JitFnPtr;
  std::map<JitFnKey, void *>::iterator It;
  {
    //TimeTraceScope T("findMap");
    It = JitCache.find(Key);
  }
  if (It == JitCache.end()) {
    //TimeTraceScope T("compileAndLink");
    JitFnPtr = Jit->compileAndLink(FnName, IR, RC, NumRuntimeConstants);
    JitCache[Key] = JitFnPtr;
    //dbgs() << "New Key " << Key << " inserting...\n";
  }
  else {
    //dbgs() << "Key " << Key << " found! cache hit\n";
    JitFnPtr = It->second;
    hits++;
  }
  //getchar();
#endif

  return JitFnPtr;
}
}
