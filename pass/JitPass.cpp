//=============================================================================
// FILE:
//    JitPass.cpp
//
// DESCRIPTION:
//    Find functions annotated with "jit" plus input arguments that are
//    amenable to runtime constant propagation. Stores the IR for those
//    functions, replaces them with a stub function that calls the jit runtime
//    library to compile the IR and call the function pointer of the jit'ed
//    version.
//
// USAGE:
//    1. Legacy PM
//      opt -enable-new-pm=0 -load libJitPass.dylib -legacy-jit-pass
//      -disable-output `\`
//        <input-llvm-file>
//    2. New PM
//      opt -load-pass-plugin=libJitPass.dylib -passes="jit-pass" `\`
//        -disable-output <input-llvm-file>
//
//
// License: MIT
//=============================================================================
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/ExecutionEngine/Orc/IndirectionUtils.h"
#include "llvm/Pass.h"
#include "llvm/Analysis/CallGraph.h"

#include <iostream>

//#define ENABLE_RECURSIVE_JIT

using namespace llvm;

//-----------------------------------------------------------------------------
// JitPass implementation
//-----------------------------------------------------------------------------
// No need to expose the internals of the pass to the outside world - keep
// everything in an anonymous namespace.
namespace {

struct JitFunctionInfo {
  Function *Fn;
  SmallVector<int, 8> ConstantArgs;
  std::string ModuleIR;
};

SmallVector<JitFunctionInfo, 8> JitFunctionInfoList;

void parseAnnotations(Module &M) {
  auto GlobalAnnotations = M.getNamedGlobal("llvm.global.annotations");
  if (GlobalAnnotations) {
    auto Array = cast<ConstantArray>(GlobalAnnotations->getOperand(0));
    dbgs() << "Array " << *Array << "\n";
    for (int i = 0; i < Array->getNumOperands(); i++) {
      auto Entry = cast<ConstantStruct>(Array->getOperand(i));
      dbgs() << "Entry " << *Entry << "\n";

      auto Fn = dyn_cast<Function>(Entry->getOperand(0));

      if (!Fn)
        continue;

      for (auto &JFI : JitFunctionInfoList)
        if (JFI.Fn == Fn)
          report_fatal_error("Duplicate jit annotation for Fn " + Fn->getName(),
                             false);

      dbgs() << "Function " << Fn->getName() << "\n";

      auto Annotation = cast<ConstantDataArray>(Entry->getOperand(1)->getOperand(0));

      dbgs() << "Annotation " << Annotation->getAsCString() << "\n";

      // TODO: needs CString for comparison to work, why?
      if (Annotation->getAsCString().compare("jit"))
        continue;

      JitFunctionInfo JFI;
      JFI.Fn = Fn;

      if (Entry->getOperand(4)->isNullValue())
        JFI.ConstantArgs = {};
      else {
        dbgs() << "AnnotArgs " << *Entry->getOperand(4)->getOperand(0) << "\n";
        dbgs() << "Type AnnotArgs " << *Entry->getOperand(4)->getOperand(0)->getType() << "\n";
        auto AnnotArgs =
            cast<ConstantStruct>(Entry->getOperand(4)->getOperand(0));

        for (int I = 0; I < AnnotArgs->getNumOperands(); ++I) {
          auto *Index = cast<ConstantInt>(AnnotArgs->getOperand(I));
          // TODO: think about types, check within function arguments bounds, -1
          // to convert to 0-start index.
          JFI.ConstantArgs.push_back(Index->getValue().getZExtValue() - 1);
        }
      }

      JitFunctionInfoList.push_back(JFI);
    }
  }
}

void getReachableFunctions(Module &M, Function &F,
                           SmallPtrSetImpl<Function *> &ReachableFunctions) {
  SmallVector<Function *, 8> ToVisit;
  ToVisit.push_back(&F);
  CallGraphWrapperPass CG;
  CG.runOnModule(M);
  while (!ToVisit.empty()) {
    Function *VisitF = ToVisit.pop_back_val();
    CallGraphNode *CGNode = CG[VisitF];

    for (const auto &Callee : *CGNode) {
      Function *CalleeF = Callee.second->getFunction();

      if (!CalleeF) {
        dbgs() << "Skip external node\n";
        continue;
      }

      if (CalleeF->isDeclaration()) {
        dbgs() << "Skip declaration of " << CalleeF->getName() << "\n";
        continue;
      }

      if (ReachableFunctions.contains(CalleeF)) {
        dbgs() << "Skip already visited " << CalleeF->getName() << "\n";
        continue;
      }

      dbgs() << "Found reachable " << CalleeF->getName() << " ... to visit\n";
      ReachableFunctions.insert(CalleeF);
      ToVisit.push_back(CalleeF);
    }
  }
}

// This method implements what the pass does
void visitor(Module &M, CallGraph &CG) {

  if (JitFunctionInfoList.empty()) {
    //dbgs() << "=== Begin Mod\n" << M << "=== End Mod\n";
    return;
  }

  //dbgs() << "=== Pre M\n" << M << "=== End of Pre M\n";

  // First pass creates the string Module IR per jit'ed function.
  for (JitFunctionInfo &JFI : JitFunctionInfoList) {
    Function *F = JFI.Fn;

    SmallPtrSet<Function *, 16> ReachableFunctions;
    getReachableFunctions(M, *F, ReachableFunctions);
    ReachableFunctions.insert(F);

    ValueToValueMapTy VMap;
    auto JitMod = CloneModule(
        M, VMap, [&ReachableFunctions,&F](const GlobalValue *GV) {
          if (const GlobalVariable *G = dyn_cast<GlobalVariable>(GV)) {
            if (!G->isConstant())
              return false;
          }

          // TODO: do not clone aliases' definitions, it this sound?
          if (const GlobalAlias *GA = dyn_cast<GlobalAlias>(GV))
            return false;

          if (const Function *OrigF = dyn_cast<Function>(GV)) {
            if (OrigF == F) {
              dbgs() << "OrigF " << OrigF->getName() << " == " << F->getName() << ", definitely keep\n";
              return true;
            }
            // Do not keep definitions of unreachable functions.
            if (!ReachableFunctions.contains(OrigF)) {
              //dbgs() << "Drop unreachable " << F->getName() << "\n";
              return false;
            }

#ifdef ENABLE_RECURSIVE_JIT
            // Enable recursive jit'ing.
            for (auto &JFIInner : JitFunctionInfoList)
              if (JFIInner.Fn == OrigF) {
                dbgs() << "Do not keep definitions of another jit function " << OrigF->getName() << "\n";
                return false;
              }
#endif

            // dbgs() << "Keep reachable " << F->getName() << "\n";
            // getchar();
          }

          // By default, clone the definition.
          return true;
        });

    Function *JitF = cast<Function>(VMap[F]);
    JitF->setLinkage(GlobalValue::ExternalLinkage);

    // Set global variables to external linkage when they are not constant or
    // llvm intrinsics.
    for (auto &GV : M.global_values()) {
      if (VMap[&GV])
        if (auto *GVar = dyn_cast<GlobalVariable>(&GV)) {
          if (GVar->isConstant())
            continue;
          if (GVar->getSection() == "llvm.metadata")
            continue;
          if (GVar->getName() == "llvm.global_ctors")
            continue;
          if (GVar->isDSOLocal())
            continue;
          GV.setLinkage(GlobalValue::ExternalLinkage);
          //dbgs() << "=== GV\n";
          //dbgs() << GV << "\n";
          //dbgs() << "Linkage " << GV.getLinkage() << "\n";
          //dbgs() << "Visibility " << GV.getVisibility() << "\n";
          //dbgs() << "Make " << GV << " External\n";
          //dbgs() << "=== End GV\n";
          //getchar();
        }
    }
#ifdef ENABLE_RECURSIVE_JIT
    // Set linkage to external for any reachable jit'ed function to enable
    // recursive jit'ing.
    for (auto &JFIInner : JitFunctionInfoList) {
      if (!ReachableFunctions.contains(JFIInner.Fn))
        continue;
      Function *JitF = cast<Function>(VMap[JFIInner.Fn]);
      JitF->setLinkage(GlobalValue::ExternalLinkage);
    }
    F->setLinkage(GlobalValue::ExternalLinkage);
#endif
    // TODO: Do we want to keep debug info?
    StripDebugInfo(*JitMod);

    if (verifyModule(*JitMod, &errs()))
      report_fatal_error("Broken module found, compilation aborted!", false);
    else
      dbgs() << "JitMod verified!\n";

    // TODO: is writing/reading the bitcode instead of the textual IR faster?
    raw_string_ostream OS(JFI.ModuleIR);
    OS << *JitMod;
    OS.flush();

    dbgs() << "=== StrIR\n" << JFI.ModuleIR << "=== End of StrIR\n";
    //dbgs() << "=== Post M\n" << M << "=== End of Post M\n";
  }

  // Create jit entry runtime function.
  Type *VoidPtrTy = Type::getInt8PtrTy(M.getContext());
  Type *Int32Ty = Type::getInt32Ty(M.getContext());
  Type *Int64Ty = Type::getInt64Ty(M.getContext());
  // Use Int64 type for the Value, big enough to hold primitives.
  StructType *RuntimeConstantTy =
      StructType::create({Int32Ty, Int64Ty}, "struct.args");

  // TODO: This works by embedding the jit.bc library.
  // Function *JitEntryFn = M.getFunction("__jit_entry");
  // assert(JitEntryFn && "Expected non-null JitEntryFn");
  // FunctionType *JitEntryFnTy = JitEntryFn->getFunctionType();
  FunctionType *JitEntryFnTy = FunctionType::get(
      VoidPtrTy,
      {VoidPtrTy, VoidPtrTy, RuntimeConstantTy->getPointerTo(), Int32Ty},
      /* isVarArg=*/false);
  Function *JitEntryFn = Function::Create(
      JitEntryFnTy, GlobalValue::ExternalLinkage, "__jit_entry", M);

  // Second pass replaces jit'ed functions in the original module with stubs to
  // call the runtime entry point that compiles and links.
  for (JitFunctionInfo &JFI : JitFunctionInfoList) {
    Function *F = JFI.Fn;

    // Replace jit'ed function with a stub function.
    std::string FnName = F->getName().str();
    F->setName("");
    Function *StubFn =
        Function::Create(F->getFunctionType(), F->getLinkage(), FnName, M);
    F->replaceAllUsesWith(StubFn);
    F->eraseFromParent();

    // Replace the body of the jit'ed function to call the jit entry, grab the
    // address of the specialized jit version and execute it.
    IRBuilder<> Builder(BasicBlock::Create(M.getContext(), "entry", StubFn));

    // Create the runtime constant array type for the runtime constants passed
    // to the jit entry function.
    ArrayType *RuntimeConstantArrayTy =
        ArrayType::get(RuntimeConstantTy, JFI.ConstantArgs.size());

    // Create globals for the function name and string IR passed to the jit
    // entry.
    auto *FnNameGlobal = Builder.CreateGlobalString(StubFn->getName());
    auto *StrIRGlobal = Builder.CreateGlobalString(JFI.ModuleIR);

    // Create the runtime constants data structure passed to the jit entry.
    auto *RuntimeConstantsAlloca = Builder.CreateAlloca(RuntimeConstantArrayTy);
    // Zero-initialize the alloca to avoid stack garbage for caching.
    Builder.CreateStore(Constant::getNullValue(RuntimeConstantArrayTy),
                        RuntimeConstantsAlloca);
    for (int ArgI = 0; ArgI < JFI.ConstantArgs.size(); ++ArgI) {
      auto *GEP = Builder.CreateInBoundsGEP(
          RuntimeConstantArrayTy, RuntimeConstantsAlloca,
          {Builder.getInt32(0), Builder.getInt32(ArgI)});
      auto *GEPArgNo = Builder.CreateStructGEP(RuntimeConstantTy, GEP, 0);

      int ArgNo = JFI.ConstantArgs[ArgI];
      Builder.CreateStore(Builder.getInt32(ArgNo), GEPArgNo);

      auto *GEPValue = Builder.CreateStructGEP(RuntimeConstantTy, GEP, 1);
      Builder.CreateStore(StubFn->getArg(ArgNo), GEPValue);
    }

    auto *JitFnPtr =
        Builder.CreateCall(JitEntryFnTy, JitEntryFn,
                           {FnNameGlobal, StrIRGlobal, RuntimeConstantsAlloca,
                            Builder.getInt32(JFI.ConstantArgs.size())});
    SmallVector<Value *, 8> Args;
    for (auto &Arg : StubFn->args())
      Args.push_back(&Arg);
    auto *RetVal =
        Builder.CreateCall(StubFn->getFunctionType(), JitFnPtr, Args);
    if (StubFn->getReturnType()->isVoidTy())
      Builder.CreateRetVoid();
    else
      Builder.CreateRet(RetVal);

    // dbgs() << "=== StubFn " << *StubFn << "=== End of StubFn\n";
    // getchar();
  }

  //dbgs() << "=== Begin Mod\n" << M << "=== End Mod\n";
  if (verifyModule(M, &errs()))
    report_fatal_error("Broken module found, compilation aborted!", false);
  else
    dbgs() << "Module verified!\n";
}

// New PM implementation
struct JitPass : PassInfoMixin<JitPass> {
  // Main entry point, takes IR unit to run the pass on (&F) and the
  // corresponding pass manager (to be queried if need be)
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM) {
    parseAnnotations(M);
    CallGraph &CG = AM.getResult<CallGraphAnalysis>(M);
    visitor(M, CG);
    // TODO: is anything preserved?
    return PreservedAnalyses::none();
    //return PreservedAnalyses::all();
  }

  // Without isRequired returning true, this pass will be skipped for functions
  // decorated with the optnone LLVM attribute. Note that clang -O0 decorates
  // all functions with optnone.
  static bool isRequired() { return true; }
};

// Legacy PM implementation
struct LegacyJitPass : public ModulePass {
  static char ID;
  LegacyJitPass() : ModulePass(ID) {}
  // Main entry point - the name conveys what unit of IR this is to be run on.
  bool runOnModule(Module &M) override {
    parseAnnotations(M);
    CallGraph &CG = getAnalysis<CallGraphWrapperPass>().getCallGraph();
    visitor(M, CG);

    // TODO: what is preserved?
    return true;
    // Doesn't modify the input unit of IR, hence 'false'
    //return false;
  }
};
} // namespace

//-----------------------------------------------------------------------------
// New PM Registration
//-----------------------------------------------------------------------------
llvm::PassPluginLibraryInfo getJitPassPluginInfo() {
  const auto callback = [](PassBuilder &PB) {
    // TODO: decide where to insert it in the pipeline. Early avoids
    // inlining jit function (which disables jit'ing) but may require more
    // optimization, hence overhead, at runtime.
    //PB.registerPipelineStartEPCallback([&](ModulePassManager &MPM, auto) {
    PB.registerPipelineEarlySimplificationEPCallback( [&](ModulePassManager &MPM, auto) {
    // XXX: LastEP can break jit'ing, jit function is inlined!
    //PB.registerOptimizerLastEPCallback([&](ModulePassManager &MPM, auto) {
      MPM.addPass(JitPass());
      return true;
    });
  };

  return {LLVM_PLUGIN_API_VERSION, "JitPass", LLVM_VERSION_STRING, callback};
}

// TODO: use by jit-pass name.
// This is the core interface for pass plugins. It guarantees that 'opt' will
// be able to recognize JitPass when added to the pass pipeline on the
// command line, i.e. via '-passes=jit-pass'
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getJitPassPluginInfo();
}

//-----------------------------------------------------------------------------
// Legacy PM Registration
//-----------------------------------------------------------------------------
// The address of this variable is used to uniquely identify the pass. The
// actual value doesn't matter.
char LegacyJitPass::ID = 0;

// This is the core interface for pass plugins. It guarantees that 'opt' will
// recognize LegacyJitPass when added to the pass pipeline on the command
// line, i.e.  via '--legacy-jit-pass'
static RegisterPass<LegacyJitPass>
    X("legacy-jit-pass", "Jit Pass",
      false, // This pass doesn't modify the CFG => false
      false // This pass is not a pure analysis pass => false
    );
