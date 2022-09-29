//=============================================================================
// FILE:
//    HelloWorld.cpp
//
// DESCRIPTION:
//    Visits all functions in a module, prints their names and the number of
//    arguments via stderr. Strictly speaking, this is an analysis pass (i.e.
//    the functions are not modified). However, in order to keep things simple
//    there's no 'print' method here (every analysis pass should implement it).
//
// USAGE:
//    1. Legacy PM
//      opt -enable-new-pm=0 -load libHelloWorld.dylib -legacy-hello-world -disable-output `\`
//        <input-llvm-file>
//    2. New PM
//      opt -load-pass-plugin=libHelloWorld.dylib -passes="hello-world" `\`
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

#include <iostream>

using namespace llvm;

//-----------------------------------------------------------------------------
// HelloWorld implementation
//-----------------------------------------------------------------------------
// No need to expose the internals of the pass to the outside world - keep
// everything in an anonymous namespace.
namespace {

SmallVector<Function *, 8> JitFunctions;
SmallVector<SmallVector<int, 8>, 8> ConstantArgsList;

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

      dbgs() << "Function " << Fn->getName() << "\n";

      auto Annotation = cast<ConstantDataArray>(Entry->getOperand(1)->getOperand(0));

      dbgs() << "Annotation " << Annotation->getAsCString() << "\n";

      // TODO: needs CString for comparison to work, why?
      if (Annotation->getAsCString().compare("jit"))
        continue;

      if (Entry->getOperand(4)->isNullValue())
        ConstantArgsList.push_back({});
      else {
        dbgs() << "AnnotArgs " << *Entry->getOperand(4)->getOperand(0) << "\n";
        dbgs() << "Type AnnotArgs " << *Entry->getOperand(4)->getOperand(0)->getType() << "\n";
        auto AnnotArgs =
            cast<ConstantStruct>(Entry->getOperand(4)->getOperand(0));

        SmallVector<int, 8> ConstantArgs;
        for (int I = 0; I < AnnotArgs->getNumOperands(); ++I) {
          auto *Index = cast<ConstantInt>(AnnotArgs->getOperand(I));
          // TODO: think about types, check within function arguments bounds, -1
          // to convert to 0-start index.
          ConstantArgs.push_back(Index->getValue().getZExtValue() - 1);
        }
        ConstantArgsList.push_back(std::move(ConstantArgs));
      }

      JitFunctions.push_back(Fn);
    }
  }
}

// This method implements what the pass does
void visitor(Module &M) {

  if (JitFunctions.empty())
    return;

  dbgs() << "=== Starting M\n" << M << "=== End of Starting M\n";
  // Clone the module to avoid jit stubs in other functions.
  // TODO: maybe we want jit stubs?
  auto OriginalM = CloneModule(M);

  // Create jit entry runtime function.
  Type *VoidPtrTy = Type::getInt8PtrTy(M.getContext());
  Type *Int32Ty = Type::getInt32Ty(M.getContext());
  Type *Int64Ty = Type::getInt64Ty(M.getContext());
  // Use Int64 type for the Value, big enough to hold primitives.
  StructType *RuntimeConstantTy =
      StructType::create({Int32Ty, Int64Ty}, "struct.args");

  FunctionType *JitEntryFnTy = FunctionType::get(
      VoidPtrTy,
      {VoidPtrTy, VoidPtrTy, RuntimeConstantTy->getPointerTo(), Int32Ty},
      /* isVarArg=*/false);
  Function *JitEntryFn = Function::Create(
      JitEntryFnTy, GlobalValue::ExternalLinkage, "__jit_entry", M);

  int Idx = 0;

  SmallVector<Function *, 8> FunctionsToBeDeleted;
  for (Function *F : JitFunctions) {
    dbgs() << "SIZE " << ConstantArgsList[Idx].size() << "\n";
    // TODO: The current design copies over only the function and globals
    // (constants are copied, others are extern'ed). It might be beneficial for
    // optimization to copy the whole module excluding functions/globals that
    // are not referenced by any of the jit-annotated functions. Whole module
    // copying should happen only one per parent module.
    auto JitMod = CloneModule(*OriginalM);

    Function *MainF = JitMod->getFunction("main");
    if (MainF)
      MainF->eraseFromParent();
    // TODO: Do we want to strip debug info?
    StripDebugInfo(*JitMod);
    SmallVector<GlobalVariable *, 8> ToRemove, ToExtern;
    for (auto &Global : JitMod->globals()) {
      if (Global.isConstant())
        continue;

      // TODO: Do we want to remove llvm.metadata section'ed data?
      if (Global.getSection() == "llvm.metadata") {
        ToRemove.push_back(&Global);
        continue;
      }

      ToExtern.push_back(&Global);
    }

    for(GlobalVariable *Global : ToExtern) {
      StringRef Name = Global->getName();
      // Remove name to avoid conflict with extern definition.
      Global->setName("");
      auto *GV = new GlobalVariable(
          *JitMod, Global->getType(), /* isConstant */ false,
          GlobalVariable::ExternalLinkage, nullptr, Name, nullptr,
          Global->getThreadLocalMode(), Global->getAddressSpace(),
          Global->isExternallyInitialized());
      Global->replaceAllUsesWith(GV);
      Global->eraseFromParent();
    }

    for(GlobalVariable *Global : ToRemove)
      Global->eraseFromParent();

    #if 0
    auto JitMod = std::make_unique<Module>("jit", M.getContext());
    Function *CloneF = Function::Create(F->getFunctionType(), F->getLinkage(),
                                        F->getName() + ".clone", *JitMod);
    ValueToValueMapTy VMap;
    // TODO: fill Returns?
    SmallVector<ReturnInst *, 8> Returns;
    CloneFunctionInto(CloneF, F, VMap, CloneFunctionChangeType::ClonedModule,
                      Returns);
    auto Attrs = CloneF->getAttributes();

    for (auto &Global : M.globals()) {
      if (Global.getSection() == "llvm.metadata")
        continue;

      GlobalVariable *GV = new GlobalVariable(
          *JitMod, Global.getValueType(), Global.isConstant(),
          (Global.isConstant() ? Global.getLinkage()
                               : GlobalVariable::ExternalLinkage),
          Global.getInitializer(), Global.getName(), (GlobalVariable *)nullptr,
          Global.getThreadLocalMode(), Global.getType()->getAddressSpace());
      GV->copyAttributesFrom(&Global);
      // VMap[&*I] = GV;
      dbgs() << "Global " << Global << "\n";
      getchar();
    }
    // Remove attributes, debug info and store IR in string.
    for(auto &A : Attrs)
      CloneF->removeFnAttrs(A);
    StripDebugInfo(*JitMod);
    #endif

    if (verifyModule(*JitMod, &errs()))
      report_fatal_error("Broken module found, compilation aborted!", false);
    else
      dbgs() << "JitMod verified!\n";

    // TODO: is saving the bitcode instead of the textual IR faster?
    std::string StrIR;
    raw_string_ostream OS(StrIR);
    OS << *JitMod;
    OS.flush();

    dbgs() << "=== StrIR\n" << StrIR << "=== End of StrIR\n";

    // Replace the body of F.
    // TODO: is clear enough or should delete?
    //F->getBasicBlockList().clear();

    StringRef FnName = F->getName();
    F->setName("");
    Function *StubFn = Function::Create(
        F->getFunctionType(), F->getLinkage(), FnName, M);
    F->replaceAllUsesWith(StubFn);
    FunctionsToBeDeleted.push_back(F);

    // Replace the body of the jit'ed function to call the jit entry, grab the
    // address of the specialized jit version and execute it.
    IRBuilder<> Builder(
        BasicBlock::Create(M.getContext(), "entry", StubFn, &StubFn->getEntryBlock()));
    // Create types for the runtime constant data structure and the jit entry
    // function.

    ArrayType *RuntimeConstantArrayTy =
        ArrayType::get(RuntimeConstantTy, ConstantArgsList[Idx].size());

    // Create globals for the function name and string IR passed to the jit
    // entry.
    auto *FnNameGlobal = Builder.CreateGlobalString(StubFn->getName());
    auto *StrIRGlobal = Builder.CreateGlobalString(StrIR);

    // Create the runtime constants data structure passed to the jit entry.
    auto *RuntimeConstantsAlloca = Builder.CreateAlloca(RuntimeConstantArrayTy);
    // Zero-initialize the alloca to avoid stack garbage for caching.
    Builder.CreateStore(Constant::getNullValue(RuntimeConstantArrayTy),
                        RuntimeConstantsAlloca);
    for (int ArgI = 0; ArgI < ConstantArgsList[Idx].size(); ++ArgI) {
      auto *GEP = Builder.CreateInBoundsGEP(
          RuntimeConstantArrayTy, RuntimeConstantsAlloca,
          {Builder.getInt32(0), Builder.getInt32(ArgI)});
      auto *GEPArgNo =
          Builder.CreateStructGEP(RuntimeConstantTy, GEP, 0);

      int ArgNo = ConstantArgsList[Idx][ArgI];
      Builder.CreateStore(Builder.getInt32(ArgNo), GEPArgNo);

      auto *GEPValue =
          Builder.CreateStructGEP(RuntimeConstantTy, GEP, 1);
      Builder.CreateStore(StubFn->getArg(ArgNo), GEPValue);
    }

    auto *JitFnPtr =
        Builder.CreateCall(JitEntryFnTy, JitEntryFn,
                           {FnNameGlobal, StrIRGlobal, RuntimeConstantsAlloca,
                            Builder.getInt32(ConstantArgsList[Idx].size())});
    SmallVector<Value *, 8> Args;
    for(auto &Arg : StubFn->args())
      Args.push_back(&Arg);
    auto *RetVal = Builder.CreateCall(StubFn->getFunctionType(), JitFnPtr, Args);
    if (StubFn->getReturnType()->isVoidTy())
      Builder.CreateRetVoid();
    else
      Builder.CreateRet(RetVal);

    dbgs() << "=== StubFn " << *StubFn << "=== End of StubFn\n";
    getchar();
    dbgs() << "=== Original Module\n" << M << "=== End of Original Module\n";

    ++Idx;
  }

  //for(Function *F : FunctionsToBeDeleted)
  //  F->eraseFromParent();

  if (verifyModule(M, &errs()))
    report_fatal_error("Broken module found, compilation aborted!", false);
  else
    dbgs() << "Module verified!\n";
}

// New PM implementation
struct HelloWorld : PassInfoMixin<HelloWorld> {
  // Main entry point, takes IR unit to run the pass on (&F) and the
  // corresponding pass manager (to be queried if need be)
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
    parseAnnotations(M);
    visitor(M);
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
struct LegacyHelloWorld : public ModulePass {
  static char ID;
  LegacyHelloWorld() : ModulePass(ID) {}
  // Main entry point - the name conveys what unit of IR this is to be run on.
  bool runOnModule(Module &M) override {
    parseAnnotations(M);
    visitor(M);

    // TODO: is anything preserved?
    return true;
    // Doesn't modify the input unit of IR, hence 'false'
    //return false;
  }
};
} // namespace

//-----------------------------------------------------------------------------
// New PM Registration
//-----------------------------------------------------------------------------
llvm::PassPluginLibraryInfo getHelloWorldPluginInfo() {
  const auto callback = [](PassBuilder &PB) {
    // TODO: decide where to insert it in the pipeline. Early avoids
    // inlining jit function (which disables jit'ing) but may require more
    // optimization, hence overhead, at runtime.
    //PB.registerPipelineStartEPCallback([&](ModulePassManager &MPM, auto) {
    PB.registerPipelineEarlySimplificationEPCallback( [&](ModulePassManager &MPM, auto) {
    // XXX: LastEP can break jit'ing, functions is inlined!
    //PB.registerOptimizerLastEPCallback([&](ModulePassManager &MPM, auto) {
      MPM.addPass(HelloWorld());
      return true;
    });
  };

  return {LLVM_PLUGIN_API_VERSION, "HelloWorld", LLVM_VERSION_STRING, callback};
}

// This is the core interface for pass plugins. It guarantees that 'opt' will
// be able to recognize HelloWorld when added to the pass pipeline on the
// command line, i.e. via '-passes=hello-world'
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getHelloWorldPluginInfo();
}

//-----------------------------------------------------------------------------
// Legacy PM Registration
//-----------------------------------------------------------------------------
// The address of this variable is used to uniquely identify the pass. The
// actual value doesn't matter.
char LegacyHelloWorld::ID = 0;

// This is the core interface for pass plugins. It guarantees that 'opt' will
// recognize LegacyHelloWorld when added to the pass pipeline on the command
// line, i.e.  via '--legacy-hello-world'
static RegisterPass<LegacyHelloWorld>
    X("legacy-hello-world", "Hello World Pass",
      true, // This pass doesn't modify the CFG => true
      false // This pass is not a pure analysis pass => false
    );
