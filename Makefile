UNAME := $(shell uname)

CXXFLAGS = -O3 -shared -gdwarf-4
#LLVM_CXXFLAGS = $(shell llvm-config --cxxflags --ldflags --system-libs --libs all)
LLVM_CXXFLAGS = $(shell llvm-config --cxxflags --ldflags --system-libs --libs core orcjit native)
LLVM_RPATH=$(shell llvm-config --libdir)
ifeq ($(UNAME), Linux)
	SUFFIX := so
	CXXFLAGS += -Wl,-rpath,$(LLVM_RPATH) -fPIC
endif
ifeq ($(UNAME), Darwin)
	SUFFIX := dylib
endif

libjit: libjit.$(SUFFIX)

test_split:
	clang++ -O3 -g test1.cpp test2.cpp -o test_split.nojit.x
	clang++ -O3 -g test1.cpp -fpass-plugin=${PWD}/pass/build/libHelloWorld.$(SUFFIX) -c -o test1.o
	clang++ -O3 -g test2.cpp -fpass-plugin=${PWD}/pass/build/libHelloWorld.$(SUFFIX) -c -o test2.o
	clang++ -O3 test1.o test2.o -o test_split.x -L . -ljit

test:
	clang++ -O3 -gdwarf-4 test.cpp -fpass-plugin=${PWD}/pass/build/libHelloWorld.$(SUFFIX) -o test.x -L . -ljit
	#clang++ -O0 -g test.cpp -fpass-plugin=${PWD}/pass/build/libHelloWorld.$(SUFFIX) -o test.x
	#clang++ -O0 -g test.cpp -fpass-plugin=${PWD}/pass/build/libHelloWorld.$(SUFFIX) -emit-llvm -S -o test.ll
	#clang++ -O0 -g test.cpp -fpass-plugin=${PWD}/pass/build/libHelloWorld.$(SUFFIX) -emit-llvm -S -o test.ll
	clang++ -O3 -fsave-optimization-record -fno-discard-value-names -g test.cpp -emit-llvm -S -o test.ll
	clang++ -O3 -g test.cpp -fpass-plugin=${PWD}/pass/build/libHelloWorld.$(SUFFIX) -c -o test.o
	clang++ -O3 -g test.cpp -o test.nojit.x

libjit.$(SUFFIX): jit.cpp
	clang++ $(CXXFLAGS) -fno-limit-debug-info jit.cpp $(LLVM_CXXFLAGS) -o libjit.$(SUFFIX)
	#clang++ -O3 -g jit.cpp `llvm-config --cxxflags --ldflags --system-libs --libs core orcjit native` -shared -o libjit.$(SUFFIX)

jit-example:
	#clang++ -g jit.cpp `llvm-config --cxxflags --ldflags --system-libs --libs core orcjit native` -O3 -o jit.x
	#clang++ -g jit.cpp `llvm-config --cxxflags --ldflags --system-libs --libs core orcjit native` -emit-llvm -S -o jit.ll
	clang++ -O0 -g jit.cpp `llvm-config --cxxflags --ldflags --system-libs --libs core orcjit native` -fpass-plugin=${PWD}/pass/build/libHelloWorld.$(SUFFIX)

clean:
	rm -rf libjit.$(SUFFIX) test.x