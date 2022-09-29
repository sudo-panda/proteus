all: jit test

test:
	#clang++ -O0 -g test.cpp -fpass-plugin=${PWD}/pass/build/libHelloWorld.dylib -o test.x
	#clang++ -O0 -g test.cpp -fpass-plugin=${PWD}/pass/build/libHelloWorld.dylib -emit-llvm -S -o test.ll
	clang++ -O3 -g test.cpp -fpass-plugin=${PWD}/pass/build/libHelloWorld.dylib -o test.x -L . -ljit
	clang++ -O3 -g test.cpp -o test.nojit.x

jit:
	#clang++ -O3 -g jit.cpp `llvm-config --cxxflags --ldflags --system-libs --libs core orcjit native` -shared -o libjit.dylib -flto
	clang++ -O3 -v -g jit.cpp `llvm-config --link-static --cxxflags --ldflags --libs core orcjit native` -flto -c --emit-static-lib
	#clang++ -O3 -g jit.cpp `llvm-config --cxxflags --ldflags --system-libs --libs core orcjit native` -emit-llvm -c

jit-example:
	#clang++ -g jit.cpp `llvm-config --cxxflags --ldflags --system-libs --libs core orcjit native` -O3 -o jit.x
	#clang++ -g jit.cpp `llvm-config --cxxflags --ldflags --system-libs --libs core orcjit native` -emit-llvm -S -o jit.ll
	clang++ -O0 -g jit.cpp `llvm-config --cxxflags --ldflags --system-libs --libs core orcjit native` -fpass-plugin=${PWD}/pass/build/libHelloWorld.dylib
