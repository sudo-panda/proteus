ml load cmake/3.23.1
ml load cuda/12.2.2

LLVM_INSTALL_DIR=$1

if [ -z "$LLVM_INSTALL_DIR" ]; then
    echo "Usage: setup-lassen.sh <LLVM installation dir>"
    return 0
fi

export PATH="$LLVM_INSTALL_DIR/bin":$PATH

mkdir build-lassen
pushd build-lassen

cmake .. \
-DLLVM_INSTALL_DIR="$LLVM_INSTALL_DIR" \
-DLLVM_VERSION=18 \
-DENABLE_CUDA=on \
-DCMAKE_CUDA_ARCHITECTURES=70 \
-DCMAKE_C_COMPILER="$LLVM_INSTALL_DIR/bin/clang" \
-DCMAKE_CXX_COMPILER="$LLVM_INSTALL_DIR/bin/clang++" \
-DCMAKE_CUDA_COMPILER="$LLVM_INSTALL_DIR/bin/clang++" \
-DCMAKE_EXPORT_COMPILE_COMMANDS=on \
-DCMAKE_INSTALL_PREFIX="install-lassen" \
-DProTorch_DIR=/usr/workspace/koparasy/Protorch/build-lassen 

popd

