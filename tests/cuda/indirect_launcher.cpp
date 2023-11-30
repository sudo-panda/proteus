// RUN: ./indirect_launcher.%ext | FileCheck %s
#include <climits>
#include <cstdio>
#include <cuda_runtime.h>

#define cudaErrCheck(CALL)                                                      \
  {                                                                            \
    cudaError_t err = CALL;                                                     \
    if (err != cudaSuccess) {                                                   \
      printf("ERROR @ %s:%d ->  %s\n", __FILE__, __LINE__,                     \
             cudaGetErrorString(err));                                          \
      abort();                                                                 \
    }                                                                          \
  }
__global__ __attribute__((annotate("jit"))) void kernel() {

  printf("Kernel\n");
}

template <typename T> cudaError_t launcher(T kernel_in) {
  return cudaLaunchKernel((const void *)kernel_in, 1, 1, 0, 0, 0);
}

int main() {
  kernel<<<1, 1>>>();
  cudaErrCheck(launcher(kernel));
  cudaErrCheck(cudaDeviceSynchronize());
  return 0;
}

// CHECK: Kernel
// CHECK: Kernel
// CHECK: JitCache hits 0 total 1
// CHECK: HashValue {{[0-9]+}} num_execs 1
