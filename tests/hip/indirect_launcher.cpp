// RUN: ./indirect_launcher.%ext | FileCheck %s
#include <climits>
#include <cstdio>
#include <hip/hip_runtime.h>

#define hipErrCheck(CALL)                                                      \
  {                                                                            \
    hipError_t err = CALL;                                                     \
    if (err != hipSuccess) {                                                   \
      printf("ERROR @ %s:%d ->  %s\n", __FILE__, __LINE__,                     \
             hipGetErrorString(err));                                          \
      abort();                                                                 \
    }                                                                          \
  }
__global__ __attribute__((annotate("jit"))) void kernel() {

  printf("Kernel\n");
}

template <typename T> hipError_t launcher(T kernel_in) {
  return hipLaunchKernel((const void *)kernel_in, 1, 1, 0, 0, 0);
}

int main() {
  kernel<<<1, 1>>>();
  hipErrCheck(launcher(kernel));
  hipErrCheck(hipDeviceSynchronize());
  return 0;
}

// CHECK: Kernel
// CHECK: Kernel
// CHECK: JitCache hits 0 total 1
// CHECK: HashValue {{[0-9]+}} num_execs 1