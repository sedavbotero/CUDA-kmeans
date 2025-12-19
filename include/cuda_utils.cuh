#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <stdio.h>

// Macro for checking and displaying cuda errors
#define CUDA_ERROR_CHECK(expr)                                                 \
  do {                                                                         \
    cudaError_t cudaStatus = expr;                                             \
    if (cudaStatus != cudaSuccess) {                                           \
      fprintf(stderr, "%s failed! At line %d, in %s\nError: %s\n\t %s\n",      \
              #expr, __LINE__, __FILE__, cudaGetErrorName(cudaStatus),         \
              cudaGetErrorString(cudaStatus));                                 \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#endif /* ifndef CUDA_UTILS_H */
