#include "cuda_utils.cuh"
#include "utils.h"
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <device_atomic_functions.h>
extern "C" {
#include "kmeans.h"
}

__device__ inline double calculate_vector_distance(int number_of_dimensions,
                                                   const double *x,
                                                   const double *y) {
  double result = 0;
  for (int i = 0; i < number_of_dimensions; i++) {
    result += (x[i] - y[i]) * (x[i] - y[i]);
  }
  return result;
}

__host__ __device__ void print_vector(double *vec, int len) {
  printf("[");
  for (int i = 0; i < len; i++) {
    printf("%lf", vec[i]);
    if (i < len - 1) {
      printf(", ");
    }
  }
  printf("]\n");
}

__global__ void print_vector_kernel(double *vec, int len) {
  print_vector(vec, len);
}

void print_dev_vector(double *vec, int len) {
  print_vector_kernel<<<1, 1>>>(vec, len);
  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}

#define SHARED_DATA_SIZE(number_of_dimensions, number_of_clusters)             \
  (sizeof(double) * number_of_dimensions * number_of_clusters +                \
   sizeof(int) * number_of_clusters)

template <int number_of_dimensions>
__global__ void
kmeans_fit_iterate_template(double *data, int number_of_observations,
                            int number_of_clusters, const double *old_centroids,
                            double *new_centroids, int *new_cluster_sizes) {
  int global_thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  extern __shared__ double shared[];
  double *new_centroids_buffer = shared;
  int *new_cluster_sizes_buffer =
      (int *)&(shared[number_of_dimensions * number_of_clusters]);
  if (threadIdx.x < number_of_dimensions * number_of_clusters) {
    new_centroids_buffer[threadIdx.x] = 0.0;
  } else if (threadIdx.x <
             number_of_dimensions * number_of_clusters + number_of_dimensions) {
    new_cluster_sizes_buffer[threadIdx.x -
                             number_of_dimensions * number_of_clusters] = 0;
  }
  if (global_thread_id >= number_of_observations) {
    return;
  }
  __syncthreads();
  // if (threadIdx.x == 0) {
  //   for (int i = 0; i < number_of_clusters; i++) {
  //     printf("i = %d\ncentroid = ", i);
  //     print_vector(&new_centroids_buffer[i * number_of_dimensions],
  //                  number_of_dimensions);
  //     printf("size = %d\n", new_cluster_sizes_buffer[i]);
  //   }
  // }

  double x[number_of_dimensions];
  memcpy(x, &data[global_thread_id * number_of_dimensions],
         number_of_dimensions * sizeof(*x));
  // print_vector(x, number_of_dimensions);

  double minimal_distance = INFINITY;
  char closest_centroid;
  for (char i = 0; i < number_of_clusters; i++) {
    double distance = calculate_vector_distance(
        number_of_dimensions, x, &old_centroids[i * number_of_dimensions]);
    if (distance < minimal_distance) {
      minimal_distance = distance;
      closest_centroid = i;
    }
  }
  printf("minimal distance for %d: %lf, at %d\n", threadIdx.x, minimal_distance,
         closest_centroid);
  atomicAdd(&new_cluster_sizes_buffer[closest_centroid], 1);
  for (int i = 0; i < number_of_dimensions; i++) {
    atomicAdd(
        &new_centroids_buffer[closest_centroid * number_of_dimensions + i],
        x[i]);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int i = 0; i < number_of_clusters; i++) {
      for (int j = 0; j < number_of_dimensions; j++) {
        atomicAdd(&new_centroids[i * number_of_dimensions + j],
                  new_centroids_buffer[i * number_of_dimensions + j]);
      }
      atomicAdd(&new_cluster_sizes[i], new_cluster_sizes_buffer[i]);
    }
  }
}

void kmeans_fit_iterate(double *data, int number_of_dimensions,
                        int number_of_observations, int number_of_clusters,
                        double *old_centroids, double *new_centroids,
                        int *new_cluster_sizes) {
  constexpr int BLOCK_SIZE = 1024;
  const int NUMBER_OF_BLOCKS = CEIL_DEV(number_of_observations, BLOCK_SIZE);
  switch (number_of_dimensions) {
  case 1:
    kmeans_fit_iterate_template<1>
        <<<NUMBER_OF_BLOCKS, BLOCK_SIZE,
           SHARED_DATA_SIZE(number_of_dimensions, number_of_clusters)>>>(
            data, number_of_observations, number_of_clusters, old_centroids,
            new_centroids, new_cluster_sizes);
    break;
  case 2:
    kmeans_fit_iterate_template<2>
        <<<NUMBER_OF_BLOCKS, BLOCK_SIZE,
           SHARED_DATA_SIZE(number_of_dimensions, number_of_clusters)>>>(
            data, number_of_observations, number_of_clusters, old_centroids,
            new_centroids, new_cluster_sizes);
    break;
  case 3:
    kmeans_fit_iterate_template<3>
        <<<NUMBER_OF_BLOCKS, BLOCK_SIZE,
           SHARED_DATA_SIZE(number_of_dimensions, number_of_clusters)>>>(
            data, number_of_observations, number_of_clusters, old_centroids,
            new_centroids, new_cluster_sizes);
    break;
  case 4:
    kmeans_fit_iterate_template<4>
        <<<NUMBER_OF_BLOCKS, BLOCK_SIZE,
           SHARED_DATA_SIZE(number_of_dimensions, number_of_clusters)>>>(
            data, number_of_observations, number_of_clusters, old_centroids,
            new_centroids, new_cluster_sizes);
    break;
  case 5:
    kmeans_fit_iterate_template<5>
        <<<NUMBER_OF_BLOCKS, BLOCK_SIZE,
           SHARED_DATA_SIZE(number_of_dimensions, number_of_clusters)>>>(
            data, number_of_observations, number_of_clusters, old_centroids,
            new_centroids, new_cluster_sizes);
    break;
  case 6:
    kmeans_fit_iterate_template<6>
        <<<NUMBER_OF_BLOCKS, BLOCK_SIZE,
           SHARED_DATA_SIZE(number_of_dimensions, number_of_clusters)>>>(
            data, number_of_observations, number_of_clusters, old_centroids,
            new_centroids, new_cluster_sizes);
    break;
  case 7:
    kmeans_fit_iterate_template<7>
        <<<NUMBER_OF_BLOCKS, BLOCK_SIZE,
           SHARED_DATA_SIZE(number_of_dimensions, number_of_clusters)>>>(
            data, number_of_observations, number_of_clusters, old_centroids,
            new_centroids, new_cluster_sizes);
    break;
  case 8:
    kmeans_fit_iterate_template<8>
        <<<NUMBER_OF_BLOCKS, BLOCK_SIZE,
           SHARED_DATA_SIZE(number_of_dimensions, number_of_clusters)>>>(
            data, number_of_observations, number_of_clusters, old_centroids,
            new_centroids, new_cluster_sizes);
    break;
  case 9:
    kmeans_fit_iterate_template<9>
        <<<NUMBER_OF_BLOCKS, BLOCK_SIZE,
           SHARED_DATA_SIZE(number_of_dimensions, number_of_clusters)>>>(
            data, number_of_observations, number_of_clusters, old_centroids,
            new_centroids, new_cluster_sizes);
    break;
  case 10:
    kmeans_fit_iterate_template<10>
        <<<NUMBER_OF_BLOCKS, BLOCK_SIZE,
           SHARED_DATA_SIZE(number_of_dimensions, number_of_clusters)>>>(
            data, number_of_observations, number_of_clusters, old_centroids,
            new_centroids, new_cluster_sizes);
    break;
  case 11:
    kmeans_fit_iterate_template<11>
        <<<NUMBER_OF_BLOCKS, BLOCK_SIZE,
           SHARED_DATA_SIZE(number_of_dimensions, number_of_clusters)>>>(
            data, number_of_observations, number_of_clusters, old_centroids,
            new_centroids, new_cluster_sizes);
    break;
  case 12:
    kmeans_fit_iterate_template<12>
        <<<NUMBER_OF_BLOCKS, BLOCK_SIZE,
           SHARED_DATA_SIZE(number_of_dimensions, number_of_clusters)>>>(
            data, number_of_observations, number_of_clusters, old_centroids,
            new_centroids, new_cluster_sizes);
    break;
  case 13:
    kmeans_fit_iterate_template<13>
        <<<NUMBER_OF_BLOCKS, BLOCK_SIZE,
           SHARED_DATA_SIZE(number_of_dimensions, number_of_clusters)>>>(
            data, number_of_observations, number_of_clusters, old_centroids,
            new_centroids, new_cluster_sizes);
    break;
  case 14:
    kmeans_fit_iterate_template<14>
        <<<NUMBER_OF_BLOCKS, BLOCK_SIZE,
           SHARED_DATA_SIZE(number_of_dimensions, number_of_clusters)>>>(
            data, number_of_observations, number_of_clusters, old_centroids,
            new_centroids, new_cluster_sizes);
    break;
  case 15:
    kmeans_fit_iterate_template<15>
        <<<NUMBER_OF_BLOCKS, BLOCK_SIZE,
           SHARED_DATA_SIZE(number_of_dimensions, number_of_clusters)>>>(
            data, number_of_observations, number_of_clusters, old_centroids,
            new_centroids, new_cluster_sizes);
    break;
  case 16:
    kmeans_fit_iterate_template<16>
        <<<NUMBER_OF_BLOCKS, BLOCK_SIZE,
           SHARED_DATA_SIZE(number_of_dimensions, number_of_clusters)>>>(
            data, number_of_observations, number_of_clusters, old_centroids,
            new_centroids, new_cluster_sizes);
    break;
  case 17:
    kmeans_fit_iterate_template<17>
        <<<NUMBER_OF_BLOCKS, BLOCK_SIZE,
           SHARED_DATA_SIZE(number_of_dimensions, number_of_clusters)>>>(
            data, number_of_observations, number_of_clusters, old_centroids,
            new_centroids, new_cluster_sizes);
    break;
  case 18:
    kmeans_fit_iterate_template<18>
        <<<NUMBER_OF_BLOCKS, BLOCK_SIZE,
           SHARED_DATA_SIZE(number_of_dimensions, number_of_clusters)>>>(
            data, number_of_observations, number_of_clusters, old_centroids,
            new_centroids, new_cluster_sizes);
    break;
  case 19:
    kmeans_fit_iterate_template<19>
        <<<NUMBER_OF_BLOCKS, BLOCK_SIZE,
           SHARED_DATA_SIZE(number_of_dimensions, number_of_clusters)>>>(
            data, number_of_observations, number_of_clusters, old_centroids,
            new_centroids, new_cluster_sizes);
    break;
  case 20:
    kmeans_fit_iterate_template<20>
        <<<NUMBER_OF_BLOCKS, BLOCK_SIZE,
           SHARED_DATA_SIZE(number_of_dimensions, number_of_clusters)>>>(
            data, number_of_observations, number_of_clusters, old_centroids,
            new_centroids, new_cluster_sizes);
    break;
  }
}
__global__ void scale_averages(double *old_centroids, double *new_centroids,
                               int *cluster_sizes, int number_of_dimensions) {
  old_centroids[threadIdx.x] =
      new_centroids[threadIdx.x] /
      (double)cluster_sizes[(int)(threadIdx.x / number_of_dimensions)];
  new_centroids[threadIdx.x] = 0.0;
  __syncthreads();
  cluster_sizes[threadIdx.x / number_of_dimensions] = 0;
}

void fit_kmeans(double *data, int number_of_dimensions,
                int number_of_observations, int number_of_clusters) {
  double *dev_data;
  CUDA_ERROR_CHECK(
      cudaMalloc(&dev_data, sizeof(*data) * number_of_observations *
                                number_of_dimensions));
  CUDA_ERROR_CHECK(cudaMemcpy(dev_data, data,
                              sizeof(*dev_data) * number_of_dimensions *
                                  number_of_observations,
                              cudaMemcpyHostToDevice));
  double *dev_centroids;
  double *dev_next_centroids;
  int *dev_next_cluster_sizes;
  CUDA_ERROR_CHECK(
      cudaMalloc(&dev_centroids, sizeof(*dev_centroids) * number_of_dimensions *
                                     number_of_clusters));
  CUDA_ERROR_CHECK(cudaMemcpy(dev_centroids, dev_data,
                              sizeof(*dev_centroids) * number_of_dimensions *
                                  number_of_clusters,
                              cudaMemcpyDeviceToDevice));
  CUDA_ERROR_CHECK(cudaMalloc(&dev_next_centroids, sizeof(*dev_next_centroids) *
                                                       number_of_dimensions *
                                                       number_of_clusters));
  // CUDA_ERROR_CHECK(cudaMemset(dev_next_centroids, 0,
  //                             sizeof(*dev_next_centroids) *
  //                                 number_of_dimensions *
  //                                 number_of_clusters));
  double zeros[number_of_dimensions * number_of_clusters];
  for (int i = 0; i < number_of_dimensions * number_of_clusters; i++) {
    zeros[i] = 0.0;
  }
  CUDA_ERROR_CHECK(cudaMemcpy(dev_next_centroids, zeros, sizeof(zeros),
                              cudaMemcpyHostToDevice));
  CUDA_ERROR_CHECK(
      cudaMalloc(&dev_next_cluster_sizes,
                 sizeof(*dev_next_cluster_sizes) * number_of_clusters));
  CUDA_ERROR_CHECK(
      cudaMemset(dev_next_cluster_sizes, 0,
                 sizeof(*dev_next_cluster_sizes) * number_of_clusters));
  kmeans_fit_iterate(dev_data, number_of_dimensions, number_of_observations,
                     number_of_clusters, dev_centroids, dev_next_centroids,
                     dev_next_cluster_sizes);
  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
  printf("old centroids:\n");
  for (int i = 0; i < number_of_clusters; i++) {
    print_dev_vector(&dev_centroids[i * number_of_dimensions],
                     number_of_dimensions);
  }
  printf("new centroids:\n");
  for (int i = 0; i < number_of_clusters; i++) {
    print_dev_vector(&dev_next_centroids[i * number_of_dimensions],
                     number_of_dimensions);
  }
  scale_averages<<<1, number_of_dimensions * number_of_clusters>>>(
      dev_centroids, dev_next_centroids, dev_next_cluster_sizes,
      number_of_dimensions);
  printf("new centroids scaled:\n");
  for (int i = 0; i < number_of_clusters; i++) {
    print_dev_vector(&dev_centroids[i * number_of_dimensions],
                     number_of_dimensions);
  }
  printf("cleaned centroids:\n");
  for (int i = 0; i < number_of_clusters; i++) {
    print_dev_vector(&dev_next_centroids[i * number_of_dimensions],
                     number_of_dimensions);
  }
  // for (int i = 3; i < 6; i++) {
  //   print_dev_vector(&data[i * number_of_dimensions], number_of_dimensions);
  // }

  CUDA_ERROR_CHECK(cudaFree(dev_next_centroids));
  CUDA_ERROR_CHECK(cudaFree(dev_next_cluster_sizes));
  CUDA_ERROR_CHECK(cudaFree(dev_centroids));
  CUDA_ERROR_CHECK(cudaFree(dev_data));
}
