#include "cuda_utils.cuh"
#include "utils.h"
#include <cuda_runtime_api.h>

extern "C" {
#include "kmeans.h"
}

template <unsigned int number_of_dimensions>
__device__ inline double calculate_vector_distance(const double *x,
                                                   const double *y) {
  double result = 0;
  for (int i = 0; i < number_of_dimensions; i++) {
    result += (x[i] - y[i]) * (x[i] - y[i]);
  }
  return result;
}

#define SHARED_DATA_SIZE(number_of_dimensions, number_of_clusters) 0

template <int number_of_dimensions>
__global__ void
kmeans_fit_iterate_template(double *data, int number_of_observations,
                            int number_of_clusters, const double *old_centroids,
                            double *new_centroids, int *new_cluster_sizes,
                            char *cluster_assignments, int *delta) {
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

  double x[number_of_dimensions];
  memcpy(x, &data[global_thread_id * number_of_dimensions],
         number_of_dimensions * sizeof(*x));

  double minimal_distance = INFINITY;
  char closest_centroid;
  for (char i = 0; i < number_of_clusters; i++) {
    double distance = calculate_vector_distance<number_of_dimensions>(
        x, &old_centroids[i * number_of_dimensions]);
    if (distance < minimal_distance) {
      minimal_distance = distance;
      closest_centroid = i;
    }
  }
  if (closest_centroid != cluster_assignments[global_thread_id]) {
    atomicAdd(delta, 1);
  }
  cluster_assignments[global_thread_id] = closest_centroid;

  atomicAdd(&new_cluster_sizes[closest_centroid], 1);
  for (int i = 0; i < number_of_dimensions; i++) {
    atomicAdd(&new_centroids[closest_centroid * number_of_dimensions + i],
              x[i]);
  }
}

template <unsigned char i = 1>
inline void kmeans_fit_iterate(double *data, const int number_of_dimensions,
                               int number_of_observations,
                               int number_of_clusters, double *old_centroids,
                               double *new_centroids, int *new_cluster_sizes,
                               char *cluster_assignments, int *delta) {

  constexpr int BLOCK_SIZE = 1024;
  const int NUMBER_OF_BLOCKS = CEIL_DEV(number_of_observations, BLOCK_SIZE);
  if (i == number_of_dimensions) {
    kmeans_fit_iterate_template<i><<<NUMBER_OF_BLOCKS, BLOCK_SIZE>>>(
        data, number_of_observations, number_of_clusters, old_centroids,
        new_centroids, new_cluster_sizes, cluster_assignments, delta);
  } else if constexpr (i < 20) {
    kmeans_fit_iterate<i + 1>(data, number_of_dimensions,
                              number_of_observations, number_of_clusters,
                              old_centroids, new_centroids, new_cluster_sizes,
                              cluster_assignments, delta);
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

void fit_kmeans_custom(double *data, int number_of_dimensions,
                       int number_of_observations, int number_of_clusters,
                       double **centroids, char **cluster_assignments) {
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
  char *dev_cluster_assignments;
  CUDA_ERROR_CHECK(
      cudaMalloc(&dev_cluster_assignments, number_of_observations));
  CUDA_ERROR_CHECK(
      cudaMemset(dev_cluster_assignments, 0, number_of_observations));
  int iter = 0;
  int delta = number_of_observations;
  int *dev_delta;
  CUDA_ERROR_CHECK(cudaMalloc(&dev_delta, sizeof(delta)));
  for (; (double)delta / number_of_observations > 0.00; iter++) {
    CUDA_ERROR_CHECK(cudaMemset(dev_delta, 0, sizeof(delta)));
    kmeans_fit_iterate(dev_data, number_of_dimensions, number_of_observations,
                       number_of_clusters, dev_centroids, dev_next_centroids,
                       dev_next_cluster_sizes, dev_cluster_assignments,
                       dev_delta);
    scale_averages<<<1, number_of_dimensions * number_of_clusters>>>(
        dev_centroids, dev_next_centroids, dev_next_cluster_sizes,
        number_of_dimensions);
    CUDA_ERROR_CHECK(
        cudaMemcpy(&delta, dev_delta, sizeof(delta), cudaMemcpyDeviceToHost));
    printf("  iteration:%5d,  changes:%10d\n", iter, delta);
  }

  *cluster_assignments =
      (char *)malloc(sizeof(**cluster_assignments) * number_of_observations);
  *centroids = (double *)malloc(sizeof(**centroids) * number_of_clusters *
                                number_of_dimensions);
  CUDA_ERROR_CHECK(
      cudaMemcpy(*cluster_assignments, dev_cluster_assignments,
                 sizeof(*dev_cluster_assignments) * number_of_observations,
                 cudaMemcpyDeviceToHost));
  CUDA_ERROR_CHECK(cudaMemcpy(*centroids, dev_centroids,
                              sizeof(*dev_centroids) * number_of_dimensions *
                                  number_of_clusters,
                              cudaMemcpyDeviceToHost));

  CUDA_ERROR_CHECK(cudaFree(dev_next_centroids));
  CUDA_ERROR_CHECK(cudaFree(dev_next_cluster_sizes));
  CUDA_ERROR_CHECK(cudaFree(dev_centroids));
  CUDA_ERROR_CHECK(cudaFree(dev_data));
}
