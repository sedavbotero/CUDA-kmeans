#include "cuda_utils.cuh"
#include "utils.h"
#include <cuda_runtime_api.h>
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

template <typename T> __host__ __device__ void print_vector(T *vec, int len) {
  printf("[");
  for (int i = 0; i < len; i++) {
    if constexpr (std::is_same_v<T, double>) {
      printf("%lf", vec[i]);
    } else if constexpr (std::is_same_v<T, int>) {
      printf("%d", vec[i]);
    }
    // else {
    //   static_assert(0, "type not supported");
    // }
    if (i < len - 1) {
      printf(", ");
    }
  }
  printf("]\n");
}

template <typename T> __global__ void print_vector_kernel(T *vec, int len) {
  print_vector<T>(vec, len);
}

template <typename T> void print_dev_vector(T *vec, int len) {
  print_vector_kernel<T><<<1, 1>>>(vec, len);
  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}

// #define SHARED_DATA_SIZE(number_of_dimensions, number_of_clusters)             \
//   (sizeof(double) * number_of_dimensions * number_of_clusters +                \
//    sizeof(int) * number_of_clusters)

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
  // double *x = &data[global_thread_id * number_of_dimensions];
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
  // printf("minimal distance for %d: %lf, at %d\n", threadIdx.x,
  // minimal_distance,
  //        closest_centroid);
  if (closest_centroid != cluster_assignments[global_thread_id]) {
    atomicAdd(delta, 1);
  }
  cluster_assignments[global_thread_id] = closest_centroid;

  atomicAdd(&new_cluster_sizes[closest_centroid], 1);
  for (int i = 0; i < number_of_dimensions; i++) {
    atomicAdd(&new_centroids[closest_centroid * number_of_dimensions + i],
              x[i]);
  }

  // atomicAdd(&new_cluster_sizes_buffer[closest_centroid], 1);
  // for (int i = 0; i < number_of_dimensions; i++) {
  //   atomicAdd(
  //       &new_centroids_buffer[closest_centroid * number_of_dimensions + i],
  //       x[i]);
  // }
  // __syncthreads();
  // __threadfence();
  // if (threadIdx.x != 0)
  //   return;
  // // print_vector<int>(new_cluster_sizes_buffer, number_of_clusters);
  // for (int i = 0; i < number_of_clusters; i++) {
  //   for (int j = 0; j < number_of_dimensions; j++) {
  //     atomicAdd(&new_centroids[i * number_of_dimensions + j],
  //               new_centroids_buffer[i * number_of_dimensions + j]);
  //   }
  //   atomicAdd(&new_cluster_sizes[i], new_cluster_sizes_buffer[i]);
  // }
}

void kmeans_fit_iterate(double *data, int number_of_dimensions,
                        int number_of_observations, int number_of_clusters,
                        double *old_centroids, double *new_centroids,
                        int *new_cluster_sizes, char *cluster_assignments,
                        int *delta) {
  constexpr int BLOCK_SIZE = 1024;
  const int NUMBER_OF_BLOCKS = CEIL_DEV(number_of_observations, BLOCK_SIZE);
  switch (number_of_dimensions) {
  case 1:
    kmeans_fit_iterate_template<1>
        <<<NUMBER_OF_BLOCKS, BLOCK_SIZE,
           SHARED_DATA_SIZE(number_of_dimensions, number_of_clusters)>>>(
            data, number_of_observations, number_of_clusters, old_centroids,
            new_centroids, new_cluster_sizes, cluster_assignments, delta);
    break;
  case 2:
    kmeans_fit_iterate_template<2>
        <<<NUMBER_OF_BLOCKS, BLOCK_SIZE,
           SHARED_DATA_SIZE(number_of_dimensions, number_of_clusters)>>>(
            data, number_of_observations, number_of_clusters, old_centroids,
            new_centroids, new_cluster_sizes, cluster_assignments, delta);
    break;
  case 3:
    kmeans_fit_iterate_template<3>
        <<<NUMBER_OF_BLOCKS, BLOCK_SIZE,
           SHARED_DATA_SIZE(number_of_dimensions, number_of_clusters)>>>(
            data, number_of_observations, number_of_clusters, old_centroids,
            new_centroids, new_cluster_sizes, cluster_assignments, delta);
    break;
  case 4:
    kmeans_fit_iterate_template<4>
        <<<NUMBER_OF_BLOCKS, BLOCK_SIZE,
           SHARED_DATA_SIZE(number_of_dimensions, number_of_clusters)>>>(
            data, number_of_observations, number_of_clusters, old_centroids,
            new_centroids, new_cluster_sizes, cluster_assignments, delta);
    break;
  case 5:
    kmeans_fit_iterate_template<5>
        <<<NUMBER_OF_BLOCKS, BLOCK_SIZE,
           SHARED_DATA_SIZE(number_of_dimensions, number_of_clusters)>>>(
            data, number_of_observations, number_of_clusters, old_centroids,
            new_centroids, new_cluster_sizes, cluster_assignments, delta);
    break;
  case 6:
    kmeans_fit_iterate_template<6>
        <<<NUMBER_OF_BLOCKS, BLOCK_SIZE,
           SHARED_DATA_SIZE(number_of_dimensions, number_of_clusters)>>>(
            data, number_of_observations, number_of_clusters, old_centroids,
            new_centroids, new_cluster_sizes, cluster_assignments, delta);
    break;
  case 7:
    kmeans_fit_iterate_template<7>
        <<<NUMBER_OF_BLOCKS, BLOCK_SIZE,
           SHARED_DATA_SIZE(number_of_dimensions, number_of_clusters)>>>(
            data, number_of_observations, number_of_clusters, old_centroids,
            new_centroids, new_cluster_sizes, cluster_assignments, delta);
    break;
  case 8:
    kmeans_fit_iterate_template<8>
        <<<NUMBER_OF_BLOCKS, BLOCK_SIZE,
           SHARED_DATA_SIZE(number_of_dimensions, number_of_clusters)>>>(
            data, number_of_observations, number_of_clusters, old_centroids,
            new_centroids, new_cluster_sizes, cluster_assignments, delta);
    break;
  case 9:
    kmeans_fit_iterate_template<9>
        <<<NUMBER_OF_BLOCKS, BLOCK_SIZE,
           SHARED_DATA_SIZE(number_of_dimensions, number_of_clusters)>>>(
            data, number_of_observations, number_of_clusters, old_centroids,
            new_centroids, new_cluster_sizes, cluster_assignments, delta);
    break;
  case 10:
    kmeans_fit_iterate_template<10>
        <<<NUMBER_OF_BLOCKS, BLOCK_SIZE,
           SHARED_DATA_SIZE(number_of_dimensions, number_of_clusters)>>>(
            data, number_of_observations, number_of_clusters, old_centroids,
            new_centroids, new_cluster_sizes, cluster_assignments, delta);
    break;
  case 11:
    kmeans_fit_iterate_template<11>
        <<<NUMBER_OF_BLOCKS, BLOCK_SIZE,
           SHARED_DATA_SIZE(number_of_dimensions, number_of_clusters)>>>(
            data, number_of_observations, number_of_clusters, old_centroids,
            new_centroids, new_cluster_sizes, cluster_assignments, delta);
    break;
  case 12:
    kmeans_fit_iterate_template<12>
        <<<NUMBER_OF_BLOCKS, BLOCK_SIZE,
           SHARED_DATA_SIZE(number_of_dimensions, number_of_clusters)>>>(
            data, number_of_observations, number_of_clusters, old_centroids,
            new_centroids, new_cluster_sizes, cluster_assignments, delta);
    break;
  case 13:
    kmeans_fit_iterate_template<13>
        <<<NUMBER_OF_BLOCKS, BLOCK_SIZE,
           SHARED_DATA_SIZE(number_of_dimensions, number_of_clusters)>>>(
            data, number_of_observations, number_of_clusters, old_centroids,
            new_centroids, new_cluster_sizes, cluster_assignments, delta);
    break;
  case 14:
    kmeans_fit_iterate_template<14>
        <<<NUMBER_OF_BLOCKS, BLOCK_SIZE,
           SHARED_DATA_SIZE(number_of_dimensions, number_of_clusters)>>>(
            data, number_of_observations, number_of_clusters, old_centroids,
            new_centroids, new_cluster_sizes, cluster_assignments, delta);
    break;
  case 15:
    kmeans_fit_iterate_template<15>
        <<<NUMBER_OF_BLOCKS, BLOCK_SIZE,
           SHARED_DATA_SIZE(number_of_dimensions, number_of_clusters)>>>(
            data, number_of_observations, number_of_clusters, old_centroids,
            new_centroids, new_cluster_sizes, cluster_assignments, delta);
    break;
  case 16:
    kmeans_fit_iterate_template<16>
        <<<NUMBER_OF_BLOCKS, BLOCK_SIZE,
           SHARED_DATA_SIZE(number_of_dimensions, number_of_clusters)>>>(
            data, number_of_observations, number_of_clusters, old_centroids,
            new_centroids, new_cluster_sizes, cluster_assignments, delta);
    break;
  case 17:
    kmeans_fit_iterate_template<17>
        <<<NUMBER_OF_BLOCKS, BLOCK_SIZE,
           SHARED_DATA_SIZE(number_of_dimensions, number_of_clusters)>>>(
            data, number_of_observations, number_of_clusters, old_centroids,
            new_centroids, new_cluster_sizes, cluster_assignments, delta);
    break;
  case 18:
    kmeans_fit_iterate_template<18>
        <<<NUMBER_OF_BLOCKS, BLOCK_SIZE,
           SHARED_DATA_SIZE(number_of_dimensions, number_of_clusters)>>>(
            data, number_of_observations, number_of_clusters, old_centroids,
            new_centroids, new_cluster_sizes, cluster_assignments, delta);
    break;
  case 19:
    kmeans_fit_iterate_template<19>
        <<<NUMBER_OF_BLOCKS, BLOCK_SIZE,
           SHARED_DATA_SIZE(number_of_dimensions, number_of_clusters)>>>(
            data, number_of_observations, number_of_clusters, old_centroids,
            new_centroids, new_cluster_sizes, cluster_assignments, delta);
    break;
  case 20:
    kmeans_fit_iterate_template<20>
        <<<NUMBER_OF_BLOCKS, BLOCK_SIZE,
           SHARED_DATA_SIZE(number_of_dimensions, number_of_clusters)>>>(
            data, number_of_observations, number_of_clusters, old_centroids,
            new_centroids, new_cluster_sizes, cluster_assignments, delta);
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
  cudaEvent_t start, mid, end;
  float time, total_1 = 0.0, total_2 = 0.0;
  CUDA_ERROR_CHECK(cudaEventCreate(&start));
  CUDA_ERROR_CHECK(cudaEventCreate(&mid));
  CUDA_ERROR_CHECK(cudaEventCreate(&end));
  for (; (double)delta / number_of_observations > 0.00; iter++) {
    CUDA_ERROR_CHECK(cudaMemset(dev_delta, 0, sizeof(delta)));
    CUDA_ERROR_CHECK(cudaEventRecord(start));
    kmeans_fit_iterate(dev_data, number_of_dimensions, number_of_observations,
                       number_of_clusters, dev_centroids, dev_next_centroids,
                       dev_next_cluster_sizes, dev_cluster_assignments,
                       dev_delta);
    // CUDA_ERROR_CHECK(cudaDeviceSynchronize());
#ifdef DEBUG
    printf("old centroids:\n");
    for (int i = 0; i < number_of_clusters; i++) {
      print_dev_vector<double>(&dev_centroids[i * number_of_dimensions],
                               number_of_dimensions);
    }
    printf("new centroids:\n");
    for (int i = 0; i < number_of_clusters; i++) {
      print_dev_vector<double>(&dev_next_centroids[i * number_of_dimensions],
                               number_of_dimensions);
    }
    printf("sizes: ");
    print_dev_vector<int>(dev_next_cluster_sizes, number_of_clusters);
#endif /* ifdef DEBUG */
    CUDA_ERROR_CHECK(cudaEventRecord(mid));
    scale_averages<<<1, number_of_dimensions * number_of_clusters>>>(
        dev_centroids, dev_next_centroids, dev_next_cluster_sizes,
        number_of_dimensions);
    CUDA_ERROR_CHECK(cudaEventRecord(end));
    CUDA_ERROR_CHECK(cudaEventSynchronize(end));
    CUDA_ERROR_CHECK(cudaEventElapsedTime(&time, start, mid));
    total_1 += time;
    CUDA_ERROR_CHECK(cudaEventElapsedTime(&time, mid, end));
    total_2 += time;
#ifdef DEBUG
    printf("new centroids scaled:\n");
    for (int i = 0; i < number_of_clusters; i++) {
      print_dev_vector<double>(&dev_centroids[i * number_of_dimensions],
                               number_of_dimensions);
    }
    printf("cleaned centroids:\n");
    for (int i = 0; i < number_of_clusters; i++) {
      print_dev_vector<double>(&dev_next_centroids[i * number_of_dimensions],
                               number_of_dimensions);
    }
    printf("delta: %d\n", delta);
#endif /* ifdef DEBUG */
    CUDA_ERROR_CHECK(
        cudaMemcpy(&delta, dev_delta, sizeof(delta), cudaMemcpyDeviceToHost));
    printf("  iteration:%5d,  changes:%10d\n", iter, delta);
  }
  printf("time iter %fms\n", total_1 / iter);
  printf("time scale %fms\n", total_2 / iter);

#ifdef DEBUG

  printf("iter = %d\n", iter);
#endif /* ifdef DEBUG */
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
  // for (int i = 3; i < 6; i++) {
  //   print_dev_vector(&data[i * number_of_dimensions], number_of_dimensions);
  // }

  CUDA_ERROR_CHECK(cudaFree(dev_next_centroids));
  CUDA_ERROR_CHECK(cudaFree(dev_next_cluster_sizes));
  CUDA_ERROR_CHECK(cudaFree(dev_centroids));
  CUDA_ERROR_CHECK(cudaFree(dev_data));
}
