#include <cstdio>
#include <cuda_runtime_api.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

extern "C" {
#include "kmeans.h"
}

template <int number_of_dimensions>
__device__ inline double calculate_vector_distance(const double *x,
                                                   const double *y) {
  double result = 0;
  for (int i = 0; i < number_of_dimensions; i++) {
    result += (x[i] - y[i]) * (x[i] - y[i]);
  }
  return result;
}

template <int number_of_dimensions> struct assignClusterFunctor {
  const double *data;
  const int number_of_clusters;
  double *centroids;
  assignClusterFunctor(const double *_data, const int _number_of_clusters,
                       double *_centroids)
      : data(_data), number_of_clusters(_number_of_clusters),
        centroids(_centroids) {}
  __device__ char operator()(const int idx) {
    double minimal_distance = INFINITY;
    char closest_centroid;
    for (char i = 0; i < number_of_clusters; i++) {
      double distance = calculate_vector_distance<number_of_dimensions>(
          &data[idx * number_of_dimensions],
          &centroids[i * number_of_dimensions]);
      if (distance < minimal_distance) {
        minimal_distance = distance;
        closest_centroid = i;
      }
    }
    return closest_centroid;
  }
};

template <int number_of_dimensions> struct dataPointReduceIntoCentroids {
  const double *data;
  const int number_of_clusters;
  double *centroids;
  int *counts;

  dataPointReduceIntoCentroids(const double *_data,
                               const int _number_of_clusters,
                               double *_centroids, int *_counts)
      : data(_data), number_of_clusters(_number_of_clusters),
        centroids(_centroids), counts(_counts) {}

  __device__ void operator()(thrust::tuple<int, char &> iterator) {
    const int idx = iterator.get<0>();
    const char label = iterator.get<1>();
    for (int i = 0; i < number_of_dimensions; i++) {
      atomicAdd(&centroids[label * number_of_dimensions + i],
                data[idx * number_of_dimensions + i]);
    }
    atomicAdd(&counts[label], 1);
  }
};

template <int number_of_dimensions>
void fit_kmeans_thrust_template(double *data, int number_of_observations,
                                int number_of_clusters, double **centroids,
                                char **cluster_assignments) {
  thrust::device_vector<double> thrust_data(
      data, data + number_of_observations * number_of_dimensions);
  thrust::device_vector<double> thrust_centroids(
      data, data + number_of_dimensions * number_of_clusters);
  thrust::device_vector<char> assignments(number_of_observations);
  thrust::device_vector<char> old_assignments(number_of_observations);
  int delta = number_of_observations;
  for (int iter = 0; (double)delta / number_of_observations > 0.00; iter++) {
    old_assignments = assignments;
    assignClusterFunctor<number_of_dimensions> assignCluster(
        thrust::raw_pointer_cast(thrust_data.data()), number_of_clusters,
        thrust::raw_pointer_cast(thrust_centroids.data()));
    thrust::counting_iterator it(0);
    thrust::transform(it, it + number_of_observations, assignments.begin(),
                      assignCluster);

    delta = thrust::inner_product(
        assignments.begin(), assignments.end(), old_assignments.begin(), 0,
        thrust::plus<int>(), thrust::not_equal_to<int>());
    printf("  iteration:%5d,  changes:%10d\n", iter, delta);
    if ((double)delta / number_of_observations <= 0.00) {
      break;
    }

    thrust::device_vector<int> counts(number_of_clusters);
    thrust_centroids.assign(number_of_dimensions * number_of_clusters, 0.0);
    dataPointReduceIntoCentroids<number_of_dimensions> reductor(
        thrust::raw_pointer_cast(thrust_data.data()), number_of_clusters,
        thrust::raw_pointer_cast(thrust_centroids.data()),
        thrust::raw_pointer_cast(counts.data()));
    auto zip_start =
        thrust::make_zip_iterator(thrust::make_tuple(it, assignments.begin()));
    auto zip_end = thrust::make_zip_iterator(
        thrust::make_tuple(it + number_of_observations, assignments.end()));
    thrust::for_each(zip_start, zip_end, reductor);

    int *counts_data = thrust::raw_pointer_cast(counts.data());
    thrust::transform(
        thrust::make_zip_iterator(thrust_centroids.begin(),
                                  thrust::counting_iterator(0)),
        thrust::make_zip_iterator(
            thrust_centroids.end(),
            thrust::counting_iterator(number_of_clusters *
                                      number_of_dimensions)),
        thrust_centroids.begin(),
        [counts_data] __device__(thrust::tuple<double &, int> it) {
          return (double)it.get<0>() /
                 ((double)counts_data[(
                     int)(((int)it.get<1>() / number_of_dimensions))]);
        });
  }

  *cluster_assignments =
      (char *)malloc(sizeof(**cluster_assignments) * number_of_observations);
  *centroids = (double *)malloc(sizeof(**centroids) * number_of_clusters *
                                number_of_dimensions);
  thrust::copy(assignments.begin(), assignments.end(), *cluster_assignments);
  thrust::copy(thrust_centroids.begin(), thrust_centroids.end(), *centroids);
}

template <int i = 1>
inline void
fit_kmeans_thrust_dispatch(double *data, int number_of_dimensions,
                           int number_of_observations, int number_of_clusters,
                           double **centroids, char **cluster_assignments) {

  if (i == number_of_dimensions) {
    fit_kmeans_thrust_template<i>(data, number_of_observations,
                                  number_of_clusters, centroids,
                                  cluster_assignments);
  } else if constexpr (i < 20) {
    fit_kmeans_thrust_dispatch<i + 1>(
        data, number_of_dimensions, number_of_observations, number_of_clusters,
        centroids, cluster_assignments);
  }
}

void fit_kmeans_thrust(double *data, int number_of_dimensions,
                       int number_of_observations, int number_of_clusters,
                       double **centroids, char **cluster_assignments) {
  fit_kmeans_thrust_dispatch(data, number_of_dimensions, number_of_observations,
                             number_of_clusters, centroids,
                             cluster_assignments);
}
