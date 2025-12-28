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

__device__ inline double calculate_vector_distance(int number_of_dimensions,
                                                   const double *x,
                                                   const double *y) {
  double result = 0;
  for (int i = 0; i < number_of_dimensions; i++) {
    result += (x[i] - y[i]) * (x[i] - y[i]);
  }
  return result;
}

struct assignClusterFunctor {
  const double *data;
  const int number_of_dimensions;
  const int number_of_clusters;
  double *centroids;
  assignClusterFunctor(const double *_data, const int _number_of_dimensions,
                       const int _number_of_clusters, double *_centroids)
      : data(_data), number_of_dimensions(_number_of_dimensions),
        number_of_clusters(_number_of_clusters), centroids(_centroids) {}
  __device__ char operator()(const int idx) {
    double minimal_distance = INFINITY;
    char closest_centroid;
    for (char i = 0; i < number_of_clusters; i++) {
      double distance = calculate_vector_distance(
          number_of_dimensions, &data[idx * number_of_dimensions],
          &centroids[i * number_of_dimensions]);
      if (distance < minimal_distance) {
        minimal_distance = distance;
        closest_centroid = i;
      }
    }
    return closest_centroid;
  }
};

struct dataPointReduceIntoCentroids {
  const double *data;
  const int number_of_dimensions;
  const int number_of_clusters;
  double *centroids;
  int *counts;

  dataPointReduceIntoCentroids(const double *_data,
                               const int _number_of_dimensions,
                               const int _number_of_clusters,
                               double *_centroids, int *_counts)
      : data(_data), number_of_dimensions(_number_of_dimensions),
        number_of_clusters(_number_of_clusters), centroids(_centroids),
        counts(_counts) {}

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

// template <int dim> struct Point {
//   double coordinates[dim];
//   __device__ __host__ double distance(const Point<dim> &l,
//                                       const Point<dim> &r) {
//     double dist = 0;
// #pragma unroll
//     for (int i = 0; i < dim; i++) {
//       dist += (l.coordinates[i] - r.coordinates[i]) *
//               (l.coordinates[i] - r.coordinates[i]);
//     }
//     return std::sqrt(dist);
//   }
//   __device__ __host__ Point<dim> operator+(const Point<dim> &other) {
//     Point<dim> out = new Point<dim>();
// #pragma unroll
//     for (int i = 0; i < dim; i++) {
//       out.coordinates[i] = this->coordinates[i] + other.coordinates[i];
//     }
//     return out;
//   }
// };

void fit_kmeans_thrust(double *data, int number_of_dimensions,
                       int number_of_observations, int number_of_clusters,
                       double **centroids, char **cluster_assignments) {
  thrust::device_vector<double> thrust_data(
      data, data + number_of_observations * number_of_dimensions);
  printf("N = %d, k = %d, d = %d\n", number_of_observations, number_of_clusters,
         number_of_dimensions);
  // printf("[");
  // for (auto el = thrust_data.begin(); el != thrust_data.end(); el++) {
  //   std::cout << el[0] << ", ";
  // }
  // printf("]\n");
  // double c[2 * 2] = {3, 3, 1, 1};
  // thrust::device_vector<double> thrust_centroids(c, c + 4);
  thrust::device_vector<double> thrust_centroids(
      data, data + number_of_dimensions * number_of_clusters);
  thrust::device_vector<char> assignments(number_of_observations);
  thrust::device_vector<char> old_assignments(number_of_observations);
  int delta = number_of_observations;
  while ((double)delta / number_of_observations > 0.00) {
    // for (int i = 0; i < 200; i++) {
    old_assignments = assignments;
    // thrust::copy(assignments.begin(), assignments.end(),
    //              old_assignments.begin());
    assignClusterFunctor assignCluster(
        thrust::raw_pointer_cast(thrust_data.data()), number_of_dimensions,
        number_of_clusters, thrust::raw_pointer_cast(thrust_centroids.data()));
    thrust::counting_iterator it(0);
    thrust::transform(it, it + number_of_observations, assignments.begin(),
                      assignCluster);

    // printf("[");
    // for (auto el = assignments.begin(); el != assignments.begin() + 5; el++)
    // {
    //   std::cout << (int)el[0] << ", ";
    // }
    // printf("]\n");
    // printf("[");
    // for (auto el = old_assignments.begin(); el != old_assignments.begin() +
    // 5;
    //      el++) {
    //   std::cout << (int)el[0] << ", ";
    // }
    // printf("]\n");
    delta = thrust::inner_product(
        assignments.begin(), assignments.end(), old_assignments.begin(), 0,
        thrust::plus<int>(), thrust::not_equal_to<int>());
    printf("delta = %d\n", delta);
    if ((double)delta / number_of_observations <= 0.00) {
      break;
    }

    thrust::device_vector<int> counts(number_of_clusters);
    thrust_centroids.assign(number_of_dimensions * number_of_clusters, 0.0);
    dataPointReduceIntoCentroids reductor(
        thrust::raw_pointer_cast(thrust_data.data()), number_of_dimensions,
        number_of_clusters, thrust::raw_pointer_cast(thrust_centroids.data()),
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
        [number_of_dimensions,
         counts_data] __device__(thrust::tuple<double &, int> it) {
          return (double)it.get<0>() /
                 ((double)counts_data[(
                     int)(((int)it.get<1>() / (int)number_of_dimensions))]);
        });
    // printf("[");
    // for (auto el = counts.begin(); el != counts.end(); el++) {
    //   std::cout << (int)el[0] << ", ";
    // }
    // printf("]\n");
  }

  // thrust::device_vector<int> indexes(number_of_dimensions *
  // number_of_clusters); thrust::transform(
  //     thrust::counting_iterator(0),
  //     thrust::counting_iterator(number_of_dimensions *
  //     number_of_clusters),
  //     thrust::constant_iterator<int>(number_of_dimensions),
  //     indexes.begin(), thrust::divides<int>());
  // thrust::device_vector<int>
  // devisors(number_of_dimensions*number_of_clusters);
  // thrust::scatter(counts.begin(), counts.end(), indexes.begin(), );

  *cluster_assignments =
      (char *)malloc(sizeof(**cluster_assignments) * number_of_observations);
  *centroids = (double *)malloc(sizeof(**centroids) * number_of_clusters *
                                number_of_dimensions);
  thrust::copy(assignments.begin(), assignments.end(), *cluster_assignments);
  thrust::copy(thrust_centroids.begin(), thrust_centroids.end(), *centroids);
}

// void fit_kmeans_thrust(double *data, int number_of_dimensions,
//                        int number_of_observations, int number_of_clusters,
//                        double **centroids, char **cluster_assignments) {
//   fit_kmeans_thrust_template(data, number_of_dimensions,
//   number_of_observations,
//                              number_of_clusters, centroids,
//                              cluster_assignments);
// }
