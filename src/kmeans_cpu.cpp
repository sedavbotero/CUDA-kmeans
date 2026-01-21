#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" {
#include "kmeans.h"
}

/**
 * Calculates L2 distance squared between two points (Host/CPU version).
 */
template <int number_of_dimensions>
inline double calculate_vector_distance(const double *x, const double *y) {
  double result = 0;
  for (int i = 0; i < number_of_dimensions; i++) {
    result += (x[i] - y[i]) * (x[i] - y[i]);
  }
  return result;
}

#define INFINITY (1.0 / 0.0)
#define MAX_CLUSTERS 20

/**
 * Sequential K-means iteration loop; handles centroid updates and convergence
 * checks.
 */
template <int number_of_dimensions>
void fit_kmeans_cpu_template(double *data, int number_of_observations,
                             int number_of_clusters, double **centroids,
                             char **cluster_assignments) {
  double *old_centroids = (double *)malloc(
      sizeof(**centroids) * number_of_dimensions * number_of_clusters);
  double *new_centroids = (double *)malloc(
      sizeof(**centroids) * number_of_dimensions * number_of_clusters);
  *cluster_assignments =
      (char *)calloc(sizeof(**cluster_assignments), number_of_observations);
  memcpy(old_centroids, data,
         sizeof(*old_centroids) * number_of_clusters * number_of_dimensions);
  int delta = number_of_observations;
  int counts[MAX_CLUSTERS];
  for (int iter = 0;; iter++) {
    memset(counts, 0, sizeof(counts));
    for (int i = 0; i < number_of_dimensions * number_of_clusters; i++) {
      new_centroids[i] = 0.0;
    }
    delta = 0;
    for (int i = 0; i < number_of_observations; i++) {
      double minimal_distance = INFINITY;
      char closest_centroid = 0;
      for (char j = 0; j < number_of_clusters; j++) {
        double distance = calculate_vector_distance<number_of_dimensions>(
            &data[i * number_of_dimensions],
            &old_centroids[j * number_of_dimensions]);
        if (distance < minimal_distance) {
          minimal_distance = distance;
          closest_centroid = j;
        }
      }
      if (closest_centroid != (*cluster_assignments)[i]) {
        delta++;
      }
      (*cluster_assignments)[i] = closest_centroid;
      counts[(int)closest_centroid]++;
      for (int j = 0; j < number_of_dimensions; j++) {
        new_centroids[closest_centroid * number_of_dimensions + j] +=
            data[i * number_of_dimensions + j];
      }
    }
    printf("  iteration:%5d,  changes:%10d\n", iter, delta);
    if ((double)delta / number_of_observations <= 0.0) {
      break;
    }
    for (int i = 0; i < number_of_clusters; i++) {
      for (int j = 0; j < number_of_dimensions; j++) {
        new_centroids[i * number_of_dimensions + j] /= counts[i];
      }
    }
    double *tmp = old_centroids;
    old_centroids = new_centroids;
    new_centroids = tmp;
  }
  free(new_centroids);
  *centroids = old_centroids;
}

/**
 * Compile-time recursion to dispatch the CPU template for specific dimensions.
 */
template <int i = 1>
void fit_kmeans_cpu_dispatch(double *data, int number_of_dimensions,
                             int number_of_observations, int number_of_clusters,
                             double **centroids, char **cluster_assignments) {
  if (i == number_of_dimensions) {
    fit_kmeans_cpu_template<i>(data, number_of_observations, number_of_clusters,
                               centroids, cluster_assignments);
  } else if constexpr (i < 20) {
    fit_kmeans_cpu_dispatch<i + 1>(data, number_of_dimensions,
                                   number_of_observations, number_of_clusters,
                                   centroids, cluster_assignments);
  }
}

/**
 * Entry point for the CPU-based K-means algorithm.
 */
void fit_kmeans_cpu(double *data, int number_of_dimensions,
                    int number_of_observations, int number_of_clusters,
                    double **centroids, char **cluster_assignments) {
  fit_kmeans_cpu_dispatch(data, number_of_dimensions, number_of_observations,
                          number_of_clusters, centroids, cluster_assignments);
}
