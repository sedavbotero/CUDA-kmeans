#include "kmeans.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

double calculate_vector_distance(int number_of_dimensions, const double *x,
                                 const double *y) {
  double result = 0;
  for (int i = 0; i < number_of_dimensions; i++) {
    result += (x[i] - y[i]) * (x[i] - y[i]);
  }
  return result;
}

#define INFINITY (1.0 / 0.0)
#define MAX_CLUSTERS 20
// #define MAX_DIMENSIONS 20
void print_vector_d(double *vec, int len) {
  printf("[");
  for (int i = 0; i < len; i++) {
    printf("%lf", vec[i]);
    if (i < len - 1) {
      printf(", ");
    }
  }
  printf("]\n");
}

void fit_kmeans_cpu(double *data, int number_of_dimensions,
                    int number_of_observations, int number_of_clusters,
                    double **centroids, char **cluster_assignments) {
  double *old_centroids =
      malloc(sizeof(**centroids) * number_of_dimensions * number_of_clusters);
  double *new_centroids =
      malloc(sizeof(**centroids) * number_of_dimensions * number_of_clusters);
  *cluster_assignments =
      calloc(sizeof(**cluster_assignments), number_of_observations);
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
      // char closest_centroid = 0;
      // double distance_sqr;
      // double min_distance_sqr = INFINITY;
      // for (int j = 0; j < number_of_clusters; j++) {
      //   distance_sqr = calculate_vector_distance(
      //       number_of_dimensions, &data[i * number_of_dimensions],
      //       &old_centroids[j * number_of_dimensions]);
      //   if (j) {
      //     printf("dist = %lf\n", distance_sqr);
      //   }
      //   if (distance_sqr < min_distance_sqr) {
      //     min_distance_sqr = distance_sqr;
      //     closest_centroid = j;
      //   }
      // }
      double minimal_distance = INFINITY;
      char closest_centroid = 0;
      for (char j = 0; j < number_of_clusters; j++) {
        double distance = calculate_vector_distance(
            number_of_dimensions, &data[i * number_of_dimensions],
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
      counts[closest_centroid]++;
      for (int j = 0; j < number_of_dimensions; j++) {
        new_centroids[closest_centroid * number_of_dimensions + j] +=
            data[i * number_of_dimensions + j];
      }
    }
    // printf("delta = %d\n", delta);
    printf("  iteration:%5d,  changes:%10d\n", iter, delta);
    if ((double)delta / number_of_observations <= 0.0) {
      break;
    }
    for (int i = 0; i < number_of_clusters; i++) {
      for (int j = 0; j < number_of_dimensions; j++) {
        new_centroids[i * number_of_dimensions + j] /= counts[i];
      }
      // printf("counts[%d] = %d\n", i, counts[i]);
    }
    // print_vector_d(old_centroids, number_of_dimensions);
    double *tmp = old_centroids;
    old_centroids = new_centroids;
    new_centroids = tmp;
  }
  free(new_centroids);
  *centroids = old_centroids;
}
