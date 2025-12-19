#include "kmeans.h"
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char *argv[]) {
  // double data[120];
  // for (int i = 0; i < 120; i++) {
  //   data[i] = i % 6 + (double)(((i / 2) % 2) * 2 - 1) * 100.0 +
  //             (double)(((i / 6) % 4) == 0) * ((i % 2) * 2 - 1) * 100.0;
  //   // data[i] = i % 6 + 100.0;
  // }
  // fit_kmeans(data, 2, 60, 3);
  int N, d, k;
  FILE *fptr = fopen("tests/KMeans_files/points_5mln_4d_5c.txt", "r");
  if (!fptr) {
    perror("fopen");
  }
  fscanf(fptr, "%d %d %d", &N, &d, &k);
  printf("N = %d, d = %d, k = %d\n", N, d, k);
  // N = 10;
  printf("N = %d, d = %d, k = %d\n", N, d, k);
  double *data = malloc(sizeof(*data) * N * d);
  for (int i = 0; i < N * d; i++) {
    fscanf(fptr, "%lf", data + i);
  }
  fclose(fptr);
  double *centroids;
  char *cluster_assignments;
  fit_kmeans(data, d, N, k, &centroids, &cluster_assignments);
  fptr = fopen("outfile", "w");
  for (int i = 0; i < k; i++) {
    for (int j = 0; j < d; j++) {
      fprintf(fptr, "%10.4lf", centroids[i * d + j]);
    }
    fprintf(fptr, "\r\n");
  }
  for (int i = 0; i < N; i++) {
    fprintf(fptr, "%3d\r\n", cluster_assignments[i]);
  }
  fflush(fptr);
  fclose(fptr);

  return EXIT_SUCCESS;
}
