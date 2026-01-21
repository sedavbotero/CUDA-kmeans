#include "kmeans.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define ERR(source)                                                            \
  (fprintf(stderr, "%s:%d\n", __FILE__, __LINE__), perror(source),             \
   exit(EXIT_FAILURE))

double *load_text_data(const char *filename, int *N, int *d, int *k) {
  FILE *fptr = fopen(filename, "r");
  if (!fptr) {
    ERR("fopen");
  }
  if (fscanf(fptr, "%d %d %d", N, d, k) != 3) {
    ERR("fscanf");
  }
  // printf("N = %d, d = %d, k = %d\n", *N, *d, *k);
  double *data = malloc(sizeof(*data) * *N * *d);
  for (int i = 0; i < *N * *d; i++) {
    if (fscanf(fptr, "%lf", data + i) != 1) {
      ERR("fscanf");
    }
  }
  fclose(fptr);
  return data;
}

double *load_binary_data(const char *filename, int *N, int *d, int *k) {
  FILE *fptr = fopen(filename, "rb");
  if (!fptr) {
    ERR("fopen");
  }
  if (fread(N, sizeof(*N), 1, fptr) != 1) {
    ERR("fread");
  }
  if (fread(d, sizeof(*d), 1, fptr) != 1) {
    ERR("fread");
  }
  if (fread(k, sizeof(*k), 1, fptr) != 1) {
    ERR("fread");
  }
  // printf("N = %d, d = %d, k = %d\n", *N, *d, *k);
  double *data = malloc(sizeof(*data) * *N * *d);
  if (fread(data, sizeof(*data), *N * *d, fptr) != *N * *d) {
    ERR("fread");
  }
  fclose(fptr);
  return data;
}

void output_kmeans_results(char *filename, double *centroids, int d, int k,
                           int N, char *cluster_assignments) {
  FILE *fptr = fopen(filename, "w");
  if (!fptr) {
    ERR("fopen");
  }
  for (int i = 0; i < k; i++) {
    for (int j = 0; j < d; j++) {
      fprintf(fptr, "%10.4lf", centroids[i * d + j]);
    }
    fprintf(fptr, "\n");
  }
  for (int i = 0; i < N; i++) {
    fprintf(fptr, "%3d\n", cluster_assignments[i]);
  }
  fflush(fptr);
  fclose(fptr);
}

enum data_format { BINARY, TEXT };
enum computation_method { GPU1, GPU2, CPU };
struct arguments {
  enum data_format data_format;
  enum computation_method computation_method;
  char *input_file;
  char *output_file;
};

void print_usage() {
  fprintf(stderr, "usage:  KMeans data_format computation_method input_file "
                  "output_file\n");
  exit(EXIT_FAILURE);
}

bool check_file_read_access(char *filename) {
  FILE *fptr = fopen(filename, "r");
  if (!fptr) {
    return false;
  }
  char buf;
  if (1 != fread(&buf, 1, 1, fptr)) {
    return false;
  }
  fclose(fptr);
  return true;
}

void parse_arguments(int argc, char *argv[],
                     struct arguments *parsed_arguments) {

  if (argc != 5) {
    print_usage();
  }
  if (!strcmp(argv[1], "txt")) {
    parsed_arguments->data_format = TEXT;
  } else if (!strcmp(argv[1], "bin")) {
    parsed_arguments->data_format = BINARY;
  } else {
    fprintf(stderr, "Unrecognised data format\ndata format may "
                    "be txt or bin\n");
    exit(EXIT_FAILURE);
  }
  if (!strcmp(argv[2], "gpu1")) {
    parsed_arguments->computation_method = GPU1;
  } else if (!strcmp(argv[2], "gpu2")) {
    parsed_arguments->computation_method = GPU2;
  } else if (!strcmp(argv[2], "cpu")) {
    parsed_arguments->computation_method = CPU;
  } else {
    fprintf(stderr, "Unrecognised computation method\ncomputation method may "
                    "be gpu1, gpu2 or cpu\n");
    exit(EXIT_FAILURE);
  }
  parsed_arguments->input_file = argv[3];
  if (!check_file_read_access(parsed_arguments->input_file)) {
    fprintf(stderr,
            "An error was encountered while trying to open file \"%s\":\n",
            parsed_arguments->input_file);
    perror("");
    exit(EXIT_FAILURE);
  }
  parsed_arguments->output_file = argv[4];
}

#define NANODIFFTIME(end, start)                                               \
  (difftime(end.tv_sec, start.tv_sec) +                                        \
   (double)(end.tv_nsec - start.tv_nsec) / 1000000000)

int main(int argc, char *argv[]) {
  struct arguments parsed_arguments;

  parse_arguments(argc, argv, &parsed_arguments);
  printf("\n number of thread: 12\n\n");
  printf("Reading data to CPU ...\n");
  printf("  data format:         %s\n",
         parsed_arguments.data_format == TEXT ? "text" : "binary");
  printf("  data type:           double\n");
  double *data;
  int N, d, k;
  struct timespec start, end, first_start;
  if (!timespec_get(&start, TIME_UTC)) {
    ERR("timespec_get");
  }
  if (parsed_arguments.data_format == TEXT) {
    data = load_text_data(parsed_arguments.input_file, &N, &d, &k);
  } else {
    data = load_binary_data(parsed_arguments.input_file, &N, &d, &k);
  }
  if (!timespec_get(&end, TIME_UTC)) {
    ERR("timespec_get");
  }
  printf("  number of points:    %8d\n", N);
  printf("  number of dimensions:%8d\n", d);
  printf("  number of clusters:  %8d\n", k);
  double time = NANODIFFTIME(end, start);
  printf("Data read, time: %.1lf seconds\n\n", time);
  double *centroids;
  char *cluster_assignments;
  first_start = start;
  switch (parsed_arguments.computation_method) {
  case GPU1:
    printf("Computing on GPU (custom kernels) ...\n");
    if (!timespec_get(&start, TIME_UTC)) {
      ERR("timespec_get");
    }
    fit_kmeans_custom(data, d, N, k, &centroids, &cluster_assignments);
    break;
  case GPU2:
    printf("Computing on GPU (thrust) ...\n");
    if (!timespec_get(&start, TIME_UTC)) {
      ERR("timespec_get");
    }
    fit_kmeans_thrust(data, d, N, k, &centroids, &cluster_assignments);
    break;
  case CPU:
    printf("Computing on CPU  ...\n");
    if (!timespec_get(&start, TIME_UTC)) {
      ERR("timespec_get");
    }
    fit_kmeans_cpu(data, d, N, k, &centroids, &cluster_assignments);
    break;
  default:
    fprintf(stderr, "Computation method not recognised\n");
    exit(EXIT_FAILURE);
    break;
  }
  if (!timespec_get(&end, TIME_UTC)) {
    ERR("timespec_get");
  }
  time = NANODIFFTIME(end, start);
  printf("Computation completed, time: %.1lf seconds\n\n", time);
  printf("Saving results to file ...\n");
  if (!timespec_get(&start, TIME_UTC)) {
    ERR("timespec_get");
  }
  output_kmeans_results(parsed_arguments.output_file, centroids, d, k, N,
                        cluster_assignments);
  if (!timespec_get(&end, TIME_UTC)) {
    ERR("timespec_get");
  }
  time = NANODIFFTIME(end, start);
  printf("Results saved, time: %.1lf seconds\n\n", time);
  time = NANODIFFTIME(end, first_start);
  printf("Total execution time: %.1lf seconds\n", time);

  return EXIT_SUCCESS;
}
