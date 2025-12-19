#include "kmeans.h"
#include <stdlib.h>
int main(int argc, char *argv[]) {
  double data[120];
  for (int i = 0; i < 120; i++) {
    data[i] = i % 6 + (double)(((i / 6) % 2) * 2 - 1) * 100.0;
  }
  fit_kmeans(data, 2, 60, 3);
  return EXIT_SUCCESS;
}
