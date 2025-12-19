#ifndef KMEANS_H
#define KMEANS_H

void fit_kmeans(double *data, int number_of_dimensions,
                int number_of_observations, int number_of_clusters,
                double **centroids, char **cluster_assignments);

#endif // !KMEANS_H
