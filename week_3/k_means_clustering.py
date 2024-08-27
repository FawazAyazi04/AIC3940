import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataset = pd.read_csv("IRIS.csv")

def euclidean_distance(x1, x2):
    """This function calculates the Euclidean distance between two points"""
    dist = np.sqrt(np.sum((x1 - x2) ** 2))
    return dist

class KMeans:
    def __init__(self, k, max_iter=100, plot_steps=False):
        """this is just to store the values"""
        self.k = k
        self.max_iters = max_iter
        self.plot_steps = plot_steps #boolean
        # List of sample indices for each cluster
        self.clusters = [[] for _ in range(self.k)]

        self.centroids = []

    def predict(self, X):
        """predict method"""
        self.X = X
        self.n_samples, self.n_features = X.shape

        # Initialize our centroids
        random_sample_indexes = np.random.choice(self.n_samples, self.k, replace=False) # false => feature not choosen more than once
        self.centroids = [self.X[idx] for idx in random_sample_indexes]

        # Optimization
        for _ in range(self.max_iters):
            # Update clusters
            self.clusters = self._create_clusters(self.centroids)

            if self.plot_steps:
                self.plot()

            # Update centroids
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            if self.plot_steps:
                #for plotting when the step is true
                self.plot()

            # Check if converged
            if self._if_converged(centroids_old, self.centroids):
                break

        # Return cluster labels
        return self._get_cluster_labels(self.clusters)

    def _get_cluster_labels(self, clusters):
        """cluster label helper function"""
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels

    def _create_clusters(self, centroids):
        """create cluster helper function"""
        clusters = [[] for _ in range(self.k)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        """closest centroid helper function"""
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_idx = np.argmin(distances) #argmin return min index
        return closest_idx

    def _get_centroids(self, clusters):
        """centroid helper function"""
        centroids = np.zeros((self.k, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _if_converged(self, centroids_old, centroids):
        """check for convergence"""
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.k)]
        return sum(distances) == 0

    def _calculate_wcss(self):
        """Calculate the Within-Cluster Sum of Squares"""
        wcss = 0
        for i, cluster in enumerate(self.clusters):
            centroid = self.centroids[i]
            wcss += np.sum([euclidean_distance(self.X[idx], centroid) ** 2 for idx in cluster])
        return wcss

    def plot(self):
        """plot the cluster points"""
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self.clusters):
            points = self.X[index]
            ax.scatter(points[:, 0], points[:, 1], label=f"Cluster of petals {i}")
            #ax.scatter(points[:, 2], points[:, 3], label=f"Cluster of sepals {i}")

        centroids = np.array(self.centroids)
        ax.scatter(centroids[:, 0], centroids[:, 1], marker="x", color="black", linewidth=2, s=100, label="Centroids of sepals")
        #ax.scatter(centroids[:, 2], centroids[:, 3], marker="o", color="black", linewidth=2, s=100, label="Centroids of petals")

        ax.legend()
        plt.xlabel("length")
        plt.ylabel("width")
        plt.title(f"K-Means Clustering for k = {self.k}")
        plt.show()

def plot_elbow_method(X, k_range):
    """elbow plot"""
    wcss = []
    for k in k_range:
        kmeans = KMeans(k=k, max_iter=100)
        kmeans.predict(X)
        wcss.append(kmeans._calculate_wcss())

    plt.figure(figsize=(10, 6))
    plt.plot(k_range, wcss, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('WCSS')
    plt.xticks(k_range)
    plt.grid(True)
    plt.show()

    diff = np.diff(wcss)
    optimal_k = np.argmin(diff[1:] - diff[:-1]) + 2
    return optimal_k

numeric_columns = dataset.select_dtypes(include=[np.number])
#normalizing the dataset unig min max values
numeric_columns = (numeric_columns - numeric_columns.min()) / (numeric_columns.max() - numeric_columns.min())

k_range = range(1, 11)
k_optimal = plot_elbow_method(numeric_columns.values, k_range)

for i in k_range:
    kmeans = KMeans(k=i, max_iter=100, plot_steps=True)
    labels = kmeans.predict(numeric_columns.values)

kmeans.plot()
