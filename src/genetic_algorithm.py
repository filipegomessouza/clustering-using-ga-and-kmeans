from typing import List
from sko.GA import RCGA
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class GeneticAlgorithm(RCGA):
    MUTATION_PROBABILITY = 0.01

    def __init__(self, dataset: np.ndarray, number_of_groups: int):
        self.dataset = dataset
        self.number_of_groups = number_of_groups
        n_features = dataset.shape[1]

        n_dimensions = number_of_groups * n_features

        super().__init__(
            func=self.fitness,
            n_dim=n_dimensions,
            size_pop=3 * n_dimensions,
            max_iter=10 * n_dimensions,
            prob_mut=self.MUTATION_PROBABILITY,
            lb=dataset.min(axis=0).tolist() * number_of_groups,
            ub=dataset.max(axis=0).tolist() * number_of_groups,
        )

    def fitness(self, individual: np.ndarray) -> float:
        centroids = individual.reshape(self.number_of_groups, self.dataset.shape[1])
        distances: np.ndarray = np.linalg.norm(self.dataset[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
        min_distances: np.ndarray = distances.min(axis=1)

        return min_distances.sum()

    def _apply_kmeans(self, idx: int) -> None:
        centroids = self.X[idx].reshape(self.number_of_groups, self.dataset.shape[1])

        kmeans = KMeans(
            n_clusters=self.number_of_groups,
            init=centroids,
            n_init=1,
        )

        kmeans.fit(self.dataset)

        refined = kmeans.cluster_centers_.flatten()
        self.Chrom[idx] = (refined - self.lb) / (self.ub - self.lb)
        self.X[idx] = refined
        self.Y[idx] = self.func(np.array([refined]))[0]

    def run(self, max_iter=None):
        self.max_iter = max_iter or self.max_iter
        kmeans_limit = self.max_iter // 10
        best = []

        for i in range(self.max_iter):
            self.X = self.chrom2x(self.Chrom)
            self.Y = self.x2y()

            if i < kmeans_limit:
                idx = int(self.Y.argmin()) if i == 0 else np.random.randint(0, self.size_pop)
                self._apply_kmeans(idx)

            self.ranking()
            self.selection()
            self.crossover()
            self.mutation()

            generation_best_index = self.FitV.argmax()
            self.generation_best_X.append(self.X[generation_best_index, :])
            self.generation_best_Y.append(self.Y[generation_best_index])
            self.all_history_Y.append(self.Y)
            self.all_history_FitV.append(self.FitV)

            if self.early_stop:
                best.append(min(self.generation_best_Y))
                if len(best) >= self.early_stop:
                    if best.count(min(best)) == len(best):
                        break
                    else:
                        best.pop(0)

        global_best_index = np.array(self.generation_best_Y).argmin()
        self.best_x = self.generation_best_X[global_best_index]
        self.best_y = self.func(np.array([self.best_x]))

        return self.best_x, self.best_y

    def get_clusters(self) -> np.ndarray:
        centroids = self.best_x.reshape(self.number_of_groups, self.dataset.shape[1])
        distances = np.linalg.norm(self.dataset[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
        return distances.argmin(axis=1)

    def plot(self, filename: str, class_names: List[str]) -> None:
        labels = self.get_clusters()
        dataset_3d, centroids_3d = self.get_3d_points_for_plot()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for i in range(self.number_of_groups):
            mask = labels == i
            ax.scatter(dataset_3d[mask, 0], dataset_3d[mask, 1], dataset_3d[mask, 2], s=20, label=class_names[i].capitalize())

        ax.scatter(centroids_3d[:, 0], centroids_3d[:, 1], centroids_3d[:, 2], s=20, c='black', label='Centroids')
        ax.legend()

        plt.tight_layout()
        plt.savefig(filename, format='png')

    def get_3d_points_for_plot(self) -> np.ndarray:
        n_features = self.dataset.shape[1]
        centroids = self.best_x.reshape(self.number_of_groups, n_features)

        if n_features > 3:
            pca = PCA(n_components=3)
            pca.fit(self.dataset)
            dataset_3d = pca.transform(self.dataset)
            centroids_3d = pca.transform(centroids)
        elif n_features == 3:
            dataset_3d = self.dataset
            centroids_3d = centroids
        elif n_features == 2:
            dataset_3d = np.hstack([self.dataset, np.zeros((len(self.dataset), 1))])
            centroids_3d = np.hstack([centroids, np.zeros((self.number_of_groups, 1))])
        elif n_features == 1:
            dataset_3d = np.hstack([self.dataset, np.zeros((len(self.dataset), 2))])
            centroids_3d = np.hstack([centroids, np.zeros((self.number_of_groups, 2))])
        else:
            raise ValueError("Dataset must have at least 1 feature")

        return dataset_3d, centroids_3d
