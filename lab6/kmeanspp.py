import numpy as np
from collections import defaultdict
from copy import deepcopy


class KMeansPP(object):
    def __init__(self, df, k):
        """
        Given a pandas dataframe and the of clusters,
        Initializes k-means to its initial K centroids
        via k-means++ init

        Params:
            df - pandas dataframe containing points to be clustered
            k - number of clusters to form
        """

        # Got some help/inspiration from this StackOverflow link:
        # http://stackoverflow.com/questions/5466323/how-exactly-does-k-means-work
        # Also read this link (a bit confusing, but helpful too):
        # https://datasciencelab.wordpress.com/2014/01/15/improved-seeding-for-clustering-with-k-means/

        self.k = np.min(k, df.shape[0])
        self.points = df.as_matrix()

        # If the number of points is less than the number
        # of clusters, then set k=number of points
        num_clusters = min(self.k, self.points.shape[0])
        # self.centers = np.zeros(num_clusters, self.points.shape[1])
        self.centers = []

        # Choose initial center uniformly at random from the points.
        c1_index = np.random.randint(low=0, high=self.points.shape[0])
        # self.centers[0] = self.points[c1_index]
        self.centers.append(self.points[c1_index])

        for i in range(1, num_clusters):
            # For each iteration, Compute the vector containing the square
            # distances between all points in the dataset

            dist_vec = np.array([np.amin((c - self.points) ** 2, axis=0)
                                 for c in self.centers])
            # choose each subsequent center from self.pixels,
            # randomly drawn from the normalized probability distribution
            # over dist_vec.
            probs = dist_vec / dist_vec.sum()
            cumprobs = probs.cumsum()
            r = np.random.rand()

            for j, p in enumerate(cumprobs):
                if r < p:
                    ci_index = j  # Index of every subsequent centroid
                    break
            self.centers.append(self.points[ci_index])

        self.clusters = defaultdict(list)

    def _find_center(self, p):
        """ Calculates the closest center for a point based
        on Euclidean distance.

        Args:
            p - a point within self.points

        Returns:
            Index of the center closest to current pixel
        """
        center_dists = np.linalg.norm(p - self.centers, axis=1)
        return np.argmin(center_dists)

    def _assign(self):
        """Assign each point to its nearest centroid."""
        clusters = defaultdict(list)
        for p in self.points:
            center_idx = self._find_center(p)
            clusters[center_idx].append(p)
        self.clusters = clusters

    def _update_centroids(self):
        """ Update centroids for a single iteration of k-means."""
        self._assign()
        new_centers = np.zeros(self.centers.shape)
        # Recompute new centroids
        for center_idx, cluster in sorted(self.clusters).items():
            # transform list of points in cluster -> ndarray of points
            cluster_pts = np.array(cluster)
            # Take the average of all points (aka along the rows, axis=0)
            # associated with the current centroid, and
            # use that as the new centroid.
            avg = np.sum(cluster_pts, axis=0) / cluster_pts.shape[0]
            new_centers[center_idx] = avg
        self.centers = new_centers

    def run(self, num_iters=10):
        """
        Runs K-means++ for num_iters iterations, or until
        centroids converge.
        """
        for i in range(num_iters):
            old_centroids = deepcopy(self.centers)
            self.update_centroids()
            if i != 0:
                if self.centers == old_centroids:
                    break
        return zip(self.centers, list(self.clusters.values()))
