import numpy as np
from collections import defaultdict
from copy import deepcopy


class KMeans(object):
    def __init__(self, df, k):
        """
        Given a pandas dataframe and the of clusters,
        Initializes k-means to its initial K centroids
        via uniform random sampling (without replacement).

        Params:
            df - pandas dataframe containing points to be clustered
            k - number of clusters to form
        """

        # Got some help/inspiration from this StackOverflow link:
        # http://stackoverflow.com/questions/5466323/how-exactly-does-k-means-work
        # Also read this link (a bit confusing, but helpful too):
        # https://datasciencelab.wordpress.com/2014/01/15/improved-seeding-for-clustering-with-k-means/

        self.k = min(k, df.shape[0])
        self.points = df.as_matrix()
        self.centers = df.sample(self._k).as_matrix()
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

    def update_centroids(self):
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
