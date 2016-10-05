import numpy as np
import pandas as pd
from numpy import linalg as LA
from collections import defaultdict
from copy import deepcopy


class SpectralCluster(object):
    def __init__(self, df, k, sigma=1):
        """
        Given a pandas dataframe and the of clusters,
        Initializes spectral clustering to its initial K centroids
        via uniform random sampling (without replacement).

        Params:
            df - pandas dataframe containing points to be clustered
            k - number of clusters to form
            sigma - scaling parameter.
        """

        # Implemented against the following paper from 2001:
        # http://ai.stanford.edu/~ang/papers/nips01-spectral.pdf

        self.k = min(k, df.shape[0])
        self.orig_points = df.as_matrix()
        self.centers = None  # Centroids in spectral subspace
        self.points = None  # Points in spectral subspace

        self._spectral_init(sigma)
        self.clusters = defaultdict(list)

    def _spectral_init(self, sigma):
        """ Map the original dataset onto the spectral
        subspace and find the initial centroids in the subspace.
        """
        # Form the affinity matrix A
        n = self.orig_points.shape[0]  # Number of points.
        points = self.orig_points
        A = points.reshape(1, -1).T ** 2 - np.tile(points, (n, 1)) ** 2
        A = np.exp(-A / (2 * np.power(sigma, 2)))
        np.fill_diagonal(A, 0)

        # Find the Laplacian matrix L, which is equal to D^(-1/2)AD^(-1/2),
        # where D is defined as a diagonal matrix with diagonal entries equal
        # to the sums of the rows of A.
        D = np.diag(A.sum(axis=1))
        D_sqrt_recip = np.reciprocal(np.power(D, .5))  # D^(-1/2)
        L = D_sqrt_recip @ A @ D_sqrt_recip
        w, v = LA.eig(L)  # Eigenvalues and eigenvectors of L
        eig_pairs = list(zip(w, v))

        # Sort eigenvalue/eigenvector pairs from high to low
        # eigenvecs should be orthogonal because L should be a symmetric matrix
        # spectral clustering assumes this to be true.
        eig_pairs.sort(key=lambda x: x[0], reverse=True)
        # only keep k largest eigenvectors
        k_largest_w = [v for w, v in eig_pairs[:self.k]]
        # Convert list into a ndarray, where eigenvectors are the column vecs
        # of the matrix sorted greatest->least as column index increases
        # (left->right)
        X = np.array(k_largest_w).T

        # Form the matrix Y from X by renormalizing each of X's rows to have
        # unit length.
        Y = X / np.sqrt((X ** 2).sum(axis=1))
        self.points = Y
        self.centers = pd.DataFrame(Y).sample(self._k).as_matrix()

    def _find_center(self, p):
        """ Calculates the closest center for a point based
        on Euclidean distance.

        Args:
            p - index of a point within self.points

        Returns:
            Index of the center closest to current pixel
        """
        center_dists = np.linalg.norm(self.points[p] - self.centers, axis=1)
        return np.argmin(center_dists)

    def _assign(self):
        """Assign each point to its nearest centroid."""
        clusters = defaultdict(list)
        for p_idx, p in enumerate(self.points):
            center_idx = self._find_center(p)
            clusters[center_idx].append(p_idx)
        self.clusters = clusters

    def update_centroids(self):
        """ Update centroids for a single iteration of k-means."""
        self._assign()
        new_centers = np.zeros(self.centers.shape)
        # Recompute new centroids
        for center_idx, cluster in sorted(self.clusters).items():
            # transform list of point indices in cluster -> ndarray of points
            cluster_pts = np.array([self.points[p_idx] for p_idx in cluster])
            # Take the average of all points (aka along the rows, axis=0)
            # associated with the current centroid, and
            # use that as the new centroid.
            avg = np.sum(cluster_pts, axis=0) / cluster_pts.shape[0]
            new_centers[center_idx] = avg
        self.centers = new_centers

    def _spectral_to_original(self):
        """ Map the spectral clustering back to the original
        data points.

        Return:
            Clustering of points in original space.
        """
        # embedding is a nested list containing the indices of the points
        # that belong to each cluster.
        embedding = self.clusters.values()
        orig_clustering = [[self.orig_points[p_idx] for p_idx in cluster]
                           for cluster in embedding]
        return orig_clustering

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
        # Map clustering in spectral subspace back to original dataset
        return zip(self.centers, self._spectral_to_original())
