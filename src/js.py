import numpy as np
from scipy.special import rel_entr
from scipy.spatial.distance import mahalanobis, squareform, pdist, jensenshannon

class JSKMeans:
    """
    K-Means clustering using Jensen-Shannon Distance.
    Optimized and Robust to floating point errors.
    """
    def __init__(self, n_clusters=5, max_iter=300, tol=1e-4, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None

    def fit(self, X):
        # X must be Probability Distributions (rows sum to 1)
        np.random.seed(self.random_state)

        # 1. Initialize Centroids (Randomly pick k samples)
        random_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.centroids = X[random_indices]

        for i in range(self.max_iter):
            # 2. Assignment Step: Calculate JS Distance
            distances = self._calc_js_distances(X, self.centroids)
            labels = np.argmin(distances, axis=1)

            # 3. Update Step: New centroids
            new_centroids = np.zeros_like(self.centroids)
            for k in range(self.n_clusters):
                mask = (labels == k)
                if np.any(mask):
                    # Arithmetic mean is a valid approximation for centroid
                    new_centroids[k] = X[mask].mean(axis=0)
                else:
                    # Handle empty cluster: keep old or re-initialize
                    new_centroids[k] = self.centroids[k]

            # 4. Check Convergence
            shift = np.linalg.norm(self.centroids - new_centroids)
            self.centroids = new_centroids
            self.labels_ = labels

            if shift < self.tol:
                break

        # Final inertia
        final_dists = self._calc_js_distances(X, self.centroids)
        self.inertia_ = np.sum(np.min(final_dists, axis=1)**2)
        return self

    def predict(self, X):
        distances = self._calc_js_distances(X, self.centroids)
        return np.argmin(distances, axis=1)

    def _calc_js_distances(self, X, centroids):
        """
        Robust calculation of JS distance between every point in X and every Centroid.
        Returns matrix (N_samples, N_clusters).
        """
        n_samples = X.shape[0]
        n_clusters = centroids.shape[0]
        dists = np.zeros((n_samples, n_clusters))

        # Loop over centroids (K is usually small, so this is fine)
        for k in range(n_clusters):
            p = X
            q = centroids[k].reshape(1, -1)

            # Mixture distribution M = 0.5(P + Q)
            m = 0.5 * (p + q)

            # KL(P || M) + KL(Q || M)
            # rel_entr(x, y) computes x * log(x / y) safely
            # We sum along axis 1 (features) to get divergence per sample
            left = rel_entr(p, m).sum(axis=1)
            right = rel_entr(q, m).sum(axis=1)

            js_divergence = 0.5 * (left + right)

            # --- FIX: Clamp to 0 to avoid sqrt(-1e-10) error ---
            js_divergence = np.maximum(js_divergence, 0.0)

            dists[:, k] = np.sqrt(js_divergence)

        return dists


def compute_robust_js_matrix(X):
    print(f"Computing JS distance matrix for {X.shape[0]} samples...")
    def js_metric(p, q):
        p = p.reshape(1, -1)
        q = q.reshape(1, -1)
        m = 0.5 * (p + q)
        left = rel_entr(p, m).sum()
        right = rel_entr(q, m).sum()
        return np.sqrt(max(0.5 * (left + right), 0.0))
    return squareform(pdist(X, metric=jensenshannon))

