"""Real-world linear-algebra examples for the "Land on Vector Spaces" module.

Each function is a small, self-contained application of an idea from the
lessons (dot products, linear transformations, least squares, eigen/SVD
decompositions). Everything is implemented with plain NumPy so the linear
algebra stays visible -- no scikit-learn shortcuts.

Design goals (robustness):
    * every public function validates its inputs and raises a clear,
      actionable error instead of letting NumPy fail deep in a computation;
    * shapes, dtypes and value ranges are checked up front;
    * functions are pure (no global state) and importable for unit testing.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "cosine_similarity",
    "rotation_matrix",
    "scaling_matrix",
    "shear_matrix",
    "transform_points",
    "least_squares_fit",
    "pca",
    "svd_compress",
    "recommend_svd",
    "lsa",
]


# --------------------------------------------------------------------------- #
# Internal validation helpers
# --------------------------------------------------------------------------- #
def _as_float_array(x, name):
    """Return ``x`` as a float ndarray or raise a clear error."""
    try:
        arr = np.asarray(x, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be array-like of numbers; got {type(x).__name__}.") from exc
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return arr


def _check_2d(arr, name):
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array; got {arr.ndim}D with shape {arr.shape}.")
    if arr.size == 0:
        raise ValueError(f"{name} is empty.")
    return arr


def _check_rank(k, limit, name="k"):
    if not isinstance(k, (int, np.integer)):
        raise TypeError(f"{name} must be an integer; got {type(k).__name__}.")
    k = int(k)
    if k < 1:
        raise ValueError(f"{name} must be >= 1; got {k}.")
    if k > limit:
        raise ValueError(f"{name}={k} exceeds the maximum of {limit}.")
    return k


# --------------------------------------------------------------------------- #
# Lesson 00 -- vectors: text / feature similarity
# --------------------------------------------------------------------------- #
def cosine_similarity(u, v):
    """Cosine of the angle between two vectors: ``u . v / (|u| |v|)``.

    The workhorse of search engines and recommender systems for comparing
    feature vectors (e.g. word-count vectors of two documents).

    Returns a float in [-1, 1]. Raises if shapes differ or a vector is zero.
    """
    u = _as_float_array(u, "u")
    v = _as_float_array(v, "v")
    if u.ndim != 1 or v.ndim != 1:
        raise ValueError(f"u and v must be 1D vectors; got shapes {u.shape} and {v.shape}.")
    if u.shape != v.shape:
        raise ValueError(f"u and v must have the same length; got {u.shape[0]} and {v.shape[0]}.")
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu == 0 or nv == 0:
        raise ValueError("cosine_similarity is undefined for a zero vector.")
    return float(np.dot(u, v) / (nu * nv))


# --------------------------------------------------------------------------- #
# Lesson 01 -- linear transformations: 2D computer graphics
# --------------------------------------------------------------------------- #
def rotation_matrix(degrees):
    """2x2 counter-clockwise rotation matrix for an angle in degrees."""
    theta = np.radians(float(degrees))
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def scaling_matrix(sx, sy=None):
    """2x2 scaling matrix. ``scaling_matrix(2)`` scales both axes by 2."""
    sy = sx if sy is None else sy
    return np.array([[float(sx), 0.0], [0.0, float(sy)]])


def shear_matrix(kx=0.0, ky=0.0):
    """2x2 shear matrix (``kx`` shears x by y, ``ky`` shears y by x)."""
    return np.array([[1.0, float(kx)], [float(ky), 1.0]])


def transform_points(points, A):
    """Apply 2x2 transformation ``A`` to an array of 2D ``points``.

    ``points`` has shape ``(N, 2)`` (one row per point). Returns the
    transformed points with the same shape -- the basis of every 2D graphics
    pipeline (rotate/scale/shear a sprite or shape).
    """
    points = _check_2d(_as_float_array(points, "points"), "points")
    A = _check_2d(_as_float_array(A, "A"), "A")
    if points.shape[1] != 2:
        raise ValueError(f"points must have shape (N, 2); got {points.shape}.")
    if A.shape != (2, 2):
        raise ValueError(f"A must be a 2x2 matrix; got shape {A.shape}.")
    return points @ A.T


# --------------------------------------------------------------------------- #
# Lesson 02 -- matrices as systems: least-squares regression
# --------------------------------------------------------------------------- #
def least_squares_fit(X, y, fit_intercept=True):
    """Solve the least-squares problem ``min ||X b - y||`` via the normal
    equations ``(X^T X) b = X^T y``.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    y : array-like, shape (n_samples,)
    fit_intercept : bool, prepend a column of ones for the bias term.

    Returns
    -------
    coef : ndarray, the fitted coefficients (first entry is the intercept
           when ``fit_intercept`` is True).

    Raises if the system is under-determined or rank-deficient (so you get a
    clear message instead of a silently meaningless fit).
    """
    X = _as_float_array(X, "X")
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    _check_2d(X, "X")
    y = _as_float_array(y, "y").ravel()
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X has {X.shape[0]} rows but y has {y.shape[0]} entries.")

    A = np.hstack([np.ones((X.shape[0], 1)), X]) if fit_intercept else X
    if A.shape[0] < A.shape[1]:
        raise ValueError(
            f"Under-determined system: {A.shape[0]} samples for {A.shape[1]} "
            "unknowns. Add more data or fewer features."
        )
    if np.linalg.matrix_rank(A) < A.shape[1]:
        raise ValueError(
            "Design matrix is rank-deficient (collinear features); the normal "
            "equations have no unique solution. Use svd-based lstsq instead."
        )
    return np.linalg.solve(A.T @ A, A.T @ y)


# --------------------------------------------------------------------------- #
# Lesson 03 -- eigenvectors: Principal Component Analysis
# --------------------------------------------------------------------------- #
def pca(X, n_components):
    """Principal Component Analysis via eigendecomposition of the covariance.

    Reduces ``X`` (shape ``(n_samples, n_features)``) to ``n_components``
    dimensions while keeping as much variance as possible -- the standard
    tool for visualising and compressing high-dimensional data.

    Returns
    -------
    projected : ndarray (n_samples, n_components)
    components : ndarray (n_components, n_features), the principal axes
    explained_variance_ratio : ndarray (n_components,), fraction of total
        variance captured by each component.
    """
    X = _check_2d(_as_float_array(X, "X"), "X")
    n_components = _check_rank(n_components, X.shape[1], "n_components")

    Xc = X - X.mean(axis=0)
    # Covariance matrix is symmetric -> use eigh (real eigenvalues, ordered).
    cov = np.cov(Xc, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]          # largest variance first
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    components = eigvecs[:, :n_components].T
    projected = Xc @ components.T
    total = eigvals.sum()
    ratio = (eigvals[:n_components] / total) if total > 0 else np.zeros(n_components)
    return projected, components, ratio


# --------------------------------------------------------------------------- #
# Lesson 04 -- SVD: image compression
# --------------------------------------------------------------------------- #
def svd_compress(image, k):
    """Rank-``k`` approximation of a grayscale ``image`` via truncated SVD.

    Returns
    -------
    approx : ndarray, the reconstructed image (same shape, clipped to the
        original value range).
    ratio : float, storage ratio ``stored_values / original_values``; lower
        is better compression.
    """
    image = _check_2d(_as_float_array(image, "image"), "image")
    m, n = image.shape
    k = _check_rank(k, min(m, n), "k")

    U, s, Vt = np.linalg.svd(image, full_matrices=False)
    approx = (U[:, :k] * s[:k]) @ Vt[:k, :]
    approx = np.clip(approx, image.min(), image.max())

    stored = k * (m + n + 1)          # U[:, :k], s[:k], Vt[:k, :]
    ratio = stored / (m * n)
    return approx, float(ratio)


# --------------------------------------------------------------------------- #
# Lesson 04 -- SVD: collaborative-filtering recommender
# --------------------------------------------------------------------------- #
def recommend_svd(ratings, k):
    """Predict missing entries of a user-item ``ratings`` matrix with a
    rank-``k`` SVD (the core idea behind the Netflix-prize methods).

    Missing ratings must be encoded as ``0``. They are mean-filled before the
    decomposition, then the low-rank reconstruction estimates them.

    Returns the full predicted ratings matrix (same shape as ``ratings``).
    """
    R = _check_2d(_as_float_array(ratings, "ratings"), "ratings")
    if np.any(R < 0):
        raise ValueError("ratings must be non-negative (use 0 for 'not rated').")
    k = _check_rank(k, min(R.shape), "k")

    mask = R > 0
    if not mask.any():
        raise ValueError("ratings matrix has no observed (non-zero) entries.")
    # Fill unknowns with each item's mean rating so they don't bias the SVD.
    item_means = np.array(
        [col[col > 0].mean() if np.any(col > 0) else 0.0 for col in R.T]
    )
    filled = np.where(mask, R, item_means)

    U, s, Vt = np.linalg.svd(filled, full_matrices=False)
    return (U[:, :k] * s[:k]) @ Vt[:k, :]


# --------------------------------------------------------------------------- #
# Lesson 04 -- SVD: Latent Semantic Analysis (NLP)
# --------------------------------------------------------------------------- #
def lsa(term_doc, k):
    """Latent Semantic Analysis: reduce a term-document matrix to ``k`` latent
    topics with truncated SVD.

    ``term_doc`` has shape ``(n_terms, n_docs)`` (entry = weight of a term in a
    document). Returns

    doc_topics : ndarray (n_docs, k), each document in latent-topic space
    term_topics : ndarray (n_terms, k), each term in latent-topic space
    doc_similarity : ndarray (n_docs, n_docs), cosine similarity between docs
        in topic space (captures meaning, not just shared words).
    """
    M = _check_2d(_as_float_array(term_doc, "term_doc"), "term_doc")
    k = _check_rank(k, min(M.shape), "k")

    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    term_topics = U[:, :k] * s[:k]
    doc_topics = (np.diag(s[:k]) @ Vt[:k, :]).T

    # Cosine similarity between documents in topic space.
    norms = np.linalg.norm(doc_topics, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = doc_topics / norms
    doc_similarity = unit @ unit.T
    return doc_topics, term_topics, doc_similarity
