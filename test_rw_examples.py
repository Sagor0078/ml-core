"""Unit tests for rw_examples -- both correctness and input validation.

Run with:  python -m pytest test_rw_examples.py   (or plain `python test_rw_examples.py`)
"""
import numpy as np
import pytest

import rw_examples as rw


# --- correctness ----------------------------------------------------------- #
def test_cosine_similarity_bounds():
    assert rw.cosine_similarity([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert rw.cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)


def test_transform_points_rotation():
    pts = np.array([[1.0, 0.0]])
    out = rw.transform_points(pts, rw.rotation_matrix(90))
    assert out[0] == pytest.approx([0.0, 1.0], abs=1e-9)


def test_least_squares_recovers_slope():
    x = np.linspace(0, 10, 50)
    y = 3 * x + 2
    b0, b1 = rw.least_squares_fit(x, y)
    assert b0 == pytest.approx(2.0, abs=1e-6)
    assert b1 == pytest.approx(3.0, abs=1e-6)


def test_pca_variance_sums_to_one():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))
    _, _, ratio = rw.pca(X, 3)
    assert ratio.sum() == pytest.approx(1.0)


def test_svd_compress_full_rank_is_lossless():
    rng = np.random.default_rng(1)
    img = rng.random((10, 8))
    approx, ratio = rw.svd_compress(img, 8)
    assert np.allclose(approx, img, atol=1e-9)
    assert 0 < ratio


def test_recommender_fills_missing():
    R = np.array([[5, 4, 0], [4, 0, 1], [0, 2, 5]], float)
    pred = rw.recommend_svd(R, 2)
    assert pred.shape == R.shape
    assert np.all(np.isfinite(pred))


def test_lsa_groups_similar_docs():
    M = np.array([[1, 1, 0, 0], [1, 1, 1, 0], [0, 0, 1, 1]], float)
    _, _, sim = rw.lsa(M, 2)
    assert sim[0, 1] > sim[0, 3]   # docs 0 & 1 more alike than 0 & 3


# --- validation / error paths ---------------------------------------------- #
@pytest.mark.parametrize("call", [
    lambda: rw.cosine_similarity([1, 2], [1, 2, 3]),       # length mismatch
    lambda: rw.cosine_similarity([0, 0], [1, 2]),          # zero vector
    lambda: rw.transform_points(np.zeros((3, 2)), np.eye(3)),  # non-2x2 matrix
    lambda: rw.least_squares_fit([[1, 2, 3]], [1]),        # under-determined
    lambda: rw.pca(np.zeros((5, 2)), 5),                   # too many components
    lambda: rw.svd_compress(np.zeros((4, 4)), 99),         # rank too large
    lambda: rw.recommend_svd(-np.ones((2, 2)), 1),         # negative ratings
    lambda: rw.lsa(np.ones((3, 3)), 0),                    # rank < 1
])
def test_invalid_inputs_raise(call):
    with pytest.raises((ValueError, TypeError)):
        call()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
