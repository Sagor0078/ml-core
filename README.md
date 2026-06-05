# Land on Vector Spaces — practical linear algebra with Python

A visual, computational tour of linear algebra using NumPy and Matplotlib.
Each notebook builds intuition for a core idea and then **applies it to a
real-world problem**.

## Notebooks

| Notebook | Concept | Real-world application |
|----------|---------|------------------------|
| `00_linear-algebra.ipynb` | Vectors, dot product | Document similarity (cosine similarity) |
| `01_vector_transformations.ipynb` | Linear transformations | 2D computer graphics (rotate / scale / shear a shape) |
| `02_matrix_operations.ipynb` | Matrices as systems | Least-squares line fitting (regression) |
| `03_eigenvalues_vectors.ipynb` | Eigen-decomposition | Principal Component Analysis (PCA) |
| `04_singular_value_decompositions.ipynb` | SVD | Image compression, a movie recommender, and Latent Semantic Analysis |

## The `rw_examples.py` helper

The real-world examples are backed by [`rw_examples.py`](rw_examples.py): small,
pure-NumPy functions (`cosine_similarity`, `transform_points`, `pca`,
`svd_compress`, `recommend_svd`, `lsa`, …). Every function **validates its
inputs** — checking shapes, dtypes and value ranges — and raises a clear,
actionable error instead of failing deep inside NumPy. This keeps the
notebooks robust when you experiment with your own data.

## Setup

```bash
pip install -r requirements.txt
jupyter lab            # or: jupyter notebook
```

No network access is required to run the examples — the image-compression demo
uses the bundled `images/washington-monument.jpg`, and all other datasets are
generated locally.

## Tests

```bash
python -m pytest test_rw_examples.py
```

Covers both correctness and the input-validation/error paths of every helper.
