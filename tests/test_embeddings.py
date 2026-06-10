import numpy as np
import pytest

from prism.embeddings.base import Embedder


def test_average_unit_vector_returns_unit_vector():
    rng = np.random.default_rng(42)
    raw = rng.standard_normal((5, 8)).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    vecs = raw / norms

    avg = Embedder.average_unit_vector(vecs)
    assert avg.shape == (8,)
    assert avg.dtype == np.float32
    assert np.isclose(np.linalg.norm(avg), 1.0, atol=1e-5)


def test_average_unit_vector_of_identical_vectors_is_that_vector():
    v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    arr = np.stack([v, v, v], axis=0)
    avg = Embedder.average_unit_vector(arr)
    assert np.allclose(avg, v)


def test_average_unit_vector_degenerate_zero_mean():
    """Opposing unit vectors average to zero — function must not blow up."""
    arr = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
    avg = Embedder.average_unit_vector(arr)
    assert avg.shape == (2,)
    assert np.allclose(avg, [0.0, 0.0])


def test_average_unit_vector_empty_raises():
    with pytest.raises(ValueError):
        Embedder.average_unit_vector(np.zeros((0, 4), dtype=np.float32))
