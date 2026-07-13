"""Upsample2D / Reshape: shapes, exact values, adjoint correctness (numeric
vs analytic), scratch reuse. All on the numpy backend — no engine needed."""
import numpy as np
import pytest

from mantissa_cnn import _numpy_backend as B
from mantissa_cnn.layers import Conv2D
from mantissa_autoencoder.layers import Reshape, Upsample2D


def _built(layer, in_shape, seed=0):
    layer.build(in_shape, np.random.default_rng(seed))
    return layer


# -- Upsample2D ----------------------------------------------------------------

def test_upsample_forward_repeats_blocks():
    up = _built(Upsample2D(2), (1, 2, 2))
    X = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
    Y = up.forward(X, B)
    assert np.array_equal(Y[0, 0], [[1, 1, 2, 2],
                                    [1, 1, 2, 2],
                                    [3, 3, 4, 4],
                                    [3, 3, 4, 4]])


@pytest.mark.parametrize("scale, in_shape", [(2, (3, 4, 5)), (3, (1, 2, 2))])
def test_upsample_shapes(scale, in_shape):
    up = _built(Upsample2D(scale), in_shape)
    c, h, w = in_shape
    assert up.out_shape == (c, h * scale, w * scale)
    X = np.random.default_rng(1).random((2,) + in_shape, dtype=np.float32)
    Y = up.forward(X, B)
    assert Y.shape == (2,) + up.out_shape and Y.dtype == np.float32
    assert up.backward(Y, B).shape == X.shape


def test_upsample_backward_is_block_sum():
    up = _built(Upsample2D(2), (1, 2, 2))
    dY = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
    dX = up.backward(dY, B)
    # block sums of [[0..3],[4..7],[8..11],[12..15]]
    assert np.array_equal(dX[0, 0], [[0 + 1 + 4 + 5, 2 + 3 + 6 + 7],
                                     [8 + 9 + 12 + 13, 10 + 11 + 14 + 15]])


def test_upsample_gradient_numeric_through_tiny_net():
    """Analytic dX through Upsample2D inside a tiny net vs central finite
    differences. Identity-activation conv keeps the net linear in X, so the
    central difference is exact up to float32 rounding."""
    rng = np.random.default_rng(42)
    up = _built(Upsample2D(2), (2, 3, 3))
    conv = _built(Conv2D(2, 3, pad=1, act="identity"), up.out_shape, seed=1)
    X = rng.random((2, 2, 3, 3), dtype=np.float32)
    R = rng.random((2,) + conv.out_shape, dtype=np.float32)

    def loss(x):
        return float((conv.forward(up.forward(x, B), B) * R).sum())

    loss(X)                                   # populate conv._X / Z scratch
    dX = up.backward(conv.backward(np.ascontiguousarray(R), B), B).copy()

    eps = 1e-2
    fd = np.empty_like(X)
    flat, fdf = X.reshape(-1), fd.reshape(-1)
    for i in range(flat.size):
        old = flat[i]
        flat[i] = old + eps
        hi = loss(X)
        flat[i] = old - eps
        lo = loss(X)
        flat[i] = old
        fdf[i] = (hi - lo) / (2 * eps)
    assert np.allclose(dX, fd, rtol=1e-3, atol=1e-3)


def test_upsample_scratch_reused_across_batches():
    up = _built(Upsample2D(2), (1, 4, 4))
    X = np.random.default_rng(5).random((6, 1, 4, 4), dtype=np.float32)
    assert up.forward(X, B) is up.forward(X, B)              # same Y buffer
    dY = np.ones((6,) + up.out_shape, dtype=np.float32)
    assert up.backward(dY, B) is up.backward(dY, B)          # same dX buffer


def test_upsample_rejects_bad_scale():
    with pytest.raises(ValueError, match="scale"):
        Upsample2D(0)


# -- Reshape -------------------------------------------------------------------

def test_reshape_roundtrip():
    rs = _built(Reshape((32, 7, 7)), (1568,))
    assert rs.out_shape == (32, 7, 7)
    X = np.random.default_rng(3).random((4, 1568), dtype=np.float32)
    Y = rs.forward(X, B)
    assert Y.shape == (4, 32, 7, 7)
    assert np.array_equal(rs.backward(Y, B), X)
    assert Y.base is X                                       # a view, no copy


def test_reshape_size_mismatch_raises():
    with pytest.raises(ValueError, match="Reshape"):
        Reshape((32, 7, 7)).build((100,), np.random.default_rng(0))


def test_layers_are_parameter_free():
    up = _built(Upsample2D(2), (1, 4, 4))
    rs = _built(Reshape((16,)), (4, 2, 2))
    assert up.param_count() == 0 and rs.param_count() == 0
    up.step(B, 0.1)                                          # no-op, no error
    rs.step(B, 0.1)
