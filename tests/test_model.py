"""Autoencoder end-to-end: learning on a synthetic pattern (both backends —
mantissa skips when the engine is absent, mirroring mantissa-cnn's parity
guard), MSE gradient vs finite differences, API and validation."""
import numpy as np
import pytest

import mantissa_cnn._engine as eng
from mantissa_cnn.layers import Conv2D, MaxPool2D

from mantissa_autoencoder.layers import Upsample2D
from mantissa_autoencoder.model import Autoencoder, mse_loss_grad


def _engine_ready() -> bool:
    try:
        eng.cnn_engine()
        return True
    except Exception:
        return False


BACKENDS = ["numpy",
            pytest.param("mantissa", marks=pytest.mark.skipif(
                not _engine_ready(),
                reason="mantissa engine with CNN primitives not available"))]


def stripe_images(n=64, size=8, seed=0):
    """Synthetic pattern with structure to learn: horizontal or vertical
    bright stripes at random offsets, plus light noise."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0.05, 0.02, size=(n, 1, size, size)).astype(np.float32)
    for i in range(n):
        j = rng.integers(0, size)
        if i % 2:
            X[i, 0, j, :] += 0.8
        else:
            X[i, 0, :, j] += 0.8
    return np.clip(X, 0.0, 1.0)


def tiny_ae(seed=0, backend="numpy"):
    return Autoencoder(
        [Conv2D(4, 3, pad=1, act="relu"), MaxPool2D(2)],
        [Upsample2D(2), Conv2D(1, 3, pad=1, act="identity")],
        seed=seed, backend=backend)


# -- learning -------------------------------------------------------------------

@pytest.mark.parametrize("backend", BACKENDS)
def test_fit_decreases_loss(backend):
    X = stripe_images()
    ae = tiny_ae(backend=backend).fit(X, epochs=8, batch_size=16, lr=0.05)
    loss = ae.history_["loss"]
    assert len(loss) == 8
    assert loss[-1] < loss[0] * 0.5                          # it learns
    assert loss[-1] == min(loss[0], loss[-1])


def test_same_seed_same_run():
    X = stripe_images()
    a = tiny_ae(seed=3).fit(X, epochs=2, batch_size=16, lr=0.05)
    b = tiny_ae(seed=3).fit(X, epochs=2, batch_size=16, lr=0.05)
    assert a.history_["loss"] == b.history_["loss"]
    assert np.array_equal(a.encoder[0].K, b.encoder[0].K)


def test_noise_is_applied_to_input_only():
    X = stripe_images(n=32)
    calls = []

    def noisy(batch):
        calls.append(batch.shape)
        return np.zeros_like(batch)          # destroy the input entirely

    ae = tiny_ae().fit(X, epochs=1, batch_size=16, noise=noisy, lr=0.0)
    assert len(calls) == 2                   # 32 samples / bs 16
    # lr=0 and zeroed input: the loss is MSE(const reconstruction, clean X),
    # so the clean target must still carry the stripes.
    assert ae.history_["loss"][0] > 0.01


def test_explicit_target_srcnn_style():
    X = stripe_images(n=32)
    blurry = X * 0.5                         # stand-in low-quality input
    ae = tiny_ae().fit(blurry, target=X, epochs=6, batch_size=16, lr=0.05)
    assert ae.history_["loss"][-1] < ae.history_["loss"][0]


def test_partial_tail_batch_is_handled():
    X = stripe_images(n=40)                  # bs 16 -> tail 8
    ae = tiny_ae().fit(X, epochs=2, batch_size=16, lr=0.01)
    assert len(ae.history_["loss"]) == 2


# -- MSE loss/gradient ------------------------------------------------------------

def test_mse_grad_matches_finite_difference():
    rng = np.random.default_rng(7)
    Y = rng.normal(size=(3, 2, 4, 4))
    T = rng.normal(size=(3, 2, 4, 4))
    dY = np.empty_like(Y)
    loss = mse_loss_grad(Y, T, dY)
    assert np.isclose(loss, np.mean((Y - T) ** 2))

    eps = 1e-6
    fd = np.empty_like(Y)
    flat, fdf = Y.reshape(-1), fd.reshape(-1)
    for i in range(flat.size):
        old = flat[i]
        flat[i] = old + eps
        hi = np.mean((Y - T) ** 2)
        flat[i] = old - eps
        lo = np.mean((Y - T) ** 2)
        flat[i] = old
        fdf[i] = (hi - lo) / (2 * eps)
    assert np.allclose(dY, fd, rtol=1e-5, atol=1e-8)


# -- API ---------------------------------------------------------------------------

def test_encode_decode_reconstruct_agree():
    X = stripe_images(n=8)
    ae = tiny_ae().fit(X, epochs=1, batch_size=8, lr=0.01)
    Z = ae.encode(X)
    assert Z.shape == (8,) + ae.latent_shape_
    R = ae.reconstruct(X)
    assert R.shape == X.shape
    assert np.allclose(ae.decode(Z), R, atol=1e-6)


def test_summary_marks_bottleneck():
    ae = tiny_ae().build((1, 8, 8))
    s = ae.summary()
    assert "bottleneck" in s and "total params" in s


def test_input_validation():
    X = stripe_images(n=8)
    ae = tiny_ae()
    with pytest.raises(RuntimeError, match="build"):
        ae.encode(X)                                    # not built yet
    ae.fit(X, epochs=1, batch_size=8)
    with pytest.raises(ValueError, match="sample shape"):
        ae.reconstruct(np.zeros((2, 1, 9, 9), dtype=np.float32))
    with pytest.raises(ValueError, match="target"):
        ae.fit(X, target=X[:, :, :4, :4], epochs=1)
    with pytest.raises(ValueError, match="samples"):
        ae.fit(X, target=X[:4], epochs=1)
    with pytest.raises(ValueError, match="backend"):
        Autoencoder([], [], backend="torch")


def test_missing_decoder_output_shape_check():
    from mantissa_cnn.layers import Flatten
    ae = Autoencoder([Conv2D(2, 3, pad=1)], [Flatten()], backend="numpy")
    with pytest.raises(ValueError, match=r"\(c, h, w\)"):
        ae.build((1, 8, 8))


@pytest.mark.skipif(not _engine_ready(),
                    reason="mantissa engine with CNN primitives not available")
def test_backend_parity_two_training_steps():
    """Same seed, same data, 2 SGD steps: C engine == numpy oracle."""
    X = stripe_images(n=16)

    def train(backend):
        ae = tiny_ae(seed=1, backend=backend)
        ae.fit(X, epochs=2, batch_size=16, lr=0.05)     # 1 batch = 1 step/epoch
        return ae

    a, b = train("mantissa"), train("numpy")
    assert np.allclose(a.history_["loss"], b.history_["loss"], rtol=1e-4)
    for la, lb in zip(a.layers, b.layers):
        for attr in ("K", "b", "W"):
            if hasattr(la, attr):
                assert np.allclose(getattr(la, attr), getattr(lb, attr),
                                   rtol=1e-4, atol=1e-5), type(la).__name__
    assert np.allclose(a.reconstruct(X), b.reconstruct(X), rtol=1e-4, atol=1e-5)
