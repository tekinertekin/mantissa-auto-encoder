"""Pure task helpers: psnr, roc_auc (against hand-computed cases), noise,
downscale/upscale, latent quantization, anomaly scores. No downloads, no
engine — tiny fabricated arrays only."""
import math

import numpy as np
import pytest

from mantissa_autoencoder import tasks


# -- psnr ------------------------------------------------------------------------

def test_psnr_identical_is_inf():
    X = np.random.default_rng(0).random((2, 1, 4, 4), dtype=np.float32)
    assert tasks.psnr(X, X) == math.inf


def test_psnr_known_value():
    a = np.zeros((1, 1, 2, 2), dtype=np.float32)
    b = np.full_like(a, 0.5)                 # MSE 0.25 -> 10*log10(1/0.25)
    assert np.isclose(tasks.psnr(a, b), 10 * math.log10(4.0))
    # doubling the peak adds 20*log10(2) dB
    assert np.isclose(tasks.psnr(a, b, peak=2.0),
                      tasks.psnr(a, b) + 20 * math.log10(2.0))


# -- roc_auc ----------------------------------------------------------------------

def test_roc_auc_hand_computed():
    # positives {0.35, 0.8} vs negatives {0.1, 0.4}: 3 of 4 pairs correctly
    # ordered -> 0.75
    assert tasks.roc_auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == 0.75


def test_roc_auc_perfect_and_inverted():
    assert tasks.roc_auc([1, 2, 3, 4], [0, 0, 1, 1]) == 1.0
    assert tasks.roc_auc([4, 3, 2, 1], [0, 0, 1, 1]) == 0.0


def test_roc_auc_ties_count_half():
    assert tasks.roc_auc([1.0, 1.0], [0, 1]) == 0.5
    # one tie among four: pairs (2>1)=1, (2=2)=0.5, (3>1)=1, (3>2)=1 -> 3.5/4
    assert tasks.roc_auc([1.0, 2.0, 2.0, 3.0], [0, 0, 1, 1]) == 0.875


def test_roc_auc_needs_both_classes():
    with pytest.raises(ValueError, match="positive"):
        tasks.roc_auc([1.0, 2.0], [1, 1])
    with pytest.raises(ValueError, match="labels"):
        tasks.roc_auc([1.0, 2.0], [1])


# -- noise --------------------------------------------------------------------------

def test_add_gaussian_noise_shape_range_and_purity():
    X = np.random.default_rng(1).random((3, 1, 6, 6), dtype=np.float32)
    X0 = X.copy()
    N = tasks.add_gaussian_noise(X, sigma=0.3, seed=0)
    assert N.shape == X.shape and N.dtype == np.float32
    assert N.min() >= 0.0 and N.max() <= 1.0
    assert np.array_equal(X, X0)                       # input untouched
    assert not np.array_equal(N, X)                    # it actually corrupts
    assert np.array_equal(N, tasks.add_gaussian_noise(X, sigma=0.3, seed=0))
    assert not np.array_equal(N, tasks.add_gaussian_noise(X, 0.3, seed=1))


# -- downscale / upscale ---------------------------------------------------------------

def test_downscale2x_is_block_mean():
    X = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
    D = tasks.downscale2x(X)
    assert D.shape == (1, 1, 2, 2) and D.dtype == np.float32
    assert np.array_equal(D[0, 0], [[2.5, 4.5], [10.5, 12.5]])
    with pytest.raises(ValueError, match="even"):
        tasks.downscale2x(np.zeros((1, 1, 5, 4), dtype=np.float32))


def test_nearest_upscale2x_repeats_and_roundtrips():
    X = np.random.default_rng(2).random((2, 3, 5, 4), dtype=np.float32)
    U = tasks.nearest_upscale2x(X)
    assert U.shape == (2, 3, 10, 8) and U.dtype == np.float32
    assert np.array_equal(U[:, :, ::2, ::2], X)
    # block mean of a constant block is that constant: exact inverse
    assert np.allclose(tasks.downscale2x(U), X, atol=1e-7)


# -- latent quantization ------------------------------------------------------------------

def test_quantize_latent_roundtrip_within_step():
    Z = np.random.default_rng(3).normal(size=(8, 32)).astype(np.float32) * 5
    q, lo, hi = tasks.quantize_latent(Z)
    assert q.dtype == np.uint8 and q.shape == Z.shape
    step = (float(hi) - float(lo)) / 255.0
    Zr = tasks.dequantize_latent(q, lo, hi)
    assert Zr.dtype == np.float32
    assert np.abs(Zr - Z).max() <= step / 2 + 1e-6


def test_quantize_latent_constant_batch():
    Z = np.full((4, 32), 1.5, dtype=np.float32)
    q, lo, hi = tasks.quantize_latent(Z)
    assert np.array_equal(q, np.zeros_like(q))
    assert np.allclose(tasks.dequantize_latent(q, lo, hi), Z)


# -- anomaly scores -------------------------------------------------------------------------

class _StubModel:
    """reconstruct() = blur to the per-sample mean — structured samples score
    high, flat samples score zero."""

    def reconstruct(self, X):
        return np.broadcast_to(X.mean(axis=(1, 2, 3), keepdims=True),
                               X.shape).astype(np.float32)


def test_anomaly_scores_per_sample_mse():
    flat = np.full((1, 1, 4, 4), 0.5, dtype=np.float32)
    spiky = np.zeros((1, 1, 4, 4), dtype=np.float32)
    spiky[0, 0, 0, 0] = 1.0
    scores = tasks.anomaly_scores(_StubModel(), np.concatenate([flat, spiky]))
    assert scores.shape == (2,)
    assert scores[0] == 0.0
    expected = np.mean((spiky - spiky.mean()) ** 2)
    assert np.isclose(scores[1], expected)
    assert tasks.roc_auc(scores, [0, 1]) == 1.0


# -- protocol recipes --------------------------------------------------------------------------

def test_task_recipes_are_fixed():
    assert set(tasks.TASKS) == {"denoise", "compress", "anomaly", "superres"}
    assert tasks.TASKS["denoise"]["dataset"] == "fashion_mnist"
    assert tasks.TASKS["denoise"]["sigma"] == 0.3
    assert tasks.TASKS["compress"]["latent"] == 32
    assert tasks.TASKS["anomaly"]["held_out_digit"] == 1
    assert tasks.TASKS["superres"]["scale"] == 2
