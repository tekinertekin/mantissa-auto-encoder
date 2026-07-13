"""The four benchmark tasks: pure numpy helpers + the fixed recipes.

Recipes (protocol-fixed; the bench harness and tests both read from here):

- **denoise** — fashion_mnist, ``denoise_ae``, Gaussian noise sigma 0.3 on
  the input only; metric PSNR(reconstruction, clean).
- **compress** — mnist, ``bottleneck_ae`` latent 32; PSNR through the float32
  code, plus a uint8-quantized variant at a REAL byte budget: 32 code bytes
  + 8 bytes (float32 lo/hi for dequantization) = 40 bytes per image vs the
  784-byte uint8 original — 19.6x. (Quoting 24.5x from the 32 code bytes
  alone would ignore the header; count what you ship.)
- **anomaly** — mnist with digit 1 held out of training; train on the other
  nine digits, score the test set by per-sample reconstruction MSE, metric
  AUC with the held-out 1s as positives (reconstruction-error anomaly
  detection: Sakurada & Yairi, 2014, "Anomaly Detection Using Autoencoders
  with Nonlinear Dimensionality Reduction", *MLSDA*).
- **superres** — mnist 28 -> 14 (2x2 mean) -> nearest-upscaled back to 28
  outside the net, ``srcnn`` refines; metric PSNR vs the original.

Everything here is memory-bound array plumbing, so it is plain numpy — no
sklearn, no scipy. PSNR peak is 1.0 throughout (images live in [0, 1]).
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["TASKS", "add_gaussian_noise", "psnr", "downscale2x",
           "nearest_upscale2x", "anomaly_scores", "roc_auc",
           "quantize_latent", "dequantize_latent"]

# The fixed recipes: dataset + zoo model + task parameters + metric.
TASKS = {
    "denoise": dict(dataset="fashion_mnist", model="denoise_ae",
                    sigma=0.3, metric="psnr"),
    "compress": dict(dataset="mnist", model="bottleneck_ae",
                     latent=32, metric="psnr"),
    "anomaly": dict(dataset="mnist", model="bottleneck_ae",
                    held_out_digit=1, metric="auc"),
    "superres": dict(dataset="mnist", model="srcnn",
                     scale=2, metric="psnr"),
}


def add_gaussian_noise(X, sigma: float = 0.3, seed=None):
    """X + N(0, sigma^2), clipped back to [0, 1], float32. Pure — the input
    is not modified. The default ``seed=None`` draws fresh noise every call:
    the denoising recipe corrupts each mini-batch anew (Vincent et al.,
    2008). Pass a seed for a reproducible corruption (e.g. the fixed test
    set of a benchmark)."""
    rng = np.random.default_rng(seed)
    out = X + rng.normal(0.0, sigma, size=X.shape).astype(np.float32)
    return np.clip(out, 0.0, 1.0, out=out)


def psnr(a, b, peak: float = 1.0) -> float:
    """Peak signal-to-noise ratio in dB: 10*log10(peak^2 / MSE(a, b)).
    Identical inputs give inf. MSE accumulates in float64."""
    mse = float(np.mean(np.square(np.asarray(a, dtype=np.float64) -
                                  np.asarray(b, dtype=np.float64))))
    if mse == 0.0:
        return math.inf
    return 10.0 * math.log10(peak * peak / mse)


def downscale2x(X):
    """NCHW -> half height/width by 2x2 block mean (h, w must be even)."""
    n, c, h, w = X.shape
    if h % 2 or w % 2:
        raise ValueError(f"downscale2x needs even h and w, got {(h, w)}")
    return X.reshape(n, c, h // 2, 2, w // 2, 2).mean(axis=(3, 5),
                                                      dtype=np.float32)


def nearest_upscale2x(X):
    """NCHW -> double height/width by pixel repetition (the pre-net upscale
    of the superres task; the srcnn zoo model refines its output)."""
    n, c, h, w = X.shape
    out = np.empty((n, c, 2 * h, 2 * w), dtype=np.float32)
    out.reshape(n, c, h, 2, w, 2)[...] = X[:, :, :, None, :, None]
    return out


def anomaly_scores(model, X):
    """Per-sample reconstruction MSE, shape (n,) float64 — higher = more
    anomalous under a model trained on normal data only."""
    R = model.reconstruct(X)
    d = np.asarray(R, dtype=np.float64) - np.asarray(X, dtype=np.float64)
    return np.mean(np.square(d), axis=(1, 2, 3))


def roc_auc(scores, labels) -> float:
    """ROC AUC of ``scores`` against binary ``labels`` (1 = positive).

    Rank statistic form (equivalent to the Mann-Whitney U): AUC is the
    probability a random positive outscores a random negative; tied scores
    get their group's average rank, i.e. count half. Pure numpy, no sklearn.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels).ravel().astype(bool)
    if len(s) != len(y):
        raise ValueError(f"{len(s)} scores for {len(y)} labels")
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("roc_auc needs at least one positive and one negative")
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    # 1-based average rank of each tie group: last member's rank minus half
    # the group's spread.
    avg_rank = np.cumsum(counts) - (counts - 1) / 2.0
    ranks = avg_rank[inv]
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def quantize_latent(Z):
    """float32 codes -> (uint8 codes, lo, hi) with one min/max range per
    batch. The compress task's real byte budget counts the uint8 code plus
    the 8-byte float32 (lo, hi) header this returns — see the module
    docstring."""
    Z = np.asarray(Z, dtype=np.float32)
    lo = float(Z.min())
    hi = float(Z.max())
    span = (hi - lo) or 1.0                 # constant batch: all zeros
    q = np.round((Z - lo) * (255.0 / span)).astype(np.uint8)
    return q, np.float32(lo), np.float32(hi)


def dequantize_latent(q, lo, hi):
    """Inverse of :func:`quantize_latent`, back to float32 codes."""
    span = (float(hi) - float(lo)) or 1.0
    return (q.astype(np.float32) * np.float32(span / 255.0)
            + np.float32(lo))
