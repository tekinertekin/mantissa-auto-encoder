# Changelog

Release notes for **mantissa-autoencoder**, newest first — in the family's
style (see mantissa/RELEASES.md): what shipped, what was measured, what was
deliberately not done.

---

## v0.1.1 — 2026-07-14

The adoption release: `Upsample2D` moves onto the engine's new
nearest-neighbor primitive (mantissa v0.2.3, `Session.upsample2d` /
`upsample2d_backward`), feature-detected per call — the numpy expressions
stay verbatim as the oracle path and the fallback for `backend="numpy"`
and older engines. Nothing breaks without v0.2.3; it is just slower.

**Why**: this package's own docstring claimed the numpy upsample ran "at
memcpy speed" and a primitive "would buy nothing". Measured, both claims
were false — the broadcast-assign forward ran at 4 vs 74 GB/s and the
fused `np.sum(axis=(3,5))` backward was ~9× under a plain slice-add
(interleaved length-k reduction axes degenerate numpy's iterator). The
docstrings now carry the measured numbers instead.

**Measured** (M4, protocol shapes):
- Primitive vs numpy per stage: fwd 200→41 / 416→37 µs, bwd 739→21 /
  1415→30 µs (up to 47×).
- denoise_ae sanity fit (fashion_mnist 2000-subset, median of 5):
  **2231 → 1549 ms (−31%)**.
- Official benchmark: TF's fit lead on the decoder-heavy tasks narrowed
  from ~1.5× to **1.14–1.17×**; ours now leads batch reconstruct on all
  four tasks (35–45 ms). Deterministic task metrics moved zero digits —
  all decoder scales are 2, where the C block-sum backward is bit-exact.

**Deliberately not done**: `sgd_update_list` adoption — the step contract
is per-layer and the layers live in mantissa_cnn; harvesting another
package's parameter tensors to bypass its `step()` is not this package's
call. Recorded for a coordinated change.

Requires `mantissa-nn >= 0.2.3` for the fast path (declared as a soft
requirement: the package imports and runs against older engines via the
numpy fallback).

---

## v0.1.0 — 2026-07-13

Initial release: convolutional autoencoders (`Autoencoder` with MSE +
mini-batch SGD, `Upsample2D`/`Reshape` decoder layers) on top of
mantissa-cnn's layers, backends and datasets; a cited three-model zoo
(denoise_ae — Vincent et al. 2008; bottleneck_ae — Hinton & Salakhutdinov
2006; srcnn — Dong et al. 2014) and four benchmark task recipes
(denoise, compress with honest 19.6× byte accounting, anomaly via
reconstruction error, super-resolution) measured against torch and
TensorFlow re-expressions of the same architectures.
