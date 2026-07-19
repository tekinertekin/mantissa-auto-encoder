"""Autoencoder: encoder/decoder layer stacks trained with MSE + SGD.

Training loop design (mantissa_cnn.Sequential's, with the classifier head
swapped for a regression loss):
- Shuffled mini-batches (seeded ``np.random.default_rng``), plain SGD.
- Loss is pixel MSE: ``loss = mean((Y - T)**2)``, ``dY = 2*(Y - T)/Y.size``
  — one vectorized numpy expression per batch each (see
  :func:`mse_loss_grad`). The loss is memory-bound (one subtract, one
  multiply-accumulate over the batch), so numpy is the right tool; the
  convolution compute on both sides of it still runs in the C engine.
- Memory: mini-batch input/target staging buffers are allocated once per
  fit (one set for full batches, one for the epoch tail) and refilled with
  ``np.take(..., out=...)``; layer scratch is allocated once per batch shape
  (see mantissa_nn.layers / mantissa_autoencoder.layers). Steady-state
  training does no per-batch allocation.

Denoising: ``fit(X, noise=...)`` applies the callable to the *input* batch
only — corrupt the input, reconstruct the clean target. That is the whole
denoising-autoencoder recipe (Vincent, Larochelle, Bengio & Manzagol, 2008,
"Extracting and Composing Robust Features with Denoising Autoencoders",
*ICML*).

Backends: the family's shared ones — ``backend="mantissa"`` (default) runs
every layer primitive in the C engine (via a per-model Session when the
engine offers one) and raises with the exact fix command when the engine is
missing; ``backend="numpy"`` is the pure-numpy reference oracle.

Data contract: X (and target, when given) is NCHW float32; the datasets
module gives [0, 1].
"""
from __future__ import annotations

import numpy as np

from mantissa_nn import _numpy_backend
from mantissa_cnn._engine import cnn_engine   # CNN feature gate: this model uses Conv2D

__all__ = ["Autoencoder", "mse_loss_grad"]


def mse_loss_grad(Y, T, dY):
    """MSE over every element: returns ``mean((Y-T)**2)`` and writes the
    gradient ``2*(Y-T)/Y.size`` into ``dY`` (caller-allocated, reused).

    ``Y.size`` counts the whole batch, so gradient magnitude is independent
    of batch and image size — one lr works across the zoo.
    """
    np.subtract(Y, T, out=dY)
    loss = float(np.vdot(dY, dY)) / Y.size
    dY *= np.float32(2.0 / Y.size)
    return loss


class Autoencoder:
    """An encoder stack and a decoder stack trained end-to-end with MSE.

    Parameters
    ----------
    encoder_layers, decoder_layers : lists of Layer
        Any base/CNN layer (Conv2D / MaxPool2D from mantissa-cnn, Dense /
        Flatten from mantissa-nn) plus
        this package's Upsample2D / Reshape. The split is what makes
        ``encode``/``decode`` meaningful; training runs the concatenation.
    seed : int
        Seeds one rng stream for weight init and epoch shuffling — two
        models with the same seed and backend train identically.
    backend : {"mantissa", "numpy"}
        "mantissa" (default) requires the C engine and raises
        ImportError/RuntimeError with the exact fix otherwise.

    Fitted attributes
    -----------------
    history_ : dict with "loss" (per-epoch mean training MSE).
    latent_shape_, out_shape_ : per-sample shapes after build().
    """

    def __init__(self, encoder_layers, decoder_layers, seed: int = 0,
                 backend: str = "mantissa"):
        if backend == "mantissa":
            tk = cnn_engine()                 # raises with the exact fix
            # mantissa >= 0.2.2: a per-model Session memoizes each buffer's
            # ctypes pointer by identity — our buffers are allocated once
            # and refilled in place, so pointer derivation becomes a dict
            # hit. Older engines just take the plain methods.
            self._backend = tk.session() if hasattr(tk, "session") else tk
        elif backend == "numpy":
            self._backend = _numpy_backend
        else:
            raise ValueError(f"backend must be 'mantissa' or 'numpy', got {backend!r}")
        self.backend = backend
        self.encoder = list(encoder_layers)
        self.decoder = list(decoder_layers)
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self._built = False

    @property
    def layers(self):
        """Encoder then decoder — the stack training actually runs."""
        return self.encoder + self.decoder

    # -- construction ---------------------------------------------------------

    def build(self, in_shape):
        """Initialize parameters for input shape (c, h, w). Called by fit()
        automatically; call it yourself to inspect summary() before training."""
        shape = tuple(int(d) for d in in_shape)
        self.in_shape_ = shape
        for layer in self.encoder:
            shape = layer.build(shape, self._rng)
        self.latent_shape_ = shape
        for layer in self.decoder:
            shape = layer.build(shape, self._rng)
        if len(shape) != 3:
            raise ValueError(f"decoder output must be an image (c, h, w), "
                             f"got shape {shape}")
        self.out_shape_ = shape
        self._built = True
        return self

    def summary(self) -> str:
        """Per-layer output shapes and parameter counts (build() first),
        with the bottleneck marked between encoder and decoder."""
        if not self._built:
            raise RuntimeError("summary() needs parameters — call build(in_shape) or fit() first")
        rows = [(type(l).__name__, str(l.out_shape), l.param_count())
                for l in self.layers]
        total = sum(r[2] for r in rows)
        w = max(max(len(r[0]) for r in rows), len("-- bottleneck --"))
        lines = [f"{'layer':<{w}}  {'out shape':<16}  params",
                 "-" * (w + 26)]
        for i, (name, shape, p) in enumerate(rows):
            if i == len(self.encoder):
                lines.append(f"{'-- bottleneck --':<{w}}  {str(self.latent_shape_):<16}")
            lines.append(f"{name:<{w}}  {shape:<16}  {p:,}")
        lines.append(f"total params: {total:,}")
        return "\n".join(lines)

    # -- training -------------------------------------------------------------

    def fit(self, X, target=None, epochs: int = 10, batch_size: int = 32,
            lr: float = 0.01, noise=None, verbose: bool = False):
        """Train with MSE between the reconstruction and ``target``.

        target defaults to X (plain / denoising autoencoder); pass a
        different array for input->output tasks (e.g. super-resolution:
        X = upscaled low-res, target = the original).
        noise, if given, is a callable applied to the INPUT mini-batch only
        (the target stays clean) — e.g.
        ``lambda x: tasks.add_gaussian_noise(x, 0.3)``.
        """
        X = self._check(X, name="X")
        n = len(X)
        if not self._built:
            self.build(X.shape[1:])
        if tuple(X.shape[1:]) != self.in_shape_:
            raise ValueError(f"X has sample shape {tuple(X.shape[1:])}, "
                             f"model was built for {self.in_shape_}")
        T = X if target is None else self._check(target, expect=self.out_shape_,
                                                 name="target")
        if target is None and tuple(X.shape[1:]) != self.out_shape_:
            raise ValueError(f"target defaults to X, but the decoder emits "
                             f"{self.out_shape_} for input {self.in_shape_} — "
                             f"pass an explicit target")
        if len(T) != n:
            raise ValueError(f"X has {n} samples but target has {len(T)}")

        backend = self._backend
        bs = min(int(batch_size), n)
        layers = self.layers
        self.history_ = {"loss": []}

        # Per-fit staging buffers, refilled in place every batch: one set for
        # full batches, one for the epoch tail. When the target is X itself
        # and the input is not corrupted, the target buffer aliases the input
        # buffer — same values, zero extra staging memory.
        tail = n % bs
        own_target = target is not None or noise is not None

        def alloc(m):
            xb = np.empty((m,) + self.in_shape_, dtype=np.float32)
            tb = (np.empty((m,) + self.out_shape_, dtype=np.float32)
                  if own_target else xb)
            return xb, tb, np.empty((m,) + self.out_shape_, dtype=np.float32)

        Xb, Tb, dY = alloc(bs)
        Xt, Tt, dYt = alloc(tail) if tail else (None, None, None)

        for epoch in range(int(epochs)):
            order = self._rng.permutation(n)
            loss_sum = 0.0
            for start in range(0, n, bs):
                idx = order[start:start + bs]
                nb = len(idx)
                bx, bt, bd = (Xb, Tb, dY) if nb == bs else (Xt, Tt, dYt)
                np.take(X, idx, axis=0, out=bx)
                if bt is not bx:
                    np.take(T, idx, axis=0, out=bt)

                out = noise(bx) if noise is not None else bx
                for layer in layers:
                    out = layer.forward(out, backend)

                loss_sum += mse_loss_grad(out, bt, bd) * nb

                grad = bd
                for i in range(len(layers) - 1, -1, -1):
                    grad = layers[i].backward(grad, backend, need_dx=i > 0)
                for layer in layers:           # after ALL grads: dX of layer i
                    layer.step(backend, lr)    # depends on its pre-step params

            self.history_["loss"].append(loss_sum / n)
            if verbose:
                print(f"epoch {epoch + 1}/{epochs}  "
                      f"loss {self.history_['loss'][-1]:.6f}")
        return self

    # -- inference --------------------------------------------------------------

    def encode(self, X):
        """Latent codes, shape (n,) + latent_shape_."""
        self._require_built()
        X = self._check(X, expect=self.in_shape_, name="X")
        return self._run(self.encoder, X, self.latent_shape_)

    def decode(self, Z):
        """Reconstructions from latent codes, shape (n,) + out_shape_."""
        self._require_built()
        Z = self._check(Z, expect=self.latent_shape_, name="Z")
        return self._run(self.decoder, Z, self.out_shape_)

    def reconstruct(self, X):
        """encode + decode in one chunked pass, shape (n,) + out_shape_."""
        self._require_built()
        X = self._check(X, expect=self.in_shape_, name="X")
        return self._run(self.layers, X, self.out_shape_)

    # -- internals ---------------------------------------------------------------

    def _run(self, layers, X, out_shape, chunk: int = 256):
        out = np.empty((len(X),) + out_shape, dtype=np.float32)
        for s in range(0, len(X), chunk):
            h = X[s:s + chunk]                 # contiguous slice view, no copy
            for layer in layers:
                h = layer.forward(h, self._backend)
            out[s:s + chunk] = h               # copy out: layer scratch is reused
        return out

    def _require_built(self):
        if not self._built:
            raise RuntimeError("model has no parameters — call build(in_shape) or fit() first")

    def _check(self, A, expect=None, name="X"):
        A = np.ascontiguousarray(A, dtype=np.float32)
        want_ndim = 1 + (len(expect) if expect is not None else 3)
        if A.ndim != want_ndim:
            raise ValueError(f"{name} must have shape (n,) + {expect or '(c, h, w)'} "
                             f"float32, got ndim={A.ndim}")
        if expect is not None and tuple(A.shape[1:]) != tuple(expect):
            raise ValueError(f"{name} has sample shape {tuple(A.shape[1:])}, "
                             f"expected {tuple(expect)}")
        return A
