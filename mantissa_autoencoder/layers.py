"""The two decoder layers mantissa-cnn lacks: Upsample2D and Reshape.

Same contract as :mod:`mantissa_nn.layers` (whose ``Layer`` base these
subclass): ``build(in_shape, rng) -> out_shape``, ``forward``/``backward``
taking a backend object, scratch allocated once per batch shape and reused
across batches and epochs, ``step`` a no-op (both layers are parameter-free).

Reshape is a pure view either way and always stays in numpy. Upsample2D is
memory-bound (nearest-neighbor does zero arithmetic per output element), and
the earlier note here claimed numpy's strided broadcast already moved the
bytes at memcpy speed so a primitive "would buy nothing" — that was measured
false. At these decoder shapes the numpy broadcast-assign runs at 4 vs 74
GB/s forward, and the backward's fused ``np.sum`` over two interleaved
length-k axes degenerates the reduction iterator (~9x slower than the block
sum needs to be). mantissa v0.2.3 adds ``tk_upsample2d_nearest_f32`` /
``_backward_f32`` (exposed as ``Session.upsample2d`` / ``upsample2d_backward``);
measured at this package's decoder shapes: forward 200->41 and 416->37 us,
backward 739->21 and 1415->30 us (up to 47x). Upsample2D uses the primitive
when the backend offers it (``hasattr(backend, "upsample2d")`` — the mantissa
Session does, the numpy oracle backend does not) and otherwise keeps the numpy
expressions verbatim, so ``backend="numpy"`` stays pure and older engines fall
back automatically.

Why these two layers exist at all: the decoders here upsample with
nearest-neighbor resize followed by a plain ``Conv2D`` instead of using
transposed convolution. Transposed convolutions overlap their kernel
footprints unevenly and imprint checkerboard artifacts on the output;
resize-then-convolve avoids the artifact by construction (Odena, Dumoulin &
Olah, 2016, "Deconvolution and Checkerboard Artifacts", *Distill*) — a
deliberate choice, not a workaround.
"""
from __future__ import annotations

import numpy as np

from mantissa_nn.layers import Layer

__all__ = ["Upsample2D", "Reshape"]


class Upsample2D(Layer):
    """Nearest-neighbor upsampling: every pixel becomes a scale x scale block.

    Forward maps each input pixel to a scale x scale output block; backward is
    the exact adjoint — the gradient of an input pixel is the sum of its
    block. Both write into reused scratch. When the backend exposes the
    v0.2.3 primitive (``upsample2d`` / ``upsample2d_backward``) the work
    crosses into C; otherwise it runs the numpy forms (broadcast assign into
    the output viewed as (n, c, h, scale, w, scale); ``np.sum`` over the two
    block axes), which stay the oracle and older-engine fallback.
    """

    def __init__(self, scale: int = 2):
        super().__init__()
        self.scale = int(scale)
        if self.scale < 1:
            raise ValueError(f"Upsample2D: scale must be >= 1, got {scale}")

    def build(self, in_shape, rng):
        c, h, w = self.in_shape = tuple(in_shape)
        self.out_shape = (c, h * self.scale, w * self.scale)
        return self.out_shape

    def _alloc(self, n):
        return {"Y": np.empty((n,) + self.out_shape, dtype=np.float32),
                "dX": np.empty((n,) + self.in_shape, dtype=np.float32)}

    def forward(self, X, backend):
        n = X.shape[0]
        s = self._bufs(n)
        c, h, w = self.in_shape
        k = self.scale
        if hasattr(backend, "upsample2d"):
            backend.upsample2d(X, s["Y"], n, c, h, w, k)
        else:
            s["Y"].reshape(n, c, h, k, w, k)[...] = X[:, :, :, None, :, None]
        return s["Y"]

    def backward(self, dY, backend, need_dx: bool = True):
        n = dY.shape[0]
        s = self._bufs(n)
        c, h, w = self.in_shape
        k = self.scale
        if hasattr(backend, "upsample2d_backward"):
            backend.upsample2d_backward(dY, s["dX"], n, c, h, w, k)
        else:
            np.sum(dY.reshape(n, c, h, k, w, k), axis=(3, 5), out=s["dX"])
        return s["dX"]


class Reshape(Layer):
    """Fixed reshape of each sample, e.g. ``(1568,) -> (32, 7, 7)`` between a
    Dense bottleneck and the convolutional decoder (Flatten's inverse, with
    the target spelled out). A view both ways on contiguous buffers — no
    copies, no scratch, no parameters."""

    def __init__(self, shape):
        super().__init__()
        self.shape = tuple(int(d) for d in shape)

    def build(self, in_shape, rng):
        self.in_shape = tuple(in_shape)
        if int(np.prod(self.in_shape)) != int(np.prod(self.shape)):
            raise ValueError(
                f"Reshape: cannot reshape {self.in_shape} "
                f"({int(np.prod(self.in_shape))} values) to {self.shape} "
                f"({int(np.prod(self.shape))} values)")
        self.out_shape = self.shape
        return self.out_shape

    def forward(self, X, backend):
        return X.reshape((X.shape[0],) + self.shape)

    def backward(self, dY, backend, need_dx: bool = True):
        return dY.reshape((dY.shape[0],) + self.in_shape)
