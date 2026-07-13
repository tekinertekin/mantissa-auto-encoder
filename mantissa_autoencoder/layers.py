"""The two decoder layers mantissa-cnn lacks: Upsample2D and Reshape.

Same contract as :mod:`mantissa_cnn.layers` (whose ``Layer`` base these
subclass): ``build(in_shape, rng) -> out_shape``, ``forward``/``backward``
taking a backend object, scratch allocated once per batch shape and reused
across batches and epochs, ``step`` a no-op (both layers are parameter-free).

Both layers are pure data movement, so they ignore the backend argument.
Nearest-neighbor upsampling does zero arithmetic per output element — it is
memory-bound, and numpy's strided broadcast assignment already moves the
bytes at memcpy speed, so it stays in numpy on both backends. An engine
primitive is future work, on the record; it would buy nothing today.

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

from mantissa_cnn.layers import Layer

__all__ = ["Upsample2D", "Reshape"]


class Upsample2D(Layer):
    """Nearest-neighbor upsampling: every pixel becomes a scale x scale block.

    Forward is one broadcast assignment into the preallocated output viewed
    as (n, c, h, scale, w, scale); backward is the exact adjoint — each input
    pixel fed a scale x scale output block, so its gradient is that block's
    sum (``np.sum`` over the two block axes, written into reused scratch).
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
        s["Y"].reshape(n, c, h, k, w, k)[...] = X[:, :, :, None, :, None]
        return s["Y"]

    def backward(self, dY, backend, need_dx: bool = True):
        n = dY.shape[0]
        s = self._bufs(n)
        c, h, w = self.in_shape
        k = self.scale
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
