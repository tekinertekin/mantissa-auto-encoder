"""Model zoo: three classic autoencoder recipes at small-image scale,
honestly named.

Each function returns a built :class:`~mantissa_autoencoder.model.Autoencoder`
(parameters initialized, so ``summary()`` works immediately). All decoders
upsample with nearest-neighbor resize + Conv2D rather than transposed
convolution — the artifact-avoiding choice (Odena, Dumoulin & Olah, 2016,
"Deconvolution and Checkerboard Artifacts", *Distill*); see
:mod:`mantissa_autoencoder.layers`. Reconstruction heads use the identity
activation: pixel targets live in [0, 1] and MSE wants unsquashed output.
"""
from __future__ import annotations

from mantissa_cnn.layers import Conv2D, Dense, Flatten, MaxPool2D

from .layers import Reshape, Upsample2D
from .model import Autoencoder

__all__ = ["denoise_ae", "bottleneck_ae", "srcnn"]


def _check_divisible(in_shape, by: int, name: str):
    c, h, w = in_shape
    if h % by or w % by:
        raise ValueError(f"{name}: input h and w must be divisible by {by} "
                         f"(two 2x pool/upsample stages), got {in_shape}")
    return c, h, w


def denoise_ae(in_shape=(1, 28, 28), seed: int = 0,
               backend: str = "mantissa") -> Autoencoder:
    """Convolutional denoising autoencoder — Vincent, Larochelle, Bengio &
    Manzagol (2008), "Extracting and Composing Robust Features with
    Denoising Autoencoders", *ICML*. Their recipe (corrupt the input,
    reconstruct the clean target) with the modern conv encoder/decoder
    rather than the paper's fully-connected stacks — flagged deviation.

    Encoder Conv 16@3x3 -> pool -> Conv 32@3x3 -> pool (28x28 -> 32@7x7,
    a spatial latent); decoder mirrors it with upsample + conv. Train with
    ``fit(X, noise=lambda x: tasks.add_gaussian_noise(x, 0.3))`` — the
    noise goes on the input only.
    """
    c, _, _ = _check_divisible(in_shape, 4, "denoise_ae")
    return Autoencoder(
        [Conv2D(16, 3, pad=1, act="relu"),
         MaxPool2D(2),
         Conv2D(32, 3, pad=1, act="relu"),
         MaxPool2D(2)],
        [Upsample2D(2),
         Conv2D(16, 3, pad=1, act="relu"),
         Upsample2D(2),
         Conv2D(c, 3, pad=1, act="identity")],
        seed=seed, backend=backend).build(in_shape)


def bottleneck_ae(in_shape=(1, 28, 28), latent: int = 32, seed: int = 0,
                  backend: str = "mantissa") -> Autoencoder:
    """Dense-bottleneck autoencoder — Hinton & Salakhutdinov (2006),
    "Reducing the Dimensionality of Data with Neural Networks", *Science*
    313(5786): squeeze the image through a code far smaller than the input
    and let reconstruction error force the code to keep what matters. Their
    30-float MNIST code is the direct ancestor of the default ``latent=32``
    here; the conv encoder/decoder around it is the modern scale-appropriate
    body, not the paper's RBM-pretrained dense stack — flagged deviation.

    Encoder Conv 16 -> pool -> Conv 32 -> pool -> Flatten -> Dense(latent)
    (identity: an unbounded linear code, as in the paper's code layer);
    decoder Dense -> Reshape -> upsample+conv mirror.
    """
    c, h, w = _check_divisible(in_shape, 4, "bottleneck_ae")
    h4, w4 = h // 4, w // 4
    return Autoencoder(
        [Conv2D(16, 3, pad=1, act="relu"),
         MaxPool2D(2),
         Conv2D(32, 3, pad=1, act="relu"),
         MaxPool2D(2),
         Flatten(),
         Dense(int(latent), act="identity")],
        [Dense(32 * h4 * w4, act="relu"),
         Reshape((32, h4, w4)),
         Upsample2D(2),
         Conv2D(16, 3, pad=1, act="relu"),
         Upsample2D(2),
         Conv2D(c, 3, pad=1, act="identity")],
        seed=seed, backend=backend).build(in_shape)


def srcnn(in_shape=(1, 28, 28), scale: int = 2, seed: int = 0,
          backend: str = "mantissa") -> Autoencoder:
    """SRCNN-style super-resolution net — Dong, Loy, He & Tang (2014/2016),
    "Image Super-Resolution Using Deep Convolutional Networks", *TPAMI*
    38(2). As in the paper, the low-res image is upscaled to the target
    size OUTSIDE the net (``tasks.nearest_upscale2x`` for ``scale=2``;
    the paper uses bicubic — flagged deviation) and the net only refines:
    Conv 32@5x5 (patch extraction) -> Conv 16@3x3 (non-linear mapping) ->
    Conv c@3x3 identity (reconstruction), all size-preserving.

    ``scale`` records the intended factor for the task recipe; the net
    itself is fully convolutional and size-preserving, so it never sees it.
    There is no bottleneck — the encoder/decoder split here is the paper's
    extraction/reconstruction split, and ``encode`` returns feature maps,
    not a compressed code.
    """
    c = int(in_shape[0])
    if int(scale) != 2:
        raise ValueError(f"srcnn: only scale=2 has task helpers "
                         f"(tasks.downscale2x / nearest_upscale2x), got {scale}")
    return Autoencoder(
        [Conv2D(32, 5, pad=2, act="relu")],
        [Conv2D(16, 3, pad=1, act="relu"),
         Conv2D(c, 3, pad=1, act="identity")],
        seed=seed, backend=backend).build(in_shape)
