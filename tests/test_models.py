"""Zoo architectures: latent/output shapes, size-preserving reconstruction,
input validation. Numpy backend — no engine needed."""
import numpy as np
import pytest

from mantissa_autoencoder.models import bottleneck_ae, denoise_ae, srcnn


def test_denoise_ae_shapes():
    ae = denoise_ae(backend="numpy")
    assert ae.latent_shape_ == (32, 7, 7)
    assert ae.out_shape_ == (1, 28, 28)


def test_bottleneck_ae_latent_is_flat_code():
    ae = bottleneck_ae(latent=32, backend="numpy")
    assert ae.latent_shape_ == (32,)
    assert ae.out_shape_ == (1, 28, 28)
    ae = bottleneck_ae(in_shape=(3, 32, 32), latent=16, backend="numpy")
    assert ae.latent_shape_ == (16,)
    assert ae.out_shape_ == (3, 32, 32)


def test_srcnn_is_size_preserving():
    ae = srcnn(backend="numpy")
    assert ae.out_shape_ == (1, 28, 28)
    with pytest.raises(ValueError, match="scale"):
        srcnn(scale=3, backend="numpy")


@pytest.mark.parametrize("factory", [denoise_ae, bottleneck_ae])
def test_pooling_models_need_divisible_input(factory):
    with pytest.raises(ValueError, match="divisible"):
        factory(in_shape=(1, 30, 30), backend="numpy")


@pytest.mark.parametrize("factory", [denoise_ae, bottleneck_ae, srcnn])
def test_zoo_reconstruction_pass(factory):
    ae = factory(backend="numpy")
    X = np.random.default_rng(6).random((2, 1, 28, 28), dtype=np.float32)
    R = ae.reconstruct(X)
    assert R.shape == X.shape and R.dtype == np.float32
    Z = ae.encode(X)
    assert np.allclose(ae.decode(Z), R, atol=1e-6)
