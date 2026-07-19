"""mantissa-autoencoder: classic convolutional autoencoders on the mantissa
C engine, built on mantissa-nn and mantissa-cnn.

>>> from mantissa_autoencoder import models, datasets, tasks
>>> X_train, _, X_test, _ = datasets.load("fashion_mnist")
>>> ae = models.denoise_ae()                    # backend="mantissa" (C engine)
>>> ae.fit(X_train, epochs=5,
...        noise=lambda x: tasks.add_gaussian_noise(x, 0.3))
>>> print(tasks.psnr(ae.reconstruct(X_test), X_test))

The shared base (Dense/Flatten layers, both backends, the image datasets)
comes from mantissa-nn; the convolution and pooling layers come from
mantissa-cnn. This package adds the two decoder layers neither has
(Upsample2D, Reshape), an MSE-trained Autoencoder, a small zoo, and the four
task helpers (denoise / compress / anomaly / superres).
"""
try:
    import mantissa_cnn  # noqa: F401  (Conv2D/MaxPool2D; pulls the mantissa-nn base)
except ImportError:
    raise ImportError(
        "mantissa-cnn is not installed — run: pip install mantissa-cnn"
    ) from None

from .layers import Reshape, Upsample2D
from .model import Autoencoder, mse_loss_grad
from . import models, tasks


def __getattr__(name):
    # PEP 562 lazy import (the family pattern): keep the datasets module (and
    # its urllib import) out of the import path until datasets are used.
    if name == "datasets":
        import importlib
        return importlib.import_module(".datasets", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__version__ = "0.1.2"
__all__ = ["Autoencoder", "Upsample2D", "Reshape", "mse_loss_grad",
           "models", "tasks", "datasets"]
