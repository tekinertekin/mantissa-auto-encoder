"""The three zoo architectures re-expressed layer-for-layer in each
framework, plus the task data prep — everything bench/speed.py needs that
is not measurement.

Estimator surface (uniform across contenders; construction is untimed,
``fit`` is the timed region):

- ``factory(arch, sigma=None)`` -> fresh estimator, weights initialized with
  the same init family as ours (He normal before relu, Glorot uniform on
  identity-activation layers, zero biases — the frameworks' own defaults
  differ, and matching the init keeps the comparison about the frameworks,
  not the initializer; exactly the cnn benchmark's policy).
- ``fit(X, T=None)`` trains under the fixed protocol (bench.protocol: plain
  SGD, MSE, seeded shuffles); ``T=None`` means reconstruct the input.
  ``sigma`` (the denoise task) corrupts each mini-batch anew INSIDE fit with
  the framework's native Gaussian noise + clip to [0, 1] — the corruption is
  part of the recipe, so it is part of the timed work.
- ``reconstruct(X)`` -> NCHW float32 numpy, whatever the native layout.
- ``encode(X)`` / ``decode(Z)`` for the compress task's quantized round trip
  (``Z`` is always numpy; codes are quantized outside the frameworks so the
  metric arithmetic is shared).

``X``/``T`` arrive in the contender's native form via its ``prep`` function
(NCHW numpy for ours, NCHW torch tensors, NHWC numpy for tensorflow) —
conversion happens once, outside the timed region.

Structural parity: ``python -m bench.contenders`` asserts and prints that
parameter counts match across frameworks for every architecture (the numpy
backend shares ours' layer objects, so its parity is by construction).

Layout note: keras is NHWC, so the bottleneck's Dense layers connect to a
(7, 7, 32) tensor instead of (32, 7, 7) — a permutation of the same weight
matrix, identical parameter count and function class.
"""
from __future__ import annotations

import os

# Keep TensorFlow's C++ banner out of benchmark output (set before any TF
# import anywhere in the process).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

from mantissa_autoencoder import datasets, models, tasks

from .protocol import BATCH_SIZE, EPOCHS, LR, N_TEST, N_TRAIN, SEED

__all__ = ["ARCHITECTURES", "CONTENDERS", "OursAE", "TorchAE", "KerasAE",
           "contenders", "task_data", "check_parity"]

ARCHITECTURES = ("denoise_ae", "bottleneck_ae", "srcnn")

# JSON keys, in protocol.CONTENDERS order (protocol pins the display names).
CONTENDERS = ("ours", "vanilla_numpy", "torch", "tensorflow")

# The compress recipe's latent must be the zoo default — the frameworks
# below hard-code the mirrored shapes around it.
assert tasks.TASKS["compress"]["latent"] == 32


# --- ours (both backends) ----------------------------------------------------

class OursAE:
    """mantissa_autoencoder.models.<arch> on the chosen backend."""

    def __init__(self, arch, backend, sigma=None):
        self._net = getattr(models, arch)(seed=SEED, backend=backend)
        self._noise = (None if sigma is None
                       else lambda x: tasks.add_gaussian_noise(x, sigma))

    def fit(self, X, T=None):
        self._net.fit(X, target=T, epochs=EPOCHS, batch_size=BATCH_SIZE,
                      lr=LR, noise=self._noise)
        self.final_loss_ = float(self._net.history_["loss"][-1])
        return self

    def reconstruct(self, X):
        return self._net.reconstruct(X)

    def encode(self, X):
        return self._net.encode(X)

    def decode(self, Z):
        return self._net.decode(Z)

    def param_count(self):
        return sum(l.param_count() for l in self._net.layers)


# --- torch ---------------------------------------------------------------------

def _torch_arch(arch):
    """Encoder/decoder nn.Sequentials mirroring models.<arch> exactly
    (same kernel sizes, pads, channel widths; nearest-neighbor upsampling)."""
    import torch.nn as nn

    def up():
        return nn.Upsample(scale_factor=2, mode="nearest")

    if arch == "denoise_ae":
        enc = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        dec = nn.Sequential(up(), nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(),
                            up(), nn.Conv2d(16, 1, 3, padding=1))
    elif arch == "bottleneck_ae":
        enc = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                            nn.Flatten(), nn.Linear(32 * 7 * 7, 32))
        dec = nn.Sequential(nn.Linear(32, 32 * 7 * 7), nn.ReLU(),
                            nn.Unflatten(1, (32, 7, 7)),
                            up(), nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(),
                            up(), nn.Conv2d(16, 1, 3, padding=1))
    elif arch == "srcnn":
        enc = nn.Sequential(nn.Conv2d(1, 32, 5, padding=2), nn.ReLU())
        dec = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(),
                            nn.Conv2d(16, 1, 3, padding=1))
    else:
        raise ValueError(f"unknown arch {arch!r}")
    return enc, dec


class TorchAE:
    """torch.nn.Sequential(enc, dec), eager mode, explicit seeded loop with
    the same shuffle-stream construction as ours (np rng permutation)."""

    def __init__(self, arch, sigma=None):
        import torch
        import torch.nn as nn
        torch.manual_seed(SEED)
        self._enc, self._dec = _torch_arch(arch)
        self._m = nn.Sequential(self._enc, self._dec)
        g = torch.Generator().manual_seed(SEED)
        for seq in (self._enc, self._dec):
            mods = list(seq)
            for i, m in enumerate(mods):
                if not isinstance(m, (nn.Conv2d, nn.Linear)):
                    continue
                nn.init.zeros_(m.bias)
                if i + 1 < len(mods) and isinstance(mods[i + 1], nn.ReLU):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu",
                                            generator=g)
                else:                     # identity-activation layer
                    nn.init.xavier_uniform_(m.weight, generator=g)
        self._sigma = sigma
        self._rng = np.random.default_rng(SEED)

    def fit(self, X, T=None):
        import torch
        T = X if T is None else T
        m = self._m
        m.train()
        opt = torch.optim.SGD(m.parameters(), lr=LR, momentum=0.0)
        loss_fn = torch.nn.MSELoss()
        n = len(X)
        for _ in range(EPOCHS):
            order = self._rng.permutation(n)
            loss_sum = 0.0
            for s in range(0, n, BATCH_SIZE):
                idx = torch.from_numpy(order[s:s + BATCH_SIZE])
                xb, tb = X[idx], T[idx]
                if self._sigma is not None:
                    xb = torch.clamp(xb + torch.randn_like(xb) * self._sigma,
                                     0.0, 1.0)
                opt.zero_grad()
                loss = loss_fn(m(xb), tb)
                loss.backward()
                opt.step()
                loss_sum += loss.item() * len(idx)
            self.final_loss_ = loss_sum / n
        return self

    def reconstruct(self, X):
        import torch
        self._m.eval()
        with torch.no_grad():
            return self._m(X).numpy()

    def encode(self, X):
        import torch
        self._m.eval()
        with torch.no_grad():
            return self._enc(X).numpy()

    def decode(self, Z):
        import torch
        self._m.eval()
        with torch.no_grad():
            z = torch.from_numpy(np.ascontiguousarray(Z, dtype=np.float32))
            return self._dec(z).numpy()

    def param_count(self):
        return sum(p.numel() for p in self._m.parameters())


# --- tensorflow / keras --------------------------------------------------------

def _keras_arch(arch):
    """Encoder/decoder keras.Sequentials mirroring models.<arch> (NHWC;
    'same' padding == pad 1 for 3x3 / pad 2 for 5x5 at stride 1)."""
    import keras
    L = keras.layers
    he = keras.initializers.HeNormal(seed=SEED)
    glorot = keras.initializers.GlorotUniform(seed=SEED)

    def conv(c, k, act, init):
        return L.Conv2D(c, k, padding="same", activation=act,
                        kernel_initializer=init)

    if arch == "denoise_ae":
        enc = keras.Sequential([keras.Input((28, 28, 1)),
                                conv(16, 3, "relu", he), L.MaxPool2D(2),
                                conv(32, 3, "relu", he), L.MaxPool2D(2)])
        dec = keras.Sequential([keras.Input((7, 7, 32)),
                                L.UpSampling2D(2), conv(16, 3, "relu", he),
                                L.UpSampling2D(2), conv(1, 3, None, glorot)])
    elif arch == "bottleneck_ae":
        enc = keras.Sequential([keras.Input((28, 28, 1)),
                                conv(16, 3, "relu", he), L.MaxPool2D(2),
                                conv(32, 3, "relu", he), L.MaxPool2D(2),
                                L.Flatten(),
                                L.Dense(32, kernel_initializer=glorot)])
        dec = keras.Sequential([keras.Input((32,)),
                                L.Dense(32 * 7 * 7, activation="relu",
                                        kernel_initializer=he),
                                L.Reshape((7, 7, 32)),
                                L.UpSampling2D(2), conv(16, 3, "relu", he),
                                L.UpSampling2D(2), conv(1, 3, None, glorot)])
    elif arch == "srcnn":
        enc = keras.Sequential([keras.Input((28, 28, 1)),
                                L.Conv2D(32, 5, padding="same", activation="relu",
                                         kernel_initializer=he)])
        dec = keras.Sequential([keras.Input((28, 28, 32)),
                                conv(16, 3, "relu", he), conv(1, 3, None, glorot)])
    else:
        raise ValueError(f"unknown arch {arch!r}")
    return enc, dec


class KerasAE:
    """tf.keras, built + compiled in the constructor — outside the timed
    region, like any one-time setup. The denoise task's per-batch corruption
    uses a tf.data map (TF's native way to corrupt inside fit while keeping
    the clean target); UpSampling2D's default interpolation is nearest."""

    def __init__(self, arch, sigma=None):
        import keras
        keras.utils.set_random_seed(SEED)   # init + shuffle + noise streams
        self._enc, self._dec = _keras_arch(arch)
        in_shape = self._enc.input_shape[1:]
        m = keras.Sequential([keras.Input(in_shape), self._enc, self._dec])
        m.compile(optimizer=keras.optimizers.SGD(learning_rate=LR, momentum=0.0),
                  loss=keras.losses.MeanSquaredError())
        self._m = m
        self._sigma = sigma

    def fit(self, X, T=None):
        T = X if T is None else T
        if self._sigma is None:
            h = self._m.fit(X, T, epochs=EPOCHS, batch_size=BATCH_SIZE,
                            shuffle=True, verbose=0)
        else:
            import tensorflow as tf
            sigma = self._sigma

            def corrupt(x, t):
                noisy = x + tf.random.normal(tf.shape(x), stddev=sigma)
                return tf.clip_by_value(noisy, 0.0, 1.0), t

            ds = (tf.data.Dataset.from_tensor_slices((X, T))
                  .shuffle(len(X), seed=SEED, reshuffle_each_iteration=True)
                  .batch(BATCH_SIZE).map(corrupt))
            h = self._m.fit(ds, epochs=EPOCHS, verbose=0)
        self.final_loss_ = float(h.history["loss"][-1])
        return self

    def reconstruct(self, X):
        out = self._m.predict(X, verbose=0)
        return np.ascontiguousarray(out.transpose(0, 3, 1, 2))

    def encode(self, X):
        Z = self._enc.predict(X, verbose=0)
        if Z.ndim == 4:                       # spatial latent -> NCHW
            Z = np.ascontiguousarray(Z.transpose(0, 3, 1, 2))
        return Z

    def decode(self, Z):
        Z = np.ascontiguousarray(Z, dtype=np.float32)
        if Z.ndim == 4:                       # NCHW codes -> NHWC
            Z = np.ascontiguousarray(Z.transpose(0, 2, 3, 1))
        out = self._dec.predict(Z, verbose=0)
        return np.ascontiguousarray(out.transpose(0, 3, 1, 2))

    def param_count(self):
        return self._m.count_params()


# --- registry -------------------------------------------------------------------
# (name, factory, prep). prep maps NCHW float32 numpy into the contender's
# native form ONCE, outside the timed region, so fit() measures training only.

def _prep_ours(A):
    return A


def _prep_torch(A):
    import torch
    return torch.from_numpy(A)


def _prep_tf(A):
    return np.ascontiguousarray(A.transpose(0, 2, 3, 1))   # NHWC


def contenders():
    reg = [
        ("ours", lambda a, sigma=None: OursAE(a, "mantissa", sigma), _prep_ours),
        ("vanilla_numpy", lambda a, sigma=None: OursAE(a, "numpy", sigma), _prep_ours),
        ("torch", TorchAE, _prep_torch),
        ("tensorflow", KerasAE, _prep_tf),
    ]
    assert tuple(n for n, *_ in reg) == CONTENDERS
    return reg


# --- task data -------------------------------------------------------------------

def task_data(task):
    """Numpy NCHW arrays for one task: the fit pair (X, T; T None means
    reconstruct the input), sigma for the denoise corruption, and whatever
    the task metric needs on the held-out test subset. Deterministic — the
    subset is seeded and the denoise test corruption is drawn once with the
    protocol seed (fresh per-batch noise stays inside fit)."""
    spec = tasks.TASKS[task]
    Xtr, ytr, Xte, yte = datasets.subset(spec["dataset"], N_TRAIN, N_TEST, SEED)
    if task == "denoise":
        return dict(arch=spec["model"], X=Xtr, T=None, sigma=spec["sigma"],
                    X_test=Xte,
                    X_test_noisy=tasks.add_gaussian_noise(Xte, spec["sigma"],
                                                          seed=SEED))
    if task == "compress":
        return dict(arch=spec["model"], X=Xtr, T=None, sigma=None, X_test=Xte)
    if task == "anomaly":
        keep = ytr != spec["held_out_digit"]
        return dict(arch=spec["model"], X=Xtr[keep], T=None, sigma=None,
                    X_test=Xte,
                    labels=(yte == spec["held_out_digit"]).astype(np.int32))
    if task == "superres":
        low_up = tasks.nearest_upscale2x(tasks.downscale2x(Xtr))
        low_up_te = tasks.nearest_upscale2x(tasks.downscale2x(Xte))
        return dict(arch=spec["model"], X=low_up, T=Xtr, sigma=None,
                    X_test=low_up_te, X_test_target=Xte)
    raise ValueError(f"unknown task {task!r}")


# --- structural parity ------------------------------------------------------------

def check_parity(verbose: bool = True):
    """Assert parameter counts match across frameworks per architecture.
    vanilla_numpy shares ours' layer objects, so it is covered by 'ours'."""
    rows = {}
    for arch in ARCHITECTURES:
        counts = {"ours": OursAE(arch, "mantissa").param_count(),
                  "torch": TorchAE(arch).param_count(),
                  "tensorflow": KerasAE(arch).param_count()}
        assert len(set(counts.values())) == 1, \
            f"parameter count mismatch for {arch}: {counts}"
        rows[arch] = counts
        if verbose:
            print(f"{arch:14s} " +
                  "  ".join(f"{k}={v:,}" for k, v in counts.items()) + "  OK")
    return rows


if __name__ == "__main__":
    check_parity()
