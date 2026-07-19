"""The image datasets this package trains on, loaded as NCHW float32.

Self-contained on purpose: the mantissa family keeps datasets independent per
package, so this module owns its own MNIST / Fashion-MNIST loaders rather than
importing them from mantissa-cnn or mantissa-nn (the tasks here use only these
two — see :data:`mantissa_autoencoder.tasks.TASKS`).

Nothing downloads implicitly. ``load(name)`` reads files from the data
directory; if any are missing it raises FileNotFoundError with the exact fix
command. The only code that touches the network is the explicit CLI::

    python -m mantissa_autoencoder.datasets download <name|all>
    python -m mantissa_autoencoder.datasets list

Data directory: ``./data/<name>/`` relative to the current working directory,
or the ``MANTISSA_AE_DATA`` environment variable. The directory is gitignored
— datasets are never committed.

``load(name)`` -> (X_train, y_train, X_test, y_test): X is NCHW float32 scaled
to [0, 1]; y is int32 class ids 0..9. IDX files are parsed with numpy and
verified (magic number, dtype code, dimension consistency).

| name          | train/test    | shape       | source |
|---------------|---------------|-------------|--------|
| mnist         | 60000 / 10000 | (1, 28, 28) | LeCun et al. — ossci-datasets S3 mirror |
| fashion_mnist | 60000 / 10000 | (1, 28, 28) | Xiao, Rasul & Vollgraf (2017), zalandoresearch/fashion-mnist |

All URLs verified fetchable 2026-07.
"""
from __future__ import annotations

import gzip
import os
import sys
import urllib.request
from pathlib import Path
from typing import NamedTuple, Tuple

import numpy as np

__all__ = ["DATASETS", "data_dir", "download", "download_command", "load", "subset"]

_DATA_ENV = "MANTISSA_AE_DATA"

_IDX4 = ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
         "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz")


class _Spec(NamedTuple):
    base_url: str
    files: Tuple[str, ...]
    note: str


DATASETS = {
    "mnist": _Spec(
        "https://ossci-datasets.s3.amazonaws.com/mnist/", _IDX4,
        "handwritten digits (LeCun, Bottou, Bengio & Haffner, 1998)"),
    "fashion_mnist": _Spec(
        "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/",
        _IDX4,
        "Zalando clothing thumbnails, drop-in MNIST replacement (Xiao et al., 2017)"),
}


def data_dir() -> Path:
    return Path(os.environ.get(_DATA_ENV, "data"))


def download_command(name: str) -> str:
    return f"python -m mantissa_autoencoder.datasets download {name}"


# -- IDX parsing ---------------------------------------------------------------

_IDX_DTYPES = {0x08: np.dtype(">u1"), 0x0B: np.dtype(">i2"),
               0x0C: np.dtype(">i4"), 0x0D: np.dtype(">f4"),
               0x0E: np.dtype(">f8")}


def _read_idx(path: Path) -> np.ndarray:
    """Parse one (gzipped) IDX file, verifying magic number and sizes."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        buf = f.read()
    magic = int.from_bytes(buf[:4], "big")
    code, ndim = (magic >> 8) & 0xFF, magic & 0xFF
    if magic >> 16 != 0 or code not in _IDX_DTYPES or not 1 <= ndim <= 4:
        raise ValueError(f"{path}: bad IDX magic 0x{magic:08x}")
    dims = np.frombuffer(buf, ">u4", count=ndim, offset=4).astype(np.int64)
    dt = _IDX_DTYPES[code]
    expected = 4 + 4 * ndim + int(dims.prod()) * dt.itemsize
    if len(buf) != expected:
        raise ValueError(f"{path}: size {len(buf)} != expected {expected} "
                         f"for dims {dims.tolist()}")
    return np.frombuffer(buf, dt, offset=4 + 4 * ndim).reshape(dims)


def _load_idx_pair(images_path: Path, labels_path: Path):
    X = _read_idx(images_path)
    y = _read_idx(labels_path)
    if X.ndim != 3 or X.shape[1:] != (28, 28):
        raise ValueError(f"{images_path}: expected (n, 28, 28) images, got {X.shape}")
    if len(y) != len(X):
        raise ValueError(f"{labels_path}: {len(y)} labels for {len(X)} images")
    return X.reshape(-1, 1, 28, 28), y.astype(np.int32)


def _to_f01(X):
    """uint8 [0,255] NCHW -> C-contiguous float32 [0,1]."""
    return np.ascontiguousarray(X.astype(np.float32) / 255.0)


def _require_paths(name: str):
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; available: {', '.join(DATASETS)}")
    d = data_dir() / name
    paths = [d / f for f in DATASETS[name].files]
    if not all(p.is_file() for p in paths):
        raise FileNotFoundError(
            f"dataset {name!r} not downloaded — run: {download_command(name)}")
    return paths


# -- public API ---------------------------------------------------------------

def _load_u8(name: str):
    """load() without the float conversion: uint8 NCHW, int32 labels."""
    paths = _require_paths(name)
    Xtr, ytr = _load_idx_pair(paths[0], paths[1])
    Xte, yte = _load_idx_pair(paths[2], paths[3])
    return Xtr, ytr, Xte, yte


def load(name: str):
    """Load dataset ``name`` -> (X_train, y_train, X_test, y_test) as NCHW
    float32 in [0, 1], int32 labels. Never downloads: raises FileNotFoundError
    with the exact fix command if any file is missing."""
    Xtr, ytr, Xte, yte = _load_u8(name)
    return _to_f01(Xtr), ytr, _to_f01(Xte), yte


def subset(name: str, n_train: int, n_test: int, seed: int = 0):
    """Seeded stratified subset -> (X_train, y_train, X_test, y_test).

    Per-class quotas are as equal as the class counts allow (largest-remainder
    split). Slices the raw uint8 arrays and converts only the slice to float32
    — same values as slicing load()'s output, at a fraction of the peak RAM."""
    Xtr, ytr, Xte, yte = _load_u8(name)
    itr = _stratified_indices(ytr, n_train, np.random.default_rng(seed))
    ite = _stratified_indices(yte, n_test, np.random.default_rng(seed + 1))
    return (_to_f01(Xtr[itr]), ytr[itr],
            _to_f01(Xte[ite]), yte[ite])


def _stratified_indices(y, n, rng):
    classes = np.unique(y)
    base, extra = divmod(n, len(classes))
    picks = []
    for i, c in enumerate(classes):
        idx = np.flatnonzero(y == c)
        take = base + (1 if i < extra else 0)
        if take > len(idx):
            raise ValueError(f"class {c} has only {len(idx)} samples, need {take}")
        picks.append(rng.permutation(idx)[:take])
    return rng.permutation(np.concatenate(picks))


# -- explicit downloader (the only networking code) ----------------------------

def download(name: str) -> None:
    """Fetch every file of dataset ``name``, verified and atomic.

    The payload is checked before it can reach the load path: length against
    the server's Content-Length and the gzip magic — a truncated body or an
    HTML error page raises OSError instead of landing on disk. The verified
    body is written to a ``.part`` file and renamed into place.
    """
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; available: {', '.join(DATASETS)}")
    spec = DATASETS[name]
    d = data_dir() / name
    d.mkdir(parents=True, exist_ok=True)
    for fname in spec.files:
        path = d / fname
        if path.is_file():
            print(f"{name}: {fname} already present")
            continue
        url = spec.base_url + fname
        print(f"{name}: {url}\n  -> {path}")
        with urllib.request.urlopen(url, timeout=60) as r:
            length = r.headers.get("Content-Length")
            body = r.read()
        if length is not None and len(body) != int(length):
            raise OSError(f"{url}: truncated — got {len(body):,} of "
                          f"{int(length):,} announced bytes")
        if not body.startswith(b"\x1f\x8b"):
            raise OSError(f"{url}: not gzip data (starts {body[:4]!r}) — "
                          f"an error page or proxy response, not the dataset")
        tmp = path.with_name(fname + ".part")
        tmp.write_bytes(body)
        tmp.replace(path)
        print(f"  done ({len(body):,} bytes)")


def _main(argv) -> int:
    if len(argv) == 1 and argv[0] == "list":
        for name, spec in DATASETS.items():
            d = data_dir() / name
            state = "present" if all((d / f).is_file() for f in spec.files) else "missing"
            print(f"{name:14} {state:8} {spec.note}")
        return 0
    if len(argv) == 2 and argv[0] == "download":
        names = list(DATASETS) if argv[1] == "all" else [argv[1]]
        failed = []
        for name in names:
            if name not in DATASETS:
                print(f"unknown dataset {name!r}; available: {', '.join(DATASETS)}",
                      file=sys.stderr)
                return 2
            try:
                download(name)
            except Exception as exc:            # keep fetching the rest
                print(f"{name}: FAILED — {exc}", file=sys.stderr)
                failed.append(name)
        if failed:
            print(f"download failed for: {', '.join(failed)}", file=sys.stderr)
            return 1
        return 0
    print("usage: python -m mantissa_autoencoder.datasets download <name|all>\n"
          "       python -m mantissa_autoencoder.datasets list", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
