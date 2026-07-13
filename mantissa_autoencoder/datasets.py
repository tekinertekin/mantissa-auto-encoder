"""Datasets are mantissa-cnn's — re-exported, never reimplemented.

``mantissa_cnn.datasets`` resolves its data directory as ``./data`` relative
to the *current working directory* (or the ``MANTISSA_CNN_DATA`` environment
variable), which works from the cnn repo but not from here. Importing this
module fixes that once: if the variable is unset and there is no local
``./data``, it points ``MANTISSA_CNN_DATA`` at the ``data/`` directory next
to the installed ``mantissa_cnn`` package — the cnn checkout's ``data/`` in
the editable dev layout, where the datasets already live. Nothing is ever
downloaded twice; the download CLI stays mantissa-cnn's::

    python -m mantissa_cnn.datasets download <name|all>
    python -m mantissa_cnn.datasets list

An explicit ``MANTISSA_CNN_DATA`` or a local ``./data`` always wins.
"""
from __future__ import annotations

import os
from pathlib import Path

from mantissa_cnn import datasets as _cnn_datasets

__all__ = ["DATASETS", "data_dir", "download", "download_command",
           "load", "subset"]

_DATA_ENV = "MANTISSA_CNN_DATA"


def _point_at_sibling_data() -> None:
    if _DATA_ENV in os.environ or Path("data").is_dir():
        return                     # the caller's choice stands
    candidate = Path(_cnn_datasets.__file__).resolve().parents[1] / "data"
    if candidate.is_dir():
        os.environ[_DATA_ENV] = str(candidate)


_point_at_sibling_data()

DATASETS = _cnn_datasets.DATASETS
data_dir = _cnn_datasets.data_dir
download = _cnn_datasets.download
download_command = _cnn_datasets.download_command
load = _cnn_datasets.load
subset = _cnn_datasets.subset
