"""Speed + task-metric + peak-RSS benchmark vs torch and tensorflow.

Protocol (fixed in bench/protocol.py; do not tune per contender):
  - Tasks: the four recipes pinned in mantissa_autoencoder.tasks.TASKS
    (denoise / compress / anomaly / superres), each with its zoo
    architecture re-expressed layer-for-layer per contender
    (bench/contenders.py; parameter counts asserted equal).
  - Data: stratified N_TRAIN/N_TEST subsets, seed SEED. The anomaly task
    trains on the subset minus the held-out digit (~1800 samples).
  - Training: EPOCHS epochs, batch BATCH_SIZE, plain SGD lr=LR, MSE,
    seeded shuffles, CPU only.
  - Repeats: REPEATS, INTERLEAVED round-robin (A,B,C,D x R) so thermal and
    background drift hit every contender equally; medians reported, raw
    samples kept. time.perf_counter(); fit() wall time only (data prepped
    and framework imported beforehand). One untimed WARMUP_N-sample fit per
    contender first — first-call runtime setup (TF graph machinery, torch
    dispatch caches, our dylib load) is a one-time JIT-like cost, excluded
    the same way imports are.
  - Task metric on the held-out test subset with the benchmark's own
    trained model (re-seeded per repeat, so every repeat trains the same
    model; the last one is scored):
      denoise   PSNR(reconstruction of seeded-noisy test, clean test)
      compress  PSNR through the uint8-quantized latent (32 code bytes +
                8-byte range header = 40 B/image vs 784 B uint8 original,
                19.6x); the float32-code PSNR is recorded alongside
      anomaly   ROC-AUC of per-sample reconstruction MSE, held-out 1s positive
      superres  PSNR(refined output, ground truth); nearest baseline recorded
  - Batch reconstruct over the test subset: median of PREDICT_CALLS calls,
    interleaved, each framework's native inference path.
  - PEAK RSS: one (contender, task) per fresh subprocess; the child imports
    its own framework, fits once, reports
    resource.getrusage(RUSAGE_SELF).ru_maxrss (BYTES on macOS, KiB on
    Linux — normalized). Import cost deliberately included: users pay it.

Gallery capture: the same fitted models reconstruct one fixed test image
per task (denoise + superres) into bench/results/gallery.npz for
bench/plots.py — the galleries show what the benchmark trained, nothing
retrained or cherry-picked.

Output: bench/results/results.json
  {"env": {...}, "protocol": {...},
   "n_fit": {"<task>": n_train_actually_fit},
   "fit_s":          {"<task>": {"<contender>": {"median":, "samples": [...]}}},
   "reconstruct_ms": {...same nesting...},
   "task_metric":    {"<task>": {"<contender>": {"value":, "unit":,
                                  ...task extras (baselines, ratios)}}},
   "final_loss":     {"<task>": {"<contender>": last-epoch training MSE}},
   "peak_rss_mb":    {"<task>": {"<contender>": MB}}}

Run from the repo root:  python -m bench.speed
(the RSS worker re-invokes:  python -m bench.speed --worker <contender> <task>)
"""
from __future__ import annotations

import json
import os
import platform

# Keep TensorFlow's C++ banner out of benchmark output (set before any TF
# import anywhere in the process).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

import numpy as np

# numpy 2.x on Apple Accelerate emits spurious FPE RuntimeWarnings from the
# BLAS matmul kernel even on finite inputs (verified in the cnn repo:
# contender weights stay bounded). They fire from the vanilla-numpy
# backend's matmuls.
warnings.filterwarnings("ignore", message=".*encountered in matmul",
                        category=RuntimeWarning)

from mantissa_autoencoder.tasks import (dequantize_latent, psnr,
                                        quantize_latent, roc_auc)

from .contenders import contenders, task_data
from .protocol import (BATCH_SIZE, EPOCHS, LR, N_TEST, N_TRAIN, REPEATS,
                       SEED, TASKS)

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = REPO_ROOT / "bench" / "results" / "results.json"
GALLERY_PATH = REPO_ROOT / "bench" / "results" / "gallery.npz"

# Harness constants (not protocol: they do not touch training or metrics).
PREDICT_CALLS = 20      # batch-reconstruct timing repeats
WARMUP_N = 64           # samples for the one untimed warm-up fit
GALLERY_IDX = 0         # fixed test image shown in the galleries
GALLERY_CONTENDERS = ("ours", "torch", "tensorflow")


# --- per-task plumbing --------------------------------------------------------

def _test_pair(task, data):
    """(model input, metric reference) on the test subset, numpy NCHW."""
    if task == "denoise":
        return data["X_test_noisy"], data["X_test"]
    if task == "superres":
        return data["X_test"], data["X_test_target"]
    return data["X_test"], data["X_test"]


def _task_metric(task, est, data, prep, R):
    """The task metric from R = est.reconstruct(test input), plus honest
    extras (baselines the model must beat, the real compression ratio)."""
    test_in, ref = _test_pair(task, data)
    if task == "denoise":
        return {"value": round(psnr(R, ref), 4), "unit": "psnr_db",
                "noisy_input_psnr_db": round(psnr(test_in, ref), 4)}
    if task == "compress":
        Z = est.encode(prep(data["X_test"]))
        Rq = est.decode(dequantize_latent(*quantize_latent(Z)))
        return {"value": round(psnr(Rq, ref), 4), "unit": "psnr_db",
                "float32_code_psnr_db": round(psnr(R, ref), 4),
                "bytes_per_image": 40,      # 32 uint8 code + 8 B range header
                "compression_ratio": round(784 / 40.0, 2)}
    if task == "anomaly":
        d = (np.asarray(R, dtype=np.float64)
             - np.asarray(ref, dtype=np.float64))
        scores = np.mean(np.square(d), axis=(1, 2, 3))
        return {"value": round(roc_auc(scores, data["labels"]), 4),
                "unit": "auc"}
    if task == "superres":
        return {"value": round(psnr(R, ref), 4), "unit": "psnr_db",
                "nearest_baseline_psnr_db": round(psnr(test_in, ref), 4)}
    raise ValueError(f"unknown task {task!r}")


def _run_task(task, reg, gallery):
    """Interleaved timing + metrics for one task. Returns
    (fit_s, reconstruct_ms, task_metric, final_loss, n_fit) dicts keyed by
    contender; appends gallery panels for denoise/superres."""
    data = task_data(task)
    arch, sigma = data["arch"], data["sigma"]
    test_in, ref = _test_pair(task, data)

    # Native forms, converted once outside the timed region.
    native = {}
    for name, _factory, prep in reg:
        native[name] = (prep(data["X"]),
                        None if data["T"] is None else prep(data["T"]),
                        prep(test_in))

    # One untimed warm-up fit per contender (WARMUP_N samples).
    for name, factory, _prep in reg:
        Xn, Tn, _ = native[name]
        factory(arch, sigma).fit(Xn[:WARMUP_N],
                                 None if Tn is None else Tn[:WARMUP_N])

    # FIT: outer loop repeats, inner loop contenders -> true round-robin.
    # Fresh estimator per repeat (fresh weights); construction is untimed.
    fit_samples = {name: [] for name, *_ in reg}
    fitted = {}
    for _ in range(REPEATS):
        for name, factory, _prep in reg:
            Xn, Tn, _ = native[name]
            est = factory(arch, sigma)
            t0 = time.perf_counter()
            est.fit(Xn, Tn)
            fit_samples[name].append(time.perf_counter() - t0)
            fitted[name] = est

    # RECONSTRUCT: batch pass over the test subset, round-robin.
    rec_samples = {name: [] for name, *_ in reg}
    for _ in range(PREDICT_CALLS):
        for name, *_rest in reg:
            Xtn = native[name][2]
            t0 = time.perf_counter()
            fitted[name].reconstruct(Xtn)
            rec_samples[name].append((time.perf_counter() - t0) * 1000.0)

    # METRIC + gallery from the models the benchmark actually trained.
    metrics, final_loss = {}, {}
    for name, _factory, prep in reg:
        R = fitted[name].reconstruct(native[name][2])
        metrics[name] = _task_metric(task, fitted[name], data, prep, R)
        final_loss[name] = round(fitted[name].final_loss_, 6)
        if task in ("denoise", "superres") and name in GALLERY_CONTENDERS:
            gallery[f"{task}_{name}"] = R[GALLERY_IDX, 0]
    if task == "denoise":
        gallery["denoise_clean"] = ref[GALLERY_IDX, 0]
        gallery["denoise_noisy"] = test_in[GALLERY_IDX, 0]
    elif task == "superres":
        gallery["superres_input"] = test_in[GALLERY_IDX, 0]
        gallery["superres_gt"] = ref[GALLERY_IDX, 0]

    fit_s = {n: {"median": median(s), "samples": s}
             for n, s in fit_samples.items()}
    rec_ms = {n: {"median": median(s), "samples": s}
              for n, s in rec_samples.items()}
    return fit_s, rec_ms, metrics, final_loss, len(data["X"])


# --- RSS worker ---------------------------------------------------------------

def _rss_mb() -> float:
    import resource
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss: bytes on macOS, KiB on Linux.
    if sys.platform == "darwin":
        return maxrss / (1024.0 * 1024.0)
    return maxrss / 1024.0


def _run_worker(contender: str, task: str) -> int:
    """Fresh subprocess: import the contender's framework, fit once under
    the full protocol, print peak RSS in MB. Import cost included on
    purpose — it is what a user pays."""
    spec = {name: (factory, prep)
            for name, factory, prep in contenders()}.get(contender)
    if spec is None:
        print(f"unknown contender {contender!r}", file=sys.stderr)
        return 2
    factory, prep = spec
    data = task_data(task)
    Xn = prep(data["X"])
    Tn = None if data["T"] is None else prep(data["T"])
    factory(data["arch"], data["sigma"]).fit(Xn, Tn)
    print(f"{_rss_mb():.4f}")
    return 0


def _measure_rss(contender: str, task: str) -> float:
    proc = subprocess.run(
        [sys.executable, "-m", "bench.speed", "--worker", contender, task],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"RSS worker failed for {contender}/{task}:\n{proc.stderr}")
    return float(proc.stdout.strip().splitlines()[-1])


# --- environment ---------------------------------------------------------------

def _cpu_name() -> str:
    if sys.platform == "darwin":
        try:
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        except Exception:
            pass
    elif sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except Exception:
            pass
    return platform.processor() or platform.machine() or "unknown"


def _env_block() -> dict:
    """Versions and thread settings — thread knobs are left at each
    framework's default and RECORDED, not equalized."""
    from mantissa_nn import MANTISSA_MIN_VERSION
    env = {
        "cpu": _cpu_name(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "mantissa": f">={MANTISSA_MIN_VERSION} (f32 CNN primitives)",
        "mantissa_threads": os.environ.get("MANTISSA_THREADS",
                                           f"default({os.cpu_count()})"),
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    }
    import torch
    env["torch"] = torch.__version__
    env["torch_threads"] = torch.get_num_threads()
    import tensorflow as tf
    import keras
    env["tensorflow"] = tf.__version__
    env["keras"] = keras.__version__
    env["tf_inter_op_threads"] = tf.config.threading.get_inter_op_parallelism_threads()
    env["tf_intra_op_threads"] = tf.config.threading.get_intra_op_parallelism_threads()
    env["tf_threads_note"] = "0 = TensorFlow default (runtime-chosen)"
    return env


# --- entrypoint ------------------------------------------------------------------

def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] == "--worker":
        return _run_worker(argv[1], argv[2])

    reg = contenders()
    names = [n for n, *_ in reg]
    print(f"contenders: {', '.join(names)}")

    fit_s, rec_ms, task_metric = {}, {}, {}
    final_loss, peak_rss_mb, n_fit = {}, {}, {}
    gallery = {}
    t_start = time.perf_counter()
    for task in TASKS:
        print(f"\n[{task}] timing (R={REPEATS}, interleaved) ...")
        f, r, m, fl, nf = _run_task(task, reg, gallery)
        fit_s[task], rec_ms[task] = f, r
        task_metric[task], final_loss[task], n_fit[task] = m, fl, nf
        for name in names:
            mv = m[name]
            print(f"  {name:14s} fit {f[name]['median']:8.3f} s   "
                  f"reconstruct {r[name]['median']:9.2f} ms   "
                  f"{mv['unit']} {mv['value']:.4f}")
        print(f"[{task}] peak RSS (fresh subprocess each) ...")
        peak_rss_mb[task] = {}
        for name in names:
            mb = _measure_rss(name, task)
            peak_rss_mb[task][name] = round(mb, 4)
            print(f"  {name:14s} {mb:8.1f} MB")

    out = {
        "env": _env_block(),
        "protocol": {"tasks": list(TASKS), "contenders": names,
                     "n_train": N_TRAIN, "n_test": N_TEST,
                     "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
                     "seed": SEED, "repeats": REPEATS,
                     "predict_calls": PREDICT_CALLS, "warmup_n": WARMUP_N},
        "n_fit": n_fit,
        "fit_s": fit_s,
        "reconstruct_ms": rec_ms,
        "task_metric": task_metric,
        "final_loss": final_loss,
        "peak_rss_mb": peak_rss_mb,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(out, indent=2) + "\n")
    np.savez_compressed(GALLERY_PATH, **gallery)
    print(f"\nwrote {RESULTS_PATH.relative_to(REPO_ROOT)} and "
          f"{GALLERY_PATH.relative_to(REPO_ROOT)} "
          f"({time.perf_counter() - t_start:.0f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
