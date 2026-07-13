"""Benchmark plots + galleries. Consumes bench/results/results.json and
bench/results/gallery.npz (produced by bench.speed) and writes PNGs to
assets/. Never invents data: exits with a message if an input is missing.

Figures (matplotlib, no seaborn, one chart per file):
  assets/fit_time.png          median fit seconds per task, grouped bars per
                               contender, log scale
  assets/task_metrics.png      the task metric per contender — mixed units
                               (PSNR dB / ROC-AUC), so one panel per task,
                               each with its honest baseline drawn in
                               (noisy input, nearest upscale, AUC chance)
  assets/peak_rss.png          peak RSS per (task, contender), fresh-process
  assets/gallery_denoise.png   clean | noisy | ours | torch | tf — one fixed
                               test image through the benchmarked models
  assets/gallery_superres.png  nearest input | ours | torch | tf | ground truth

Run from the repo root:  python -m bench.plots
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "bench" / "results"
ASSETS = REPO_ROOT / "assets"

# One stable color per contender across every plot — the cnn repo's palette
# (categorical slots validated there with the dataviz six-checks script
# against this light surface: worst adjacent-pair CVD dE 13.9; the two
# lower-contrast hues are relieved by the direct value label every bar
# carries).
COLORS = {
    "ours": "#2a78d6",           # blue
    "vanilla_numpy": "#1baf7a",  # aqua
    "torch": "#e34948",          # red
    "tensorflow": "#d96b2f",     # orange
}
LABELS = {
    "ours": "ours (mantissa)",
    "vanilla_numpy": "vanilla numpy",
    "torch": "torch",
    "tensorflow": "tensorflow",
}
ORDER = ["ours", "vanilla_numpy", "torch", "tensorflow"]

# Opaque light surface so the PNG reads on GitHub light AND dark themes.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

METRIC_LABEL = {"psnr_db": "PSNR — dB (higher is better)",
                "auc": "ROC-AUC"}


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": AXIS,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
    })


def _load_results():
    path = RESULTS / "results.json"
    if not path.is_file():
        raise SystemExit(f"{path.relative_to(REPO_ROOT)} missing — run "
                         f"python -m bench.speed first")
    return json.loads(path.read_text())


def _load_gallery():
    path = RESULTS / "gallery.npz"
    if not path.is_file():
        raise SystemExit(f"{path.relative_to(REPO_ROOT)} missing — run "
                         f"python -m bench.speed first")
    return np.load(path)


def _short_env(env) -> str:
    bits = [env.get("cpu", "?"), f"Python {env.get('python', '?')}"]
    if env.get("date"):
        bits.append(env["date"])
    return "  ·  ".join(bits)


def _contender_order(per_task):
    seen = {c for row in per_task.values() for c in row}
    return [c for c in ORDER if c in seen] + sorted(seen - set(ORDER))


def _grouped_bars(ax, contenders, tasks, values, log=False, fmt="{:.2f}"):
    """values[contender][task] -> float. One group per task."""
    n_series = len(contenders)
    x = np.arange(len(tasks))
    width = 0.8 / n_series
    all_h = []
    for si, c in enumerate(contenders):
        heights = [values[c].get(t, 0.0) for t in tasks]
        offset = (si - (n_series - 1) / 2) * width
        bars = ax.bar(x + offset, heights, width, label=LABELS[c],
                      color=COLORS[c], edgecolor=SURFACE, linewidth=0.8,
                      zorder=3)
        for rect, h in zip(bars, heights):
            if h <= 0:
                continue
            all_h.append(h)
            ax.annotate(fmt.format(h), (rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=6.5, rotation=90,
                        color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    if log:
        ax.set_yscale("log")
        # headroom for the rotated value labels + legend row above tall bars
        ax.set_ylim(min(all_h) / 3.0, max(all_h) * 14.0)
    else:
        ax.margins(y=0.22)


def plot_fit_time(res):
    tasks = res["protocol"]["tasks"]
    per_task = {t: {c: v["median"] for c, v in res["fit_s"][t].items()}
                for t in tasks}
    contenders = _contender_order(per_task)
    values = {c: {t: per_task[t][c] for t in tasks} for c in contenders}
    r = res["protocol"]["repeats"]
    e = res["protocol"]["epochs"]

    fig, ax = plt.subplots(figsize=(9.5, 4.4), dpi=150)
    _grouped_bars(ax, contenders, tasks, values, log=True, fmt="{:.2f}")
    ax.set_ylabel("median fit time — s (log scale)")
    ax.set_title(f"Training time per task — median of {r} interleaved fits, "
                 f"{e} epochs", color=INK, fontsize=13, fontweight="bold",
                 pad=30, loc="left")
    ax.text(0, 1.05, _short_env(res["env"]), transform=ax.transAxes,
            fontsize=8, color=INK2, va="bottom")
    ax.legend(loc="upper left", framealpha=0.9, facecolor=SURFACE,
              edgecolor=GRID, fontsize=7.5, ncol=len(contenders))
    _save(fig, "fit_time.png")


# Honest baseline per task panel: (label, value-key in the metric extras,
# or a literal for AUC chance).
_BASELINES = {
    "denoise": ("noisy input", "noisy_input_psnr_db"),
    "superres": ("nearest upscale", "nearest_baseline_psnr_db"),
    "anomaly": ("chance (0.5)", 0.5),
}


def plot_task_metrics(res):
    """One panel per task — PSNR and AUC must not share an axis."""
    tasks = res["protocol"]["tasks"]
    contenders = _contender_order(res["task_metric"])
    fig, axes = plt.subplots(1, len(tasks), figsize=(11.5, 3.6), dpi=150)
    for ti, (task, ax) in enumerate(zip(tasks, np.atleast_1d(axes))):
        row = res["task_metric"][task]
        unit = row[contenders[0]]["unit"]
        x = np.arange(len(contenders))
        heights = [row[c]["value"] for c in contenders]
        fmt = "{:.2f}" if unit == "psnr_db" else "{:.3f}"
        bars = ax.bar(x, heights, 0.62, color=[COLORS[c] for c in contenders],
                      edgecolor=SURFACE, linewidth=0.8, zorder=3)
        for rect, h in zip(bars, heights):
            ax.annotate(fmt.format(h), (rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7.5, color=INK2)
        base = _BASELINES.get(task)
        if base is not None:
            label, key = base
            val = key if isinstance(key, float) else row[contenders[0]][key]
            ax.axhline(val, color=INK2, linewidth=1.0, linestyle="--",
                       zorder=2)
            ax.annotate(label, (0.02, val), xycoords=("axes fraction", "data"),
                        xytext=(0, 2), textcoords="offset points",
                        fontsize=6.5, color=INK2, va="bottom")
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[c].replace(" ", "\n") for c in contenders],
                           fontsize=6.5)
        ax.set_title(task, color=INK, fontsize=11, fontweight="bold",
                     loc="left")
        ax.set_ylabel(METRIC_LABEL[unit], fontsize=8)
        ax.set_axisbelow(True)
        ax.grid(axis="x", visible=False)
        ax.margins(y=0.22)
        if unit == "auc":
            ax.set_ylim(0, 1.0)
    fig.suptitle("Task metrics on the held-out test subset — dashed = the "
                 "baseline the model must beat", color=INK, fontsize=12,
                 fontweight="bold", x=0.005, ha="left")
    fig.text(0.005, 0.905, _short_env(res["env"]), fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    _save(fig, "task_metrics.png", tight=False)


def plot_peak_rss(res):
    tasks = res["protocol"]["tasks"]
    per_task = res["peak_rss_mb"]
    contenders = _contender_order(per_task)
    values = {c: {t: per_task[t][c] for t in tasks} for c in contenders}

    fig, ax = plt.subplots(figsize=(9.5, 4.2), dpi=150)
    _grouped_bars(ax, contenders, tasks, values, log=False, fmt="{:.0f}")
    ax.set_ylabel("peak RSS — MB")
    ax.set_title("Peak memory — import + one fit, fresh process each",
                 color=INK, fontsize=13, fontweight="bold", pad=30, loc="left")
    ax.text(0, 1.05, _short_env(res["env"]), transform=ax.transAxes,
            fontsize=8, color=INK2, va="bottom")
    ax.legend(loc="upper left", framealpha=0.9, facecolor=SURFACE,
              edgecolor=GRID, fontsize=7.5, ncol=len(contenders))
    _save(fig, "peak_rss.png")


# --- galleries -------------------------------------------------------------------

def _psnr(a, b) -> float:
    mse = float(np.mean(np.square(a.astype(np.float64) -
                                  b.astype(np.float64))))
    return float("inf") if mse == 0 else 10.0 * np.log10(1.0 / mse)


def _gallery(gal, name, cols, ref_key, title):
    """One row of 28x28 panels; every model panel is captioned with its
    PSNR against the reference for THIS image (the table reports the
    1000-image test-set numbers)."""
    ref = gal[ref_key]
    fig, axes = plt.subplots(1, len(cols), figsize=(1.62 * len(cols), 2.15),
                             dpi=150)
    for (key, label), ax in zip(cols, axes):
        img = gal[key]
        ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(AXIS)
        cap = label
        if key != ref_key:
            cap += f"\n{_psnr(img, ref):.1f} dB"
        ax.set_title(cap, fontsize=7.5, color=INK, pad=4)
    fig.suptitle(title, color=INK, fontsize=10, fontweight="bold",
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, name, tight=False)


def plot_galleries(gal):
    _gallery(gal, "gallery_denoise.png",
             [("denoise_clean", "clean"),
              ("denoise_noisy", "noisy σ=0.3"),
              ("denoise_ours", "ours (mantissa)"),
              ("denoise_torch", "torch"),
              ("denoise_tensorflow", "tensorflow")],
             ref_key="denoise_clean",
             title="Denoising — one fashion_mnist test image "
                   "(per-image PSNR vs clean)")
    _gallery(gal, "gallery_superres.png",
             [("superres_input", "nearest ×2 input"),
              ("superres_ours", "ours (mantissa)"),
              ("superres_torch", "torch"),
              ("superres_tensorflow", "tensorflow"),
              ("superres_gt", "ground truth")],
             ref_key="superres_gt",
             title="Super-resolution — one mnist test image, 28→14→28 "
                   "(per-image PSNR vs truth)")


def _save(fig, name, tight=True):
    ASSETS.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    # metadata=Software:None -> byte-stable across runs (no version/timestamp).
    fig.savefig(ASSETS / name, dpi=150, metadata={"Software": None})
    plt.close(fig)
    print(f"wrote assets/{name}")


def main() -> int:
    _style()
    res = _load_results()
    gal = _load_gallery()
    plot_fit_time(res)
    plot_task_metrics(res)
    plot_peak_rss(res)
    plot_galleries(gal)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
