"""Benchmark protocol constants — fixed BEFORE any benchmark code runs, so
the numbers cannot be tuned after the fact (the family rule: measure, don't
assume). The harness (speed/accuracy/plots, mantissa-cnn's bench layout) is
a later phase; nothing here executes a benchmark.

Contenders: ours (mantissa engine), ours (numpy backend), torch,
tensorflow — the same architecture re-expressed layer-for-layer in each,
identical hyperparameters, CPU only. Metrics per (task, contender): fit
wall-time (median of interleaved repeats), the task metric (PSNR for
denoise/compress/superres, AUC for anomaly), and peak RSS in a fresh
subprocess with import cost included.

Task recipes (dataset, model, task parameters) are pinned in
:data:`mantissa_autoencoder.tasks.TASKS`.
"""
N_TRAIN = 2000          # stratified subset sizes, mantissa_cnn.datasets.subset
N_TEST = 1000
SEED = 0
EPOCHS = 5
BATCH_SIZE = 32
LR = 0.01
REPEATS = 5             # interleaved A/B/C/A/B/C..., median reported

TASKS = ("denoise", "compress", "anomaly", "superres")
CONTENDERS = ("ours (mantissa)", "vanilla numpy", "torch", "tensorflow")
METRICS = ("fit_s", "task_metric", "peak_rss_mb")
