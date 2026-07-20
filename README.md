# mantissa-autoencoder

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-3776AB.svg)
[![Base](https://img.shields.io/badge/base-mantissa--cnn-4B8BBE.svg)](https://github.com/tekinertekin/mantissa-cnn)
[![Engine](https://img.shields.io/badge/engine-mantissa-00599C.svg)](https://github.com/tekinertekin/mantissa)

**Classic autoencoders, with a C engine.**

Convolutional autoencoders (`fit` / `encode` / `decode` / `reconstruct`)
built on top of [mantissa-cnn](https://github.com/tekinertekin/mantissa-cnn):
its Conv2D / MaxPool2D / Flatten / Dense layers, its
[mantissa](https://github.com/tekinertekin/mantissa) C-engine and pure-numpy
backends, and its dataset loaders are all reused, not reimplemented. This
package adds only what an autoencoder needs on top of a classifier: the two
decoder layers (`Upsample2D`, `Reshape`), an MSE-trained `Autoencoder`, a
three-model zoo, and helpers for the four tasks autoencoders are classically
used for — denoising, compression, anomaly detection, super-resolution.

Deliberately minimal, like the rest of the family: NCHW float32 images,
pixel MSE, plain SGD. No autograd graph, no optimizer zoo. The convolutions
and the nearest-neighbor upsample run in C on zero-copy float32 buffers
(the upsample since mantissa 0.2.3, with an automatic numpy fallback for
older engines); the loss is memory-bound array movement and honestly stays
in numpy. Layers allocate their scratch once per batch shape and reuse it —
steady-state training does no per-batch allocation.

## The mantissa family

Part of the **mantissa** family: a low-precision engine written in C, with
small Python packages built on top. Each package sits under the one it depends
on — ⭐ marks where you are, and every other name links to its repo.

- [mantissa](https://github.com/tekinertekin/mantissa) — low-precision neural-network engine in C (the core)
  - [mantissa-perceptron](https://github.com/tekinertekin/mantissa-perceptron) — perceptron & ADALINE, the linear classics
  - [mantissa-nn](https://github.com/tekinertekin/mantissa-nn) — shared neural-net primitives (layers, engine binding)
    - [mantissa-cnn](https://github.com/tekinertekin/mantissa-cnn) — convolutional networks for images
      - ⭐ **mantissa-auto-encoder** — autoencoders for denoising & super-resolution *(you are here)*
      - [mantissa-interpret](https://github.com/tekinertekin/mantissa-interpret) — CNN interpretability (occlusion, saliency, Grad-CAM)
    - [mantissa-mlp](https://github.com/tekinertekin/mantissa-mlp) — multilayer perceptrons, fully-connected nets


## Install

```sh
pip install mantissa-autoencoder
```

This pulls in `mantissa-cnn >= 0.1.0` (which pulls the engine
`mantissa-core >= 0.2.1`).

The engine-accelerated decoder path (nearest-neighbor upsample in C) needs
`mantissa-core >= 0.2.3` for `Session.upsample2d`; older engines and
`backend="numpy"` fall back to the numpy expressions automatically — nothing
breaks, it is just slower.

From checkouts (works today, no PyPI needed): clone this repo, `cnn`, and
[mantissa](https://github.com/tekinertekin/mantissa) side by side, build the
engine (`make dist` there), then here:

```sh
pip install -e ../cnn && pip install -e ".[dev]"
```

mantissa-cnn finds the sibling engine checkout automatically, and this
package finds mantissa-cnn's `data/` directory automatically (set
`MANTISSA_CNN_DATA` to override).

## Quickstart

```sh
# datasets are mantissa-cnn's; nothing downloads implicitly — fetch once:
python -m mantissa_cnn.datasets download fashion_mnist
```

```python
from mantissa_autoencoder import models, datasets, tasks

X_train, _, X_test, _ = datasets.load("fashion_mnist")   # labels unused
ae = models.denoise_ae()                 # C engine; backend="numpy" also works
print(ae.summary())

# the denoising recipe: corrupt the INPUT, reconstruct the clean target
ae.fit(X_train, epochs=5, batch_size=32, lr=0.01,
       noise=lambda x: tasks.add_gaussian_noise(x, sigma=0.3), verbose=True)

noisy = tasks.add_gaussian_noise(X_test, sigma=0.3)
print("PSNR noisy   :", tasks.psnr(noisy, X_test))
print("PSNR denoised:", tasks.psnr(ae.reconstruct(noisy), X_test))
```

Or compose your own from mantissa-cnn's layers plus the two added here:

```python
from mantissa_cnn import Conv2D, MaxPool2D, Flatten, Dense
from mantissa_autoencoder import Autoencoder, Upsample2D, Reshape

ae = Autoencoder(
    encoder_layers=[Conv2D(16, 3, pad=1), MaxPool2D(2),
                    Flatten(), Dense(32)],           # 784 pixels -> 32 floats
    decoder_layers=[Dense(16 * 14 * 14, act="relu"), Reshape((16, 14, 14)),
                    Upsample2D(2), Conv2D(1, 3, pad=1, act="identity")],
    seed=0)
```

## New to autoencoders? The idea, and the four things it buys

An autoencoder is a network trained to output its own input. That sounds
useless until you make it *hard*: the **encoder** squeezes the image down —
pooling away resolution, or all the way to a few numbers — into a
**bottleneck** code, and the **decoder** has to rebuild the image from that
code alone. Reconstruction error is the whole training signal (no labels),
so the only way to do well is for the code to keep what actually matters
about the image and drop the rest. The compression *is* the learning.

<img src="https://raw.githubusercontent.com/tekinertekin/mantissa-auto-encoder/main/assets/concepts/autoencoder.png" width="380" alt="autoencoder schema: input through encoder to a code, decoder reconstructs the input">

**Why our decoders upsample-then-convolve.** A decoder must grow small
feature maps back to image size. The obvious inverse of convolution —
transposed convolution — overlaps its kernel footprints unevenly and stamps
a checkerboard pattern into the output. Nearest-neighbor resize followed by
a plain convolution cannot produce that artifact by construction, at the
same parameter cost (Odena, Dumoulin & Olah, 2016, "Deconvolution and
Checkerboard Artifacts", *Distill*). That is `Upsample2D` + `Conv2D`
everywhere in this zoo, and why there is no `ConvTranspose2D`.

**Denoising.** Corrupt the input, keep the target clean, and the identity
shortcut is gone: the network can only succeed by learning what digits and
sleeves *look like* and using that to fill in what the noise destroyed.
Denoising started as a way to force robust features, not as an application —
the cleaned-up image was the byproduct (Vincent, Larochelle, Bengio &
Manzagol, 2008, "Extracting and Composing Robust Features with Denoising
Autoencoders", *ICML*).

**Compression.** Make the bottleneck a handful of numbers and the code
becomes a learned, lossy compression of the image — Hinton & Salakhutdinov
squeezed MNIST digits through 30 floats and reconstructed recognizably,
beating PCA at equal dimension because the mapping is nonlinear (2006,
"Reducing the Dimensionality of Data with Neural Networks", *Science*
313(5786)). Our compress task quantizes the 32-float code to uint8 and
counts real shipped bytes: 40 per image against the 784-byte original.

**Anomaly detection.** Train on normal data only. The autoencoder learns to
reconstruct what it has seen — and only that. Feed it something it never saw
and the reconstruction goes wrong, so per-sample reconstruction error is an
anomaly score, no anomaly labels needed at training time (Sakurada & Yairi,
2014, "Anomaly Detection Using Autoencoders with Nonlinear Dimensionality
Reduction", *MLSDA*). Our task holds digit 1 out of training and asks the
error to rank the unseen 1s highest.

**Super-resolution.** Upscale a low-res image by simple interpolation
outside the net, then train a small stack of convolutions to refine that
blurry guess toward the original — SRCNN showed three conv layers
(extraction → mapping → reconstruction) beat the classical
sparse-coding pipeline (Dong, Loy, He & Tang, 2014/2016, "Image
Super-Resolution Using Deep Convolutional Networks", *TPAMI* 38(2)). No
bottleneck here; it earns its place as the input→target form of the same
MSE trainer.

**Deliberate non-goals.** U-Net's skip connections pass encoder maps
straight to the decoder, which needs a graph, not a chain — out of scope
for `Sequential`-style stacks (Ronneberger, Fischer & Brox, 2015, "U-Net:
Convolutional Networks for Biomedical Image Segmentation", *MICCAI*, is the
pointer). Variational autoencoders sample the code through the
reparameterization trick and add a KL term — machinery this trainer does
not have (Kingma & Welling, 2014, "Auto-Encoding Variational Bayes",
*ICLR*).

<sub>Autoencoder schema by Michela Massi, via
[Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Autoencoder_schema.png),
licensed [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) —
redistributed here with attribution, scaled to 500 px.</sub>

## Model zoo

Honest names: classic recipes at small-image scale, with deviations from
the papers flagged in each docstring.

| model | architecture | paper |
|-------|--------------|-------|
| `denoise_ae` | Conv 16 → pool → Conv 32 → pool ‖ Upsample → Conv 16 → Upsample → Conv 1 (identity); spatial 32@7×7 latent. Conv body instead of the paper's dense stacks — flagged | Vincent, Larochelle, Bengio & Manzagol (2008), "Extracting and Composing Robust Features with Denoising Autoencoders", *ICML* |
| `bottleneck_ae` | same conv body around Flatten → Dense(32) ‖ Dense → Reshape; a 32-float linear code (theirs was 30, RBM-pretrained — flagged) | Hinton & Salakhutdinov (2006), "Reducing the Dimensionality of Data with Neural Networks", *Science* 313(5786) |
| `srcnn` | Conv 32@5×5 → Conv 16@3×3 → Conv 1@3×3, size-preserving; input is nearest-upscaled low-res, upscaled *outside* the net (paper uses bicubic — flagged) | Dong, Loy, He & Tang (2014/2016), "Image Super-Resolution Using Deep Convolutional Networks", *TPAMI* 38(2) |

## Datasets

mantissa-cnn's loaders, re-exported (`mantissa_autoencoder.datasets` just
points them at the right `data/` directory from this repo). Same contract:
NCHW float32 in [0, 1], nothing downloads implicitly, `subset()` gives
seeded stratified slices. mnist / fashion_mnist / kmnist / qmnist / cifar10 —
sources, sample gallery and download CLI are documented in
[mantissa-cnn](https://github.com/tekinertekin/mantissa-cnn#datasets).

## Results

<!-- BEGIN:BENCH (bench/speed.py + bench/plots.py output; do not edit outside these markers) -->
Protocol (fixed in `bench/protocol.py` before the first number was
measured): the **same architecture, re-expressed layer-for-layer in each
framework** (`torch.nn.Sequential` eager, `tf.keras.Sequential`, our
encoder/decoder stacks — parameter counts asserted equal by
`python -m bench.contenders`), identical hyperparameters everywhere —
plain SGD, lr 0.01, batch 32, 5 epochs, MSE, seed 0 — on stratified
2000-train / 1000-test subsets, CPU only. Fit wall-time is the median of
5 interleaved repeats (one untimed warm-up each); reconstruct is a batch
pass over the 1000-image test subset, median of 20 interleaved calls;
peak RSS is one fresh subprocess per (contender, task), import cost
included. Task metrics come from the models the benchmark itself trained.
`vanilla numpy` is our pure-numpy reference backend — no mantissa engine —
showing what the C core buys.

**denoise** — fashion_mnist, Gaussian σ 0.3 on the input only; the noisy
test input scores **12.77 dB** against clean, the floor every model must
beat:

| contender | fit (s) ↓ | reconstruct (ms) ↓ | PSNR (dB) ↑ | peak RSS (MB) ↓ |
|-----------|----------:|-------------------:|------------:|----------------:|
| tensorflow | **1.352** | 53.3 | 14.17 | 632 |
| **ours (mantissa)** | 1.541 | **35.1** | **14.31** | **190** |
| torch | 4.025 | 41.3 | 14.23 | 358 |
| vanilla numpy | 6.449 | 225.4 | 14.30 | 223 |

**compress** — mnist through a 32-float code, then uint8-quantized: 32
code bytes + an 8-byte range header = 40 B/image vs the 784-byte uint8
original, an honest **19.6×** (quantization costs < 0.001 dB at this code
size — the float32-code PSNR is in the JSON):

| contender | fit (s) ↓ | reconstruct (ms) ↓ | PSNR @ 19.6× (dB) ↑ | peak RSS (MB) ↓ |
|-----------|----------:|-------------------:|--------------------:|----------------:|
| tensorflow | **1.399** | 55.6 | **10.85** | 618 |
| **ours (mantissa)** | 1.643 | **37.6** | 10.45 | **188** |
| torch | 3.977 | 41.5 | 10.60 | 383 |
| vanilla numpy | 6.202 | 207.9 | 10.45 | 222 |

**anomaly** — mnist, digit 1 held out of training (1800 fit samples),
per-sample reconstruction MSE as the score, held-out 1s positive:

| contender | fit (s) ↓ | reconstruct (ms) ↓ | ROC-AUC | peak RSS (MB) ↓ |
|-----------|----------:|-------------------:|--------:|----------------:|
| tensorflow | **1.290** | 56.8 | 0.131 | 611 |
| **ours (mantissa)** | 1.479 | **37.9** | 0.174 | **185** |
| torch | 3.635 | 41.8 | 0.064 | 381 |
| vanilla numpy | 5.649 | 210.1 | 0.174 | 217 |

**superres** — mnist 28 → 14 (2×2 mean) → nearest-upscaled back to 28
outside the net, `srcnn` refines; the nearest-upscaled input scores
**17.82 dB**, the floor:

| contender | fit (s) ↓ | reconstruct (ms) ↓ | PSNR (dB) ↑ | peak RSS (MB) ↓ |
|-----------|----------:|-------------------:|------------:|----------------:|
| **ours (mantissa)** | **1.789** | **44.5** | **18.12** | **197** |
| tensorflow | 1.859 | 64.9 | 18.06 | 557 |
| torch | 5.476 | 84.7 | 17.74 | 388 |
| vanilla numpy | 6.935 | 197.5 | 18.12 | 256 |

![median fit time per task per contender](https://raw.githubusercontent.com/tekinertekin/mantissa-auto-encoder/main/assets/fit_time.png)
![task metric per contender, one panel per task with its baseline](https://raw.githubusercontent.com/tekinertekin/mantissa-auto-encoder/main/assets/task_metrics.png)
![peak RSS per task per contender](https://raw.githubusercontent.com/tekinertekin/mantissa-auto-encoder/main/assets/peak_rss.png)

The galleries below are the models the benchmark trained — same seed,
same test image, nothing retrained or cherry-picked:

![denoising gallery: clean, noisy, ours, torch, tensorflow](https://raw.githubusercontent.com/tekinertekin/mantissa-auto-encoder/main/assets/gallery_denoise.png)
![super-resolution gallery: nearest input, ours, torch, tensorflow, ground truth](https://raw.githubusercontent.com/tekinertekin/mantissa-auto-encoder/main/assets/gallery_superres.png)

**The honest read.**
- **Fit: TensorFlow's compiled graph still leads three of four tasks,
  but the gap collapsed from ~1.5× to 1.14–1.17×.** The previous run's
  recorded next target — an engine-side upsample primitive — landed in
  mantissa 0.2.3 (`upsample2d`; at these decoder shapes the numpy
  forward ran at 4 vs 74 GB/s and the backward's fused `np.sum` over two
  interleaved length-k axes was ~9× off, up to 47× end-to-end), and our
  denoise fit fell 2.34 → 1.54 s. Ours keeps superres — the control
  architecture with **no upsample and no pooling**, pure engine
  convolutions — and beats torch eager 2.4–3.1× on every task; the
  numpy backend trails ours 3.8–4.2×. TF's remaining lead has the same
  mechanical explanation as before, now smaller: decoders convolve at
  **full 28×28 resolution** — the heavy-conv regime where the cnn
  repo's benchmark also found TF's graph executor strongest — and the
  stages still running as numpy between engine calls (the denoise
  corruption, batch staging, the MSE/loss step) are what TF fuses into
  one graph.
- **Reconstruct is now ours on all four tasks** (35–45 ms; torch
  41–85, tf 53–65 — what was a three-way photo finish broke open when
  the decoder's upsample forward moved into C); the numpy backend is
  4.4–6.4× slower than ours.
- **Peak memory is ours across the board**: 185–197 MB against
  358–388 MB for torch and 557–632 MB for tensorflow — a ~2× and
  ~3× gap, fresh process, import included.
- **Task quality lands in the same band for everyone**, as it must with
  identical structure and budget: denoise within 0.14 dB (all 1.4–1.5 dB
  above the noisy input), superres ours/tf above the nearest baseline
  with torch 0.08 dB below it, compress spread 0.4 dB with TF ahead —
  differences of this size are init/shuffle-stream noise (seeded per
  framework; they cannot be made bit-identical across libraries), not
  framework superiority. The engine's metrics match its numpy oracle's
  to within 0.0002 dB on every deterministic task — same model, just
  faster. Adopting the C upsample changed no deterministic-task metric
  by even a printed digit versus the 0.2.2 run: every decoder here
  upsamples by 2, where the C backward's block sum is bit-exact against
  the numpy reduction.
- **The anomaly recipe fails honestly, for every framework** (AUC
  0.06–0.17, far *below* chance 0.5). With digit 1 held out,
  reconstruction-error detection breaks: 1s are the lowest-complexity
  digit, and an autoencoder trained on the other nine reconstructs their
  thin strokes *better* than average — reconstruction MSE is confounded
  with image complexity, a known failure mode of the Sakurada & Yairi
  recipe. The protocol pinned digit 1 before any measurement, so the
  number is reported as measured; all four contenders agree, which is
  exactly what the column is for — it compares frameworks, not the
  recipe's wisdom.

**Fairness caveats.** TF's one-time graph tracing is excluded from fit
timing via an identical untimed warm-up for every contender (as imports
are); torch runs eager, its default mode. CPU only — no MPS/Metal for
anyone. The per-batch denoise corruption uses each framework's native
RNG (numpy / `torch.randn` / a `tf.data` map) — same distribution,
different streams; ours draws it unseeded, so the denoise PSNR of
ours/vanilla-numpy wobbles a few hundredths of a dB between benchmark
runs (14.29 → 14.31 and 14.23 → 14.30 across the last two passes,
vanilla's code unchanged) while the seeded torch/tf values repeat
exactly. Keras is NHWC, so its dense bottleneck connects to a
permuted flatten — identical parameter count and function class. Thread
settings left at each framework's defaults and recorded in the JSON. All
raw samples live in `bench/results/results.json` (regenerable,
gitignored).

**Environment.** Apple M4 · Python 3.9.6 · numpy 2.0.2 · torch 2.8.0 ·
tensorflow 2.20.0 · mantissa 0.2.3 (f32 CNN primitives + the upsample2d
decoder op) · 2026-07-14. Full run: 398 s.
Reproduce: `python -m bench.contenders && python -m bench.speed &&
python -m bench.plots`.
<!-- END:BENCH -->

### Methodology

Identical architectures, subsets, epochs, batch size, learning rate and
seeds for every contender; timings are medians over interleaved repeats on
one machine, library versions recorded in the results JSON. Peak RSS is
measured per contender in a fresh subprocess because that is what a user
pays. *Measure, don't assume.*

## License

MIT — © Tekin Ertekin. Base package:
[mantissa-cnn](https://github.com/tekinertekin/mantissa-cnn); engine:
[mantissa](https://github.com/tekinertekin/mantissa) — same author, MIT.
