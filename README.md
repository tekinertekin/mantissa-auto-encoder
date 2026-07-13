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
run in C on zero-copy float32 buffers; the loss and the nearest-neighbor
upsample are memory-bound array movement and honestly stay in numpy. Layers
allocate their scratch once per batch shape and reuse it — steady-state
training does no per-batch allocation.

## Install

```sh
pip install mantissa-autoencoder   # after PyPI publication
```

This pulls in `mantissa-cnn >= 0.1.0` (which pulls the engine
`mantissa-nn >= 0.2.1`).

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

<img src="assets/concepts/autoencoder.png" width="380" alt="autoencoder schema: input through encoder to a code, decoder reconstructs the input">

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

<!-- BEGIN:BENCH (bench harness output will replace this block; do not edit outside the markers) -->
**Coming.** The protocol is fixed before the first number is measured
(constants pinned in `bench/protocol.py`, recipes in
`mantissa_autoencoder.tasks.TASKS`):

- **Four tasks** — denoise (fashion_mnist, Gaussian σ 0.3), compress
  (mnist, 32-float code, plus the uint8-quantized 40-byte variant),
  anomaly (mnist, digit 1 held out of training), superres (mnist,
  28 → 14 → 28).
- **Contenders** — ours (mantissa engine), ours (numpy backend), torch,
  tensorflow; the same architecture re-expressed layer-for-layer in each,
  CPU only.
- **Budget** — stratified 2000-train / 1000-test subsets, seed 0; 5 epochs,
  batch 32, lr 0.01, plain SGD everywhere; 5 interleaved repeats, medians
  reported.
- **Metrics** — fit wall-time; the task metric (PSNR for denoise /
  compress / superres, AUC for anomaly); peak RSS in a fresh subprocess
  per contender, import cost included.
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
