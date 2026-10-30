# Method

Freq-DINO adapts Grounding DINO to few-shot defect detection in dense lithographic
patterns. The implementation is in [`my_custom_module3.py`](../my_custom_module3.py).
The detector uses Swin-B visual features and a BERT text encoder. An explicit FFT
prior enters the four neck feature levels before the detector's multimodal
encoder and decoder.

## Frequency prior

The prior receives the same augmented, channel-normalized, batch-padded tensor as
the visual backbone. The implementation:

1. Projects the three channels to grayscale and subtracts the spatial mean.
2. Computes the 2D FFT and centers the spectrum.
3. Excludes a central disk from the **peak-search magnitude map**.
4. Selects three peak pairs by magnitude. For each selection it suppresses both
   the selected peak and its conjugate counterpart in the complex spectrum,
   then excludes their neighborhoods from subsequent peak searches.
5. Takes the magnitude of the inverse FFT.
6. Measures both lower-tail and upper-tail deviations using quantiles from the
   interior image region, applies a soft border mask, normalizes, smooths with a
   Gaussian kernel, and normalizes again.

Subtracting the mean removes DC. The central search exclusion does **not** zero
the entire central disk of the complex spectrum. Three means three conjugate
peak pairs, not three individual peaks. The selection does not impose fixed
angles or a first-shell constraint.

| Parameter | Default |
|---|---:|
| Peak pairs `freq_num_peaks` | 3 |
| Central search exclusion radius | 30 |
| Peak suppression radius | 8 |
| Search exclusion radius around each selected peak | 11 |
| Lower / upper quantiles | 0.5% / 99.5% |
| Border ratio | 0.03 |
| Gaussian sigma / kernel size | 20 / 121 × 121 |

The radii refer to the processed FFT grid. The code does not convert them to a
physical length. Batch padding is included in the FFT. The prior computation
runs without gradients and requires no heatmap annotations.

## Encoding and fusion

A shared convolutional encoder maps the heatmap through channels
1 → 32 → 64 → 128 → 256 using GroupNorm and LeakyReLU. Bilinear interpolation
matches its output to four visual feature sizes. Each level has a separate
normalization and gate:

```text
frequency = GroupNorm(encoded_prior)
gate = sigmoid(gate_network(concat(visual, frequency)))
fused = visual + gate * frequency
```

The gate is a single-channel spatial weight broadcast over feature channels.
Its final convolution starts with zero weights and bias −2, giving an initial
gate of approximately 0.119. Gate values are fusion weights, not calibrated
defect probabilities.

## Training

The paper configurations freeze the complete Swin-B backbone and BERT encoder.
The neck, cross-modal encoder/decoder, detection heads, projections, query
embeddings, frequency encoder, and fusion modules are trainable. In particular,
`encoder.text_layers` and `text_feat_map` remain trainable after the frozen BERT
encoder. Despite the historical class name `GroundingDINOLoRA`, these
configurations set `lora_r=0` and do not use LoRA.

Each support set is presented through OD and VG views with the same image and
box annotations. This does not create a second independent support set.
Training uses the existing Grounding DINO grounding and box losses, including
L1 and GIoU terms and auxiliary decoder/denoising losses.

The provided training variants are the fine-tuned baseline, FFT prior with
learned gating, FFT prior with direct normalized residual addition, and a
grayscale-prior control. See [EXPERIMENTS.md](EXPERIMENTS.md) for the commands and
the distinction between saved AP-selected checkpoints and AP50-selected
validation epochs.
