# Spectrum Foundation Model — Update Slides (2026-06-17)

Paste-ready slide content. Each `---` block is one slide: a heading, a short
description written out in full (no abbreviations), and a placeholder marking
where to drop the figure. Source figures come from
`notebooks/07_visualize_predictions.ipynb`.

---

## Slide — Masked Spectrum Reconstruction on DESI Spectra (In-Distribution)

**What this shows.** A test of whether the model has genuinely learned spectral
structure rather than copying it. We hide a fraction of the spectrum tokens from
the encoder (50 percent, then 90 percent) and ask the model to reconstruct the
hidden regions. Because the hidden tokens are absent from the encoder, the model
cannot copy them through cross-attention — it has to infer them.

**How to read it.** Each spectrum has three stacked panels:
- Top: the true spectrum (blue, after the tokenizer round trip) versus the
  model's blind reconstruction (red). Grey shading marks the regions hidden from
  the encoder.
- Middle: the residual (true minus reconstruction).
- Bottom: the raw measurement noise that comes with the observation
  (one standard deviation per pixel, computed from the inverse variance).

**Why it matters.** If the residual stays within the raw noise band, the
reconstruction is honest. Comparing 50 percent to 90 percent masking shows how
much real structure the model recovers as it is given fewer and fewer anchors.

_[Add plots: DESI blind reconstruction panels at 50 percent and 90 percent masking]_

---

## Slide — Redshift Prediction on DESI Spectra (Full Spectrum, No Masking)

**What this shows.** The core deliverable of a spectrum foundation model: given a
full observed spectrum with no masking, how accurately can it measure the
redshift? The encoder sees the entire spectrum; only the redshift itself is
withheld, because that is the quantity being predicted.

**Sample.** Evaluated over the entire set of local DESI spectra.

**How to read it.**
- Left: predicted redshift versus true redshift (points on the diagonal are
  correct).
- Right: the distribution of the normalized redshift error, defined as the
  difference between predicted and true redshift divided by one plus the true
  redshift.

**Headline metrics.**
- Normalized Median Absolute Deviation of the redshift error — a robust measure
  of scatter.
- Catastrophic outlier fraction — the fraction of spectra whose normalized
  redshift error exceeds a fixed threshold.
- Note: with 256 redshift bins the quantization step sets a hard resolution
  floor; a finer redshift head (4096 bins) is planned to push below it.

_[Add plots: DESI redshift scatter and normalized-error histogram]_

---

## Slide — Masked Spectrum Reconstruction on SDSS Spectra (Out-of-Distribution)

**What this shows.** The same blind reconstruction test, but on spectra from a
different instrument (the Sloan Digital Sky Survey), which the model never trained
on. SDSS spectra use a different wavelength sampling and lower resolution, and are
resampled onto the DESI wavelength grid before being passed through the model.

**How to read it.** Three stacked panels per spectrum, as in the DESI case:
- Top: true spectrum (blue) versus blind reconstruction (red). Orange shading
  marks the regions hidden from the encoder; grey marks wavelengths outside the
  SDSS coverage.
- Middle: the residual.
- Bottom: the raw measurement noise (one standard deviation per pixel).

**Why it matters.** This probes generalization: can the model infer hidden
structure on out-of-distribution data, or does it fall back to a generic
in-distribution prior? The masking levels (50 percent and 90 percent) show how
that ability degrades as context is removed.

_[Add plots: SDSS blind reconstruction panels at 50 percent and 90 percent masking]_

---

## Slide — Redshift Prediction on SDSS Spectra (Full Spectrum, No Masking)

**What this shows.** Out-of-distribution redshift measurement: the full SDSS
spectrum is given to the encoder with no masking, and only the redshift is
withheld so the model must measure it from the spectrum alone.

**Sample.** Evaluated over one hundred SDSS spectra.

**How to read it.**
- Left: predicted redshift versus true redshift.
- Right: the distribution of the normalized redshift error.

**Headline metrics.** Same as for DESI — Normalized Median Absolute Deviation and
catastrophic outlier fraction. The difference between the SDSS and DESI numbers
quantifies the cost of applying the model to a new instrument.

_[Add plots: SDSS redshift scatter and normalized-error histogram]_

---

## Slide — Pre-Tokenizing the Corpus with the Frozen Tokenizer

**What we are doing.** The spectrum tokenizer (a convolutional encoder paired with
a lookup-free quantizer) is now frozen, so the discrete tokens it produces for any
given spectrum never change. That lets us run the tokenizer once over the entire
data set and cache the resulting tokens per spectrum, instead of recomputing them
on every training step.

**Why it matters.** Previously every transformer training step paid for a full
convolutional encoder pass plus reading and stitching the raw observation files.
With the cache, each step becomes a small array read. This removes the throughput
bottleneck and makes large batches and faster numerical formats practical —
directly unblocking the next phase of transformer training.

**How it is stored.** One compressed file per combination of survey, program, and
sky region. Each file holds, for every spectrum: the discrete token identifiers
(272 tokens per spectrum), the true redshift, quality flags, and identifiers for
later cross-referencing.

**Scale and status.** Running over the full first data release on the
supercomputing cluster (tens of thousands of sky regions). The process is
resumable across sessions and produces a cache of a few gigabytes, copied to
long-term storage once complete. A built-in verification step re-tokenizes a
random sample and confirms it matches the cache exactly.

_[Optional diagram: raw spectra → frozen tokenizer (run once) → cached token shards → fast transformer training]_

---
