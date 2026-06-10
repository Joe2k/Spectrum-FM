"""
Sequence construction for transformer training.

Given a raw batch (flux, ivar, z) — produced by `DR1IndexedDataset` or
any equivalent dataset — tokenize spectra (frozen `SpectrumTokenizer`)
and redshifts (`RedshiftTokenizer`), then build encoder/decoder/target
sequences for Approach A (encoder sees redshift) or B (encoder masked
of redshift).

Also supports BERT-style encoder masking: replace a fraction of the
encoder's spectrum tokens with `MASK_TOKEN` before the model sees them,
without changing the decoder input or target. Returns the boolean
masked-positions tensor so eval/metrics can report accuracy on the
masked positions only (the honest spectrum-reconstruction number).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from src.models.transformer import (
    EOS_TOKEN,
    MASK_TOKEN,
    REDMASK_TOKEN,
    REDSHIFT_TOKEN_OFFSET,
    SOS_TOKEN,
    SPECTRUM_TOKEN_OFFSET,
)


def tokenize_and_build(
    raw_batch: dict,
    spec_tok,
    z_tok,
    approach: str,
    device: torch.device,
    encoder_mask_ratio: float = 0.0,
    redshift_mask_ratio: float = 0.0,
    rng: Optional[torch.Generator] = None,
    wavelength_aware: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Convert a raw spectrum batch into transformer-ready sequences.

    Args:
        raw_batch: dict with "flux" (B, L), "ivar" (B, L), "z" (B,) tensors.
            Produced by `collate_dr1_skip_none` or equivalent.
        spec_tok: a frozen `SpectrumTokenizer` (eval mode).
        z_tok: a fitted `RedshiftTokenizer`.
        approach: 'a' (encoder sees redshift) or 'b' (encoder does not).
        device: where the spectrum tokenizer + transformer live.
        encoder_mask_ratio: fraction of encoder spectrum positions to
            replace with `MASK_TOKEN` (BERT-style). Decoder input and
            target are NOT modified. Default 0.0 = no masking.
        redshift_mask_ratio: probability (per sample) of replacing the
            encoder's redshift token with `REDMASK_TOKEN` — redshift
            conditioning dropout. Forces the model to predict redshift
            from the spectrum instead of copying it from the encoder.
            Decoder input and target keep the TRUE redshift token, so
            position-0 is still supervised against the real value.
            Train with a value in (0, 1) (e.g. 0.5); inference passes
            1.0 (always hide z). Ignored for Approach B (encoder has no
            redshift). Default 0.0 = z always visible (no dropout).
        rng: optional `torch.Generator` for reproducible masking.
        wavelength_aware: pass the batch's "wavelength" array to the
            spectrum tokenizer so it resamples onto the fixed wavelength
            grid instead of length-stretching. Use ONLY with a tokenizer
            trained wavelength-aware (v2+); v1 expects the legacy stretch.

    Returns:
        encoder_input: (B, L_enc) long tensor.
        decoder_input: (B, L_dec) long tensor (teacher-forced).
        target: (B, L_dec) long tensor.
        masked_positions: (B, T_spec) bool tensor — True where the
            encoder's spectrum tokens were replaced by MASK. None if
            `encoder_mask_ratio == 0.0`. T_spec = number of spectrum
            positions per sequence (e.g. 272 from the tokenizer).
    """
    if approach not in ("a", "b"):
        raise ValueError(f"approach must be 'a' or 'b', got {approach!r}")
    if not 0.0 <= encoder_mask_ratio <= 1.0:
        raise ValueError(f"encoder_mask_ratio must be in [0, 1], got {encoder_mask_ratio}")
    if not 0.0 <= redshift_mask_ratio <= 1.0:
        raise ValueError(f"redshift_mask_ratio must be in [0, 1], got {redshift_mask_ratio}")

    flux = raw_batch["flux"].to(device, non_blocking=True)
    ivar = raw_batch["ivar"].to(device, non_blocking=True)
    z_vals = raw_batch["z"]  # may stay on CPU; encode is per-item

    istd = torch.sqrt(ivar.clamp(min=1e-10))
    x = torch.stack([flux, istd], dim=1)  # (B, 2, L)

    wave = None
    if wavelength_aware and "wavelength" in raw_batch:
        wave = raw_batch["wavelength"].to(device, non_blocking=True)

    with torch.no_grad():
        # Only pass the kwarg when set, so legacy tokenizer wrappers/stubs
        # without a `wavelength` parameter keep working.
        if wave is not None:
            spec_indices, _ = spec_tok.encode(x, wavelength=wave)
        else:
            spec_indices, _ = spec_tok.encode(x)  # (B, n_tokens) or (B, 1, n_tokens)
    if spec_indices.dim() == 3:
        spec_indices = spec_indices.squeeze(1)
    spec_tokens = spec_indices.long() + SPECTRUM_TOKEN_OFFSET  # (B, T_spec)

    z_batch = z_vals.float().to(device=z_tok.device)
    redshift_idx = z_tok.encode(z_batch).long().to(device)
    redshift_tokens = redshift_idx + REDSHIFT_TOKEN_OFFSET  # (B,)

    B, T_spec = spec_tokens.shape
    sos = torch.full((B, 1), SOS_TOKEN, dtype=torch.long, device=device)
    eos = torch.full((B, 1), EOS_TOKEN, dtype=torch.long, device=device)
    rz = redshift_tokens.unsqueeze(1)  # (B, 1)

    # Apply masking to the encoder's spectrum tokens only.
    masked_positions: Optional[torch.Tensor] = None
    spec_tokens_enc = spec_tokens
    if encoder_mask_ratio > 0.0:
        if rng is None:
            mask = torch.rand(B, T_spec, device=device) < encoder_mask_ratio
        else:
            mask = torch.rand(B, T_spec, device=device, generator=rng) < encoder_mask_ratio
        masked_positions = mask
        spec_tokens_enc = torch.where(
            mask,
            torch.full_like(spec_tokens, MASK_TOKEN),
            spec_tokens,
        )

    # Redshift conditioning dropout (Approach A only): replace the encoder's
    # redshift token with REDMASK so the model must infer z from the spectrum
    # rather than copy it. decoder_input/target keep the true rz below.
    rz_enc = rz
    if approach == "a" and redshift_mask_ratio > 0.0:
        if rng is None:
            rmask = torch.rand(B, 1, device=device) < redshift_mask_ratio
        else:
            rmask = torch.rand(B, 1, device=device, generator=rng) < redshift_mask_ratio
        rz_enc = torch.where(rmask, torch.full_like(rz, REDMASK_TOKEN), rz)

    if approach == "a":
        encoder_input = torch.cat([sos, rz_enc, spec_tokens_enc, eos], dim=1)
    else:  # 'b'
        encoder_input = torch.cat([sos, spec_tokens_enc, eos], dim=1)

    # Decoder input and target use the UNMASKED spec_tokens.
    decoder_input = torch.cat([sos, rz, spec_tokens], dim=1)
    target = torch.cat([rz, spec_tokens, eos], dim=1)

    return encoder_input, decoder_input, target, masked_positions


def lr_at(step: int, base_lr: float, warmup: int, total: int) -> float:
    """Linear warmup -> cosine decay to 1/10 of base."""
    import math
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))
