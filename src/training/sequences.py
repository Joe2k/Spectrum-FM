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
    mask_targets_only: bool = False,
    decouple_masks: bool = False,
    blind_z_frac: float = 0.5,
    recon_z_shown_frac: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Convert a raw spectrum batch into transformer-ready sequences.

    Args:
        raw_batch: dict with "flux" (B, L), "ivar" (B, L), "z" (B,) tensors
            (produced by `collate_dr1_skip_none`), OR the pre-tokenized form
            with "spec_indices" (B, n_tokens) raw codes + "z" (B,) from
            `collate_cached_skip_none` (the X1 cache) — in which case the
            spectrum encode is skipped and `spec_tok` may be None.
        spec_tok: a frozen `SpectrumTokenizer` (eval mode); unused/None when
            `raw_batch` carries "spec_indices".
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
        mask_targets_only: MAE-style spectrum objective. When True, the
            spectrum reconstruction is supervised ONLY at the positions the
            encoder masked — the target at the *un*masked spectrum positions
            is set to -100 (ignore_index), so the model can't earn reward by
            copying visible tokens across cross-attention. The redshift token
            (position 0) and EOS stay supervised. Requires `encoder_mask_ratio
            > 0` to have any effect; with no encoder masking `masked_positions`
            is None and this is a no-op. Default False = supervise every
            spectrum position (legacy behavior).
        decouple_masks: Approach-A only. When True, replace the two
            INDEPENDENT masks (which let a "predict-z" example also have a
            half-masked spectrum) with a per-sample three-mode split:
              * Z-task (prob `blind_z_frac`): encoder spectrum UNMASKED,
                redshift -> REDMASK. Redshift is supervised (position 0);
                the spectrum block is all -100 (nothing to reconstruct).
                z is supervised ONLY in this mode, always from a full
                spectrum — matching inference-time blind-z.
              * Recon+z-shown (prob `(1-blind_z_frac)*recon_z_shown_frac`):
                encoder spectrum masked per `encoder_mask_ratio`, redshift
                token SHOWN (true z). Spectrum supervised; position-0
                redshift target -> -100 (no trivial z-copy gradient).
              * Recon+z-hidden (the remainder): encoder spectrum masked,
                redshift -> REDMASK. Spectrum supervised; position-0 -> -100.
            Requires `mask_targets_only=True` and `encoder_mask_ratio>0` for
            the recon modes to supervise anything. Ignored for Approach B.
            Default False = legacy independent masks.

            Note (2026-06-30): a "surgical alignment" variant that kept
            the true z at position-0 on the recon+z-shown rows (restoring
            the trivial-copy z-supervision) was implemented and tested in
            run abmask_dec_v2_zv2 (W&B 42bj6sck). It did NOT recover V3/V4
            parity: at step 4000 val_z/z_nmad 0.084, val_z/redshift_acc
            0.0013, train/redshift_acc still essentially 0 — the same
            trajectory as the pure DECOUPLE run (ot51pk0s). The trivial
            z-copy loss is too close to 0 to materially change the
            encoder's gradient; the Z-task rows still have no spectrum
            gradient, so the encoder still builds mode-specialized
            representations. See the 2026-06-30 RESEARCH_LOG entry for
            the full negative result.
        blind_z_frac: fraction of samples that are the Z-task (full
            spectrum, predict z). Only used when `decouple_masks=True`.
        recon_z_shown_frac: within the recon samples, fraction that SHOW the
            true z to the encoder. 0.0 => z hidden everywhere (pure
            foundation-model "always predict z, never given" objective);
            recon eval (z hidden) is then exactly in-distribution. Only used
            when `decouple_masks=True`.

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
    if decouple_masks:
        if not 0.0 <= blind_z_frac <= 1.0:
            raise ValueError(f"blind_z_frac must be in [0, 1], got {blind_z_frac}")
        if not 0.0 <= recon_z_shown_frac <= 1.0:
            raise ValueError(f"recon_z_shown_frac must be in [0, 1], got {recon_z_shown_frac}")

    z_vals = raw_batch["z"]  # may stay on CPU; encode is per-item

    if "spec_indices" in raw_batch:
        # Pre-tokenized path (X1 cache): the frozen tokenizer was run offline,
        # so the ConvNeXt encode is skipped entirely. spec_tok may be None.
        # Cached indices are the raw codes (0..1023), same as encode() returns.
        spec_indices = raw_batch["spec_indices"].to(device)
    else:
        flux = raw_batch["flux"].to(device, non_blocking=True)
        ivar = raw_batch["ivar"].to(device, non_blocking=True)
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

    def _rand(*shape):
        if rng is None:
            return torch.rand(*shape, device=device)
        return torch.rand(*shape, device=device, generator=rng)

    # Apply masking to the encoder's spectrum tokens only.
    masked_positions: Optional[torch.Tensor] = None
    spec_tokens_enc = spec_tokens
    is_recon: Optional[torch.Tensor] = None  # set in decoupled mode; (B,1) bool

    if decouple_masks and approach == "a":
        # Three-mode per-sample split (see docstring). z is supervised ONLY in
        # the Z-task (full spectrum); the recon modes supervise the spectrum.
        is_ztask = _rand(B, 1) < blind_z_frac          # full spectrum, predict z
        is_recon = ~is_ztask
        # Within recon, whether the encoder is shown the TRUE z.
        show_z = is_recon & (_rand(B, 1) < recon_z_shown_frac)

        # Spectrum mask: only recon rows get masked; Z-task rows stay full.
        if encoder_mask_ratio > 0.0:
            mask = (_rand(B, T_spec) < encoder_mask_ratio) & is_recon  # (B,1)->(B,T)
            masked_positions = mask
            spec_tokens_enc = torch.where(
                mask, torch.full_like(spec_tokens, MASK_TOKEN), spec_tokens)

        # Redshift visible to the encoder only when (recon AND show_z).
        rz_enc = torch.where(~show_z, torch.full_like(rz, REDMASK_TOKEN), rz)
    else:
        # Legacy: the two masks are drawn INDEPENDENTLY.
        if encoder_mask_ratio > 0.0:
            mask = _rand(B, T_spec) < encoder_mask_ratio
            masked_positions = mask
            spec_tokens_enc = torch.where(
                mask, torch.full_like(spec_tokens, MASK_TOKEN), spec_tokens)

        # Redshift conditioning dropout (Approach A only): replace the encoder's
        # redshift token with REDMASK so the model must infer z from the
        # spectrum rather than copy it. decoder_input/target keep the true rz.
        rz_enc = rz
        if approach == "a" and redshift_mask_ratio > 0.0:
            rmask = _rand(B, 1) < redshift_mask_ratio
            rz_enc = torch.where(rmask, torch.full_like(rz, REDMASK_TOKEN), rz)

    if approach == "a":
        encoder_input = torch.cat([sos, rz_enc, spec_tokens_enc, eos], dim=1)
    else:  # 'b'
        encoder_input = torch.cat([sos, spec_tokens_enc, eos], dim=1)

    # Decoder input and target use the UNMASKED spec_tokens.
    decoder_input = torch.cat([sos, rz, spec_tokens], dim=1)
    target = torch.cat([rz, spec_tokens, eos], dim=1)

    # MAE-style objective: supervise only the encoder-masked spectrum
    # positions. The spectrum block sits at target indices [1, 1+T_spec);
    # set the unmasked ones to -100 so the loss/metrics ignore them. The
    # redshift token (index 0) and EOS (index 1+T_spec) stay supervised.
    if mask_targets_only and masked_positions is not None:
        spec_target = target[:, 1:1 + T_spec]
        spec_target[~masked_positions] = -100

    if decouple_masks and is_recon is not None:
        # Supervise z (position 0) ONLY in the Z-task rows (full spectrum).
        # Recon rows — whether z was shown or hidden — get position-0 -> -100,
        # so the redshift gradient never comes from a half-masked spectrum.
        target[is_recon.squeeze(1), 0] = -100

    return encoder_input, decoder_input, target, masked_positions


def lr_at(step: int, base_lr: float, warmup: int, total: int) -> float:
    """Linear warmup -> cosine decay to 1/10 of base."""
    import math
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))
