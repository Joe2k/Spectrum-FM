"""
Load official release checkpoints and run inference.

Checkpoints live under checkpoints/release/ with names matching W&B artifacts.
See checkpoints/release/MANIFEST.json for the registry.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.models.transformer import (
    REDSHIFT_TOKEN_OFFSET,
    SOS_TOKEN,
    SPECTRUM_TOKEN_OFFSET,
    SpectrumTransformer,
)
from src.tokenizers.redshift import RedshiftTokenizer
from src.tokenizers.spectrum import SpectrumTokenizer, N_TOKENS
from src.training.sequences import tokenize_and_build

REPO_ROOT = Path(__file__).resolve().parents[2]
RELEASE_ROOT = REPO_ROOT / "checkpoints" / "release"
MANIFEST_PATH = RELEASE_ROOT / "MANIFEST.json"

DEFAULT_TOKENIZER_ID = "spectrum_tokenizer_v1"
DEFAULT_TRANSFORMER_ID = "transformer_approach_a_fm_v1_10k_ddp4_rw10_v8"


def format_results_table(manifest: Optional[dict] = None) -> str:
    """Human-readable metrics table from MANIFEST (W&B-verified)."""
    manifest = manifest or load_manifest()

    def pct(x) -> str:
        if x is None or x == "":
            return "—"
        return f"{100 * float(x):.1f}%"

    lines = [f"Metrics source: {manifest.get('metrics_source', 'MANIFEST.json')}", ""]
    hdr = f"{'Model':<16} {'run':<26} {'z_TF':>7} {'z_AR':>7} {'spec_TF':>8} {'spec_AR':>8}"
    lines.extend([hdr, "-" * len(hdr)])
    for label, key in [
        ("A release", "approach_a_results"),
        ("B metrics", "approach_b_results"),
    ]:
        row = manifest.get(key, {})
        lines.append(
            f"{label:<16} {row.get('training_run', ''):<26} "
            f"{pct(row.get('val_redshift_acc_teacher_forced')):>7} "
            f"{pct(row.get('val_redshift_acc_autoregressive')):>7} "
            f"{pct(row.get('val_spectrum_acc_teacher_forced')):>8} "
            f"{pct(row.get('val_spectrum_acc_autoregressive')):>8}"
        )
    ref = manifest.get("reference_runs", {}).get("phase10_mask50_a_big")
    if ref:
        lines.append(
            f"{'A phase10 ref':<16} {ref.get('training_run', ''):<26} "
            f"{pct(ref.get('val_redshift_acc_teacher_forced')):>7} "
            f"{pct(ref.get('val_redshift_acc_autoregressive')):>7} "
            f"{pct(ref.get('val_spectrum_acc_teacher_forced')):>8} "
            f"{pct(ref.get('val_spectrum_acc_autoregressive')):>8}"
        )
    return "\n".join(lines)


def load_manifest() -> dict:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"Release manifest not found: {MANIFEST_PATH}\n"
            "Run: python scripts/setup_release_checkpoints.py\n"
            "Then: git lfs pull  (if checkpoints are tracked with LFS)"
        )
    return json.loads(MANIFEST_PATH.read_text())


def list_release_models() -> List[dict]:
    """Return loadable models from MANIFEST (excludes metrics-only entries)."""
    manifest = load_manifest()
    rows = []
    for model_id, info in manifest.get("models", {}).items():
        ckpt = RELEASE_ROOT / info["checkpoint"]
        rows.append({
            "model_id": model_id,
            "display_name": info.get("display_name", model_id),
            "checkpoint": str(ckpt),
            "exists": ckpt.is_file(),
            "wandb_artifact": info.get("wandb_artifact"),
        })
    return rows


def _checkpoint_path(model_id: str) -> Path:
    manifest = load_manifest()
    if model_id not in manifest.get("models", {}):
        raise KeyError(
            f"Unknown model_id {model_id!r}. "
            f"Available: {list(manifest.get('models', {}).keys())}"
        )
    rel = manifest["models"][model_id]["checkpoint"]
    path = (RELEASE_ROOT / rel).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint missing: {path}\n"
            "Run: python scripts/setup_release_checkpoints.py\n"
            "Then: git lfs pull"
        )
    return path


def _load_model_config(model_id: str) -> dict:
    cfg_path = RELEASE_ROOT / model_id / "config.json"
    if cfg_path.is_file():
        return json.loads(cfg_path.read_text())
    manifest = load_manifest()
    return manifest["models"].get(model_id, {})


def _restore_z_tokenizer(ckpt: dict) -> RedshiftTokenizer:
    z_state = ckpt["z_tokenizer"]
    z_tok = RedshiftTokenizer(
        n_levels=int(z_state["n_levels"]),
        gaussian_range=float(z_state.get("gaussian_range", 3.0)),
    )
    # Keep empirical CDF table on CPU (small); encode() moves inputs to match.
    z_tok._sorted_z = z_state["sorted_z"].clone().cpu()
    z_tok._min_z = z_tok._sorted_z[0].item()
    z_tok._max_z = z_tok._sorted_z[-1].item()
    return z_tok


def _infer_transformer_dims(sd: dict) -> dict:
    enc_idx = {int(k.split(".")[1]) for k in sd if k.startswith("encoder_layers.")}
    dec_idx = {int(k.split(".")[1]) for k in sd if k.startswith("decoder_layers.")}
    d_model = int(sd["token_embedding.weight"].shape[1])
    return {
        "d_model": d_model,
        "n_encoder_layers": len(enc_idx),
        "n_decoder_layers": len(dec_idx),
        "n_heads": d_model // 64,
    }


def load_spectrum_tokenizer(
    model_id: str = DEFAULT_TOKENIZER_ID,
    device: Optional[torch.device] = None,
) -> SpectrumTokenizer:
    device = device or torch.device("cpu")
    path = _checkpoint_path(model_id)
    tok = SpectrumTokenizer().to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    tok.load_state_dict(sd)
    tok.eval()
    for p in tok.parameters():
        p.requires_grad_(False)
    return tok


def load_release_models(
    model_id: str = DEFAULT_TRANSFORMER_ID,
    device: Optional[torch.device] = None,
) -> Tuple[SpectrumTransformer, SpectrumTokenizer, RedshiftTokenizer, dict]:
    """
    Load transformer + paired spectrum tokenizer + redshift tokenizer.

    Returns:
        (model, spec_tok, z_tok, config)
    """
    device = device or torch.device("cpu")
    cfg = _load_model_config(model_id)
    tokenizer_id = cfg.get("requires_tokenizer", DEFAULT_TOKENIZER_ID)

    spec_tok = load_spectrum_tokenizer(tokenizer_id, device=device)

    path = _checkpoint_path(model_id)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt

    dims = _infer_transformer_dims(sd)
    model = SpectrumTransformer(
        d_model=cfg.get("d_model", dims["d_model"]),
        n_encoder_layers=cfg.get("n_encoder_layers", dims["n_encoder_layers"]),
        n_decoder_layers=cfg.get("n_decoder_layers", dims["n_decoder_layers"]),
        n_heads=cfg.get("n_heads", dims["n_heads"]),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(sd)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    if isinstance(ckpt, dict) and ckpt.get("z_tokenizer") is not None:
        z_tok = _restore_z_tokenizer(ckpt)
    else:
        raise ValueError(
            f"Checkpoint {path} has no embedded z_tokenizer; "
            "cannot decode redshift predictions."
        )

    out_cfg = {
        **cfg,
        "model_id": model_id,
        "approach": ckpt.get("approach", cfg.get("approach", "a")) if isinstance(ckpt, dict) else "a",
        "transformer_checkpoint": str(path),
        "tokenizer_checkpoint": str(_checkpoint_path(tokenizer_id)),
    }
    return model, spec_tok, z_tok, out_cfg


def numpy_spectrum_batch(
    flux: np.ndarray,
    ivar: np.ndarray,
    z: Optional[np.ndarray] = None,
) -> dict:
    """
    Build a batch dict from numpy arrays for inference.

    Args:
        flux: (B, L) or (L,) flux arrays
        ivar: (B, L) or (L,) inverse variance
        z: optional (B,) redshifts for evaluation (use 0.0 if unknown)
    """
    flux = np.asarray(flux, dtype=np.float32)
    ivar = np.asarray(ivar, dtype=np.float32)
    if flux.ndim == 1:
        flux = flux[None, :]
        ivar = ivar[None, :]
    B = flux.shape[0]
    if z is None:
        z = np.zeros(B, dtype=np.float32)
    else:
        z = np.asarray(z, dtype=np.float32).reshape(B)
    return {
        "flux": torch.from_numpy(flux),
        "ivar": torch.from_numpy(ivar),
        "z": torch.from_numpy(z),
    }


@torch.no_grad()
def predict_teacher_forced(
    batch: dict,
    model: SpectrumTransformer,
    spec_tok: SpectrumTokenizer,
    z_tok: RedshiftTokenizer,
    approach: str,
    device: torch.device,
    encoder_mask_ratio: float = 0.0,
    redshift_mask_ratio: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (enc, dec, target, pred, denorm).

    `redshift_mask_ratio` defaults to 1.0: the released model always predicts
    redshift from the spectrum without seeing it (Approach A). Pass 0.0 to
    deliberately give the model the true redshift (e.g. for diagnostics).
    """
    enc, dec, tgt, _ = tokenize_and_build(
        batch, spec_tok, z_tok, approach, device,
        encoder_mask_ratio=encoder_mask_ratio,
        redshift_mask_ratio=redshift_mask_ratio,
    )
    logits, _ = model(enc, dec, targets=tgt)
    pred = logits.argmax(dim=-1)
    flux = batch["flux"].to(device)
    ivar = batch["ivar"].to(device)
    istd = torch.sqrt(ivar.clamp(min=1e-10))
    x = torch.stack([flux, istd], dim=1)
    _, denorm = spec_tok.encode(x)
    return enc, dec, tgt, pred, denorm


@torch.no_grad()
def predict_autoregressive(
    batch: dict,
    model: SpectrumTransformer,
    spec_tok: SpectrumTokenizer,
    z_tok: RedshiftTokenizer,
    approach: str,
    device: torch.device,
    temperature: float = 1.0,
    encoder_mask_ratio: float = 0.0,
    redshift_mask_ratio: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (generated, target).

    `redshift_mask_ratio` defaults to 1.0: z is hidden from the encoder so the
    model generates redshift from the spectrum (then conditions the spectrum on
    its own generated z). Pass 0.0 to give the model the true redshift.
    """
    enc, _, tgt, _ = tokenize_and_build(
        batch, spec_tok, z_tok, approach, device,
        encoder_mask_ratio=encoder_mask_ratio,
        redshift_mask_ratio=redshift_mask_ratio,
    )
    generated = model.generate(
        enc,
        decoder_start_token=SOS_TOKEN,
        max_new_tokens=tgt.shape[1],
        temperature=temperature,
    )
    return generated, tgt


def decode_z_from_token(token_id: int, z_tok: RedshiftTokenizer) -> float:
    fsq_index = int(token_id) - REDSHIFT_TOKEN_OFFSET
    fsq_index = max(0, min(fsq_index, z_tok.n_levels - 1))
    z = z_tok.decode(fsq_index)
    return float(z.reshape(-1)[0].item())


def decode_spectrum_tokens_to_flux(
    spec_token_ids: torch.Tensor,
    spec_tok: SpectrumTokenizer,
    denorm: torch.Tensor,
) -> torch.Tensor:
    """Map global spectrum token IDs to flux (B, L_grid)."""
    lfq_idx = spec_token_ids - SPECTRUM_TOKEN_OFFSET
    valid = (lfq_idx >= 0) & (lfq_idx < 1024)
    lfq_idx = torch.where(valid, lfq_idx, torch.zeros_like(lfq_idx))
    decoded = spec_tok.decode(lfq_idx, denorm)
    return decoded[:, 0]


def get_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
