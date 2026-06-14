"""
DR1TokenizedDataset
===================
Wraps DR1IndexedDataset and produces transformer-ready sequences:
(encoder_input, decoder_input, target) per Approach A or B.

Mirrors src.datasets.tokenized_dataset.TokenizedSpectrumDataset but
reads spectra on demand from the manifest instead of holding everything
in memory.

The spectrum tokenizer is run in eval mode with no grad. The redshift
tokenizer must be `fit()` on a representative redshift sample before
this dataset is used.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from dr1_dataset import DR1IndexedDataset  # noqa: E402
from src.tokenizers.spectrum import SpectrumTokenizer  # noqa: E402
from src.tokenizers.redshift import RedshiftTokenizer  # noqa: E402
from src.models.transformer import (  # noqa: E402
    PAD_TOKEN,
    build_approach_a_sequences,
    build_approach_b_sequences,
    encode_redshift_token,
    encode_spectrum_token,
)


class DR1TokenizedDataset(Dataset):
    """Per-item tokenization of a DR1IndexedDataset.

    Args:
        base: an underlying DR1IndexedDataset.
        spectrum_tokenizer: a SpectrumTokenizer in eval mode (caller's
            responsibility to load weights and set requires_grad=False).
        redshift_tokenizer: a fitted RedshiftTokenizer.
        approach: 'a' (joint) or 'b' (masked redshift).
        device: device on which to run the tokenizer encode pass.
            Tokenized outputs are returned on CPU for the DataLoader
            to pin/transfer.
    """

    def __init__(
        self,
        base: DR1IndexedDataset,
        spectrum_tokenizer: SpectrumTokenizer,
        redshift_tokenizer: RedshiftTokenizer,
        approach: str = "a",
        device: torch.device = torch.device("cpu"),
    ):
        approach = approach.lower()
        assert approach in ("a", "b"), f"approach must be 'a' or 'b', got {approach!r}"
        self.base = base
        self.spectrum_tokenizer = spectrum_tokenizer
        self.redshift_tokenizer = redshift_tokenizer
        self.approach = approach
        self.device = device

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        spec = self.base[idx]
        if spec is None:
            return None

        flux = spec["flux"].unsqueeze(0).to(self.device, non_blocking=True)
        ivar = spec["ivar"].unsqueeze(0).to(self.device, non_blocking=True)
        istd = torch.sqrt(ivar.clamp(min=1e-10))
        x = torch.stack([flux, istd], dim=1)

        with torch.no_grad():
            indices, _ = self.spectrum_tokenizer.encode(x)

        spectrum_tokens = encode_spectrum_token(indices.squeeze(0)).cpu()

        z = float(spec["z"])
        redshift_idx = self.redshift_tokenizer.encode(z)
        redshift_token = encode_redshift_token(redshift_idx).cpu()

        if self.approach == "a":
            enc, dec_in, target = build_approach_a_sequences(redshift_token, spectrum_tokens)
        else:
            enc, dec_in, target = build_approach_b_sequences(redshift_token, spectrum_tokens)

        return {
            "encoder_input": enc,
            "decoder_input": dec_in,
            "target": target,
            "redshift": torch.tensor(z, dtype=torch.float32),
        }


def collate_tokenized_skip_none(batch):
    """Drop None entries (filtered rows), pad sequences, return dict of tensors."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    n = len(batch)
    max_enc = max(len(b["encoder_input"]) for b in batch)
    max_dec = max(len(b["decoder_input"]) for b in batch)
    max_tgt = max(len(b["target"]) for b in batch)

    encoder_input = torch.full((n, max_enc), PAD_TOKEN, dtype=torch.long)
    encoder_mask = torch.zeros(n, max_enc, dtype=torch.bool)  # True = padding
    decoder_input = torch.full((n, max_dec), PAD_TOKEN, dtype=torch.long)
    decoder_mask = torch.zeros(n, max_dec, dtype=torch.bool)
    target = torch.full((n, max_tgt), -100, dtype=torch.long)

    for i, b in enumerate(batch):
        le, ld, lt = len(b["encoder_input"]), len(b["decoder_input"]), len(b["target"])
        encoder_input[i, :le] = b["encoder_input"]
        encoder_mask[i, le:] = True
        decoder_input[i, :ld] = b["decoder_input"]
        decoder_mask[i, ld:] = True
        target[i, :lt] = b["target"]

    return {
        "encoder_input": encoder_input,
        "decoder_input": decoder_input,
        "target": target,
        "encoder_mask": encoder_mask,
        "decoder_mask": decoder_mask,
        "redshift": torch.stack([b["redshift"] for b in batch]),
    }


class DR1CachedTokenDataset(Dataset):
    """Reads pre-tokenized shards written by `pretokenize_corpus.py` (X1).

    Each shard is one (survey, program, healpix) `.npz` with parallel arrays
    over the coadd's rows: `indices` (N, n_tokens) raw codes 0..1023, `z`,
    `denorm`, `zwarn`, `fiberstatus`, `nonzero_flux`, `spectype`, `targetid`,
    `row`, plus scalar `survey`/`program`/`healpix`. This makes every
    transformer step a fast array read instead of a ConvNeXt encode — the
    item carries the RAW codes, so `tokenize_and_build` adds the offset and
    builds sequences exactly as on the on-the-fly path.

    The quality filter (zwarn / fiberstatus / nonzero-flux) is applied from
    the STORED flags at index-build time, so the default read reproduces the
    training cut even when the shards were written with `--quality all`.

    Args:
        shards: a directory (globs `*.npz`) or an explicit list of shard paths.
        require_good_zwarn: keep only rows with zwarn == 0 and fiberstatus == 0.
        require_nonzero_flux: keep only rows flagged nonzero-flux.
        max_spectra: cap total kept rows (smoke).
        cache_size: number of recently-touched shards kept in memory.
    """

    def __init__(
        self,
        shards,
        require_good_zwarn: bool = True,
        require_nonzero_flux: bool = True,
        max_spectra=None,
        cache_size: int = 2,
    ):
        import numpy as np

        if isinstance(shards, (str, Path)):
            shard_paths = sorted(Path(shards).glob("*.npz"))
        else:
            shard_paths = [Path(s) for s in shards]
        if not shard_paths:
            raise FileNotFoundError(f"no .npz shards found under {shards!r}")
        self.shard_paths = shard_paths
        self.cache_size = max(1, cache_size)

        self.flat_index = []                 # (shard_idx, row_in_shard)
        self.shard_meta = []                 # (survey, program, healpix) per shard
        stop = False
        for si, p in enumerate(shard_paths):
            with np.load(p, allow_pickle=False) as d:
                n = int(d["indices"].shape[0])
                zwarn = d["zwarn"] if "zwarn" in d.files else np.zeros(n, np.int16)
                fstat = d["fiberstatus"] if "fiberstatus" in d.files else np.zeros(n, np.int32)
                nzf = d["nonzero_flux"] if "nonzero_flux" in d.files else np.ones(n, np.int8)
                self.shard_meta.append((
                    str(d["survey"]) if "survey" in d.files else "unknown",
                    str(d["program"]) if "program" in d.files else "unknown",
                    int(d["healpix"]) if "healpix" in d.files else -1,
                ))
            keep = np.ones(n, dtype=bool)
            if require_good_zwarn:
                keep &= (zwarn == 0) & (fstat == 0)
            if require_nonzero_flux:
                keep &= nzf.astype(bool)
            for r in np.nonzero(keep)[0]:
                self.flat_index.append((si, int(r)))
                if max_spectra is not None and len(self.flat_index) >= max_spectra:
                    stop = True
                    break
            if stop:
                break

        self._cache: dict = {}
        self._order: list = []

    def __len__(self):
        return len(self.flat_index)

    def meta_for_index(self, idx):
        """(survey, program) for a flat index — parity with the audit tooling."""
        si, _ = self.flat_index[idx]
        s, p, _ = self.shard_meta[si]
        return (s, p)

    def _load(self, si):
        import numpy as np

        if si in self._cache:
            return self._cache[si]
        if len(self._order) >= self.cache_size:
            ev = self._order.pop(0)
            self._cache.pop(ev, None)
        with np.load(self.shard_paths[si], allow_pickle=False) as d:
            data = {
                "indices": d["indices"],
                "z": d["z"],
                "spectype": d["spectype"] if "spectype" in d.files else None,
                "targetid": d["targetid"] if "targetid" in d.files else None,
            }
        self._cache[si] = data
        self._order.append(si)
        return data

    def __getitem__(self, idx):
        import numpy as np

        si, r = self.flat_index[idx]
        d = self._load(si)
        item = {
            "spec_indices": torch.from_numpy(d["indices"][r].astype(np.int64)),
            "z": torch.tensor(float(d["z"][r]), dtype=torch.float32),
        }
        if d["spectype"] is not None:
            item["spectype"] = str(d["spectype"][r])
        if d["targetid"] is not None:
            item["targetid"] = int(d["targetid"][r])
        return item


def collate_cached_skip_none(batch):
    """Stack pre-tokenized items. spec_indices is fixed width, so no padding."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return {
        "spec_indices": torch.stack([b["spec_indices"] for b in batch]),
        "z": torch.stack([b["z"] for b in batch]),
    }


def collect_redshifts_from_cache(shards, max_files=None):
    """Pull good-zwarn redshifts from pre-tokenized shards to fit RedshiftTokenizer.

    Avoids reopening redrock FITS when training from the X1 cache.
    """
    import numpy as np

    if isinstance(shards, (str, Path)):
        shard_paths = sorted(Path(shards).glob("*.npz"))
    else:
        shard_paths = [Path(s) for s in shards]
    n_scan = len(shard_paths) if max_files is None else min(max_files, len(shard_paths))
    zs = []
    for p in shard_paths[:n_scan]:
        with np.load(p, allow_pickle=False) as d:
            z = d["z"]
            zwarn = d["zwarn"] if "zwarn" in d.files else np.zeros(len(z), np.int16)
            zs.append(z[zwarn == 0].astype("float32").copy())
    if not zs:
        return torch.zeros(0)
    return torch.from_numpy(np.concatenate(zs))


def collect_redshifts(records, max_files=None):
    """Open redrock files in a manifest and pull all Z values.

    Used to fit RedshiftTokenizer before training. Cheap -- only reads
    the REDSHIFTS table's Z column.

    Args:
        records: list of manifest records (from load_manifest)
        max_files: optionally cap how many files to scan (subsample)

    Returns:
        torch.Tensor of redshifts (1D)
    """
    from astropy.io import fits

    zs = []
    n_scan = len(records) if max_files is None else min(max_files, len(records))
    for rec in records[:n_scan]:
        with fits.open(rec["redrock"], memmap=True) as h:
            z = h["REDSHIFTS"].data["Z"]
            zwarn = h["REDSHIFTS"].data["ZWARN"]
            good = zwarn == 0
            zs.append(z[good].astype("float32").copy())
    if not zs:
        return torch.zeros(0)
    import numpy as np
    return torch.from_numpy(np.concatenate(zs))
