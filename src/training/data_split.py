"""
Dataset-agnostic split helpers.

`split_records_by_healpix`: partition a manifest (list of dicts, each
with a `coadd` or `healpix` field) into train/val by *record*, not by
row. This avoids same-pointing leakage where spectra from the same
healpix file land in both train and val.

The function is intentionally type-agnostic about records — it accepts
any list of dicts and partitions by index. Caller decides what's a
"record" (here, one record = one healpix coadd file).
"""

from __future__ import annotations

from collections import OrderedDict
from typing import List, Sequence, Tuple

import torch


def split_records_by_healpix(
    records: List[dict],
    holdout_frac: float,
    seed: int = 42,
    stratify_keys: Sequence[str] = ("survey", "program"),
) -> Tuple[List[dict], List[dict]]:
    """Partition records into (train, val) by record (healpix), not by row.

    The split is a SEEDED RANDOM shuffle — never a sequential block — so val
    is a random subset of healpix, not whatever survey/program happens to sort
    last in the manifest. When `stratify_keys` are present on the records
    (the DR1 manifest carries 'survey' and 'program'), the holdout is taken
    independently WITHIN each (survey, program) group, so every category is
    represented in val at ~`holdout_frac` regardless of how imbalanced the
    corpus is (e.g. full DR1, where main-survey healpix vastly outnumber sv1).
    This both prevents a category from dominating val by chance and makes
    per-(survey, program) / per-spectype val metrics trustworthy.

    Records missing the stratify keys fall back to a single random split
    (identical to a plain seeded `randperm` holdout) — back-compatible.

    Args:
        records: list of manifest records (each typically has 'coadd',
            'redrock', 'healpix', and ideally 'survey'/'program' fields).
        holdout_frac: fraction of records (per group) to put in val. In (0, 1).
        seed: torch RNG seed for the permutation.
        stratify_keys: record fields to stratify on. Pass () to disable and
            split the whole list at random.

    Returns:
        (train_records, val_records). Disjoint, original order preserved.
        Each group with >1 record contributes at least one healpix to val.

    Raises:
        ValueError if `holdout_frac` is not in (0, 1) or the split would leave
        an empty train or val set.
    """
    n = len(records)
    if not 0.0 < holdout_frac < 1.0:
        raise ValueError(f"holdout_frac must be in (0,1), got {holdout_frac}")

    # Group record indices by the stratify keys (single None group when the
    # keys are absent or disabled → plain random split).
    groups: "OrderedDict[object, list]" = OrderedDict()
    for i, r in enumerate(records):
        if stratify_keys and all(k in r for k in stratify_keys):
            key = tuple(r[k] for k in stratify_keys)
        else:
            key = None
        groups.setdefault(key, []).append(i)

    g = torch.Generator().manual_seed(seed)
    val_idx = set()
    for key in sorted(groups, key=lambda k: ("",) if k is None else tuple(map(str, k))):
        idxs = groups[key]
        m = len(idxs)
        perm = torch.randperm(m, generator=g).tolist()
        # Hold out at least one healpix from any group with >1 record, so even
        # tiny categories (sv1) appear in val; a 1-record group stays in train.
        n_val_g = max(1, int(round(m * holdout_frac))) if m > 1 else 0
        for p in perm[-n_val_g:] if n_val_g > 0 else []:
            val_idx.add(idxs[p])

    if not val_idx:
        raise ValueError(
            f"holdout_frac={holdout_frac} on {n} records yields an empty val set"
        )
    if len(val_idx) >= n:
        raise ValueError(
            f"holdout_frac={holdout_frac} on {n} records leaves no train set"
        )

    train = [records[i] for i in range(n) if i not in val_idx]
    val = [records[i] for i in range(n) if i in val_idx]
    return train, val
