"""Evaluation utilities (physical redshift metrics, continuous decoding,
uncertainty / calibration / outlier rejection)."""

from src.eval.redshift_metrics import (
    CATASTROPHIC_DZ,
    decode_redshift,
    redshift_bin_centers,
    redshift_metrics,
)
from src.eval.redshift_uncertainty import (
    DEFAULT_REJECTION_SCORE,
    apply_pit_recalibration,
    apply_temperature,
    coverage_quality_curve,
    fit_pit_recalibrator,
    fit_temperature,
    outlier_auroc,
    pit_calibration_error,
    pit_histogram,
    pit_values,
    redshift_posterior,
    uncertainty_scores,
)

__all__ = [
    "CATASTROPHIC_DZ",
    "decode_redshift",
    "redshift_bin_centers",
    "redshift_metrics",
    "DEFAULT_REJECTION_SCORE",
    "apply_pit_recalibration",
    "apply_temperature",
    "coverage_quality_curve",
    "fit_pit_recalibrator",
    "fit_temperature",
    "outlier_auroc",
    "pit_calibration_error",
    "pit_histogram",
    "pit_values",
    "redshift_posterior",
    "uncertainty_scores",
]
