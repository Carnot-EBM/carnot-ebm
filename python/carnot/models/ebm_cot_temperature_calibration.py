"""Post-hoc temperature calibration for Exp 1401 EBM-CoT scores.

The Exp 1401 hinge-only probe improved AUROC but made paraphrase energy deltas
more spread out.  This module keeps the trained ranking fixed and fits one
positive scalar temperature on a validation split, then applies that same scale
to held-out EBM-CoT energies.  Positive scalar division preserves ranking, so
AUROC should remain unchanged while the measured energy-delta variance contracts
by the same temperature scale.

Spec: REQ-VERIFY-1416, SCENARIO-VERIFY-1416
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from carnot.eval.metrics import auroc
from carnot.models.ebm_cot_energy_calibration_probe import (
    DEFAULT_EXP1401_ARTIFACT_PATH,
    DEFAULT_FOVER_PATH,
    DEFAULT_HINGE_MARGIN,
    DEFAULT_MODELS_DIR,
    DEFAULT_N_EPOCHS,
    EBMCoTKANEnergyCalibrator,
    EXP1384_SPLIT_SEED,
    EXP1401_RUN_DATE,
    FoVerSplit,
    FoVerStepCase,
    HINGE_ONLY_CONSISTENCY_WEIGHT,
    load_fover_verified_cases,
    make_balanced_split,
    paraphrase_positive_step,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ARTIFACT_PATH = (
    REPO_ROOT / "results" / "experiment_1416_ebm_cot_v3_temperature_calibration.json"
)
DEFAULT_TEMPERATURE_CANDIDATES = (0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0)
DEFAULT_VALIDATION_FRACTION = 0.20
DEFAULT_AUROC_TOLERANCE = 1e-12
DEFAULT_VARIANCE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class FittedTemperature:
    """Validation-selected scalar temperature and its calibration losses.

    The validation losses are kept in the result artifact so reviewers can see
    that the selected temperature came from the validation split rather than
    from the held-out test scores.

    Spec: REQ-VERIFY-1416
    """

    best_temperature: float
    validation_losses: dict[float, float]


@dataclass(frozen=True)
class TemperatureCalibrationScores:
    """Precomputed validation/test EBM-CoT energies for temperature scaling.

    Energies use the EBM convention: lower energy means a more likely correct
    reasoning step.  Paraphrase deltas are original-positive energy minus the
    deterministic paraphrase energy for positive cases.

    Spec: REQ-VERIFY-1416
    """

    validation_labels: np.ndarray
    validation_energies: np.ndarray
    validation_paraphrase_deltas: np.ndarray
    test_labels: np.ndarray
    test_energies: np.ndarray
    test_paraphrase_deltas: np.ndarray
    baseline_auroc: float
    exp1401_reference_delta: float
    corpus_cases_used: int
    validation_cases_used: int
    test_cases_used: int


@dataclass(frozen=True)
class TemperatureCalibrationResult:
    """Measured before/after temperature scaling gates for Exp 1416.

    Spec: REQ-VERIFY-1416, SCENARIO-VERIFY-1416
    """

    best_temperature: float
    validation_losses: dict[float, float]
    auroc_before: float
    auroc_after: float
    calibration_auroc_delta_before: float
    calibration_auroc_delta_after: float
    variance_before_temp_scaling: float
    variance_after_temp_scaling: float
    variance_worsened: bool
    auroc_preserved: bool
    corpus_cases_used: int
    validation_cases_used: int
    test_cases_used: int


def _as_float_array(values: Iterable[float] | np.ndarray, *, name: str) -> np.ndarray:
    raw_values = values if isinstance(values, np.ndarray) else list(values)
    array = np.asarray(raw_values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    return array


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _binary_nll(labels: np.ndarray, energies: np.ndarray, temperature: float) -> float:
    probabilities = _sigmoid(-apply_temperature_to_energies(energies, temperature))
    probabilities = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    losses = labels * np.log(probabilities) + (1.0 - labels) * np.log(1.0 - probabilities)
    return float(-np.mean(losses))


def apply_temperature_to_energies(energies: np.ndarray, temperature: float) -> np.ndarray:
    """Divide EBM-CoT energies by one positive scalar temperature.

    A positive scalar changes calibration scale but not score order.  That is
    the property Exp 1416 relies on: temperature scaling can reduce energy
    variance while preserving the AUROC ranking from Exp 1401.

    Spec: REQ-VERIFY-1416
    """

    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError("temperature must be a positive finite scalar")
    return np.asarray(energies, dtype=np.float64) / float(temperature)


def auroc_from_energies(labels: np.ndarray, energies: np.ndarray) -> float:
    """Compute AUROC with lower EBM energy mapped to higher correctness score.

    Spec: REQ-VERIFY-1416
    """

    label_array = _as_float_array(labels, name="labels")
    energy_array = _as_float_array(energies, name="energies")
    if len(label_array) != len(energy_array):
        raise ValueError("labels and energies must have the same length")
    if len(set(label_array.tolist())) < 2:
        return 0.5
    return float(auroc(label_array, -energy_array))


def paraphrase_variance_after_temperature(
    paraphrase_deltas: np.ndarray,
    temperature: float,
) -> float:
    """Return the variance of paraphrase energy deltas after temperature scaling.

    Spec: REQ-VERIFY-1416
    """

    deltas = _as_float_array(paraphrase_deltas, name="paraphrase_deltas")
    if len(deltas) == 0:
        return 0.0
    scaled = apply_temperature_to_energies(deltas, temperature)
    return float(np.var(scaled))


def fit_best_temperature(
    validation_labels: np.ndarray,
    validation_energies: np.ndarray,
    *,
    candidate_temperatures: Iterable[float] = DEFAULT_TEMPERATURE_CANDIDATES,
) -> FittedTemperature:
    """Fit scalar `T*` by minimizing validation negative log likelihood.

    Only validation labels and energies enter this routine.  The test split is
    deliberately absent from the signature so a caller cannot accidentally tune
    the temperature on held-out test scores.

    Spec: REQ-VERIFY-1416
    """

    labels = _as_float_array(validation_labels, name="validation_labels")
    energies = _as_float_array(validation_energies, name="validation_energies")
    if len(labels) != len(energies):
        raise ValueError("validation labels and energies must have the same length")
    if len(labels) == 0:
        raise ValueError("validation split must not be empty")

    losses: dict[float, float] = {}
    for candidate in candidate_temperatures:
        temperature = float(candidate)
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("candidate temperatures must be positive finite scalars")
        losses[temperature] = _binary_nll(labels, energies, temperature)
    if not losses:
        raise ValueError("at least one candidate temperature is required")

    best_temperature = min(losses, key=lambda value: (losses[value], value))
    return FittedTemperature(best_temperature=best_temperature, validation_losses=losses)


def calibrate_temperature_scores(
    scores: TemperatureCalibrationScores,
    *,
    candidate_temperatures: Iterable[float] = DEFAULT_TEMPERATURE_CANDIDATES,
    auroc_tolerance: float = DEFAULT_AUROC_TOLERANCE,
    variance_tolerance: float = DEFAULT_VARIANCE_TOLERANCE,
) -> TemperatureCalibrationResult:
    """Fit `T*` on validation scores and measure held-out AUROC/variance gates.

    Spec: REQ-VERIFY-1416, SCENARIO-VERIFY-1416
    """

    fitted = fit_best_temperature(
        scores.validation_labels,
        scores.validation_energies,
        candidate_temperatures=candidate_temperatures,
    )
    scaled_test_energies = apply_temperature_to_energies(
        scores.test_energies,
        fitted.best_temperature,
    )
    auroc_before = auroc_from_energies(scores.test_labels, scores.test_energies)
    auroc_after = auroc_from_energies(scores.test_labels, scaled_test_energies)
    delta_before = float(auroc_before - scores.baseline_auroc)
    delta_after = float(auroc_after - scores.baseline_auroc)
    variance_before = paraphrase_variance_after_temperature(scores.test_paraphrase_deltas, 1.0)
    variance_after = paraphrase_variance_after_temperature(
        scores.test_paraphrase_deltas,
        fitted.best_temperature,
    )
    variance_worsened = variance_after > variance_before + float(variance_tolerance)
    auroc_preserved = (
        scores.exp1401_reference_delta > 0.0
        and delta_before > 0.0
        and delta_after > 0.0
        and abs(auroc_after - auroc_before) <= float(auroc_tolerance)
    )
    return TemperatureCalibrationResult(
        best_temperature=fitted.best_temperature,
        validation_losses=fitted.validation_losses,
        auroc_before=auroc_before,
        auroc_after=auroc_after,
        calibration_auroc_delta_before=delta_before,
        calibration_auroc_delta_after=delta_after,
        variance_before_temp_scaling=variance_before,
        variance_after_temp_scaling=variance_after,
        variance_worsened=variance_worsened,
        auroc_preserved=auroc_preserved,
        corpus_cases_used=scores.corpus_cases_used,
        validation_cases_used=scores.validation_cases_used,
        test_cases_used=scores.test_cases_used,
    )


def _split_train_validation(
    split: FoVerSplit,
    *,
    validation_fraction: float,
) -> tuple[
    list[FoVerStepCase],
    list[FoVerStepCase],
    list[FoVerStepCase],
    list[FoVerStepCase],
]:
    if not 0.0 < float(validation_fraction) < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    n_validation = max(1, int(round(len(split.train_positive) * float(validation_fraction))))
    if n_validation >= len(split.train_positive) or n_validation >= len(split.train_negative):
        raise ValueError("validation split would consume all training pairs")
    train_positive = split.train_positive[:-n_validation]
    train_negative = split.train_negative[:-n_validation]
    validation_positive = split.train_positive[-n_validation:]
    validation_negative = split.train_negative[-n_validation:]
    return train_positive, train_negative, validation_positive, validation_negative


def _energy_array(
    calibrator: EBMCoTKANEnergyCalibrator,
    cases: list[FoVerStepCase],
) -> np.ndarray:
    return np.asarray([calibrator.energy(case) for case in cases], dtype=np.float64)


def _label_array(cases: list[FoVerStepCase]) -> np.ndarray:
    return np.asarray([case.label for case in cases], dtype=np.float64)


def _paraphrase_delta_array(
    calibrator: EBMCoTKANEnergyCalibrator,
    positive_cases: list[FoVerStepCase],
) -> np.ndarray:
    deltas = []
    for case in positive_cases:
        paraphrase = FoVerStepCase(
            case_id=f"{case.case_id}:paraphrase",
            question=case.question,
            step_text=paraphrase_positive_step(case.step_text),
            label=1,
        )
        deltas.append(calibrator.energy(case) - calibrator.energy(paraphrase))
    return np.asarray(deltas, dtype=np.float64)


def regenerate_temperature_calibration_scores(
    *,
    exp1401_reference: dict[str, Any],
    fover_path: Path | str = DEFAULT_FOVER_PATH,
    models_dir: Path | str = DEFAULT_MODELS_DIR,
    n_epochs: int = DEFAULT_N_EPOCHS,
    hinge_margin: float = DEFAULT_HINGE_MARGIN,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    max_pairs_per_class: int | None = None,
) -> TemperatureCalibrationScores:
    """Regenerate the minimal deterministic score arrays needed for Exp 1416.

    The Exp 1401 artifact stores summary metrics, not per-case raw scores.  To
    avoid fresh LLM inference or new data mining, this routine reconstructs the
    same FoVer split seed, holds out a validation slice for `T*`, retrains the
    small KAN hinge objective on the remaining local rows, and scores only the
    validation/test cases needed by the temperature pass.

    Spec: REQ-VERIFY-1416
    """

    cases = load_fover_verified_cases(fover_path)
    split = make_balanced_split(
        cases,
        seed=EXP1384_SPLIT_SEED,
        max_pairs_per_class=max_pairs_per_class,
    )
    train_positive, train_negative, validation_positive, validation_negative = (
        _split_train_validation(
            split,
            validation_fraction=validation_fraction,
        )
    )
    calibrator = EBMCoTKANEnergyCalibrator.load_current_checkpoint(models_dir)
    baseline_auroc = calibrator.evaluate_auroc(split.test_cases)
    calibrator.train_ebm_cot(
        train_positive,
        train_negative,
        n_epochs=n_epochs,
        hinge_margin=hinge_margin,
        consistency_weight=HINGE_ONLY_CONSISTENCY_WEIGHT,
    )

    validation_cases = validation_positive + validation_negative
    test_positive = [case for case in split.test_cases if case.label == 1]
    return TemperatureCalibrationScores(
        validation_labels=_label_array(validation_cases),
        validation_energies=_energy_array(calibrator, validation_cases),
        validation_paraphrase_deltas=_paraphrase_delta_array(calibrator, validation_positive),
        test_labels=_label_array(split.test_cases),
        test_energies=_energy_array(calibrator, split.test_cases),
        test_paraphrase_deltas=_paraphrase_delta_array(calibrator, test_positive),
        baseline_auroc=baseline_auroc,
        exp1401_reference_delta=float(exp1401_reference["calibration_auroc_delta"]),
        corpus_cases_used=split.corpus_cases_used,
        validation_cases_used=len(validation_cases),
        test_cases_used=len(split.test_cases),
    )


def _honest_verdict(result: TemperatureCalibrationResult) -> str:
    if result.auroc_preserved and not result.variance_worsened:
        return "temperature_scaling_reduced_variance_and_preserved_auroc"
    if result.auroc_preserved:
        return "temperature_scaling_preserved_auroc_but_variance_worsened"
    if not result.variance_worsened:
        return "temperature_scaling_reduced_variance_but_lost_auroc"
    return "temperature_scaling_failed_variance_and_auroc_gates"


def build_temperature_calibration_artifact(
    *,
    result: TemperatureCalibrationResult,
    exp1401_reference: dict[str, Any],
    duration_s: float,
    run_date: str = EXP1401_RUN_DATE,
) -> dict[str, Any]:
    """Build the complete Exp 1416 JSON artifact.

    Spec: REQ-VERIFY-1416, SCENARIO-VERIFY-1416
    """

    return {
        "status": "complete",
        "run_date": run_date,
        "experiment": 1416,
        "title": "EBM-CoT v3 post-hoc temperature calibration on hinge-only scores",
        "temperature_scaling_applied": result.best_temperature != 1.0,
        "best_temperature": float(result.best_temperature),
        "calibration_auroc_delta_before": float(result.calibration_auroc_delta_before),
        "calibration_auroc_delta_after": float(result.calibration_auroc_delta_after),
        "paraphrase_energy_variance_before_temp_scaling": float(
            result.variance_before_temp_scaling
        ),
        "paraphrase_energy_variance_after_temp_scaling": float(
            result.variance_after_temp_scaling
        ),
        "variance_worsened": result.variance_worsened,
        "auroc_preserved": result.auroc_preserved,
        "honest_verdict": _honest_verdict(result),
        "auroc_before_temp_scaling": float(result.auroc_before),
        "auroc_after_temp_scaling": float(result.auroc_after),
        "exp1401_reference_calibration_auroc_delta": float(
            exp1401_reference["calibration_auroc_delta"]
        ),
        "exp1401_reference_auroc": exp1401_reference.get("ebm_cot_v2_auroc"),
        "exp1401_reference_variance_after": exp1401_reference.get(
            "paraphrase_energy_variance_after"
        ),
        "candidate_temperatures": list(DEFAULT_TEMPERATURE_CANDIDATES),
        "validation_losses": {
            str(temperature): float(loss)
            for temperature, loss in sorted(result.validation_losses.items())
        },
        "corpus_cases_used": result.corpus_cases_used,
        "validation_cases_used": result.validation_cases_used,
        "test_cases_used": result.test_cases_used,
        "fit_split": "validation",
        "test_split_used_for_temperature_fit": False,
        "duration_s": float(duration_s),
        "paper_reference": "arXiv:2604.07172",
        "source_reference": (
            "research-references.md Temperature Scaling for Semantic Uncertainty "
            "Quantification entry and results/experiment_1401_ebm_cot_v2_hinge_only.json"
        ),
    }


def write_temperature_calibration_artifact(path: Path | str, artifact: dict[str, Any]) -> None:
    """Atomically write an Exp 1416 JSON artifact.

    Spec: REQ-VERIFY-1416
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(target)


def write_in_progress_artifact(path: Path | str = DEFAULT_ARTIFACT_PATH) -> None:
    """Write the required Exp 1416 bootstrap artifact before fitting starts.

    Spec: REQ-VERIFY-1416
    """

    write_temperature_calibration_artifact(
        path,
        {
            "status": "in_progress",
            "temperature_scaling_applied": False,
            "best_temperature": None,
            "calibration_auroc_delta_before": None,
            "calibration_auroc_delta_after": None,
            "paraphrase_energy_variance_before_temp_scaling": None,
            "paraphrase_energy_variance_after_temp_scaling": None,
            "variance_worsened": None,
            "auroc_preserved": None,
            "honest_verdict": "in_progress",
        },
    )


def run_temperature_calibration_pass(
    *,
    artifact_path: Path | str = DEFAULT_ARTIFACT_PATH,
    exp1401_artifact_path: Path | str = DEFAULT_EXP1401_ARTIFACT_PATH,
    fover_path: Path | str = DEFAULT_FOVER_PATH,
    models_dir: Path | str = DEFAULT_MODELS_DIR,
    n_epochs: int = DEFAULT_N_EPOCHS,
    candidate_temperatures: Iterable[float] = DEFAULT_TEMPERATURE_CANDIDATES,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    max_pairs_per_class: int | None = None,
    scores: TemperatureCalibrationScores | None = None,
    exp1401_reference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run Exp 1416 end to end and write the final JSON artifact.

    Optional `scores` and `exp1401_reference` arguments let unit tests exercise
    the runner without retraining the KAN probe.  Production calls leave them as
    `None`, causing deterministic local score regeneration from FoVer rows.

    Spec: REQ-VERIFY-1416
    """

    started_at = time.time()
    write_in_progress_artifact(artifact_path)
    reference = (
        dict(exp1401_reference)
        if exp1401_reference is not None
        else json.loads(Path(exp1401_artifact_path).read_text(encoding="utf-8"))
    )
    calibration_scores = (
        scores
        if scores is not None
        else regenerate_temperature_calibration_scores(
            exp1401_reference=reference,
            fover_path=fover_path,
            models_dir=models_dir,
            n_epochs=n_epochs,
            validation_fraction=validation_fraction,
            max_pairs_per_class=max_pairs_per_class,
        )
    )
    result = calibrate_temperature_scores(
        calibration_scores,
        candidate_temperatures=candidate_temperatures,
    )
    artifact = build_temperature_calibration_artifact(
        result=result,
        exp1401_reference=reference,
        duration_s=round(time.time() - started_at, 3),
    )
    write_temperature_calibration_artifact(artifact_path, artifact)
    return artifact
