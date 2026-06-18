"""Exp 4393: skeptic-proof the Exp 4392 first-error localizer win.

Spec refs: REQ-VERIFY-4393, SCENARIO-VERIFY-4393.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from carnot import experiment_4392_verifiable_process_data_localizer as exp4392


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = ROOT / "results" / "experiment_4393_localizer_skeptic_proof.json"
EXP4392_ARTIFACT_PATH = ROOT / "results" / "experiment_4392_verifiable_process_data_localizer.json"
FOVER_STEP_CORPUS_PATH = ROOT / "data" / "step_level_prm_training.jsonl"
VERIFIER_GAPS_PATH = ROOT / "ops" / "verifier_gaps.md"

RANDOM_SEED = 4393
RANDOM_SEEDS_USED = (4393, 4394)
BOOTSTRAP_RESAMPLES = 2500
TEMPLATE_DROP_FLOOR = 0.05
ENSEMBLE_BASELINE_F1 = exp4392.ENSEMBLE_BASELINE_F1
INFERENCE_SUBSTRATE = exp4392.INFERENCE_SUBSTRATE
SPEC_REFS = ["REQ-VERIFY-4393", "SCENARIO-VERIFY-4393"]

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "localizer_win_is_genuine",
    "beats_position_only_baseline",
    "template_ablation_drop",
    "held_out_real_localization_delta_ci95",
    "verifier_is_oracle",
    "preconditions_checked",
    "random_seed",
    "reproducibility_checksum",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. A genuine PASS (the localizer headline graduates) "
        "and a FAIL (the A1 win quarantined as template-leak/position/overfit-"
        "confounded) are BOTH decision-grade."
    ),
    "localizer_win_is_genuine": (
        "BARE bool: the capstone reads this; true iff the A1 real-split "
        "advantage beats a content-blind position-only baseline AND degrades "
        "under template-ablation AND holds on a held-out REAL split (all "
        "CI95-excl-0) -- the diagnostic that the localizer win is real "
        "first-error structure, not synthetic-template leakage / position bias "
        "/ overfit."
    ),
    "beats_position_only_baseline": (
        "BARE bool: true iff the A1 localizer beats a content-blind position-only "
        "baseline on the REAL split (CI95-excl-0) -- rules out the 'localizer "
        "just learned where errors sit' artifact."
    ),
    "template_ablation_drop": (
        "BARE float: the REAL-split F1 drop when the synthetic injection template "
        "is shuffled/randomized -- a material drop confirms the localizer "
        "learned real first-error structure, not the template; ~0 drop = "
        "template leakage."
    ),
    "held_out_real_localization_delta_ci95": (
        "Bootstrap CI95 of the localization delta on a HELD-OUT REAL split -- "
        "excluding 0 rules out synthetic-distribution / single-split overfit."
    ),
    "verifier_is_oracle": "BARE bool=false -- the localizer is oracle-distinct.",
    "preconditions_checked": (
        "Records the A1 win + corpus + REAL split + TRM-stand-down verified; "
        "pre-empts the silent-missing-resource fabrication mode."
    ),
    "random_seed": (
        "Determinism precondition for the template-ablation + the position-only "
        "baseline + the held-out split + the bootstrap."
    ),
    "reproducibility_checksum": (
        "Hash of the A1 localizer + the ablations + the held-out split; lets a third party re-run."
    ),
}


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime configuration for Exp 4393."""

    repo_root: Path = ROOT
    exp4392_artifact_path: Path = EXP4392_ARTIFACT_PATH
    fover_step_corpus_path: Path = FOVER_STEP_CORPUS_PATH
    artifact_path: Path = ARTIFACT_PATH
    verifier_gaps_path: Path = VERIFIER_GAPS_PATH
    random_seed: int = RANDOM_SEED
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES
    started_at: float | None = None
    clock: Callable[[], float] = time.time

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at


@dataclass(frozen=True)
class PositionOnlyBaseline:
    """Content-blind predictor using only empirical first-error step counts."""

    position_counts: dict[int, int]

    @classmethod
    def fit(cls, traces: Sequence[exp4392.ProcessTrace]) -> "PositionOnlyBaseline":
        counts: Counter[int] = Counter()
        for trace in traces:
            if trace.first_error_index is not None:
                counts[int(trace.first_error_index)] += 1
        return cls(position_counts=dict(sorted(counts.items())))

    def predict_first_error_index(self, trace: exp4392.ProcessTrace) -> int | None:
        if not trace.steps or not self.position_counts:
            return None
        valid = [idx for idx in self.position_counts if idx < len(trace.steps)]
        if not valid:
            return len(trace.steps) - 1
        return max(valid, key=lambda idx: (self.position_counts[idx], -idx))


AdversarialVerifyRunner = Callable[[Path], dict[str, Any]]


def _round_float(value: float | None, digits: int = 6) -> float | None:
    return exp4392.round_float(value, digits=digits)


def _write_artifact(path: Path, artifact: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json_dict(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # pragma: no cover - invalid JSON path is a blocked-resource guard.
        return None
    return payload if isinstance(payload, dict) else None


def _localizer_from_a1(payload: dict[str, Any]) -> exp4392.LocalizerModel | None:
    localizer = payload.get("model_specs", {}).get("localizer")
    if not isinstance(localizer, dict):
        return None
    weights = localizer.get("weights")
    if not isinstance(weights, dict):
        return None
    try:
        numeric_weights = {str(key): float(value) for key, value in weights.items()}
        threshold = float(localizer.get("threshold", 0.0))
    except (TypeError, ValueError):  # pragma: no cover - malformed A1 model guard.
        return None
    summary = localizer.get("training_summary", {})
    return exp4392.LocalizerModel(
        weights=numeric_weights,
        threshold=threshold,
        training_summary=summary if isinstance(summary, dict) else {},
    )


def _synthetic_n_from_a1(payload: dict[str, Any]) -> int:
    synthesis = payload.get("synthesis_verification", {})
    if isinstance(synthesis, dict) and synthesis.get("n_synthetic_traces") is not None:
        return int(synthesis["n_synthetic_traces"])
    config = payload.get("model_specs", {}).get("synthesis_config", {})
    if isinstance(config, dict) and config.get("n") is not None:
        return int(config["n"])
    return exp4392.MIN_SYNTHETIC_TRACES


def split_real_traces(
    traces: Sequence[exp4392.ProcessTrace],
    *,
    seed: int,
) -> tuple[list[exp4392.ProcessTrace], list[exp4392.ProcessTrace]]:
    ordered = sorted(traces, key=lambda trace: trace.trace_id)
    rng = random.Random(seed)
    shuffled = list(ordered)
    rng.shuffle(shuffled)
    cut = len(shuffled) // 2
    return shuffled[:cut], shuffled[cut:]


def successes_for_predictor(
    traces: Sequence[exp4392.ProcessTrace],
    predict: Callable[[exp4392.ProcessTrace], int | None],
) -> list[int]:
    successes: list[int] = []
    for trace in traces:
        if trace.first_error_index is None:
            continue
        successes.append(int(predict(trace) == trace.first_error_index))
    return successes


def f1_from_successes(successes: Sequence[int]) -> float:
    return sum(int(value) for value in successes) / len(successes) if successes else 0.0


def _paired_delta_ci95(
    left_successes: Sequence[int],
    right_successes: Sequence[int],
    *,
    seed: int,
    resamples: int,
) -> list[float | None]:
    if not left_successes or len(left_successes) != len(right_successes) or resamples <= 0:
        return [None, None]
    rng = random.Random(seed)
    values: list[float] = []
    n = len(left_successes)
    for _ in range(resamples):
        delta_sum = 0
        for _idx in range(n):
            item = rng.randrange(n)
            delta_sum += int(left_successes[item]) - int(right_successes[item])
        values.append(delta_sum / n)
    values.sort()
    lo = int(0.025 * (len(values) - 1))
    hi = int(0.975 * (len(values) - 1))
    return [_round_float(values[lo]), _round_float(values[hi])]


def _baseline_delta_ci95(
    successes: Sequence[int],
    *,
    baseline_f1: float,
    seed: int,
    resamples: int,
) -> list[float | None]:
    return exp4392._bootstrap_delta_ci95(
        successes,
        baseline_f1=baseline_f1,
        seed=seed,
        resamples=resamples,
    )


def scramble_synthetic_first_error_structure(
    corpus: Sequence[exp4392.ProcessTrace],
    *,
    seed: int,
) -> list[exp4392.ProcessTrace]:
    rng = random.Random(seed)
    scrambled: list[exp4392.ProcessTrace] = []
    for trace in corpus:
        if not trace.steps:
            scrambled.append(trace)
            continue
        current = trace.first_error_index
        candidates = [idx for idx in range(len(trace.steps)) if idx != current]
        new_index = rng.choice(candidates) if candidates else current
        steps = tuple(
            exp4392.ProcessStep(
                step_index=step.step_index,
                text=step.text,
                first_error_target=idx == new_index,
                features=dict(step.features),
                prefix_invalidity_verified=False,
                trajectory_consistent=step.trajectory_consistent,
            )
            for idx, step in enumerate(trace.steps)
        )
        scrambled.append(
            exp4392.ProcessTrace(
                trace_id=trace.trace_id,
                source_domain=trace.source_domain,
                steps=steps,
                first_error_index=new_index,
                error_class="template_ablation_scrambled_first_error",
            )
        )
    return scrambled


def position_only_report(
    heldout_traces: Sequence[exp4392.ProcessTrace],
    a1_localizer: exp4392.LocalizerModel,
    baseline: PositionOnlyBaseline,
    *,
    seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    a1_successes = successes_for_predictor(heldout_traces, a1_localizer.predict_first_error_index)
    position_successes = successes_for_predictor(heldout_traces, baseline.predict_first_error_index)
    delta = f1_from_successes(a1_successes) - f1_from_successes(position_successes)
    ci95 = _paired_delta_ci95(
        a1_successes,
        position_successes,
        seed=seed,
        resamples=bootstrap_resamples,
    )
    return {
        "a1_f1": _round_float(f1_from_successes(a1_successes)),
        "position_only_f1": _round_float(f1_from_successes(position_successes)),
        "delta": _round_float(delta),
        "delta_ci95": ci95,
        "beats_position_only_baseline": bool(ci95[0] is not None and ci95[0] > 0.0),
        "position_counts": baseline.position_counts,
        "n_error_traces": len(a1_successes),
    }


def template_ablation_report(
    heldout_traces: Sequence[exp4392.ProcessTrace],
    a1_localizer: exp4392.LocalizerModel,
    ablated_localizer: exp4392.LocalizerModel,
    *,
    seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    a1_successes = successes_for_predictor(heldout_traces, a1_localizer.predict_first_error_index)
    ablated_successes = successes_for_predictor(
        heldout_traces,
        ablated_localizer.predict_first_error_index,
    )
    drop = f1_from_successes(a1_successes) - f1_from_successes(ablated_successes)
    ci95 = _paired_delta_ci95(
        a1_successes,
        ablated_successes,
        seed=seed,
        resamples=bootstrap_resamples,
    )
    degrades = bool(drop >= TEMPLATE_DROP_FLOOR and ci95[0] is not None and ci95[0] > 0.0)
    return {
        "a1_f1": _round_float(f1_from_successes(a1_successes)),
        "template_ablated_f1": _round_float(f1_from_successes(ablated_successes)),
        "drop": _round_float(drop),
        "drop_ci95": ci95,
        "material_drop_floor": TEMPLATE_DROP_FLOOR,
        "degrades_under_template_ablation": degrades,
        "n_error_traces": len(a1_successes),
    }


def heldout_real_report(
    heldout_traces: Sequence[exp4392.ProcessTrace],
    a1_localizer: exp4392.LocalizerModel,
    *,
    seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    successes = successes_for_predictor(heldout_traces, a1_localizer.predict_first_error_index)
    a1_f1 = f1_from_successes(successes)
    delta = a1_f1 - ENSEMBLE_BASELINE_F1
    ci95 = _baseline_delta_ci95(
        successes,
        baseline_f1=ENSEMBLE_BASELINE_F1,
        seed=seed,
        resamples=bootstrap_resamples,
    )
    return {
        "a1_f1": _round_float(a1_f1),
        "ensemble_baseline_0096": _round_float(ENSEMBLE_BASELINE_F1, digits=3),
        "delta": _round_float(delta),
        "delta_ci95": ci95,
        "holds_on_held_out_real_split": bool(ci95[0] is not None and ci95[0] > 0.0),
        "n_traces": len(heldout_traces),
        "n_error_traces": len(successes),
        "exact_match_count": int(sum(successes)),
    }


def _checksum(source_paths: Sequence[Path], payload: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for path in sorted({Path(path) for path in source_paths}, key=lambda item: str(item)):
        digest.update(str(path).encode("utf-8"))
        if not path.exists():
            digest.update(b"\0MISSING\0")
            continue
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return "sha256:" + digest.hexdigest()


def _precondition(resource: str, available: bool, detail: str) -> dict[str, Any]:
    return {"resource": resource, "available": bool(available), "detail": detail}


def build_blocked_artifact(
    *,
    honest_verdict: str,
    preconditions_checked: list[dict[str, Any]],
    source_paths: Sequence[Path],
    duration_s: float,
    random_seed: int,
) -> dict[str, Any]:
    return {
        "experiment": "experiment_4393_localizer_skeptic_proof",
        "schema": "carnot.localizer_skeptic_proof.v1",
        "honest_verdict": honest_verdict,
        "localizer_win_is_genuine": False,
        "beats_position_only_baseline": False,
        "template_ablation_drop": 0.0,
        "held_out_real_localization_delta_ci95": [None, None],
        "verifier_is_oracle": False,
        "preconditions_checked": preconditions_checked,
        "random_seed": int(random_seed),
        "random_seeds_used": list(RANDOM_SEEDS_USED),
        "reproducibility_checksum": _checksum(
            source_paths,
            payload={"blocked": honest_verdict, "random_seed": random_seed},
        ),
        "model_specs": {
            "a1_artifact": str(source_paths[0]) if source_paths else str(EXP4392_ARTIFACT_PATH),
            "fover_step_corpus": str(source_paths[1])
            if len(source_paths) > 1
            else str(FOVER_STEP_CORPUS_PATH),
            "trm_training": "stood_down_not_invoked",
            "live_generation": False,
            "verifier_is_oracle": False,
        },
        "diagnostics": {},
        "a1_win_quarantined": False,
        "missing_verifier_gaps": [],
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "methodology_note": "blocked before re-test controls; no skeptic-proof metrics fabricated",
        "duration_s": _round_float(duration_s, digits=3),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "adversarial_verify": {"skipped": "blocked"},
    }


def build_complete_artifact(
    *,
    a1_artifact: dict[str, Any],
    a1_localizer: exp4392.LocalizerModel,
    ablated_localizer: exp4392.LocalizerModel,
    synthetic_corpus: Sequence[exp4392.ProcessTrace],
    primary_traces: Sequence[exp4392.ProcessTrace],
    heldout_traces: Sequence[exp4392.ProcessTrace],
    preconditions_checked: list[dict[str, Any]],
    source_paths: Sequence[Path],
    position_report: dict[str, Any],
    template_report: dict[str, Any],
    heldout_report: dict[str, Any],
    duration_s: float,
    random_seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    localizer_win_is_genuine = bool(
        position_report["beats_position_only_baseline"]
        and template_report["degrades_under_template_ablation"]
        and heldout_report["holds_on_held_out_real_split"]
    )
    gaps = (
        []
        if localizer_win_is_genuine
        else [_missing_verifier_gap(position_report, template_report)]
    )
    checksum_payload = {
        "a1_localizer": a1_localizer.as_dict(),
        "template_ablated_localizer": ablated_localizer.as_dict(),
        "position_report": position_report,
        "template_report": template_report,
        "heldout_report": heldout_report,
        "heldout_trace_ids": [trace.trace_id for trace in heldout_traces],
        "random_seed": random_seed,
    }
    return {
        "experiment": "experiment_4393_localizer_skeptic_proof",
        "schema": "carnot.localizer_skeptic_proof.v1",
        "honest_verdict": (
            "success: localizer_win_is_genuine"
            if localizer_win_is_genuine
            else "complete: a1_win_quarantined_as_artifact_confounded"
        ),
        "localizer_win_is_genuine": localizer_win_is_genuine,
        "beats_position_only_baseline": bool(position_report["beats_position_only_baseline"]),
        "template_ablation_drop": float(template_report["drop"]),
        "held_out_real_localization_delta_ci95": heldout_report["delta_ci95"],
        "verifier_is_oracle": False,
        "preconditions_checked": preconditions_checked,
        "random_seed": int(random_seed),
        "random_seeds_used": list(RANDOM_SEEDS_USED),
        "reproducibility_checksum": _checksum(source_paths, payload=checksum_payload),
        "model_specs": {
            "a1_artifact": str(source_paths[0]),
            "a1_reproducibility_checksum": a1_artifact.get("reproducibility_checksum"),
            "a1_localizer": a1_localizer.as_dict(),
            "template_ablated_localizer": ablated_localizer.as_dict(),
            "synthetic_corpus": {
                "source": "deterministic_reconstruction_from_exp4392_synthesis_config",
                "n": len(synthetic_corpus),
            },
            "fover_step_corpus": str(source_paths[1]),
            "heldout_split_seed": int(random_seed),
            "bootstrap_resamples": int(bootstrap_resamples),
            "position_only_baseline": {
                "feature_inputs": ["first_error_step_index_distribution_only"],
                "content_blind": True,
                "position_counts": position_report["position_counts"],
            },
            "trm_training": "stood_down_not_invoked",
            "live_generation": False,
            "verifier_is_oracle": False,
        },
        "diagnostics": {
            "held_out_real_split": heldout_report,
            "position_only_baseline": position_report,
            "template_ablation": template_report,
            "split_sizes": {
                "primary_real_traces": len(primary_traces),
                "held_out_real_traces": len(heldout_traces),
            },
        },
        "a1_win_quarantined": not localizer_win_is_genuine,
        "missing_verifier_gaps": gaps,
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "methodology_note": (
            "The diagnostic uses the Exp 4392 A1 localizer as the model under "
            "test, reconstructs the deterministic synthetic corpus for the "
            "template-ablation retrain, and evaluates only cached REAL FoVer "
            "traces. No TRM training or live generation is invoked."
        ),
        "duration_s": _round_float(duration_s, digits=3),
        "inference_substrate": INFERENCE_SUBSTRATE,
    }


def _missing_verifier_gap(
    position_report: dict[str, Any],
    template_report: dict[str, Any],
) -> dict[str, Any]:
    if not position_report["beats_position_only_baseline"]:
        confounder = "position_only_baseline_ties_a1"
        missing = "A held-out real split with non-degenerate first-error positions."
    elif not template_report["degrades_under_template_ablation"]:
        confounder = "template_ablation_no_material_drop"
        missing = "A template-invariant feature that fails when synthetic structure is scrambled."
    else:
        confounder = "heldout_real_advantage_not_stable"
        missing = "A second-seed real split where the localization advantage excludes zero."
    return {
        "gap_id": "GAP-4393-LOCALIZER-POSITION-OR-TEMPLATE-CONFOUND",
        "status": "open",
        "confounder": confounder,
        "missing_discriminator": missing,
        "candidate_design": (
            "Collect or construct REAL first-error traces with varied first-error "
            "positions and retrain the localizer with template-family holdouts."
        ),
        "priority": "high",
    }


def append_missing_verifier_gaps(path: Path, gaps: Sequence[dict[str, Any]]) -> None:
    if not gaps:
        return
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    additions: list[str] = []
    for gap in gaps:
        gap_id = str(gap["gap_id"])
        if gap_id in existing:
            continue
        additions.append(
            "\n".join(
                [
                    f"### {gap_id}: Exp 4393 localizer skeptic-proof residual",
                    f"- status: {gap['status']}",
                    "- evidence: `results/experiment_4393_localizer_skeptic_proof.json`.",
                    f"- confounder: {gap['confounder']}",
                    f"- missing discriminator: {gap['missing_discriminator']}",
                    f"- candidate design: {gap['candidate_design']}",
                    f"- priority: {gap['priority']}",
                    "",
                ]
            )
        )
    if additions:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(existing.rstrip() + "\n\n" + "\n".join(additions), encoding="utf-8")


def artifact_schema_errors(artifact: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing {field}")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not (
        verdict.startswith("success:")
        or verdict.startswith("complete:")
        or verdict.startswith("blocked_")
    ):
        errors.append("honest_verdict must be terminal-prefixed")
    if not isinstance(artifact.get("localizer_win_is_genuine"), bool):
        errors.append("localizer_win_is_genuine must be bare bool")
    if not isinstance(artifact.get("beats_position_only_baseline"), bool):
        errors.append("beats_position_only_baseline must be bare bool")
    if not isinstance(artifact.get("template_ablation_drop"), float):
        errors.append("template_ablation_drop must be bare float")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    return errors


def _preconditions_hold(checks: Sequence[dict[str, Any]]) -> bool:
    return all(bool(check["available"]) for check in checks)


def run_experiment(
    config: ExperimentConfig | None = None,
    *,
    adversarial_verify_runner: AdversarialVerifyRunner = exp4392.run_adversarial_verify,
    write: bool = True,
) -> dict[str, Any]:
    cfg = config or ExperimentConfig()
    started = cfg.start_time()
    source_paths = [cfg.exp4392_artifact_path, cfg.fover_step_corpus_path]
    checks: list[dict[str, Any]] = []

    a1_artifact = _load_json_dict(cfg.exp4392_artifact_path)
    if a1_artifact is None:
        checks.append(_precondition("exp4392_a1_artifact", False, "missing_or_unreadable"))
        artifact = build_blocked_artifact(
            honest_verdict="blocked_a1_artifact",
            preconditions_checked=checks,
            source_paths=source_paths,
            duration_s=cfg.clock() - started,
            random_seed=cfg.random_seed,
        )
        if write:
            _write_artifact(cfg.artifact_path, artifact)
        return artifact

    a1_win = a1_artifact.get("localizer_beats_ensemble_baseline") is True
    checks.append(
        _precondition(
            "exp4392_localizer_beats_ensemble_baseline",
            a1_win,
            f"value={a1_artifact.get('localizer_beats_ensemble_baseline')!r}",
        )
    )
    if not a1_win:
        artifact = build_blocked_artifact(
            honest_verdict="blocked_no_win_to_validate",
            preconditions_checked=checks,
            source_paths=source_paths,
            duration_s=cfg.clock() - started,
            random_seed=cfg.random_seed,
        )
        if write:
            _write_artifact(cfg.artifact_path, artifact)
        return artifact

    a1_localizer = _localizer_from_a1(a1_artifact)
    checks.append(
        _precondition(
            "a1_localizer",
            a1_localizer is not None,
            "weights present" if a1_localizer is not None else "missing weights",
        )
    )
    n_synthetic = _synthetic_n_from_a1(a1_artifact)
    synthetic = exp4392.synthesize_verifiable_first_error_corpus(
        n_traces=n_synthetic,
        seed=int(a1_artifact.get("random_seed", exp4392.RANDOM_SEED)),
    )
    synthetic_summary = exp4392.synthesis_verification_summary(synthetic)
    expected_n = a1_artifact.get("synthesis_verification", {}).get("n_synthetic_traces")
    synthetic_ok = expected_n is None or int(expected_n) == synthetic_summary["n_synthetic_traces"]
    checks.append(
        _precondition(
            "synthetic_corpus_reconstruction",
            synthetic_ok and len(synthetic) > 0,
            f"reconstructed={len(synthetic)}; expected={expected_n}",
        )
    )
    if cfg.fover_step_corpus_path.is_file():
        real_traces = exp4392._read_fover_real_traces(cfg.fover_step_corpus_path)
    else:
        real_traces = []
    primary_traces, heldout_traces = split_real_traces(real_traces, seed=cfg.random_seed)
    primary_errors = sum(1 for trace in primary_traces if trace.first_error_index is not None)
    heldout_errors = sum(1 for trace in heldout_traces if trace.first_error_index is not None)
    checks.append(
        _precondition(
            "held_out_real_split",
            primary_errors > 0 and heldout_errors > 0,
            (
                f"primary_traces={len(primary_traces)}; primary_errors={primary_errors}; "
                f"heldout_traces={len(heldout_traces)}; heldout_errors={heldout_errors}"
            ),
        )
    )
    checks.append(_precondition("trm_training_stand_down", True, "not invoked"))

    if a1_localizer is None or not _preconditions_hold(checks):
        failed = next(check["resource"] for check in checks if not check["available"])
        verdict = "blocked_a1_localizer" if failed == "a1_localizer" else f"blocked_{failed}"
        artifact = build_blocked_artifact(
            honest_verdict=verdict,
            preconditions_checked=checks,
            source_paths=source_paths,
            duration_s=cfg.clock() - started,
            random_seed=cfg.random_seed,
        )
        if write:
            _write_artifact(cfg.artifact_path, artifact)
        return artifact

    position_baseline = PositionOnlyBaseline.fit(primary_traces)
    ablated_synthetic = scramble_synthetic_first_error_structure(
        synthetic,
        seed=cfg.random_seed,
    )
    ablated_localizer = exp4392.train_contrastive_localizer(ablated_synthetic)
    pos_report = position_only_report(
        heldout_traces,
        a1_localizer,
        position_baseline,
        seed=cfg.random_seed,
        bootstrap_resamples=cfg.bootstrap_resamples,
    )
    tmpl_report = template_ablation_report(
        heldout_traces,
        a1_localizer,
        ablated_localizer,
        seed=cfg.random_seed + 1,
        bootstrap_resamples=cfg.bootstrap_resamples,
    )
    heldout_report = heldout_real_report(
        heldout_traces,
        a1_localizer,
        seed=cfg.random_seed + 2,
        bootstrap_resamples=cfg.bootstrap_resamples,
    )
    artifact = build_complete_artifact(
        a1_artifact=a1_artifact,
        a1_localizer=a1_localizer,
        ablated_localizer=ablated_localizer,
        synthetic_corpus=synthetic,
        primary_traces=primary_traces,
        heldout_traces=heldout_traces,
        preconditions_checked=checks,
        source_paths=source_paths,
        position_report=pos_report,
        template_report=tmpl_report,
        heldout_report=heldout_report,
        duration_s=cfg.clock() - started,
        random_seed=cfg.random_seed,
        bootstrap_resamples=cfg.bootstrap_resamples,
    )
    if write:
        _write_artifact(cfg.artifact_path, artifact)
        artifact["adversarial_verify"] = adversarial_verify_runner(cfg.artifact_path)
        if artifact["adversarial_verify"].get("returncode") not in (0, None):
            artifact["flagged_adversarial"] = True
            artifact["a1_win_quarantined"] = True
        _write_artifact(cfg.artifact_path, artifact)
        append_missing_verifier_gaps(cfg.verifier_gaps_path, artifact["missing_verifier_gaps"])
    else:
        artifact["adversarial_verify"] = {"returncode": None, "skipped": True}
    return artifact


def main() -> int:  # pragma: no cover - exercised through results/ CLI in integration runs.
    artifact = run_experiment(write=True)
    print(
        "[exp4393] "
        f"{artifact['honest_verdict']} "
        f"genuine={artifact['localizer_win_is_genuine']} "
        f"position_control={artifact['beats_position_only_baseline']} "
        f"template_drop={artifact['template_ablation_drop']} -> {ARTIFACT_PATH}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
