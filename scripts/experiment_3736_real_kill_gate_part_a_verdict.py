#!/usr/bin/env python3
"""Aggregate Exp 3734/3735 into the real Thesis-A kill-gate part-(a) verdict.

Spec: REQ-EBT-3736, SCENARIO-EBT-3736-PASS,
SCENARIO-EBT-3736-UNTESTED, SCENARIO-EBT-3736-DIVERGED.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]

SOURCE_REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SOURCE_REPO_ROOT
OUTPUT_REL_PATH = Path("results/experiment_3736_real_kill_gate_part_a_verdict.json")
EXP3727_REL_PATH = Path("results/experiment_3727_matched_compute_eval_harness.json")
EXP3729_REL_PATH = Path("results/experiment_3729_stability_kill_gate_verdict.json")
EXP3734_REL_PATH = Path("results/experiment_3734_fix_harness_and_bounded_train_chunk1.json")
EXP3735_REL_PATH = Path("results/experiment_3735_bounded_train_chunk2_resume.json")

RANDOM_SEED = 3736
INFERENCE_SUBSTRATE = (
    "aggregation_from_upstream_artifacts (principle: a verdict over the "
    "real-run diagnostics, no live model)."
)
PASS_VERDICT = (
    "complete: real_kill_gate_part_a_PASS_ebt_trained_stably_green_light_342_"
    "supersedes_exp3729_false_negative"
)
DIVERGED_VERDICT = (
    "complete: real_kill_gate_part_a_genuinely_bounded_ebt_diverged_in_real_run_"
    "honest_negative"
)
UNTESTED_VERDICT = "complete: real_kill_gate_part_a_untested_training_did_not_complete"

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "ebt_trained_stably",
    "green_light_342",
    "training_actually_ran",
    "kill_gate_conclusion",
    "supersedes_exp3729",
    "cited_upstream_artifacts",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal prefix stating the GENUINE kill-gate outcome "
        "(stable->green-light / diverged->bounded / not-run->untested)."
    ),
    "inference_substrate": (
        "A verdict over the real-run diagnostics, no live model."
    ),
    "ebt_trained_stably": (
        "The kill-gate part-(a) boolean over the GENUINE run -- the single most "
        "load-bearing field of the milestone."
    ),
    "green_light_342": "BARE bool. Gates whether the matched-compute comparison runs.",
    "training_actually_ran": (
        "True iff cumulative_steps_trained>0 -- distinguishes a real stability "
        "result from the exp3729 infra-block false-negative."
    ),
    "kill_gate_conclusion": (
        "Honest one-paragraph conclusion (green-light / genuinely-bounded-with-"
        "evidence / untested) so the record is self-explanatory and does not "
        "repeat the false-negative."
    ),
    "supersedes_exp3729": (
        "Explicitly records that this verdict replaces exp3729's infra-false-"
        "negative as the authoritative part-(a) result."
    ),
    "cited_upstream_artifacts": (
        "Provenance: makes the verdict traceable to the real training run "
        "(anti-fabrication audit trail)."
    ),
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Content hash catches drift.",
    "duration_s": "Wall-clock plausibility floor.",
}

UPSTREAM_FIELDS = {
    3727: [
        "honest_verdict",
        "flop_model_description",
        "matched_compute_report.ebt_total_flops",
        "matched_compute_report.ar_total_flops",
        "matched_compute_report.budget_match.ar_best_of_m",
        "matched_compute_report.budget_match.target_total_flops",
        "matched_compute_report.budget_match.within_tolerance",
        "random_seed",
        "reproducibility_checksum",
        "duration_s",
    ],
    3729: [
        "honest_verdict",
        "ebt_trained_stably",
        "green_light_342",
        "kill_gate_conclusion",
        "reproducibility_checksum",
    ],
    3734: [
        "honest_verdict",
        "harness_fix_applied",
        "cumulative_steps_trained",
        "ebt_loss_curve",
        "ar_loss_curve",
        "nan_or_divergence_events",
        "gradient_norms_bounded",
        "gradient_norm_curve",
        "stabilizers_applied",
        "peak_vram_mb",
        "preconditions_checked.cuda",
        "preconditions_checked.ebt_vendored",
        "preconditions_checked.corpus_ok",
        "random_seed",
        "reproducibility_checksum",
        "duration_s",
    ],
    3735: [
        "honest_verdict",
        "cumulative_steps_trained",
        "ebt_loss_curve",
        "ar_loss_curve",
        "ebt_converged",
        "nan_or_divergence_events",
        "gradient_norms_bounded",
        "gradient_norm_curve",
        "stabilizers_applied",
        "peak_vram_mb",
        "preconditions_checked.cuda",
        "preconditions_checked.ebt_vendored",
        "preconditions_checked.checkpoint_present",
        "random_seed",
        "reproducibility_checksum",
        "duration_s",
    ],
}


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    adversarial_verify_report: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Build the Exp 3736 verdict from checked-in upstream artifacts."""

    root_path = Path(root)
    upstream = _load_upstreams(root_path)
    diagnostics = _real_run_diagnostics(upstream)
    outcome = _classify_outcome(diagnostics)
    setup = _recommended_part_b_setup(root_path, upstream.get(3727), outcome == "pass")
    conclusion = _kill_gate_conclusion(outcome, diagnostics, setup)
    verify_report = _compact_verify_report(adversarial_verify_report or {"flags": []})

    payload: JsonDict = {
        "schema": "carnot.experiment_3736_real_kill_gate_part_a_verdict.v1",
        "experiment": 3736,
        "honest_verdict": _verdict_for_outcome(outcome),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "ebt_trained_stably": outcome == "pass",
        "green_light_342": outcome == "pass",
        "training_actually_ran": bool(diagnostics["training_actually_ran"]),
        "kill_gate_conclusion": conclusion,
        "supersedes_exp3729": True,
        "recommended_part_b_setup": setup,
        "real_run_diagnostics": diagnostics,
        "cited_upstream_artifacts": _citations(root_path, upstream),
        "adversarial_verify_clean": verify_report["critical_flag_count"] == 0,
        "adversarial_verify_report": verify_report,
        "field_principles": dict(FIELD_PRINCIPLES),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "duration_s": _duration(started_s, now_s),
    }
    payload["reproducibility_checksum"] = payload_checksum(payload)
    validate_artifact(payload)
    return payload


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate schema and decision semantics for the Exp 3736 verdict."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        raise ValueError("field_principles must be a mapping")
    missing_principles = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in principles]
    if missing_principles:
        raise ValueError(f"missing field principles: {missing_principles}")
    if artifact.get("honest_verdict") not in {PASS_VERDICT, DIVERGED_VERDICT, UNTESTED_VERDICT}:
        raise ValueError("terminal verdict must match an Exp 3736 terminal verdict")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be the Exp 3736 aggregation substrate")
    if not isinstance(artifact.get("ebt_trained_stably"), bool):
        raise ValueError("ebt_trained_stably must be boolean")
    if not isinstance(artifact.get("green_light_342"), bool):
        raise ValueError("green_light_342 must be boolean")
    if artifact.get("green_light_342") != artifact.get("ebt_trained_stably"):
        raise ValueError("green_light_342 must equal ebt_trained_stably")
    if not isinstance(artifact.get("training_actually_ran"), bool):
        raise ValueError("training_actually_ran must be boolean")
    if artifact.get("supersedes_exp3729") is not True:
        raise ValueError("supersedes_exp3729 must be true")
    if not str(artifact.get("kill_gate_conclusion") or "").strip():
        raise ValueError("kill_gate_conclusion must be present")
    _validate_citations(artifact.get("cited_upstream_artifacts"))
    if artifact.get("random_seed") != RANDOM_SEED:
        raise ValueError("random_seed must equal 3736")
    duration = artifact.get("duration_s")
    if not _finite_number(duration) or float(duration) < 0.0001:
        raise ValueError("duration_s must be numeric with the aggregation plausibility floor")
    if _has_live_model_markers(artifact):
        raise ValueError("live-model markers must not be present")
    verify_report = artifact.get("adversarial_verify_report")
    if _critical_flag_count(verify_report) > 0:
        raise ValueError("adversarial verifier must report no critical flags")
    checksum = artifact.get("reproducibility_checksum")
    if not _is_sha256(checksum):
        raise ValueError("reproducibility_checksum must be a sha256 hex string")
    if checksum != payload_checksum(artifact):
        raise ValueError("reproducibility_checksum does not match artifact content")


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Write the final Exp 3736 artifact and return its path."""

    root_path = Path(root)
    output_path = root_path / OUTPUT_REL_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_artifact(root_path, started_s=started_s, now_s=now_s)
    _write_json(output_path, payload)

    report = _run_adversarial_verify(output_path)
    payload["adversarial_verify_report"] = _compact_verify_report(report)
    payload["adversarial_verify_clean"] = payload["adversarial_verify_report"]["critical_flag_count"] == 0
    payload["reproducibility_checksum"] = payload_checksum(payload)
    validate_artifact(payload)
    _write_json(output_path, payload)
    return output_path


def sha256_file(path: Path) -> str:
    """Return the SHA256 hash for a source artifact."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Return the checksum for artifact content excluding its checksum field."""

    filtered = {key: value for key, value in payload.items() if key != "reproducibility_checksum"}
    encoded = json.dumps(filtered, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for the Exp 3736 real kill-gate verdict."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    output_path = write_artifact(args.root)
    payload = _read_json_object(output_path, required=True)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _load_upstreams(root: Path) -> dict[int, JsonDict | None]:
    return {
        3727: _read_json_object(root / EXP3727_REL_PATH, required=True),
        3729: _read_json_object(root / EXP3729_REL_PATH, required=False),
        3734: _read_json_object(root / EXP3734_REL_PATH, required=False),
        3735: _read_json_object(root / EXP3735_REL_PATH, required=False),
    }


def _read_json_object(path: Path, *, required: bool) -> JsonDict | None:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _real_run_diagnostics(upstream: Mapping[int, Mapping[str, Any] | None]) -> JsonDict:
    exp3734 = upstream.get(3734)
    exp3735 = upstream.get(3735)
    artifacts = [artifact for artifact in (exp3734, exp3735) if artifact is not None]
    steps_by_exp = {
        str(exp_id): _safe_int((upstream.get(exp_id) or {}).get("cumulative_steps_trained"))
        for exp_id in (3734, 3735)
        if upstream.get(exp_id) is not None
    }
    cumulative_steps = max(steps_by_exp.values(), default=0)
    missing_or_blocked = [
        exp_id
        for exp_id in (3734, 3735)
        if upstream.get(exp_id) is None or _is_blocked(upstream[exp_id])
    ]
    loss_curve = _selected_loss_curve(exp3734, exp3735)
    finite_loss_curve = _finite_curve(loss_curve)
    no_nan_inf_or_divergence = _no_nan_inf_or_divergence(artifacts, finite_loss_curve, loss_curve)
    gradient_norms_bounded = _gradient_norms_bounded(artifacts)
    non_runaway_convergence = _non_runaway_convergence(finite_loss_curve)
    ebt_converged_flag = exp3735 is not None and exp3735.get("ebt_converged") is True
    bounded_run_completed = bool(
        exp3734 is not None
        and exp3735 is not None
        and not _is_blocked(exp3734)
        and not _is_blocked(exp3735)
        and _terminal_complete(exp3735)
        and cumulative_steps > 0
    )
    genuine_divergence = bool(
        cumulative_steps > 0
        and (
            not no_nan_inf_or_divergence
            or _source_verdict_contains(artifacts, "diverged")
            or _runaway_collapse(finite_loss_curve)
            or _gradient_norms_exploded(artifacts)
        )
    )
    return {
        "source_verdicts": {
            str(exp_id): (upstream.get(exp_id) or {}).get("honest_verdict")
            for exp_id in (3734, 3735)
            if upstream.get(exp_id) is not None
        },
        "cumulative_steps_by_experiment": steps_by_exp,
        "cumulative_steps_trained": cumulative_steps,
        "training_actually_ran": cumulative_steps > 0,
        "bounded_run_completed": bounded_run_completed,
        "missing_or_blocked_artifacts": missing_or_blocked,
        "no_nan_inf_or_divergence": no_nan_inf_or_divergence,
        "gradient_norms_bounded": gradient_norms_bounded,
        "non_runaway_convergence": non_runaway_convergence,
        "ebt_converged_flag": ebt_converged_flag,
        "genuine_divergence": genuine_divergence,
        "finite_ebt_loss_curve": finite_loss_curve,
        "selected_loss_curve_source": 3735 if exp3735 is not None and exp3735.get("ebt_loss_curve") else 3734,
    }


def _classify_outcome(diagnostics: Mapping[str, Any]) -> str:
    stable = bool(
        diagnostics.get("bounded_run_completed")
        and diagnostics.get("training_actually_ran")
        and diagnostics.get("no_nan_inf_or_divergence")
        and diagnostics.get("gradient_norms_bounded")
        and diagnostics.get("non_runaway_convergence")
        and diagnostics.get("ebt_converged_flag")
    )
    if stable:
        return "pass"
    if diagnostics.get("genuine_divergence"):
        return "diverged"
    return "untested"


def _verdict_for_outcome(outcome: str) -> str:
    return {
        "pass": PASS_VERDICT,
        "diverged": DIVERGED_VERDICT,
        "untested": UNTESTED_VERDICT,
    }[outcome]


def _kill_gate_conclusion(outcome: str, diagnostics: Mapping[str, Any], setup: str) -> str:
    steps = diagnostics.get("cumulative_steps_trained")
    source_verdicts = diagnostics.get("source_verdicts")
    if outcome == "pass":
        return (
            "GREEN-LIGHT: the real Exp 3734/3735 run completed with "
            f"cumulative_steps_trained={steps}, no NaN/inf/divergence events, "
            "bounded gradient diagnostics, and finite non-runaway convergence. "
            f"Recommended part-(b) setup: {setup}"
        )
    if outcome == "diverged":
        return (
            "BOUNDED: the real Exp 3734/3735 run trained for nonzero steps but "
            "recorded genuine divergence evidence within the bounded budget. "
            "energy-as-generator is bounded at small scale on this corpus and "
            "budget; consider only an explicitly budgeted stabilization recipe "
            "before retirement, not the matched-compute part-(b) run."
        )
    return (
        "UNTESTED: training did not complete -- part-(a) remains untested, not "
        f"bounded. Recorded cumulative_steps_trained={steps} with source "
        f"verdicts={source_verdicts}; rerun the bounded training before "
        "green-lighting the matched-compute part-(b) comparison."
    )


def _recommended_part_b_setup(root: Path, exp3727: Mapping[str, Any] | None, stable: bool) -> str:
    flop_budget = _get_nested(exp3727 or {}, "matched_compute_report.ebt_total_flops")
    flop_text = f"{flop_budget} FLOPs from Exp 3727's harness" if flop_budget is not None else "Exp 3727 harness FLOP budget"
    checkpoint = _checkpoint_evidence(root)
    if stable:
        return f"Run part-(b) from {checkpoint} using {flop_text}."
    return f"Do not run part-(b) until part-(a) completes; when it does, use {checkpoint} and {flop_text}."


def _checkpoint_evidence(root: Path) -> str:
    for rel in (
        Path("results/experiment_3735_checkpoint.pt"),
        Path("results/experiment_3734_checkpoint.pt"),
    ):
        if (root / rel).exists():
            return rel.as_posix()
    return "the completed Exp 3735 checkpoint evidence"


def _selected_loss_curve(
    exp3734: Mapping[str, Any] | None,
    exp3735: Mapping[str, Any] | None,
) -> Any:
    if exp3735 is not None and exp3735.get("ebt_loss_curve"):
        return exp3735.get("ebt_loss_curve")
    if exp3734 is not None:
        return exp3734.get("ebt_loss_curve")
    return []


def _finite_curve(values: Any) -> list[float]:
    if not isinstance(values, list):
        return []
    curve: list[float] = []
    for value in values:
        if not _finite_number(value):
            return []
        curve.append(float(value))
    return curve


def _non_runaway_convergence(curve: list[float]) -> bool:
    if len(curve) < 2 or curve[-1] >= curve[0]:
        return False
    if _runaway_collapse(curve):
        return False
    deltas = [curve[i + 1] - curve[i] for i in range(len(curve) - 1)]
    decreases = sum(1 for delta in deltas if delta <= 0)
    return decreases >= max(1, math.ceil(0.6 * len(deltas)))


def _runaway_collapse(curve: list[float]) -> bool:
    if len(curve) < 2:
        return False
    if any(abs(value) > 1_000_000 for value in curve):
        return True
    scale = max(1.0, abs(curve[0]))
    return curve[-1] < 0 and abs(curve[-1]) > 100.0 * scale


def _no_nan_inf_or_divergence(
    artifacts: list[Mapping[str, Any]],
    finite_curve: list[float],
    raw_curve: Any,
) -> bool:
    if any(artifact.get("nan_or_divergence_events") is True for artifact in artifacts):
        return False
    if isinstance(raw_curve, list) and raw_curve and not finite_curve:
        return False
    return not _source_verdict_contains(artifacts, "diverged")


def _gradient_norms_bounded(artifacts: list[Mapping[str, Any]]) -> bool:
    seen_gradient_evidence = False
    for artifact in artifacts:
        if artifact.get("gradient_norms_bounded") is True:
            return True
        if artifact.get("gradient_norms_bounded") is False:
            seen_gradient_evidence = True
        for field in ("gradient_norm_curve", "gradient_norms", "ebt_gradient_norms", "ebt_grad_norms"):
            values = artifact.get(field)
            if isinstance(values, list):
                seen_gradient_evidence = True
                curve = _finite_curve(values)
                if curve and max(curve) <= 100.0 and not _curve_explodes(curve):
                    return True
    return False if seen_gradient_evidence else False


def _gradient_norms_exploded(artifacts: list[Mapping[str, Any]]) -> bool:
    for artifact in artifacts:
        if artifact.get("gradient_norms_bounded") is False:
            return True
        for field in ("gradient_norm_curve", "gradient_norms", "ebt_gradient_norms", "ebt_grad_norms"):
            values = artifact.get(field)
            if isinstance(values, list):
                curve = _finite_curve(values)
                if not curve or max(curve) > 100.0 or _curve_explodes(curve):
                    return True
    return False


def _curve_explodes(curve: list[float]) -> bool:
    if len(curve) < 2:
        return False
    return curve[-1] > max(100.0, 10.0 * max(1.0, curve[0]))


def _is_blocked(artifact: Mapping[str, Any] | None) -> bool:
    if artifact is None:
        return True
    verdict = str(artifact.get("honest_verdict") or "").lower()
    return verdict.startswith("blocked") or "blocked_" in verdict


def _terminal_complete(artifact: Mapping[str, Any] | None) -> bool:
    return str((artifact or {}).get("honest_verdict") or "").startswith("complete:")


def _source_verdict_contains(artifacts: list[Mapping[str, Any]], marker: str) -> bool:
    return any(marker in str(artifact.get("honest_verdict") or "").lower() for artifact in artifacts)


def _citations(root: Path, upstream: Mapping[int, Mapping[str, Any] | None]) -> list[JsonDict]:
    citations: list[JsonDict] = []
    for experiment_id, rel_path in (
        (3727, EXP3727_REL_PATH),
        (3729, EXP3729_REL_PATH),
        (3734, EXP3734_REL_PATH),
        (3735, EXP3735_REL_PATH),
    ):
        artifact = upstream.get(experiment_id)
        if artifact is None:
            continue
        path = root / rel_path
        fields = [
            field
            for field in UPSTREAM_FIELDS[experiment_id]
            if _get_nested(artifact, field) is not None
        ]
        citations.append(
            {
                "experiment_id": experiment_id,
                "fields_imported": fields,
                "sha256": sha256_file(path),
            }
        )
    return citations


def _validate_citations(citations: Any) -> None:
    if not isinstance(citations, list) or not citations:
        raise ValueError("cited_upstream_artifacts must cite upstream artifacts")
    ids = {item.get("experiment_id") for item in citations if isinstance(item, Mapping)}
    if 3727 not in ids:
        raise ValueError("cited_upstream_artifacts must cite the Exp 3727 harness")
    for item in citations:
        if not isinstance(item, Mapping):
            raise ValueError("each citation must be an object")
        if not item.get("fields_imported"):
            raise ValueError("each citation must include fields_imported")
        if not _is_sha256(item.get("sha256")):
            raise ValueError("each citation must include a sha256 hex string")


def _get_nested(artifact: Mapping[str, Any], field: str) -> Any:
    current: Any = artifact
    for part in field.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _compact_verify_report(report: Mapping[str, Any]) -> JsonDict:
    flags_raw = report.get("flags", [])
    flags = [dict(flag) for flag in flags_raw if isinstance(flag, Mapping)] if isinstance(flags_raw, list) else []
    return {
        "flag_count": len(flags),
        "critical_flag_count": sum(
            1 for flag in flags if str(flag.get("severity", "")).lower() == "critical"
        ),
        "flags": flags,
    }


def _critical_flag_count(report: Any) -> int:
    if not isinstance(report, Mapping):
        return 0
    count = report.get("critical_flag_count")
    if isinstance(count, int) and not isinstance(count, bool):
        return count
    flags = report.get("flags", [])
    if not isinstance(flags, list):
        return 0
    return sum(
        1
        for flag in flags
        if isinstance(flag, Mapping) and str(flag.get("severity", "")).lower() == "critical"
    )


def _run_adversarial_verify(path: Path) -> JsonDict:
    verifier_path = SOURCE_REPO_ROOT / "scripts" / "adversarial_verify.py"
    spec = importlib.util.spec_from_file_location("carnot_adversarial_verify_exp3736", verifier_path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"could not load adversarial verifier from {verifier_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.verify_artifact(path)
    if not isinstance(report, dict):  # pragma: no cover
        raise RuntimeError("adversarial verifier returned a non-object report")
    return report


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _duration(started_s: float | None, now_s: float | None) -> float:
    start = time.time() if started_s is None else float(started_s)
    end = time.time() if now_s is None else float(now_s)
    return round(max(0.0001, end - start), 6)


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _has_live_model_markers(artifact: Mapping[str, Any]) -> bool:
    encoded = json.dumps(artifact, sort_keys=True)
    forbidden = ("model_specs", "target_model", "models_tested", "GGUF", "torch.cuda", ".cuda(")
    return any(marker in encoded for marker in forbidden)


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
