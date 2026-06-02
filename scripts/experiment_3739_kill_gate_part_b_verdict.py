#!/usr/bin/env python3
"""Aggregate Exp 3738 into the Thesis-A kill-gate part-(b) verdict.

Spec: REQ-EBT-3739, SCENARIO-EBT-3739-WIN,
SCENARIO-EBT-3739-BOUNDED, SCENARIO-EBT-3739-INVALID,
SCENARIO-EBT-3739-NOT-RUN.
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
OUTPUT_REL_PATH = Path("results/experiment_3739_kill_gate_part_b_verdict.json")
EXP3736_REL_PATH = Path("results/experiment_3736_real_kill_gate_part_a_verdict.json")
EXP3738_REL_PATH = Path("results/experiment_3738_matched_compute_comparison.json")

RANDOM_SEED = 3739
MIN_HELDOUT_N = 100
INFERENCE_SUBSTRATE = (
    "aggregation_from_upstream_artifacts (principle: a verdict over "
    "upstream numbers, no live model)."
)

PART_A_PASS_VERDICT = (
    "complete: real_kill_gate_part_a_PASS_ebt_trained_stably_green_light_342_"
    "supersedes_exp3729_false_negative"
)
BOUNDED_VERDICT = (
    "complete: kill_gate_part_b_BOUNDED_ebt_does_not_beat_ar_at_equal_"
    "compute_honest_negative"
)
INVALID_VERDICT = (
    "complete: kill_gate_part_b_INVALID_flops_not_matched_rerun_exp3738_"
    "tighter_budget"
)
NOT_RUN_PART_A_VERDICT = "complete: kill_gate_part_b_not_run_part_a_did_not_green_light"
NOT_RUN_EXP3738_VERDICT = "complete: kill_gate_part_b_not_run_exp3738_absent_or_blocked"

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "thesis_a_outcome",
    "ebt_beats_ar_at_matched_compute",
    "accuracy_delta_cited",
    "flops_matched_cited",
    "next_step_recommendation",
    "cited_upstream_artifacts",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal prefix stating the part-(b) thesis outcome "
        "(win / bounded / invalid / not-run)."
    ),
    "inference_substrate": "A verdict over upstream numbers, no live model.",
    "thesis_a_outcome": (
        "The .342 Thesis-A result string "
        "(ebt_beats_ar_at_matched_compute / bounded_at_small_scale / "
        "comparison_invalid / part_b_not_run)."
    ),
    "ebt_beats_ar_at_matched_compute": (
        "BARE bool. True only if delta>0, FLOPs matched, and n>=100."
    ),
    "accuracy_delta_cited": "The Exp 3738 delta carried up for the record.",
    "flops_matched_cited": "Whether the comparison was compute-fair.",
    "next_step_recommendation": "Honest bounded next move from the verdict.",
    "cited_upstream_artifacts": "Provenance from the verdict to upstream numbers.",
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Content hash catches drift.",
    "duration_s": "Wall-clock plausibility floor.",
}

UPSTREAM_FIELDS = {
    3736: [
        "honest_verdict",
        "green_light_342",
        "ebt_trained_stably",
        "training_actually_ran",
        "kill_gate_conclusion",
        "real_run_diagnostics.training_actually_ran",
        "real_run_diagnostics.bounded_run_completed",
        "random_seed",
        "reproducibility_checksum",
        "duration_s",
    ],
    3738: [
        "honest_verdict",
        "accuracy_delta",
        "heldout_accuracy_delta",
        "matched_compute_report.accuracy_delta",
        "ebt_accuracy",
        "ar_accuracy",
        "matched_compute_report.ebt_accuracy",
        "matched_compute_report.ar_accuracy",
        "flops_matched_within_tolerance",
        "flops_matched",
        "matched_compute_report.flops_matched_within_tolerance",
        "matched_compute_report.budget_match.within_tolerance",
        "n_heldout",
        "heldout_n",
        "n_examples",
        "matched_compute_report.n_heldout",
        "matched_compute_report.n_examples",
        "gap_narrowing",
        "gap_trend",
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
    """Build the Exp 3739 verdict from checked-in upstream artifacts."""

    root_path = Path(root)
    part_a_path = root_path / EXP3736_REL_PATH
    part_a = _read_json_object(part_a_path, required=False)
    exp3738_path, exp3738 = _find_exp3738(root_path)
    inputs = _decision_inputs(part_a, exp3738)
    decision = _classify(inputs)
    verify_report = _compact_verify_report(adversarial_verify_report or {"flags": []})

    payload: JsonDict = {
        "schema": "carnot.experiment_3739_kill_gate_part_b_verdict.v1",
        "experiment": 3739,
        "honest_verdict": decision["honest_verdict"],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "thesis_a_outcome": decision["thesis_a_outcome"],
        "ebt_beats_ar_at_matched_compute": decision[
            "ebt_beats_ar_at_matched_compute"
        ],
        "accuracy_delta_cited": inputs["accuracy_delta"],
        "flops_matched_cited": inputs["flops_matched"],
        "n_heldout_cited": inputs["n_heldout"],
        "gap_narrowing_cited": inputs["gap_narrowing"],
        "part_b_not_run_reason": decision["part_b_not_run_reason"],
        "decision_basis": decision["decision_basis"],
        "next_step_recommendation": decision["next_step_recommendation"],
        "cited_upstream_artifacts": _citations(
            (3736, part_a_path, part_a),
            (3738, exp3738_path, exp3738),
        ),
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
    """Validate schema and decision semantics for the Exp 3739 verdict."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        raise ValueError("field_principles must be a mapping")
    missing_principles = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in principles]
    if missing_principles:
        raise ValueError(f"missing field principles: {missing_principles}")
    if not str(artifact.get("honest_verdict") or "").startswith(
        "complete: kill_gate_part_b_"
    ):
        raise ValueError("terminal verdict must be an Exp 3739 terminal verdict")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be the Exp 3739 aggregation substrate")

    outcome = artifact.get("thesis_a_outcome")
    valid_outcomes = {
        "ebt_beats_ar_at_matched_compute",
        "bounded_at_small_scale",
        "comparison_invalid",
        "part_b_not_run",
    }
    if outcome not in valid_outcomes:
        raise ValueError("thesis_a_outcome must be an Exp 3739 outcome")
    beats = artifact.get("ebt_beats_ar_at_matched_compute")
    if not isinstance(beats, bool):
        raise ValueError("ebt_beats_ar_at_matched_compute must be a bare bool")

    delta = artifact.get("accuracy_delta_cited")
    flops = artifact.get("flops_matched_cited")
    n_heldout = artifact.get("n_heldout_cited")
    if delta is not None and not _finite_number(delta):
        raise ValueError("accuracy_delta_cited must be numeric or null")
    if flops is not None and not isinstance(flops, bool):
        raise ValueError("flops_matched_cited must be boolean or null")
    if n_heldout is not None and (
        not isinstance(n_heldout, int) or isinstance(n_heldout, bool)
    ):
        raise ValueError("n_heldout_cited must be an integer or null")
    if not str(artifact.get("next_step_recommendation") or "").strip():
        raise ValueError("next_step_recommendation must be present")

    _validate_decision_semantics(artifact, str(outcome), beats, delta, flops, n_heldout)
    _validate_citations(artifact.get("cited_upstream_artifacts"), str(outcome))

    if artifact.get("random_seed") != RANDOM_SEED:
        raise ValueError("random_seed must equal 3739")
    duration = artifact.get("duration_s")
    if not _finite_number(duration) or float(duration) < 0.0001:
        raise ValueError("duration_s must be numeric with the aggregation plausibility floor")
    if _has_live_model_markers(artifact):
        raise ValueError("live-model markers must not be present")
    if _critical_flag_count(artifact.get("adversarial_verify_report")) > 0:
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
    """Write the final Exp 3739 artifact and return its path."""

    root_path = Path(root)
    output_path = root_path / OUTPUT_REL_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_artifact(root_path, started_s=started_s, now_s=now_s)
    _write_json(output_path, payload)

    report = _run_adversarial_verify(output_path)
    payload["adversarial_verify_report"] = _compact_verify_report(report)
    payload["adversarial_verify_clean"] = (
        payload["adversarial_verify_report"]["critical_flag_count"] == 0
    )
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

    filtered = {
        key: value for key, value in payload.items() if key != "reproducibility_checksum"
    }
    encoded = json.dumps(filtered, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for the Exp 3739 part-(b) verdict."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    output_path = write_artifact(args.root)
    payload = _read_json_object(output_path, required=True)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _decision_inputs(
    part_a: Mapping[str, Any] | None,
    exp3738: Mapping[str, Any] | None,
) -> JsonDict:
    part_a_green = bool(part_a and part_a.get("green_light_342") is True)
    exp3738_blocked = _is_blocked(exp3738)
    accuracy_delta = None if exp3738 is None or exp3738_blocked else _extract_accuracy_delta(exp3738)
    flops_matched = None if exp3738 is None or exp3738_blocked else _extract_flops_matched(exp3738)
    n_heldout = None if exp3738 is None or exp3738_blocked else _extract_n_heldout(exp3738)
    return {
        "part_a_present": part_a is not None,
        "part_a_green": part_a_green,
        "part_a_conclusion": (part_a or {}).get("kill_gate_conclusion"),
        "exp3738_present": exp3738 is not None,
        "exp3738_blocked": exp3738_blocked,
        "exp3738_verdict": (exp3738 or {}).get("honest_verdict"),
        "accuracy_delta": accuracy_delta,
        "flops_matched": flops_matched,
        "n_heldout": n_heldout,
        "gap_narrowing": False
        if exp3738 is None or accuracy_delta is None
        else _gap_narrowing(exp3738, accuracy_delta),
    }


def _classify(inputs: Mapping[str, Any]) -> JsonDict:
    if not inputs["part_a_green"]:
        reason = "part-(a) did not green-light"
        conclusion = str(inputs.get("part_a_conclusion") or "")
        if "training did not complete" in conclusion:
            reason = f"{reason}: training did not complete"
        return _not_run_decision(NOT_RUN_PART_A_VERDICT, reason)
    if not inputs["exp3738_present"]:
        return _not_run_decision(
            NOT_RUN_EXP3738_VERDICT,
            "Exp 3738 artifact absent or unreadable",
        )
    if inputs["exp3738_blocked"]:
        verdict = inputs.get("exp3738_verdict")
        return _not_run_decision(
            NOT_RUN_EXP3738_VERDICT,
            f"Exp 3738 blocked before valid heldout comparison (verdict={verdict})",
        )

    delta = inputs.get("accuracy_delta")
    flops = inputs.get("flops_matched")
    n_heldout = inputs.get("n_heldout")
    if flops is not True:
        return {
            "honest_verdict": INVALID_VERDICT,
            "thesis_a_outcome": "comparison_invalid",
            "ebt_beats_ar_at_matched_compute": False,
            "part_b_not_run_reason": None,
            "decision_basis": (
                "INVALID: FLOPs were not matched within tolerance, so the "
                "comparison has a compute confound; preserve the delta but do "
                "not call a winner."
            ),
            "next_step_recommendation": (
                "Re-run Exp 3738 with a tighter budget match before calling a winner."
            ),
        }
    if delta is None:
        return _invalid_missing_metric("accuracy_delta is missing")
    if n_heldout is None or n_heldout < MIN_HELDOUT_N:
        return _invalid_missing_metric(f"n_heldout={n_heldout} is below the 100-case floor")
    if delta > 0:
        return {
            "honest_verdict": (
                "complete: kill_gate_part_b_ebt_BEATS_ar_at_matched_compute_"
                f"delta_{_format_delta(delta)}_n{n_heldout}_bounded_scaleup_recommended"
            ),
            "thesis_a_outcome": "ebt_beats_ar_at_matched_compute",
            "ebt_beats_ar_at_matched_compute": True,
            "part_b_not_run_reason": None,
            "decision_basis": (
                f"WIN: accuracy_delta={delta} with matched FLOPs and "
                f"n_heldout={n_heldout}; the EBT beats AR at equal compute."
            ),
            "next_step_recommendation": (
                "Bounded scale-up: run one pre-registered 2x-training and "
                "heldout expansion using the same matched-FLOP accounting; "
                "do not commit to open-ended scale until that passes."
            ),
        }
    return _bounded_decision(delta, bool(inputs.get("gap_narrowing")))


def _not_run_decision(verdict: str, reason: str) -> JsonDict:
    return {
        "honest_verdict": verdict,
        "thesis_a_outcome": "part_b_not_run",
        "ebt_beats_ar_at_matched_compute": False,
        "part_b_not_run_reason": reason,
        "decision_basis": f"NOT-RUN: {reason}; no part-(b) thesis verdict was measured.",
        "next_step_recommendation": (
            "Do not score part-(b) until part-(a) green-lights and Exp 3738 "
            "exists; rerun bounded training first when training did not complete."
        ),
    }


def _invalid_missing_metric(reason: str) -> JsonDict:
    return {
        "honest_verdict": INVALID_VERDICT,
        "thesis_a_outcome": "comparison_invalid",
        "ebt_beats_ar_at_matched_compute": False,
        "part_b_not_run_reason": None,
        "decision_basis": f"INVALID: {reason}; do not call a winner.",
        "next_step_recommendation": (
            "Re-run Exp 3738 with complete heldout metrics and a tighter budget match."
        ),
    }


def _bounded_decision(delta: float, gap_narrowing: bool) -> JsonDict:
    if gap_narrowing:
        recommendation = (
            "Run exactly one 2x-training attempt because the matched-FLOP gap "
            "is narrowing; keep the same heldout and FLOP accounting and "
            "retire the route if the delta does not turn positive."
        )
    else:
        recommendation = (
            "Retire the route at small scale because the matched-FLOP gap is "
            "flat or negative."
        )
    return {
        "honest_verdict": BOUNDED_VERDICT,
        "thesis_a_outcome": "bounded_at_small_scale",
        "ebt_beats_ar_at_matched_compute": False,
        "part_b_not_run_reason": None,
        "decision_basis": (
            f"BOUNDED: accuracy_delta={delta} at matched FLOPs; "
            "energy-as-generator does NOT beat AR at equal compute at this scale."
        ),
        "next_step_recommendation": recommendation,
    }


def _validate_decision_semantics(
    artifact: Mapping[str, Any],
    outcome: str,
    beats: bool,
    delta: Any,
    flops: Any,
    n_heldout: Any,
) -> None:
    if outcome == "ebt_beats_ar_at_matched_compute":
        if not beats:
            raise ValueError("win outcome must set ebt_beats_ar_at_matched_compute")
        if not (_finite_number(delta) and float(delta) > 0 and flops is True):
            raise ValueError("win outcome requires positive delta and matched FLOPs")
        if not (isinstance(n_heldout, int) and n_heldout >= MIN_HELDOUT_N):
            raise ValueError("win outcome requires n_heldout>=100")
        return
    if beats:
        raise ValueError("non-win outcome cannot set ebt_beats_ar_at_matched_compute")
    if outcome == "bounded_at_small_scale":
        if artifact.get("honest_verdict") != BOUNDED_VERDICT:
            raise ValueError("bounded outcome must use the bounded terminal verdict")
        if not (_finite_number(delta) and float(delta) <= 0 and flops is True):
            raise ValueError("bounded outcome requires non-positive delta at matched FLOPs")
    elif outcome == "comparison_invalid":
        if artifact.get("honest_verdict") != INVALID_VERDICT:
            raise ValueError("invalid outcome must use the invalid terminal verdict")
    elif outcome == "part_b_not_run":
        if artifact.get("honest_verdict") not in {
            NOT_RUN_PART_A_VERDICT,
            NOT_RUN_EXP3738_VERDICT,
        }:
            raise ValueError("not-run outcome must use a not-run terminal verdict")
        if not str(artifact.get("part_b_not_run_reason") or "").strip():
            raise ValueError("not-run outcome must explain the fallback reason")


def _citations(*items: tuple[int, Path | None, Mapping[str, Any] | None]) -> list[JsonDict]:
    citations: list[JsonDict] = []
    for experiment_id, path, artifact in items:
        if path is None or artifact is None:
            continue
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


def _validate_citations(citations: Any, outcome: str) -> None:
    if not isinstance(citations, list) or not citations:
        raise ValueError("cited_upstream_artifacts must cite upstream artifacts")
    for item in citations:
        if not isinstance(item, Mapping):
            raise ValueError("each citation must be an object")
        if not item.get("fields_imported"):
            raise ValueError("each citation must include fields_imported")
        if not _is_sha256(item.get("sha256")):
            raise ValueError("each citation must include a sha256 hex string")
    ids = {item.get("experiment_id") for item in citations if isinstance(item, Mapping)}
    if 3736 not in ids:
        raise ValueError("cited_upstream_artifacts must cite Exp 3736")
    if outcome != "part_b_not_run" and 3738 not in ids:
        raise ValueError("cited_upstream_artifacts must cite Exp 3738")


def _find_exp3738(root: Path) -> tuple[Path | None, JsonDict | None]:
    exact = root / EXP3738_REL_PATH
    if exact.exists():
        return exact, _read_json_object(exact, required=True)
    results_dir = root / "results"
    for path in sorted(results_dir.glob("experiment_3738*.json")):
        return path, _read_json_object(path, required=True)
    return None, None


def _read_json_object(path: Path, *, required: bool) -> JsonDict | None:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _extract_accuracy_delta(artifact: Mapping[str, Any]) -> float | None:
    direct = _first_number(
        artifact,
        [
            "accuracy_delta",
            "accuracy_delta_cited",
            "heldout_accuracy_delta",
            "matched_accuracy_delta",
            "matched_compute_report.accuracy_delta",
        ],
    )
    if direct is not None:
        return direct
    ebt = _first_number(artifact, ["matched_compute_report.ebt_accuracy", "ebt_accuracy"])
    ar = _first_number(artifact, ["matched_compute_report.ar_accuracy", "ar_accuracy"])
    if ebt is None or ar is None:
        return None
    return round(ebt - ar, 12)


def _extract_flops_matched(artifact: Mapping[str, Any]) -> bool | None:
    return _extract_bool(
        artifact,
        [
            "flops_matched_within_tolerance",
            "flops_matched",
            "flops_matched_cited",
            "within_tolerance",
            "matched_compute_report.flops_matched_within_tolerance",
            "matched_compute_report.budget_match.within_tolerance",
        ],
    )


def _extract_n_heldout(artifact: Mapping[str, Any]) -> int | None:
    for field in (
        "n_heldout",
        "heldout_n",
        "n_examples",
        "n_samples",
        "matched_compute_report.n_heldout",
        "matched_compute_report.heldout_n",
        "matched_compute_report.n_examples",
        "matched_compute_report.n_samples",
    ):
        value = _get_nested(artifact, field)
        parsed = _safe_int(value)
        if parsed is not None:
            return parsed
    return None


def _extract_bool(artifact: Mapping[str, Any], fields: list[str]) -> bool | None:
    for field in fields:
        value = _get_nested(artifact, field)
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in {"true", "false"}:
            return value.lower() == "true"
    return None


def _first_number(artifact: Mapping[str, Any], fields: list[str]) -> float | None:
    for field in fields:
        value = _get_nested(artifact, field)
        if _finite_number(value):
            return float(value)
    return None


def _gap_narrowing(artifact: Mapping[str, Any], delta: float) -> bool:
    if artifact.get("gap_narrowing") is True:
        return True
    trend = str(artifact.get("gap_trend") or "").lower()
    if "narrow" in trend:
        return True
    return False


def _get_nested(artifact: Mapping[str, Any], field: str) -> Any:
    current: Any = artifact
    for part in field.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _is_blocked(artifact: Mapping[str, Any] | None) -> bool:
    if artifact is None:
        return False
    verdict = str(artifact.get("honest_verdict") or "").lower()
    return verdict.startswith("blocked") or "blocked_" in verdict


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
    spec = importlib.util.spec_from_file_location("carnot_adversarial_verify_exp3739", verifier_path)
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


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _format_delta(delta: float | None) -> str:
    if delta is None:
        return "unknown"
    return f"{float(delta):.6g}"


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
