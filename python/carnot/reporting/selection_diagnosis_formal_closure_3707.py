"""Formal closure record for the energy discrimination-vs-selection diagnosis.

The purpose of Exp 3707 is deliberately narrow: preserve the evidence that the
selection question is now low-value to re-grind, and recommend operator-curated
retirement. It reads prior artifacts and reference text only; it does not score
candidates, generate a corpus, or edit the exclusion manifest.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_REL_PATH = Path(
    "results/experiment_3707_selection_diagnosis_formal_closure.json"
)
RANDOM_SEED = 3707
INFERENCE_SUBSTRATE = (
    "aggregation_from_upstream_artifacts (principle: reads prior artifacts; "
    "no live inference; no compute-bound marker)."
)
CLOSED_VERDICT = (
    "complete: selection_diagnosis_formally_closed_retirement_recommended_to_operator"
)
OPEN_VERDICT = "complete: selection_diagnosis_cannot_close_open_question"
TERMINAL_VERDICTS = (CLOSED_VERDICT, OPEN_VERDICT)

EXP3672_REL_PATH = Path("results/experiment_3672_ensemble_selection_where_sc_weak.json")
EXP3682_REL_PATH = Path("results/experiment_3682_discrimination_vs_selection_gap.json")
EXP3694_REL_PATH = Path("results/experiment_3694_selection_gap_proper_rediagnosis.json")
REFERENCES_REL_PATH = Path("research-references.md")
ROADMAP_REL_PATH = Path("research-roadmap.yaml")
MANIFEST_REL_PATH = Path("ops/exclusion_manifest.yaml")
CONDUCTOR_REL_PATH = Path("scripts/research_conductor.py")

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "earned_negative_source",
    "failed_diagnosis_attempts",
    "bounded_thesis_basis",
    "operator_retirement_recommendation",
    "question_closed",
    "manifest_unmodified_assert",
    "adversarial_verify_clean",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix for reconciler classification.",
    "inference_substrate": (
        "Reads prior artifacts and reference text only; no live inference and "
        "no compute-bound marker should be present."
    ),
    "earned_negative_source": (
        "exp3672 is the terminal earned-negative the closure rests on."
    ),
    "failed_diagnosis_attempts": (
        "exp3682 was degenerate and exp3694 was blocked, documenting why a "
        "third diagnosis grind is low-value."
    ),
    "bounded_thesis_basis": (
        "The settled-bounded thesis plus arXiv:2512.23067 and arXiv:2605.30619 "
        "are the published reason the decoupling is expected."
    ),
    "operator_retirement_recommendation": (
        "Records the recommended exclusion-manifest retirement text for "
        "operator action so future planners do not re-propose this diagnosis."
    ),
    "question_closed": (
        "Bare bool. True iff the closure is recorded with a retirement "
        "recommendation rather than a third diagnosis."
    ),
    "manifest_unmodified_assert": (
        "Asserts ops/exclusion_manifest.yaml was not edited by this task."
    ),
    "adversarial_verify_clean": "True iff no critical adversarial flag.",
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Drift detection.",
    "duration_s": "Plausibility floor.",
}


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    adversarial_verify_clean: bool = False,
    adversarial_verify_report: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Read the real repo inputs and build the Exp 3707 artifact."""

    root_path = Path(root)
    started = time.time() if started_s is None else started_s
    now = time.time() if now_s is None else now_s
    manifest = root_path / MANIFEST_REL_PATH
    conductor = root_path / CONDUCTOR_REL_PATH
    return build_artifact_from_inputs(
        exp3672=_read_json_object(root_path / EXP3672_REL_PATH),
        exp3682=_read_json_object(root_path / EXP3682_REL_PATH),
        exp3694=_read_json_object(root_path / EXP3694_REL_PATH),
        references_text=_read_text(root_path / REFERENCES_REL_PATH),
        roadmap_text=_read_text(root_path / ROADMAP_REL_PATH),
        manifest_hash_before=_sha256_path(manifest),
        manifest_hash_after=_sha256_path(manifest),
        conductor_hash_before=_sha256_path(conductor),
        conductor_hash_after=_sha256_path(conductor),
        started_s=started,
        now_s=now,
        adversarial_verify_clean=adversarial_verify_clean,
        adversarial_verify_report=adversarial_verify_report or {"flags": []},
    )


def build_artifact_from_inputs(
    *,
    exp3672: Mapping[str, Any],
    exp3682: Mapping[str, Any],
    exp3694: Mapping[str, Any],
    references_text: str,
    roadmap_text: str,
    manifest_hash_before: str,
    manifest_hash_after: str,
    conductor_hash_before: str,
    conductor_hash_after: str,
    started_s: float,
    now_s: float,
    adversarial_verify_clean: bool,
    adversarial_verify_report: Mapping[str, Any],
) -> JsonDict:
    """Construct a closure or open-question artifact from supplied evidence."""

    manifest_unmodified = manifest_hash_before == manifest_hash_after
    conductor_unmodified = conductor_hash_before == conductor_hash_after
    earned_negative = _earned_negative(exp3672)
    degenerate = _degenerate_attempt(exp3682)
    blocked = _blocked_attempt(exp3694)
    thesis = _bounded_thesis_present(references_text, roadmap_text)
    question_closed = all(
        [earned_negative, degenerate, blocked, thesis, manifest_unmodified, conductor_unmodified]
    )
    artifact: JsonDict = {
        "schema": "carnot.selection_diagnosis_formal_closure.v1",
        "honest_verdict": CLOSED_VERDICT if question_closed else OPEN_VERDICT,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "earned_negative_source": _earned_negative_source(exp3672, earned_negative),
        "failed_diagnosis_attempts": _failed_attempts_source(exp3682, exp3694, degenerate, blocked),
        "bounded_thesis_basis": _bounded_thesis_basis(references_text, roadmap_text, thesis),
        "operator_retirement_recommendation": _operator_recommendation(question_closed),
        "question_closed": question_closed,
        "manifest_unmodified_assert": manifest_unmodified,
        "scripts_research_conductor_modified": not conductor_unmodified,
        "adversarial_verify_clean": adversarial_verify_clean,
        "adversarial_verify_report": compact_adversarial_report(adversarial_verify_report),
        "closure_evidence": {
            "exp3672_earned_negative": earned_negative,
            "exp3682_methodology_failed_degenerate": degenerate,
            "exp3694_methodology_failed_blocked": blocked,
            "bounded_thesis_and_references_present": thesis,
            "operator_manifest_left_unmodified": manifest_unmodified,
            "research_conductor_left_unmodified": conductor_unmodified,
        },
        "source_artifact_hashes": {
            "exp3672": _sha256_payload(exp3672),
            "exp3682": _sha256_payload(exp3682),
            "exp3694": _sha256_payload(exp3694),
        },
        "field_principles": dict(FIELD_PRINCIPLES),
        "random_seed": RANDOM_SEED,
        "duration_s": max(float(now_s) - float(started_s), 0.0001),
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build, adversarial-check, validate, and persist the Exp 3707 artifact."""

    root_path = Path(root)
    started = time.time() if started_s is None else started_s
    now = time.time() if now_s is None else now_s
    output = _repo_path(root_path, Path(output_path))
    artifact = build_artifact(root_path, started_s=started, now_s=now)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = run_adversarial_verify_report(output)
    artifact["adversarial_verify_report"] = compact_adversarial_report(report)
    artifact["adversarial_verify_clean"] = adversarial_report_is_clean(report)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Reject schema drift that would make the closure ambiguous to reconcile."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing: raise ValueError(f"missing required fields: {missing}")
    if artifact["honest_verdict"] not in TERMINAL_VERDICTS: raise ValueError("terminal verdict")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE: raise ValueError("inference_substrate")
    if type(artifact["question_closed"]) is not bool: raise ValueError("question_closed must be bare bool")
    if type(artifact["manifest_unmodified_assert"]) is not bool: raise ValueError("manifest_unmodified_assert must be bare bool")
    if type(artifact["adversarial_verify_clean"]) is not bool: raise ValueError("adversarial_verify_clean must be bare bool")
    duration = artifact["duration_s"]
    if isinstance(duration, bool) or not isinstance(duration, int | float) or duration < 0.0001: raise ValueError("duration_s")
    checksum = artifact["reproducibility_checksum"]
    if not isinstance(checksum, str) or len(checksum) != 64: raise ValueError("reproducibility_checksum")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping): raise ValueError("field_principles")
    principle_missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in principles]
    if principle_missing: raise ValueError(f"field_principles missing: {principle_missing}")


def run_adversarial_verify_report(path: Path) -> JsonDict:
    """Run the repository adversarial verifier against an artifact path."""

    verifier_path = REPO_ROOT / "scripts" / "adversarial_verify.py"
    spec = importlib.util.spec_from_file_location("carnot_adversarial_verify_3707", verifier_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {verifier_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return dict(module.verify_artifact(path))


def compact_adversarial_report(report: Mapping[str, Any]) -> JsonDict:
    """Keep only the stable flag summary needed by the closure artifact."""

    flags = [dict(flag) for flag in report.get("flags", []) if isinstance(flag, Mapping)]
    return {"flag_count": len(flags), "flags": flags}


def adversarial_report_is_clean(report: Mapping[str, Any]) -> bool:
    """Return true when the adversarial report contains no critical flag."""

    flags = report.get("flags", [])
    if not isinstance(flags, list):
        return False
    return not any(
        isinstance(flag, Mapping) and str(flag.get("severity", "")).lower() == "critical"
        for flag in flags
    )


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the closure payload while excluding fields recomputed after verify."""

    payload = {
        key: value
        for key, value in artifact.items()
        if key not in {"reproducibility_checksum", "adversarial_verify_report"}
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _earned_negative(exp3672: Mapping[str, Any]) -> bool:
    verdict = str(exp3672.get("honest_verdict", "")).lower()
    ensemble = _coerce_float(exp3672.get("ensemble_selection_accuracy"))
    sc = _coerce_float(exp3672.get("sc_accuracy"))
    oracle = _coerce_float(exp3672.get("oracle_bestofn_accuracy"))
    flips = _coerce_float(exp3672.get("flip_count"))
    return (
        "earned_negative" in verdict
        and exp3672.get("positive_control_valid") is True
        and ensemble is not None
        and sc is not None
        and oracle is not None
        and flips is not None
        and ensemble < sc
        and oracle > sc
        and flips > 0
    )


def _degenerate_attempt(exp3682: Mapping[str, Any]) -> bool:
    verdict = str(exp3682.get("honest_verdict", "")).lower()
    values = [
        _coerce_float(exp3682.get("ensemble_selection_accuracy")),
        _coerce_float(exp3682.get("selection_accuracy_per_question_normalized")),
        _coerce_float(exp3682.get("self_certainty_selection_accuracy")),
    ]
    identical_fixes = None not in values and len({round(float(value), 6) for value in values}) == 1
    return (
        "selection_gap" in verdict
        and exp3682.get("selection_gap_closed") is False
        and (exp3682.get("flagged_adversarial") is True or identical_fixes)
    )


def _blocked_attempt(exp3694: Mapping[str, Any]) -> bool:
    verdict = str(exp3694.get("honest_verdict", "")).lower()
    reason = str(exp3694.get("block_reason", "")).lower()
    examples = _coerce_float(exp3694.get("n_examples"))
    return (
        "blocked_no_multi_candidate_corpus" in verdict
        or ("multi-candidate" in reason and "corpus" in reason and (examples is None or examples == 0))
    )


def _bounded_thesis_present(references_text: str, roadmap_text: str) -> bool:
    combined = f"{references_text}\n{roadmap_text}"
    return (
        "arXiv:2512.23067" in combined
        and "arXiv:2605.30619" in combined
        and (
            "project_energy_selection_thesis_bounded" in combined
            or "settled-bounded" in combined
        )
    )


def _earned_negative_source(exp3672: Mapping[str, Any], valid: bool) -> str:
    if valid:
        return (
            "exp3672 earned-negative: ensemble selection underperformed "
            "self-consistency despite valid oracle headroom and nonzero flips."
        )
    return "exp3672 evidence incomplete or contradictory; earned-negative not established."


def _failed_attempts_source(
    exp3682: Mapping[str, Any],
    exp3694: Mapping[str, Any],
    degenerate: bool,
    blocked: bool,
) -> str:
    exp3682_status = (
        "exp3682 degenerate tautology/no-op methodology failure"
        if degenerate
        else f"exp3682 status insufficient: {exp3682.get('honest_verdict')}"
    )
    exp3694_status = (
        "exp3694 blocked on missing multi-candidate corpus"
        if blocked
        else f"exp3694 status insufficient: {exp3694.get('honest_verdict')}"
    )
    return f"{exp3682_status}; {exp3694_status}."


def _bounded_thesis_basis(references_text: str, roadmap_text: str, present: bool) -> str:
    if present:
        return (
            "project_energy_selection_thesis_bounded says energy-selection is "
            "settled-bounded; arXiv:2512.23067 and arXiv:2605.30619 explain why "
            "discrimination and best-of-N selection utility can decouple."
        )
    observed = []
    for token in ["project_energy_selection_thesis_bounded", "arXiv:2512.23067", "arXiv:2605.30619"]:
        observed.append(f"{token}={'present' if token in references_text + roadmap_text else 'missing'}")
    return "bounded thesis basis incomplete: " + ", ".join(observed) + "."


def _operator_recommendation(question_closed: bool) -> str:
    if question_closed:
        return (
            "OPERATOR RECOMMENDATION: add the discrimination-vs-selection "
            "diagnosis question to ops/exclusion_manifest.yaml under operator "
            "authority so future planners do not re-propose a third diagnosis. "
            "Caveat: a new human-seeded thesis may reopen it; a loop-self-"
            "initiated re-grind should not."
        )
    return (
        "Retirement not recommended as a completed action: one or more closure "
        "inputs is missing or contradictory, so the question remains open."
    )


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _repo_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _read_json_object(path: Path) -> JsonDict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _sha256_path(path: Path) -> str:
    data = path.read_bytes() if path.exists() else b""
    return hashlib.sha256(data).hexdigest()


def _sha256_payload(payload: Mapping[str, Any]) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()
