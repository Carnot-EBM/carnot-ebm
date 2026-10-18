"""Exp 1473 adversarial audit for live telemetry and BEAVER-lite evidence.

This module asks a deliberately skeptical question: can the `.113` telemetry
and BEAVER-lite artifacts satisfy their local gates without measuring a real
verifier signal?  The answer matters because logprob availability, response
shape, prompt family, and deterministic bound bookkeeping can all look useful
while only proving that the experiment pipeline ran.

Spec: REQ-VERIFY-1473, SCENARIO-VERIFY-1473.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from carnot.reporting.halt_spilled_energy_telemetry_diagnostic import (
    build_diagnostic_payload,
    evaluate_rank_signals,
)


DEFAULT_RUN_DATE = "20260507"
DEFAULT_EXP1468_ARTIFACT_PATH = Path(
    "results/experiment_1468_live_sota_logprob_telemetry_preflight.json"
)
DEFAULT_EXP1469_ARTIFACT_PATH = Path(
    "results/experiment_1469_halt_spilled_energy_telemetry_diagnostic.json"
)
DEFAULT_EXP1470_ARTIFACT_PATH = Path(
    "results/experiment_1470_beaver_lite_deterministic_bound_smoke.json"
)
DEFAULT_EXP1468_MANIFEST_PATH = Path("results/live_sota_telemetry_manifest_1468.jsonl")
DEFAULT_OUTPUT_PATH = Path(
    "results/experiment_1473_live_telemetry_adversarial_validity_audit.json"
)
DEFAULT_AUDIT_NOTE_PATH = Path(
    "docs/research-notes/live_telemetry_adversarial_validity_audit.md"
)
REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "artifacts_audited",
    "length_confound_checked",
    "format_confound_checked",
    "prompt_family_confound_checked",
    "mock_logprob_leakage_checked",
    "superficial_baseline_results",
    "telemetry_validity_verdict",
    "claim_allowed",
    "audit_note_path",
    "honest_verdict",
)
SUPERFICIAL_BASELINE_FEATURES: tuple[str, ...] = (
    "completion_tokens",
    "token_count",
    "response_char_length",
    "json_like_response",
    "exact_answer_format",
    "prompt_family_fover",
    "prompt_family_gsm8k",
    "mock_logprobs",
    "live_logprobs",
)
LENGTH_BASELINE_FEATURES: tuple[str, ...] = (
    "completion_tokens",
    "token_count",
    "response_char_length",
)
FORMAT_BASELINE_FEATURES: tuple[str, ...] = ("json_like_response", "exact_answer_format")
PROMPT_FAMILY_FEATURES: tuple[str, ...] = ("prompt_family_fover", "prompt_family_gsm8k")
MOCK_LIVE_FEATURES: tuple[str, ...] = ("mock_logprobs", "live_logprobs")
NEAR_CONFOUND_MARGIN = 0.10

JsonDict = dict[str, Any]


def _write_json(path: Path, payload: Mapping[str, Any]) -> JsonDict:
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = dict(payload)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def _resolve_path(project_root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else project_root / candidate


def _display_path(project_root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:  # pragma: no cover - only hit for caller-supplied external paths.
        return str(path)


def _read_json(path: Path) -> JsonDict:
    if not path.is_file():  # pragma: no cover - the experiment inputs are mandatory.
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _read_jsonl(path: Path) -> list[JsonDict]:
    if not path.is_file():  # pragma: no cover - the experiment manifest is mandatory.
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _artifact_summary(project_root: Path, label: str, path: Path, artifact: Mapping[str, Any]) -> JsonDict:
    return {
        "label": label,
        "path": _display_path(project_root, path),
        "status": artifact.get("status", "missing"),
        "honest_verdict": artifact.get("honest_verdict"),
    }


def write_in_progress_artifact(
    path: str | Path = DEFAULT_OUTPUT_PATH,
    *,
    audit_note_path: str | Path = DEFAULT_AUDIT_NOTE_PATH,
    project_root: str | Path = Path("."),
) -> JsonDict:
    """Write the durable startup marker before reading source artifacts.

    The conductor may be interrupted at any point.  A status-only artifact
    makes that interruption visible instead of letting a missing file look like
    the experiment was never attempted.

    Spec: REQ-VERIFY-1473
    """
    root = Path(project_root)
    output = _resolve_path(root, path)
    note = _resolve_path(root, audit_note_path)
    return _write_json(
        output,
        {
            "status": "in_progress",
            "artifacts_audited": [],
            "length_confound_checked": False,
            "format_confound_checked": False,
            "prompt_family_confound_checked": False,
            "mock_logprob_leakage_checked": False,
            "superficial_baseline_results": {},
            "telemetry_validity_verdict": "in_progress",
            "claim_allowed": False,
            "audit_note_path": _display_path(root, note),
            "honest_verdict": "in_progress",
        },
    )


def _augment_cases_with_superficial_features(cases: Sequence[JsonDict], rows: Sequence[JsonDict]) -> list[JsonDict]:
    augmented: list[JsonDict] = []
    for case, row in zip(cases, rows, strict=False):
        copied = dict(case)
        features = dict(copied.get("features") or {})
        family = str(row.get("family", ""))
        source = str(row.get("generation_source", ""))
        features["token_count"] = features.get("completion_tokens", 0.0)
        features["prompt_family_fover"] = 1.0 if family == "fover_style" else 0.0
        features["prompt_family_gsm8k"] = 1.0 if family == "gsm8k_style" else 0.0
        features["mock_logprobs"] = 1.0 if source != "live_sota_llamacpp" else 0.0
        features["live_logprobs"] = 1.0 if source == "live_sota_llamacpp" else 0.0
        copied["features"] = features
        augmented.append(copied)
    return augmented


def _score(signal: Mapping[str, Any] | None) -> float:
    if not signal:
        return 0.0
    return float(signal.get("oriented_auroc") or 0.0)


def _reported_signal(exp1469_artifact: Mapping[str, Any], cases: Sequence[JsonDict], label_key: str) -> JsonDict | None:
    rank_signal = ((exp1469_artifact.get("auroc_or_rank_signal") or {}).get("rank_signal") or {})
    best = rank_signal.get("best_signal")
    if isinstance(best, dict):
        return dict(best)
    computed = evaluate_rank_signals(cases, label_key=label_key)
    return computed.get("best_signal")


def _baseline_summary(
    cases: Sequence[JsonDict],
    *,
    features: Sequence[str],
    label_key: str,
) -> JsonDict:
    return evaluate_rank_signals(cases, candidate_features=features, label_key=label_key)


def _confound_result(summary: Mapping[str, Any], proposed_score: float) -> JsonDict:
    best = summary.get("best_signal")
    best_score = _score(best if isinstance(best, Mapping) else None)
    fail = best_score >= proposed_score or best_score >= proposed_score - NEAR_CONFOUND_MARGIN
    return {
        "checked": True,
        "verdict": "fail" if fail else "pass",
        "best_signal": best,
        "best_oriented_auroc": best_score,
        "near_margin": NEAR_CONFOUND_MARGIN,
    }


def audit_telemetry_confounds(
    exp1468_rows: Sequence[JsonDict],
    exp1469_artifact: Mapping[str, Any],
    *,
    run_date: str,
) -> JsonDict:
    """Compare reported telemetry evidence against superficial baselines.

    The audit uses the same row-level feature extraction as Exp 1469, then adds
    deliberately boring predictors: length, token count, format flags, prompt
    family, and whether logprobs were mock or live.  If those predictors match
    the proposed diagnostic, the telemetry evidence is not a verifier signal.

    Spec: REQ-VERIFY-1473, SCENARIO-VERIFY-1473
    """
    diagnostic = build_diagnostic_payload(exp1468_rows, run_date=run_date)
    cases = _augment_cases_with_superficial_features(diagnostic["cases"], exp1468_rows)
    default_label_key = str(diagnostic["label_key"])
    reported = _reported_signal(exp1469_artifact, cases, default_label_key)
    label_key = str((reported or {}).get("label_key") or default_label_key)
    proposed_score = _score(reported)
    all_baselines = _baseline_summary(
        cases,
        features=SUPERFICIAL_BASELINE_FEATURES,
        label_key=label_key,
    )
    length = _confound_result(
        _baseline_summary(cases, features=LENGTH_BASELINE_FEATURES, label_key=label_key),
        proposed_score,
    )
    format_ = _confound_result(
        _baseline_summary(cases, features=FORMAT_BASELINE_FEATURES, label_key=label_key),
        proposed_score,
    )
    prompt_family = _confound_result(
        _baseline_summary(cases, features=PROMPT_FAMILY_FEATURES, label_key=label_key),
        proposed_score,
    )
    mock_live = _confound_result(
        _baseline_summary(cases, features=MOCK_LIVE_FEATURES, label_key=label_key),
        proposed_score,
    )
    best_baseline = all_baselines.get("best_signal")
    best_baseline_score = _score(best_baseline if isinstance(best_baseline, Mapping) else None)
    matches_or_exceeds = best_baseline_score >= proposed_score
    source_retired = bool(exp1469_artifact.get("diagnostic_lineage_retired"))
    blockers: list[str] = []
    if matches_or_exceeds:
        blockers.append("superficial_baseline_matches_or_exceeds")
    if source_retired:
        blockers.append("source_diagnostic_lineage_retired")
    return {
        "rows_audited": len(exp1468_rows),
        "label_key": label_key,
        "reported_signal": reported,
        "reported_oriented_auroc": proposed_score,
        "best_superficial_baseline": best_baseline,
        "best_superficial_oriented_auroc": best_baseline_score,
        "superficial_baseline_matches_or_exceeds": matches_or_exceeds,
        "source_diagnostic_lineage_retired": source_retired,
        "length_confound": length,
        "format_confound": format_,
        "prompt_family_confound": prompt_family,
        "mock_live_confound": mock_live,
        "all_superficial_baselines": all_baselines,
        "claim_blockers": blockers,
    }


def _constraint_is_surface_only(constraint: Mapping[str, Any]) -> bool:
    description = str(constraint.get("description", "")).lower()
    return bool(constraint.get("terminal_only")) and "terminal response text" in description and "integer" in description


def audit_beaver_lite_artifact(exp1470_artifact: Mapping[str, Any]) -> JsonDict:
    """Check BEAVER-lite live/mock provenance and whether its gate is semantic.

    A sound bound over a terminal-integer toy constraint can be a valid smoke
    test, but it is not evidence that Carnot measured semantic verifier
    correctness.  This distinction is the main adversarial check for Exp 1470.

    Spec: REQ-VERIFY-1473
    """
    mode = str(exp1470_artifact.get("mock_or_live_logprobs", ""))
    n_completions = [int(value) for value in exp1470_artifact.get("n_completions", [])]
    constraints = [dict(item) for item in exp1470_artifact.get("prefix_closed_constraint", [])]
    surface_constraint_only = bool(constraints) and all(
        _constraint_is_surface_only(item) for item in constraints
    )
    single_logged_completion_gate = bool(n_completions) and max(n_completions) <= 1
    can_pass_without_real_signal = (
        exp1470_artifact.get("bound_is_sound") is True
        and surface_constraint_only
        and single_logged_completion_gate
    )
    return {
        "mock_or_live_logprobs": mode,
        "mock_logprobs_used": mode == "mock_logprobs",
        "mock_live_label_clear": mode in {"live_exp1468", "mock_logprobs"},
        "bound_is_sound": exp1470_artifact.get("bound_is_sound") is True,
        "unsafe_mass_bounds": exp1470_artifact.get("unsafe_mass_bounds", []),
        "empirical_violation_rates": exp1470_artifact.get("empirical_violation_rates", []),
        "n_completions": n_completions,
        "surface_constraint_only": surface_constraint_only,
        "single_logged_completion_gate": single_logged_completion_gate,
        "can_pass_without_real_verifier_signal": can_pass_without_real_signal,
        "verdict": "fail_surface_constraint_only"
        if can_pass_without_real_signal
        else "pass_no_surface_only_gate",
    }


def _telemetry_verdict(blockers: Sequence[str]) -> str:
    return (
        "invalid_for_headline_claim_superficial_or_mechanical_gate"
        if blockers
        else "validity_not_falsified_by_superficial_baselines"
    )


def build_audit_payload(
    *,
    project_root: Path,
    run_date: str,
    exp1468_path: Path,
    exp1469_path: Path,
    exp1470_path: Path,
    manifest_path: Path,
    audit_note_path: Path,
) -> JsonDict:
    exp1468 = _read_json(exp1468_path)
    exp1469 = _read_json(exp1469_path)
    exp1470 = _read_json(exp1470_path)
    rows = _read_jsonl(manifest_path)
    telemetry = audit_telemetry_confounds(rows, exp1469, run_date=run_date)
    beaver = audit_beaver_lite_artifact(exp1470)
    blockers = list(telemetry["claim_blockers"])
    if beaver["can_pass_without_real_verifier_signal"]:
        blockers.append("beaver_surface_constraint_only")
    claim_allowed = not blockers
    return {
        "schema_version": 1,
        "run_date": run_date,
        "status": "complete",
        "artifacts_audited": [
            _artifact_summary(project_root, "exp1468_live_sota_logprob_telemetry_preflight", exp1468_path, exp1468),
            _artifact_summary(project_root, "exp1469_halt_spilled_energy_telemetry_diagnostic", exp1469_path, exp1469),
            _artifact_summary(project_root, "exp1470_beaver_lite_deterministic_bound_smoke", exp1470_path, exp1470),
        ],
        "length_confound_checked": True,
        "format_confound_checked": True,
        "prompt_family_confound_checked": True,
        "mock_logprob_leakage_checked": beaver["mock_live_label_clear"],
        "superficial_baseline_results": {
            "telemetry": telemetry,
            "beaver_lite": beaver,
            "claim_blockers": blockers,
        },
        "telemetry_validity_verdict": _telemetry_verdict(blockers),
        "claim_allowed": claim_allowed,
        "audit_note_path": _display_path(project_root, audit_note_path),
        "honest_verdict": "telemetry_claim_allowed_after_adversarial_audit"
        if claim_allowed
        else "telemetry_claim_blocked_adversarial_audit",
    }


def render_audit_note(artifact: Mapping[str, Any]) -> str:
    """Render the human-readable pass/fail audit note.

    The JSON artifact is the machine contract.  The markdown note is for the
    next researcher: it names why an attractive metric should or should not be
    promoted into a paper claim.

    Spec: REQ-VERIFY-1473
    """
    results = artifact["superficial_baseline_results"]
    telemetry = results["telemetry"]
    beaver = results["beaver_lite"]
    length = telemetry["length_confound"]
    format_ = telemetry["format_confound"]
    prompt = telemetry["prompt_family_confound"]
    mock_live = telemetry["mock_live_confound"]
    lines = [
        "# Live Telemetry Adversarial Validity Audit",
        "",
        f"Run date: `{artifact['run_date']}`",
        "",
        "## Verdict",
        "",
        f"- Telemetry validity verdict: `{artifact['telemetry_validity_verdict']}`",
        f"- Claim allowed: `{str(artifact['claim_allowed']).lower()}`",
        f"- Honest verdict: `{artifact['honest_verdict']}`",
        "",
        "## Confound Checks",
        "",
        f"- Length/token count: **{str(length['verdict']).upper()}**; best baseline `{(length.get('best_signal') or {}).get('name')}` oriented AUROC `{length['best_oriented_auroc']:.6f}`.",
        f"- JSON/schema or exact-answer format: **{str(format_['verdict']).upper()}**; best baseline `{(format_.get('best_signal') or {}).get('name')}` oriented AUROC `{format_['best_oriented_auroc']:.6f}`.",
        f"- Prompt family: **{str(prompt['verdict']).upper()}**; best baseline `{(prompt.get('best_signal') or {}).get('name')}` oriented AUROC `{prompt['best_oriented_auroc']:.6f}`.",
        f"- Mock/live logprob leakage: **{str(mock_live['verdict']).upper()}** for telemetry baselines; BEAVER label clear `{str(beaver['mock_live_label_clear']).lower()}` with mode `{beaver['mock_or_live_logprobs']}`.",
        "",
        "## BEAVER-Lite",
        "",
        f"- Bound is sound: `{str(beaver['bound_is_sound']).lower()}`.",
        f"- Surface constraint only: `{str(beaver['surface_constraint_only']).lower()}`.",
        f"- Single logged completion gate: `{str(beaver['single_logged_completion_gate']).lower()}`.",
        f"- Can pass without real verifier signal: `{str(beaver['can_pass_without_real_verifier_signal']).lower()}`.",
        "",
        "## Claim Boundary",
        "",
        "The audited artifacts are useful as telemetry plumbing and a deterministic bound smoke, but they do not support a headline claim that live logprob telemetry measured a robust verifier signal.",
    ]
    return "\n".join(lines) + "\n"


def write_audit_note(path: Path, artifact: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_audit_note(artifact), encoding="utf-8")


def run_audit(
    *,
    project_root: str | Path = Path("."),
    run_date: str = DEFAULT_RUN_DATE,
    exp1468_artifact_path: str | Path = DEFAULT_EXP1468_ARTIFACT_PATH,
    exp1469_artifact_path: str | Path = DEFAULT_EXP1469_ARTIFACT_PATH,
    exp1470_artifact_path: str | Path = DEFAULT_EXP1470_ARTIFACT_PATH,
    exp1468_manifest_path: str | Path = DEFAULT_EXP1468_MANIFEST_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    audit_note_path: str | Path = DEFAULT_AUDIT_NOTE_PATH,
) -> JsonDict:
    """Run Exp 1473 and write the terminal JSON and markdown note.

    Spec: REQ-VERIFY-1473, SCENARIO-VERIFY-1473
    """
    root = Path(project_root)
    output = _resolve_path(root, output_path)
    note = _resolve_path(root, audit_note_path)
    exp1468 = _resolve_path(root, exp1468_artifact_path)
    exp1469 = _resolve_path(root, exp1469_artifact_path)
    exp1470 = _resolve_path(root, exp1470_artifact_path)
    manifest = _resolve_path(root, exp1468_manifest_path)
    write_in_progress_artifact(output, audit_note_path=note, project_root=root)
    artifact = build_audit_payload(
        project_root=root,
        run_date=run_date,
        exp1468_path=exp1468,
        exp1469_path=exp1469,
        exp1470_path=exp1470,
        manifest_path=manifest,
        audit_note_path=note,
    )
    write_audit_note(note, artifact)
    return _write_json(output, artifact)


def _parse_args() -> argparse.Namespace:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--run-date", default=DEFAULT_RUN_DATE)
    parser.add_argument("--exp1468-artifact-path", default=str(DEFAULT_EXP1468_ARTIFACT_PATH))
    parser.add_argument("--exp1469-artifact-path", default=str(DEFAULT_EXP1469_ARTIFACT_PATH))
    parser.add_argument("--exp1470-artifact-path", default=str(DEFAULT_EXP1470_ARTIFACT_PATH))
    parser.add_argument("--exp1468-manifest-path", default=str(DEFAULT_EXP1468_MANIFEST_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--audit-note-path", default=str(DEFAULT_AUDIT_NOTE_PATH))
    return parser.parse_args()


def main() -> None:  # pragma: no cover - CLI wrapper.
    args = _parse_args()
    run_audit(
        project_root=args.project_root,
        run_date=args.run_date,
        exp1468_artifact_path=args.exp1468_artifact_path,
        exp1469_artifact_path=args.exp1469_artifact_path,
        exp1470_artifact_path=args.exp1470_artifact_path,
        exp1468_manifest_path=args.exp1468_manifest_path,
        output_path=args.output_path,
        audit_note_path=args.audit_note_path,
    )


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    main()


__all__ = [
    "REQUIRED_ARTIFACT_FIELDS",
    "audit_beaver_lite_artifact",
    "audit_telemetry_confounds",
    "build_audit_payload",
    "render_audit_note",
    "run_audit",
    "write_audit_note",
    "write_in_progress_artifact",
]
