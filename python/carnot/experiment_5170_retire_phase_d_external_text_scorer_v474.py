"""Exp 5170: retire Phase D external-text energy/reward verifier scorers.

Spec refs: REQ-REPORT-5170, SCENARIO-REPORT-5170-LINT,
SCENARIO-REPORT-5170-NARROW-SCOPE.

This is a reporting harness, not a new verifier run. It consolidates the Phase D
LoRA-EBM/uPRM/EBRM external generated-text and logprob scorer lineage, proves
the exclusion-manifest entry is load-bearing, and confirms the hidden-state
verifier pilot remains outside the retired scope.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

import yaml

from carnot.experiment_5150_archive_471_activate_472 import (
    CommandResult,
    _bool,
    _list,
    _mapping,
    file_sha256,
    payload_checksum,
    read_json_mapping,
    run_adversarial_verification,
    verification_payload,
    write_json,
)


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5170_retire_phase_d_external_text_scorer_v474.json")
MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
VERIFIER_GAPS_RELATIVE_PATH = Path("ops/verifier_gaps.md")
KNOWN_ISSUES_RELATIVE_PATH = Path("ops/known-issues.md")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
RESEARCH_REFERENCES_RELATIVE_PATH = Path("research-references.md")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
LINTER_RELATIVE_PATH = Path("scripts/exclusion_manifest_lint.py")
LINTER_MODULE_PATH = REPO_ROOT / LINTER_RELATIVE_PATH

EXPERIMENT = "experiment_5170_retire_phase_d_external_text_scorer_v474"
EXPERIMENT_ID = "exp5170-retire-phase-d-external-text-scorer-v474"
MILESTONE = "2026.07.474"
SCHEMA = "carnot.experiment_5170_retire_phase_d_external_text_scorer.v1"
RANDOM_SEED = 5170
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
ENTRY_ID = "phase_d_external_text_scorer_retired_exp5163_v474"
COMPLETE_VERDICT = (
    "complete: phase_d_external_text_scorer_scope_retired_and_hidden_state_exception_preserved"
)
INCOMPLETE_VERDICT = "complete: phase_d_external_text_scorer_retirement_hygiene_incomplete"
TERMINAL_PREFIXES = ("complete:", "complete_", "success:", "success_")
SANCTIONED_EXCEPTION_DOC_MARKER = (
    "2026-07-02 retirement note: Phase D external-text scorer moat closed, "
    "hidden-state verifiers remain open"
)

SOURCE_RESULT_PATHS: dict[str, Path] = {
    "exp4940": Path("results/experiment_4940_distributional_energy_verifier_executable_spec.json"),
    "distributional_energy_verifier_musr": Path("results/distributional_energy_verifier_musr.json"),
    "exp5003": Path("results/experiment_5003_lora_ebm_scorer_musr.json"),
    "exp5004": Path("results/experiment_5004_uprm_replication.json"),
    "exp5005": Path("results/experiment_5005_ebrm_uncertainty_verifier.json"),
    "exp5007": Path("results/experiment_5007_moat_gate_resolution.json"),
    "exp5015": Path("results/experiment_5015_genuine_sc_baseline_fix.json"),
    "exp5017": Path("results/experiment_5017_lora_ebm_scorer_musr_v2.json"),
    "exp5018": Path("results/experiment_5018_uprm_replication_v2.json"),
    "exp5022": Path("results/experiment_5022_moat_gate_resolution_v2.json"),
    "exp5029": Path("results/experiment_5029_shared_logprob_candidate_cache_v2.json"),
    "exp5031": Path("results/experiment_5031_lora_ebm_scorer_musr_v3.json"),
    "exp5032": Path("results/experiment_5032_uprm_replication_v3.json"),
    "exp5033": Path("results/experiment_5033_ebrm_uncertainty_verifier_v3.json"),
    "exp5036": Path("results/experiment_5036_moat_gate_resolution_v3.json"),
    "exp5045": Path("results/experiment_5045_powered_lora_ebm_eorm_musr.json"),
    "exp5046": Path("results/experiment_5046_vpr_process_reward_repair.json"),
    "exp5047": Path("results/experiment_5047_kan_purm_energy_calibration.json"),
    "exp5050": Path("results/experiment_5050_moat_gate_resolution_v464.json"),
    "exp5059": Path("results/experiment_5059_d1_sota_refresh_audit.json"),
    "exp5060": Path("results/experiment_5060_second_corpus_audit_v2.json"),
    "exp5063": Path("results/experiment_5063_moat_gate_resolution_v465.json"),
    "exp5072": Path("results/experiment_5072_uprm_logprob_cache_v466.json"),
    "exp5086": Path("results/experiment_5086_uprm_logprob_cache_retry_v467.json"),
    "exp5087": Path("results/experiment_5087_uprm_process_verifier_retry_v467.json"),
    "exp5088": Path("results/experiment_5088_temporal_consistency_prm_v467.json"),
    "exp5126": Path("results/experiment_5126_distributional_energy_ranker_v470.json"),
    "exp5163": Path("results/experiment_5163_mmlu_pro_verifier_rescale_v473.json"),
}

EXPECTED_EXPERIMENT_IDS = [
    source_id for source_id in SOURCE_RESULT_PATHS if source_id.startswith("exp")
]
EXPECTED_BLOCKED_PATTERNS = [
    "train lora ebm scorer v2 on off-arc reasoning corpus",
    "rerun uprm text scorer on off-arc reasoning corpus",
    "rerun ebrm external text reward scorer on off-arc reasoning corpus",
    "phase d external text scorer rerun",
]
EXPERIMENT_SCOPE = (
    "external-TEXT-based energy/reward verifier scoring (LoRA-EBM holistic scorer / "
    "uPRM / EBRM style) vs. self-consistency on off-ARC reasoning corpora"
)
RETIREMENT_REASON = (
    "retire_if_same_verdict: Exp 5163 is the terminal Phase D continuation after seven "
    "milestones of null or marginal external generated-text/logprob scorer evidence. "
    "LoRA-EBM, uPRM, EBRM, process-reward, and distributional-energy ranker attempts "
    "failed to produce a decision-grade win over genuine tuned self-consistency on "
    "headroom-present off-ARC corpora. Future reruns need an operator_override citing "
    "a genuinely different mechanism; hidden-state/internal-representation verifiers, "
    "ARC oracle-distinct verifier work, and the FoVer production ensemble are outside "
    "this retired scope."
)
SPEC_REFS = [
    "REQ-REPORT-5170",
    "SCENARIO-REPORT-5170-LINT",
    "SCENARIO-REPORT-5170-NARROW-SCOPE",
]

FIELD_PRINCIPLES: dict[str, str] = {
    "phase_d_artifacts_enumerated": (
        "A retirement based on 4 examples out of ~20+ artifacts is under-evidenced; "
        "this field proves the consolidation is genuinely comprehensive."
    ),
    "exclusion_manifest_entry_added": "manifest retirement must be present and schema-compatible",
    "entry_id": "traceability to the retired_extras entry",
    "false_positive_check_against_exp5178": (
        "Must confirm the new entry does not accidentally block this milestone's own hidden-state "
        "verifier pilot -- an over-broad retirement would be a self-inflicted wound, per the "
        "literature's own distinction between external-text and internal-representation scoring."
    ),
    "synthetic_match_check_passed": (
        "Confirms the retirement is load-bearing, not just documentation, per the 2026-07-01 "
        "BLOCKED_PATTERN_MATCHED mechanical-fix precedent and the exp5165 self-test pattern."
    ),
    "sanctioned_exception_documented": (
        "The retirement note must explicitly preserve the hidden-state verifier research direction "
        "as open, not accidentally chill it via ambiguous prose even where the mechanical "
        "blocked_patterns are correctly scoped."
    ),
    "inference_substrate": "aggregation_from_upstream_artifacts",
    "honest_verdict": "Must start with complete:/complete_/success:/success_.",
}

REQUIRED_SCHEMA_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "milestone",
    "spec_refs",
    "result_path",
    "run_date",
    "field_principles",
    "inference_substrate",
    "duration_s",
    "random_seed",
    "source_artifacts_read",
    "source_artifact_summary",
    "lineage_stage_summary",
    "manifest_entry_audit",
    "current_roadmap_lint",
    "synthetic_match_lint",
    "sanctioned_exception_doc",
    "adversarial_verification",
    "flagged_adversarial",
    "tests_run",
    "reproducibility_checksum",
    *FIELD_PRINCIPLES,
)

DEFAULT_TESTS_RUN = [
    ".venv/bin/pytest tests/python/test_experiment_5170_retire_phase_d_external_text_scorer_v474.py -q -o addopts=''",
    ".venv/bin/coverage run --rcfile=/dev/null --include='*/experiment_5170_retire_phase_d_external_text_scorer_v474.py' -m pytest tests/python/test_experiment_5170_retire_phase_d_external_text_scorer_v474.py -q --no-cov -o addopts=''",
    ".venv/bin/coverage report --rcfile=/dev/null -m --include='*/experiment_5170_retire_phase_d_external_text_scorer_v474.py' --fail-under=100",
    "python scripts/check_spec_coverage.py tests/python/test_experiment_5170_retire_phase_d_external_text_scorer_v474.py",
    ".venv/bin/python scripts/exclusion_manifest_lint.py research-roadmap.yaml",
    ".venv/bin/pytest tests/python -q",
]


def expected_manifest_entry() -> JsonDict:
    return {
        "id": ENTRY_ID,
        "experiment_scope": EXPERIMENT_SCOPE,
        "reason": RETIREMENT_REASON,
        "experiment_ids": list(EXPECTED_EXPERIMENT_IDS),
        "retired_milestone": MILESTONE,
        "retired_by_artifact": str(SOURCE_RESULT_PATHS["exp5163"]),
        "recorded_by_artifact": str(RESULT_RELATIVE_PATH),
        "operator_reopen_required": True,
        "retire_if_same_verdict": True,
        "blocked_patterns": list(EXPECTED_BLOCKED_PATTERNS),
    }


def _load_yaml_mapping(path: Path) -> JsonDict:
    if not path.exists():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return dict(loaded) if isinstance(loaded, Mapping) else {}


def _load_linter_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_exp5170_exclusion_manifest_lint", LINTER_MODULE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {LINTER_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _risk_to_dict(risk: Any) -> JsonDict:
    return {
        "task_id": str(getattr(risk, "task_id", "")),
        "task_title": str(getattr(risk, "task_title", "")),
        "violation_class": str(getattr(risk, "violation_class", "")),
        "severity": str(getattr(risk, "severity", "")),
        "detail": str(getattr(risk, "detail", "")),
        "retirement_reason": str(getattr(risk, "retirement_reason", "")),
    }


def lint_roadmap(root: Path, roadmap_path: Path) -> list[JsonDict]:
    module = _load_linter_module()
    module.PROJECT_ROOT = root
    return [_risk_to_dict(risk) for risk in module.lint(roadmap_path)]


def _find_manifest_entry(root: Path) -> JsonDict:
    manifest = _load_yaml_mapping(root / MANIFEST_RELATIVE_PATH)
    for entry in _list(manifest.get("retired_extras")):
        entry_map = _mapping(entry)
        if entry_map.get("id") == ENTRY_ID:
            return dict(entry_map)
    return {}


def _entry_audit(entry: JsonMap) -> JsonDict:
    expected = expected_manifest_entry()
    errors: list[str] = []
    for key in (
        "id",
        "experiment_scope",
        "reason",
        "experiment_ids",
        "retired_milestone",
        "retired_by_artifact",
        "recorded_by_artifact",
        "operator_reopen_required",
        "retire_if_same_verdict",
        "blocked_patterns",
    ):
        if entry.get(key) != expected[key]:
            errors.append(f"{key}.mismatch")
    return {
        "entry_id": str(entry.get("id", "")),
        "found": bool(entry),
        "errors": errors,
        "blocked_patterns": _list(entry.get("blocked_patterns")),
    }


def _entry_blocked_pattern_risks(risks: Sequence[JsonMap]) -> list[JsonDict]:
    return [
        dict(risk)
        for risk in risks
        if risk.get("violation_class") == "BLOCKED_PATTERN_MATCHED"
        and ENTRY_ID in str(risk.get("detail", ""))
    ]


def check_current_roadmap_false_positive(root: Path) -> JsonDict:
    roadmap_path = root / ROADMAP_RELATIVE_PATH
    risks = lint_roadmap(root, roadmap_path)
    entry_risks = _entry_blocked_pattern_risks(risks)
    exp5178_ids = [
        str(task.get("id", ""))
        for task in _list(_load_yaml_mapping(roadmap_path).get("tasks"))
        if str(task.get("id", "")).startswith("exp5178")
    ]
    exp5178_risks = [
        risk for risk in entry_risks if str(risk.get("task_id", "")).startswith("exp5178")
    ]
    return {
        "roadmap_path": str(ROADMAP_RELATIVE_PATH),
        "all_risks": risks,
        "entry_blocked_pattern_risks": entry_risks,
        "entry_blocked_pattern_task_ids": [str(risk.get("task_id", "")) for risk in entry_risks],
        "exp5178_task_ids": exp5178_ids,
        "exp5178_entry_risks": exp5178_risks,
        "passed": len(exp5178_ids) == 1 and not exp5178_risks,
    }


def _synthetic_roadmap_text() -> str:
    return yaml.safe_dump(
        {
            "milestone": MILESTONE,
            "tasks": [
                {
                    "id": "exp9999-lora-ebm-external-text-rerun",
                    "title": "train lora ebm scorer v2 on off-arc reasoning corpus",
                    "prompt": (
                        "Rerun the retired external generated-text scorer against a tuned "
                        "self-consistency baseline without operator override."
                    ),
                    "agent_type": "codex",
                }
            ],
        },
        sort_keys=False,
    )


def check_synthetic_match(root: Path) -> JsonDict:
    with tempfile.TemporaryDirectory(prefix="exp5170_lint_") as tmp:
        roadmap_path = Path(tmp) / "research-roadmap-next.yaml"
        roadmap_path.write_text(_synthetic_roadmap_text(), encoding="utf-8")
        risks = lint_roadmap(root, roadmap_path)
    matched = _entry_blocked_pattern_risks(risks)
    return {
        "synthetic_title": "train lora ebm scorer v2 on off-arc reasoning corpus",
        "matched_risks": matched,
        "all_risks": risks,
        "passed": any(risk.get("severity") == "HARD" for risk in matched),
    }


def _source_row(root: Path, *, source_id: str, path: Path, kind: str) -> JsonDict:
    full_path = root / path
    return {
        "source_id": source_id,
        "kind": kind,
        "path": str(path),
        "exists": full_path.exists(),
        "sha256": file_sha256(full_path),
    }


def build_source_artifacts_read(root: Path) -> list[JsonDict]:
    rows = [
        _source_row(root, source_id=source_id, path=path, kind="prior_result")
        for source_id, path in SOURCE_RESULT_PATHS.items()
    ]
    rows.extend(
        [
            _source_row(
                root, source_id="exclusion_manifest", path=MANIFEST_RELATIVE_PATH, kind="ops_yaml"
            ),
            _source_row(
                root,
                source_id="verifier_gaps",
                path=VERIFIER_GAPS_RELATIVE_PATH,
                kind="ops_doc",
            ),
            _source_row(
                root, source_id="known_issues", path=KNOWN_ISSUES_RELATIVE_PATH, kind="ops_doc"
            ),
            _source_row(
                root,
                source_id="research_references",
                path=RESEARCH_REFERENCES_RELATIVE_PATH,
                kind="research_doc",
            ),
            _source_row(
                root, source_id="active_roadmap", path=ROADMAP_RELATIVE_PATH, kind="roadmap_yaml"
            ),
            _source_row(
                root, source_id="exclusion_manifest_lint", path=LINTER_RELATIVE_PATH, kind="script"
            ),
        ]
    )
    return rows


def _value(value: Any) -> Any:
    if isinstance(value, Mapping) and "value" in value:
        return value.get("value")
    return value


def _number_value(value: Any) -> float | None:
    value = _value(value)
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _source_artifact_summary(root: Path) -> JsonDict:
    summary: JsonDict = {}
    for source_id, path in SOURCE_RESULT_PATHS.items():
        payload, status = read_json_mapping(root / path)
        payload_map = _mapping(payload)
        summary[source_id] = {
            "path": str(path),
            "loadable": _bool(status.get("loadable")),
            "honest_verdict": str(_value(payload_map.get("honest_verdict", ""))),
            "flagged_adversarial": payload_map.get("flagged_adversarial"),
        }
        for key in (
            "delta_vs_tuned_sc",
            "distributional_energy_delta",
            "energy_minus_sc_delta",
            "energy_minus_sc_ci95",
            "paired_ci95",
            "delta_ci95",
            "verifier_vs_cheap_delta",
            "verifier_vs_cheap_delta_ci95",
            "self_consistency_accuracy",
            "moat_realized",
            "moat_retired_bounded",
            "bounded_retirement_ok",
        ):
            if key in payload_map:
                summary[source_id][key] = _value(payload_map.get(key))
    return summary


def _loadable_source_ids(summary: JsonMap) -> list[str]:
    return [
        source_id
        for source_id in SOURCE_RESULT_PATHS
        if _bool(_mapping(summary.get(source_id)).get("loadable"))
    ]


def _lineage_stage_summary(summary: JsonMap) -> JsonDict:
    cleanest_source = "exp5031"
    return {
        "stage_count": 7,
        "scope_retired": EXPERIMENT_SCOPE,
        "overall_verdict": (
            "No Phase D external generated-text/logprob scorer produced a decision-grade "
            "win over genuine tuned self-consistency; the best point estimate touches zero."
        ),
        "cleanest_point_estimate": {
            "source_id": cleanest_source,
            "delta_vs_tuned_sc": _mapping(summary.get(cleanest_source)).get("delta_vs_tuned_sc"),
            "ci95": _mapping(summary.get(cleanest_source)).get("paired_ci95"),
            "interpretation": "positive point estimate, but CI95 lower bound touches zero",
        },
        "decisive_live_musr_negative": {
            "source_id": "distributional_energy_verifier_musr",
            "energy_minus_sc_delta": _mapping(
                summary.get("distributional_energy_verifier_musr")
            ).get("energy_minus_sc_delta"),
            "ci95": _mapping(summary.get("distributional_energy_verifier_musr")).get(
                "energy_minus_sc_ci95"
            ),
        },
        "terminal_continuation": {
            "source_id": "exp5163",
            "delta": _number_value(_mapping(summary.get("exp5163")).get("verifier_vs_cheap_delta")),
            "ci95": _mapping(summary.get("exp5163")).get("verifier_vs_cheap_delta_ci95"),
            "flagged_adversarial": _mapping(summary.get("exp5163")).get("flagged_adversarial"),
        },
        "sanctioned_exceptions": [
            "hidden-state/internal-representation verifiers",
            "ARC-domain oracle-distinct verifier work",
            "FoVer production ensemble",
        ],
    }


def check_sanctioned_exception_doc(root: Path) -> JsonDict:
    paths = [VERIFIER_GAPS_RELATIVE_PATH, KNOWN_ISSUES_RELATIVE_PATH]
    hits: list[str] = []
    required_terms = (
        "external generated-text/logprob scorer",
        "hidden-state/internal-representation verifier",
        "TrajSelector",
        "VerifySteer",
        "remain open",
    )
    for path in paths:
        full_path = root / path
        text = full_path.read_text(encoding="utf-8") if full_path.exists() else ""
        if SANCTIONED_EXCEPTION_DOC_MARKER in text and all(term in text for term in required_terms):
            hits.append(str(path))
    return {
        "paths_checked": [str(path) for path in paths],
        "marker": SANCTIONED_EXCEPTION_DOC_MARKER,
        "matching_paths": hits,
        "updated": bool(hits),
    }


def _honest_verdict(
    *,
    enumerated_ok: bool,
    entry_ok: bool,
    current_ok: bool,
    synthetic_ok: bool,
    doc_ok: bool,
) -> str:
    if enumerated_ok and entry_ok and current_ok and synthetic_ok and doc_ok:
        return COMPLETE_VERDICT
    return INCOMPLETE_VERDICT


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    duration_s: float,
    run_date: str,
    verification: JsonMap,
    tests_run: Sequence[Any] = DEFAULT_TESTS_RUN,
) -> JsonDict:
    entry = _find_manifest_entry(root)
    entry_audit = _entry_audit(entry)
    current_lint = check_current_roadmap_false_positive(root)
    synthetic_lint = check_synthetic_match(root)
    doc_check = check_sanctioned_exception_doc(root)
    source_summary = _source_artifact_summary(root)
    enumerated = _loadable_source_ids(source_summary)
    entry_ok = not _list(entry_audit.get("errors")) and bool(entry_audit.get("found"))
    enumerated_ok = len(enumerated) >= 7 and "exp5163" in enumerated
    current_ok = _bool(current_lint.get("passed"))
    synthetic_ok = _bool(synthetic_lint.get("passed"))
    doc_ok = _bool(doc_check.get("updated"))
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "spec_refs": list(SPEC_REFS),
        "result_path": str(RESULT_RELATIVE_PATH),
        "run_date": run_date,
        "field_principles": dict(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(max(duration_s, 0.0001), 6),
        "random_seed": RANDOM_SEED,
        "source_artifacts_read": build_source_artifacts_read(root),
        "source_artifact_summary": source_summary,
        "lineage_stage_summary": _lineage_stage_summary(source_summary),
        "manifest_entry_audit": entry_audit,
        "current_roadmap_lint": current_lint,
        "synthetic_match_lint": synthetic_lint,
        "sanctioned_exception_doc": doc_check,
        "adversarial_verification": dict(verification),
        "flagged_adversarial": _bool(verification.get("flagged_adversarial")),
        "tests_run": list(tests_run),
        "phase_d_artifacts_enumerated": enumerated,
        "exclusion_manifest_entry_added": entry_ok,
        "entry_id": ENTRY_ID,
        "false_positive_check_against_exp5178": current_ok,
        "synthetic_match_check_passed": synthetic_ok,
        "sanctioned_exception_documented": doc_ok,
        "honest_verdict": _honest_verdict(
            enumerated_ok=enumerated_ok,
            entry_ok=entry_ok,
            current_ok=current_ok,
            synthetic_ok=synthetic_ok,
            doc_ok=doc_ok,
        ),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def artifact_schema_errors(artifact: JsonMap) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_SCHEMA_FIELDS:
        if field not in artifact:
            errors.append(f"missing.{field}")
    for field, principle in FIELD_PRINCIPLES.items():
        if _mapping(artifact.get("field_principles")).get(field) != principle:
            errors.append(f"field_principle.{field}")
    checks = [
        (artifact.get("experiment_id") != EXPERIMENT_ID, "experiment_id.invalid"),
        (artifact.get("milestone") != MILESTONE, "milestone.invalid"),
        (artifact.get("entry_id") != ENTRY_ID, "entry_id.invalid"),
        (artifact.get("inference_substrate") != INFERENCE_SUBSTRATE, "inference_substrate.invalid"),
        (
            not str(artifact.get("honest_verdict", "")).startswith(TERMINAL_PREFIXES),
            "honest_verdict.not_terminal",
        ),
        (
            len(_list(artifact.get("phase_d_artifacts_enumerated"))) < 7
            or "exp5163" not in _list(artifact.get("phase_d_artifacts_enumerated")),
            "phase_d_artifacts_enumerated.insufficient",
        ),
        (
            artifact.get("exclusion_manifest_entry_added") is not True,
            "exclusion_manifest_entry_added.invalid",
        ),
        (
            artifact.get("false_positive_check_against_exp5178") is not True,
            "false_positive_check_against_exp5178.invalid",
        ),
        (
            artifact.get("synthetic_match_check_passed") is not True,
            "synthetic_match_check_passed.invalid",
        ),
        (
            artifact.get("sanctioned_exception_documented") is not True,
            "sanctioned_exception_documented.invalid",
        ),
        (
            not isinstance(artifact.get("duration_s"), (int, float))
            or artifact.get("duration_s", 0) <= 0,
            "duration_s.invalid",
        ),
        (not _list(artifact.get("source_artifacts_read")), "source_artifacts_read.empty"),
        (not _list(artifact.get("tests_run")), "tests_run.empty"),
        (
            artifact.get("reproducibility_checksum") != payload_checksum(artifact),
            "reproducibility_checksum.invalid",
        ),
    ]
    errors.extend(code for failed, code in checks if failed)
    return errors


def validate_artifact(artifact: JsonMap) -> None:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError(f"invalid Exp 5170 retirement artifact: {errors}")


def run(
    *,
    root: Path = REPO_ROOT,
    output: Path | None = None,
    run_date: str | None = None,
    clock: Any = time.perf_counter,
    verification_runner: Any = run_adversarial_verification,
    tests_run: Sequence[Any] = DEFAULT_TESTS_RUN,
) -> Path:
    root = Path(root)
    output_path = output or root / RESULT_RELATIVE_PATH
    start = clock()
    placeholder = verification_payload(
        CommandResult(command=(), exit_code=0, stdout='{"flags":[]}', stderr="")
    )
    artifact = build_artifact(
        root=root,
        duration_s=clock() - start,
        run_date=run_date or time.strftime("%Y%m%d"),
        verification=placeholder,
        tests_run=tests_run,
    )
    write_json(output_path, artifact)
    verification = verification_payload(verification_runner(root, output_path))
    artifact = {
        **artifact,
        "adversarial_verification": verification,
        "flagged_adversarial": _bool(verification.get("flagged_adversarial")),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    write_json(output_path, artifact)
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--date", default=None)
    args = parser.parse_args(argv)
    run(root=args.root, output=args.output, run_date=args.date)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
