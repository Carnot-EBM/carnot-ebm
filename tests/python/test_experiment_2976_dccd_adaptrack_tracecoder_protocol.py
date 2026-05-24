"""Tests for Exp 2976 intent-preserving DCCD repair protocol.

Spec refs: REQ-VERIFY-2976, SCENARIO-VERIFY-2976.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from carnot.experiment_2976_dccd_adaptrack_tracecoder_protocol import (
    ARTIFACT_FILENAME,
    MANDATORY_HEADLINE_MODEL_IDS,
    REQUIRED_ARTIFACT_FIELDS,
    build_artifact,
    run_experiment,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "verification" / "spec.md"
SOURCE_FILES = (
    "experiment_2963_dccd_repair_protocol_manifest_v1.json",
    "experiment_2964_sota_dccd_repair_replication_v1.json",
    "experiment_2952_sota_taxonomy_guided_code_repair_eval_v1.json",
    "experiment_2953_code_verifier_threshold_policy_v1.json",
)
MANIFEST_REQUIRED_FIELDS = (
    "draft_intent",
    "constrained_patch",
    "backtracking_steps",
    "execution_trace",
    "verifier_result",
    "schema_result",
    "syntax_result",
    "false_accept_audit",
    "acceptance_reason",
)


def _clock() -> object:
    value = 200.0

    def monotonic() -> float:
        nonlocal value
        value += 0.25
        return value

    return monotonic


def _copy_sources(dst: Path, *, research_sweep: bool = True) -> None:
    results_dir = dst / "results"
    results_dir.mkdir(parents=True)
    for filename in SOURCE_FILES:
        shutil.copy2(REPO_ROOT / "results" / filename, results_dir / filename)
    text = (
        "2026-05-24 Post-.279 Planning Sweep\n"
        "AdapTrack\nTraceCoder\nThinking Before Constraining\nbacktracking\n"
        if research_sweep
        else "research references without the required sweep\n"
    )
    (dst / "research-references.md").write_text(text, encoding="utf-8")


def test_exp2976_spec_entry_present() -> None:
    """REQ-VERIFY-2976: the verification spec anchors the protocol artifact."""
    spec = SPEC_PATH.read_text(encoding="utf-8")
    assert "REQ-VERIFY-2976" in spec
    assert "SCENARIO-VERIFY-2976" in spec
    assert ARTIFACT_FILENAME in spec


def test_exp2976_build_artifact_aggregates_failure_guards() -> None:
    """SCENARIO-VERIFY-2976: upstream schema collapse becomes deterministic gates."""
    artifact = build_artifact(repo_root=REPO_ROOT, monotonic=_clock())

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    assert missing == []
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["intent_preserving_repair_protocol_ready"] is True
    assert artifact["trace_execution_plan_ready"] is True
    assert artifact["prior_failure_addressed"] is True
    assert artifact["downstream_min_tasks"] == 20
    assert artifact["duration_s"] == 0.25
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"

    assert artifact["mandatory_headline_model_ids"] == list(MANDATORY_HEADLINE_MODEL_IDS)
    assert artifact["legacy_model_policy"] == (
        "Legacy small models are allowed only for CPU smoke tests and must never "
        "be reported as headline repair results."
    )
    assert artifact["required_model_specs"][0]["selection_rule"] == "call_cached_sota_pair_first"
    assert artifact["required_model_specs"][0]["minimum_headline_models"] == 1

    condition_ids = [condition["condition_id"] for condition in artifact["baseline_conditions"]]
    assert condition_ids == [
        "baseline_no_taxonomy",
        "schema_only_dccd",
        "intent_preserving_dccd",
        "trace_aware_repair",
    ]

    manifest_schema = artifact["repair_manifest_schema"]
    assert manifest_schema["schema_version"] == "carnot.intent_preserving_repair_manifest.v1"
    assert manifest_schema["required"] == list(MANIFEST_REQUIRED_FIELDS)
    assert all(field in manifest_schema["properties"] for field in MANIFEST_REQUIRED_FIELDS)
    assert manifest_schema["properties"]["execution_trace"]["min_items"] == 1
    assert manifest_schema["properties"]["backtracking_steps"]["items"]["required"] == [
        "step_index",
        "constraint_trigger",
        "rollback_target",
        "intent_preservation_check",
    ]

    comparison = artifact["upstream_failure_comparison"]
    assert comparison["exp2964_dccd_structured"]["schema_failure_rate"] == 1.0
    assert comparison["exp2964_dccd_structured"]["syntax_failure_rate"] == 1.0
    assert comparison["exp2964_dccd_structured"]["pass_at_1"] == 0.0
    assert comparison["exp2964_dccd_structured"]["false_accept_rate"] == 0.0
    assert comparison["schema_failure_rate_delta"] == 0.95
    assert comparison["syntax_failure_rate_delta"] == 0.30000000000000004
    assert comparison["pass_at_1_delta"] == -0.2

    correlations = artifact["dccd_failure_correlations"]
    assert correlations["schema_failures"]["correlated_dccd_fields"] == [
        "mode=dccd_structured",
        "schema_valid=false",
        "parser_status=not_run",
        "schema_errors_present",
    ]
    assert correlations["false_accepts"]["interpretation"] == (
        "zero false accepts in schema-only DCCD is not progress because pass@1 and "
        "pass@k are both zero"
    )

    gates = artifact["acceptance_gates"]
    assert gates["minimum_tasks"]["op"] == ">="
    assert gates["minimum_tasks"]["value"] == 20
    assert gates["pass_at_1_delta"]["op"] == ">"
    assert gates["schema_failure_rate_delta"]["op"] == "<="
    assert gates["runtime_trace_coverage"]["value"] == 0.8
    assert gates["condition_coverage"]["required_conditions"] == condition_ids

    assert artifact["schema_regression_guard"]["blocked_prior_pattern"]["observed_rate"] == 1.0
    assert artifact["syntax_regression_guard"]["blocked_prior_pattern"]["observed_rate"] == 1.0
    assert artifact["false_accept_guard"]["max_false_accept_rate"] == 0.010135135135
    assert artifact["false_accept_guard"]["verifier_only_accepts_count_as_pass"] is False


def test_exp2976_blocks_when_research_sweep_missing(tmp_path: Path) -> None:
    """REQ-VERIFY-2976: the post-.279 research sweep is a real precondition."""
    _copy_sources(tmp_path, research_sweep=False)

    artifact = build_artifact(repo_root=tmp_path, monotonic=_clock())

    assert artifact["honest_verdict"] == "blocked_missing_post_279_research_sweep"
    assert artifact["intent_preserving_repair_protocol_ready"] is False
    assert artifact["trace_execution_plan_ready"] is False
    assert artifact["prior_failure_addressed"] is False
    assert artifact["repair_manifest_schema"]["required"] == []
    assert artifact["preconditions_checked"][-1]["resource"] == "post_279_research_sweep"
    assert artifact["preconditions_checked"][-1]["available"] is False


def test_exp2976_blocks_when_upstream_artifact_missing(tmp_path: Path) -> None:
    """REQ-VERIFY-2976: upstream .279 repair artifacts must exist first."""
    (tmp_path / "results").mkdir()
    (tmp_path / "research-references.md").write_text(
        "2026-05-24 Post-.279 Planning Sweep\n"
        "AdapTrack\nTraceCoder\nThinking Before Constraining\nbacktracking\n",
        encoding="utf-8",
    )

    artifact = build_artifact(repo_root=tmp_path, monotonic=_clock())

    assert artifact["honest_verdict"] == "blocked_missing_upstream_repair_artifacts"
    assert artifact["intent_preserving_repair_protocol_ready"] is False
    assert artifact["trace_execution_plan_ready"] is False
    assert artifact["preconditions_checked"][0]["resource"] == "exp2963"
    assert artifact["preconditions_checked"][0]["available"] is False


def test_exp2976_run_experiment_writes_terminal_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-2976: the runner writes the stable terminal JSON."""
    _copy_sources(tmp_path)
    destination = tmp_path / "results" / ARTIFACT_FILENAME

    artifact = run_experiment(repo_root=tmp_path, artifact_path=destination, monotonic=_clock())
    written = json.loads(destination.read_text(encoding="utf-8"))

    assert written == artifact
    assert written["intent_preserving_repair_protocol_ready"] is True
    assert written["trace_execution_plan_ready"] is True
