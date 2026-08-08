"""Tests for Exp6210 V537 adversarial capstone.

Spec refs: REQ-INFRA-6210, SCENARIO-INFRA-6210-1,
SCENARIO-INFRA-6210-2, SCENARIO-INFRA-6210-3,
SCENARIO-INFRA-6210-4, SCENARIO-INFRA-6210-5.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_6210_v537_adversarial_capstone as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-harnesses/spec.md"


def _write_text(root: Path, rel_path: Path | str, text: str = "fixture\n") -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(root: Path, rel_path: Path | str, payload: Any) -> None:
    _write_text(root, rel_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _prompt(*fields: str) -> str:
    return "TASK\nRequired deliverable: fixture\nREQUIRED ARTIFACT FIELDS: " + ", ".join(fields)


def _roadmap_payload() -> JsonDict:
    tasks: list[JsonDict] = [
        {
            "id": "exp6197-v537-terminal-artifact-contract",
            "milestone": mod.MILESTONE,
            "title": "Fail-closed terminal-artifact contract after V536 bootstrap-only completions",
            "track": "infrastructure",
            "deliverable": "results/experiment_6197_v537_terminal_artifact_contract.json",
            "prompt": _prompt("status", "honest_verdict", "duration_s"),
        },
        {
            "id": "exp6198-v537-post-marker-source-scope-audit",
            "milestone": mod.MILESTONE,
            "title": "Post-V537-marker SOTA delta plus staged-roadmap scope audit",
            "track": "infrastructure",
            "deliverable": "results/experiment_6198_v537_post_marker_source_scope_audit.json",
            "requires": ["exp6197-v537-terminal-artifact-contract"],
            "prompt": _prompt("status", "honest_verdict", "duration_s", "accepted_count"),
        },
        {
            "id": "exp6199-gatemate-terminal-action-audit-v537",
            "milestone": mod.MILESTONE,
            "title": "GateMate cached terminal-action audit with changed-state authorization only",
            "track": "hardware",
            "deliverable": "results/experiment_6199_gatemate_terminal_action_audit_v537.json",
            "requires": ["exp6197-v537-terminal-artifact-contract"],
            "prompt": _prompt(
                "status",
                "honest_verdict",
                "duration_s",
                "speed_power_energy_terminal_tsu_kona_claim_counts",
            ),
        },
        {
            "id": "exp6200-three-family-raw-code-transport-canary",
            "milestone": mod.MILESTONE,
            "title": "Three-family raw-code transport canary",
            "track": "phase-d",
            "deliverable": "results/experiment_6200_three_family_raw_code_transport_canary.json",
            "requires": ["exp6197-v537-terminal-artifact-contract"],
            "prompt": _prompt(
                "status",
                "honest_verdict",
                "duration_s",
                "phase_d_transport_ready_score",
                "csl_transport_ready_score",
            ),
        },
        {
            "id": "exp6201-livecodebench-k8-pool-v2",
            "milestone": mod.MILESTONE,
            "title": "Authentic Gemma-4-31B K=8 pool",
            "track": "phase-d",
            "deliverable": "results/experiment_6201_livecodebench_k8_pool_v2.json",
            "requires": ["exp6200-three-family-raw-code-transport-canary"],
            "gated_on": [
                {
                    "upstream": "exp6200-three-family-raw-code-transport-canary",
                    "artifact_field": "phase_d_transport_ready_score",
                    "op": "==",
                    "value": 1,
                }
            ],
            "prompt": _prompt(
                "status",
                "honest_verdict",
                "duration_s",
                "pool_integrity_ready_score",
                "raw_shard_paths_hashes_and_seal_receipt",
            ),
        },
        {
            "id": "exp6202-livecodebench-headroom-v2",
            "milestone": mod.MILESTONE,
            "title": "Code competence and selectable-headroom audit",
            "track": "phase-d",
            "deliverable": "results/experiment_6202_livecodebench_headroom_v2.json",
            "requires": ["exp6201-livecodebench-k8-pool-v2"],
            "gated_on": [
                {
                    "upstream": "exp6201-livecodebench-k8-pool-v2",
                    "artifact_field": "pool_integrity_ready_score",
                    "op": "==",
                    "value": 1,
                }
            ],
            "prompt": _prompt("status", "honest_verdict", "duration_s", "headroom_ready_score"),
        },
        {
            "id": "exp6203-matching-base-code-hidden-state-v2",
            "milestone": mod.MILESTONE,
            "title": "Matching-base hidden-state surface",
            "track": "phase-d",
            "deliverable": "results/experiment_6203_matching_base_code_hidden_state_v2.json",
            "requires": ["exp6202-livecodebench-headroom-v2"],
            "gated_on": [
                {
                    "upstream": "exp6202-livecodebench-headroom-v2",
                    "artifact_field": "headroom_ready_score",
                    "op": "==",
                    "value": 1,
                }
            ],
            "prompt": _prompt("status", "honest_verdict", "duration_s", "surface_ready_score"),
        },
        {
            "id": "exp6204-calibration-code-selector-v2",
            "milestone": mod.MILESTONE,
            "title": "Calibration-only selector freeze",
            "track": "phase-d",
            "deliverable": "results/experiment_6204_calibration_code_selector_v2.json",
            "requires": ["exp6203-matching-base-code-hidden-state-v2"],
            "gated_on": [
                {
                    "upstream": "exp6203-matching-base-code-hidden-state-v2",
                    "artifact_field": "surface_ready_score",
                    "op": "==",
                    "value": 1,
                }
            ],
            "prompt": _prompt("status", "honest_verdict", "duration_s", "selector_ready_score"),
        },
        {
            "id": "exp6205-held-code-selection-v2",
            "milestone": mod.MILESTONE,
            "title": "One-shot held code selection",
            "track": "phase-d",
            "deliverable": "results/experiment_6205_held_code_selection_v2.json",
            "requires": ["exp6204-calibration-code-selector-v2"],
            "gated_on": [
                {
                    "upstream": "exp6204-calibration-code-selector-v2",
                    "artifact_field": "selector_ready_score",
                    "op": "==",
                    "value": 1,
                }
            ],
            "prompt": _prompt("status", "honest_verdict", "duration_s"),
        },
        {
            "id": "exp6206-live-strategy-seed-v2",
            "milestone": mod.MILESTONE,
            "title": "Transport-qualified two-family strategy seed",
            "track": "continuous-self-learning",
            "deliverable": "results/experiment_6206_live_strategy_seed_v2.json",
            "requires": ["exp6200-three-family-raw-code-transport-canary"],
            "gated_on": [
                {
                    "upstream": "exp6200-three-family-raw-code-transport-canary",
                    "artifact_field": "csl_transport_ready_score",
                    "op": "==",
                    "value": 1,
                }
            ],
            "prompt": _prompt("status", "honest_verdict", "duration_s", "seed_stream_ready_score"),
        },
        {
            "id": "exp6207-prospective-procedural-memory-csl",
            "milestone": mod.MILESTONE,
            "title": "Prospective procedural-memory continuous-learning A/B",
            "track": "continuous-self-learning",
            "deliverable": "results/experiment_6207_prospective_procedural_memory_csl.json",
            "requires": ["exp6206-live-strategy-seed-v2"],
            "gated_on": [
                {
                    "upstream": "exp6206-live-strategy-seed-v2",
                    "artifact_field": "seed_stream_ready_score",
                    "op": "==",
                    "value": 1,
                }
            ],
            "prompt": _prompt("status", "honest_verdict", "duration_s"),
        },
        {
            "id": "exp6208-mode-jump-runtime-integration",
            "milestone": mod.MILESTONE,
            "title": "Mode-jump runtime integration",
            "track": "sampler",
            "deliverable": "results/experiment_6208_mode_jump_runtime_integration.json",
            "requires": ["exp6197-v537-terminal-artifact-contract"],
            "prompt": _prompt("status", "honest_verdict", "duration_s"),
        },
        {
            "id": "exp6209-arc-loo-task-aware-shadow",
            "milestone": mod.MILESTONE,
            "title": "Single ARC slot leave-one-game-out shadow generalization",
            "track": "arc",
            "deliverable": "results/experiment_6209_arc_loo_task_aware_shadow.json",
            "requires": ["exp6197-v537-terminal-artifact-contract"],
            "prompt": _prompt("status", "honest_verdict", "duration_s"),
        },
    ]
    tasks.append(
        {
            "id": mod.EXPERIMENT_ID,
            "milestone": mod.MILESTONE,
            "title": "V537 exact-path adversarial capstone",
            "track": "infrastructure",
            "deliverable": mod.RESULT_RELATIVE_PATH.as_posix(),
            "requires": [task["id"] for task in tasks],
            "prompt": _prompt(*mod.REQUIRED_ARTIFACT_FIELDS),
            "prior_failures": [
                {
                    "experiment_id": "exp6196-v536-capstone",
                    "verdict": "blocked: bootstrap only; capstone reconciliation checks not complete",
                    "retire_if_same_verdict": True,
                }
            ],
        }
    )
    return {"milestone": mod.MILESTONE, "tasks": tasks}


def _artifact(task_id: str) -> JsonDict:
    base: JsonDict = {
        "schema": f"fixture.{task_id}",
        "status": "complete",
        "honest_verdict": "complete: fixture terminal",
        "duration_s": 1.0,
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "reproducibility_checksum": "sha256:fixture",
    }
    payloads: dict[str, JsonDict] = {
        "exp6197-v537-terminal-artifact-contract": {
            **base,
            "status": "complete_ready",
            "honest_verdict": "complete: classifier ready",
        },
        "exp6198-v537-post-marker-source-scope-audit": {
            **base,
            "status": "complete",
            "honest_verdict": "complete_null: accepted_count=0",
            "accepted_count": 0,
        },
        "exp6199-gatemate-terminal-action-audit-v537": {
            **base,
            "status": "blocked_missing_receipt",
            "honest_verdict": "blocked_missing_receipt: unchanged hardware state",
            "speed_power_energy_terminal_tsu_kona_claim_counts": {
                "energy": 0,
                "kona": 0,
                "power": 0,
                "speed": 0,
                "terminal": 0,
                "terminal_hardware": 0,
                "tsu": 0,
            },
        },
        "exp6200-three-family-raw-code-transport-canary": {
            **base,
            "status": "complete_ready",
            "honest_verdict": "complete_ready: phase_d ready csl blocked",
            "phase_d_transport_ready_score": 1,
            "csl_transport_ready_score": 0,
        },
        "exp6201-livecodebench-k8-pool-v2": {
            **base,
            "status": "complete_ready",
            "honest_verdict": "complete_ready: pool ready but missing a required receipt",
            "pool_integrity_ready_score": 1,
        },
        "exp6203-matching-base-code-hidden-state-v2": {
            **base,
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "gates_evaluated": [
                {
                    "upstream": "exp6202-livecodebench-headroom-v2",
                    "artifact_field": "headroom_ready_score",
                    "expected": 1,
                    "actual": None,
                    "passed": False,
                }
            ],
        },
        "exp6205-held-code-selection-v2": {
            **base,
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "gates_evaluated": [
                {
                    "upstream": "exp6204-calibration-code-selector-v2",
                    "artifact_field": "selector_ready_score",
                    "expected": 1,
                    "actual": None,
                    "passed": False,
                }
            ],
        },
        "exp6206-live-strategy-seed-v2": {
            **base,
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "gates_evaluated": [
                {
                    "upstream": "exp6200-three-family-raw-code-transport-canary",
                    "artifact_field": "csl_transport_ready_score",
                    "expected": 1,
                    "actual": 0,
                    "passed": False,
                }
            ],
        },
        "exp6208-mode-jump-runtime-integration": {
            **base,
            "status": "complete_ready",
            "honest_verdict": "complete_ready: runtime sampler integrated",
            "test_exit_codes": {"cargo fmt --all -- --check": 1},
            "nonzero_command_classification": [
                {
                    "command": "cargo fmt --all -- --check",
                    "exit_code": 1,
                    "classification": "unrelated_preexisting",
                }
            ],
        },
        "exp6209-arc-loo-task-aware-shadow": {
            **base,
            "status": "complete_positive",
            "honest_verdict": "complete_positive: shadow generalization no solve",
            "solve_claimed": False,
            "arc_solve_registry_delta": [],
        },
    }
    return payloads[task_id]


def _conductor_log() -> str:
    lines = []
    for task in _roadmap_payload()["tasks"]:
        if task["id"] == mod.EXPERIMENT_ID:
            continue
        status = "GATE_BLOCK" if "6206" in task["id"] else "OK"
        lines.append(f"| 2026-08-07 16:00 UTC | {task['title'][:52]} | {status} | fixture |")
    return "\n".join(lines)


def _make_root(root: Path) -> None:
    roadmap = _roadmap_payload()
    _write_text(root, mod.ROADMAP_RELATIVE_PATH, yaml.safe_dump(roadmap))
    _write_text(root, mod.ROADMAP_DOC_RELATIVE_PATH, "last reconciled 2026-07-03\n")
    _write_text(root, mod.CONDUCTOR_LOG_RELATIVE_PATH, _conductor_log())
    _write_text(root, mod.CODEX_RELATIVE_PATH, "CODEX fixture\n")
    _write_text(root, mod.CLAUDE_RELATIVE_PATH, "CLAUDE fixture\n")
    _write_text(root, mod.EXCLUSION_MANIFEST_RELATIVE_PATH, "retired_experiments: []\n")
    _write_text(root, mod.RESEARCH_COMPLETE_RELATIVE_PATH, "milestones: []\n")
    for rel_path in (
        mod.STATUS_RELATIVE_PATH,
        mod.CHANGELOG_RELATIVE_PATH,
        mod.TRACEABILITY_RELATIVE_PATH,
        mod.ARCHITECTURE_RELATIVE_PATH,
    ):
        _write_text(root, rel_path, f"{rel_path.as_posix()} fixture\n")

    missing = {
        "exp6202-livecodebench-headroom-v2",
        "exp6204-calibration-code-selector-v2",
        "exp6207-prospective-procedural-memory-csl",
    }
    for task in roadmap["tasks"]:
        task_id = str(task["id"])
        if task_id in missing or task_id == mod.EXPERIMENT_ID:
            continue
        _write_json(root, task["deliverable"], _artifact(task_id))
    _write_json(
        root,
        "results/experiment_6202_sidecar.json",
        {"status": "complete", "honest_verdict": "complete: ignored sidecar"},
    )


def _receipt(task_id: str, path: str, *, critical: bool = False) -> JsonDict:
    flags = [{"kind": "DURATION_TOO_SHORT", "severity": "critical"}] if critical else []
    report = {
        "artifact": path,
        "loaded": True,
        "flag_count": len(flags),
        "flags": flags,
        "max_severity": 5 if flags else -1,
    }
    return {
        "task_id": task_id,
        "artifact_path": path,
        "adversarial": {
            "command": f".venv/bin/python scripts/adversarial_verify.py --json {path}",
            "exit_code": 1 if critical else 0,
            "stdout_json": {"reports": [report], "flagged_count": len(flags)},
            "stderr": "",
        },
        "summary": {
            "command": f".venv/bin/python scripts/summarize_artifact.py {path}",
            "exit_code": 2 if critical else 0,
            "stdout_tail": "summary",
            "stderr_tail": "",
        },
    }


def _receipts() -> dict[str, JsonDict]:
    receipts: dict[str, JsonDict] = {}
    for task in _roadmap_payload()["tasks"]:
        task_id = str(task["id"])
        if task_id in {
            mod.EXPERIMENT_ID,
            "exp6202-livecodebench-headroom-v2",
            "exp6204-calibration-code-selector-v2",
            "exp6207-prospective-procedural-memory-csl",
        }:
            continue
        receipts[task_id] = _receipt(
            task_id,
            str(task["deliverable"]),
            critical=task_id == "exp6201-livecodebench-k8-pool-v2",
        )
    return receipts


def _build(root: Path) -> JsonDict:
    _make_root(root)
    return mod.build_report(
        root,
        date="20260807",
        verifier_receipts=_receipts(),
        tests_run={
            ".venv/bin/pytest tests/python/test_experiment_6210_v537_adversarial_capstone.py -q --no-cov -n 0": 0
        },
        duration_s=3.0,
    )


def test_req_infra_6210_spec_declares_exact_capstone_contract() -> None:
    """REQ-INFRA-6210: OpenSpec names the V537 capstone contract."""

    text = SPEC_PATH.read_text(encoding="utf-8")
    section = text[text.index("REQ-INFRA-6210") :]
    for marker in (
        "REQ-INFRA-6210",
        "SCENARIO-INFRA-6210-1",
        "SCENARIO-INFRA-6210-2",
        "SCENARIO-INFRA-6210-3",
        "SCENARIO-INFRA-6210-4",
        "SCENARIO-INFRA-6210-5",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        "Exp6197 through Exp6209",
        "Exp6197 terminal-artifact classifier",
        "aggregation_from_upstream_artifacts",
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_infra_6210_exact_manifest_and_classifier_preserve_missing(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFRA-6210-1/2: exact paths and classifier results outrank receipts."""

    report = _build(tmp_path)

    manifest = report["exact_deliverable_manifest"]
    exp6202 = manifest["exp6202-livecodebench-headroom-v2"]
    assert exp6202["present"] is False
    assert exp6202["same_number_alias_used"] is False
    assert exp6202["same_number_alias_candidates_ignored"] == [
        "results/experiment_6202_sidecar.json"
    ]

    classifier = report["terminal_classifier_path_hash_and_results"]["results"]
    assert classifier["exp6202-livecodebench-headroom-v2"]["classification"] == "missing"
    assert classifier["exp6202-livecodebench-headroom-v2"]["conductor_receipt_status"] == "OK"
    assert classifier["exp6202-livecodebench-headroom-v2"]["receipt_overrode"] is False

    classes = report["task_terminal_classes"]
    assert classes["exp6201-livecodebench-k8-pool-v2"]["classifier_class"] == "ready"
    assert classes["exp6201-livecodebench-k8-pool-v2"]["terminal_class"] == "flagged"
    assert classes["exp6201-livecodebench-k8-pool-v2"]["missing_required_fields"] == [
        "raw_shard_paths_hashes_and_seal_receipt"
    ]
    assert classes["exp6206-live-strategy-seed-v2"]["terminal_class"] == "skipped"

    counts = report["missing_nonterminal_blocked_skipped_null_retired_flagged_counts"]
    assert counts["missing"] == 3
    assert counts["flagged"] == 1
    assert counts["blocked"] == 1
    assert counts["skipped"] == 3
    assert counts["null"] == 1


def test_scenario_infra_6210_gate_recompute_and_headline_eligibility(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFRA-6210-3/4: gates and verifier flags drive headline outcomes."""

    report = _build(tmp_path)

    gates = report["structured_gate_recomputation"]
    assert gates["exp6201-livecodebench-k8-pool-v2"]["declared_gates"][0]["passed"] is True
    seed_gate = gates["exp6206-live-strategy-seed-v2"]["declared_gates"][0]
    assert seed_gate["actual"] == 0
    assert seed_gate["passed"] is False
    assert gates["exp6206-live-strategy-seed-v2"]["conductor_gate_block"] is True

    receipts = report["adversarial_verify_receipts_by_artifact"]
    assert receipts["exp6201-livecodebench-k8-pool-v2"]["critical_flag_count"] == 1
    assert receipts["exp6208-mode-jump-runtime-integration"]["summary"]["exit_code"] == 0

    phase = report["phase_d_headline_eligibility_and_reason"]
    assert phase["eligible"] is False
    assert "exp6201-livecodebench-k8-pool-v2:critical_adversarial_flag" in phase["blocking_reasons"]
    assert "exp6202-livecodebench-headroom-v2:missing" in phase["blocking_reasons"]

    csl = report["continuous_self_learning_headline_eligibility_and_reason"]
    assert csl["eligible"] is False
    assert "exp6206-live-strategy-seed-v2:failed_gate" in csl["blocking_reasons"]
    assert "exp6207-prospective-procedural-memory-csl:missing" in csl["blocking_reasons"]

    assert report["sampler_integration_headline_eligibility_and_reason"]["eligible"] is True
    assert report["arc_generalization_headline_eligibility_and_reason"]["eligible"] is True
    assert report["hardware_continuity_state"]["terminal_class"] == "blocked"
    assert report["hardware_continuity_state"]["forbidden_claim_counts"] == {
        "energy": 0,
        "kona": 0,
        "power": 0,
        "speed": 0,
        "terminal": 0,
        "terminal_hardware": 0,
        "tsu": 0,
    }
    assert report["source_delta_state"]["terminal_class"] == "null"
    assert report["source_delta_state"]["accepted_count"] == 0


def test_scenario_infra_6210_write_validate_checksum_and_zero_counts(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6210-5: the artifact is stable, atomic, and non-mutating."""

    _make_root(tmp_path)
    report = mod.write_capstone(
        root=tmp_path,
        date="20260807",
        verifier_receipts=_receipts(),
        tests_run={
            ".venv/bin/pytest tests/python/test_experiment_6210_v537_adversarial_capstone.py -q --no-cov -n 0": 0
        },
        duration_s=3.0,
        env={},
    )

    assert (tmp_path / mod.RESULT_RELATIVE_PATH).exists()
    assert mod.validate_report(report) == []
    assert report["status"] == "complete"
    assert report["honest_verdict"].startswith("complete:")
    assert report["reproducibility_checksum"] == mod.payload_checksum(report)
    assert report["protected_historical_artifact_mutation_count"] == 0
    assert all(value == 0 for value in report["forbidden_claim_counts"].values())
    assert (
        report["spec_traceability_status_changelog_reconciliation_receipts"][
            "ops_status_changelog_traceability_modified"
        ]
        is False
    )
    assert report["architecture_freshness_warning"]["stale"] is True
    assert report["verifier_is_oracle"] is False
    assert report["inference_substrate"] == mod.INFERENCE_SUBSTRATE


def test_req_infra_6210_defensive_helpers_and_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-INFRA-6210: helpers fail closed and validation rejects laundering."""

    report = _build(tmp_path)

    assert mod._read_json_mapping(tmp_path / "missing.json")[1]["error"] == "missing"
    _write_text(tmp_path, "bad.json", "{")
    assert mod._read_json_mapping(tmp_path / "bad.json")[1]["error"].startswith("json_error:")
    _write_text(tmp_path, "array.json", "[]")
    assert mod._read_json_mapping(tmp_path / "array.json")[1]["error"] == "json_not_mapping"
    assert mod._read_yaml_mapping(tmp_path / "missing.yaml") == {}
    _write_text(tmp_path, "not_mapping.yaml", "[]\n")
    assert mod._read_yaml_mapping(tmp_path / "not_mapping.yaml") == {}
    assert (
        mod._same_number_aliases(tmp_path / "no-results", "not-an-exp", Path("results/x.json"))
        == []
    )
    assert mod._latest_conductor_receipt("", "not present") == {
        "present": False,
        "status": None,
        "line": None,
        "detail": None,
    }
    assert mod._gate_passed(1, "==", 1) is True
    assert mod._gate_passed(1, "!=", 1) is False
    assert mod._required_fields_from_prompt("no field list") == []
    assert mod._required_fields_from_prompt(
        "REQUIRED ARTIFACT FIELDS: `status`, honest_verdict."
    ) == [
        "status",
        "honest_verdict",
    ]
    assert mod._required_fields_from_prompt(
        "REQUIRED ARTIFACT FIELDS: status, and honest_verdict."
    ) == [
        "status",
        "honest_verdict",
    ]
    assert mod._missing_required_fields({"status": "complete"}, ["status", "x"]) == ["x"]
    assert mod._unclassified_nonzero_commands({"test_exit_codes": {"cmd": 1}}) == ["cmd"]
    assert (
        mod._unclassified_nonzero_commands(
            {
                "test_exit_codes": {"cmd": 1},
                "nonzero_command_classification": [{"command": "cmd"}],
            }
        )
        == []
    )
    assert (
        mod._unclassified_nonzero_commands(
            {
                "test_exit_codes": {"mapped": 1},
                "nonzero_command_classification": {"mapped": "preexisting"},
            }
        )
        == []
    )
    assert mod._unclassified_nonzero_commands(
        {
            "test_exit_codes": {"bad-int": "nope"},
            "task_owned_test_commands_and_exit_codes": {
                "command_receipts": [
                    {"command": "nested", "exit_code": 3, "classification": "preexisting"},
                    {"command": "nested-unclassified", "exit_code": 4},
                ]
            },
            "full_suite_command_and_classified_exit_code": {
                "command": "full",
                "exit_code": 2,
                "classification": "classified",
            },
        }
    ) == ["bad-int", "nested-unclassified"]
    assert mod._same_number_aliases(tmp_path, "exp6202-any", Path("results/nope.json")) == [
        "results/experiment_6202_sidecar.json"
    ]
    fallback_root = tmp_path / "fallback"
    _write_text(
        fallback_root,
        mod.ROADMAP_RELATIVE_PATH,
        yaml.safe_dump(
            {
                "tasks": [
                    {
                        "id": "exp6197-fallback",
                        "deliverable": "results/experiment_6197_fallback.json",
                    },
                    {
                        "id": "exp6211-out-of-range",
                        "deliverable": "results/experiment_6211_out.json",
                    },
                    {
                        "id": mod.EXPERIMENT_ID,
                        "requires": ["exp6197-fallback", "exp6211-out-of-range"],
                        "deliverable": mod.RESULT_RELATIVE_PATH.as_posix(),
                    },
                ]
            }
        ),
    )
    fallback_declared, _fallback_roadmap, fallback_capstone = mod._roadmap_declared_tasks(
        fallback_root
    )
    assert [row["task_id"] for row in fallback_declared] == ["exp6197-fallback"]
    assert fallback_capstone["id"] == mod.EXPERIMENT_ID
    no_capstone_root = tmp_path / "no-capstone"
    _write_text(
        no_capstone_root,
        mod.ROADMAP_RELATIVE_PATH,
        yaml.safe_dump(
            {
                "tasks": [
                    {
                        "id": "exp6198-fallback",
                        "deliverable": "results/experiment_6198_fallback.json",
                    }
                ]
            }
        ),
    )
    no_capstone_declared, _roadmap, no_capstone_row = mod._roadmap_declared_tasks(no_capstone_root)
    assert [row["task_id"] for row in no_capstone_declared] == ["exp6198-fallback"]
    assert no_capstone_row == {}
    assert mod._run_commands(
        tmp_path, ["alpha beta"], lambda argv, _root: {"command": " ".join(argv), "exit_code": 0}
    ) == [{"command": "alpha beta", "exit_code": 0}]
    assert mod._normalize_tests(None)[1]
    assert mod._normalize_tests([{"command": "cmd", "exit_code": 2}]) == (["cmd"], {"cmd": 2})
    top_level = mod._normalize_verifier_receipts(
        [
            {
                "task_id": "expTop",
                "command": "adv",
                "exit_code": 0,
                "stdout_json": {"flagged_count": 4},
            }
        ]
    )
    assert top_level["expTop"]["flag_count"] == 4
    assert mod._normalize_verifier_receipts([{"task_id": "expX"}])["expX"]["task_id"] == "expX"
    assert mod._flag_count({"adversarial": []}) == 0
    assert mod._critical_flag_count({"adversarial": {"stdout_json": {}}}) == 0
    assert (
        mod._critical_flag_count({"adversarial": {"stdout_json": {"reports": [{"flags": "bad"}]}}})
        == 0
    )
    assert mod._artifact_forbidden_claim_counts({}) == {
        "energy": 0,
        "kona": 0,
        "power": 0,
        "speed": 0,
        "terminal": 0,
        "terminal_hardware": 0,
        "tsu": 0,
    }
    assert (
        mod._artifact_forbidden_claim_counts(
            {"speed_power_energy_terminal_tsu_kona_claim_counts": {"speed": 0}}
        )["speed"]
        == 0
    )
    assert (
        mod._artifact_forbidden_claim_counts(
            {"speed_power_energy_terminal_tsu_kona_claim_counts": {"speed": "bad"}}
        )["speed"]
        == 1
    )
    assert "unclassified_nonzero_command" in mod._task_issues(
        "task",
        {
            "terminal_class": "complete",
            "critical_adversarial_flag_count": 0,
            "missing_required_fields": [],
            "unclassified_nonzero_commands": ["cmd"],
        },
        {"declared_gates": []},
    )
    assert (
        mod._prior_failure_actions(
            [
                {
                    "task_id": "task",
                    "prior_failures": [
                        {"experiment_id": "bad-shape"},
                        {
                            "experiment_id": "prior",
                            "verdict": "complete: same",
                            "retire_if_same_verdict": True,
                        },
                    ],
                }
            ],
            {},
            {"task": {"honest_verdict": "complete: same"}},
        )[0]["same_verdict"]
        is True
    )

    monkeypatch.setattr(mod, "_run_artifact_verifiers", lambda _root, _present: _receipts())
    live_report = mod.build_report(
        tmp_path,
        date="20260807",
        verifier_receipts=None,
        tests_run={".venv/bin/pytest tests/python -q": 0},
        duration_s=3.0,
    )
    assert (
        live_report["adversarial_verify_receipts_by_artifact"]["exp6201-livecodebench-k8-pool-v2"][
            "critical_flag_count"
        ]
        == 1
    )
    command_report = mod.build_report(
        tmp_path,
        date="20260807",
        verifier_receipts=_receipts(),
        tests_run=None,
        command_runner=lambda argv, _root: {"command": " ".join(argv), "exit_code": 0},
        duration_s=3.0,
    )
    assert set(command_report["test_commands"]) == set(mod.DEFAULT_TEST_COMMANDS)
    assert all(code == 0 for code in command_report["test_exit_codes"].values())

    broken = deepcopy(report)
    broken.pop("status")
    assert "missing:status" in mod.validate_report(broken)

    bad_status = deepcopy(report)
    bad_status["status"] = "complete_partial"
    bad_status["reproducibility_checksum"] = mod.payload_checksum(bad_status)
    assert "status" in mod.validate_report(bad_status)

    bad_checksum = deepcopy(report)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum" in mod.validate_report(bad_checksum)

    bad_zero = deepcopy(report)
    bad_zero["protected_historical_artifact_mutation_count"] = 1
    bad_zero["reproducibility_checksum"] = mod.payload_checksum(bad_zero)
    assert "protected_historical_artifact_mutation_count" in mod.validate_report(bad_zero)

    bad_forbidden = deepcopy(report)
    bad_forbidden["forbidden_claim_counts"]["speed"] = 1
    bad_forbidden["reproducibility_checksum"] = mod.payload_checksum(bad_forbidden)
    assert "forbidden_claim_counts" in mod.validate_report(bad_forbidden)

    bad_docs = deepcopy(report)
    bad_docs["spec_traceability_status_changelog_reconciliation_receipts"][
        "ops_status_changelog_traceability_modified"
    ] = True
    bad_docs["reproducibility_checksum"] = mod.payload_checksum(bad_docs)
    assert "spec_traceability_status_changelog_reconciliation_receipts" in mod.validate_report(
        bad_docs
    )

    bad_oracle = deepcopy(report)
    bad_oracle["verifier_is_oracle"] = True
    bad_oracle["reproducibility_checksum"] = mod.payload_checksum(bad_oracle)
    assert "verifier_is_oracle" in mod.validate_report(bad_oracle)

    bad_substrate = deepcopy(report)
    bad_substrate["inference_substrate"] = "live_llm_inference"
    bad_substrate["reproducibility_checksum"] = mod.payload_checksum(bad_substrate)
    assert "inference_substrate" in mod.validate_report(bad_substrate)

    bad_verdict = deepcopy(report)
    bad_verdict["honest_verdict"] = "done"
    bad_verdict["reproducibility_checksum"] = mod.payload_checksum(bad_verdict)
    assert "honest_verdict_prefix" in mod.validate_report(bad_verdict)

    bad_provenance = deepcopy(report)
    bad_provenance["field_provenance"]["status"]["principle"] = "wrong"
    bad_provenance["reproducibility_checksum"] = mod.payload_checksum(bad_provenance)
    assert "field_provenance:status" in mod.validate_report(bad_provenance)

    no_provenance = deepcopy(report)
    no_provenance["field_provenance"] = []
    no_provenance["reproducibility_checksum"] = mod.payload_checksum(no_provenance)
    assert "field_provenance:not_mapping" in mod.validate_report(no_provenance)

    no_principles = deepcopy(report)
    no_principles["field_principles"] = []
    no_principles["reproducibility_checksum"] = mod.payload_checksum(no_principles)
    assert "field_principles:not_mapping" in mod.validate_report(no_principles)

    missing_principle = deepcopy(report)
    missing_principle["field_principles"]["status"] = ""
    missing_principle["reproducibility_checksum"] = mod.payload_checksum(missing_principle)
    assert "field_principles:status" in mod.validate_report(missing_principle)

    monkeypatch.setattr(mod, "validate_report", lambda _report: ["forced"])
    with pytest.raises(ValueError, match="invalid Exp6210 capstone"):
        mod.write_capstone(
            root=tmp_path,
            date="20260807",
            verifier_receipts=_receipts(),
            tests_run={".venv/bin/pytest tests/python -q": 0},
        )
