"""Tests for Exp6224 V538 adversarial capstone.

Spec refs: REQ-CAPSTONE-6224, SCENARIO-CAPSTONE-6224,
SCENARIO-CAPSTONE-6224-EXACT-PATH,
SCENARIO-CAPSTONE-6224-BRANCH-INDEPENDENCE,
SCENARIO-CAPSTONE-6224-GATEMATE,
SCENARIO-CAPSTONE-6224-ARC-REGISTRY,
SCENARIO-CAPSTONE-6224-FIELD-PRINCIPLES.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_6224_v538_adversarial_capstone as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/capstone/spec.md"


UPSTREAM_TASKS: tuple[tuple[str, str, str], ...] = (
    (
        "exp6211-v538-post-marker-source-scope-prereg",
        "infrastructure",
        "results/experiment_6211_v538_post_marker_source_scope_prereg.json",
    ),
    (
        "exp6212-three-family-gguf-runtime-recovery",
        "infrastructure",
        "results/experiment_6212_three_family_gguf_runtime_recovery.json",
    ),
    (
        "exp6213-arc-object-delta-perception-wiring",
        "arc",
        "results/experiment_6213_arc_object_delta_perception_wiring.json",
    ),
    (
        "exp6214-arc-object-delta-heldout-ab",
        "arc",
        "results/experiment_6214_arc_object_delta_heldout_ab.json",
    ),
    (
        "exp6215-arc-trajectory-transfer-ab",
        "arc",
        "results/experiment_6215_arc_object_relative_trajectory_transfer_ab.json",
    ),
    (
        "exp6216-arc-budget-aware-search-ab",
        "arc",
        "results/experiment_6216_arc_budget_aware_search_ab.json",
    ),
    (
        "exp6217-arc-gemma31-think-ab",
        "arc",
        "results/experiment_6217_arc_gemma31_think_ab.json",
    ),
    (
        "exp6218-arc-admissible-lever-portfolio-heldout",
        "arc",
        "results/experiment_6218_arc_admissible_lever_portfolio_heldout.json",
    ),
    (
        "exp6219-two-timescale-constraint-csl",
        "continuous-learning",
        "results/experiment_6219_two_timescale_constraint_csl.json",
    ),
    (
        "exp6220-mode-jump-runtime-ab",
        "sampler",
        "results/experiment_6220_mode_jump_runtime_ab.json",
    ),
    (
        "exp6221-three-family-code-transport-canary-v3",
        "phase-d",
        "results/experiment_6221_three_family_code_transport_canary_v3.json",
    ),
    (
        "exp6222-livecodebench-k8-pool-v3",
        "phase-d",
        "results/experiment_6222_livecodebench_k8_pool_v3.json",
    ),
    (
        "exp6223-livecodebench-headroom-v3",
        "phase-d",
        "results/experiment_6223_livecodebench_headroom_v3.json",
    ),
)


def _write_text(root: Path, rel_path: Path | str, text: str = "fixture\n") -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(root: Path, rel_path: Path | str, payload: Any) -> None:
    _write_text(root, rel_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _prompt(*fields: str) -> str:
    return "TASK\nRequired deliverable: fixture\nREQUIRED ARTIFACT FIELDS: " + ", ".join(fields)


def _roadmap_payload() -> JsonDict:
    tasks: list[JsonDict] = []
    for task_id, track, deliverable in UPSTREAM_TASKS:
        gated_on: list[JsonDict] = []
        if "6217" in task_id:
            gated_on.append(
                {
                    "upstream": "exp6212-three-family-gguf-runtime-recovery",
                    "artifact_field": "gemma_4_31b_runtime_ready_score",
                    "op": "==",
                    "value": 1,
                }
            )
        if "6221" in task_id:
            gated_on.append(
                {
                    "upstream": "exp6212-three-family-gguf-runtime-recovery",
                    "artifact_field": "three_family_runtime_ready_score",
                    "op": "==",
                    "value": 1,
                }
            )
        if "6222" in task_id:
            gated_on.append(
                {
                    "upstream": "exp6221-three-family-code-transport-canary-v3",
                    "artifact_field": "phase_d_transport_ready_score",
                    "op": "==",
                    "value": 1,
                }
            )
        if "6223" in task_id:
            gated_on.append(
                {
                    "upstream": "exp6222-livecodebench-k8-pool-v3",
                    "artifact_field": "pool_integrity_ready_score",
                    "op": "==",
                    "value": 1,
                }
            )
        tasks.append(
            {
                "id": task_id,
                "milestone": mod.MILESTONE,
                "title": task_id.replace("-", " ")[:80],
                "track": track,
                "deliverable": deliverable,
                "gated_on": gated_on,
                "prompt": _prompt("status", "honest_verdict", "duration_s"),
            }
        )
    tasks.append(
        {
            "id": mod.EXPERIMENT_ID,
            "milestone": mod.MILESTONE,
            "title": "V538 exact-path adversarial capstone",
            "track": "capstone",
            "deliverable": mod.RESULT_RELATIVE_PATH.as_posix(),
            "requires": [task[0] for task in UPSTREAM_TASKS],
            "prompt": _prompt(*mod.REQUIRED_ARTIFACT_FIELDS),
            "prior_failures": [
                {
                    "experiment_id": "exp6210-v537-adversarial-capstone",
                    "verdict": "complete: prior",
                    "retire_if_same_verdict": True,
                }
            ],
        }
    )
    return {"milestone": mod.MILESTONE, "tasks": tasks}


def _artifact(task_id: str) -> JsonDict:
    base: JsonDict = {
        "status": "complete",
        "honest_verdict": "complete: fixture terminal",
        "duration_s": 1.0,
        "reproducibility_checksum": "sha256:fixture",
    }
    payloads: dict[str, JsonDict] = {
        "exp6211-v538-post-marker-source-scope-prereg": {
            **base,
            "status": "complete_null",
            "honest_verdict": "complete_null: no new sources",
        },
        "exp6212-three-family-gguf-runtime-recovery": {
            **base,
            "status": "complete_partial",
            "honest_verdict": "complete_partial: runtime evidence still blocked",
            "gemma_4_31b_runtime_ready_score": 0,
            "three_family_runtime_ready_score": 0,
        },
        "exp6213-arc-object-delta-perception-wiring": {
            **base,
            "status": "complete_ready",
            "honest_verdict": "complete_ready: object wiring ready no solve",
            "object_delta_wiring_ready_score": 1,
            "solve_claimed": False,
        },
        "exp6214-arc-object-delta-heldout-ab": {
            **base,
            "status": "complete_positive",
            "honest_verdict": "complete_positive: object delta improved no solve",
            "ab_complete_score": 1,
            "object_delta_promotion_ready_score": 1,
            "solve_claimed": False,
            "level_credit_delta": 0,
        },
        "exp6215-arc-trajectory-transfer-ab": {
            **base,
            "status": "complete_positive",
            "honest_verdict": "complete_positive: transfer reduced calls no solve",
            "ab_complete_score": 1,
            "solve_claimed": False,
            "level_credit_delta": 0,
        },
        "exp6216-arc-budget-aware-search-ab": {
            **base,
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "gate_check_summary": "fixture gate failed",
            "solve_claimed": False,
            "level_credit_delta": 0,
        },
        "exp6217-arc-gemma31-think-ab": {
            **base,
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "gate_check_summary": "dense runtime was not ready",
        },
        "exp6218-arc-admissible-lever-portfolio-heldout": {
            **base,
            "status": "complete_null",
            "honest_verdict": "complete_null: fewer than two levers admissible",
            "solve_claimed": False,
            "level_credit_delta": 0,
        },
        "exp6219-two-timescale-constraint-csl": {
            **base,
            "status": "complete_positive",
            "honest_verdict": "complete_positive: csl branch independent",
            "continuous_self_learning_ready_score": 1,
        },
        "exp6220-mode-jump-runtime-ab": {
            **base,
            "status": "blocked",
            "honest_verdict": "blocked: sampler quality threshold unmet",
            "sampler_runtime_ready_score": 0,
            "hardware_claim_count": 0,
        },
        "exp6221-three-family-code-transport-canary-v3": {
            **base,
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "gate_check_summary": "runtime gate failed",
            "phase_d_transport_ready_score": 0,
        },
        "exp6223-livecodebench-headroom-v3": {
            **base,
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "gate_check_summary": "pool missing",
            "headroom_ready_score": 0,
        },
    }
    return payloads[task_id]


def _gatemate_receipt() -> JsonDict:
    return {
        "status": "blocked_missing_receipt",
        "honest_verdict": "blocked_missing_receipt: no newer dated GateMate receipt",
        "duration_s": 0.01,
        "current_dated_operator_receipt": {
            "exists": False,
            "receipt_date": None,
        },
        "detect_attempt_count_command_stdout_stderr_exit_code": {"attempt_count": 0},
        "hardware_command_authorized": False,
        "physical_state_changed": False,
        "speed_power_energy_terminal_tsu_kona_claim_counts": {
            "energy": 0,
            "kona": 0,
            "power": 0,
            "speed": 0,
            "terminal": 0,
            "terminal_hardware": 0,
            "tsu": 0,
        },
    }


def _make_root(root: Path) -> None:
    _write_text(root, mod.ROADMAP_RELATIVE_PATH, yaml.safe_dump(_roadmap_payload()))
    _write_text(root, mod.ROADMAP_DOC_RELATIVE_PATH, "roadmap fixture\n")
    _write_text(root, mod.CONDUCTOR_LOG_RELATIVE_PATH, _conductor_log())
    _write_text(root, mod.CODEX_RELATIVE_PATH, "CODEX fixture\n")
    _write_text(root, mod.CLAUDE_RELATIVE_PATH, "CLAUDE fixture\n")
    _write_text(root, mod.EXCLUSION_MANIFEST_RELATIVE_PATH, "retired_experiments: []\n")
    _write_text(root, mod.RESEARCH_COMPLETE_RELATIVE_PATH, "milestones: []\n")
    _write_text(root, mod.ARC_REGISTRY_RELATIVE_PATH, "games: []\n")
    _write_text(root, mod.SPEC_RELATIVE_PATH, "REQ-CAPSTONE-6224\n")
    for rel_path in (
        mod.STATUS_RELATIVE_PATH,
        mod.CHANGELOG_RELATIVE_PATH,
        mod.TRACEABILITY_RELATIVE_PATH,
        mod.CONDUCTOR_RELATIVE_PATH,
    ):
        _write_text(root, rel_path, f"{rel_path.as_posix()} fixture\n")
    _write_json(root, mod.GATEMATE_RECEIPT_RELATIVE_PATH, _gatemate_receipt())
    for task_id, _track, deliverable in UPSTREAM_TASKS:
        if task_id == "exp6222-livecodebench-k8-pool-v3":
            continue
        _write_json(root, deliverable, _artifact(task_id))
    _write_json(
        root,
        "results/experiment_6222_sidecar.json",
        {"status": "complete", "honest_verdict": "complete: ignored sidecar"},
    )


def _conductor_log() -> str:
    lines = []
    for task_id, _track, _deliverable in UPSTREAM_TASKS:
        status = "GATE_BLOCK" if task_id.startswith(("exp6217", "exp622")) else "OK"
        lines.append(
            f"| 2026-08-09 12:00 UTC | {task_id.replace('-', ' ')[:52]} | {status} | fixture |"
        )
    return "\n".join(lines)


def _receipt(task_id: str, path: str, *, critical: bool = False) -> JsonDict:
    flags = [{"kind": "DURATION_TOO_SHORT", "severity": "critical"}] if critical else []
    return {
        "task_id": task_id,
        "artifact_path": path,
        "adversarial": {
            "command": f".venv/bin/python scripts/adversarial_verify.py --json {path}",
            "exit_code": 1 if critical else 0,
            "stdout_json": {
                "reports": [
                    {
                        "artifact": path,
                        "loaded": True,
                        "flag_count": len(flags),
                        "flags": flags,
                    }
                ],
                "flagged_count": len(flags),
            },
            "stderr": "",
        },
        "summary": {
            "command": f".venv/bin/python scripts/summarize_artifact.py {path}",
            "exit_code": 0,
            "stdout_tail": "summary",
            "stderr_tail": "",
        },
    }


def _receipts() -> dict[str, JsonDict]:
    receipts: dict[str, JsonDict] = {}
    for task_id, _track, deliverable in UPSTREAM_TASKS:
        if task_id == "exp6222-livecodebench-k8-pool-v3":
            continue
        receipts[task_id] = _receipt(
            task_id,
            deliverable,
            critical=task_id == "exp6214-arc-object-delta-heldout-ab",
        )
    return receipts


def _tests_run() -> dict[str, int]:
    return {
        ".venv/bin/pytest tests/python/test_experiment_6224_v538_adversarial_capstone.py -q --no-cov -n 0": 0,
        ".venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_6224_v538_adversarial_capstone.py --fail-under=100": 0,
        ".venv/bin/pytest tests/python -q": 0,
    }


def _command_receipts() -> list[JsonDict]:
    return [
        {
            "command": ".venv/bin/python scripts/publication_gate.py --json",
            "exit_code": 0,
            "classification": "passed",
            "stdout_json": {"paper_ready": False, "unmet_gates": ["G2"]},
        },
        {
            "command": ".venv/bin/python scripts/audit_roadmap_gates.py research-roadmap.yaml",
            "exit_code": 0,
            "classification": "passed",
        },
    ]


def _build(root: Path) -> JsonDict:
    _make_root(root)
    return mod.build_report(
        root,
        date="20260809",
        verifier_receipts=_receipts(),
        command_receipts=_command_receipts(),
        tests_run=_tests_run(),
        duration_s=4.0,
    )


def test_req_capstone_6224_spec_declares_contract() -> None:
    """REQ-CAPSTONE-6224: OpenSpec anchors the V538 exact-path capstone."""

    text = SPEC_PATH.read_text(encoding="utf-8")
    section = text[text.index("REQ-CAPSTONE-6224") :]
    normalized = " ".join(section.split())
    for marker in (
        "REQ-CAPSTONE-6224",
        "SCENARIO-CAPSTONE-6224",
        "SCENARIO-CAPSTONE-6224-EXACT-PATH",
        "SCENARIO-CAPSTONE-6224-BRANCH-INDEPENDENCE",
        "SCENARIO-CAPSTONE-6224-GATEMATE",
        "SCENARIO-CAPSTONE-6224-ARC-REGISTRY",
        "SCENARIO-CAPSTONE-6224-FIELD-PRINCIPLES",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_capstone_6224_exact_paths_and_counts(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-6224-EXACT-PATH: exact paths outrank aliases."""

    report = _build(tmp_path)

    declared = report["declared_task_ids_and_deliverables"]
    assert len(declared["tasks"]) == 13
    assert declared["task_ids"] == [task_id for task_id, _track, _path in UPSTREAM_TASKS]

    rows = report["exact_artifact_paths_hashes_and_terminal_classifications"]
    exp6222 = rows["exp6222-livecodebench-k8-pool-v3"]
    assert exp6222["present"] is False
    assert exp6222["classification"] == "missing"
    assert exp6222["same_number_alias_candidates_ignored"] == [
        "results/experiment_6222_sidecar.json"
    ]
    assert exp6222["receipt_overrode"] is False

    counts = report["missing_nonterminal_blocked_skipped_null_retired_and_flagged_counts"]
    assert counts["missing"] == 1
    assert counts["nonterminal"] == 2
    assert counts["blocked"] == 1
    assert counts["skipped"] == 4
    assert counts["null"] == 2
    assert counts["retired"] == 0
    assert counts["flagged"] == 1
    assert report["conductor_receipt_override_count"] == 0


def test_scenario_capstone_6224_branch_independence_and_zero_arc_credit(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-6224-BRANCH-INDEPENDENCE: branches stay separate."""

    report = _build(tmp_path)

    arc = report["arc_lever_and_portfolio_eligibility"]
    assert arc["eligible"] is False
    assert (
        "exp6214-arc-object-delta-heldout-ab:critical_adversarial_flag" in arc["blocking_reasons"]
    )
    assert "exp6216-arc-budget-aware-search-ab:skipped" in arc["blocking_reasons"]

    assert report["gguf_runtime_eligibility"]["eligible"] is False
    assert (
        "exp6212-three-family-gguf-runtime-recovery:three_family_runtime_ready_score=0"
        in report["gguf_runtime_eligibility"]["blocking_reasons"]
    )
    assert report["continuous_self_learning_eligibility"]["eligible"] is True
    assert report["sampler_runtime_eligibility"]["eligible"] is False
    assert report["phase_d_transport_pool_and_headroom_eligibility"]["eligible"] is False
    assert (
        "exp6222-livecodebench-k8-pool-v3:missing"
        in report["phase_d_transport_pool_and_headroom_eligibility"]["blocking_reasons"]
    )

    assert report["arc_solve_claim_count"] == 0
    assert report["arc_level_credit_delta"] == 0
    assert report["arc_registry_hash_before_after"]["unchanged"] is True
    assert (
        report["arc_registry_hash_before_after"]["before_sha256"]
        == report["arc_registry_hash_before_after"]["after_sha256"]
    )


def test_scenario_capstone_6224_gate_gatemate_publication_and_write(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-6224-GATEMATE: hardware stays blocked and unwritten."""

    _make_root(tmp_path)
    report = mod.write_capstone(
        root=tmp_path,
        date="20260809",
        verifier_receipts=_receipts(),
        command_receipts=_command_receipts(),
        tests_run=_tests_run(),
        duration_s=4.0,
        env={},
    )

    assert (tmp_path / mod.RESULT_RELATIVE_PATH).exists()
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == report
    assert mod.validate_report(report) == []
    assert report["status"] == "complete"
    assert report["honest_verdict"].startswith("complete:")
    assert report["gatemate_cached_state_and_new_receipt_count"]["new_receipt_count"] == 0
    assert report["gatemate_cached_state_and_new_receipt_count"]["board_command_count"] == 0
    assert report["hardware_claim_eligibility"]["eligible"] is False
    assert all(
        value == 0
        for value in report["hardware_claim_eligibility"][
            "unauthorized_hardware_claim_counts"
        ].values()
    )
    assert report["publication_gate_snapshot"]["paper_ready"] is False
    assert report["protected_files_unchanged"]["unchanged"] is True
    assert report["reproducibility_checksum"] == mod.payload_checksum(report)


def test_req_capstone_6224_validation_rejects_laundering(tmp_path: Path) -> None:
    """REQ-CAPSTONE-6224: validator rejects nonzero promotion counts."""

    report = _build(tmp_path)

    missing = deepcopy(report)
    missing.pop("status")
    assert "missing:status" in mod.validate_report(missing)

    bad_checksum = deepcopy(report)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum" in mod.validate_report(bad_checksum)

    bad_override = deepcopy(report)
    bad_override["conductor_receipt_override_count"] = 1
    bad_override["reproducibility_checksum"] = mod.payload_checksum(bad_override)
    assert "conductor_receipt_override_count" in mod.validate_report(bad_override)

    bad_solve = deepcopy(report)
    bad_solve["arc_solve_claim_count"] = 1
    bad_solve["reproducibility_checksum"] = mod.payload_checksum(bad_solve)
    assert "arc_solve_claim_count" in mod.validate_report(bad_solve)

    bad_credit = deepcopy(report)
    bad_credit["arc_level_credit_delta"] = 1
    bad_credit["reproducibility_checksum"] = mod.payload_checksum(bad_credit)
    assert "arc_level_credit_delta" in mod.validate_report(bad_credit)

    bad_hardware = deepcopy(report)
    bad_hardware["hardware_claim_eligibility"]["unauthorized_hardware_claim_counts"]["speed"] = 1
    bad_hardware["reproducibility_checksum"] = mod.payload_checksum(bad_hardware)
    assert "hardware_claim_eligibility" in mod.validate_report(bad_hardware)

    bad_registry = deepcopy(report)
    bad_registry["arc_registry_hash_before_after"]["after_sha256"] = "sha256:changed"
    bad_registry["arc_registry_hash_before_after"]["unchanged"] = False
    bad_registry["reproducibility_checksum"] = mod.payload_checksum(bad_registry)
    assert "arc_registry_hash_before_after" in mod.validate_report(bad_registry)

    bad_provenance = deepcopy(report)
    bad_provenance["field_provenance"]["status"]["principle"] = "wrong"
    bad_provenance["reproducibility_checksum"] = mod.payload_checksum(bad_provenance)
    assert "field_provenance:status" in mod.validate_report(bad_provenance)

    bad_verdict = deepcopy(report)
    bad_verdict["honest_verdict"] = "blocked_missing"
    bad_verdict["reproducibility_checksum"] = mod.payload_checksum(bad_verdict)
    assert "honest_verdict" in mod.validate_report(bad_verdict)

    with pytest.raises(ValueError, match="invalid Exp6224 capstone"):
        mod.write_capstone(
            root=tmp_path,
            date="20260809",
            verifier_receipts=_receipts(),
            command_receipts=_command_receipts(),
            tests_run=_tests_run(),
            duration_s=4.0,
            validator=lambda _report: ["forced"],
        )


def test_req_capstone_6224_defensive_helpers_and_validation_edges(
    tmp_path: Path,
) -> None:
    """REQ-CAPSTONE-6224: helper edges fail closed without promotion."""

    report = _build(tmp_path)

    assert mod._read_yaml_mapping(tmp_path / "missing.yaml") == {}
    _write_text(tmp_path, "array.yaml", "[]\n")
    assert mod._read_yaml_mapping(tmp_path / "array.yaml") == {}

    fallback = tmp_path / "fallback"
    _write_text(
        fallback,
        mod.ROADMAP_RELATIVE_PATH,
        yaml.safe_dump(
            {
                "tasks": [
                    {
                        "id": "exp6211-fallback",
                        "deliverable": "results/experiment_6211_fallback.json",
                    },
                    {
                        "id": "exp6224-out-of-range",
                        "deliverable": "results/experiment_6224_out.json",
                    },
                ]
            }
        ),
    )
    declared, _roadmap, capstone = mod._roadmap_declared_tasks(fallback)
    assert [row["task_id"] for row in declared] == ["exp6211-fallback"]
    assert capstone == {}
    requires_root = tmp_path / "requires"
    _write_text(
        requires_root,
        mod.ROADMAP_RELATIVE_PATH,
        yaml.safe_dump(
            {
                "tasks": [
                    {
                        "id": "exp6211-in-range",
                        "deliverable": "results/experiment_6211_in_range.json",
                    },
                    {
                        "id": "exp6224-out-of-range",
                        "deliverable": "results/experiment_6224_out.json",
                    },
                    {
                        "id": mod.EXPERIMENT_ID,
                        "requires": ["exp6211-in-range", "exp6224-out-of-range"],
                        "deliverable": mod.RESULT_RELATIVE_PATH.as_posix(),
                    },
                ]
            }
        ),
    )
    explicit, _roadmap, capstone = mod._roadmap_declared_tasks(requires_root)
    assert [row["task_id"] for row in explicit] == ["exp6211-in-range"]
    assert capstone["id"] == mod.EXPERIMENT_ID

    command_rows = mod._command_receipts(
        tmp_path,
        None,
        lambda argv, _root: {"command": " ".join(argv), "exit_code": 0},
    )
    assert len(command_rows) == len(mod.DEFAULT_COMMANDS)
    assert mod._argv("alpha beta") == ("alpha", "beta")
    commands, exits = mod._normalize_test_commands(
        None, [{"command": "", "exit_code": 7}, {"command": "cmd", "exit_code": 2}]
    )
    assert commands == ["cmd"]
    assert exits == {"cmd": 2}

    assert mod._publication_snapshot(
        [
            {
                "command": ".venv/bin/python scripts/publication_gate.py --json",
                "stdout_tail": json.dumps({"paper_ready": True}),
            }
        ]
    ) == {"paper_ready": True}
    assert mod._publication_snapshot(
        [
            {
                "command": ".venv/bin/python scripts/publication_gate.py --json",
                "stdout_tail": "{",
            }
        ]
    )["unmet_gates"] == ["publication_gate_not_run"]
    assert mod._publication_snapshot([])["unmet_gates"] == ["publication_gate_not_run"]

    assert mod._real_field_items({"field_principles": {"x": 1}}) == []
    assert mod._real_field_items({"rows": [1, {"value": 2}]}) == [
        (("rows", "0"), 1),
        (("rows", "1", "value"), 2),
    ]
    assert mod._positive_int(True) == 1
    assert mod._positive_int(3.5) == 3
    assert mod._arc_claim_counts(
        {
            "arc": {
                "solve_claimed": True,
                "solve_claim_count": 2,
                "level_credit_delta": 1,
                "field_principles": {"arc_level_credit_delta": 99},
            }
        }
    ) == (3, 1)
    assert (
        mod._hardware_claim_counts(
            {
                "hw": {
                    "speed_power_energy_terminal_tsu_kona_claim_counts": {
                        "speed": 2,
                        "power": 0,
                    },
                    "hardware_claim_count": 1,
                    "hardware_speedup_claim": True,
                    "hardware_execution_claimed": True,
                }
            }
        )["hardware_execution"]
        == 1
    )
    assert "missing_required_field" in mod._task_blockers(
        "task",
        {
            "task": {
                "terminal_class": "complete",
                "critical_adversarial_flag_count": 0,
                "missing_required_fields": ["x"],
                "unclassified_nonzero_commands": ["cmd"],
            }
        },
        {"task": {"declared_gates": []}},
        {"task": {}},
    )

    bad_status = deepcopy(report)
    bad_status["status"] = "blocked"
    bad_status["reproducibility_checksum"] = mod.payload_checksum(bad_status)
    assert "status" in mod.validate_report(bad_status)

    bad_substrate = deepcopy(report)
    bad_substrate["inference_substrate"] = "live_llm_inference"
    bad_substrate["reproducibility_checksum"] = mod.payload_checksum(bad_substrate)
    assert "inference_substrate" in mod.validate_report(bad_substrate)

    bad_oracle = deepcopy(report)
    bad_oracle["verifier_is_oracle"] = True
    bad_oracle["reproducibility_checksum"] = mod.payload_checksum(bad_oracle)
    assert "verifier_is_oracle" in mod.validate_report(bad_oracle)

    bad_protected = deepcopy(report)
    bad_protected["protected_files_unchanged"]["unchanged"] = False
    bad_protected["reproducibility_checksum"] = mod.payload_checksum(bad_protected)
    assert "protected_files_unchanged" in mod.validate_report(bad_protected)

    bad_docs = deepcopy(report)
    bad_docs["spec_trace_status_changelog_reconciliation"][
        "ops_status_changelog_traceability_modified"
    ] = True
    bad_docs["reproducibility_checksum"] = mod.payload_checksum(bad_docs)
    assert "spec_trace_status_changelog_reconciliation" in mod.validate_report(bad_docs)

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
