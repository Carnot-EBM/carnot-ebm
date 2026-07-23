"""Tests for the Exp5837 V520 transition receipt.

Spec refs: REQ-REPORT-5837, SCENARIO-REPORT-5837,
SCENARIO-REPORT-5837-VERIFIER-RECEIPTS,
SCENARIO-REPORT-5837-COLLISION-BLOCK,
SCENARIO-REPORT-5837-FIELD-PROVENANCE.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_5837_transition_v520 as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-reporting/spec.md"


def _write_json(root: Path, rel_path: Path | str, payload: Any) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(root: Path, rel_path: Path | str, text: str = "context\n") -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _artifact_payload(task_id: str, exp5828_sha: str | None = None) -> JsonDict:
    payloads: dict[str, JsonDict] = {
        "exp5823-transition-v519": {
            "status": "complete",
            "honest_verdict": "complete: archived terminal .518 evidence by exact paths",
            "next_task_range": "exp5823-exp5836",
            "next_range_collision_count": 0,
        },
        "exp5824-v519-source-delta-ingestion": {
            "status": "complete",
            "honest_verdict": "complete: no accepted post-V519 source deltas",
            "accepted_finding_count": 0,
        },
        "exp5825-certified-adaptive-memory-contract": {
            "status": "complete",
            "honest_verdict": "complete: certified_adaptive_memory_contract_ready",
            "duration_s": 0.167314,
            "inference_substrate": "aggregation_from_upstream_artifacts",
        },
        "exp5826-out-of-template-constraint-stream": {
            "status": "complete",
            "honest_verdict": "complete: out_of_template_constraint_event_stream_ready",
            "duration_s": 0.963263,
            "inference_substrate": "deterministic_exact_solver_dataset_generation_no_llm",
        },
        "exp5827-minimal-core-structural-acquisition-ab": {
            "status": "complete",
            "honest_verdict": "complete: structural_learning_credited",
            "duration_s": 0.947768,
            "inference_substrate": "online_exact_membership_query_sidecar_no_llm",
        },
        "exp5828-future-validated-structural-memory": {
            "status": "complete",
            "honest_verdict": "complete: future_validated_structural_memory_lifecycle_credited",
            "duration_s": 0.793358,
            "inference_substrate": "online_exact_membership_query_sidecar_no_llm",
            "flagged_adversarial": True,
            "corrigendum_pending": [
                {
                    "kind": "DURATION_TOO_SHORT",
                    "severity": "critical",
                    "detail": "too fast for declared compute markers",
                },
                {
                    "kind": "METHODOLOGY_MISSING",
                    "severity": "warn",
                    "detail": "missing model_specs/target_model",
                },
            ],
        },
        "exp5829-transfer-selective-replay-audit": {
            "status": "complete",
            "honest_verdict": "positive: signature_compatible_replay_credited",
            "duration_s": 3.615075,
            "inference_substrate": "aggregation_from_upstream_artifacts",
            "upstream_artifact_hashes": {
                "exp5828_lifecycle_artifact": exp5828_sha or "sha256:fixture"
            },
        },
    }
    return payloads[task_id]


def _research_complete_payload(*, duplicate_519_blocks: int = 1) -> JsonDict:
    v519_block = {
        "id": mod.MILESTONE_FROM,
        "title": "Terminal V519",
        "doc": "openspec/change-proposals/research-roadmap-vNEXT.md",
        "completed": "2026-07-23",
        "finding": "See conductor log for per-experiment results.",
        "tasks": [
            {
                "id": task_id,
                "title": f"title for {task_id}",
                "deliverable": rel_path.as_posix(),
                "result": "OK (conductor)",
            }
            for task_id, rel_path in mod.TASK_ARTIFACT_PATHS.items()
        ],
    }
    return {"milestones": [v519_block for _ in range(duplicate_519_blocks)]}


def _active_roadmap_payload() -> JsonDict:
    active_ids = (
        "exp5837-transition-v520",
        "exp5838-v520-source-delta-ingestion",
        "exp5839-v519-evidence-qualification",
        "exp5840-exact-counterfactual-embedding-fixture",
    )
    return {
        "milestone": mod.MILESTONE_TO,
        "tasks": [
            {
                "id": task_id,
                "milestone": mod.MILESTONE_TO,
                "deliverable": mod.NEXT_TASK_ARTIFACT_PATHS[task_id].as_posix(),
            }
            for task_id in active_ids
        ],
    }


def _vnext_doc() -> str:
    lines = [
        "# Research Roadmap vNEXT",
        "",
        "**Milestone:** 2026.07.520",
        "**Task range:** Exp5837-Exp5848",
        "",
    ]
    lines.extend(
        f"`{mod.NEXT_TASK_ARTIFACT_PATHS[task_id].as_posix()}`" for task_id in mod.NEXT_TASK_IDS
    )
    lines.extend(f"proposal-only `{task_id}`" for task_id in mod.RESERVED_UNACTIVATED_TASK_IDS)
    return "\n".join(lines) + "\n"


def _conductor_log() -> str:
    return "\n".join(
        [
            "| 2026-07-23 00:09 UTC | Archive terminal .518 evidence, retire finite-ID a | OK | 118 passed |",
            "| 2026-07-23 00:33 UTC | Time-windowed post-V519 literature and implementat | OK | 88 passed |",
            "| 2026-07-23 01:15 UTC | Canonical certified-event and adaptive-memory pref | FAIL | bootstrap |",
            "| 2026-07-23 01:34 UTC | Canonical certified-event and adaptive-memory pref | OK | 140 passed |",
            "| 2026-07-23 01:57 UTC | Gated on Exp5825 contract: chronological out-of-te | OK | 88 passed |",
            "| 2026-07-23 02:18 UTC | Gated on Exp5826 stream: minimal-core and active-q | OK | 88 passed |",
            "| 2026-07-23 02:52 UTC | Gated on Exp5827 structural lift: future-validated | FAIL | bootstrap |",
            "| 2026-07-23 03:14 UTC | Gated on Exp5827 structural lift: future-validated | FLAGGED | adversarial_verify CRITICAL: DURATION_TOO_SHORT |",
            "| 2026-07-23 03:37 UTC | Gated on Exp5828 durable memory: transfer-selectiv | OK | 88 passed |",
        ]
    )


def _receipt(
    task_id: str, *, critical: tuple[str, ...] = (), warn: tuple[str, ...] = ()
) -> JsonDict:
    path = mod.TASK_ARTIFACT_PATHS[task_id].as_posix()
    flags = [
        {"kind": kind, "severity": "critical", "detail": f"{kind} detail"} for kind in critical
    ]
    flags.extend({"kind": kind, "severity": "warn", "detail": f"{kind} detail"} for kind in warn)
    report = {
        "reports": [
            {
                "path": path,
                "flag_count": len(flags),
                "flags": flags,
                "max_severity": 2 if critical else 1 if warn else -1,
            }
        ],
        "flagged_count": 1 if flags else 0,
    }
    encoded = json.dumps(report, sort_keys=True).encode("utf-8")
    return {
        "task_id": task_id,
        "artifact_path": path,
        "command": f".venv/bin/python scripts/adversarial_verify.py --json {path}",
        "exit_code": 1 if flags else 0,
        "stdout_json": report,
        "stderr": "",
        "receipt_hash": mod.sha256_bytes(encoded),
    }


def _clean_receipts() -> list[JsonDict]:
    return [
        _receipt("exp5825-certified-adaptive-memory-contract"),
        _receipt("exp5826-out-of-template-constraint-stream"),
        _receipt("exp5827-minimal-core-structural-acquisition-ab"),
        _receipt(
            "exp5828-future-validated-structural-memory",
            critical=("DURATION_TOO_SHORT",),
            warn=("METHODOLOGY_MISSING",),
        ),
        _receipt("exp5829-transfer-selective-replay-audit"),
    ]


def _make_root(
    root: Path,
    *,
    duplicate_519_blocks: int = 1,
    include_research_complete: bool = True,
) -> None:
    for task_id, rel_path in mod.TASK_ARTIFACT_PATHS.items():
        if task_id == "exp5829-transfer-selective-replay-audit":
            continue
        _write_json(root, rel_path, _artifact_payload(task_id))
    exp5828_sha = mod.path_sha256(
        root / mod.TASK_ARTIFACT_PATHS["exp5828-future-validated-structural-memory"]
    )
    _write_json(
        root,
        mod.TASK_ARTIFACT_PATHS["exp5829-transfer-selective-replay-audit"],
        _artifact_payload("exp5829-transfer-selective-replay-audit", exp5828_sha),
    )
    _write_text(root, mod.ROADMAP_RELATIVE_PATH, yaml.safe_dump(_active_roadmap_payload()))
    if include_research_complete:
        _write_text(
            root,
            mod.RESEARCH_COMPLETE_RELATIVE_PATH,
            yaml.safe_dump(_research_complete_payload(duplicate_519_blocks=duplicate_519_blocks)),
        )
    _write_text(root, mod.VNEXT_RELATIVE_PATH, _vnext_doc())
    _write_text(root, mod.CONDUCTOR_LOG_RELATIVE_PATH, _conductor_log())
    _write_text(root, mod.EXCLUSION_MANIFEST_RELATIVE_PATH, "retired_experiments: []\n")
    _write_text(root, mod.CONDUCTOR_RELATIVE_PATH, "# conductor fixture\n")
    _write_text(root, mod.EVIDENCE_INDEX_RELATIVE_PATH, "# evidence index fixture\n")
    _write_text(root, mod.DOC_RECONCILE_RELATIVE_PATH, "# reconcile fixture\n")
    _write_text(root, mod.SPEC_RELATIVE_PATH, "### REQ-REPORT-5837\n")


def _clean_build(root: Path, *, receipts: list[JsonDict] | None = None) -> JsonDict:
    return mod.build_report(
        root,
        adversarial_receipts=receipts or _clean_receipts(),
        tests_run=[{"command": "unit", "exit_code": 0}],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
        duration_s=1.25,
    )


def test_spec_contains_req_report_5837_contract() -> None:
    """REQ-REPORT-5837: OpenSpec names exact identity, verifier, and range gates."""

    text = SPEC_PATH.read_text(encoding="utf-8")
    section = text[text.index("### REQ-REPORT-5837") :]

    assert str(mod.RESULT_RELATIVE_PATH) in section
    assert "(milestone, task_id, declared_deliverable)" in section
    assert "SCENARIO-REPORT-5837-VERIFIER-RECEIPTS" in section
    assert "Exp5830 through Exp5836" in section
    assert "Exp5837 through Exp5848" in section
    assert "DURATION_TOO_SHORT" in section
    assert "METHODOLOGY_MISSING" in section
    for field in mod.REQUIRED_PRINCIPLE_FIELDS:
        assert f"`{field}`" in section


def test_scenario_report_5837_archives_terminal_v519_by_exact_identity(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5837: declared V519 paths and disjoint classes are canonical."""

    _make_root(tmp_path)
    report = _clean_build(tmp_path)

    assert report["status"] == "complete"
    assert report["honest_verdict"].startswith("complete:")
    assert report["milestone_transition"] == {
        "source_milestone": "2026.07.519",
        "destination_milestone": "2026.07.520",
        "task_identity_tuple": ["milestone", "task_id", "declared_deliverable"],
    }
    assert len(report["declared_deliverable_matrix"]) == 7
    assert report["outcome_classification"]["clean_positive_task_ids"] == [
        "exp5823-transition-v519",
        "exp5825-certified-adaptive-memory-contract",
        "exp5826-out-of-template-constraint-stream",
        "exp5827-minimal-core-structural-acquisition-ab",
    ]
    assert report["outcome_classification"]["clean_null_task_ids"] == [
        "exp5824-v519-source-delta-ingestion"
    ]
    assert report["outcome_classification"]["flagged_task_ids"] == [
        "exp5828-future-validated-structural-memory"
    ]
    assert report["outcome_classification"]["flagged_upstream_provisional_task_ids"] == [
        "exp5829-transfer-selective-replay-audit"
    ]
    assert report["outcome_classification"]["clean_success_task_ids"] == [
        "exp5823-transition-v519",
        "exp5825-certified-adaptive-memory-contract",
        "exp5826-out-of-template-constraint-stream",
        "exp5827-minimal-core-structural-acquisition-ab",
    ]
    assert report["outcome_classification"]["proposal_only_task_ids"] == list(
        mod.RESERVED_UNACTIVATED_TASK_IDS
    )

    exp5828 = next(
        row
        for row in report["declared_deliverable_matrix"]
        if row["task_id"] == "exp5828-future-validated-structural-memory"
    )
    assert exp5828["outcome_class"] == "flagged"
    assert exp5828["artifact_flagged_adversarial"] is True
    assert exp5828["adversarial_critical_findings"] == ["DURATION_TOO_SHORT"]
    assert exp5828["adversarial_warn_findings"] == ["METHODOLOGY_MISSING"]

    exp5829 = report["declared_deliverable_matrix"][-1]
    assert exp5829["task_id"] == "exp5829-transfer-selective-replay-audit"
    assert exp5829["outcome_class"] == "flagged-upstream/provisional"
    assert exp5829["upstream_taint_source_task_id"] == "exp5828-future-validated-structural-memory"

    assert report["flagged_evidence_preserved"] is True
    assert report["reserved_unactivated_task_ids"] == list(mod.RESERVED_UNACTIVATED_TASK_IDS)
    assert report["research_complete_append_count"] == 0
    assert report["preconditions_checked"]["roadmaps"]["next"]["present"] is False
    assert report["preconditions_checked"]["resource_receipts"]["disk_free_bytes"] > 0
    assert report["next_task_range"] == "exp5837-exp5848"
    assert report["next_range_collision_count"] == 0
    assert report["docs_reconciled"]["operator_owned_docs_deferred"] is True
    assert report["research_roadmap_unchanged"] is True
    assert report["conductor_unchanged"] is True


def test_scenario_report_5837_verifier_receipts_preserve_flag_and_taint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-REPORT-5837-VERIFIER-RECEIPTS: live receipts are authoritative."""

    _make_root(tmp_path)
    report = _clean_build(tmp_path)
    receipts = {row["task_id"]: row for row in report["adversarial_verifier_receipts"]}

    assert receipts["exp5825-certified-adaptive-memory-contract"]["headline_eligible"] is True
    assert receipts["exp5828-future-validated-structural-memory"]["exit_code"] == 1
    assert receipts["exp5828-future-validated-structural-memory"]["critical_findings"] == [
        {
            "kind": "DURATION_TOO_SHORT",
            "severity": "critical",
            "detail": "DURATION_TOO_SHORT detail",
        }
    ]
    assert receipts["exp5828-future-validated-structural-memory"]["warn_findings"] == [
        {"kind": "METHODOLOGY_MISSING", "severity": "warn", "detail": "METHODOLOGY_MISSING detail"}
    ]
    assert receipts["exp5828-future-validated-structural-memory"]["headline_eligible"] is False
    assert receipts["exp5829-transfer-selective-replay-audit"]["headline_eligible"] is False
    assert receipts["exp5829-transfer-selective-replay-audit"]["headline_ineligible_reason"] == (
        "upstream_flagged_exp5828"
    )

    missing_receipt = _clean_build(tmp_path, receipts=_clean_receipts()[:-1])
    assert (
        "adversarial_receipts_missing=['exp5829-transfer-selective-replay-audit']"
        in (missing_receipt["preconditions_checked"]["failed_preconditions"])
    )

    stale_receipts = _clean_receipts()
    stale_receipts[3] = _receipt("exp5828-future-validated-structural-memory")
    stale_report = _clean_build(tmp_path, receipts=stale_receipts)
    assert (
        "exp5828_live_stamp_not_preserved"
        in stale_report["preconditions_checked"]["failed_preconditions"]
    )

    completed = mod.run_adversarial_verifier(
        tmp_path,
        mod.TASK_ARTIFACT_PATHS["exp5825-certified-adaptive-memory-contract"],
        subprocess_run=lambda *_args, **_kwargs: mod.ProcessResult(
            returncode=0,
            stdout=json.dumps({"reports": [{"path": "x", "flags": [], "flag_count": 0}]}),
            stderr="",
        ),
    )
    assert completed["exit_code"] == 0
    assert completed["task_id"] == "exp5825-certified-adaptive-memory-contract"
    assert completed["receipt_hash"].startswith("sha256:")

    failed = mod.run_adversarial_verifier(
        tmp_path,
        mod.TASK_ARTIFACT_PATHS["exp5828-future-validated-structural-memory"],
        subprocess_run=lambda *_args, **_kwargs: mod.ProcessResult(
            returncode=1,
            stdout="{not-json",
            stderr="broken",
        ),
    )
    assert failed["stdout_parse_error"]

    monkeypatch.setattr(mod, "VERIFIER_TASK_IDS", ("exp9999-other",))
    try:
        assert mod.normalize_adversarial_verifier_receipts([])[0]["task_id"] == "exp9999-other"
    finally:
        monkeypatch.setattr(
            mod,
            "VERIFIER_TASK_IDS",
            (
                "exp5825-certified-adaptive-memory-contract",
                "exp5826-out-of-template-constraint-stream",
                "exp5827-minimal-core-structural-acquisition-ab",
                "exp5828-future-validated-structural-memory",
                "exp5829-transfer-selective-replay-audit",
            ),
        )


def test_scenario_report_5837_collision_blocks_allocation(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5837-COLLISION-BLOCK: occupied V520 ids fail closed."""

    _make_root(tmp_path)
    _write_json(tmp_path, "results/experiment_5847_existing_collision.json", {"status": "old"})
    payload = _research_complete_payload()
    payload["milestones"].append(
        {
            "id": "2026.07.400",
            "title": "stale next-range reference",
            "finding": "mentions exp5843-sparse-oracle-continuous-learning",
            "tasks": [],
        }
    )
    _write_text(tmp_path, mod.RESEARCH_COMPLETE_RELATIVE_PATH, yaml.safe_dump(payload))
    _write_text(
        tmp_path,
        mod.EXCLUSION_MANIFEST_RELATIVE_PATH,
        "retired_experiments:\n- scope_key: exp5846-bounded-adaptive-memory-microkernel\n",
    )
    _write_json(
        tmp_path,
        "results/experiment_5700_transition_alloc.json",
        {"next_task_range": "exp5837-exp5848"},
    )

    report = _clean_build(tmp_path)

    assert report["status"] == "blocked"
    assert report["honest_verdict"].startswith("blocked:")
    assert report["next_range_collision_count"] == 4
    assert [row["path"] for row in report["collision_scan"]["preexisting_collisions"]] == [
        "ops/exclusion_manifest.yaml",
        "research-complete.yaml",
        "results/experiment_5700_transition_alloc.json",
        "results/experiment_5847_existing_collision.json",
    ]

    alias_root = tmp_path / "alias"
    _make_root(alias_root)
    _write_json(
        alias_root, "results/experiment_5828_same_number_alias.json", {"status": "complete"}
    )
    alias_report = _clean_build(alias_root)
    assert alias_report["same_number_alias_groups"]["5828"]["aliases"][0]["path"] == (
        "results/experiment_5828_same_number_alias.json"
    )


def test_scenario_report_5837_emit_report_field_provenance_checksum_and_append(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5837-FIELD-PROVENANCE: emitted artifact is stable."""

    _make_root(tmp_path)
    history_before = (tmp_path / mod.RESEARCH_COMPLETE_RELATIVE_PATH).read_bytes()
    output = tmp_path / mod.RESULT_RELATIVE_PATH
    report = mod.emit_report(
        tmp_path,
        output_path=output,
        adversarial_receipts=_clean_receipts(),
        tests_run=[{"command": "unit", "exit_code": 0}],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
        duration_s=1.25,
    )
    written = json.loads(output.read_text(encoding="utf-8"))

    assert (tmp_path / mod.RESEARCH_COMPLETE_RELATIVE_PATH).read_bytes() == history_before
    assert written == report
    assert mod.payload_checksum(written) == written["reproducibility_checksum"]
    assert set(mod.REQUIRED_PRINCIPLE_FIELDS).issubset(report["field_principles"])
    assert set(mod.REQUIRED_PRINCIPLE_FIELDS).issubset(report["field_provenance"])
    assert all(report["field_principles"][field] for field in mod.REQUIRED_PRINCIPLE_FIELDS)
    assert all(
        report["field_provenance"][field]["sources"] for field in mod.REQUIRED_PRINCIPLE_FIELDS
    )
    assert all(
        "sha256_by_source" in report["field_provenance"][field]
        for field in mod.REQUIRED_PRINCIPLE_FIELDS
    )
    assert report["duration_s"] == 1.25
    assert report["inference_substrate"] == "aggregation_from_upstream_artifacts"

    append_root = tmp_path / "append"
    _make_root(append_root, include_research_complete=False)
    appended = _clean_build(append_root)
    assert appended["research_complete_append_count"] == 1
    assert mod._research_complete_blocks(append_root)
    second = _clean_build(append_root)
    assert second["research_complete_append_count"] == 0

    original = mod.FIELD_PRINCIPLES.pop("status")
    try:
        with pytest.raises(KeyError, match="missing field principles"):
            mod.build_report(
                tmp_path,
                adversarial_receipts=_clean_receipts(),
                modification_overrides={
                    mod.ROADMAP_RELATIVE_PATH: False,
                    mod.CONDUCTOR_RELATIVE_PATH: False,
                },
                duration_s=1.25,
            )
    finally:
        mod.FIELD_PRINCIPLES["status"] = original


def test_scenario_report_5837_defensive_preconditions_and_parsers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-5837: malformed inputs, identity ambiguity, and mutation fail closed."""

    assert mod._task_signature({"tasks": "not-list"}) == ()
    assert mod._artifact_terminal_status({}, {"exists": False, "loadable": False}) == "missing"
    assert mod._artifact_terminal_status({}, {"exists": True, "loadable": False}) == "malformed"
    assert (
        mod._artifact_terminal_status(
            {"honest_verdict": "blocked: no input"},
            {"exists": True, "loadable": True},
        )
        == "blocked"
    )
    assert (
        mod._artifact_terminal_status(
            {"honest_verdict": "positive: provisional"},
            {"exists": True, "loadable": True},
        )
        == "complete"
    )
    assert mod._artifact_terminal_status({}, {"exists": True, "loadable": True}) == "unknown"
    assert mod._task_number("not-an-exp") is None
    assert mod._conductor_outcomes(tmp_path)[1] == list(mod.EXPECTED_TASK_IDS)
    assert mod._parse_conductor_log(tmp_path / "missing-log-file") == []
    short_log_root = tmp_path / "short-log"
    short_log_root.mkdir()
    _write_text(short_log_root, mod.CONDUCTOR_LOG_RELATIVE_PATH, "| too | short |\n")
    assert mod._parse_conductor_log(short_log_root) == []
    assert mod._task_id_for_artifact_path(Path("results/not_declared.json")) == ""
    assert mod._flag_rows_from_stdout(None) == []
    assert mod._flag_rows_from_stdout({"reports": []}) == []
    assert mod._flag_rows_from_stdout({"reports": ["not-a-map"]}) == []
    assert mod._receipt_kinds({"critical_findings": "not-list"}, "critical_findings") == []
    nonzero = mod.normalize_adversarial_verifier_receipts(
        [
            {
                "task_id": "exp5825-certified-adaptive-memory-contract",
                "artifact_path": "results/experiment_5825_certified_adaptive_memory_contract.json",
                "command": "nonzero",
                "exit_code": 2,
                "stdout_json": {"reports": [{"flags": [], "flag_count": 0}]},
            }
        ]
    )[0]
    assert nonzero["headline_ineligible_reason"] == "verifier_exit_nonzero"
    assert (
        mod._adversarial_receipt_failures(
            {
                "exp5825-certified-adaptive-memory-contract": {
                    "present": True,
                    "critical_findings": [
                        {"kind": "UNEXPECTED", "severity": "critical", "detail": "bad"}
                    ],
                    "warn_findings": [],
                    "exit_code": 1,
                },
                "exp5826-out-of-template-constraint-stream": {
                    "present": True,
                    "critical_findings": [],
                    "warn_findings": [],
                    "exit_code": 0,
                },
                "exp5827-minimal-core-structural-acquisition-ab": {
                    "present": True,
                    "critical_findings": [],
                    "warn_findings": [],
                    "exit_code": 0,
                },
                "exp5828-future-validated-structural-memory": {
                    "present": True,
                    "critical_findings": [{"kind": "DURATION_TOO_SHORT", "severity": "critical"}],
                    "warn_findings": [{"kind": "METHODOLOGY_MISSING", "severity": "warn"}],
                    "exit_code": 1,
                },
                "exp5829-transfer-selective-replay-audit": {
                    "present": True,
                    "critical_findings": [],
                    "warn_findings": [],
                    "exit_code": 0,
                },
            }
        )[0]
        == "exp5825-certified-adaptive-memory-contract_unexpected_critical_findings=['UNEXPECTED']"
    )

    malformed_yaml = tmp_path / "bad.yaml"
    malformed_yaml.write_text("not: [closed\n", encoding="utf-8")
    _payload, malformed_meta = mod._read_yaml_with_meta(malformed_yaml)
    assert malformed_meta["parsed"] is False
    assert malformed_meta["error"]
    list_yaml = tmp_path / "list.yaml"
    list_yaml.write_text("- item\n", encoding="utf-8")
    _payload, list_meta = mod._read_yaml_with_meta(list_yaml)
    assert list_meta["parsed"] is False
    assert list_meta["error"] == "expected mapping, got list"

    ambiguous_root = tmp_path / "ambiguous-history"
    _make_root(ambiguous_root)
    ambiguous_payload = _research_complete_payload()
    altered = json.loads(json.dumps(ambiguous_payload["milestones"][0]))
    altered["tasks"][0]["deliverable"] = "results/other.json"
    ambiguous_payload["milestones"].append(altered)
    _write_text(
        ambiguous_root, mod.RESEARCH_COMPLETE_RELATIVE_PATH, yaml.safe_dump(ambiguous_payload)
    )
    ambiguous = _clean_build(ambiguous_root)
    assert (
        "ambiguous_research_complete_declared_task_blocks"
        in ambiguous["preconditions_checked"]["failed_preconditions"]
    )

    duplicate_task_root = tmp_path / "duplicate-task"
    _make_root(duplicate_task_root)
    duplicate_payload = _research_complete_payload()
    duplicate_task = dict(duplicate_payload["milestones"][0]["tasks"][0])
    duplicate_task["deliverable"] = "results/conflicting.json"
    duplicate_payload["milestones"][0]["tasks"].append(duplicate_task)
    _write_text(
        duplicate_task_root,
        mod.RESEARCH_COMPLETE_RELATIVE_PATH,
        yaml.safe_dump(duplicate_payload),
    )
    duplicate_task_report = _clean_build(duplicate_task_root)
    assert any(
        item.startswith("duplicate_task_id_conflicts=")
        for item in duplicate_task_report["preconditions_checked"]["failed_preconditions"]
    )
    assert any(
        item.startswith("declared_task_ids_mismatch=")
        for item in duplicate_task_report["preconditions_checked"]["failed_preconditions"]
    )

    missing_artifact_root = tmp_path / "missing-artifact"
    _make_root(missing_artifact_root)
    (
        missing_artifact_root
        / mod.TASK_ARTIFACT_PATHS["exp5825-certified-adaptive-memory-contract"]
    ).unlink()
    missing_artifact = _clean_build(missing_artifact_root)
    assert (
        "missing_or_malformed_declared_deliverables=['exp5825-certified-adaptive-memory-contract']"
        in missing_artifact["preconditions_checked"]["failed_preconditions"]
    )

    no_stamp_root = tmp_path / "no-stamp"
    _make_root(no_stamp_root)
    exp5828 = _artifact_payload("exp5828-future-validated-structural-memory")
    exp5828["flagged_adversarial"] = False
    _write_json(
        no_stamp_root,
        mod.TASK_ARTIFACT_PATHS["exp5828-future-validated-structural-memory"],
        exp5828,
    )
    no_stamp = _clean_build(no_stamp_root)
    assert (
        "exp5828_artifact_stamp_not_preserved"
        in no_stamp["preconditions_checked"]["failed_preconditions"]
    )

    no_taint_root = tmp_path / "no-taint"
    _make_root(no_taint_root)
    exp5829 = _artifact_payload("exp5829-transfer-selective-replay-audit", "sha256:wrong")
    _write_json(
        no_taint_root,
        mod.TASK_ARTIFACT_PATHS["exp5829-transfer-selective-replay-audit"],
        exp5829,
    )
    no_taint = _clean_build(no_taint_root)
    assert (
        "exp5829_upstream_taint_not_preserved"
        in no_taint["preconditions_checked"]["failed_preconditions"]
    )

    bad_active_root = tmp_path / "bad-active"
    _make_root(bad_active_root)
    _write_text(bad_active_root, mod.ROADMAP_RELATIVE_PATH, "bad: [yaml\n")
    bad_active = _clean_build(bad_active_root)
    assert (
        "active_roadmap_unparseable" in bad_active["preconditions_checked"]["failed_preconditions"]
    )

    bad_task_root = tmp_path / "bad-task"
    _make_root(bad_task_root)
    _write_text(
        bad_task_root,
        mod.ROADMAP_RELATIVE_PATH,
        yaml.safe_dump(
            {
                "milestone": mod.MILESTONE_TO,
                "tasks": [{"id": "exp9999-not-allocated", "deliverable": "results/x.json"}],
            }
        ),
    )
    bad_task = _clean_build(bad_task_root)
    assert any(
        item.startswith("active_roadmap_task_ids=")
        for item in bad_task["preconditions_checked"]["failed_preconditions"]
    )

    bad_next_root = tmp_path / "bad-next"
    _make_root(bad_next_root)
    _write_text(bad_next_root, mod.ROADMAP_NEXT_RELATIVE_PATH, "bad: [yaml\n")
    bad_next = _clean_build(bad_next_root)
    assert "next_roadmap_unparseable" in bad_next["preconditions_checked"]["failed_preconditions"]

    bad_complete_root = tmp_path / "bad-complete"
    _make_root(bad_complete_root)
    _write_text(bad_complete_root, mod.RESEARCH_COMPLETE_RELATIVE_PATH, "bad: [yaml\n")
    bad_complete = _clean_build(bad_complete_root)
    assert (
        "research_complete_unparseable"
        in bad_complete["preconditions_checked"]["failed_preconditions"]
    )

    existing_complete_root = tmp_path / "existing-complete"
    _make_root(existing_complete_root, include_research_complete=False)
    _write_text(existing_complete_root, mod.RESEARCH_COMPLETE_RELATIVE_PATH, "milestones:")
    existing_complete = _clean_build(existing_complete_root)
    assert existing_complete["research_complete_append_count"] == 1
    assert mod._research_complete_blocks(existing_complete_root)

    missing_log_root = tmp_path / "missing-log"
    _make_root(missing_log_root)
    _write_text(missing_log_root, mod.CONDUCTOR_LOG_RELATIVE_PATH, "no rows\n")
    missing_log = _clean_build(missing_log_root)
    assert any(
        item.startswith("missing_conductor_outcomes=")
        for item in missing_log["preconditions_checked"]["failed_preconditions"]
    )

    blocked_class = mod._classify_outcomes(
        {"exp5825-certified-adaptive-memory-contract": {"status": "blocked"}},
        {"exp5825-certified-adaptive-memory-contract": {"outcome": "GATE_BLOCK"}},
        {},
    )
    assert blocked_class["blocked_skipped_task_ids"] == [
        "exp5825-certified-adaptive-memory-contract"
    ]

    collision_branch_root = tmp_path / "collision-branches"
    _make_root(collision_branch_root)
    (collision_branch_root / "results/experiment_5844_directory").mkdir()
    _write_text(collision_branch_root, "results/experiment_5700_transition_bad.json", "{")
    collision_branch = _clean_build(collision_branch_root)
    assert collision_branch["next_range_collision_count"] == 0

    modified_root = tmp_path / "modified"
    _make_root(modified_root)
    modified = mod.build_report(
        modified_root,
        adversarial_receipts=_clean_receipts(),
        tests_run=[{"command": "unit", "exit_code": 1}],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: True,
            mod.CONDUCTOR_RELATIVE_PATH: True,
        },
    )
    failed = modified["preconditions_checked"]["failed_preconditions"]
    assert "research_roadmap_modified" in failed
    assert "research_conductor_modified" in failed
    assert any(item.startswith("test_failures=") for item in failed)
    assert modified["duration_s"] > 0
    assert (
        mod._tests_failed([{"command": "global baseline", "exit_code": 1, "blocking": False}]) == []
    )

    monkeypatch.setattr(mod, "EXPECTED_TASK_IDS", ("not-an-exp-task",))
    assert mod._same_number_alias_groups(tmp_path, {}) == {}
