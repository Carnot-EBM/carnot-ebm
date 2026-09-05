"""Independent belief-ledger cold-audit tests.

Spec refs: REQ-CSL-7022, SCENARIO-CSL-7022-PRECONDITIONS,
SCENARIO-CSL-7022-ISOLATED-REPLAY, SCENARIO-CSL-7022-MUTATION-SENSITIVITY,
SCENARIO-CSL-7022-ORDER-AUTHORITY-AND-RETRIEVAL,
SCENARIO-CSL-7022-CONFLICT-CAPACITY-AND-POISON,
SCENARIO-CSL-7022-RESTART-INTERRUPTION-ROLLBACK,
SCENARIO-CSL-7022-ROW-RECOMPUTATION, and
SCENARIO-CSL-7022-SEPARATE-DECISIONS.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from carnot.agentic import arc_belief_ledger as ledger_mod
from carnot.agentic import arc_belief_ledger_cold_audit as exp


REPO_ROOT = Path(__file__).resolve().parents[2]


def _passing_runtime() -> dict:
    return {
        "fresh_process": True,
        "network_namespace_isolated": True,
        "network_disabled": True,
        "external_lookup_rejected": True,
        "gpu_devices_visible": [],
        "cuda_visible_devices": "",
        "game_source_access_rejected": True,
        "registry_access_rejected": True,
        "source_tree_read_only": True,
        "passed": True,
    }


@pytest.fixture(scope="module")
def source_bundle() -> dict:
    loaded = exp.load_hashed_sources(REPO_ROOT)
    assert loaded["passed"] is True
    return loaded


@pytest.fixture(scope="module")
def frozen_artifact(tmp_path_factory: pytest.TempPathFactory) -> dict:
    work_root = tmp_path_factory.mktemp("exp7022-audit")
    return exp.build_from_repo(
        REPO_ROOT,
        run_date=exp.RUN_DATE,
        output_path=work_root / "result.json",
        work_root=work_root,
        runtime_receipt=_passing_runtime(),
        write_output=False,
    )


def test_req_csl_7022_spec_precedes_implementation() -> None:
    """REQ-CSL-7022 owns all audit surfaces before implementation."""

    text = (REPO_ROOT / exp.SPEC_PATH).read_text(encoding="utf-8")
    section = text.split("## REQ-CSL-7022", 1)[1]
    for marker in (
        "SCENARIO-CSL-7022-PRECONDITIONS",
        "SCENARIO-CSL-7022-ISOLATED-REPLAY",
        "SCENARIO-CSL-7022-MUTATION-SENSITIVITY",
        "SCENARIO-CSL-7022-ORDER-AUTHORITY-AND-RETRIEVAL",
        "SCENARIO-CSL-7022-CONFLICT-CAPACITY-AND-POISON",
        "SCENARIO-CSL-7022-RESTART-INTERRUPTION-ROLLBACK",
        "SCENARIO-CSL-7022-ROW-RECOMPUTATION",
        "SCENARIO-CSL-7022-SEPARATE-DECISIONS",
        exp.INFERENCE_SUBSTRATE,
        "blocked_belief_ledger_cold_audit",
    ):
        assert marker in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_7022_preconditions_hash_before_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-CSL-7022-PRECONDITIONS rejects changed bytes before JSON parsing."""

    source = tmp_path / "exp7020.json"
    source.write_text('{"belief_ledger_ready_score":1}\n', encoding="utf-8")

    def forbidden_parse(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("source headline parsed before its hash passed")

    monkeypatch.setattr(exp.json, "loads", forbidden_parse)
    loaded = exp.load_hashed_sources(
        tmp_path,
        source_paths={"exp7020": Path("exp7020.json")},
        expected_hashes={"exp7020": "sha256:wrong"},
    )

    assert loaded["passed"] is False
    assert loaded["values"] == {}
    assert loaded["checks"][0]["check"] == "source_hash:exp7020"
    assert loaded["checks"][0]["observed_value"] == exp.sha256_path(source)


def test_scenario_7022_exact_sources_and_upstream_gates(source_bundle: dict) -> None:
    """SCENARIO-CSL-7022-PRECONDITIONS freezes both gates and every source hash."""

    assert source_bundle["hashes"] == exp.EXPECTED_SOURCE_HASHES
    checks = {row["check"]: row for row in source_bundle["checks"]}
    assert all(row["passed"] for row in checks.values())
    assert checks["exp7020_belief_ledger_ready_score"]["observed_value"] == 1
    assert checks["exp7021_comparison_complete_score"]["observed_value"] == 1
    assert checks["producer_code_importable"]["observed_value"] is True


def test_req_7022_source_loader_preserves_missing_and_invalid_inputs(tmp_path: Path) -> None:
    """REQ-CSL-7022 keeps absent bytes and invalid fixture rows explicit."""

    assert exp.sha256_path(tmp_path / "missing") is None
    source = tmp_path / "exp7020.json"
    source.write_text('{"belief_ledger_ready_score":1}\n', encoding="utf-8")
    without_fixture = exp.load_hashed_sources(
        tmp_path,
        source_paths={"exp7020": Path("exp7020.json")},
        expected_hashes={"exp7020": exp.sha256_path(source)},
    )
    assert without_fixture["passed"] is False
    assert without_fixture["values"]["exp7020"]["belief_ledger_ready_score"] == 1

    fixture = tmp_path / "fixture.jsonl"
    fixture.write_text('{"schema":"wrong"}\n', encoding="utf-8")
    invalid_fixture = exp.load_hashed_sources(
        tmp_path,
        source_paths={"construction_fixture": Path("fixture.jsonl")},
        expected_hashes={"construction_fixture": exp.sha256_path(fixture)},
    )
    checks = {row["check"]: row for row in invalid_fixture["checks"]}
    assert checks["construction_fixture_loadable"]["passed"] is False
    assert invalid_fixture["values"]["events"] is None


def test_req_7022_empty_reducers_preserve_unsupported_metrics() -> None:
    """REQ-CSL-7022 does not invent a metric or interval for empty rows."""

    assert exp._mean([]) is None
    assert exp._cluster_interval({}, seed=7, resamples=11) == {
        "point_estimate": None,
        "lower": None,
        "upper": None,
        "cluster_count": 0,
        "resamples": 11,
        "seed": 7,
    }


@pytest.mark.parametrize("field_name", sorted(ledger_mod.FORBIDDEN_UPDATER_KEYS))
def test_scenario_7022_each_prohibited_mutation_rejects_without_state_change(
    field_name: str, tmp_path: Path
) -> None:
    """SCENARIO-CSL-7022-MUTATION-SENSITIVITY rejects every denied field."""

    event = ledger_mod._fixture_event(0, mechanic="mutation", outcome="clean")
    row = exp.audit_one_forbidden_mutation(event, field_name, tmp_path / field_name)

    assert row["field_name"] == field_name
    assert row["write_rejected"] is True
    assert row["query_rejected"] is True
    assert row["state_unchanged"] is True
    assert row["passed"] is True
    assert row["terminal"] is True


@pytest.mark.parametrize("field_name", sorted(ledger_mod.FORBIDDEN_UPDATER_KEYS))
def test_scenario_7022_red_mutation_fails_if_field_auditor_is_bypassed(
    field_name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-CSL-7022-MUTATION-SENSITIVITY keeps every RED mutation sensitive."""

    monkeypatch.setattr(ledger_mod, "find_forbidden_paths", lambda _value: [])
    event = ledger_mod._fixture_event(0, mechanic="bypass", outcome="unsafe")
    row = exp.audit_one_forbidden_mutation(event, field_name, tmp_path / field_name)

    assert row["write_rejected"] is False
    assert row["state_unchanged"] is False
    assert row["passed"] is False


def test_req_7022_query_and_order_bypass_rows_turn_red(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CSL-7022 records a failed row if query or order rejection is bypassed."""

    event = ledger_mod._fixture_event(0, mechanic="bypass", outcome="unsafe")
    monkeypatch.setattr(ledger_mod.BeliefLedger, "query", lambda *_args, **_kwargs: {})
    mutation = exp.audit_one_forbidden_mutation(event, "game_id", tmp_path / "query")
    assert mutation["query_rejected"] is False
    assert mutation["passed"] is False

    monkeypatch.setattr(ledger_mod, "load_updater_events", lambda _path: [])
    order = exp._order_rows(
        [event, ledger_mod._fixture_event(1, mechanic="b", outcome="b")], tmp_path / "order"
    )
    assert all(row["rejected"] is False and row["passed"] is False for row in order)


def test_scenario_7022_isolated_replay_and_full_control_matrix(
    source_bundle: dict, tmp_path: Path
) -> None:
    """SCENARIO-CSL-7022-ISOLATED-REPLAY runs every clean replay control."""

    evidence = exp.run_audit_matrix(
        source_bundle["values"]["events"],
        source_bundle["values"]["exp7020"],
        tmp_path,
        _passing_runtime(),
    )

    for field in exp.SAFETY_ROW_TABLES:
        assert evidence[field]
        assert all(row["terminal"] for row in evidence[field])
    assert all(row["passed"] for field in exp.SAFETY_ROW_TABLES for row in evidence[field])
    assert len(evidence["future_leakage_mutation_rows"]) == 4
    assert len(evidence["game_identity_mutation_rows"]) == 10
    assert evidence["fresh_process_rows"][0]["replay_digest_matches_exp7020"] is True
    assert evidence["capacity_rows"][0]["within_item_capacity"] is True
    assert evidence["capacity_rows"][0]["within_byte_capacity"] is True
    assert evidence["poison_rows"][0]["admitted_poison_count"] == 0
    assert evidence["retention_rows"][0]["protected_case_harm_count"] == 0


def test_scenario_7022_order_serialization_authority_and_retrieval_details(
    source_bundle: dict, tmp_path: Path
) -> None:
    """SCENARIO-CSL-7022-ORDER-AUTHORITY-AND-RETRIEVAL checks exact failure modes."""

    evidence = exp.run_audit_matrix(
        source_bundle["values"]["events"],
        source_bundle["values"]["exp7020"],
        tmp_path,
        _passing_runtime(),
    )

    assert {row["mutation"] for row in evidence["serialization_mutation_rows"]} == {
        "semantic_key_and_whitespace_change",
        "changed_event_bytes_without_row_hash",
    }
    assert {row["mutation"] for row in evidence["order_mutation_rows"]} == {
        "adjacent_event_swap",
        "reverse_event_order",
    }
    assert evidence["authority_conflict_rows"][0]["reason"] == "authority_conflict"
    assert evidence["retrieval_collision_rows"][0]["cross_key_fact_retrieved"] is False


def test_scenario_7022_conflict_capacity_poison_and_durable_controls(
    source_bundle: dict, tmp_path: Path
) -> None:
    """SCENARIO-CSL-7022-CONFLICT-CAPACITY-AND-POISON keeps safety controls exact."""

    evidence = exp.run_audit_matrix(
        source_bundle["values"]["events"],
        source_bundle["values"]["exp7020"],
        tmp_path,
        _passing_runtime(),
    )

    assert evidence["supersession_rows"][0]["generation_increased"] is True
    assert evidence["restart_rows"][0]["committed_bytes_survived"] is True
    interruption = {row["boundary"]: row for row in evidence["interruption_rows"]}
    assert interruption["after_prepare"]["recovery_phase"] == "abort_recovered"
    assert interruption["after_publish"]["recovery_phase"] == "commit_recovered"
    rollback = {row["case"]: row for row in evidence["rollback_rows"]}
    assert rollback["invalid_snapshot"]["rejected"] is True
    assert rollback["exact_parent"]["byte_exact"] is True
    assert rollback["tampered_journal_tail"]["rejected"] is True
    assert rollback["truncated_commit_tail"]["recovery_phase"] == "commit_recovered"


def test_req_7022_durable_bypass_rows_turn_red(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CSL-7022 records failed rows if snapshot or journal rejection is bypassed."""

    original_rollback = ledger_mod.BeliefLedger.rollback
    original_restart = ledger_mod.BeliefLedger.restart

    def bypass_invalid_snapshot(
        self: ledger_mod.BeliefLedger,
        parent_bytes: bytes,
        *,
        transaction_id: str,
        reason: str,
    ) -> dict:
        if parent_bytes == b"{invalid":
            return {"rolled_back": False}
        return original_rollback(
            self,
            parent_bytes,
            transaction_id=transaction_id,
            reason=reason,
        )

    def bypass_tampered_tail(self: ledger_mod.BeliefLedger) -> ledger_mod.BeliefLedger:
        if self.root.name == "tampered-tail":
            return self
        return original_restart(self)

    monkeypatch.setattr(ledger_mod.BeliefLedger, "rollback", bypass_invalid_snapshot)
    monkeypatch.setattr(ledger_mod.BeliefLedger, "restart", bypass_tampered_tail)
    rows = exp._durability_rows(tmp_path)
    rollback = {row["case"]: row for row in rows["rollback_rows"]}
    assert rollback["invalid_snapshot"]["passed"] is False
    assert rollback["tampered_journal_tail"]["passed"] is False


def test_scenario_7022_recomputes_every_exp7021_headline_and_pair(
    source_bundle: dict,
) -> None:
    """SCENARIO-CSL-7022-ROW-RECOMPUTATION derives metrics without producer helpers."""

    value = exp.audit_exp7021_rows(source_bundle["values"]["exp7021"])

    assert value["recomputation_passed"] is True
    assert value["row_positive_gate_passed"] is False
    assert len(value["aggregate_recomputation_rows"]) == 8
    assert all(row["passed"] for row in value["aggregate_recomputation_rows"])
    gates = {row["check"]: row for row in value["promotion_gate_rows"]}
    assert gates["strictly_beats_both_controls"]["passed"] is True
    assert gates["paired_interval_lower_bounds_nonnegative"]["passed"] is False
    assert gates["protected_retention_does_not_regress"]["passed"] is True


def test_scenario_7022_headline_or_paired_drift_disqualifies_value(
    source_bundle: dict,
) -> None:
    """SCENARIO-CSL-7022-ROW-RECOMPUTATION detects both aggregate drift classes."""

    headline = deepcopy(source_bundle["values"]["exp7021"])
    headline["rows"][0]["action_ranking_accuracy"] = 0.75
    headline_audit = exp.audit_exp7021_rows(headline)
    assert headline_audit["recomputation_passed"] is False
    assert any(not row["passed"] for row in headline_audit["aggregate_recomputation_rows"])

    paired = deepcopy(source_bundle["values"]["exp7021"])
    paired["paired_delta_rows"][0]["lower"] = 0.0
    paired_audit = exp.audit_exp7021_rows(paired)
    assert paired_audit["recomputation_passed"] is False
    assert any(
        row["record_type"] == "paired_delta" and not row["passed"]
        for row in paired_audit["aggregate_recomputation_rows"]
    )


def test_scenario_7022_safe_but_nonuseful_is_null(frozen_artifact: dict) -> None:
    """SCENARIO-CSL-7022-SEPARATE-DECISIONS keeps shadow safety separate from value."""

    artifact = frozen_artifact
    assert exp.validate_artifact(artifact) == []
    assert artifact["belief_shadow_safe_score"] == 1
    assert artifact["belief_promotion_ready_score"] == 0
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"].startswith("complete_null_")
    assert artifact["verifier_is_oracle"] is False
    assert set(artifact["field_principles"]) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["source_artifact_hashes"] == exp.EXPECTED_SOURCE_HASHES


def test_req_7022_leakage_or_nonrecomputation_is_disqualified(
    frozen_artifact: dict,
) -> None:
    """REQ-CSL-7022 assigns disqualified to leakage and non-recomputable evidence."""

    leaky = deepcopy(frozen_artifact)
    leaky["future_leakage_mutation_rows"][0]["passed"] = False
    reduced = exp.reduce_terminal_decision(leaky)
    assert reduced["belief_shadow_safe_score"] == 0
    assert reduced["belief_promotion_ready_score"] == 0
    assert reduced["verdict_class"] == "disqualified"

    drift = deepcopy(frozen_artifact)
    drift["aggregate_recomputation_rows"][0]["passed"] = False
    reduced = exp.reduce_terminal_decision(drift)
    assert reduced["belief_shadow_safe_score"] == 1
    assert reduced["belief_promotion_ready_score"] == 0
    assert reduced["verdict_class"] == "disqualified"


def test_req_7022_all_positive_rows_are_required_for_positive() -> None:
    """REQ-CSL-7022 reaches positive only when safety, recomputation, and value pass."""

    artifact = exp._base_artifact(exp.RUN_DATE, 0.1, exp.EXPECTED_SOURCE_HASHES)
    artifact["preconditions_checked"] = [exp.gate_check("preconditions", True, True)]
    for field in exp.SAFETY_ROW_TABLES:
        artifact[field] = [{"passed": True, "terminal": True}]
    artifact["aggregate_recomputation_rows"] = [{"passed": True, "terminal": True}]
    artifact["promotion_gate_rows"] = [{"passed": True, "terminal": True}]
    reduced = exp.reduce_terminal_decision(artifact)
    assert reduced["belief_shadow_safe_score"] == 1
    assert reduced["belief_promotion_ready_score"] == 1
    assert reduced["verdict_class"] == "positive"
    assert reduced["honest_verdict"].startswith("complete_positive_")


def test_scenario_7022_blocked_artifact_names_exact_first_failure() -> None:
    """SCENARIO-CSL-7022-PRECONDITIONS emits the required blocked verdict."""

    checks = [exp.gate_check("exp7020_hash", "sha256:expected", "sha256:observed")]
    artifact = exp.build_blocked_artifact(
        run_date=exp.RUN_DATE,
        duration_s=0.25,
        checks=checks,
        source_hashes={"exp7020": "sha256:observed"},
    )

    assert exp.validate_artifact(artifact) == []
    assert artifact["belief_shadow_safe_score"] == 0
    assert artifact["belief_promotion_ready_score"] == 0
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == "blocked_belief_ledger_cold_audit"
    assert artifact["gate_check_summary"]["failed_check"] == "exp7020_hash"
    assert artifact["gate_check_summary"]["expected_value"] == "sha256:expected"
    assert artifact["gate_check_summary"]["observed_value"] == "sha256:observed"


def test_req_7022_validator_and_atomic_writer_fail_closed(
    frozen_artifact: dict, tmp_path: Path
) -> None:
    """REQ-CSL-7022 validates structure, decisions, and checksums before publication."""

    target = tmp_path / "artifact.json"
    exp.write_validated_artifact(target, frozen_artifact)
    assert json.loads(target.read_text(encoding="utf-8")) == frozen_artifact

    cases = []
    missing = deepcopy(frozen_artifact)
    missing.pop("rows")
    cases.append((missing, "required_fields_missing"))
    principles = deepcopy(frozen_artifact)
    principles["field_principles"] = {}
    cases.append((principles, "field_principles_mismatch"))
    score = deepcopy(frozen_artifact)
    score["belief_shadow_safe_score"] = True
    cases.append((score, "shadow_score_must_be_bare_integer"))
    terminal = deepcopy(frozen_artifact)
    terminal["capacity_rows"][0]["terminal"] = False
    cases.append((terminal, "nonterminal_row:capacity_rows"))
    prefix = deepcopy(frozen_artifact)
    prefix["honest_verdict"] = "wrong"
    cases.append((prefix, "verdict_prefix_mismatch"))
    oracle = deepcopy(frozen_artifact)
    oracle["verifier_is_oracle"] = True
    cases.append((oracle, "verifier_is_oracle_mismatch"))
    preconditions = deepcopy(frozen_artifact)
    preconditions["preconditions_checked"] = "wrong"
    cases.append((preconditions, "preconditions_invalid"))
    substrate = deepcopy(frozen_artifact)
    substrate["inference_substrate"] = "wrong"
    cases.append((substrate, "inference_substrate_mismatch"))
    verdict = deepcopy(frozen_artifact)
    verdict["verdict_class"] = "wrong"
    cases.append((verdict, "verdict_class_invalid"))
    gate = deepcopy(frozen_artifact)
    gate["gate_check_summary"] = {}
    cases.append((gate, "gate_check_summary_invalid"))
    null_score = deepcopy(frozen_artifact)
    null_score["belief_shadow_safe_score"] = 0
    cases.append((null_score, "null_score_mismatch"))
    positive_score = deepcopy(frozen_artifact)
    positive_score["verdict_class"] = "positive"
    positive_score["honest_verdict"] = "complete_positive_test"
    cases.append((positive_score, "positive_score_mismatch"))
    disqualified_score = deepcopy(frozen_artifact)
    disqualified_score["verdict_class"] = "disqualified"
    disqualified_score["honest_verdict"] = "disqualified_test"
    disqualified_score["belief_promotion_ready_score"] = 1
    cases.append((disqualified_score, "disqualified_promotion_mismatch"))

    assert exp.validate_artifact([]) == ["artifact_object_required"]

    for artifact, expected in cases:
        assert expected in exp.validate_artifact(artifact)
    with pytest.raises(ValueError, match="required_fields_missing"):
        exp.write_validated_artifact(tmp_path / "invalid.json", missing)

    blocked = exp.build_blocked_artifact(
        run_date=exp.RUN_DATE,
        duration_s=0.1,
        checks=[exp.gate_check("blocked", True, False)],
        source_hashes={},
    )
    blocked["honest_verdict"] = "blocked_wrong"
    blocked["belief_shadow_safe_score"] = 1
    blocked["gate_check_summary"] = exp.gate_summary([])
    blocked_errors = exp.validate_artifact(blocked)
    assert "blocked_verdict_mismatch" in blocked_errors
    assert "blocked_score_mismatch" in blocked_errors
    assert "blocked_without_failed_gate" in blocked_errors


def test_req_7022_atomic_writer_cleans_failed_temporary(
    frozen_artifact: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CSL-7022 removes a temporary document when publication fails."""

    monkeypatch.setattr(exp.os, "replace", lambda *_args: (_ for _ in ()).throw(OSError()))
    with pytest.raises(OSError):
        exp.write_validated_artifact(tmp_path / "failed.json", frozen_artifact)
    assert not list(tmp_path.glob(".failed.json.*"))


def test_req_7022_build_from_repo_blocked_write_and_success_write(
    tmp_path: Path,
) -> None:
    """REQ-CSL-7022 publishes both blocked and complete child outcomes."""

    blocked_path = tmp_path / "blocked.json"
    blocked = exp.build_from_repo(
        REPO_ROOT,
        output_path=blocked_path,
        work_root=tmp_path,
        runtime_receipt={"passed": False},
        write_output=True,
    )
    assert blocked["verdict_class"] == "blocked"
    assert blocked_path.is_file()

    success_path = tmp_path / "success.json"
    success = exp.build_from_repo(
        REPO_ROOT,
        output_path=success_path,
        work_root=tmp_path,
        runtime_receipt=_passing_runtime(),
        write_output=True,
    )
    assert success["verdict_class"] == "null"
    assert success_path.is_file()


def test_req_7022_build_rejects_internal_validation_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CSL-7022 refuses to publish an internally invalid complete result."""

    monkeypatch.setattr(exp, "validate_artifact", lambda _artifact: ["forced invalid"])
    with pytest.raises(RuntimeError, match="forced invalid"):
        exp.build_from_repo(
            REPO_ROOT,
            output_path=tmp_path / "invalid.json",
            work_root=tmp_path,
            runtime_receipt=_passing_runtime(),
            write_output=False,
        )


def test_req_7022_sandbox_command_and_controller_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CSL-7022 requires bubblewrap, hidden CUDA, and one writable child path."""

    command = exp.fresh_process_command(
        executable=Path("/python"),
        wrapper=Path("/wrapper"),
        repo_root=REPO_ROOT,
        writable_root=tmp_path,
        output_path=tmp_path / "child.json",
        run_date=exp.RUN_DATE,
    )
    assert command[0] == "bwrap"
    assert "--unshare-net" in command
    assert command[command.index("CUDA_VISIBLE_DEVICES") + 1] == ""
    assert command[command.index("HF_HUB_OFFLINE") + 1] == "1"

    monkeypatch.setattr(exp.shutil, "which", lambda _name: None)
    output = tmp_path / "blocked.json"
    artifact = exp.run_controller(repo_root=REPO_ROOT, result_path=output)
    assert artifact["honest_verdict"] == "blocked_belief_ledger_cold_audit"
    assert artifact["gate_check_summary"]["failed_check"] == "bubblewrap_available"
    assert json.loads(output.read_text(encoding="utf-8")) == artifact


def test_req_7022_controller_accepts_valid_child(
    frozen_artifact: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CSL-7022 copies one validated child result after source hashes stay fixed."""

    monkeypatch.setattr(exp.shutil, "which", lambda _name: "/usr/bin/bwrap")

    def completed(command: list[str], **_kwargs: object) -> SimpleNamespace:
        child_path = Path(command[command.index("--output") + 1])
        child_path.write_text(json.dumps(frozen_artifact), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(exp.subprocess, "run", completed)
    output = tmp_path / "controller.json"
    artifact = exp.run_controller(repo_root=REPO_ROOT, result_path=output)
    assert artifact["belief_shadow_safe_score"] == 1
    assert artifact["verdict_class"] == "null"
    assert exp.validate_artifact(artifact) == []
    assert json.loads(output.read_text(encoding="utf-8")) == artifact


def test_req_7022_runtime_receipt_fails_in_parent(tmp_path: Path) -> None:
    """REQ-CSL-7022 measures isolation instead of trusting environment labels."""

    receipt = exp.sandbox_runtime_receipt(tmp_path)
    assert receipt["fresh_process"] is False
    assert receipt["source_tree_read_only"] is False
    assert receipt["passed"] is False


def test_req_7022_runtime_receipt_records_read_only_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CSL-7022 records a denied source-tree write as measured evidence."""

    original_write_text = Path.write_text

    def deny_probe(self: Path, *args: object, **kwargs: object) -> int:
        if self.name == ".exp7022-write-probe":
            raise OSError("read only")
        return original_write_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", deny_probe)
    receipt = exp.sandbox_runtime_receipt(tmp_path)
    assert receipt["source_tree_read_only"] is True


def test_req_7022_controller_child_failure_and_validation_failure(
    frozen_artifact: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CSL-7022 blocks a missing child result and rejects an invalid copied result."""

    monkeypatch.setattr(exp.shutil, "which", lambda _name: "/usr/bin/bwrap")
    monkeypatch.setattr(
        exp.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=7, stdout="", stderr="failed"),
    )
    blocked = exp.run_controller(repo_root=REPO_ROOT, result_path=tmp_path / "blocked-child.json")
    assert blocked["verdict_class"] == "blocked"
    assert blocked["gate_check_summary"]["failed_check"] == "fresh_process_exit_code"

    def completed(command: list[str], **_kwargs: object) -> SimpleNamespace:
        child_path = Path(command[command.index("--output") + 1])
        child_path.write_text(json.dumps(frozen_artifact), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(exp.subprocess, "run", completed)
    monkeypatch.setattr(exp, "validate_artifact", lambda _artifact: ["forced invalid"])
    with pytest.raises(RuntimeError, match="forced invalid"):
        exp.run_controller(repo_root=REPO_ROOT, result_path=tmp_path / "invalid-child.json")
