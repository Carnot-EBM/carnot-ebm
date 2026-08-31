"""Tests for the independent byte-grounded CSL cold audit.

Spec refs: REQ-CL-6798 and SCENARIO-CL-6798-*.
"""

from __future__ import annotations

import base64
from copy import deepcopy
import inspect
import json
from pathlib import Path
import runpy
import sys

import pytest

from carnot import experiment_6798_csl_causal_safety_byte_audit as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / exp.SPEC_RELATIVE_PATH
SOURCE_PATHS = {name: REPO_ROOT / relative for name, relative in exp.SOURCE_RELATIVE_PATHS.items()}


def _state(records: list[dict] | None = None, *, version: int = 0) -> dict:
    """Build a small canonical store for unit-level replay checks."""

    return {
        "arm": "compositional_online",
        "order_id": "order_1",
        "records": records or [],
        "schema": "carnot.experiment_6791.isolated_transaction_store.v1",
        "version": version,
    }


def _factor(
    factor_id: str = "factor:order_1:event-1",
    *,
    route: str = "dependency_suffix",
) -> dict:
    """Build one exact prior factor with the fields used by the cold replay."""

    return {
        "evidence_hash": "sha256:receipt-1",
        "exact_provenance": True,
        "factor_id": factor_id,
        "factor_type": "dependency_route_factor",
        "motif_id": "motif:easy:balanced_odd:groups_1",
        "placebo_shuffled": False,
        "source_event_id": "event-1",
        "source_position": 1,
        "source_topology_family": "directed_implication_chain",
        "stratum": "easy",
        "target_route": route,
        "update_target_event_id": "event-1",
        "update_target_position": 1,
    }


def _event() -> dict:
    """Build one exact-receipt event whose suffix route finds the failure."""

    return {
        "available_actions": list(exp.TIE_BREAK_ORDER),
        "difficulty": "easy",
        "event_id": "event-2",
        "exact_failed_dependency_ids": ["d03"],
        "exact_failed_factor_ids": ["dependency:d03"],
        "exact_receipt": {"exact_valid": False},
        "held_future": False,
        "legal_observation": {
            "reusable_motif_id": "motif:easy:balanced_odd:groups_1",
            "topology_family": "directed_implication_chain",
        },
        "poison_status": "none",
    }


@pytest.fixture(scope="session")
def sources() -> dict[str, dict]:
    """Load the checked-in evidence once because the audit does not mutate it."""

    return {name: exp.read_json_object(path) for name, path in SOURCE_PATHS.items()}


@pytest.fixture(scope="session")
def full_artifact(sources: dict[str, dict]) -> dict:
    """Run the complete checked-in replay once for all terminal assertions."""

    return exp.build_artifact(
        sources=sources,
        source_paths=SOURCE_PATHS,
        run_date=exp.RUN_DATE,
        duration_s=1.0,
    )


def test_req_cl_6798_spec_owns_the_byte_audit_contract() -> None:
    """REQ-CL-6798 anchors implementation and every required output field."""

    section = SPEC_PATH.read_text(encoding="utf-8").split("## REQ-CL-6798", 1)[1]
    for marker in (
        "SCENARIO-CL-6798-PRECONDITIONS",
        "SCENARIO-CL-6798-SERIALIZER",
        "SCENARIO-CL-6798-CHAIN-REPLAY",
        "SCENARIO-CL-6798-CAUSAL-CREDIT",
        "SCENARIO-CL-6798-ATTACKS",
        "SCENARIO-CL-6798-RESTART-ROLLBACK",
        "SCENARIO-CL-6798-TERMINAL",
        exp.INFERENCE_SUBSTRATE,
        exp.MODULE_RELATIVE_PATH.as_posix(),
        exp.SCRIPT_RELATIVE_PATH.as_posix(),
        exp.RESULT_RELATIVE_PATH.as_posix(),
    ):
        assert marker in section
    for field in exp.REQUIRED_AUDIT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_cl_6798_serializer_round_trips_and_rejects_corruption() -> None:
    """SCENARIO-CL-6798-SERIALIZER owns bytes without producer helpers."""

    state = _state([_factor()], version=1)
    raw = exp.canonical_json_bytes(state)
    encoded = base64.b64encode(raw).decode("ascii")

    assert raw.endswith(b"\n")
    assert exp.decode_snapshot(encoded) == raw
    assert exp.parse_state(raw) == state
    assert exp.canonical_json_bytes(exp.parse_state(raw)) == raw
    assert exp.sha256_bytes(raw).startswith("sha256:")
    with pytest.raises(ValueError, match="base64"):
        exp.decode_snapshot("not base64")
    with pytest.raises(ValueError, match="canonical"):
        exp.parse_state(b'{"version":0}')


def test_scenario_cl_6798_chain_replay_extends_parent_bytes() -> None:
    """SCENARIO-CL-6798-CHAIN-REPLAY checks state ownership and extension."""

    parent = exp.canonical_json_bytes(_state())
    new = exp.canonical_json_bytes(_state([_factor()], version=1))
    receipt = {
        "arm": "compositional_online",
        "chain_index": 1,
        "chain_predecessor": "sha256:genesis",
        "committed": True,
        "event_id": "event-1",
        "factor_id": "factor:order_1:event-1",
        "new_state_bytes": base64.b64encode(new).decode("ascii"),
        "new_state_hash": exp.sha256_bytes(new),
        "order_id": "order_1",
        "parent_hash": exp.sha256_bytes(parent),
        "parent_state_bytes": base64.b64encode(parent).decode("ascii"),
        "position": 1,
        "receipt_hash": "sha256:receipt-1",
        "transaction_id": "tx:order_1:event-1:compositional_online",
    }

    replay = exp.verify_transaction_chains([receipt])
    assert replay["all_passed"] is True
    assert replay["parent_byte_count"] == 1
    assert replay["new_state_byte_count"] == 1
    assert replay["byte_hash_match_count"] == 1
    changed = deepcopy(receipt)
    changed["parent_hash"] = "sha256:wrong"
    assert exp.verify_transaction_chains([changed])["all_passed"] is False


def test_scenario_cl_6798_chain_replay_derives_action_and_exact_utility() -> None:
    """SCENARIO-CL-6798-CHAIN-REPLAY uses state bytes and exact receipts."""

    state = _state([_factor()], version=1)
    event = _event()
    selected = exp.select_route(state, event, "dependency_prefix", retrieval_enabled=True)
    no_retrieval = exp.select_route(state, event, "dependency_prefix", retrieval_enabled=False)
    selected_result = exp.evaluate_route(event, selected["selected_action"])
    baseline_result = exp.evaluate_route(event, no_retrieval["selected_action"])

    assert selected["retrieved_factor_ids"] == ["factor:order_1:event-1"]
    assert selected["selected_action"] == "dependency_suffix"
    assert no_retrieval["selected_action"] == "dependency_prefix"
    assert selected_result["utility"] == pytest.approx(0.85)
    assert baseline_result["utility"] == pytest.approx(-0.4)


def test_scenario_cl_6798_causal_credit_needs_action_and_utility() -> None:
    """SCENARIO-CL-6798-CAUSAL-CREDIT requires both exact witnesses."""

    state = _state([_factor()], version=1)
    witness = exp.replay_factor_ablation(
        state,
        _event(),
        "dependency_prefix",
        "factor:order_1:event-1",
    )
    neutral_state = _state([_factor(route="dependency_prefix")], version=1)
    neutral = exp.replay_factor_ablation(
        neutral_state,
        _event(),
        "dependency_prefix",
        "factor:order_1:event-1",
    )

    assert witness["credited"] is True
    assert witness["action_changed"] is True
    assert witness["utility_difference"] == pytest.approx(1.25)
    assert witness["retrieval_disabled_action"] == "dependency_prefix"
    assert neutral["credited"] is False


def test_scenario_cl_6798_preconditions_bind_the_exact_snapshot_fixture(
    sources: dict[str, dict],
) -> None:
    """SCENARIO-CL-6798-PRECONDITIONS checks every fixed source count."""

    summary = exp.evaluate_preconditions(sources, SOURCE_PATHS)
    assert summary["all_passed"] is True
    assert summary["failed_checks"] == []
    observed = summary["observed_counts"]
    assert observed == {
        "unique_rows": 4800,
        "order_hashes": 5,
        "committed_transactions": 3189,
        "parent_snapshots": 3189,
        "new_state_snapshots": 3189,
        "matching_byte_hashes": 3189,
    }

    changed = deepcopy(sources)
    changed["experiment_6797"]["transaction_byte_snapshot_fixture_ready"] = False
    changed["experiment_6797"]["rows"] = changed["experiment_6797"]["rows"][:-1]
    changed["experiment_6797"]["order_hashes"].pop("order_5")
    changed["experiment_6797"]["committed_transaction_count"] = 3188
    changed["experiment_6797"]["parent_byte_snapshot_count"] = 3188
    changed["experiment_6797"]["new_state_byte_snapshot_count"] = 3188
    changed["experiment_6797"]["byte_hash_match_count"] = 3188
    summary = exp.evaluate_preconditions(
        changed,
        SOURCE_PATHS,
        hash_overrides={"experiment_6797": exp.EXPECTED_SOURCE_HASHES["experiment_6797"]},
    )
    assert summary["all_passed"] is False
    assert set(summary["failed_checks"]) == {
        "transaction_byte_snapshot_fixture_ready",
        "unique_rows",
        "five_order_hashes",
        "committed_transaction_count",
        "parent_byte_snapshot_count",
        "new_state_byte_snapshot_count",
        "byte_hash_match_count",
    }


def test_scenario_cl_6798_attacks_reject_invalid_state_and_restore_capacity() -> None:
    """SCENARIO-CL-6798-ATTACKS keeps invalid factors outside active state."""

    state = _state([_factor()], version=1)
    result = exp.run_attack_suite(state, _event(), "dependency_prefix")

    assert [row["attack_id"] for row in result["attack_results"]] == list(exp.ATTACK_IDS)
    assert all(row["invalid_admitted"] is False for row in result["attack_results"])
    assert all(row["invalid_influenced"] is False for row in result["attack_results"])
    assert result["admitted_poison_count"] == 0
    assert result["influenced_poison_count"] == 0
    assert result["restart_byte_identity"] is True
    assert result["restart_action_identity"] is True
    assert result["rollback_byte_identity"] is True
    assert result["rollback_action_identity"] is True
    assert result["capacity_eviction_receipts"]


def test_source_is_read_without_importing_either_producer() -> None:
    """REQ-CL-6798 keeps both producer modules outside the cold process."""

    source = inspect.getsource(exp)
    for forbidden in (
        "experiment_6797_canonical_transaction_byte_replay import",
        "from carnot import experiment_6797",
        "experiment_6791_compositional_online_constraint_routing_ab import",
        "from carnot import experiment_6791",
    ):
        assert forbidden not in source


def test_scenario_cl_6798_terminal_full_replay_is_positive_and_complete(
    full_artifact: dict,
) -> None:
    """SCENARIO-CL-6798-TERMINAL credits only fully witnessed learning."""

    artifact = full_artifact
    assert set(artifact) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert set(artifact["field_principles"]) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["transaction_byte_counts"] == {
        "committed_transactions": 3189,
        "parent_snapshots": 3189,
        "new_state_snapshots": 3189,
        "matching_byte_hashes": 3189,
    }
    assert len(artifact["chain_replay_receipts"]) == 15
    assert artifact["cold_recomputed_metrics"]["row_count"] == 4800
    assert artifact["cold_recomputed_metrics"]["all_actions_replayed"] is True
    assert artifact["cold_recomputed_metrics"]["all_utilities_replayed"] is True
    assert artifact["cold_recomputed_metrics"]["uncredited_write_count"] >= 0
    assert artifact["credited_factor_count"] == len(artifact["factors_with_changed_action_witness"])
    assert artifact["credited_factor_count"] > 0
    assert artifact["admitted_poison_count"] == 0
    assert artifact["influenced_poison_count"] == 0
    assert artifact["restart_byte_identity"] is True
    assert artifact["restart_action_identity"] is True
    assert artifact["rollback_byte_identity"] is True
    assert artifact["rollback_action_identity"] is True
    assert artifact["source_verdict_supported"] is True
    assert artifact["csl_causal_audit_completed"] is True
    assert artifact["gate_check_summary"]["all_passed"] is True
    assert artifact["verifier_is_oracle"] is False
    assert artifact["verdict_class"] == "positive"
    assert artifact["honest_verdict"].startswith("complete_positive:")
    assert len(artifact["rows"]) == 4800 + len(exp.ATTACK_IDS)
    assert all(value["matches"] for value in artifact["headline_differences"].values())
    assert exp.validate_artifact(artifact) == []


def test_scenario_cl_6798_preconditions_emit_complete_blocked_artifact(
    sources: dict[str, dict],
) -> None:
    """SCENARIO-CL-6798-PRECONDITIONS never shrinks a failed audit."""

    changed = deepcopy(sources)
    changed["experiment_6797"]["transaction_byte_snapshot_fixture_ready"] = False
    artifact = exp.build_artifact(
        sources=changed,
        source_paths=SOURCE_PATHS,
        hash_overrides={"experiment_6797": exp.EXPECTED_SOURCE_HASHES["experiment_6797"]},
        duration_s=0.1,
    )

    assert artifact["status"].startswith("complete_blocked_")
    assert artifact["verdict_class"] == "blocked"
    assert artifact["rows"] == []
    assert artifact["chain_replay_receipts"] == []
    assert artifact["csl_causal_audit_completed"] is False
    assert artifact["gate_check_summary"]["failed_checks"] == [
        "transaction_byte_snapshot_fixture_ready"
    ]
    assert exp.validate_artifact(artifact) == []


def test_validator_writer_and_cli_fail_closed(
    full_artifact: dict, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-CL-6798 validates the terminal schema before an atomic write."""

    target = tmp_path / "artifact.json"
    receipt = exp.write_artifact(target, full_artifact)
    assert receipt["atomic_rename"] is True
    assert receipt["sha256"] == exp.sha256_file(target)
    assert exp.main(["--validate", "--output", str(target)]) == 0
    assert "complete_positive:" in capsys.readouterr().out

    invalid = deepcopy(full_artifact)
    invalid["verdict_class"] = "unknown"
    invalid["reproducibility_checksum"] = exp.reproducibility_checksum(invalid)
    assert "verdict class is outside the closed enum" in exp.validate_artifact(invalid)
    with pytest.raises(ValueError, match="verdict class"):
        exp.write_artifact(tmp_path / "invalid.json", invalid)

    malformed = tmp_path / "malformed.json"
    malformed.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object required"):
        exp.read_json_object(malformed)
    with pytest.raises(ValueError, match="YYYYMMDD"):
        exp.build_artifact(sources={}, source_paths={}, run_date="2026-08-31", duration_s=0.1)


def test_reproducibility_checksum_ignores_only_measured_duration(
    full_artifact: dict,
) -> None:
    """REQ-CL-6798 binds stable evidence but permits wall-clock variation."""

    changed = deepcopy(full_artifact)
    changed["duration_s"] = 99.0
    assert exp.reproducibility_checksum(changed) == full_artifact["reproducibility_checksum"]
    changed["credited_factor_count"] += 1
    assert exp.reproducibility_checksum(changed) != full_artifact["reproducibility_checksum"]


def test_req_cl_6798_rejects_malformed_snapshots_and_chain_receipts(
    tmp_path: Path,
) -> None:
    """REQ-CL-6798 fail-closes every malformed byte and chain boundary."""

    assert exp.sha256_file(tmp_path / "missing.json") is None
    for raw in (b"\xff", b"{"):
        with pytest.raises(ValueError, match="canonical JSON"):
            exp.parse_state(raw)
    for malformed in (
        {"arm": "a", "order_id": "o", "records": {}, "schema": "s", "version": 0},
        {"arm": "a", "order_id": "o", "records": [], "schema": "s", "version": "0"},
    ):
        with pytest.raises(ValueError, match="canonical store"):
            exp.parse_state(exp.canonical_json_bytes(malformed))
    with pytest.raises(ValueError, match="state bytes are not canonical"):
        exp.parse_state(b'{"arm":"a", "order_id":"o","records":[],"schema":"s","version":0}\n')

    valid = exp.canonical_json_bytes(_state())
    observed = exp._observe_snapshot_bytes(
        [
            {"committed": False},
            {
                "committed": True,
                "parent_state_bytes": base64.b64encode(valid).decode("ascii"),
            },
            {
                "committed": True,
                "new_state_bytes": base64.b64encode(valid).decode("ascii"),
            },
            {
                "committed": True,
                "parent_state_bytes": "not-base64",
                "new_state_bytes": base64.b64encode(valid).decode("ascii"),
            },
        ]
    )
    assert observed == {
        "committed_transactions": 3,
        "parent_snapshots": 2,
        "new_state_snapshots": 2,
        "matching_byte_hashes": 0,
    }

    def receipt(
        parent: dict,
        new: dict,
        *,
        arm: str = "compositional_online",
        order_id: str = "order_1",
        chain_index: int = 1,
        predecessor: str = "sha256:genesis",
        factor_id: str = "factor:order_1:event-1",
    ) -> dict:
        parent_raw = exp.canonical_json_bytes(parent)
        new_raw = exp.canonical_json_bytes(new)
        return {
            "arm": arm,
            "chain_index": chain_index,
            "chain_predecessor": predecessor,
            "committed": True,
            "event_id": f"event-{chain_index}",
            "factor_id": factor_id,
            "new_state_bytes": base64.b64encode(new_raw).decode("ascii"),
            "new_state_hash": exp.sha256_bytes(new_raw),
            "order_id": order_id,
            "parent_hash": exp.sha256_bytes(parent_raw),
            "parent_state_bytes": base64.b64encode(parent_raw).decode("ascii"),
            "position": chain_index,
            "receipt_hash": f"sha256:receipt-{chain_index}",
            "transaction_id": f"tx:{order_id}:{chain_index}",
        }

    parent = _state()
    new = _state([_factor()], version=1)
    cases: list[tuple[dict, str]] = []
    missing = receipt(parent, new)
    missing.pop("factor_id")
    cases.append((missing, "missing_required_fields"))
    malformed_bytes = receipt(parent, new)
    malformed_bytes["parent_state_bytes"] = "not-base64"
    cases.append((malformed_bytes, "strict base64"))
    wrong_hash = receipt(parent, new)
    wrong_hash["new_state_hash"] = "sha256:wrong"
    cases.append((wrong_hash, "snapshot_hash_mismatch"))
    wrong_index = receipt(parent, new, chain_index=2)
    cases.append((wrong_index, "chain_index_mismatch"))
    wrong_arm = receipt(parent, new, arm="wrong_arm")
    cases.append((wrong_arm, "arm_owner_mismatch"))
    wrong_order = receipt(parent, new, order_id="wrong_order")
    cases.append((wrong_order, "order_owner_mismatch"))
    wrong_version = receipt(parent, {**new, "version": 3})
    cases.append((wrong_version, "version_not_incremented"))
    no_append = receipt(parent, {**parent, "version": 1})
    cases.append((no_append, "record_append_mismatch"))
    wrong_factor = receipt(parent, new, factor_id="factor:wrong")
    cases.append((wrong_factor, "factor_identity_mismatch"))
    for attacked, expected in cases:
        assert expected in exp.verify_transaction_chains([attacked])["errors"][0]

    first = receipt(parent, new)
    second_parent = _state()
    second_new = _state([_factor("factor:order_1:event-2")], version=1)
    second = receipt(
        second_parent,
        second_new,
        chain_index=2,
        predecessor="sha256:wrong",
        factor_id="factor:order_1:event-2",
    )
    linked = exp.verify_transaction_chains([first, second])
    assert any("parent_does_not_extend_prior_new" in error for error in linked["errors"])
    assert any("predecessor_mismatch" in error for error in linked["errors"])

    with pytest.raises(ValueError, match="unknown live route"):
        exp.evaluate_route(_event(), "not-a-route")


def test_req_cl_6798_large_json_loader_streams_transaction_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CL-6798 streams large receipt bytes and preserves mapping behavior."""

    monkeypatch.setattr(exp, "LARGE_JSON_THRESHOLD_BYTES", 0)
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "schema": "test",
                "transaction_receipts": [
                    {"transaction_id": "tx-1", "committed": False},
                    {"transaction_id": "tx-2", "committed": True},
                ],
            }
        ),
        encoding="utf-8",
    )
    loaded = exp.read_json_object(source)
    receipts = loaded["transaction_receipts"]
    assert len(receipts) == 2
    assert receipts[0]["transaction_id"] == "tx-1"
    assert receipts[-1]["transaction_id"] == "tx-2"
    assert [row["transaction_id"] for row in receipts[:]] == ["tx-1", "tx-2"]
    assert receipts.transaction("tx-2")["committed"] is True
    assert deepcopy(receipts) is receipts
    with pytest.raises(IndexError):
        _ = receipts[2]
    with pytest.raises(IndexError):
        _ = receipts[-3]
    with pytest.raises(KeyError, match="missing"):
        receipts.transaction("missing")

    no_array = tmp_path / "no-array.json"
    no_array.write_text(json.dumps({"transaction_receipts": {}}), encoding="utf-8")
    assert exp.read_json_object(no_array)["transaction_receipts"] == {}

    wrong_root = tmp_path / "wrong-root.json"
    wrong_root.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object required"):
        exp.read_json_object(wrong_root)

    owner = exp.tempfile.TemporaryDirectory(prefix="carnot-6798-test-")
    invalid_rows = Path(owner.name) / "invalid.jsonl"
    invalid_rows.write_text("1\n", encoding="utf-8")
    invalid = exp._LazyJsonArray(invalid_rows, 1, {"tx": [0, 2]}, owner)
    with pytest.raises(ValueError, match="object rows"):
        list(invalid)
    with pytest.raises(ValueError, match="transaction row"):
        invalid.transaction("tx")
    short_rows = Path(owner.name) / "short.jsonl"
    short_rows.write_text("{}\n", encoding="utf-8")
    short = exp._LazyJsonArray(short_rows, 2, {}, owner)
    with pytest.raises(IndexError):
        _ = short[1]

    real_json_load = exp.json.load

    def invalid_index(handle: object) -> object:
        if Path(str(getattr(handle, "name", ""))).name == "transaction_index.json":
            return []
        return real_json_load(handle)

    monkeypatch.setattr(exp.json, "load", invalid_index)
    with pytest.raises(ValueError, match="transaction index"):
        exp.read_json_object(source)


def test_scenario_cl_6798_replay_and_restart_diagnostics_cover_bad_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-CL-6798-CHAIN-REPLAY keeps all row diagnostics visible."""

    monkeypatch.setattr(exp, "EXPECTED_ORDER_HASHES", {"order_1": "sha256:order"})
    monkeypatch.setattr(exp, "ARMS", ("frozen_controller",))
    monkeypatch.setattr(exp, "RETRIEVAL_ARMS", set())
    event = {
        **_event(),
        "event_id": "event-1",
        "all_factor_ids": [
            "local:g00",
            "local:g01",
            "local:g02",
            "local:g03",
            "local:g04",
            "dependency:d00",
            "dependency:d01",
            "dependency:d02",
            "dependency:d03",
        ],
    }
    rejected = {"transaction_id": "tx:rejected", "committed": False}
    new_raw = exp.canonical_json_bytes(exp._empty_state("frozen_controller", "order_1"))
    malformed_commit = {
        "transaction_id": "tx:malformed",
        "committed": True,
        "parent_state_bytes": "not-base64",
        "new_state_bytes": base64.b64encode(new_raw).decode("ascii"),
    }
    rows = []
    for position in range(240):
        transaction = malformed_commit if position == 239 else rejected
        rows.append(
            {
                "row_key": f"row-{position}",
                "order_id": "order_1",
                "event_id": "event-1",
                "arm": "frozen_controller",
                "position": position,
                "difficulty": "easy",
                "held_future": False,
                "retention_partition": False,
                "baseline_action": "local_prefix",
                "selected_action": "local_suffix",
                "route_utility": 99.0,
                "hidden_receipt_hash": "sha256:row",
                "revealed_post_action_receipt": {"source_receipt_hash": "sha256:revealed"},
                "transaction": {
                    "transaction_id": transaction["transaction_id"],
                    "committed": transaction["committed"],
                },
            }
        )
    replay_source = {
        "rows": rows,
        "transaction_receipts": [rejected, malformed_commit],
    }
    routing_source = {
        "frozen_manifest": {"events": [event]},
        "rows": [
            {
                "order_id": "order_1",
                "event_id": "event-1",
                "hidden_receipt_hash": "sha256:source",
            }
        ],
    }
    replay = exp.replay_all_rows(replay_source, routing_source)
    assert len(replay["action_errors"]) == 240
    assert len(replay["utility_errors"]) == 240
    assert len(replay["receipt_errors"]) == 240
    assert replay["attack_context"]["baseline_action"] == "local_prefix"
    assert replay["rows"][-1]["transaction_parent_identity"] is False

    no_baseline = exp.select_route(
        exp._empty_state("frozen_controller", "order_1"),
        {"available_actions": ["local_prefix"], "difficulty": "easy"},
        "not-available",
        retrieval_enabled=False,
        include_broad=False,
    )
    assert no_baseline["selected_action"] == "local_prefix"

    monkeypatch.setattr(exp, "RESTART_CHAIN_INDICES", (1, 2))
    receipt_row = {
        "committed": True,
        "order_id": "order_1",
        "arm": "frozen_controller",
        "chain_index": 1,
        "position": 239,
        "new_state_bytes": base64.b64encode(new_raw).decode("ascii"),
        "new_state_hash": exp.sha256_bytes(new_raw),
    }
    chain_receipt = {"order_id": "order_1", "arm": "frozen_controller"}
    restart = exp._restart_checks(
        {"rows": [], "transaction_receipts": [receipt_row]},
        {"frozen_manifest": {"events": []}, "rows": []},
        [chain_receipt],
    )
    assert restart == {"byte_identity": True, "action_identity": True}
    assert chain_receipt["restart_boundaries"][0]["replay_action"] is None


def test_scenario_cl_6798_build_classifies_null_disqualified_and_internal_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-CL-6798-TERMINAL distinguishes null evidence from broken gates."""

    sources = {
        "experiment_6797": {"transaction_receipts": []},
        "experiment_6790": {},
        "experiment_6791": {},
    }
    preconditions = {"all_passed": True, "checks": [], "failed_checks": [], "failures": []}
    chain = {
        "all_passed": True,
        "errors": [],
        "parent_byte_count": 0,
        "new_state_byte_count": 0,
        "byte_hash_match_count": 0,
        "chain_receipts": [],
    }
    replay = {
        "rows": [],
        "credited_factors": [],
        "retrieval_disable_effects": [],
        "action_errors": [],
        "utility_errors": [],
        "receipt_errors": [],
        "attack_context": {
            "state": _state(),
            "event": _event(),
            "baseline_action": "local_prefix",
        },
    }
    metrics = {field: {} for field in exp.SOURCE_HEADLINE_FIELDS}
    metrics["writes_by_arm_order"] = {"compositional_online": {"order_1": 0}}
    attacks = {
        "attack_results": [],
        "admitted_poison_count": 0,
        "influenced_poison_count": 0,
        "capacity_eviction_receipts": [],
        "restart_byte_identity": True,
        "restart_action_identity": True,
        "rollback_byte_identity": True,
        "rollback_action_identity": True,
        "retention_after_phase": {},
        "hard_case_harm_after_phase": {},
        "rollback_triggered": True,
    }
    monkeypatch.setattr(exp, "evaluate_preconditions", lambda *args, **kwargs: preconditions)
    monkeypatch.setattr(exp, "_source_hashes", lambda *args, **kwargs: {})
    monkeypatch.setattr(exp, "verify_transaction_chains", lambda receipts: chain)
    monkeypatch.setattr(exp, "replay_all_rows", lambda *args: replay)
    monkeypatch.setattr(exp, "_reduce_metrics", lambda rows: deepcopy(metrics))
    monkeypatch.setattr(
        exp,
        "_headline_differences",
        lambda *args: {field: {"matches": True} for field in exp.SOURCE_HEADLINE_FIELDS},
    )
    monkeypatch.setattr(
        exp,
        "_restart_checks",
        lambda *args: {"byte_identity": True, "action_identity": True},
    )
    monkeypatch.setattr(exp, "run_attack_suite", lambda *args: attacks)

    null_artifact = exp.build_artifact(
        sources=sources,
        source_paths={},
        run_date=exp.RUN_DATE,
        duration_s=0.1,
    )
    assert null_artifact["verdict_class"] == "null"
    assert null_artifact["gate_check_summary"]["all_passed"] is True

    replay["credited_factors"] = [
        {
            "credited": True,
            "action_changed": True,
            "utility_difference": 1.0,
            "same_parent_bytes": True,
        }
    ]
    chain["all_passed"] = False
    chain["errors"] = ["forced_chain_error"]
    disqualified = exp.build_artifact(
        sources=sources,
        source_paths={},
        run_date=exp.RUN_DATE,
        duration_s=0.1,
    )
    assert disqualified["verdict_class"] == "disqualified"
    assert disqualified["honest_verdict"].startswith("complete_disqualified:")

    monkeypatch.setattr(exp, "validate_artifact", lambda artifact: ["forced validation error"])
    with pytest.raises(ValueError, match="forced validation error"):
        exp.build_artifact(
            sources=sources,
            source_paths={},
            run_date=exp.RUN_DATE,
            duration_s=0.1,
        )

    blocked = {
        "all_passed": False,
        "checks": [],
        "failed_checks": ["forced"],
        "failures": [{"check": "forced", "expected": True, "observed": False}],
    }
    monkeypatch.setattr(exp, "evaluate_preconditions", lambda *args, **kwargs: blocked)
    with pytest.raises(ValueError, match="forced validation error"):
        exp.build_artifact(
            sources=sources,
            source_paths={},
            run_date=exp.RUN_DATE,
            duration_s=0.1,
        )


def test_req_cl_6798_validator_and_cli_cover_every_fail_closed_branch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CL-6798 reports every schema, completion, and CLI validation failure."""

    artifact = exp.read_json_object(REPO_ROOT / exp.RESULT_RELATIVE_PATH)
    malformed = deepcopy(artifact)
    malformed.pop("schema")
    malformed["field_principles"] = {}
    malformed["inference_substrate"] = "wrong"
    malformed["random_seed"] = -1
    malformed["verifier_is_oracle"] = True
    malformed["verdict_class"] = "unknown"
    malformed["honest_verdict"] = "not terminal"
    errors = exp.validate_artifact(malformed)
    assert {
        "required field set mismatch",
        "field principle coverage mismatch",
        "inference substrate mismatch",
        "random seed mismatch",
        "verifier_is_oracle must be false",
        "verdict class is outside the closed enum",
        "honest verdict lacks a terminal prefix",
        "reproducibility checksum mismatch",
    } <= set(errors)

    blocked = deepcopy(artifact)
    blocked["status"] = "complete_blocked_test"
    blocked["verdict_class"] = "positive"
    blocked["csl_causal_audit_completed"] = True
    blocked["gate_check_summary"] = {"all_passed": True, "failures": []}
    blocked_errors = exp.validate_artifact(blocked)
    assert {
        "blocked verdict_class mismatch",
        "blocked audit cannot be complete",
        "blocked audit contains replay evidence",
        "blocked audit lacks a failed gate",
    } <= set(blocked_errors)

    incomplete = deepcopy(artifact)
    incomplete["verdict_class"] = "null"
    incomplete["csl_causal_audit_completed"] = False
    assert "full audit must declare completion" in exp.validate_artifact(incomplete)

    unsupported = deepcopy(artifact)
    unsupported["source_verdict_supported"] = False
    unsupported["gate_check_summary"]["all_passed"] = False
    unsupported["credited_factor_count"] = 0
    unsupported_errors = exp.validate_artifact(unsupported)
    assert {
        "positive verdict lacks source support",
        "positive verdict has a failed gate",
        "positive verdict lacks a credited factor",
    } <= set(unsupported_errors)

    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text(json.dumps(malformed), encoding="utf-8")
    with pytest.raises(ValueError, match="required field set mismatch"):
        exp.main(["--validate", "--output", str(invalid_path)])

    written: list[Path] = []
    monkeypatch.setattr(exp, "build_artifact", lambda **kwargs: artifact)
    monkeypatch.setattr(
        exp,
        "write_artifact",
        lambda path, value: written.append(Path(path)) or {"atomic_rename": True},
    )
    output = tmp_path / "cli.json"
    assert exp.main(["--output", str(output)]) == 0
    assert written == [output]

    valid_path = tmp_path / "valid.json"
    valid_path.write_text(json.dumps(artifact), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [str(exp.MODULE_RELATIVE_PATH), "--validate", "--output", str(valid_path)],
    )
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(REPO_ROOT / exp.MODULE_RELATIVE_PATH), run_name="__main__")
    assert stopped.value.code == 0
