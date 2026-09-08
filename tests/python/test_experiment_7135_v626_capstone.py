"""Tests for REQ-CAPSTONE-7135 and its evidence-boundary scenarios."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from carnot import experiment_7135_v626_capstone as exp


REPO = Path(__file__).resolve().parents[2]


def _clean_verifier(_path: Path) -> dict[str, object]:
    return {"loaded": True, "flag_count": 0, "flags": []}


def _critical_verifier(_path: Path) -> dict[str, object]:
    return {
        "loaded": True,
        "flag_count": 1,
        "flags": [{"kind": "LIVE_CRITICAL", "severity": "critical"}],
    }


def _payload(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "run_date": "20260908",
        "inference_substrate": "aggregation_from_upstream_artifacts: fixture",
        "inference_substrate_class": "aggregation",
        "execution_venue": "host",
        "verifier_is_oracle": False,
        "verdict_class": "positive",
        "honest_verdict": "complete_positive_fixture",
        "rows": [{"score": 1}],
        "source_artifact_hashes": {},
    }
    value.update(updates)
    return value


def _write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _task(
    task_id: str,
    path: Path,
    *,
    gates: list[dict[str, object]] | None = None,
    prior_failures: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "id": task_id,
        "number": int(task_id[3:7]),
        "title": task_id,
        "deliverable": str(path),
        "path": path,
        "gates": gates or [],
        "prior_failures": prior_failures or [],
    }


def test_req_capstone_7135_spec_precedes_implementation() -> None:
    """REQ-CAPSTONE-7135 owns every required matrix field and scenario."""

    text = (REPO / exp.SPEC_PATH).read_text(encoding="utf-8")
    section = text.split("REQ-CAPSTONE-7135", 1)[1]

    for anchor in (
        "SCENARIO-CAPSTONE-7135-MISSING",
        "SCENARIO-CAPSTONE-7135-INGRESS",
        "SCENARIO-CAPSTONE-7135-GATES",
        "SCENARIO-CAPSTONE-7135-VERDICTS",
        "SCENARIO-CAPSTONE-7135-ROWS",
        "SCENARIO-CAPSTONE-7135-BRANCHES",
        "SCENARIO-CAPSTONE-7135-RETIREMENT",
        "SCENARIO-CAPSTONE-7135-ARTIFACT",
    ):
        assert anchor in section
    assert exp.INFERENCE_SUBSTRATE in section
    assert all(field in section for field in exp.REQUIRED_ARTIFACT_FIELDS)


def test_req_capstone_7135_resolves_exact_active_contract() -> None:
    """REQ-CAPSTONE-7135 resolves 11 upstream paths and four gates from YAML."""

    tasks, contract_rows = exp.load_contract(REPO)

    assert tuple(task["id"] for task in tasks) == exp.EXPECTED_UPSTREAM_IDS
    assert len(tasks) == 11
    assert sum(len(task["gates"]) for task in tasks) == 4
    assert len(contract_rows) == 12
    assert all(row["passed"] for row in contract_rows)
    assert tasks[0]["path"] == REPO / "results/experiment_7124_v626_contract_preflight.json"


def test_scenario_capstone_7135_ingress_excludes_stamp_and_live_critical(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-7135-INGRESS excludes both quarantine sources."""

    stamped_path = tmp_path / "stamped.json"
    critical_path = tmp_path / "critical.json"
    _write(stamped_path, _payload(flagged_adversarial=True))
    _write(critical_path, _payload())
    stamped = exp.ingest_upstreams(
        [_task("exp7124-stamped", stamped_path)], verifier=_clean_verifier
    )
    critical = exp.ingest_upstreams(
        [_task("exp7125-critical", critical_path)], verifier=_critical_verifier
    )

    assert stamped["accepted_upstreams"] == []
    assert stamped["excluded_flagged_upstreams"][0]["flag_sources"] == ["artifact"]
    assert critical["accepted_upstreams"] == []
    assert critical["excluded_flagged_upstreams"][0]["flag_sources"] == ["verifier"]


def test_scenario_capstone_7135_ingress_detects_consumer_bound_stale_hash(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-7135-INGRESS rejects changed producer bytes."""

    producer_path = tmp_path / "producer.json"
    consumer_path = tmp_path / "consumer.json"
    _write(producer_path, _payload(sota_constraint_bank_ready_score=1))
    _write(
        consumer_path,
        _payload(source_artifact_hashes={str(producer_path): "sha256:" + "0" * 64}),
    )
    producer = _task("exp7129-producer", producer_path)
    consumer = _task(
        "exp7130-consumer",
        consumer_path,
        gates=[
            {
                "upstream": "exp7129-producer",
                "artifact_field": "sota_constraint_bank_ready_score",
                "op": "==",
                "value": 1,
            }
        ],
    )

    ingress = exp.ingest_upstreams([producer, consumer], verifier=_clean_verifier)
    rejected = {row["task_id"]: row for row in ingress["rejected_upstream_rows"]}

    assert "hash_drift" in rejected["exp7129-producer"]["reason_codes"]
    assert {row["task_id"] for row in ingress["accepted_upstreams"]} == {"exp7130-consumer"}


def test_scenario_capstone_7135_gates_keep_failure_modes_distinct() -> None:
    """SCENARIO-CAPSTONE-7135-GATES uses only exact top-level producer fields."""

    gate = {
        "upstream": "exp7129-producer",
        "artifact_field": "sota_constraint_bank_ready_score",
        "op": "==",
        "value": 1,
    }
    consumer = _task("exp7130-consumer", Path("consumer.json"), gates=[gate])
    producer = _task("exp7129-producer", Path("producer.json"))

    passed = exp.recompute_gates(
        [producer, consumer],
        {"exp7129-producer": {"payload": {"sota_constraint_bank_ready_score": 1}}},
        {},
    )[0]
    failed = exp.recompute_gates(
        [producer, consumer],
        {"exp7129-producer": {"payload": {"sota_constraint_bank_ready_score": 0}}},
        {},
    )[0]
    missing_field = exp.recompute_gates(
        [producer, consumer],
        {"exp7129-producer": {"payload": {"nested": {"sota_constraint_bank_ready_score": 1}}}},
        {},
    )[0]
    missing = exp.recompute_gates(
        [producer, consumer],
        {},
        {"exp7129-producer": {"reason_codes": ["artifact_not_found"]}},
    )[0]
    excluded = exp.recompute_gates(
        [producer, consumer],
        {},
        {"exp7129-producer": {"reason_codes": ["verifier_flagged_adversarial"]}},
    )[0]
    stale = exp.recompute_gates(
        [producer, consumer],
        {},
        {"exp7129-producer": {"reason_codes": ["hash_drift"]}},
    )[0]
    unreadable = exp.recompute_gates(
        [producer, consumer],
        {},
        {"exp7129-producer": {"reason_codes": ["artifact_unreadable"]}},
    )[0]

    assert passed["status"] == "passed"
    assert failed["status"] == "valid_gate_failure"
    assert missing_field["status"] == "missing_field"
    assert missing["status"] == "missing_artifact"
    assert excluded["status"] == "excluded_flagged_upstream"
    assert stale["status"] == "stale_hash"
    assert unreadable["status"] == "missing_artifact"


def test_scenario_capstone_7135_verdicts_bound_circular_and_blocked() -> None:
    """SCENARIO-CAPSTONE-7135-VERDICTS preserves structural classes."""

    circular = exp.classify_verdict(_payload(verifier_is_oracle=True, verdict_class="positive"))
    blocked = exp.classify_verdict(
        _payload(verdict_class="partial", honest_verdict="blocked_gate_check_failed")
    )

    assert circular["structural_verdict_class"] == "circular_positive"
    assert circular["effective_verdict_class"] == "circular_positive"
    assert blocked["structural_verdict_class"] == "blocked"
    assert blocked["effective_verdict_class"] == "blocked"
    assert blocked["consistent"] is False


def test_scenario_capstone_7135_rows_recompute_and_catch_reversal() -> None:
    """SCENARIO-CAPSTONE-7135-ROWS catches arithmetic and directional reversals."""

    arc = _payload(
        verdict_class="null",
        honest_verdict="complete_null_pair",
        withheld_levels=1,
        control_levels=0,
        level_delta=1,
        arc_loo_cell_complete_score=1,
        rows=[
            {"arm": "adapter_withheld", "levels": 1, "executed_transition_count": 2},
            {"arm": "adapter_visible_control", "levels": 0, "executed_transition_count": 2},
        ],
    )
    recomputed = exp.recompute_headlines(7127, arc)
    by_field = {row["field"]: row for row in recomputed}
    reversal = exp.row_reversal_rows(
        "exp7134-fixture",
        _payload(rows=[{"paired_delta": 1.0}, {"paired_delta": -2.0}]),
    )
    unavailable = exp.recompute_headlines(7132, _payload(rows=[]))

    assert by_field["level_delta"]["recomputed_value"] == 1
    assert all(row["status"] == "agrees" for row in recomputed)
    assert reversal[0]["kind"] == "row_reversal"
    assert unavailable[0]["status"] == "unavailable"


def test_scenario_capstone_7135_task_specific_reducers_and_defensive_shapes() -> None:
    """SCENARIO-CAPSTONE-7135-ROWS covers all planned comparison shapes."""

    assert (
        exp.recompute_headlines(7127, _payload(rows=[{"arm": "wrong"}]))[0]["status"]
        == "unavailable"
    )
    assert (
        exp.recompute_headlines(
            7127,
            _payload(
                rows=[
                    {"arm": "adapter_withheld", "levels": None},
                    {"arm": "adapter_visible_control", "levels": 0},
                ]
            ),
        )[0]["status"]
        == "unavailable"
    )
    bank = _payload(
        planned_cell_count=2,
        completed_cell_count=2,
        sota_constraint_bank_ready_score=1,
        rows=[{"terminal_state": "complete"}, {"terminal_state": "complete"}],
    )
    assert all(row["status"] == "agrees" for row in exp.recompute_headlines(7129, bank))

    future = [
        {"arm": "fixed_schema", "exact_success": 1},
        {"arm": "no_memory", "exact_success": 0},
        {"arm": "free_note", "exact_success": 0},
    ]
    learning = _payload(
        rows=[{"score": 1}],
        future_episode_rows=future,
        protected_retention_rows=[{"arm": "fixed_schema", "exact_success": 1}],
        later_value_delta=1.0,
        protected_retention_delta=0.0,
        model_facing_csl_complete_score=1,
    )
    assert all(row["status"] == "agrees" for row in exp.recompute_headlines(7131, learning))
    missing_arms = deepcopy(learning)
    missing_arms["future_episode_rows"] = [{"arm": "fixed_schema", "exact_success": 1}]
    assert exp.recompute_headlines(7131, missing_arms)[0]["status"] == "unavailable"

    models = ("a", "b", "c")
    directions = [
        {"writer_model": writer, "reader_model": reader}
        for writer in models
        for reader in models
        if writer != reader
    ]
    portability = _payload(
        rows=[{"score": 1}],
        direction_rows=directions,
        six_ordered_pairs_complete=True,
        directional_memory_portability_complete_score=1,
    )
    assert all(row["status"] == "agrees" for row in exp.recompute_headlines(7132, portability))
    assert exp.recompute_headlines(7132, _payload())[0]["status"] == "unavailable"
    assert (
        exp.recompute_headlines(7134, _payload(rows=[{"arm": "exact_law"}]))[0]["status"]
        == "unavailable"
    )
    assert exp.recompute_headlines(9999, _payload())[0]["status"] == "unavailable"
    assert (
        exp.row_reversal_rows("null", _payload(verdict_class="null", honest_verdict="null_result"))
        == []
    )
    assert exp.row_reversal_rows("bad-rows", _payload(rows="not-a-list")) == []


def test_req_capstone_7135_defensive_contract_hash_and_verdict_helpers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CAPSTONE-7135 fails closed on malformed contracts and provenance."""

    (tmp_path / exp.ROADMAP_PATH).write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="task list"):
        exp.load_contract(tmp_path)

    (tmp_path / exp.ROADMAP_PATH).write_text(
        "tasks:\n  - not-a-mapping\n  - id: exp1-outside\n", encoding="utf-8"
    )
    (tmp_path / exp.DESIGN_PATH).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / exp.DESIGN_PATH).write_text("no contract\n", encoding="utf-8")
    tasks, rows = exp.load_contract(tmp_path)
    assert tasks == []
    assert len(rows) == 12 and not any(row["passed"] for row in rows)

    producer = {"deliverable": "producer.json", "path": tmp_path / "producer.json"}
    digest = "sha256:" + "1" * 64
    assert (
        exp._source_hash_for(
            {"source_artifact_hashes": {"producer.json": {"sha256": digest}}}, producer
        )
        == digest
    )
    assert (
        exp._source_hash_for(
            {"source_artifact_hashes": [{"path": "producer.json", "sha256": digest}]}, producer
        )
        == digest
    )
    assert (
        exp._source_hash_for({"source_artifact_hashes": {"upstream_artifact": digest}}, producer)
        == digest
    )
    assert exp._source_hash_for({"upstream_artifact_hash": digest}, producer) == digest
    assert exp._source_hash_for({}, producer) is None
    assert exp._scalar({"value": 3, "principle": "why"}) == 3
    assert (
        exp.classify_verdict(_payload(verdict_class="partial", honest_verdict="partial_retry"))[
            "effective_verdict_class"
        ]
        == "partial"
    )
    assert (
        exp.classify_verdict(_payload(verdict_class="invalid", honest_verdict="unknown"))[
            "effective_verdict_class"
        ]
        == "disqualified"
    )
    assert exp._compare(2, ">=", 1) is True
    assert exp._compare(True, ">=", 1) is False
    assert exp._compare(1, "!=", 1) is False
    assert exp._consumer_gate_claim({}, "field") == (None, None)
    assert exp.recompute_headlines(7131, _payload())[0]["status"] == "unavailable"

    orphan = _task(
        "exp7130-orphan",
        tmp_path / "orphan.json",
        gates=[{"upstream": "not-in-contract", "artifact_field": "ready", "op": "==", "value": 1}],
    )
    _write(orphan["path"], _payload())
    assert (
        exp.ingest_upstreams([orphan], verifier=_clean_verifier)["consumer_bound_expected_hashes"]
        == {}
    )

    task = _task("exp7131-bad-prior", tmp_path / "none.json", prior_failures=["bad"])
    assert exp.build_retirement_rows([task], {}) == []
    monkeypatch.setattr(exp.importlib.util, "spec_from_file_location", lambda *_: None)
    with pytest.raises(RuntimeError, match="cannot load adversarial verifier"):
        exp._load_verifier(tmp_path)
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    assert exp._read_json(malformed) is None
    malformed.write_text("[]", encoding="utf-8")
    assert exp._read_json(malformed) is None


def test_req_capstone_7135_build_records_internal_contradictions(tmp_path: Path) -> None:
    """REQ-CAPSTONE-7135 makes verdict, headline, and reversal defects visible."""

    for relative in (
        exp.ROADMAP_PATH,
        exp.DESIGN_PATH,
        exp.SPEC_PATH,
        exp.INGRESS_PATH,
        exp.ADVERSARIAL_PATH,
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / relative, target)
    tasks, _ = exp.load_contract(tmp_path)
    by_number = {task["number"]: task for task in tasks}
    _write(
        by_number[7124]["path"],
        _payload(verdict_class="partial", honest_verdict="blocked_gate_check_failed"),
    )
    _write(
        by_number[7127]["path"],
        _payload(
            withheld_levels=1,
            control_levels=0,
            level_delta=99,
            arc_loo_cell_complete_score=1,
            rows=[
                {"arm": "adapter_withheld", "levels": 1, "executed_transition_count": 1},
                {"arm": "adapter_visible_control", "levels": 0, "executed_transition_count": 1},
            ],
        ),
    )
    _write(
        by_number[7134]["path"],
        _payload(rows=[{"paired_delta": 1.0}, {"paired_delta": -1.0}]),
    )

    artifact = exp.build_artifact(
        tmp_path,
        "20260908",
        verifier=_clean_verifier,
        output_path=tmp_path / "results/experiment_7135.json",
    )
    kinds = {row["kind"] for row in artifact["contradiction_rows"]}

    assert {"verdict_class_mismatch", "headline_row_mismatch", "row_reversal"} <= kinds


def test_scenario_capstone_7135_retirement_and_branch_actions() -> None:
    """SCENARIO-CAPSTONE-7135-RETIREMENT prevents unchanged failed reruns."""

    task = _task(
        "exp7131-learning",
        Path("missing.json"),
        prior_failures=[
            {
                "experiment_id": "old-learning",
                "verdict": "complete_null_repeat",
                "addressed_by": "Changed model-facing evidence.",
                "retire_if_same_verdict": True,
            }
        ],
    )
    outcomes = {
        "exp7131-learning": {
            "effective_verdict_class": "null",
            "honest_verdict": "complete_null_repeat",
        }
    }
    retirement = exp.build_retirement_rows([task], outcomes)
    disposition = exp.build_branch_disposition(
        "continuous_learning", ["exp7131-learning"], outcomes, retirement
    )

    assert retirement[0]["verdict_repeated"] is True
    assert retirement[0]["action"] == "retire"
    assert disposition["verdict_class"] == "null"
    assert disposition["recommended_action"] == "retire"
    assert disposition["failed_scope_unchanged"] is False


def test_req_capstone_7135_current_artifact_is_complete_without_promotion() -> None:
    """REQ-CAPSTONE-7135 completes the matrix while preserving branch failures."""

    artifact = exp.build_artifact(REPO, "20260908")

    assert artifact["v626_capstone_complete_score"] == 1
    assert artifact["verdict_class"] == "positive"
    assert artifact["honest_verdict"] == exp.COMPLETE_VERDICT
    assert artifact["expected_upstream_ids"] == list(exp.EXPECTED_UPSTREAM_IDS)
    assert {row["task_id"] for row in artifact["missing_upstream_rows"]} == {
        "exp7131-model-facing-fixed-schema-csl",
        "exp7132-directional-memory-portability-audit",
    }
    assert {row["task_id"] for row in artifact["excluded_flagged_upstreams"]} == {
        "exp7129-hardness-controlled-sota-constraint-bank"
    }
    assert artifact["arc_branch_disposition"]["verdict_class"] == "disqualified"
    assert artifact["continuous_learning_disposition"]["verdict_class"] == "blocked"
    assert artifact["portability_disposition"]["verdict_class"] == "blocked"
    assert artifact["sampling_disposition"]["verdict_class"] == "positive"
    assert all(
        not row["promoted_as_positive_science"]
        for row in artifact["claim_ceiling_rows"]
        if row["verdict_class"] != "positive"
    )
    assert exp.validate_artifact(artifact) == []

    mutations = []
    for key, value in (
        ("field_principles", {}),
        ("inference_substrate", "wrong"),
        ("inference_substrate_class", "wrong"),
        ("execution_venue", "gpu"),
        ("verifier_is_oracle", True),
        ("verdict_class", "wrong"),
        ("honest_verdict", "blocked_wrong"),
        ("rows", []),
        ("v626_capstone_complete_score", 0),
    ):
        changed = deepcopy(artifact)
        changed[key] = value
        changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
        mutations.append(changed)
    promoted = deepcopy(artifact)
    promoted["claim_ceiling_rows"][0]["promoted_as_positive_science"] = True
    promoted["reproducibility_checksum"] = exp.reproducibility_checksum(promoted)
    mutations.append(promoted)
    assert all(exp.validate_artifact(changed) for changed in mutations)


def test_req_capstone_7135_blocked_precondition_and_validator(tmp_path: Path) -> None:
    """REQ-CAPSTONE-7135 writes a schema-complete capstone-owned block."""

    artifact = exp.build_artifact(tmp_path, "20260908", verifier=_clean_verifier)

    assert artifact["v626_capstone_complete_score"] == 0
    assert artifact["verdict_class"] == "blocked"
    assert artifact["inference_substrate_class"] == "blocked_no_run"
    assert artifact["gate_check_summary"]["failed_check"]
    assert exp.validate_artifact(artifact) == []

    broken = deepcopy(artifact)
    broken.pop("rows")
    assert exp.validate_artifact(broken)[0].startswith("missing_required_fields")

    broken = deepcopy(artifact)
    broken["gate_check_summary"] = {"failed_check": None}
    broken["reproducibility_checksum"] = exp.reproducibility_checksum(broken)
    assert "blocked_gate_summary_invalid" in exp.validate_artifact(broken)

    broken = deepcopy(artifact)
    broken["reproducibility_checksum"] = "sha256:wrong"
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(broken)


def test_req_capstone_7135_cli_writes_and_validates(tmp_path: Path) -> None:
    """REQ-CAPSTONE-7135 exposes the exact command writer and validator."""

    output = tmp_path / "experiment_7135.json"
    assert exp.main(["--root", str(REPO), "--date", "20260908", "--output", str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert exp.validate_artifact(payload) == []
    assert exp.main(["--validate", "--output", str(output)]) == 0
    output.write_text("{}", encoding="utf-8")
    assert exp.main(["--validate", "--output", str(output)]) == 1


def test_req_capstone_7135_checksum_ignores_duration() -> None:
    """SCENARIO-CAPSTONE-7135-ARTIFACT binds evidence, not wall-clock noise."""

    artifact = {"duration_s": 1.0, "reproducibility_checksum": "old", "rows": []}
    first = exp.reproducibility_checksum(artifact)
    artifact["duration_s"] = 99.0

    assert first == exp.reproducibility_checksum(artifact)
    assert (
        first
        == "sha256:"
        + hashlib.sha256(
            json.dumps(
                {"duration_s": None, "reproducibility_checksum": None, "rows": []},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    )
