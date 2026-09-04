"""Tests for the independent certified-selection cold audit.

Spec refs: REQ-VERIFY-6960 and SCENARIO-VERIFY-6960-*.
"""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from carnot import experiment_6960_certified_selection_cold_audit as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / exp.SPEC_PATH


@pytest.fixture(scope="module")
def frozen_sources() -> dict[str, dict]:
    """Load pinned artifacts once because attack tests only copy their small row lists."""

    loaded = exp.load_frozen_sources(REPO_ROOT)
    assert loaded["passed"] is True
    return loaded["sources"]


@pytest.fixture(scope="module")
def artifact() -> dict:
    """Build one complete real replay for all terminal artifact checks."""

    return exp.build_from_repo(REPO_ROOT, run_date=exp.RUN_DATE)


def test_req_verify_6960_spec_precedes_implementation() -> None:
    """REQ-VERIFY-6960 owns every required surface before code can satisfy it."""

    section = SPEC_PATH.read_text(encoding="utf-8").split("### REQ-VERIFY-6960", 1)[1]
    for marker in (
        "SCENARIO-VERIFY-6960-PRECONDITIONS",
        "SCENARIO-VERIFY-6960-AUTHORITY",
        "SCENARIO-VERIFY-6960-ISOLATION",
        "SCENARIO-VERIFY-6960-ORDER",
        "SCENARIO-VERIFY-6960-PAIRED",
        "SCENARIO-VERIFY-6960-CONTRADICTION",
        exp.INFERENCE_SUBSTRATE,
        exp.MODULE_PATH.as_posix(),
        exp.WRAPPER_PATH.as_posix(),
        exp.RESULT_PATH.as_posix(),
    ):
        assert marker in section
    for field in exp.REQUIRED_FIELDS:
        assert f"`{field}`" in section


def test_scenario_verify_6960_has_independent_fresh_process_entrypoint() -> None:
    """SCENARIO-VERIFY-6960-PRECONDITIONS excludes every upstream producer import."""

    source = inspect.getsource(exp)
    for producer in (
        "experiment_6956_three_family_reformulation_bank",
        "experiment_6957_smt_mapping_certification",
        "experiment_6958_convex_factor_energy_canary",
        "experiment_6959_certified_energy_selection",
    ):
        assert f"import {producer}" not in source
        assert f"import carnot.{producer}" not in source
    wrapper = (REPO_ROOT / exp.WRAPPER_PATH).read_text(encoding="utf-8")
    assert "experiment_6960_certified_selection_cold_audit import main" in wrapper
    assert "raise SystemExit(main())" in wrapper
    assert "--fresh-child" in source


def test_scenario_verify_6960_hashes_are_checked_before_json(monkeypatch, tmp_path: Path) -> None:
    """SCENARIO-VERIFY-6960-PRECONDITIONS rejects changed bytes before JSON parsing."""

    source = tmp_path / "source.json"
    source.write_text('{"valid": true}\n', encoding="utf-8")

    def forbidden_json_loads(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("JSON was parsed before its byte hash passed")

    monkeypatch.setattr(exp.json, "loads", forbidden_json_loads)
    loaded = exp.load_frozen_sources(
        tmp_path,
        source_paths={"source": Path("source.json")},
        expected_hashes={"source": "sha256:wrong"},
    )

    assert loaded["passed"] is False
    assert loaded["sources"] == {}
    assert loaded["hash_rows"][0]["check"] == "source_hash:source"
    assert loaded["hash_rows"][0]["terminal"] is True


def test_scenario_verify_6960_missing_and_modified_raw_rows_fail(
    frozen_sources: dict[str, dict],
) -> None:
    """SCENARIO-VERIFY-6960-PRECONDITIONS keeps all raw proposal bytes binding."""

    bank = frozen_sources["proposal_bank"]
    clean = exp.audit_proposal_bank(bank)
    assert clean["passed"] is True
    assert len(clean["candidate_rows"]) == exp.EXPECTED_CANDIDATE_COUNT

    missing = deepcopy(bank)
    missing["raw_output_rows"].pop()
    missing_result = exp.audit_proposal_bank(missing)
    assert missing_result["passed"] is False
    assert "raw_output_row_count" in missing_result["failed_checks"]

    changed = deepcopy(bank)
    changed["attempt_rows"][0]["raw_text"] += " "
    changed_result = exp.audit_proposal_bank(changed)
    assert changed_result["passed"] is False
    assert changed_result["candidate_rows"][0]["raw_hash_matches"] is False


def test_scenario_verify_6960_reordered_candidates_and_budget_fail(
    frozen_sources: dict[str, dict],
) -> None:
    """SCENARIO-VERIFY-6960-PRECONDITIONS detects order and group-budget drift."""

    bank = deepcopy(frozen_sources["proposal_bank"])
    bank["attempt_rows"][0], bank["attempt_rows"][1] = (
        bank["attempt_rows"][1],
        bank["attempt_rows"][0],
    )
    reordered = exp.audit_proposal_bank(bank)
    assert reordered["passed"] is False
    assert "candidate_ordinal_order" in reordered["failed_checks"]

    bank = deepcopy(frozen_sources["proposal_bank"])
    bank["attempt_rows"].pop()
    budget = exp.audit_proposal_bank(bank)
    assert budget["passed"] is False
    assert "proposal_row_count" in budget["failed_checks"]
    assert any(row["terminal"] for row in budget["proposal_budget_rows"])


def test_scenario_verify_6960_stale_checkpoint_fails(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-6960-PRECONDITIONS binds checkpoint bytes before loading tensors."""

    checkpoint = tmp_path / "factor.pt"
    checkpoint.write_bytes(b"stale")
    rows = exp.audit_checkpoint_hashes(
        tmp_path,
        [{"path": "factor.pt", "expected_sha256": "sha256:expected"}],
    )

    assert rows[0]["passed"] is False
    assert rows[0]["observed_value"] == exp.sha256_path(checkpoint)
    assert rows[0]["terminal"] is True


def test_scenario_verify_6960_replays_both_certificate_authorities(
    artifact: dict,
) -> None:
    """SCENARIO-VERIFY-6960-AUTHORITY reruns Z3 and enumeration for admitted rows."""

    rows = artifact["certificate_replay_rows"]
    assert len(rows) == exp.EXPECTED_CANDIDATE_COUNT
    assert all(row["terminal"] for row in rows)
    assert all(row["status_matches"] and row["label_matches"] for row in rows)
    admitted = [row for row in rows if row["enumeration_status"] in {"proved", "counterexample"}]
    assert admitted
    assert all(row["z3_executed"] and row["enumeration_executed"] for row in admitted)
    assert any(row["enumeration_label"] == "equivalent" for row in admitted)
    assert any(row["enumeration_label"] == "non_equivalent" for row in admitted)


def test_scenario_verify_6960_label_bearing_features_fail() -> None:
    """SCENARIO-VERIFY-6960-ISOLATION rejects exact labels at any payload depth."""

    clean = exp.audit_label_payload("candidate", {"factors": [[0.0]], "metadata": {"seed": 1}})
    leaked = exp.audit_label_payload(
        "candidate",
        {"factors": [[0.0]], "metadata": {"nested": {"exact_mapping_correct": True}}},
    )

    assert clean["passed"] is True
    assert leaked["passed"] is False
    assert leaked["forbidden_paths"] == ["metadata.nested.exact_mapping_correct"]


def test_scenario_verify_6960_score_direction_and_tie_policy_are_exact() -> None:
    """SCENARIO-VERIFY-6960-ORDER catches score inversion and tie drift."""

    candidates = [
        exp.synthetic_candidate("a", "direct_affine", 1.0),
        exp.synthetic_candidate("b", "domain_first", 2.0),
        exp.synthetic_candidate("c", "objective_first", 2.0),
    ]
    minimum = exp.rank_group(candidates, "convex_factor_energy", direction="min")
    inverted = exp.rank_group(candidates, "convex_factor_energy", direction="max")
    tie = exp.rank_group(
        [
            exp.synthetic_candidate("z", "direct_affine", 1.0),
            exp.synthetic_candidate("a", "domain_first", 1.0),
        ],
        "convex_factor_energy",
        direction="min",
    )

    assert minimum["selected_attempt_key"] == "a"
    assert inverted["selected_attempt_key"] == "b"
    assert exp.compare_selection_policy(minimum, inverted)["passed"] is False
    assert tie["selected_attempt_key"] == "z"
    drifted = dict(tie, selected_attempt_key="a")
    assert exp.compare_selection_policy(tie, drifted)["passed"] is False


def test_scenario_verify_6960_unequal_groups_use_pair_bootstrap() -> None:
    """SCENARIO-VERIFY-6960-PAIRED resamples pair IDs instead of candidate rows."""

    rows = [
        {"pair_id": "p1", "paired_top1_delta": 1},
        {"pair_id": "p1", "paired_top1_delta": -1},
        {"pair_id": "p1", "paired_top1_delta": 1},
        {"pair_id": "p2", "paired_top1_delta": 0},
    ]
    first = exp.paired_bootstrap_by_pair(rows, seed=9, samples=200)
    second = exp.paired_bootstrap_by_pair(rows, seed=9, samples=200)

    assert first == second
    assert first["bootstrap_unit"] == "pair_id"
    assert first["paired_pair_count"] == 2
    assert first["paired_group_count"] == 4
    assert first["mean_delta"] == pytest.approx(0.25)
    empty = exp.paired_bootstrap_by_pair([], seed=9, samples=20)
    assert empty["ci95_lower"] is None


def test_scenario_verify_6960_aggregate_contradiction_disqualifies(
    artifact: dict,
    frozen_sources: dict[str, dict],
) -> None:
    """SCENARIO-VERIFY-6960-CONTRADICTION lets row replay override a headline."""

    upstream = deepcopy(frozen_sources["upstream_selection"])
    replay = exp.replay_summary_from_artifact(artifact)
    assert exp.compare_upstream_aggregates(upstream, replay)["contradictions"] == []

    upstream["arm_rows"][0]["top1_accuracy"] = 0.99
    compared = exp.compare_upstream_aggregates(upstream, replay)
    assert compared["contradictions"]
    verdict = exp.reduce_verdict(
        audit_complete=True,
        replay_positive=False,
        upstream=upstream,
        contradictions=compared["contradictions"],
    )
    assert verdict == (0, "disqualified", "complete_disqualified_certified_selection_cold_audit")


def test_req_verify_6960_never_upgrades_null_or_disqualified() -> None:
    """REQ-VERIFY-6960 preserves the upstream claim ceiling."""

    for verdict_class in ("null", "disqualified"):
        upstream = {
            "verdict_class": verdict_class,
            "certified_energy_positive_score": 0,
            "certified_selection_run_complete_score": 1,
        }
        audited = exp.reduce_verdict(
            audit_complete=True,
            replay_positive=True,
            upstream=upstream,
            contradictions=[],
        )
        assert audited[0] == 0
        assert audited[1] in {"null", "disqualified"}
    positive = exp.reduce_verdict(
        audit_complete=True,
        replay_positive=True,
        upstream={
            "verdict_class": "positive",
            "certified_energy_positive_score": 1,
            "certified_selection_run_complete_score": 1,
        },
        contradictions=[],
    )
    assert positive[0:2] == (1, "positive")


def test_req_verify_6960_real_artifact_is_complete_null_and_row_derived(
    artifact: dict,
) -> None:
    """REQ-VERIFY-6960 produces the required stable artifact from frozen rows."""

    assert set(exp.REQUIRED_FIELDS) <= set(artifact)
    assert set(artifact["field_principles"]) == set(exp.REQUIRED_FIELDS)
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False
    assert artifact["certified_selection_audit_complete_score"] == 1
    assert artifact["audited_certified_energy_positive_score"] == 0
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"] == "complete_null_certified_selection_cold_audit"
    assert len(artifact["selection_rows"]) == exp.EXPECTED_GROUP_COUNT * len(exp.ARM_ORDER)
    assert len(artifact["checkpoint_reload_rows"]) == 12
    assert len(artifact["candidate_order_rows"]) == exp.EXPECTED_GROUP_COUNT * len(exp.ARM_ORDER)
    assert artifact["confidence_interval_rows"][0]["ci95_lower"] <= 0.0
    assert artifact["gate_check_summary"]["available_oracle_headroom"] == 0
    assert artifact["contradiction_report_rows"] == []
    assert exp.validate_artifact(artifact) == []


def test_req_verify_6960_blocked_artifact_names_failed_gate(tmp_path: Path) -> None:
    """REQ-VERIFY-6960 writes all fields when a pinned source is absent."""

    blocked = exp.build_from_repo(tmp_path, run_date=exp.RUN_DATE)

    assert set(exp.REQUIRED_FIELDS) <= set(blocked)
    assert blocked["verdict_class"] == "blocked"
    assert blocked["honest_verdict"] == "blocked_certified_selection_cold_audit"
    assert blocked["certified_selection_audit_complete_score"] == 0
    assert blocked["audited_certified_energy_positive_score"] == 0
    assert blocked["gate_check_summary"]["failed_checks"]
    assert all(
        {"check", "expected_value", "observed_value"} <= set(row)
        for row in blocked["gate_check_summary"]["failed_checks"]
    )
    assert exp.validate_artifact(blocked) == []


def test_req_verify_6960_validator_and_writer_fail_closed(artifact: dict, tmp_path: Path) -> None:
    """REQ-VERIFY-6960 validates schema, aggregates, checksum, and atomic output."""

    target = tmp_path / "audit.json"
    exp.write_output(artifact, target)
    assert json.loads(target.read_text(encoding="utf-8")) == artifact

    missing = deepcopy(artifact)
    missing.pop("tie_policy_rows")
    assert "missing_fields:tie_policy_rows" in exp.validate_artifact(missing)


def _valid_attempt(frozen_sources: dict[str, dict]) -> dict:
    """Return one independently reparsed proposal that passed the public schema."""

    replay = exp.audit_proposal_bank(frozen_sources["proposal_bank"])
    return next(row for row in replay["reparsed_attempts"] if row["parse"]["schema_valid"])


def test_req_verify_6960_parse_and_schema_fail_closed_branches(
    frozen_sources: dict[str, dict], tmp_path: Path
) -> None:
    """REQ-VERIFY-6960 reports every malformed raw and exact-schema failure."""

    invalid = tmp_path / "invalid.json"
    invalid.write_bytes(b"\xff")
    loaded = exp.load_frozen_sources(
        tmp_path,
        source_paths={"source": Path("invalid.json")},
        expected_hashes={"source": exp.sha256_path(invalid)},
    )
    assert loaded["passed"] is False
    assert loaded["hash_rows"][-1]["check"] == "source_json_parse"

    attempt = _valid_attempt(frozen_sources)
    assert exp.strict_parse("", attempt)["failure_reason"] == "empty_output"
    assert exp.strict_parse("[]", attempt)["failure_reason"] == "response_object_required"
    assert exp.strict_parse("{}", attempt)["failure_reason"] == "response_keys"
    assert exp.strict_parse('{"mapping": {}, "confidence": true}', attempt)["failure_reason"] == (
        "confidence_range"
    )

    mapping = deepcopy(attempt["parse"]["parsed_candidate"]["mapping"])
    mutations = (
        ("mapping_schema_version", lambda row: row.__setitem__("schema_version", "wrong")),
        ("variable_rows", lambda row: row.__setitem__("variables", {})),
        ("domain_clause_rows", lambda row: row.__setitem__("domain_clauses", {})),
        ("objective_keys", lambda row: row.__setitem__("objective", {})),
        (
            "rational_string_required",
            lambda row: row["variables"][0].__setitem__("scale", False),
        ),
        (
            "objective_direction",
            lambda row: row["objective"].__setitem__("source_direction", "sideways"),
        ),
        ("claimed_relation", lambda row: row.__setitem__("claimed_relation", "unknown")),
    )
    for reason, mutate in mutations:
        changed = deepcopy(mapping)
        mutate(changed)
        assert exp._mapping_schema_error(changed, attempt) == reason


def test_req_verify_6960_exact_mapping_defensive_branches(
    frozen_sources: dict[str, dict], monkeypatch
) -> None:
    """REQ-VERIFY-6960 rejects every exact-engine schema and typing ambiguity."""

    attempt = _valid_attempt(frozen_sources)
    source = attempt["source_formulation"]
    target = attempt["target_formulation"]
    mapping = deepcopy(attempt["parse"]["parsed_candidate"]["mapping"])
    with pytest.raises(ValueError, match="non_exact_rational"):
        exp._fraction(True)
    with pytest.raises(ValueError, match="mapping_keys"):
        exp._canonical_mapping([], source, target)

    bad_containers = deepcopy(mapping)
    bad_containers["variables"] = {}
    with pytest.raises(ValueError, match="mapping_container_type"):
        exp._canonical_mapping(bad_containers, source, target)

    mutations = (
        ("source_variable_coverage", lambda row: row["variables"].pop()),
        (
            "target_variable_coverage",
            lambda row: row["variables"][0].__setitem__("target", "not-a-target"),
        ),
        ("variable_mapping", lambda row: row["variables"][0].__setitem__("scale", "0")),
        ("domain_clause_coverage", lambda row: row["domain_clauses"].pop()),
        (
            "domain_clause_keys",
            lambda row: row["domain_clauses"][0].__setitem__("extra", "field"),
        ),
        ("objective_mapping", lambda row: row["objective"].__setitem__("scale", "0")),
        (
            "objective_direction",
            lambda row: row["objective"].__setitem__("source_direction", "sideways"),
        ),
    )
    for reason, mutate in mutations:
        changed = deepcopy(mapping)
        mutate(changed)
        with pytest.raises(ValueError, match=reason):
            exp._canonical_mapping(changed, source, target)

    boolean_formulation = {
        "variables": [{"name": "b", "kind": "boolean", "universe": [False, True]}]
    }
    assert exp._typed_assignment(boolean_formulation, {}) is None
    assert exp._typed_assignment(boolean_formulation, {"b": Fraction(2)}) is None

    class UnknownSolver:
        def set(self, **_kwargs: object) -> None:
            pass

        def add(self, *_clauses: object) -> None:
            pass

        def check(self) -> object:
            return object()

    monkeypatch.setattr(exp.z3, "Solver", UnknownSolver)
    assert exp._z3_status(True) == "unknown"
    monkeypatch.setattr(exp, "_z3_status", lambda *_clauses: "unknown")
    assert exp.certify_with_z3(source, target, mapping, "case")["status"] == "unknown"


def test_req_verify_6960_replay_and_feature_error_branches(
    frozen_sources: dict[str, dict], artifact: dict
) -> None:
    """REQ-VERIFY-6960 keeps malformed admitted rows terminal without leaking labels."""

    attempt = deepcopy(_valid_attempt(frozen_sources))
    attempt["parse"]["parsed_candidate"]["mapping"]["variables"][0]["scale"] = "0"
    replay = exp.replay_certificates(
        [attempt],
        frozen_sources["certificates"],
        frozen_sources["certificate_checkpoint"],
    )
    assert replay["rows"][0]["enumeration_status"] == "schema_rejected"
    assert exp._unique_row({}, "name", "x") is None

    malformed_affine = deepcopy(_valid_attempt(frozen_sources))
    malformed_affine["parse"]["parsed_candidate"]["mapping"]["variables"][0].pop("scale")
    assert any(row[2] == 1.0 for row in exp.structural_factors(malformed_affine)[:-2])
    malformed_objective = deepcopy(_valid_attempt(frozen_sources))
    malformed_objective["parse"]["parsed_candidate"]["mapping"]["objective"].pop("scale")
    assert exp.structural_factors(malformed_objective)[-2:] == [
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
    ]
    assert exp._likelihood_score({"runtime_receipt": []}) is None
    assert exp._likelihood_score({"mean_logprob": -0.25}) == -0.25
    assert exp._row_contradictions([{"passed": True}, {"passed": False}], "row") == [
        {
            "check": "row:1",
            "expected_value": True,
            "observed_value": False,
            "terminal": True,
        }
    ]
    assert artifact["verdict_class"] == "null"


def test_req_verify_6960_checkpoint_selection_and_aggregate_guards(
    frozen_sources: dict[str, dict], tmp_path: Path, monkeypatch
) -> None:
    """REQ-VERIFY-6960 covers checkpoint, ranking, and group-integrity guards."""

    checkpoint = tmp_path / "convex_factor_energy_seed_11.pt"
    checkpoint.write_bytes(b"factor")
    assert exp._resolve_checkpoint_path(tmp_path, str(checkpoint)) == checkpoint
    assert exp.load_factor_checkpoints(tmp_path, {}, [])["passed"] is False

    monkeypatch.setattr(
        exp.torch, "load", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("bad"))
    )
    loaded = exp.load_factor_checkpoints(
        tmp_path,
        {"checkpoint_paths": [str(checkpoint)]},
        [{"passed": True}],
    )
    assert loaded["rows"][0]["error"] == "OSError:bad"

    with pytest.raises(ValueError, match="empty_candidate_group"):
        exp.rank_group([], "convex_factor_energy")
    candidate = exp.synthetic_candidate("a", "direct_affine", 1.0)
    with pytest.raises(ValueError, match="unknown_arm"):
        exp.rank_group([candidate], "unknown")
    with pytest.raises(ValueError, match="unknown_direction"):
        exp.rank_group([candidate], "convex_factor_energy", direction="sideways")
    assert exp._calibration([]) == (None, None)

    duplicate = {
        "group_id": "g",
        "model_family": "m",
        "arm": "syntax_heuristic",
        "selection_probability": 0.5,
        "selected_exact_correct": False,
    }
    with pytest.raises(ValueError, match="duplicate_group_arm"):
        exp.aggregate_selection_rows(
            [duplicate, duplicate], "model_family", "m", "syntax_heuristic"
        )
    assert (
        exp.reduce_verdict(
            audit_complete=False,
            replay_positive=False,
            upstream={"verdict_class": "partial"},
            contradictions=[],
        )[1]
        == "partial"
    )
    assert exp._source_hash_map([{"hash_kind": "factor_checkpoint"}]) == {}
    assert frozen_sources["upstream_selection"]["verdict_class"] == "null"


def test_req_verify_6960_preflight_and_reload_blocks(
    frozen_sources: dict[str, dict], monkeypatch, tmp_path: Path
) -> None:
    """REQ-VERIFY-6960 emits blocked artifacts at both post-hash fail-closed gates."""

    sources = frozen_sources
    monkeypatch.setattr(
        exp,
        "load_frozen_sources",
        lambda _root: {"passed": True, "sources": sources, "hash_rows": []},
    )
    monkeypatch.setattr(exp, "audit_proposal_bank", lambda _bank: {"raw_hash_rows": []})
    monkeypatch.setattr(
        exp,
        "preflight_loaded_sources",
        lambda *_args: {"checks": [exp._gate("preflight", True, False)]},
    )
    monkeypatch.setattr(exp, "_checkpoint_entries", lambda _energy: [])
    monkeypatch.setattr(exp, "audit_checkpoint_hashes", lambda *_args: [])
    assert exp.build_from_repo(tmp_path)["verdict_class"] == "blocked"

    entries = [{"path": "factor.pt"}] * len(exp.EXPECTED_FACTOR_HASHES)
    hash_rows = [{"passed": True}] * len(exp.EXPECTED_FACTOR_HASHES)
    monkeypatch.setattr(
        exp,
        "preflight_loaded_sources",
        lambda *_args: {"checks": [exp._gate("preflight", True, True)]},
    )
    monkeypatch.setattr(exp, "_checkpoint_entries", lambda _energy: entries)
    monkeypatch.setattr(exp, "audit_checkpoint_hashes", lambda *_args: hash_rows)
    monkeypatch.setattr(
        exp,
        "load_factor_checkpoints",
        lambda *_args: {"passed": False, "models": {}, "rows": []},
    )
    assert exp.build_from_repo(tmp_path)["gate_check_summary"]["failed_checks"][-1]["check"] == (
        "factor_checkpoint_reload"
    )


def test_req_verify_6960_validator_and_process_helpers(
    artifact: dict, tmp_path: Path, monkeypatch, capsys
) -> None:
    """REQ-VERIFY-6960 covers every validator and isolated-child helper outcome."""

    invalid = deepcopy(artifact)
    invalid["field_principles"] = {}
    invalid["inference_substrate"] = "wrong"
    invalid["verifier_is_oracle"] = True
    invalid["audited_certified_energy_positive_score"] = 1
    invalid["certified_selection_audit_complete_score"] = 0
    invalid["contradiction_report_rows"] = [{"check": "x"}]
    invalid["verdict_class"] = "null"
    invalid["honest_verdict"] = "complete_positive_certified_selection_cold_audit"
    errors = exp.validate_artifact(invalid)
    assert {
        "field_principles_mismatch",
        "inference_substrate_mismatch",
        "verifier_is_oracle_mismatch",
        "positive_without_complete_audit",
        "contradiction_requires_disqualified_verdict",
        "verdict_class_prefix_mismatch",
        "reproducibility_checksum_mismatch",
    } <= set(errors)
    with pytest.raises(ValueError, match="missing_fields"):
        exp.write_output({}, tmp_path / "invalid.json")

    command = exp._child_command(
        date="20260904", repo_root=tmp_path, output_path=tmp_path / "out.json", parent_pid=7
    )
    assert command[2:4] == ["--fresh-child", "--date"]
    monkeypatch.setattr(
        exp.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout="child-out\n", stderr="child-error\n", returncode=2
        ),
    )
    assert exp.launch_fresh_process("20260904", tmp_path, tmp_path / "out.json") == 2
    captured = capsys.readouterr()
    assert "child-out" in captured.out
    assert "child-error" in captured.err

    changed = deepcopy(artifact)
    changed["selection_rows"][0]["selected_exact_correct"] = not changed["selection_rows"][0][
        "selected_exact_correct"
    ]
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(changed)

    contradiction = deepcopy(artifact)
    contradiction["contradiction_report_rows"] = [{"check": "injected", "terminal": True}]
    contradiction["verdict_class"] = "null"
    contradiction["reproducibility_checksum"] = exp.reproducibility_checksum(contradiction)
    assert "contradiction_requires_disqualified_verdict" in exp.validate_artifact(contradiction)
