"""Tests for the independent selective-arbiter authority attack shard.

Spec refs: REQ-CONSTRAINT-6825, SCENARIO-CONSTRAINT-6825-PRECONDITIONS,
SCENARIO-CONSTRAINT-6825-PRIORITY, SCENARIO-CONSTRAINT-6825-SAFE-BYTES,
SCENARIO-CONSTRAINT-6825-CERTIFICATES, SCENARIO-CONSTRAINT-6825-FEATURES,
SCENARIO-CONSTRAINT-6825-ROW-INTEGRITY, and
SCENARIO-CONSTRAINT-6825-COMPLETION.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path

import pytest

from carnot import experiment_6825_selective_arbiter_authority_attacks as attacks


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATHS = attacks.source_paths_for_root(REPO_ROOT)


@pytest.fixture(scope="module")
def sources() -> dict[str, dict]:
    """Load the three frozen artifacts once because no test edits them."""

    return attacks.load_sources(SOURCE_PATHS)


@pytest.fixture(scope="module")
def source_cases(sources: dict[str, dict]) -> list[dict]:
    """Build all public-contract cases without reading producer decisions."""

    return attacks.build_source_cases(sources)


@pytest.fixture(scope="module")
def attack_rows(source_cases: list[dict]) -> list[dict]:
    """Run the complete bounded mutation matrix once for row assertions."""

    return attacks.run_all_attacks(source_cases)


def _case(source_cases: list[dict], family: str) -> dict:
    """Return the first deterministic scenario from one required family."""

    return next(row for row in source_cases if row["family"] == family)


def _fresh_receipt(rows: list[dict]) -> dict:
    """Make a stable successful replay receipt for pure artifact tests."""

    digest = attacks.attack_rows_digest(rows)
    return {
        "byte_identical": True,
        "fresh_process": True,
        "replay_rows_sha256": digest,
        "rows_sha256": digest,
    }


def test_req_constraint_6825_spec_precedes_implementation() -> None:
    """REQ-CONSTRAINT-6825 owns each required field, scenario, and path."""

    spec = (REPO_ROOT / attacks.SPEC_RELATIVE_PATH).read_text(encoding="utf-8")
    section = spec.split("### REQ-CONSTRAINT-6825", 1)[1]
    for requirement_id in attacks.OPEN_SPEC_IDS[1:]:
        assert requirement_id in section
    for field in attacks.TASK_REQUIRED_FIELDS:
        assert f"`{field}`" in section
    for path in (
        attacks.MODULE_RELATIVE_PATH,
        attacks.SCRIPT_RELATIVE_PATH,
        attacks.RESULT_RELATIVE_PATH,
    ):
        assert path.as_posix() in section


def test_scenario_constraint_6825_preconditions_accept_frozen_sources(
    sources: dict[str, dict], source_cases: list[dict]
) -> None:
    """SCENARIO-CONSTRAINT-6825-PRECONDITIONS seals all 48 raw cases."""

    summary = attacks.check_preconditions(sources, SOURCE_PATHS)
    assert summary["passed"] is True
    assert summary["failed_checks"] == []
    assert len(source_cases) == 48
    assert {row["family"] for row in source_cases} == {
        "already_safe_proposals",
        "competing_authorities",
        "consequence",
        "fallback",
        "soft_conflict",
        "stale_prerequisites",
    }
    for case in source_cases:
        assert len(case["representative_row_ids"]) == 2
        assert case["raw_output_sha256"].startswith("sha256:")
        assert case["source_case_sha256"] == attacks.sha256_json(
            {
                "raw_output_sha256": case["raw_output_sha256"],
                "representative_row_ids": case["representative_row_ids"],
                "scenario": case["scenario"],
            }
        )


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        ("schema", "frozen_obligation_schema"),
        ("source_hash", "source_artifact_hashes"),
        ("raw_rows", "raw_representative_rows"),
        ("completion", "selective_arbiter_ab_completed"),
    ],
)
def test_scenario_constraint_6825_preconditions_fail_closed(
    sources: dict[str, dict], mutation: str, failed_check: str
) -> None:
    """SCENARIO-CONSTRAINT-6825-PRECONDITIONS names each blocked gate."""

    changed = deepcopy(sources)
    paths = dict(SOURCE_PATHS)
    if mutation == "schema":
        changed["exp6811"]["priority_order"] = ["soft", "binding", "hard"]
    elif mutation == "source_hash":
        changed["exp6813"]["source_artifact_sha256"] = "sha256:wrong"
    elif mutation == "raw_rows":
        changed["exp6812"]["raw_output_manifest"] = []
    else:
        changed["exp6813"]["selective_arbiter_ab_completed"] = False
    summary = attacks.check_preconditions(changed, paths)
    assert summary["passed"] is False
    assert failed_check in summary["failed_checks"]
    failed = next(row for row in summary["checks"] if row["check"] == failed_check)
    assert failed["passed"] is False
    assert "expected" in failed and "observed" in failed


def test_req_constraint_6825_source_loading_retains_errors(tmp_path: Path) -> None:
    """REQ-CONSTRAINT-6825 converts unreadable inputs into blocked receipts."""

    invalid = tmp_path / "invalid.json"
    invalid.write_text("[]", encoding="utf-8")
    missing = tmp_path / "missing.json"
    loaded = attacks.load_sources({"invalid": invalid, "missing": missing})
    assert loaded["invalid"] == {}
    assert loaded["missing"] == {}
    assert set(loaded["__load_errors__"]) == {"invalid", "missing"}


def test_scenario_constraint_6825_priority_and_authority_fail_closed(
    source_cases: list[dict],
) -> None:
    """SCENARIO-CONSTRAINT-6825-PRIORITY blocks soft and spoofed authority."""

    case = _case(source_cases, "competing_authorities")
    obligations = attacks.rebuild_obligations(case["scenario"], case["obligation_schema"])
    assert [row["priority_class"] for row in obligations] == ["hard", "binding"]
    assert [row["authority_order"] for row in obligations] == [0, 1]

    for attack_id in ("priority_inversion", "authority_spoofing"):
        row = attacks.run_attack(case, attack_id, source_cases)
        assert row["applicable"] is True
        assert row["passed"] is True
        assert row["failed_closed"] is True
        assert row["observed"]["selected_candidate_id"] == "candidate_1"
        assert row["observed"]["accepted_hard_violation"] is False


def test_scenario_constraint_6825_stale_fallback_and_no_candidate(
    source_cases: list[dict],
) -> None:
    """SCENARIO-CONSTRAINT-6825-PRIORITY sends stale or unsafe sets to fallback."""

    case = _case(source_cases, "stale_prerequisites")
    for attack_id in ("stale_prerequisite", "no_candidate"):
        row = attacks.run_attack(case, attack_id, source_cases)
        assert row["passed"] is True
        assert row["observed"]["selected_candidate_id"] is None
        assert row["observed"]["selected_action_bytes_b64"] == case["fallback_bytes_b64"]
        assert row["observed"]["certificate"]["kind"] == "no_candidate"


def test_scenario_constraint_6825_contract_mutations_fail_before_selection(
    source_cases: list[dict],
) -> None:
    """SCENARIO-CONSTRAINT-6825-PRIORITY rejects three sealed contract changes."""

    case = _case(source_cases, "consequence")
    for attack_id in ("fallback_deletion", "consequence_weakening"):
        row = attacks.run_attack(case, attack_id, source_cases)
        assert row["passed"] is True
        assert row["failed_closed"] is True
        assert row["observed"]["error"] in {
            "frozen_contract_hash_mismatch",
            "missing contract field: execution_consequence",
            "missing contract field: fallback",
        }

    inverted = deepcopy(case["scenario"])
    inverted["obligations"][0]["contract"]["priority"]["class"] = "unknown"
    with pytest.raises(attacks.AuthorityAttackError, match="priority class"):
        attacks.rebuild_obligations(inverted, case["obligation_schema"])


def test_scenario_constraint_6825_tie_and_canonical_bytes(
    source_cases: list[dict],
) -> None:
    """SCENARIO-CONSTRAINT-6825-SAFE-BYTES keeps the stable tie and byte seal."""

    case = _case(source_cases, "soft_conflict")
    tie = attacks.run_attack(case, "tie_reorder", source_cases)
    assert tie["passed"] is True
    assert tie["observed"]["selected_candidate_id"] == "candidate_0"
    canonical = attacks.run_attack(case, "canonical_byte_mutation", source_cases)
    assert canonical["passed"] is True
    assert canonical["observed"]["mutated_candidate_rejected"] is True
    assert canonical["observed"]["first_conflict"] == "response_canonical_bytes"


def test_scenario_constraint_6825_safe_action_mutation_is_detected(
    source_cases: list[dict],
) -> None:
    """SCENARIO-CONSTRAINT-6825-SAFE-BYTES rejects any safe output rewrite."""

    safe = _case(source_cases, "already_safe_proposals")
    row = attacks.run_attack(safe, "safe_action_mutation", source_cases)
    assert row["applicable"] is True
    assert row["passed"] is True
    assert row["observed"]["baseline_safe_action_identity"] is True
    assert row["observed"]["mutated_safe_action_identity"] is False

    other = _case(source_cases, "fallback")
    not_applicable = attacks.run_attack(other, "safe_action_mutation", source_cases)
    assert not_applicable["applicable"] is False
    assert not_applicable["passed"] is True


@pytest.mark.parametrize(
    "attack_id",
    ["model_label_influence", "exact_valid_label_influence", "future_outcome_leakage"],
)
def test_scenario_constraint_6825_prohibited_features_have_no_influence(
    source_cases: list[dict], attack_id: str
) -> None:
    """SCENARIO-CONSTRAINT-6825-FEATURES strips each denied proposal label."""

    case = _case(source_cases, "soft_conflict")
    row = attacks.run_attack(case, attack_id, source_cases)
    assert row["applicable"] is True
    assert row["passed"] is True
    assert row["observed"]["selection_changed"] is False
    assert row["observed"]["denied_field_removed"] is True


def test_scenario_constraint_6825_certificate_truth_is_local(
    source_cases: list[dict],
) -> None:
    """SCENARIO-CONSTRAINT-6825-CERTIFICATES rejects a later false conflict."""

    case = _case(source_cases, "competing_authorities")
    row = attacks.run_attack(case, "fabricated_certificates", source_cases)
    assert row["passed"] is True
    assert row["observed"]["valid_certificate_accepted"] is True
    assert row["observed"]["fabricated_certificate_accepted"] is False
    assert row["observed"]["first_unsatisfied_obligation"].endswith("_hard")


@pytest.mark.parametrize(
    ("attack_id", "fault"),
    [
        ("row_deletion", "missing_source_case"),
        ("duplicate_rows", "duplicate_source_case"),
        ("row_reorder", "source_case_reorder"),
    ],
)
def test_scenario_constraint_6825_row_integrity_attacks_fail_closed(
    source_cases: list[dict], attack_id: str, fault: str
) -> None:
    """SCENARIO-CONSTRAINT-6825-ROW-INTEGRITY names every roster mutation."""

    row = attacks.run_attack(source_cases[5], attack_id, source_cases)
    assert row["passed"] is True
    assert row["failed_closed"] is True
    assert row["observed"]["integrity_error"] == fault


def test_req_constraint_6825_roster_validator_accepts_only_exact_order(
    source_cases: list[dict],
) -> None:
    """REQ-CONSTRAINT-6825 uses identity and order as a closed roster seal."""

    ids = [row["source_case_id"] for row in source_cases]
    assert attacks.validate_roster(ids, ids) == {"passed": True, "error": None}
    assert attacks.validate_roster(ids, ids[:-1])["error"] == "missing_source_case"
    assert attacks.validate_roster(ids, [*ids, ids[0]])["error"] == "duplicate_source_case"
    assert attacks.validate_roster(ids, [ids[1], ids[0], *ids[2:]])["error"] == (
        "source_case_reorder"
    )
    assert attacks.validate_roster(ids, [*ids, "extra"])["error"] == "extra_source_case"


def test_scenario_constraint_6825_complete_matrix_and_summaries(
    source_cases: list[dict], attack_rows: list[dict]
) -> None:
    """SCENARIO-CONSTRAINT-6825-COMPLETION retains every case and mutation."""

    assert len(attack_rows) == len(source_cases) * len(attacks.ATTACK_IDS) == 768
    assert len({row["row_id"] for row in attack_rows}) == 768
    expected_pairs = {
        (case["source_case_id"], attack_id)
        for case in source_cases
        for attack_id in attacks.ATTACK_IDS
    }
    assert {(row["source_case_id"], row["attack_id"]) for row in attack_rows} == (expected_pairs)
    reduced = attacks.summarize_attack_rows(
        attack_rows,
        [row["source_case_id"] for row in source_cases],
        _fresh_receipt(attack_rows),
    )
    assert reduced["authority_attack_shard_complete"] is True
    assert reduced["hard_authority_supported"] is True
    assert all(row["passed"] for row in reduced["priority_attack_results"])
    assert all(row["byte_identity_enforced"] for row in reduced["safe_action_attack_results"])
    assert all(row["local_conflict_truth"] for row in reduced["certificate_attack_results"])
    assert all(not row["influence_detected"] for row in reduced["prohibited_feature_findings"])
    assert all(row["failed_closed"] for row in reduced["row_integrity_attacks"])


def test_scenario_constraint_6825_completion_is_independent_of_findings(
    source_cases: list[dict], attack_rows: list[dict]
) -> None:
    """SCENARIO-CONSTRAINT-6825-COMPLETION separates coverage from support."""

    changed = deepcopy(attack_rows)
    changed[0]["passed"] = False
    reduced = attacks.summarize_attack_rows(
        changed,
        [row["source_case_id"] for row in source_cases],
        _fresh_receipt(changed),
    )
    assert reduced["authority_attack_shard_complete"] is True
    assert reduced["hard_authority_supported"] is False

    incomplete = attacks.summarize_attack_rows(
        changed[:-1],
        [row["source_case_id"] for row in source_cases],
        _fresh_receipt(changed[:-1]),
    )
    assert incomplete["authority_attack_shard_complete"] is False


def test_req_constraint_6825_fresh_process_replay_is_byte_identical(
    attack_rows: list[dict],
) -> None:
    """REQ-CONSTRAINT-6825 proves deterministic rows in a new CPU process."""

    receipt = attacks.fresh_process_replay(SOURCE_PATHS, attack_rows)
    assert receipt["fresh_process"] is True
    assert receipt["byte_identical"] is True
    assert receipt["rows_sha256"] == receipt["replay_rows_sha256"]


def test_req_constraint_6825_artifact_fields_and_checksum(
    sources: dict[str, dict], source_cases: list[dict], attack_rows: list[dict]
) -> None:
    """REQ-CONSTRAINT-6825 emits each principled field without adoption."""

    receipt = _fresh_receipt(attack_rows)
    first = attacks.build_artifact(
        sources,
        source_paths=SOURCE_PATHS,
        run_date="20260831",
        duration_s=1.25,
        source_cases=source_cases,
        attack_rows=attack_rows,
        fresh_process_receipt=receipt,
    )
    second = attacks.build_artifact(
        sources,
        source_paths=SOURCE_PATHS,
        run_date="20260831",
        duration_s=9.75,
        source_cases=source_cases,
        attack_rows=attack_rows,
        fresh_process_receipt=receipt,
    )
    assert set(attacks.REQUIRED_ARTIFACT_FIELDS).issubset(first)
    assert set(first["field_principles"]) == set(first)
    assert first["inference_substrate"] == attacks.INFERENCE_SUBSTRATE
    assert first["verifier_is_oracle"] is False
    assert first["verdict_class"] == "positive"
    assert first["honest_verdict"].startswith("complete:")
    assert first["authority_attack_shard_complete"] is True
    assert first["hard_authority_supported"] is True
    assert first["adoption_decision"] == "not_evaluated"
    assert first["reproducibility_checksum"] == second["reproducibility_checksum"]
    assert first["source_artifact_hashes"]["exp6813"]["sha256"] == attacks.sha256_file(
        SOURCE_PATHS["exp6813"]
    )
    assert first["independent_attack_harness_id"]["imports_exp6813"] is False
    assert "6813" not in {
        name for name in first["independent_attack_harness_id"]["imported_modules"]
    }


def test_scenario_constraint_6825_blocked_artifact_has_no_rows(
    sources: dict[str, dict],
) -> None:
    """SCENARIO-CONSTRAINT-6825-PRECONDITIONS stops before mutations."""

    changed = deepcopy(sources)
    changed["exp6813"]["selective_arbiter_ab_completed"] = False
    artifact = attacks.build_artifact(
        changed,
        source_paths=SOURCE_PATHS,
        run_date="20260831",
        duration_s=0.25,
    )
    assert artifact["status"] == attacks.BLOCKED_STATUS
    assert artifact["verdict_class"] == "blocked"
    assert artifact["authority_attack_shard_complete"] is False
    assert artifact["rows"] == []
    assert artifact["gate_check_summary"]["failed_checks"] == ["selective_arbiter_ab_completed"]
    assert set(artifact["field_principles"]) == set(artifact)


def test_req_constraint_6825_module_identity_uses_ast_imports() -> None:
    """REQ-CONSTRAINT-6825 proves independence from imports, not substrings."""

    identity = attacks.independent_harness_identity()
    assert identity["imports_exp6813"] is False
    assert identity["path"] == attacks.MODULE_RELATIVE_PATH.as_posix()
    assert identity["sha256"].startswith("sha256:")
    assert "experiment_6813_selective_priority_arbiter_ab" not in inspect.getsource(attacks)


def test_req_constraint_6825_write_and_cli_use_explicit_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CONSTRAINT-6825 writes atomically to a caller-owned test path."""

    output = tmp_path / "artifact.json"
    monkeypatch.setattr(
        attacks,
        "fresh_process_replay",
        lambda paths, rows: _fresh_receipt(rows),
    )
    assert (
        attacks.main(["--date", "20260831", "--root", str(REPO_ROOT), "--output", str(output)]) == 0
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["authority_attack_shard_complete"] is True
    assert payload["run_date"] == "20260831"
    assert not list(tmp_path.glob("*.tmp"))

    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert len(expected) == 64


def test_req_constraint_6825_malformed_public_inputs_fail_closed(
    sources: dict[str, dict], source_cases: list[dict]
) -> None:
    """REQ-CONSTRAINT-6825 covers malformed raw, schema, and binding inputs."""

    assert attacks._valid_raw_receipt({}) is False
    assert (
        attacks._representative_cells(
            {
                "frozen_manifest": {"scenarios": "invalid"},
                "raw_output_manifest": [],
                "rows": [],
            }
        )
        == []
    )

    scenario = sources["exp6812"]["frozen_manifest"]["scenarios"][0]
    receipt = next(
        row
        for row in sources["exp6812"]["raw_output_manifest"]
        if row["scenario_id"] == scenario["scenario_id"] and row["arm"] == "direct_typed"
    )
    one_row = next(
        row for row in sources["exp6812"]["rows"] if row["cell_id"] == receipt["cell_id"]
    )
    assert (
        attacks._representative_cells(
            {
                "frozen_manifest": {"scenarios": [scenario]},
                "raw_output_manifest": [receipt],
                "rows": [one_row],
            }
        )
        == []
    )

    invalid_schema = {"exp6811": {"obligation_schema": {}}, "exp6812": {}}
    with pytest.raises(attacks.AuthorityAttackError, match="schema"):
        attacks.build_source_cases(invalid_schema)
    missing_rows = {
        "exp6811": {"obligation_schema": attacks.EXPECTED_OBLIGATION_SCHEMA},
        "exp6812": {},
    }
    with pytest.raises(attacks.AuthorityAttackError, match="representative"):
        attacks.build_source_cases(missing_rows)

    competing = _case(source_cases, "competing_authorities")
    incomplete = attacks._reference_candidates(competing["scenario"])[0]
    incomplete["parse_state"] = "incomplete"
    assert (
        attacks._evaluate_candidate(
            competing["scenario"], competing["obligation_schema"], incomplete
        )["first_conflict"]
        == "response_schema"
    )
    binding_only = attacks._reference_candidates(competing["scenario"])[0]
    binding_only["authority_chain"] = ["system_authority"]
    binding_evidence = attacks._evaluate_candidate(
        competing["scenario"], competing["obligation_schema"], binding_only
    )
    assert binding_evidence["hard_violation_count"] == 0
    assert binding_evidence["first_conflict"].endswith("_binding")
    with pytest.raises(attacks.AuthorityAttackError, match="unknown attack"):
        attacks.run_attack(competing, "unknown", source_cases)


def test_scenario_constraint_6825_disqualified_and_schema_guards(
    sources: dict[str, dict],
    source_cases: list[dict],
    attack_rows: list[dict],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-CONSTRAINT-6825-COMPLETION retains adverse and invalid results."""

    changed_rows = [dict(row) for row in attack_rows]
    changed_rows[0]["passed"] = False
    adverse = attacks.build_artifact(
        sources,
        source_paths=SOURCE_PATHS,
        run_date="20260831",
        duration_s=1.0,
        source_cases=source_cases,
        attack_rows=changed_rows,
        fresh_process_receipt=_fresh_receipt(changed_rows),
    )
    assert adverse["authority_attack_shard_complete"] is True
    assert adverse["hard_authority_supported"] is False
    assert adverse["verdict_class"] == "disqualified"

    blocked_sources = {
        **sources,
        "exp6813": {**sources["exp6813"], "selective_arbiter_ab_completed": False},
    }
    with monkeypatch.context() as patch:
        patch.setattr(attacks, "REQUIRED_ARTIFACT_FIELDS", ("missing",))
        with pytest.raises(attacks.AuthorityAttackError, match="field set"):
            attacks.build_artifact(
                blocked_sources,
                source_paths=SOURCE_PATHS,
                run_date="20260831",
                duration_s=1.0,
            )
    with monkeypatch.context() as patch:
        principles = dict(attacks.FIELD_PRINCIPLES)
        del principles["adoption_decision"]
        patch.setattr(attacks, "FIELD_PRINCIPLES", principles)
        with pytest.raises(attacks.AuthorityAttackError, match="one principle"):
            attacks.build_artifact(
                blocked_sources,
                source_paths=SOURCE_PATHS,
                run_date="20260831",
                duration_s=1.0,
            )


def test_req_constraint_6825_replay_cli_success_and_block(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-CONSTRAINT-6825 runs fresh replay and blocks an unsealed root."""

    assert attacks.main(["--root", str(REPO_ROOT), "--replay-only"]) == 0
    replay = json.loads(capsys.readouterr().out)
    assert replay["rows_sha256"].startswith("sha256:")
    assert attacks._replay_only(tmp_path) == 2
