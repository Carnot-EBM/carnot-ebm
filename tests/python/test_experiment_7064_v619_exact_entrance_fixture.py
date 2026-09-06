"""Tests for the exact V619 entrance fixture.

Spec refs: REQ-VERIFY-7064,
SCENARIO-VERIFY-7064-OPERANDS, SCENARIO-VERIFY-7064-DIVISION,
SCENARIO-VERIFY-7064-EXHAUSTIVE, SCENARIO-VERIFY-7064-WITNESS,
SCENARIO-VERIFY-7064-GROUPS, SCENARIO-VERIFY-7064-MRV,
SCENARIO-VERIFY-7064-LEAKAGE, SCENARIO-VERIFY-7064-MUTATION.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_7064_v619_exact_entrance_fixture as mod


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def written_fixture(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict[str, Any], Path]:
    """Build the full fixture once because all tests inspect the same frozen rows."""

    work = tmp_path_factory.mktemp("exp7064")
    artifact = mod.write_artifact(
        root=ROOT,
        output_path=work / "fixture.json",
        fixture_dir=work,
        proposal_output_path=work / "proposal-does-not-exist.json",
    )
    return artifact, work


def _rehash(artifact: dict[str, Any]) -> dict[str, Any]:
    artifact["reproducibility_checksum"] = mod.artifact_checksum(artifact)
    return artifact


def test_req_verify_7064_operand_multiplicity_and_commutative_canonicalization() -> None:
    """REQ-VERIFY-7064 and SCENARIO-VERIFY-7064-OPERANDS."""

    entrances = mod.enumerate_legal_entrances((2, 2, 3))
    keys = [(row["operand_pair"], row["operator"]) for row in entrances]

    assert keys.count(([2, 2], "+")) == 1
    assert keys.count(([2, 2], "*")) == 1
    assert keys.count(([2, 3], "+")) == 1
    assert keys.count(([2, 3], "*")) == 1
    assert all(pair == sorted(pair) for pair, _operator in keys)
    assert not any(pair == [3, 3] for pair, _operator in keys)

    residual = mod.apply_entrance(
        (2, 2, 3),
        next(row for row in entrances if row["operand_pair"] == [2, 2] and row["operator"] == "+"),
    )
    assert residual == (3, 4)
    with pytest.raises(ValueError, match="multiplicity"):
        mod.apply_entrance((2, 3), {"operand_pair": [2, 2], "operator": "+"})
    with pytest.raises(ValueError, match="two values"):
        mod.apply_entrance((2, 3), {"operand_pair": [2], "operator": "+"})
    with pytest.raises(ValueError, match="illegal"):
        mod.apply_entrance((2, 3), {"operand_pair": [2, 3], "operator": "/"})
    with pytest.raises(ValueError, match="positive integers"):
        mod.enumerate_legal_entrances((0, 2))


def test_req_verify_7064_legal_division_uses_exact_integer_semantics() -> None:
    """REQ-VERIFY-7064 and SCENARIO-VERIFY-7064-DIVISION."""

    divisible = mod.enumerate_legal_entrances((3, 4, 9))
    division = [row for row in divisible if row["operator"] == "/"]
    assert [
        (row["operand_pair"], row["left"], row["right"], row["result"]) for row in division
    ] == [([3, 9], 9, 3, 3)]

    equal = mod.enumerate_legal_entrances((4, 4))
    equal_division = next(row for row in equal if row["operator"] == "/")
    assert (equal_division["left"], equal_division["right"], equal_division["result"]) == (4, 4, 1)
    assert not any(row["operator"] == "/" for row in mod.enumerate_legal_entrances((3, 8)))


def test_req_verify_7064_exhaustive_entrance_labels_and_witness_replay() -> None:
    """REQ-VERIFY-7064 and SCENARIO-VERIFY-7064-EXHAUSTIVE/WITNESS."""

    unit = {
        "unit_id": "tiny",
        "source_group_id": "tiny-group",
        "split": "calibration",
        "numbers": [2, 3, 6],
        "target": 12,
    }
    rows = mod.label_unit_entrances(unit)
    keys = {(tuple(row["operand_pair"]), row["operator"]) for row in rows}

    assert len(rows) == len(keys) == 11
    assert keys == {
        ((2, 3), "+"),
        ((2, 3), "-"),
        ((2, 3), "*"),
        ((2, 6), "+"),
        ((2, 6), "-"),
        ((2, 6), "*"),
        ((2, 6), "/"),
        ((3, 6), "+"),
        ((3, 6), "-"),
        ((3, 6), "*"),
        ((3, 6), "/"),
    }
    assert any(row["reachable"] for row in rows)
    for row in rows:
        assert row["reachable"] is mod.replay_entrance_witness(unit["numbers"], unit["target"], row)

    reachable = next(row for row in rows if row["reachable"])
    broken = deepcopy(reachable)
    broken["continuation_witness"] = [{"left": 99, "operator": "+", "right": 1, "result": 100}]
    assert mod.replay_entrance_witness(unit["numbers"], unit["target"], broken) is False
    no_witness = deepcopy(reachable)
    no_witness["continuation_witness"] = None
    assert mod.replay_entrance_witness(unit["numbers"], unit["target"], no_witness) is False
    with_steps = next(row for row in rows if row["reachable"] and row["continuation_witness"])
    wrong_result = deepcopy(with_steps)
    wrong_result["continuation_witness"][0]["result"] += 1
    assert mod.replay_entrance_witness(unit["numbers"], unit["target"], wrong_result) is False


def test_req_verify_7064_source_groups_splits_counts_and_hashes(
    written_fixture: tuple[dict[str, Any], Path],
) -> None:
    """REQ-VERIFY-7064 and SCENARIO-VERIFY-7064-GROUPS."""

    artifact, _work = written_fixture
    calibration = set(artifact["split_manifest"]["calibration_source_group_ids"])
    held = set(artifact["split_manifest"]["held_source_group_ids"])
    ordered_ids = [row["unit_id"] for row in artifact["unit_rows"]]

    assert artifact["unit_count"] >= 96
    assert artifact["source_group_count"] >= 12
    assert len(artifact["diversity_subset_ids"]) >= 48
    assert calibration and held and calibration.isdisjoint(held)
    assert {row["source_group_id"] for row in artifact["unit_rows"]} == calibration | held
    assert all(row["reachable_entrance_count"] > 0 for row in artifact["unit_rows"])
    assert all(row["unreachable_entrance_count"] > 0 for row in artifact["unit_rows"])
    assert all(
        row["reachable_family_count"] >= 2
        for row in artifact["unit_rows"]
        if row["unit_id"] in artifact["diversity_subset_ids"]
    )
    assert artifact["split_manifest"]["ordered_unit_ids_hash"] == mod.sha256_json(ordered_ids)
    assert artifact["split_manifest_hash"] == mod.sha256_json(artifact["split_manifest"])
    assert artifact["split_manifest"]["sealed_before_model_output"] is True
    assert artifact["split_manifest"]["proposal_output_absent_at_seal"] is True


def test_req_verify_7064_mrv_is_label_and_witness_blind(
    written_fixture: tuple[dict[str, Any], Path],
) -> None:
    """REQ-VERIFY-7064 and SCENARIO-VERIFY-7064-MRV."""

    artifact, _work = written_fixture
    source = deepcopy(artifact["entrance_rows"][:40])
    expected = mod.build_mrv_rows(source)
    for row in source:
        row["reachable"] = not row["reachable"]
        row["continuation_witness"] = [{"solver_secret": "changed"}]

    assert mod.build_mrv_rows(source) == expected
    assert all(row["label_fields_read"] is False for row in expected)
    assert all(row["residual_domain_sizes"] for row in expected)
    assert mod.mrv_score((2, 3, 6)) == mod.mrv_score((6, 2, 3))


def test_req_verify_7064_model_visible_allowlist_and_hash(
    written_fixture: tuple[dict[str, Any], Path],
) -> None:
    """REQ-VERIFY-7064 and SCENARIO-VERIFY-7064-LEAKAGE."""

    artifact, _work = written_fixture
    allowed = set(mod.MODEL_VISIBLE_FIELDS)
    rows = artifact["model_visible_rows"]

    assert mod.validate_model_visible_rows(rows, artifact["model_visible_schema"])
    assert all(set(row) == allowed for row in rows)
    assert artifact["model_visible_rows_hash"] == mod.sha256_json(rows)
    assert not any(
        marker in mod.canonical_json(rows).lower()
        for marker in ("reachable", "witness", "solver", "source_group", "split", "label")
    )

    leaky = deepcopy(rows)
    leaky[0]["reachable"] = True
    with pytest.raises(ValueError, match="allowlist"):
        mod.validate_model_visible_rows(leaky, artifact["model_visible_schema"])


def test_req_verify_7064_ready_artifact_validates_and_is_circular(
    written_fixture: tuple[dict[str, Any], Path],
) -> None:
    """REQ-VERIFY-7064 requires an oracle-aware ready verdict."""

    artifact, _work = written_fixture
    assert mod.validate_artifact(artifact, check_files=True)
    assert artifact["entrance_fixture_ready_score"] == 1
    assert type(artifact["entrance_fixture_ready_score"]) is int
    assert artifact["verifier_is_oracle"] is True
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["inference_substrate"] == "deterministic_verifier"
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert all(artifact["field_principles"][field] for field in mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["exhaustive_enumerator_receipt"]["all_legal_entrances_labeled"] is True
    assert all(row["witness_valid"] for row in artifact["witness_replay_rows"])
    assert artifact["gate_check_summary"]["passed"] is True
    assert artifact["reproducibility_checksum"] == mod.artifact_checksum(artifact)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda value: value["unit_rows"][0].update(target=999999), "unit row"),
        (
            lambda value: value["entrance_rows"][0].update(
                reachable=not value["entrance_rows"][0]["reachable"]
            ),
            "entrance row",
        ),
        (lambda value: value["split_manifest"].update(held_source_group_ids=[]), "split manifest"),
        (lambda value: value["mrv_rows"][0].update(mrv_score=999), "MRV"),
        (lambda value: value.update(entrance_fixture_ready_score=0), "readiness"),
        (lambda value: value.update(verdict_class="positive"), "oracle"),
    ],
)
def test_req_verify_7064_mutated_artifact_is_rejected(
    written_fixture: tuple[dict[str, Any], Path],
    mutator: Any,
    message: str,
) -> None:
    """REQ-VERIFY-7064 and SCENARIO-VERIFY-7064-MUTATION."""

    artifact, _work = written_fixture
    changed = deepcopy(artifact)
    mutator(changed)
    _rehash(changed)
    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(changed)


def test_req_verify_7064_row_file_mutation_and_replacement_are_rejected(
    written_fixture: tuple[dict[str, Any], Path],
) -> None:
    """REQ-VERIFY-7064 and SCENARIO-VERIFY-7064-MUTATION cover row bytes."""

    artifact, work = written_fixture
    row_path = Path(artifact["fixture_paths"]["unit_rows"]["path"])
    original = row_path.read_bytes()
    try:
        row_path.write_bytes(original + b"{}\n")
        with pytest.raises(ValueError, match="fixture file"):
            mod.validate_artifact(artifact, check_files=True)
    finally:
        row_path.write_bytes(original)

    with pytest.raises(FileExistsError, match="immutable fixture"):
        mod.write_artifact(
            root=ROOT,
            output_path=work / "fixture.json",
            fixture_dir=work,
            proposal_output_path=work / "proposal-does-not-exist.json",
        )


def test_req_verify_7064_failed_precondition_writes_terminal_block(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-7064 preconditions fail closed with exact diagnostics."""

    failed = [
        {
            "check": "deterministic_arithmetic_solver_stack",
            "available": False,
            "expected_value": True,
            "observed_value": "z3 unavailable",
        }
    ]
    output = tmp_path / "blocked.json"
    artifact = mod.write_artifact(
        root=ROOT,
        output_path=output,
        fixture_dir=tmp_path,
        proposal_output_path=tmp_path / "proposal-does-not-exist.json",
        preconditions_override=failed,
    )

    assert output.exists()
    assert artifact["entrance_fixture_ready_score"] == 0
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("complete_blocked_")
    assert artifact["gate_check_summary"] == {
        "passed": False,
        "failed_checks": [
            {
                "failed_check": "deterministic_arithmetic_solver_stack",
                "expected_value": True,
                "observed_value": "z3 unavailable",
            }
        ],
    }
    assert mod.validate_artifact(json.loads(output.read_text(encoding="utf-8")))


def test_req_verify_7064_artifact_schema_and_blocked_guards_fail_closed() -> None:
    """REQ-VERIFY-7064 rejects malformed terminal evidence before row replay."""

    preconditions = [
        {
            "check": "fixture_path",
            "available": False,
            "expected_value": True,
            "observed_value": "not writable",
        }
    ]
    base = mod._blocked_artifact(preconditions, {"all_present": False}, 0.1)

    missing = dict(base)
    missing.pop("rows")
    with pytest.raises(ValueError, match="missing required"):
        mod.validate_artifact(missing)

    no_principle = deepcopy(base)
    no_principle["field_principles"].pop("rows")
    with pytest.raises(ValueError, match="missing field principles"):
        mod.validate_artifact(no_principle)

    wrong_substrate = dict(base, inference_substrate="live_llm_inference")
    with pytest.raises(ValueError, match="inference substrate"):
        mod.validate_artifact(wrong_substrate)

    wrong_oracle = dict(base, verifier_is_oracle=False)
    with pytest.raises(ValueError, match="oracle declaration"):
        mod.validate_artifact(wrong_oracle)

    boolean_score = dict(base, entrance_fixture_ready_score=False)
    with pytest.raises(ValueError, match="bare integer"):
        mod.validate_artifact(boolean_score)

    bad_checksum = dict(base, reproducibility_checksum="sha256:bad")
    with pytest.raises(ValueError, match="checksum"):
        mod.validate_artifact(bad_checksum)

    wrong_block = dict(base, honest_verdict="complete_wrong")
    _rehash(wrong_block)
    with pytest.raises(ValueError, match="blocked verdict"):
        mod.validate_artifact(wrong_block)

    no_failure = deepcopy(base)
    no_failure["gate_check_summary"] = {"passed": True, "failed_checks": []}
    _rehash(no_failure)
    with pytest.raises(ValueError, match="exact failures"):
        mod.validate_artifact(no_failure)

    malformed_failure = deepcopy(base)
    malformed_failure["gate_check_summary"]["failed_checks"][0].pop("observed_value")
    _rehash(malformed_failure)
    with pytest.raises(ValueError, match="failure shape"):
        mod.validate_artifact(malformed_failure)


def test_req_verify_7064_ready_gate_summary_cannot_be_forged(
    written_fixture: tuple[dict[str, Any], Path],
) -> None:
    """REQ-VERIFY-7064 binds the ready class to a passing gate summary."""

    artifact, _work = written_fixture
    changed = dict(artifact)
    changed["gate_check_summary"] = {"passed": False, "failed_checks": []}
    _rehash(changed)
    with pytest.raises(ValueError, match="ready gate summary"):
        mod.validate_artifact(changed)


def test_req_verify_7064_cli_parser_writes_to_explicit_temp_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-7064 exposes the requested dated command without tracked writes."""

    captured: dict[str, Any] = {}

    def fake_write_artifact(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"entrance_fixture_ready_score": 1}

    monkeypatch.setattr(mod, "write_artifact", fake_write_artifact)
    output = tmp_path / "cli.json"
    assert (
        mod.main(["--date", "20260906", "--output", str(output), "--fixture-dir", str(tmp_path)])
        == 0
    )
    assert captured["output_path"] == output
    assert captured["fixture_dir"] == tmp_path
    with pytest.raises(SystemExit, match="execution date"):
        mod.main(["--date", "20260905", "--output", str(output), "--fixture-dir", str(tmp_path)])
