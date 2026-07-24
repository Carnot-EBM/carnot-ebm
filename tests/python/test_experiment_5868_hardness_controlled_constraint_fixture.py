"""Tests for Exp5868 hardness-controlled constraint fixture.

Spec refs: REQ-VERIFY-5868, SCENARIO-VERIFY-5868-GENERATION,
SCENARIO-VERIFY-5868-CERTIFICATES, SCENARIO-VERIFY-5868-RELABELS-AND-CONTROLS,
SCENARIO-VERIFY-5868-REPLAY-AND-BLOCKED.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5868_hardness_controlled_constraint_fixture as mod


REPO = Path(__file__).resolve().parents[2]
VERIFY_SPEC = REPO / "openspec/capabilities/verification/spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5868_hardness_controlled_constraint_fixture.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5868_hardness_controlled_constraint_fixture.py "
    "-m pytest tests/python/test_experiment_5868_hardness_controlled_constraint_fixture.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5868_hardness_controlled_constraint_fixture.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = ".venv/bin/python scripts/check_spec_coverage.py"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5868_hardness_controlled_constraint_fixture.json"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
PROTECTED_FILE_COMMAND = (
    '.venv/bin/python -c "from pathlib import Path; '
    "assert Path('scripts/research_conductor.py').exists()\""
)
TEST_COMMANDS = [
    TEST_COMMAND,
    COVERAGE_COMMAND,
    FULL_TEST_COMMAND,
    SPEC_COMMAND,
    ADVERSARIAL_COMMAND,
    ROOT_CLUTTER_COMMAND,
    PROTECTED_FILE_COMMAND,
]
TEST_EXIT_CODES = {command: 0 for command in TEST_COMMANDS}


def _preconditions(tmp_path: Path) -> dict[str, Any]:
    return mod.collect_preconditions(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        row_file_path=tmp_path / mod.ROW_FILE_RELATIVE_PATH.name,
        memory_probe=lambda: {"available_mb": 32768, "required_mb": 1024, "ok": True},
        disk_probe=lambda root: {"available_mb": 32768, "required_mb": 1024, "ok": True},
    )


@pytest.fixture(scope="module")
def exp5868_fixture(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    """REQ-VERIFY-5868: build the deterministic exact fixture once."""

    base = tmp_path_factory.mktemp("exp5868")
    conductor = REPO / mod.PROTECTED_FILES[0]
    before_hash = mod.sha256_file(conductor)
    artifact = mod.run(
        result_path=base / mod.RESULT_RELATIVE_PATH.name,
        row_file_path=base / mod.ROW_FILE_RELATIVE_PATH.name,
        preconditions_checked=_preconditions(base),
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        duration_s=12.0,
        write=True,
    )
    rows = mod.read_row_file(base / mod.ROW_FILE_RELATIVE_PATH.name)
    assert mod.sha256_file(conductor) == before_hash
    return artifact, rows, base


def test_req_verify_5868_spec_declares_hardness_fixture_contract() -> None:
    """REQ-VERIFY-5868: OpenSpec names every required field and principle."""

    text = VERIFY_SPEC.read_text(encoding="utf-8")
    section = text[text.index("### REQ-VERIFY-5868") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-VERIFY-5868",
        "SCENARIO-VERIFY-5868-GENERATION",
        "SCENARIO-VERIFY-5868-CERTIFICATES",
        "SCENARIO-VERIFY-5868-RELABELS-AND-CONTROLS",
        "SCENARIO-VERIFY-5868-REPLAY-AND-BLOCKED",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.ROW_FILE_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "`hardness_controlled_fixture_ready_score`",
    ):
        assert marker in section
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_req_verify_5868_terminal_artifact_is_hash_bound_and_replayable(
    exp5868_fixture: tuple[dict[str, Any], list[dict[str, Any]], Path],
) -> None:
    """REQ-VERIFY-5868: terminal JSON/JSONL commitments replay exactly."""

    artifact, rows, base = exp5868_fixture
    rerun = mod.run(
        result_path=base / "rerun.json",
        row_file_path=base / "rerun.rows.jsonl",
        preconditions_checked=_preconditions(base / "rerun"),
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        duration_s=99.0,
        write=True,
    )

    assert mod.validate_artifact(artifact) is True
    assert mod.verify_row_file(rows, artifact) is True
    assert json.loads((base / mod.RESULT_RELATIVE_PATH.name).read_text()) == artifact
    assert artifact["status"] == "complete"
    assert artifact["honest_verdict"].startswith("ready:")
    assert artifact["hardness_controlled_fixture_ready_score"] == pytest.approx(1.0)
    assert isinstance(artifact["hardness_controlled_fixture_ready_score"], float)
    assert artifact["duration_s"] == pytest.approx(12.0)
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert artifact["preconditions_checked"]["preconditions_ready"] is True
    assert artifact["source_method_receipt"]["source_id"] == "arxiv:2607.17047"
    assert artifact["row_file_receipt"]["sha256"] == mod.sha256_file(
        base / mod.ROW_FILE_RELATIVE_PATH.name
    )
    assert artifact["row_file_receipt"]["sha256"] == rerun["row_file_receipt"]["sha256"]
    assert artifact["reproducibility_checksum"] == rerun["reproducibility_checksum"]
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(artifact["test_exit_codes"]) == set(artifact["test_commands"])
    assert all(code == 0 for code in artifact["test_exit_codes"].values())
    assert artifact["protected_files_unchanged"]["all_unchanged"] is True

    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert artifact["field_provenance"][field]["principle"] == principle


def test_scenario_verify_5868_generation_matches_family_bins_density_and_length(
    exp5868_fixture: tuple[dict[str, Any], list[dict[str, Any]], Path],
) -> None:
    """SCENARIO-VERIFY-5868-GENERATION: hard/easy families are matched per bin."""

    artifact, rows, _base = exp5868_fixture
    definitions = artifact["family_and_size_bin_definitions"]
    matching = artifact["density_width_and_length_matching"]

    assert definitions["all_bins_have_both_families"] is True
    assert definitions["all_bins_have_both_labels"] is True
    assert matching["all_matching_passed"] is True
    assert matching["max_clause_width_matched"] is True
    assert matching["max_density_delta"] <= mod.DENSITY_TOLERANCE
    assert matching["max_surface_token_delta"] <= mod.LENGTH_TOLERANCE_TOKENS

    canonical_counts = Counter()
    labels_by_family_bin: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    control_counts = Counter(row["control_kind"] for row in rows)
    row_ids = [row["row_id"] for row in rows]
    assert len(row_ids) == len(set(row_ids))
    assert set(control_counts) == set(mod.SURFACE_CONTROL_KINDS)

    for row in rows:
        assert row["schema"] == mod.ROW_SCHEMA
        assert row["row_hash"] == mod.row_hash(row)
        assert row["family"] in mod.FAMILIES
        assert row["size_bin"] in mod.SIZE_BIN_NAMES
        assert row["expected_label"] in mod.LABELS
        assert row["max_clause_width"] <= mod.MAX_CLAUSE_WIDTH
        assert row["canonical_formula_text"].startswith("p cnf ")
        assert row["canonical_formula_hash"].startswith("sha256:")
        assert row["surface_token_count"] == row["target_surface_token_count"]
        assert mod.clause_density(row["clauses"], row["n_vars"]) == pytest.approx(
            row["clause_density"]
        )
        labels_by_family_bin[(row["family"], row["size_bin"])][row["expected_label"]] += 1
        if row["control_kind"] == "canonical":
            canonical_counts[(row["family"], row["size_bin"], row["expected_label"])] += 1

    for family in mod.FAMILIES:
        for size_bin in mod.SIZE_BIN_NAMES:
            assert labels_by_family_bin[(family, size_bin)]["satisfiable"] == labels_by_family_bin[
                (family, size_bin)
            ]["unsatisfiable"]
            for label in mod.LABELS:
                assert canonical_counts[(family, size_bin, label)] == 1


def test_scenario_verify_5868_certificates_and_multi_solver_oracles_agree(
    exp5868_fixture: tuple[dict[str, Any], list[dict[str, Any]], Path],
) -> None:
    """SCENARIO-VERIFY-5868-CERTIFICATES: witnesses and solver configs own labels."""

    artifact, rows, _base = exp5868_fixture
    balance = artifact["label_and_certificate_balance"]
    solver_receipts = artifact["solver_versions_and_oracle_receipts"]
    covariates = artifact["proof_hardness_covariates"]

    assert balance["all_labels_balanced"] is True
    assert balance["all_certificate_checks_passed"] is True
    assert solver_receipts["all_solvers_agree"] is True
    assert solver_receipts["solver_configuration_count"] >= 2
    assert covariates["conflict_count_is_label"] is False
    assert covariates["time_covariate_is_label"] is False

    saw_sat = saw_unsat = False
    for row in rows:
        labels = {result["label"] for result in row["solver_results"].values()}
        assert labels == {row["expected_label"]}
        assert row["solver_disagreement"] is False
        assert row["solver_timeout"] is False
        assert row["certificate"]["validated"] is True
        assert mod.validate_certificate(row) is True
        assert row["proof_hardness_covariates"]["solver_conflicts"] >= 0
        assert row["proof_hardness_covariates"]["deterministic_time_proxy_s"] >= 0.0
        assert row["proof_hardness_covariates"]["used_as_label"] is False
        if row["expected_label"] == "satisfiable":
            saw_sat = True
            assert row["certificate"]["kind"] == "satisfying_assignment"
        else:
            saw_unsat = True
            assert row["certificate"]["kind"] == "tseitin_parity_contradiction"
    assert saw_sat and saw_unsat


def test_scenario_verify_5868_relabels_and_controls_preserve_truth(
    exp5868_fixture: tuple[dict[str, Any], list[dict[str, Any]], Path],
) -> None:
    """SCENARIO-VERIFY-5868-RELABELS-AND-CONTROLS: surface controls fail closed."""

    artifact, rows, _base = exp5868_fixture
    relabels = artifact["proof_preserving_relabel_receipts"]
    controls = artifact["surface_and_no_information_controls"]

    assert relabels["all_relabel_checks_passed"] is True
    assert controls["all_controls_present"] is True
    assert controls["all_control_labels_preserved"] is True
    assert controls["no_information_surface_token_count"] > 0

    for row in rows:
        receipt = row["proof_preserving_relabel"]
        assert receipt["label_preserved"] is True
        assert receipt["certificate_preserved"] is True
        assert receipt["relabel_formula_hash"].startswith("sha256:")
        assert mod.validate_relabel_receipt(row) is True
        if row["control_kind"] == "variable_renaming":
            assert row["surface_formula_hash"] == receipt["relabel_formula_hash"]
            assert row["surface_formula_hash"] != row["canonical_formula_hash"]
        if row["control_kind"] == "density_mismatched":
            assert row["clause_count"] > row["canonical_clause_count"]
            assert row["expected_label"] == row["canonical_expected_label"]
        if row["control_kind"] == "no_information":
            tokens = set(row["surface_formula_text"].split())
            assert tokens.issubset(set(mod.NO_INFORMATION_TOKENS))


def test_scenario_verify_5868_fail_closed_for_missing_inputs_and_tampering(
    tmp_path: Path,
    exp5868_fixture: tuple[dict[str, Any], list[dict[str, Any]], Path],
) -> None:
    """SCENARIO-VERIFY-5868-REPLAY-AND-BLOCKED: bad evidence cannot look ready."""

    blocked = mod.run(
        root=tmp_path,
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        row_file_path=tmp_path / mod.ROW_FILE_RELATIVE_PATH.name,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        duration_s=0.0,
        write=True,
    )
    assert blocked["status"] == "blocked"
    assert blocked["honest_verdict"].startswith("blocked:")
    assert blocked["hardness_controlled_fixture_ready_score"] == 0.0
    assert "missing_source_paper_receipt" in blocked["preconditions_checked"]["blocked_reasons"]
    assert blocked["row_file_receipt"]["row_count"] == 0

    artifact, rows, _base = exp5868_fixture
    tampered = deepcopy(artifact)
    tampered["test_exit_codes"][TEST_COMMAND] = 1
    assert mod.hardness_controlled_fixture_ready_score(tampered) == 0.0
    with pytest.raises(ValueError, match="hardness_controlled_fixture_ready_score"):
        mod.validate_artifact(tampered)

    duplicate_rows = [deepcopy(rows[0]), deepcopy(rows[0])]
    with pytest.raises(ValueError, match="duplicate_row_id"):
        mod.verify_rows(duplicate_rows)

    bad_certificate = deepcopy(rows[0])
    bad_certificate["certificate"]["validated"] = False
    assert mod.label_and_certificate_balance([bad_certificate])["all_certificate_checks_passed"] is False
    assert mod.rows_to_jsonl([]) == ""
    assert mod.read_row_file(tmp_path / "missing.rows.jsonl") == []


def test_scenario_verify_5868_defensive_branches_are_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exp5868_fixture: tuple[dict[str, Any], list[dict[str, Any]], Path],
) -> None:
    """SCENARIO-VERIFY-5868-REPLAY-AND-BLOCKED: defensive checks name failures."""

    artifact, rows, base = exp5868_fixture
    row = deepcopy(rows[0])

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object required"):
        mod._read_json(bad_json)

    bad_jsonl = tmp_path / "bad.rows.jsonl"
    bad_jsonl.write_text("\n[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSONL object required"):
        mod.read_row_file(bad_jsonl)
    blank_jsonl = tmp_path / "blank.rows.jsonl"
    blank_jsonl.write_text("\n" + mod.rows_to_jsonl([row]), encoding="utf-8")
    assert len(mod.read_row_file(blank_jsonl)) == 1

    assert mod._source_receipt_text("no matching paper") == ""
    corrupt_root = tmp_path / "corrupt"
    (corrupt_root / mod.EXP5840_ARTIFACT_RELATIVE_PATH.parent).mkdir(parents=True)
    (corrupt_root / mod.EXP5840_ARTIFACT_RELATIVE_PATH).write_text("{", encoding="utf-8")
    (corrupt_root / mod.EXP5840_ROWS_RELATIVE_PATH).write_text("{}", encoding="utf-8")
    assert mod._exp5840_receipt(corrupt_root)["ok"] is False

    assert sum(mod.charges_for_label(3, "satisfiable", 1)) % 2 == 0
    with pytest.raises(ValueError, match="unknown_family"):
        mod.graph_for_family("bad_family", 4)
    with pytest.raises(ValueError, match="unknown_label"):
        mod.charges_for_label(4, "bad_label", 1)
    with pytest.raises(ValueError, match="unknown_solver_config"):
        mod.solve_cnf_dpll([[1]], 1, config="bad")
    with pytest.raises(ValueError, match="no_branch_variable"):
        mod._choose_branch_variable((), {1: True}, 1, mod.SOLVER_CONFIGS[0])
    assert mod.solve_cnf_dpll([[1]], 1, config=mod.SOLVER_CONFIGS[0], max_decisions=-1)[
        "timeout"
    ] is True
    with pytest.raises(ValueError, match="surface_text_exceeds"):
        mod._pad_to_target_tokens("too many tokens", 1)
    assert mod._certificate_for_instance(
        label="satisfiable",
        clauses=[[1]],
        vertices=1,
        edges=[],
        charges=[0],
        assignment=None,
    )["validated"] is False
    invalid_label_row = deepcopy(row)
    invalid_label_row["expected_label"] = "invalid"
    invalid_label_row["certificate"]["validated"] = True
    assert mod.validate_certificate(invalid_label_row) is False

    original_gf2_solution = mod._gf2_solution
    monkeypatch.setattr(mod, "_gf2_solution", lambda *_args: None)
    with pytest.raises(ValueError, match="label_generation_mismatch"):
        mod._base_instance("expander_tseitin", mod.SIZE_BINS[0], "satisfiable")
    monkeypatch.setattr(mod, "_gf2_solution", original_gf2_solution)

    for field, expected in (
        ("schema", "row_schema"),
        ("row_hash", "row_hash"),
        ("certificate", "certificate"),
        ("solver_results", "solver_label"),
        ("solver_disagreement", "solver_status"),
        ("proof_preserving_relabel", "relabel"),
    ):
        tampered = deepcopy(row)
        if field == "schema":
            tampered[field] = "bad"
        elif field == "row_hash":
            tampered[field] = "sha256:bad"
        elif field == "certificate":
            tampered[field]["validated"] = False
            tampered["row_hash"] = mod.row_hash(tampered)
        elif field == "solver_results":
            first = next(iter(tampered[field]))
            tampered[field][first]["label"] = "unsatisfiable" if row["expected_label"] == "satisfiable" else "satisfiable"
            tampered["row_hash"] = mod.row_hash(tampered)
        elif field == "solver_disagreement":
            tampered[field] = True
            tampered["row_hash"] = mod.row_hash(tampered)
        else:
            tampered[field]["label_preserved"] = False
            tampered["row_hash"] = mod.row_hash(tampered)
        with pytest.raises(ValueError, match=expected):
            mod.verify_rows([tampered])

    bad_artifact = deepcopy(artifact)
    bad_artifact["row_file_receipt"]["path"] = "bad"
    with pytest.raises(ValueError, match="row_file_receipt"):
        mod.verify_row_file(rows, bad_artifact)
    bad_artifact = deepcopy(artifact)
    bad_artifact["row_file_receipt"]["row_count"] = 1
    with pytest.raises(ValueError, match="row_count"):
        mod.verify_row_file(rows, bad_artifact)
    bad_artifact = deepcopy(artifact)
    bad_artifact["row_file_receipt"]["row_hashes"][rows[0]["row_id"]] = "sha256:bad"
    with pytest.raises(ValueError, match="row_hash_receipt"):
        mod.verify_row_file(rows, bad_artifact)
    bad_artifact = deepcopy(artifact)
    bad_artifact["row_file_receipt"]["sha256"] = "sha256:bad"
    with pytest.raises(ValueError, match="row_file_sha256"):
        mod.verify_row_file(rows, bad_artifact)

    width_row = deepcopy(row)
    width_row["canonical_max_clause_width"] = 2
    peer_width_row = deepcopy(row)
    peer_width_row["family"] = "ladder_tseitin"
    peer_width_row["canonical_max_clause_width"] = 2
    assert mod.density_width_and_length_matching([width_row, peer_width_row])["width_failures"]
    solver_bad = deepcopy(row)
    first_solver = next(iter(solver_bad["solver_results"]))
    solver_bad["solver_results"][first_solver]["label"] = "bad"
    assert mod.solver_versions_and_oracle_receipts([solver_bad])["label_failure_count"] == 1
    relabel_bad = deepcopy(row)
    relabel_bad["proof_preserving_relabel"]["label_preserved"] = False
    assert mod.validate_relabel_receipt(relabel_bad) is False
    control_bad = deepcopy(row)
    control_bad["canonical_expected_label"] = "bad"
    assert mod.surface_and_no_information_controls([control_bad])["control_label_failure_count"] == 1

    substrate_bad = deepcopy(artifact)
    substrate_bad["inference_substrate"] = "bad"
    substrate_bad["verifier_is_oracle"] = False
    reasons = mod.blocked_reasons(substrate_bad)
    assert "inference_substrate" in reasons
    assert "verifier_is_oracle" in reasons

    with pytest.raises(ValueError, match="missing_fields"):
        mod.validate_artifact({})
    checksum_bad = deepcopy(artifact)
    checksum_bad["honest_verdict"] = "ready: edited"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(checksum_bad)
    status_bad = deepcopy(artifact)
    status_bad["status"] = "blocked"
    status_bad["reproducibility_checksum"] = mod.reproducibility_checksum(status_bad)
    with pytest.raises(ValueError, match="status"):
        mod.validate_artifact(status_bad)
    verdict_bad = deepcopy(artifact)
    verdict_bad["honest_verdict"] = "blocked: edited"
    verdict_bad["reproducibility_checksum"] = mod.reproducibility_checksum(verdict_bad)
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(verdict_bad)

    nonwrite = mod.run(
        result_path=base / "nonwrite.json",
        row_file_path=base / "nonwrite.rows.jsonl",
        preconditions_checked=_preconditions(base / "nonwrite"),
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        duration_s=5.0,
        write=False,
    )
    assert nonwrite["hardness_controlled_fixture_ready_score"] == 1.0
