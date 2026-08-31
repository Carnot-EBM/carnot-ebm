"""Tests for the independent real-output fixed-point cold audit.

Spec refs: REQ-VERIFY-6801 and SCENARIO-VERIFY-6801-*.
"""

from __future__ import annotations

import ast
import base64
from copy import deepcopy
import json
from pathlib import Path
import zlib

import pytest

from carnot import experiment_6801_real_output_fixed_point_cold_audit as mod


@pytest.fixture(scope="module")
def sources() -> dict:
    """REQ-VERIFY-6801 loads frozen bytes through the cold loader."""

    return mod.load_sources(mod.REPO_ROOT)


@pytest.fixture(scope="module")
def authority(sources: dict) -> dict:
    """SCENARIO-VERIFY-6801-INDEPENDENT-EXACT rebuilds graph authority once."""

    return mod.rebuild_graph_authority(sources["exp6799"])


@pytest.fixture(scope="module")
def audited_rows(sources: dict, authority: dict) -> list[dict]:
    """REQ-VERIFY-6801 recomputes each source row once for focused tests."""

    return mod.audit_source_rows(sources["exp6800"], authority)


@pytest.fixture(scope="module")
def artifact(sources: dict) -> dict:
    """REQ-VERIFY-6801 builds one complete terminal audit for all tests."""

    return mod.build_artifact(
        sources,
        repo_root=mod.REPO_ROOT,
        run_date="20260831",
        duration_s=0.5,
        bootstrap_resamples=mod.BOOTSTRAP_RESAMPLES,
    )


def test_req_verify_6801_spec_precedes_implementation() -> None:
    """REQ-VERIFY-6801 owns the exact, cluster, isolation, control, and block rules."""

    text = (mod.REPO_ROOT / mod.SPEC_PATH).read_text(encoding="utf-8")
    start = text.index("### REQ-VERIFY-6801")
    section = text[start : text.index("### SCENARIO-VERIFY-6745-DUAL", start)]
    for anchor in (
        "REQ-VERIFY-6801",
        "SCENARIO-VERIFY-6801-INDEPENDENT-EXACT",
        "SCENARIO-VERIFY-6801-CLUSTERED",
        "SCENARIO-VERIFY-6801-ISOLATION",
        "SCENARIO-VERIFY-6801-CONTROLS",
        "SCENARIO-VERIFY-6801-BLOCKED",
        "complete_blocked_real_output_fixed_point_audit",
        "model_output_fixed_point_audit_completed",
    ):
        assert anchor in section


def test_scenario_verify_6801_uses_only_standard_library_and_no_producer_import() -> None:
    """SCENARIO-VERIFY-6801-INDEPENDENT-EXACT forbids producer and numeric imports."""

    tree = ast.parse((mod.REPO_ROOT / mod.MODULE_PATH).read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert not any("experiment_6800" in name for name in imports)
    assert not any(name.split(".")[0] in {"numpy", "scipy", "torch", "jax"} for name in imports)


def test_scenario_verify_6801_preconditions_fail_closed(sources: dict) -> None:
    """SCENARIO-VERIFY-6801-BLOCKED checks hashes, rows, pairs, strata, and receipts."""

    clean = mod.check_preconditions(sources, repo_root=mod.REPO_ROOT)
    assert clean["all_passed"] is True
    assert clean["failed_checks"] == []

    mutations = {
        "comparison_completed": lambda value: value["exp6800"].__setitem__(
            "model_output_fixed_point_comparison_completed", False
        ),
        "planned_rows": lambda value: value["exp6800"]["rows"].pop(),
        "unique_pairs": lambda value: value["exp6800"]["rows"][1].__setitem__(
            "paired_key", "wrong-pair"
        ),
        "complete_strata": lambda value: value["exp6800"]["rows"][0].__setitem__(
            "transformation", "wrong-label"
        ),
        "frozen_training_receipts": lambda value: value["exp6800"][
            "training_isolation_receipts"
        ][0].__setitem__("training_source", "exp6799"),
    }
    for expected, mutate in mutations.items():
        changed = deepcopy(sources)
        mutate(changed)
        failed = {row["check"] for row in mod.check_preconditions(changed, repo_root=mod.REPO_ROOT)["failed_checks"]}
        assert expected in failed


def test_scenario_verify_6801_rebuilds_cnf_semantics_and_operation_labels(
    sources: dict, authority: dict
) -> None:
    """SCENARIO-VERIFY-6801-INDEPENDENT-EXACT enumerates without valid-set trust."""

    first_group = sources["exp6799"]["probe_groups"][0]
    record = first_group["graphs"]["base"]
    replay = mod.enumerate_graph(record["graph"])
    assert replay["valid_assignments"]
    assert replay["valid_set_hash"] == record["valid_set_hash"]
    assert replay["graph_hash"] == record["graph_hash"]
    assert replay["solution_count"] == record["solution_count"]

    valid = replay["valid_assignments"][0]
    checked = mod.evaluate_assignment(record["graph"], valid, replay["valid_assignments"])
    assert checked["exact_valid"] is True
    assert checked["dependency_violation_count"] == 0
    assert checked["distance_to_nearest_valid"] == 0

    invalid = deepcopy(valid)
    invalid.pop(next(iter(invalid)))
    checked = mod.evaluate_assignment(record["graph"], invalid, replay["valid_assignments"])
    assert checked["local_checks_passed"] is False
    assert checked["exact_valid"] is False

    assert len(authority["receipts"]) == mod.PROBE_GRAPH_COUNT
    assert authority["mismatches"] == []
    assert authority["operation_label_violations"] == [
        "gemma4_26b_middle_moe|exp6744-ladder_tseitin-small-sat-674403-base:"
        "restructuring_semantics"
    ]
    with pytest.raises(mod.AuditInputError, match="variables"):
        mod.enumerate_graph({"variables": [], "clauses": []})


def test_scenario_verify_6801_rejects_malformed_or_conflicting_authority(
    sources: dict, authority: dict, audited_rows: list[dict], tmp_path: Path
) -> None:
    """SCENARIO-VERIFY-6801-INDEPENDENT-EXACT rejects malformed cold inputs."""

    malformed = (
        ({"variables": ["x1"], "clauses": []}, "clauses"),
        ({"variables": ["wrong"], "clauses": [[1]]}, "xN"),
        ({"variables": ["x1"], "clauses": [[]]}, "nonempty"),
        ({"variables": ["x1"], "clauses": [[2]]}, "outside"),
    )
    for graph, message in malformed:
        with pytest.raises(mod.AuditInputError, match=message):
            mod.enumerate_graph(graph)

    nonobject = tmp_path / "list.json"
    nonobject.write_text("[]", encoding="utf-8")
    with pytest.raises(mod.AuditInputError, match="JSON root"):
        mod.load_json_object(nonobject)
    assert mod.digest_file(tmp_path / "missing.json") is None

    group = deepcopy(sources["exp6799"]["probe_groups"][0])
    group["graphs"]["refinement"] = deepcopy(group["graphs"]["base"])
    group["graphs"]["refinement"]["operation_class"] = "refinement"
    group["graphs"]["refinement"]["solution_count"] = -1
    conflicted = mod.rebuild_graph_authority({"probe_groups": [group], "rows": []})
    assert any("refinement_semantics" in value for value in conflicted["operation_label_violations"])
    assert any("graph_hash_collision" in value for value in conflicted["mismatches"])

    missing = deepcopy(authority)
    missing["records"] = {}
    with pytest.raises(mod.AuditInputError, match="missing graph authority"):
        mod.audit_source_rows({"rows": [audited_rows[0]]}, missing)


def test_req_verify_6801_numeric_and_pairing_edge_cases(audited_rows: list[dict]) -> None:
    """REQ-VERIFY-6801 handles percentile boundaries and refuses incomplete pairs."""

    with pytest.raises(ValueError, match="nonempty"):
        mod.percentile([], 0.5)
    with pytest.raises(ValueError, match="quantile"):
        mod.percentile([1.0], 2.0)
    assert mod.percentile([2.0], 0.5) == 2.0
    assert mod._paired_values([audited_rows[0]]) == []
    assert mod._compare_values({"a": 1}, {"b": 1})[0] == ["a", "b"]
    assert mod._compare_values([1], [1, 2])[0] == [""]


def test_req_verify_6801_recomputes_every_source_row(
    audited_rows: list[dict], sources: dict
) -> None:
    """REQ-VERIFY-6801 recomputes candidate labels, distance, support, and work."""

    assert len(audited_rows) == mod.PLANNED_SOURCE_ROW_COUNT
    assert len({row["source_row_id"] for row in audited_rows}) == mod.PLANNED_SOURCE_ROW_COUNT
    assert all(row["candidate_hashes_match"] for row in audited_rows)
    assert all(row["source_exact_outcomes_match"] for row in audited_rows)
    assert all(row["candidate_hashes_unchanged"] for row in audited_rows)
    assert all(row["candidate_budget"] == 3 for row in audited_rows)
    assert sum(row["candidate_work"] for row in audited_rows) == sum(
        row["candidate_work"] for row in sources["exp6800"]["rows"]
    )
    assert {row["arm"] for row in audited_rows} == set(mod.ARMS)


def test_scenario_verify_6801_clusters_underlying_cases_and_interaction(
    audited_rows: list[dict], sources: dict
) -> None:
    """SCENARIO-VERIFY-6801-CLUSTERED retains models and seeds inside 36 cases."""

    reduced = mod.aggregate_rows(
        audited_rows,
        resamples=mod.BOOTSTRAP_RESAMPLES,
        seed=mod.BOOTSTRAP_SEED,
    )
    assert reduced["paired_exact_valid_deltas"]["paired_key_count"] == mod.PAIRED_KEY_COUNT
    assert all(
        value["case_count"] == mod.UNDERLYING_CASE_COUNT
        for value in reduced["clustered_confidence_intervals"].values()
    )
    interaction = reduced["transformation_interaction"]
    assert interaction["contrast"] == "restructuring_minus_refinement"
    assert interaction["case_count"] == mod.UNDERLYING_CASE_COUNT
    assert interaction["resamples"] == mod.BOOTSTRAP_RESAMPLES
    assert reduced["work_matching"]["candidate_totals_by_arm"] == sources["exp6800"][
        "work_matching"
    ]["candidate_totals_by_arm"]


def test_scenario_verify_6801_proves_training_isolation(sources: dict) -> None:
    """SCENARIO-VERIFY-6801-ISOLATION allows only Exp6786 train IDs."""

    result = mod.verify_training_isolation(sources, repo_root=mod.REPO_ROOT)
    assert result["passed"] is True
    assert result["oracle_feature_violations"] == []
    assert result["receipt_count"] == 10
    assert result["checkpoint"]["payload_hashes_match"] is True
    assert result["training_unit_ids"] == result["expected_exp6786_train_unit_ids"]

    changed = deepcopy(sources)
    changed["exp6800"]["training_isolation_receipts"][0]["exp6799_fields_seen"] = [
        "valid_assignments"
    ]
    attacked = mod.verify_training_isolation(changed, repo_root=mod.REPO_ROOT)
    assert attacked["passed"] is False
    assert attacked["oracle_feature_violations"]


def test_scenario_verify_6801_rejects_checkpoint_and_oracle_attacks(sources: dict) -> None:
    """SCENARIO-VERIFY-6801-ISOLATION detects every frozen leak and receipt failure."""

    def payload(value: object) -> dict:
        raw = mod.canonical_json(value).encode("utf-8")
        return {
            "encoding": "zlib_base64_canonical_json",
            "data": base64.b64encode(zlib.compress(raw)).decode("ascii"),
            "row_sha256": mod.digest_value(value),
        }

    with pytest.raises(mod.AuditInputError, match="encoding"):
        mod._decode_checkpoint_payload({"encoding": "wrong"})
    with pytest.raises(mod.AuditInputError, match="object"):
        mod._decode_checkpoint_payload(payload([]))
    wrong_hash = payload({"a": 1})
    wrong_hash["row_sha256"] = "sha256:wrong"
    with pytest.raises(mod.AuditInputError, match="hash"):
        mod._decode_checkpoint_payload(wrong_hash)

    changed = deepcopy(sources)
    source = changed["exp6800"]
    source["training_isolation_receipts"] = [source["training_isolation_receipts"][0]]
    receipt = source["training_isolation_receipts"][0]
    receipt["training_source"] = "wrong"
    receipt["train_unit_ids"] = []
    source["feature_allowlist"].append("source_model")
    definition = next(iter(source["frozen_arm_definitions"].values()))
    definition["legal_observations"].append("source_case_id")
    definition["training_source"] = "wrong"
    definition["transfer_updates"] = 1
    definition["decoder"] = "oracle"
    source["rows"] = deepcopy(source["rows"][:2])
    source["rows"][1]["proposal_input_hash"] = "sha256:row-identity"

    original = deepcopy(source["rows"][0])

    def envelope(row: dict, *, row_id: str | None = None) -> dict:
        value = payload(row)
        return {
            "row_id": row_id or row["row_id"],
            "payload": value,
            "payload_hash": mod.checkpoint_payload_digest(value),
            "start_receipt": {"unit_id": row["unit_id"]},
            "end_receipt": {"candidate_hashes": row["candidate_hashes"]},
        }

    invalid = envelope(original)
    invalid["payload"]["encoding"] = "wrong"
    bad_payload_hash = envelope(original)
    bad_payload_hash["payload_hash"] = "sha256:wrong"
    mismatch_row = deepcopy(original)
    mismatch_row["runtime_s"] += 1.0
    mismatch = envelope(mismatch_row)
    bad_start = envelope(original)
    bad_start["start_receipt"] = {"unit_id": "wrong"}
    bad_end = envelope(original)
    bad_end["end_receipt"] = {"candidate_hashes": []}
    oracle_row = deepcopy(original)
    oracle_row["exact_evaluation_receipt"]["model_feedback_applied"] = True
    oracle = envelope(oracle_row)
    changed["checkpoint"]["rows"] = [
        invalid,
        bad_payload_hash,
        mismatch,
        bad_start,
        bad_end,
        oracle,
    ]
    attacked = mod.verify_training_isolation(changed, repo_root=mod.REPO_ROOT)
    assert attacked["passed"] is False
    violations = "\n".join(attacked["oracle_feature_violations"])
    for marker in (
        "training_source",
        "train_unit_ids",
        "feature_allowlist",
        "frozen_arm",
        "proposal_input_hash_depends_on_row_identity",
        "unknown checkpoint payload encoding",
        "payload_hash",
        "payload_mismatch",
        "start_receipt",
        "end_receipt",
        "oracle_order",
    ):
        assert marker in violations


def test_scenario_verify_6801_surface_relabel_detects_changed_outcome(
    audited_rows: list[dict], authority: dict
) -> None:
    """SCENARIO-VERIFY-6801-CONTROLS detects a surface-control contradiction."""

    changed = deepcopy(audited_rows[0])
    changed["exact_outcomes"][0]["exact_valid"] = not changed["exact_outcomes"][0][
        "exact_valid"
    ]
    result = mod._surface_relabel_check([changed], authority)
    assert result["mismatch_count"] == 1
    assert result["semantics_preserved"] is False


def test_scenario_verify_6801_runs_all_destructive_controls(
    audited_rows: list[dict], sources: dict, authority: dict
) -> None:
    """SCENARIO-VERIFY-6801-CONTROLS separates eight shortcut classes."""

    first = mod.run_destructive_controls(
        audited_rows,
        source_6800=sources["exp6800"],
        authority=authority,
    )
    second = mod.run_destructive_controls(
        audited_rows,
        source_6800=sources["exp6800"],
        authority=authority,
    )
    assert first == second
    assert set(first["results"]) == set(mod.CONTROLS)
    assert len(first["rows"]) == len(mod.CONTROLS)
    assert first["results"]["identical_arm"]["paired_exact_valid_delta"] == 0.0
    assert first["results"]["injected_aggregate_contradiction"]["detected"] is True
    assert first["results"]["solution_preserving_surface_relabel"]["semantics_preserved"] is True
    assert first["results"]["transformation_label_swap"]["mislabel_detected"] is True
    assert first["results"]["model_id_permutation"]["global_effect_unchanged"] is True


def test_req_verify_6801_artifact_is_row_derived_and_terminal(artifact: dict) -> None:
    """REQ-VERIFY-6801 emits all required evidence and a closed verdict."""

    assert artifact["model_output_fixed_point_audit_completed"] is True
    assert len(artifact["rows"]) == mod.PLANNED_AUDIT_ROW_COUNT
    assert artifact["field_principles"].keys() == artifact.keys()
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["independent_evaluator_hash"].startswith("sha256:")
    assert artifact["reproducibility_checksum"].startswith("sha256:")
    assert artifact["verifier_is_oracle"] is False
    assert artifact["verdict_class"] == "disqualified"
    assert artifact["source_verdict_supported"] is False
    assert artifact["honest_verdict"].startswith(mod.TERMINAL_PREFIXES)
    assert artifact["gate_check_summary"]["all_passed"] is True
    assert mod.validate_artifact(artifact) == []

    changed = deepcopy(artifact)
    changed["exact_recomputed_metrics"]["work_matching"]["candidate_totals_by_arm"][
        mod.GROUPED_ARM
    ] += 1
    changed["clustered_confidence_intervals"]["base"]["lower"] -= 1.0
    findings = mod.validate_artifact(changed)
    assert "row-derived metrics mismatch" in findings
    assert "row-derived intervals mismatch" in findings


def test_req_verify_6801_verdict_branches_are_fail_closed(
    sources: dict, audited_rows: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-VERIFY-6801 maps cold gates to null, positive, partial, or rejection."""

    gates = {
        "all_passed": True,
        "source_artifact_hashes": {},
        "grid_observation": {},
    }
    control_results = {
        mod.SURFACE_RELABEL: {"semantics_preserved": True},
        mod.TRANSFORMATION_LABEL_SWAP: {"mislabel_detected": True},
        mod.MODEL_ID_PERMUTATION: {"global_effect_unchanged": True},
        mod.IDENTICAL_ARM: {"paired_exact_valid_delta": 0.0},
        mod.AGGREGATE_CONTRADICTION: {"detected": True},
        mod.GROUP_PERMUTATION: {"all_source_control_outcomes_recomputed": True},
        mod.EDGE_DELETION: {"all_source_control_outcomes_recomputed": True},
        mod.DUPLICATE_CASE_REMOVAL: {},
    }
    monkeypatch.setattr(mod, "check_preconditions", lambda *_args, **_kwargs: gates)
    monkeypatch.setattr(
        mod,
        "rebuild_graph_authority",
        lambda *_args: {"mismatches": [], "operation_label_violations": []},
    )
    monkeypatch.setattr(mod, "audit_source_rows", lambda *_args: [audited_rows[0]])
    monkeypatch.setattr(
        mod,
        "verify_training_isolation",
        lambda *_args, **_kwargs: {"passed": True, "oracle_feature_violations": []},
    )
    monkeypatch.setattr(
        mod,
        "run_destructive_controls",
        lambda *_args, **_kwargs: {"results": control_results, "rows": []},
    )
    monkeypatch.setattr(mod, "validate_artifact", lambda _artifact: [])

    def invoke(*, positive: bool, headlines_match: bool) -> dict:
        lower = 0.1 if positive else -0.1
        aggregates = {
            "clustered_confidence_intervals": {
                "base": {"lower": lower},
                "refinement": {"lower": lower},
                "restructuring": {"lower": lower},
            },
            "transformation_interaction": {},
            "support_contraction": {},
            "convergence_harm": {},
            "work_matching": {
                "planned_budgets_match": True,
                "no_grouped_work_harm": True,
            },
        }
        monkeypatch.setattr(mod, "aggregate_rows", lambda *_args, **_kwargs: aggregates)
        monkeypatch.setattr(
            mod,
            "headline_differences",
            lambda *_args: {"all_match": headlines_match},
        )
        return mod.build_artifact(
            sources,
            repo_root=mod.REPO_ROOT,
            run_date="20260831",
            duration_s=0.0,
            bootstrap_resamples=1,
        )

    assert invoke(positive=False, headlines_match=True)["verdict_class"] == "null"
    assert invoke(positive=True, headlines_match=True)["verdict_class"] == "positive"
    assert invoke(positive=False, headlines_match=False)["verdict_class"] == "partial"
    monkeypatch.setattr(mod, "validate_artifact", lambda _artifact: ["injected invalidity"])
    with pytest.raises(ValueError, match="injected invalidity"):
        invoke(positive=False, headlines_match=True)


def test_req_verify_6801_validator_rejects_schema_and_blocked_mutations(artifact: dict) -> None:
    """REQ-VERIFY-6801 validator rejects malformed full and blocked artifacts."""

    malformed = deepcopy(artifact)
    malformed.pop("schema")
    malformed["field_principles"] = {}
    malformed["inference_substrate"] = "LLM"
    malformed["random_seed"] = -1
    malformed["verifier_is_oracle"] = True
    malformed["verdict_class"] = "invented"
    malformed["honest_verdict"] = "unfinished"
    malformed["duration_s"] = -1
    malformed["rows"].pop()
    findings = mod.validate_artifact(malformed)
    for marker in (
        "required field set mismatch",
        "field principle coverage mismatch",
        "inference substrate mismatch",
        "random seed mismatch",
        "verifier_is_oracle must be false",
        "verdict class outside closed enum",
        "honest verdict lacks terminal prefix",
        "duration_s must be non-negative",
        "audit row count mismatch",
    ):
        assert marker in findings

    blocked = deepcopy(artifact)
    blocked["model_output_fixed_point_audit_completed"] = False
    blocked["status"] = "wrong"
    blocked["gate_check_summary"]["all_passed"] = True
    blocked["reproducibility_checksum"] = mod.reproducibility_checksum(blocked)
    findings = mod.validate_artifact(blocked)
    assert "blocked status mismatch" in findings
    assert "blocked artifact must not contain rows" in findings
    assert "blocked artifact must retain failed gates" in findings


def test_scenario_verify_6801_blocked_artifact_has_no_fallback_rows(sources: dict) -> None:
    """SCENARIO-VERIFY-6801-BLOCKED stops before exact audit work."""

    changed = deepcopy(sources)
    changed["exp6800"]["model_output_fixed_point_comparison_completed"] = False
    blocked = mod.build_artifact(
        changed,
        repo_root=mod.REPO_ROOT,
        run_date="20260831",
        duration_s=0.1,
        bootstrap_resamples=32,
    )
    assert blocked["status"] == "complete_blocked_real_output_fixed_point_audit"
    assert blocked["model_output_fixed_point_audit_completed"] is False
    assert blocked["rows"] == []
    assert blocked["verdict_class"] == "blocked"
    assert blocked["gate_check_summary"]["failed_checks"]
    assert mod.validate_artifact(blocked) == []

    missing = deepcopy(sources)
    missing["exp6800"]["rows"].pop()
    disqualified = mod.build_artifact(
        missing,
        repo_root=mod.REPO_ROOT,
        run_date="20260831",
        duration_s=0.1,
        bootstrap_resamples=32,
    )
    assert disqualified["verdict_class"] == "disqualified"
    assert disqualified["rows"] == []


def test_req_verify_6801_writer_and_cli_use_explicit_test_path(
    artifact: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-VERIFY-6801 writes only the explicit output supplied by the caller."""

    output = tmp_path / "audit.json"
    mod.write_output(artifact, output)
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert mod.parse_run_date("20260831") == "20260831"
    with pytest.raises(ValueError, match="YYYYMMDD"):
        mod.parse_run_date("2026-08-31")

    monkeypatch.setattr(mod, "build_from_repo", lambda **_: deepcopy(artifact))
    cli_output = tmp_path / "cli-audit.json"
    assert mod.main(["--date", "20260831", "--artifact-path", str(cli_output)]) == 0
    assert json.loads(cli_output.read_text(encoding="utf-8")) == artifact
    assert artifact["honest_verdict"] in capsys.readouterr().out


def test_req_verify_6801_writer_cleans_failed_publish(
    artifact: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-VERIFY-6801 removes temporary bytes when atomic publication fails."""

    def fail_replace(_source: str, _target: Path) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr(mod.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected"):
        mod.write_output(artifact, tmp_path / "failed.json")
    assert list(tmp_path.iterdir()) == []

    with monkeypatch.context() as context:
        context.setattr(mod.os, "replace", fail_replace)
        context.setattr(mod.os, "unlink", lambda _path: (_ for _ in ()).throw(FileNotFoundError()))
        with pytest.raises(OSError, match="injected"):
            mod.write_output(artifact, tmp_path / "leftover.json")
    for path in tmp_path.iterdir():
        path.unlink()


def test_req_verify_6801_build_from_repo_measures_and_rechecks_checksum(
    artifact: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-VERIFY-6801 measures the repository audit outside stable evidence."""

    ticks = iter((10.0, 12.5))
    monkeypatch.setattr(mod.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(mod, "load_sources", lambda _root: {"frozen": True})
    monkeypatch.setattr(mod, "build_artifact", lambda *_args, **_kwargs: deepcopy(artifact))
    built = mod.build_from_repo(repo_root=mod.REPO_ROOT, run_date="20260831")
    assert built["duration_s"] == 2.5
    assert built["reproducibility_checksum"] == mod.reproducibility_checksum(built)
