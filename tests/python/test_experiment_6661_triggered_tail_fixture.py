"""Tests for the Exp6661 trigger-switched tail fixture.

Spec refs: REQ-CONSTRAINT-6661,
SCENARIO-CONSTRAINT-6661-DELAYED-SYNTAX,
SCENARIO-CONSTRAINT-6661-SEMANTIC-FREE-GRAMMAR,
SCENARIO-CONSTRAINT-6661-IMMUTABLE-MANIFEST,
SCENARIO-CONSTRAINT-6661-FAIL-CLOSED-PARSERS,
SCENARIO-CONSTRAINT-6661-ATTACK-AND-READINESS.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from carnot import experiment_6661_triggered_tail_fixture as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/constraint-verification/spec.md"
PASSING_TESTS = [
    {
        "command": ".venv/bin/pytest tests/python/test_experiment_6661_triggered_tail_fixture.py -q",
        "exit_code": 0,
        "summary": "focused fixture tests passed",
    },
    {
        "command": ".venv/bin/coverage report --fail-under=100",
        "exit_code": 0,
        "summary": "scoped module coverage is 100%",
    },
    {
        "command": (
            ".venv/bin/python scripts/check_spec_coverage.py "
            "tests/python/test_experiment_6661_triggered_tail_fixture.py"
        ),
        "exit_code": 0,
        "summary": "all focused tests trace to OpenSpec",
    },
]


@pytest.fixture(scope="module")
def manifest() -> list[dict]:
    return mod.build_frozen_task_manifest()


@pytest.fixture(scope="module")
def arms() -> dict[str, dict]:
    return mod.build_arm_contracts()


def test_req_6661_spec_declares_full_fixture_contract() -> None:
    """REQ-CONSTRAINT-6661: OpenSpec owns the complete fixture boundary."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("### REQ-CONSTRAINT-6661") :]
    for marker in (
        "SCENARIO-CONSTRAINT-6661-DELAYED-SYNTAX",
        "SCENARIO-CONSTRAINT-6661-SEMANTIC-FREE-GRAMMAR",
        "SCENARIO-CONSTRAINT-6661-IMMUTABLE-MANIFEST",
        "SCENARIO-CONSTRAINT-6661-FAIL-CLOSED-PARSERS",
        "SCENARIO-CONSTRAINT-6661-ATTACK-AND-READINESS",
        mod.RESULT_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "triggered_tail_fixture_ready",
    ):
        assert marker in section


def test_scenario_6661_manifest_is_deterministic_balanced_and_not_id_transport(
    manifest: list[dict],
) -> None:
    """SCENARIO-CONSTRAINT-6661-IMMUTABLE-MANIFEST: task bytes stay frozen."""

    assert manifest == mod.build_frozen_task_manifest()
    assert len(manifest) == mod.EXPECTED_TASK_COUNT == 18
    assert {task["family"] for task in manifest} == set(mod.FAMILY_ORDER)
    assert {
        family: sum(task["family"] == family for task in manifest) for family in mod.FAMILY_ORDER
    } == {family: 6 for family in mod.FAMILY_ORDER}
    assert len({task["task_id"] for task in manifest}) == 18
    assert len({task["task_sha256"] for task in manifest}) == 18

    for task in manifest:
        material = {key: value for key, value in task.items() if key != "task_sha256"}
        assert task["task_sha256"] == mod.sha256_json(material)
        assert task["target"] not in task["prompt"]
        assert task["task_id"] not in task["prompt"]
        assert task["checker"]["finite_answer_id_transport"] is False
        assert task["checker"]["executable"] is True
        assert mod.check_certificate(task, task["target"])["exact_valid"] is True
        assert mod.check_certificate(task, mod.wrong_certificate(task))["exact_valid"] is False


def test_req_6661_exact_checkers_are_family_specific_and_independent(
    manifest: list[dict],
) -> None:
    """REQ-CONSTRAINT-6661: exact family checkers execute outside grammar code."""

    identities = {task["family"]: task["checker"] for task in manifest}
    assert set(identities) == set(mod.FAMILY_ORDER)
    assert len({row["function"] for row in identities.values()}) == 3
    assert all(row["sha256"].startswith("sha256:") for row in identities.values())
    assert all(row["grammar_is_authority"] is False for row in identities.values())

    controls = mod.build_exact_checker_rows(manifest)
    assert len(controls) == 36
    assert {(row["task_id"], row["control_kind"]) for row in controls} == {
        (task["task_id"], kind)
        for task in manifest
        for kind in ("known_positive", "known_negative")
    }
    assert all(row["passed"] for row in controls)
    assert all(row["observed_exact_valid"] == row["expected_exact_valid"] for row in controls)


def test_scenario_6661_arm_contracts_freeze_prompts_schemas_and_budgets(
    arms: dict[str, dict],
) -> None:
    """SCENARIO-CONSTRAINT-6661-IMMUTABLE-MANIFEST: all arms freeze first."""

    assert tuple(arms) == mod.ARM_ORDER
    assert {row["parser_version"] for row in arms.values()} == set(mod.PARSER_VERSIONS.values())
    assert {row["total_token_budget"] for row in arms.values()} == {256}
    assert arms["triggered_tail"]["reasoning_token_budget"] == 192
    assert arms["triggered_tail"]["tail_token_budget"] == 64
    assert arms["triggered_tail"]["trigger_token"] == mod.TRIGGER_TOKEN
    assert arms["immediate_json"]["reasoning_token_budget"] == 0
    assert arms["natural"]["schema"] is None
    assert arms["immediate_json"]["schema"] == mod.TAIL_SCHEMA
    assert arms["triggered_tail"]["schema"] == mod.TAIL_SCHEMA
    assert all(row["contract_sha256"] == mod.arm_contract_hash(row) for row in arms.values())


@pytest.mark.parametrize("arm", mod.ARM_ORDER)
def test_req_6661_each_arm_parses_a_known_positive_and_reaches_exact_checker(
    arm: str, manifest: list[dict]
) -> None:
    """REQ-CONSTRAINT-6661: each frozen transport can carry a certificate."""

    task = manifest[0]
    output = mod.render_known_output(arm, task["target"])
    parsed = mod.parse_arm_output(arm, output)
    assert parsed == {
        "parsed": True,
        "failure": None,
        "certificate": task["target"],
        "trigger_count": 1 if arm == "triggered_tail" else 0,
    }
    assert mod.check_certificate(task, parsed["certificate"])["exact_valid"] is True


@pytest.mark.parametrize(
    ("arm", "output", "failure"),
    (
        ("natural", "reason\nFINAL CERTIFICATE missing colon", "natural_marker_missing"),
        (
            "natural",
            "FINAL CERTIFICATE: a\nFINAL CERTIFICATE: b",
            "natural_marker_count",
        ),
        ("natural", f"reason {mod.TRIGGER_TOKEN}\nFINAL CERTIFICATE: a", "trigger_forbidden"),
        ("immediate_json", '{"certificate": "x",}', "json_malformed"),
        ("immediate_json", '{"certificate": "x", "extra": 1}', "unknown_fields"),
        (
            "immediate_json",
            '{"certificate": "x", "certificate": "y"}',
            "duplicate_fields",
        ),
        ("immediate_json", '{"certificate": 7}', "wrong_primitive_type"),
        ("triggered_tail", '{"certificate": "x"}', "missing_trigger"),
        (
            "triggered_tail",
            f'reason {mod.TRIGGER_TOKEN} more {mod.TRIGGER_TOKEN} {{"certificate": "x"}}',
            "trigger_count",
        ),
        (
            "triggered_tail",
            f'{mod.TRIGGER_TOKEN}{{"certificate": "x"}}',
            "premature_trigger",
        ),
        (
            "triggered_tail",
            f'reason\n{mod.TRIGGER_TOKEN}\n{{"certificate": "x", "extra": 1}}',
            "unknown_fields",
        ),
    ),
)
def test_scenario_6661_parsers_fail_closed(arm: str, output: str, failure: str) -> None:
    """SCENARIO-CONSTRAINT-6661-FAIL-CLOSED-PARSERS: bad transport has no label."""

    parsed = mod.parse_arm_output(arm, output)
    assert parsed["parsed"] is False
    assert parsed["failure"] == failure
    assert parsed["certificate"] is None


def test_req_6661_parser_rejects_unknown_arm_and_non_text() -> None:
    """REQ-CONSTRAINT-6661: parser dispatch does not coerce unknown input."""

    assert mod.parse_arm_output("unknown", "x")["failure"] == "unknown_arm"
    assert mod.parse_arm_output("natural", b"x")["failure"] == "output_not_text"
    assert mod.parse_arm_output("natural", "FINAL CERTIFICATE:   ")["failure"] == (
        "empty_certificate"
    )
    assert mod.parse_arm_output("immediate_json", "[]")["failure"] == "tail_not_object"


def test_scenario_6661_grammar_is_task_independent_and_semantic_free(
    manifest: list[dict],
) -> None:
    """SCENARIO-CONSTRAINT-6661-SEMANTIC-FREE-GRAMMAR: syntax has no answer."""

    receipt = mod.build_syntax_only_grammar_receipt(manifest)
    altered = deepcopy(manifest)
    for task in altered:
        task["target"] = mod.wrong_certificate(task)
        task["task_id"] = "renamed-" + task["task_id"]
    altered_receipt = mod.build_syntax_only_grammar_receipt(altered)

    assert receipt["grammar"] == mod.SYNTAX_ONLY_GBNF
    assert receipt["grammar_sha256"] == altered_receipt["grammar_sha256"]
    assert receipt["schema_sha256"] == altered_receipt["schema_sha256"]
    assert receipt["answer_semantics_absent"] is True
    assert receipt["finite_answer_enumeration"] is False
    assert receipt["task_ids_present"] == []
    assert receipt["targets_present"] == []
    assert receipt["labels_present"] == []
    assert receipt["grammar_only_sample"] == '{"certificate":""}'
    assert receipt["grammar_only_exact_success_count"] == 0
    assert receipt["allowed_syntax"] == {
        "field_names": ["certificate"],
        "primitive_types": {"certificate": "string"},
    }


def test_req_6661_fixture_rows_cover_every_task_and_arm(
    manifest: list[dict], arms: dict[str, dict]
) -> None:
    """REQ-CONSTRAINT-6661: fixture rows bind each task to all arm contracts."""

    rows = mod.build_fixture_rows(manifest, arms)
    assert len(rows) == 18
    assert {row["task_id"] for row in rows} == {task["task_id"] for task in manifest}
    assert all(tuple(row["arm_rows"]) == mod.ARM_ORDER for row in rows)
    assert all(
        arm_row["parse_result"]["parsed"] is True and arm_row["exact_result"]["exact_valid"] is True
        for row in rows
        for arm_row in row["arm_rows"].values()
    )
    assert all(row["row_sha256"] == mod.fixture_row_hash(row) for row in rows)


def test_scenario_6661_all_task_arm_attacks_are_retained_and_pass(
    manifest: list[dict], arms: dict[str, dict]
) -> None:
    """SCENARIO-CONSTRAINT-6661-ATTACK-AND-READINESS: all attacks remain rows."""

    rows = mod.build_leakage_attack_rows(manifest, arms)
    assert len(rows) == mod.EXPECTED_ATTACK_ROW_COUNT == 18 * 3 * 10
    assert {(row["task_id"], row["arm"], row["attack_type"]) for row in rows} == {
        (task["task_id"], arm, attack)
        for task in manifest
        for arm in mod.ARM_ORDER
        for attack in mod.ATTACK_TYPES
    }
    assert all(row["passed"] for row in rows)
    assert all(row["row_sha256"] == mod.attack_row_hash(row) for row in rows)

    wrong = [
        row for row in rows if row["attack_type"] == "semantically_wrong_syntactically_valid_tail"
    ]
    assert all(row["observed"]["parsed"] is True for row in wrong)
    assert all(row["observed"]["exact_valid"] is False for row in wrong)

    malformed = [row for row in rows if row["attack_type"] == "malformed_tail"]
    assert all(row["observed"]["parsed"] is False for row in malformed)

    renamed = [row for row in rows if row["attack_type"] == "label_renaming"]
    assert all(row["observed"]["exact_label_unchanged"] is True for row in renamed)
    assert all(row["observed"]["grammar_hash_unchanged"] is True for row in renamed)

    grammar_only = [row for row in rows if row["attack_type"] == "grammar_only_generation"]
    assert all(row["observed"]["exact_valid"] is not True for row in grammar_only)
    assert all(row["observed"]["answer_recovered"] is False for row in grammar_only)


def test_req_6661_reducer_fails_closed_on_missing_failed_or_leaking_rows(
    manifest: list[dict], arms: dict[str, dict]
) -> None:
    """REQ-CONSTRAINT-6661: readiness comes only from complete passing rows."""

    fixtures = mod.build_fixture_rows(manifest, arms)
    controls = mod.build_exact_checker_rows(manifest)
    attacks = mod.build_leakage_attack_rows(manifest, arms)
    clean = mod.recompute_aggregate_rows(
        manifest=manifest,
        arm_contracts=arms,
        fixture_rows=fixtures,
        exact_checker_rows=controls,
        leakage_attack_rows=attacks,
    )
    assert clean["ready"] is True
    assert clean["counts"] == {
        "tasks": 18,
        "arm_contracts": 3,
        "fixture_rows": 18,
        "checker_controls": 36,
        "attack_rows": 540,
        "expected_attack_rows": 540,
        "leakage_findings": 0,
    }
    assert clean["failed_checks"] == []

    cases = (
        (fixtures[:-1], controls, attacks, "fixture_row_keys"),
        (fixtures, controls[:-1], attacks, "checker_control_keys"),
        (fixtures, controls, attacks[:-1], "attack_row_keys"),
    )
    for fixture_rows, checker_rows, attack_rows, failed in cases:
        result = mod.recompute_aggregate_rows(
            manifest=manifest,
            arm_contracts=arms,
            fixture_rows=fixture_rows,
            exact_checker_rows=checker_rows,
            leakage_attack_rows=attack_rows,
        )
        assert result["ready"] is False
        assert failed in result["failed_checks"]

    failed_attack = deepcopy(attacks)
    failed_attack[0]["passed"] = False
    result = mod.recompute_aggregate_rows(
        manifest=manifest,
        arm_contracts=arms,
        fixture_rows=fixtures,
        exact_checker_rows=controls,
        leakage_attack_rows=failed_attack,
    )
    assert "attack_outcomes" in result["failed_checks"]


def test_scenario_6661_terminal_artifact_has_required_fields_and_recomputes(
    tmp_path: Path,
) -> None:
    """SCENARIO-CONSTRAINT-6661-ATTACK-AND-READINESS: artifact closes from rows."""

    artifact = mod.build_artifact(
        root=REPO,
        date="20260827",
        duration_s=1.25,
        tests_run=PASSING_TESTS,
    )
    assert mod.validate_artifact(artifact) == []
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["verdict_class"] == "null"
    assert artifact["triggered_tail_fixture_ready"] is True
    assert artifact["verifier_is_oracle"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["protected_files_unchanged"]["unchanged"] is True
    assert (
        artifact["protected_files_unchanged"]["before"]
        == artifact["protected_files_unchanged"]["after"]
    )
    assert artifact["aggregate_row_recomputation"]["ready"] is True
    assert len(artifact["per_unit_rows"]) == 18 + 3 + 36 + 540
    assert artifact["reproducibility_checksum"] == mod.artifact_checksum(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_provenance"])
    for row in artifact["field_provenance"].values():
        assert set(row) >= {"source_path", "parser", "function", "sha256", "principle"}

    output = tmp_path / "artifact.json"
    mod.write_artifact_atomic(output, artifact)
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded == artifact
    assert mod.validate_artifact(loaded) == []
    assert not output.with_suffix(output.suffix + ".tmp").exists()


def test_req_6661_preconditions_capture_sources_tools_resources_and_no_llm(
    manifest: list[dict],
) -> None:
    """REQ-CONSTRAINT-6661: measured provenance binds the no-model substrate."""

    before = mod.protected_hashes(REPO)
    receipt = mod.collect_preconditions(REPO, manifest)
    after = mod.protected_files_receipt(REPO, before)

    assert set(receipt["input_hashes"]) >= {path.as_posix() for path in mod.SOURCE_CORPUS_PATHS}
    assert set(receipt["checker_compiler_identities"]) == {
        "scheduling",
        "graph_constraints",
        "arithmetic_logic",
    }
    assert receipt["parser_versions"] == mod.PARSER_VERSIONS
    assert receipt["cpu"]["logical_count"] >= 1
    assert receipt["ram"]["total_bytes"] > 0
    assert receipt["disk"]["free_bytes"] > 0
    assert receipt["no_llm_substrate"]["model_loaded"] is False
    assert receipt["no_llm_substrate"]["model_inference_called"] is False
    assert receipt["llama_cpp_helpers"]["llama_cpp_importable"] is True
    assert receipt["llama_cpp_helpers"]["model_instantiated"] is False
    assert after["unchanged"] is True
    assert after["before"] == after["after"] == before


def test_req_6661_validator_rejects_tampering_and_failed_tests() -> None:
    """REQ-CONSTRAINT-6661: artifact edits cannot preserve a ready verdict."""

    artifact = mod.build_artifact(
        root=REPO,
        date="20260827",
        duration_s=1.0,
        tests_run=PASSING_TESTS,
    )
    mutations = (
        ("missing_required_fields", lambda row: row.pop("status")),
        ("task_count", lambda row: row["frozen_task_manifest"].pop()),
        ("attack_row_count", lambda row: row["leakage_attack_rows"].pop()),
        ("checker_control_failed", lambda row: row["exact_checker_rows"][0].update(passed=False)),
        ("attack_failed", lambda row: row["leakage_attack_rows"][0].update(passed=False)),
        (
            "readiness_mismatch",
            lambda row: row.update(triggered_tail_fixture_ready=False),
        ),
        ("verdict_class_mismatch", lambda row: row.update(verdict_class="positive")),
        (
            "protected_files_changed",
            lambda row: row["protected_files_unchanged"].update(unchanged=False),
        ),
        ("test_command_failed", lambda row: row["tests_run"][0].update(exit_code=1)),
        (
            "field_provenance_missing",
            lambda row: row["field_provenance"].pop("status"),
        ),
        (
            "inference_substrate_mismatch",
            lambda row: row.update(inference_substrate="live_llm_inference"),
        ),
        ("oracle_boundary_mismatch", lambda row: row.update(verifier_is_oracle=False)),
        ("checksum_mismatch", lambda row: row.update(reproducibility_checksum="bad")),
    )
    for expected, mutate in mutations:
        changed = deepcopy(artifact)
        mutate(changed)
        assert expected in mod.validate_artifact(changed)


def test_req_6661_honest_failed_test_receipt_is_a_valid_terminal_blocker() -> None:
    """REQ-CONSTRAINT-6661: a measured test failure must still write evidence."""

    failed_tests = deepcopy(PASSING_TESTS)
    failed_tests.append(
        {
            "command": ".venv/bin/pytest tests/python -q",
            "exit_code": 1,
            "summary": "measured legacy-suite failures",
        }
    )
    artifact = mod.build_artifact(
        root=REPO,
        date="20260827",
        duration_s=1.0,
        tests_run=failed_tests,
    )

    assert artifact["status"] == "blocked_fixture_contract"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["triggered_tail_fixture_ready"] is False
    assert artifact["gate_check_summary"]["first_failed_check"] == "tests"
    assert artifact["gate_check_summary"]["observed"]["failed_test_commands"] == [
        ".venv/bin/pytest tests/python -q"
    ]
    assert mod.validate_artifact(artifact) == []


def test_req_6661_run_writes_atomically_with_supplied_real_receipts(tmp_path: Path) -> None:
    """REQ-CONSTRAINT-6661: orchestration accepts measured test receipts."""

    output = tmp_path / "experiment_6661.json"
    artifact = mod.run(
        date="20260827",
        root=REPO,
        output_path=output,
        tests_run=PASSING_TESTS,
    )
    assert artifact["triggered_tail_fixture_ready"] is True
    assert json.loads(output.read_text(encoding="utf-8")) == artifact


def test_req_6661_command_receipts_capture_exit_and_summary(tmp_path: Path) -> None:
    """REQ-CONSTRAINT-6661: verification commands retain measured outcomes."""

    calls: list[list[str]] = []

    def runner(command: list[str], cwd: Path) -> dict:
        calls.append(command)
        assert cwd == tmp_path
        return {
            "command": " ".join(command),
            "exit_code": 0,
            "summary": "ok",
            "duration_s": 0.01,
        }

    rows = mod.run_verification_commands(tmp_path, command_runner=runner)
    assert len(rows) == len(mod.VERIFICATION_COMMANDS)
    assert calls == [list(command) for command in mod.VERIFICATION_COMMANDS]
    assert all(row["exit_code"] == 0 and row["summary"] == "ok" for row in rows)


def test_req_6661_hash_helpers_are_canonical_and_missing_files_fail_closed(
    tmp_path: Path,
) -> None:
    """REQ-CONSTRAINT-6661: stable hashes expose missing provenance."""

    assert mod.canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'
    assert mod.sha256_bytes(b"x") == "sha256:" + hashlib.sha256(b"x").hexdigest()
    assert mod.sha256_file(tmp_path / "missing") == "missing"
    assert mod.arm_contract_hash({"contract_sha256": "ignored", "x": 1}) == mod.sha256_json(
        {"x": 1}
    )


def test_req_6661_exact_checker_defensive_assignment_branches(
    manifest: list[dict],
) -> None:
    """REQ-CONSTRAINT-6661: malformed, duplicate, missing, and ranged data fail."""

    graph = next(task for task in manifest if task["family"] == "graph_constraints")
    arithmetic = next(task for task in manifest if task["family"] == "arithmetic_logic")

    assert mod.check_certificate(graph, "not-an-assignment")["reason"] == ("malformed_assignment")
    assert mod.check_certificate(graph, "a=1;a=2")["reason"] == "duplicate_assignment"
    assert mod.check_certificate(graph, "a=1")["reason"] == "assignment_domain_mismatch"

    graph_out_of_range = str(graph["target"]).replace("a=1", "a=0")
    assert mod.check_certificate(graph, graph_out_of_range)["reason"] == "color_out_of_range"

    first_variable = arithmetic["checker_input"]["variables"][0]
    assert mod.check_certificate(arithmetic, f"{first_variable}=1")["reason"] == (
        "assignment_domain_mismatch"
    )
    arithmetic_out_of_range = str(arithmetic["target"]).replace(
        f"{first_variable}=4", f"{first_variable}=21"
    )
    assert mod.check_certificate(arithmetic, arithmetic_out_of_range)["reason"] == (
        "integer_out_of_range"
    )

    unknown = {**graph, "family": "unknown"}
    assert mod.check_certificate(unknown, str(graph["target"]))["reason"] == "unknown_family"
    with pytest.raises(ValueError, match="unknown arm"):
        mod.render_known_output("unknown", "x")
    assert mod._package_version("package-that-does-not-exist-exp6661") is None


def test_req_6661_validator_and_run_refuse_internal_reducer_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CONSTRAINT-6661: internal validation errors stop terminal output."""

    artifact = mod.build_artifact(
        root=REPO,
        date="20260827",
        duration_s=1.0,
        tests_run=PASSING_TESTS,
    )
    changed = deepcopy(artifact)
    changed["arm_contracts"] = None
    assert "aggregate_recomputation_failed" in mod.validate_artifact(changed)

    monkeypatch.setattr(mod, "validate_artifact", lambda payload: ["injected_error"])
    with pytest.raises(ValueError, match="injected_error"):
        mod.run(date="20260827", root=REPO, tests_run=PASSING_TESTS)


def test_req_6661_default_command_runner_measures_real_process(tmp_path: Path) -> None:
    """REQ-CONSTRAINT-6661: the production command adapter records real output."""

    row = mod._default_command_runner(
        [mod.sys.executable, "-c", "print('measured command')"], tmp_path
    )
    assert row["exit_code"] == 0
    assert row["summary"] == "measured command"
    assert row["output_sha256"].startswith("sha256:")
    assert row["duration_s"] >= 0.0
