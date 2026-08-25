"""Tests for the Exp6604 exact two-level plan corpus.

Spec refs: REQ-CONSTRAINT-6604,
SCENARIO-CONSTRAINT-6604-GENERATION-AND-SPLITS,
SCENARIO-CONSTRAINT-6604-TWO-LEVEL-COMPILATION,
SCENARIO-CONSTRAINT-6604-INDEPENDENT-EXECUTION,
SCENARIO-CONSTRAINT-6604-INCOMPLETE-ENCODING,
SCENARIO-CONSTRAINT-6604-ROW-RETENTION-AND-ATOMIC-OUTPUT,
SCENARIO-CONSTRAINT-6604-ADVERSARIAL-CONTROLS.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from carnot import experiment_6604_exact_two_level_plan_corpus as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/constraint-verification/spec.md"
TESTS_RUN = [
    {
        "command": (
            ".venv/bin/pytest tests/python/test_experiment_6604_exact_two_level_plan_corpus.py -q"
        ),
        "exit_code": 0,
        "duration_s": 1.0,
    },
    {
        "command": "ruff check python/carnot/experiment_6604_exact_two_level_plan_corpus.py",
        "exit_code": 0,
        "duration_s": 1.0,
    },
]


def _feasible_task() -> dict[str, object]:
    return next(task for task in mod.generate_plan_tasks() if task["known_feasible"])


def test_req_constraint_6604_spec_declares_the_full_contract() -> None:
    """REQ-CONSTRAINT-6604: OpenSpec owns the exact corpus contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("### REQ-CONSTRAINT-6604") :]
    for marker in (
        "SCENARIO-CONSTRAINT-6604-GENERATION-AND-SPLITS",
        "SCENARIO-CONSTRAINT-6604-TWO-LEVEL-COMPILATION",
        "SCENARIO-CONSTRAINT-6604-INDEPENDENT-EXECUTION",
        "SCENARIO-CONSTRAINT-6604-INCOMPLETE-ENCODING",
        "SCENARIO-CONSTRAINT-6604-ROW-RETENTION-AND-ATOMIC-OUTPUT",
        "SCENARIO-CONSTRAINT-6604-ADVERSARIAL-CONTROLS",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "headroom_fixture_ready_score",
    ):
        assert marker in section


def test_scenario_6604_generation_and_frozen_splits_are_deterministic() -> None:
    """SCENARIO-CONSTRAINT-6604-GENERATION-AND-SPLITS: bytes and splits freeze."""

    first = mod.generate_plan_tasks()
    second = mod.generate_plan_tasks()

    assert first == second
    assert len(first) == 72
    assert mod.corpus_checksum(first) == mod.corpus_checksum(second)
    assert {task["split"] for task in first} == {"calibration", "held"}
    assert sum(task["split"] == "calibration" for task in first) == 36
    assert sum(task["split"] == "held" for task in first) == 36
    assert len({task["task_id"] for task in first}) == 72
    assert len({task["source_sha256"] for task in first}) == 72

    for split in ("calibration", "held"):
        strata = {
            tuple(task["stratum"][name] for name in mod.STRATUM_AXES)
            for task in first
            if task["split"] == split
        }
        assert len(strata) == 36

    for task in first:
        source = str(task["source_bytes"])
        prompt = str(task["model_prompt_bytes"])
        assert hashlib.sha256(source.encode()).hexdigest() == task["source_sha256"]
        assert hashlib.sha256(prompt.encode()).hexdigest() == task["model_prompt_sha256"]
        assert "gold_witness" not in prompt
        assert "known_feasible" not in prompt
        assert task["known_feasible"] == mod.search_exact_feasibility(task)["feasible"]


def test_scenario_6604_two_level_compilers_accept_gold_with_replayable_state() -> None:
    """SCENARIO-CONSTRAINT-6604-TWO-LEVEL-COMPILATION: both levels replay."""

    task = _feasible_task()
    plan = str(task["gold_witness"])
    syntax_program = mod.TokenSyntaxCompiler().compile(task)
    syntax = syntax_program.run(plan)
    semantic_program = mod.ActionSemanticCompiler().compile(task)
    semantic = semantic_program.run(syntax["meta_tokens"])

    assert syntax["accepted"] is True
    assert semantic["accepted"] is True
    assert syntax_program.compiler_version == mod.TOKEN_COMPILER_VERSION
    assert semantic_program.compiler_version == mod.SEMANTIC_COMPILER_VERSION
    assert syntax_program.meta_token_mapping
    assert semantic["transition_rows"]
    assert semantic["final_state"]
    assert semantic["goal_satisfied"] is True
    assert mod.canonical_json(syntax_program.receipt()) == mod.canonical_json(
        mod.TokenSyntaxCompiler().compile(task).receipt()
    )
    assert syntax_program.run(" " + plan)["errors"] == [
        "ambiguous_or_noncanonical_whitespace",
        f"unknown_or_ill_typed_action: {plan.splitlines()[0]}",
    ]
    assert semantic_program.run(["<UNKNOWN>"])["reason"] == "unknown_meta_token"


def test_scenario_6604_independent_executor_classifies_all_mutation_families() -> None:
    """SCENARIO-CONSTRAINT-6604-INDEPENDENT-EXECUTION: exact failures stay distinct."""

    task = _feasible_task()
    executor = mod.IndependentExactExecutor()
    valid = executor.execute(task, str(task["gold_witness"]))
    assert valid == executor.execute(task, str(task["gold_witness"]))
    assert valid["valid"] is True
    assert valid["reason"] == "valid_goal_reached"

    rows = {row["mutation_type"]: row for row in mod.build_task_mutations(task)}
    assert set(rows) == {
        "syntax_error",
        "precondition_violation",
        "ordering_violation",
        "unmet_goal",
        "parser_ambiguity",
        "semantic_state_attack",
    }
    assert rows["syntax_error"]["syntax_accept"] is False
    assert rows["parser_ambiguity"]["syntax_accept"] is False
    assert rows["precondition_violation"]["exact_reason"] == "precondition_violation"
    assert rows["ordering_violation"]["exact_reason"] == "ordering_violation"
    assert rows["unmet_goal"]["exact_reason"] == "unmet_goal"
    assert rows["semantic_state_attack"]["syntax_accept"] is True
    assert rows["semantic_state_attack"]["semantic_accept"] is False
    assert all(row["failed_as_expected"] for row in rows.values())
    assert executor.execute(task, " BAD(x)")["detail"] == ["ambiguous_action_text: BAD(x)"]
    assert executor.execute(task, "BAD (x)")["detail"] == ["noncanonical_action_text:BAD (x)"]


def test_req_6604_defensive_generator_search_and_corpus_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CONSTRAINT-6604: defensive generation and validation paths fail closed."""

    tasks = mod.generate_plan_tasks()
    too_short = deepcopy(tasks[:-1])
    errors = mod.validate_frozen_corpus(too_short)["errors"]
    assert "task_count" in errors
    assert "split_counts" in errors

    bad_hash = deepcopy(tasks)
    bad_hash[0]["source_sha256"] = "0" * 64
    assert "source_hash_mismatch" in mod.validate_frozen_corpus(bad_hash)["errors"]

    bad_stratum = deepcopy(tasks)
    bad_stratum[1]["stratum"] = deepcopy(bad_stratum[0]["stratum"])
    assert "stratum_coverage" in mod.validate_frozen_corpus(bad_stratum)["errors"]

    capped = deepcopy(tasks[0])
    capped["known_feasible"] = False
    capped["goal_predicates"] = ["never:true"]
    capped["max_plan_steps"] = 0
    assert mod.search_exact_feasibility(capped)["feasible"] is False

    original_execute = mod.IndependentExactExecutor.execute
    calls = 0

    def drifting_execute(self: object, task: object, plan: str) -> dict[str, object]:
        nonlocal calls
        calls += 1
        result = original_execute(self, task, plan)
        return {**result, "injected_call": calls}

    monkeypatch.setattr(mod.IndependentExactExecutor, "execute", drifting_execute)
    assert "nondeterministic_execution" in mod.validate_frozen_corpus(tasks)["errors"]
    monkeypatch.setattr(mod.IndependentExactExecutor, "execute", original_execute)
    monkeypatch.setattr(mod, "detect_compiler_executor_sharing", lambda source=None: True)
    assert "compiler_executor_code_sharing" in mod.validate_frozen_corpus(tasks)["errors"]

    monkeypatch.setattr(
        mod,
        "search_exact_feasibility",
        lambda task: {
            "feasible": not bool(task["known_feasible"]),
            "witness": None,
            "states_expanded": 0,
            "search_complete": True,
        },
    )
    with pytest.raises(RuntimeError, match="generator feasibility mismatch"):
        mod._build_task(
            split="calibration",
            stratum_index=0,
            lexical="plain",
            temporal="inspect_early",
            branching="direct",
            distractor="none",
        )


def test_scenario_6604_incomplete_encoding_is_not_self_certifying() -> None:
    """SCENARIO-CONSTRAINT-6604-INCOMPLETE-ENCODING: the executor rejects omission."""

    task = _feasible_task()
    row = mod.build_omitted_obligation_attack(task)

    assert row["mutation_type"] == "omitted_obligation"
    assert row["omitted_obligation_id"] == mod.OMITTED_OBLIGATION_ID
    assert row["syntax_accept"] is True
    assert row["semantic_accept"] is True
    assert row["both_encoded_automata_accept"] is True
    assert row["exact_valid"] is False
    assert row["exact_reason"] == "precondition_violation"
    assert row["failed_as_expected"] is True


def test_scenario_6604_impossible_tasks_and_attacks_fail_closed() -> None:
    """SCENARIO-CONSTRAINT-6604-ADVERSARIAL-CONTROLS: attacks are detected."""

    tasks = mod.generate_plan_tasks()
    impossible = [task for task in tasks if not task["known_feasible"]]
    assert impossible
    assert all(task["gold_witness"] is None for task in impossible)
    assert all(mod.search_exact_feasibility(task)["feasible"] is False for task in impossible)

    attacks = mod.build_attack_rows(tasks, mod.PROTECTED_BASELINE_SHA256)
    expected = {
        "split_leakage",
        "duplicate_source_bytes",
        "seed_drift",
        "nondeterministic_execution",
        "goal_answer_leakage",
        "compiler_executor_code_sharing",
        "impossible_task_mislabeling",
        "protected_file_mutation",
        "incomplete_semantic_encoding",
    }
    assert {row["attack_type"] for row in attacks} == expected
    assert all(row["detected"] for row in attacks)
    assert all(row["failed_closed"] for row in attacks)

    clean = mod.validate_frozen_corpus(tasks)
    assert clean["passed"] is True
    assert clean["errors"] == []

    drifted = deepcopy(tasks)
    drifted[0]["seed"] = int(drifted[0]["seed"]) + 1
    assert "seed_drift" in mod.validate_frozen_corpus(drifted)["errors"]

    duplicated = deepcopy(tasks)
    duplicated[1]["source_bytes"] = duplicated[0]["source_bytes"]
    duplicated[1]["source_sha256"] = duplicated[0]["source_sha256"]
    assert "duplicate_source_bytes" in mod.validate_frozen_corpus(duplicated)["errors"]


def test_scenario_6604_terminal_artifact_retains_rows_and_writes_atomically(
    tmp_path: Path,
) -> None:
    """SCENARIO-CONSTRAINT-6604-ROW-RETENTION-AND-ATOMIC-OUTPUT: artifact closes."""

    output = tmp_path / "experiment_6604.json"
    artifact = mod.build_artifact(
        repo_root=REPO,
        output_path=output,
        date="20260825",
        duration_s=1.0,
        tests_run=TESTS_RUN,
        write=True,
    )
    written = json.loads(output.read_text(encoding="utf-8"))

    assert written == artifact
    assert mod.validate_artifact(artifact) == []
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["verdict_class"] == "null"
    assert artifact["headroom_fixture_ready_score"] == 1.0
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert artifact["protected_files_unchanged"] is True
    assert len(artifact["fixture_and_split_receipts"]) == 72
    assert len(artifact["plan_fixture_rows"]) == 72
    assert len(artifact["mutation_rows"]) > 72
    assert all(row["failed_as_expected"] for row in artifact["mutation_rows"])
    assert all(row["failed_closed"] for row in artifact["attack_rows"])
    assert artifact["independent_exact_executor_receipts"]["oracle_distinct"] is True
    assert artifact["independent_exact_executor_receipts"]["hand_checked_gold_subset"]
    assert artifact["atomic_output_receipt"]["atomic_replace"] is True
    assert not output.with_suffix(output.suffix + ".tmp").exists()
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_provenance"])
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])


def test_req_6604_artifact_validation_rejects_tampering(tmp_path: Path) -> None:
    """REQ-CONSTRAINT-6604: row, gate, test, protection, and checksum edits fail."""

    artifact = mod.build_artifact(
        repo_root=REPO,
        output_path=tmp_path / "unused.json",
        date="20260825",
        duration_s=1.0,
        tests_run=TESTS_RUN,
        write=False,
    )
    changes = (
        ("missing_required_fields", lambda value: value.pop("status")),
        ("missing_fixture_row", lambda value: value["plan_fixture_rows"].pop()),
        ("readiness_mismatch", lambda value: value.update(headroom_fixture_ready_score=0.0)),
        ("verdict_class_mismatch", lambda value: value.update(verdict_class="positive")),
        ("protected_files_changed", lambda value: value.update(protected_files_unchanged=False)),
        (
            "test_command_failed",
            lambda value: value["tests_run"][0].update(exit_code=1),
        ),
        (
            "attack_not_closed",
            lambda value: value["attack_rows"][0].update(failed_closed=False),
        ),
        (
            "omitted_encoding_proof_missing",
            lambda value: value["mutation_rows"].__setitem__(
                slice(None),
                [
                    row
                    for row in value["mutation_rows"]
                    if row["mutation_type"] != "omitted_obligation"
                ],
            ),
        ),
        (
            "mutation_expectation_failed",
            lambda value: value["mutation_rows"][0].update(failed_as_expected=False),
        ),
        ("field_principles_missing", lambda value: value["field_principles"].pop("status")),
        ("field_provenance_missing", lambda value: value["field_provenance"].pop("status")),
        (
            "inference_substrate_mismatch",
            lambda value: value.update(inference_substrate="live_llm_inference"),
        ),
        ("oracle_boundary_mismatch", lambda value: value.update(verifier_is_oracle=False)),
        ("checksum_mismatch", lambda value: value.update(reproducibility_checksum="bad")),
    )
    for expected, mutate in changes:
        altered = deepcopy(artifact)
        mutate(altered)
        assert expected in mod.validate_artifact(altered)
