"""REQ-CONSTRAINT-6832 operational-obligation saturation fixture tests."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from carnot import experiment_6832_operational_obligation_saturation_fixture as exp


@pytest.fixture(scope="module")
def scenarios() -> list[dict]:
    """Build the frozen rows once because all tests inspect the same bytes."""

    return exp.generate_scenarios()


@pytest.fixture(scope="module")
def artifact() -> dict:
    """Build the complete fixture without writing into the research record."""

    return exp.build_artifact(duration_s=1.25, worktree_status="")


def _candidate_bytes(action_ids: list[str]) -> bytes:
    return exp.canonical_bytes({"selected_action_ids": action_ids})


def test_req_6832_spec_exists_before_implementation() -> None:
    """REQ-CONSTRAINT-6832 exists before the producer implementation."""

    spec = (exp.REPO_ROOT / exp.SPEC_PATH).read_text(encoding="utf-8")
    assert "## REQ-CONSTRAINT-6832:" in spec
    assert "### SCENARIO-CONSTRAINT-6832-PRECONDITIONS:" in spec
    assert "### SCENARIO-CONSTRAINT-6832-READINESS:" in spec


def test_scenario_6832_schema_and_principles_are_complete(artifact: dict) -> None:
    """REQ-CONSTRAINT-6832 fixes the terminal schema and every field principle."""

    assert set(artifact) == set(exp.FIELD_PRINCIPLES)
    assert set(artifact["field_principles"]) == set(artifact)
    assert artifact["schema"] == exp.ARTIFACT_SCHEMA
    assert artifact["inference_substrate"] == (
        "deterministic CPU exact-checker transactional fixture; "
        "deterministic CPU fixture generation, no LLM"
    )
    assert artifact["verifier_is_oracle"] is False
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"].startswith("complete_")
    assert exp.validate_artifact(artifact) == []


def test_scenario_6832_count_and_dependency_balance(scenarios: list[dict]) -> None:
    """SCENARIO-CONSTRAINT-6832-BALANCE freezes 30 rows and a 15/15 split per count."""

    assert len(scenarios) == 150
    assert len({row["scenario_id"] for row in scenarios}) == 150
    for count in exp.OBLIGATION_COUNTS:
        rows = [row for row in scenarios if row["obligation_count"] == count]
        assert len(rows) == 30
        assert sum(row["dependency_mode"] == "independent" for row in rows) == 15
        assert sum(row["dependency_mode"] == "interacting" for row in rows) == 15
        assert all(len(row["obligations"]) == count for row in rows)


def test_scenario_6832_typed_semantics_consume_all_fields(scenarios: list[dict]) -> None:
    """SCENARIO-CONSTRAINT-6832-TYPED-SEMANTICS consumes all five contract fields."""

    expected = set(exp.OBLIGATION_FIELDS)
    for scenario in scenarios:
        for obligation in scenario["obligations"]:
            assert set(obligation["contract"]) == expected
            result = exp.resolve_scenario(scenario)["obligations"][obligation["obligation_id"]]
            assert result["disposition"] in {
                "constructive",
                "fallback",
                "preempted",
                "fail_closed",
            }


def test_scenario_6832_permutation_and_arm_information_invariance(
    scenarios: list[dict],
) -> None:
    """SCENARIO-CONSTRAINT-6832-PERMUTATION preserves truth under every order."""

    by_template: dict[str, list[dict]] = {}
    for scenario in scenarios:
        by_template.setdefault(scenario["template_id"], []).append(scenario)
        typed = exp.extract_prompt_information(scenario["prompts"]["typed"])
        compressed = exp.extract_prompt_information(scenario["prompts"]["compressed"])
        assert typed == compressed == exp.prompt_information(scenario)
        assert scenario["prompt_information_sha256"] == exp.sha256_json(typed)
    assert len(by_template) == 50
    for rows in by_template.values():
        assert len(rows) == 3
        assert len({row["permutation_id"] for row in rows}) == 3
        assert len({tuple(row["candidate_prompt_order"]) for row in rows}) == 3
        assert len({tuple(row["legal_action_ids"]) for row in rows}) == 1
        canonical = exp.prompt_information(rows[0])
        assert all(exp.prompt_information(row) == canonical for row in rows)


def test_scenario_6832_unique_nonempty_solution_sets(scenarios: list[dict]) -> None:
    """SCENARIO-CONSTRAINT-6832-UNIQUE-SOLUTION rejects each one-action perturbation."""

    for scenario in scenarios:
        legal = scenario["legal_action_ids"]
        assert legal
        assert exp.check_joint(scenario, _candidate_bytes(legal))["passed"] is True
        for action in scenario["candidates"]:
            if action["action_id"] not in legal:
                changed = sorted({*legal, action["action_id"]})
                assert exp.check_joint(scenario, _candidate_bytes(changed))["passed"] is False


def test_scenario_6832_safe_noop_and_impossible_cases(scenarios: list[dict]) -> None:
    """SCENARIO-CONSTRAINT-6832-SAFE-FAIL-CLOSED keeps both boundaries exact."""

    safe = [row for row in scenarios if row["semantic_class"] == "safe_no_op"]
    impossible = [
        row for row in scenarios if row["semantic_class"] == "intentionally_unsatisfiable"
    ]
    assert len(safe) == 30
    assert len(impossible) == 30
    assert all(
        set(row["legal_action_ids"])
        == {obligation["contract"]["fallback"]["action_id"] for obligation in row["obligations"]}
        for row in safe
    )
    assert all(row["legal_action_ids"] == [row["fail_closed_action_id"]] for row in impossible)


@pytest.mark.parametrize(
    ("raw", "code"),
    [
        (b"not-json", "invalid_json"),
        (b'{"selected_action_ids":[] }', "non_canonical_json"),
        (b'{"extra":[],"selected_action_ids":[]}', "invalid_response_fields"),
        (b'{"selected_action_ids":"x"}', "invalid_action_list"),
        (b'{"selected_action_ids":[""]}', "invalid_action_id"),
        (b'{"selected_action_ids":["x","x"]}', "duplicate_action_id"),
        (b"\xff", "invalid_utf8"),
    ],
)
def test_req_6832_parser_fails_closed(raw: bytes, code: str) -> None:
    """REQ-CONSTRAINT-6832 parses canonical JSON without extraction or repair."""

    with pytest.raises(exp.FixtureError, match=code):
        exp.parse_candidate_response(raw)


def test_scenario_6832_field_and_joint_checker_mutations(scenarios: list[dict]) -> None:
    """SCENARIO-CONSTRAINT-6832-CHECKERS validates five mutations for every row."""

    for scenario in scenarios:
        results = exp.run_checker_mutations(scenario)
        assert set(results["cases"]) == {"legal", "violation", "omission", "conflict", "reorder"}
        assert results["cases"]["legal"]["joint_passed"] is True
        assert results["cases"]["reorder"]["joint_passed"] is True
        assert results["cases"]["violation"]["joint_passed"] is False
        assert results["cases"]["omission"]["joint_passed"] is False
        assert results["cases"]["conflict"]["joint_passed"] is False
        assert results["all_expected"] is True
        for case in ("legal", "reorder"):
            assert all(
                check["passed"]
                and set(check["fields"]) == set(exp.OBLIGATION_FIELDS)
                and all(check["fields"].values())
                for check in results["cases"][case]["obligation_checks"].values()
            )


def test_scenario_6832_candidate_response_order_is_ignored(scenarios: list[dict]) -> None:
    """SCENARIO-CONSTRAINT-6832-PERMUTATION treats action identifiers as a set."""

    multi = next(row for row in scenarios if len(row["legal_action_ids"]) > 1)
    forward = _candidate_bytes(multi["legal_action_ids"])
    reverse = _candidate_bytes(list(reversed(multi["legal_action_ids"])))
    assert exp.parse_candidate_response(forward) == exp.parse_candidate_response(reverse)
    assert exp.check_joint(multi, reverse)["passed"] is True


def test_scenario_6832_leakage_audit_catches_each_class(scenarios: list[dict]) -> None:
    """SCENARIO-CONSTRAINT-6832-LEAKAGE rejects answer and fixture-only prompt data."""

    audit = exp.audit_prompt_leakage(scenarios)
    assert audit["passed"] is True
    assert audit["violations"] == []
    changed = deepcopy(scenarios)
    changed[0]["prompts"]["typed"] += "\nanswer key"
    bad = exp.audit_prompt_leakage(changed)
    assert bad["passed"] is False
    assert bad["violations"][0]["reason"] == "forbidden_vocabulary:answer key"


def test_req_6832_preconditions_bind_both_sources_and_worktree() -> None:
    """SCENARIO-CONSTRAINT-6832-PRECONDITIONS binds gates, hashes, and cleanliness."""

    exp6811 = (exp.REPO_ROOT / exp.EXP6811_PATH).read_bytes()
    exp6831 = (exp.REPO_ROOT / exp.EXP6831_PATH).read_bytes()
    checks = exp.evaluate_preconditions(exp6811, exp6831, worktree_status="")
    assert all(row["passed"] for row in checks)

    changed_contract = json.loads(exp6831)
    changed_contract["v597_contract_ready"] = False
    blocked = exp.evaluate_preconditions(
        exp6811,
        exp.canonical_bytes(changed_contract),
        worktree_status=" M results/source.json\n",
    )
    failed = {row["check"] for row in blocked if not row["passed"]}
    assert "exp6831_file_sha256" in failed
    assert "v597_contract_ready" in failed
    assert "owned_source_tree_clean" in failed


def test_scenario_6832_blocked_artifact_stops_before_generation() -> None:
    """SCENARIO-CONSTRAINT-6832-PRECONDITIONS emits the exact blocked terminal state."""

    artifact = exp.build_artifact(duration_s=0.5, worktree_status=" M source.py\n")
    assert artifact["status"] == "complete_blocked_operational_obligation_saturation_fixture"
    assert artifact["honest_verdict"] == (
        "complete_blocked_operational_obligation_saturation_fixture"
    )
    assert artifact["verdict_class"] == "blocked"
    assert artifact["scenarios"] == []
    assert artifact["checker_mutation_results"] == []
    assert artifact["gate_check_summary"]["failed_check"] == "owned_source_tree_clean"
    assert exp.validate_artifact(artifact) == []


def test_scenario_6832_readiness_and_manifests_are_exact(artifact: dict) -> None:
    """SCENARIO-CONSTRAINT-6832-READINESS depends only on exact fixture checks."""

    assert artifact["operational_saturation_fixture_ready"] is True
    assert artifact["obligation_counts"] == [1, 2, 4, 6, 8]
    assert len(artifact["scenario_manifest"]) == 150
    assert len(artifact["prompt_arm_manifest"]) == 300
    assert len(artifact["checker_mutation_results"]) == 150
    assert artifact["checker_manifest"]["obligation_checker_count"] == 630
    assert artifact["checker_manifest"]["joint_checker_count"] == 150
    assert artifact["leakage_audit"]["passed"] is True
    assert artifact["legal_action_headroom"]["all_required_nonzero"] is True
    assert artifact["gate_check_summary"]["passed"] is True
    assert artifact["source_artifact_hashes"]["exp6811"]["file_sha256"] == (
        exp.EXPECTED_EXP6811_FILE_SHA256
    )
    assert artifact["source_artifact_hashes"]["exp6831"]["file_sha256"] == (
        exp.EXPECTED_EXP6831_FILE_SHA256
    )


def test_req_6832_checksum_is_deterministic_and_binds_content(artifact: dict) -> None:
    """REQ-CONSTRAINT-6832 excludes duration but binds code, scenarios, and output."""

    rebuilt = exp.build_artifact(duration_s=999.0, worktree_status="")
    assert rebuilt["reproducibility_checksum"] == artifact["reproducibility_checksum"]
    changed = deepcopy(artifact)
    changed["scenarios"][0]["observed_facts"].append("mutated")
    assert exp.reproducibility_checksum(changed) != artifact["reproducibility_checksum"]
    assert set(artifact["implementation_hashes"]) == {"module", "wrapper"}


def test_req_6832_validator_rejects_readiness_and_checksum_mutations(artifact: dict) -> None:
    """REQ-CONSTRAINT-6832 validates readiness, principles, hashes, and closed enums."""

    changed = deepcopy(artifact)
    changed["field_principles"].pop("schema")
    changed["verdict_class"] = "unknown"
    changed["operational_saturation_fixture_ready"] = False
    errors = exp.validate_artifact(changed)
    assert "field principles do not cover every top-level field" in errors
    assert "verdict class is outside the closed set" in errors
    assert "reproducibility checksum mismatch" in errors
    assert "complete artifact is not ready" in errors


def test_req_6832_defensive_source_and_worktree_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-CONSTRAINT-6832-PRECONDITIONS reports unreadable and dirty inputs."""

    assert exp._json_object(b"not-json") is None
    assert exp._json_object(b"[]") is None
    assert exp._read_source_bytes(tmp_path, Path("missing.json")).startswith(b'{"path"')
    assert exp._sha256_file(tmp_path / "missing.py") is None

    monkeypatch.setattr(
        exp.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=7, stderr="failed", stdout=""),
    )
    assert exp._owned_worktree_status(tmp_path) == "git_status_error:7:failed"
    monkeypatch.setattr(
        exp.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stderr="", stdout="clean"),
    )
    assert exp._owned_worktree_status(tmp_path) == "clean"


def test_req_6832_defensive_prompt_and_checker_paths(scenarios: list[dict]) -> None:
    """REQ-CONSTRAINT-6832 returns stable failures for unknown prompt and check identities."""

    with pytest.raises(exp.FixtureError, match="unknown_prompt_representation"):
        exp.extract_prompt_information("plain text")
    scenario = scenarios[0]
    unknown = exp.check_obligation(scenario, scenario["legal_action_ids"], "missing")
    assert unknown["passed"] is False
    assert not any(unknown["fields"].values())
    parse_failure = exp.check_joint(scenario, b"not-json")
    assert parse_failure == {
        "obligation_checks": {},
        "parse_error": "invalid_json",
        "parsed": False,
        "passed": False,
        "selected_action_ids": [],
    }


def test_scenario_6832_leakage_detects_response_and_information(
    scenarios: list[dict],
) -> None:
    """SCENARIO-CONSTRAINT-6832-LEAKAGE detects exact response and arm drift."""

    changed = deepcopy(scenarios[:1])
    changed[0]["prompts"]["typed"] = _candidate_bytes(changed[0]["legal_action_ids"]).decode()
    reasons = {row["reason"] for row in exp.audit_prompt_leakage(changed)["violations"]}
    assert "exact_response_leakage" in reasons
    assert "information_mismatch" in reasons


def test_req_6832_validator_covers_all_terminal_failures(artifact: dict) -> None:
    """REQ-CONSTRAINT-6832 reports every malformed complete and blocked terminal field."""

    ready = deepcopy(artifact)
    ready.pop("scenario_manifest")
    ready["verifier_is_oracle"] = True
    ready["duration_s"] = True
    ready["gate_check_summary"] = {"passed": False, "failed_check": "x"}
    ready["scenarios"] = []
    ready["prompt_arm_manifest"] = []
    ready["checker_mutation_results"] = []
    ready["leakage_audit"] = {"passed": False}
    ready["legal_action_headroom"] = {"all_required_nonzero": False}
    ready["verdict_class"] = "positive"
    ready["honest_verdict"] = "bad"
    errors = exp.validate_artifact(ready)
    assert "top-level fields differ from the declared contract" in errors
    assert "verifier_is_oracle must be false" in errors
    assert "duration_s must be a nonnegative number" in errors
    assert "ready artifact has a failed gate" in errors
    assert "ready artifact scenario count mismatch" in errors
    assert "ready artifact prompt count mismatch" in errors
    assert "ready artifact checker count mismatch" in errors
    assert "ready artifact failed leakage audit" in errors
    assert "ready artifact lacks required action headroom" in errors
    assert "ready terminal verdict mismatch" in errors

    blocked = exp.build_artifact(duration_s=0.1, worktree_status="dirty")
    blocked["operational_saturation_fixture_ready"] = True
    blocked["scenarios"] = [{}]
    blocked["checker_mutation_results"] = [{}]
    blocked["verdict_class"] = "partial"
    blocked["honest_verdict"] = "bad"
    blocked["reproducibility_checksum"] = exp.reproducibility_checksum(blocked)
    blocked_errors = exp.validate_artifact(blocked)
    assert "blocked artifact lacks an exact failed gate" in blocked_errors
    assert "blocked artifact generated fixture rows" in blocked_errors
    assert "blocked terminal verdict mismatch" in blocked_errors


def test_req_6832_atomic_execute_and_validate(tmp_path: Path) -> None:
    """REQ-CONSTRAINT-6832 runs the task command against a caller-owned output."""

    output = tmp_path / "fixture.json"
    artifact = exp.execute(exp.REPO_ROOT, "20260901", output, worktree_status="")
    assert output.is_file()
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert exp.validate_artifact(artifact) == []
    assert exp.main(["--date", "20260901", "--output", str(output), "--validate"]) == 0
    assert exp.main(["--date", "2026-09-01", "--output", str(output)]) == 2
    with pytest.raises(exp.FixtureError, match="invalid_run_date"):
        exp.execute(exp.REPO_ROOT, "20261301", output, worktree_status="")


def test_req_6832_execute_and_main_defensive_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact: dict,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-CONSTRAINT-6832 makes invalid artifacts and command states terminal."""

    output = tmp_path / "artifact.json"
    monkeypatch.setattr(exp, "build_artifact", lambda **kwargs: {})
    with pytest.raises(exp.FixtureError, match="invalid_artifact"):
        exp.execute(exp.REPO_ROOT, "20260901", output, worktree_status="")

    invalid = deepcopy(artifact)
    invalid["verdict_class"] = "bad"
    output.write_bytes(exp.canonical_bytes(invalid))
    assert exp.main(["--output", str(output), "--validate"]) == 1
    assert "verdict class is outside the closed set" in capsys.readouterr().err

    output.write_text("not-json", encoding="utf-8")
    assert exp.main(["--output", str(output), "--validate"]) == 2
    assert "Expecting value" in capsys.readouterr().err

    monkeypatch.setattr(exp, "execute", lambda *args, **kwargs: artifact)
    assert exp.main(["--output", str(output)]) == 0
    assert '"artifact"' in capsys.readouterr().out
