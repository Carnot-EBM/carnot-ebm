"""V595 owned-contract and active-manifest preflight tests.

Spec refs: REQ-AGENTIC-6810-1, REQ-AGENTIC-6810-2,
REQ-CONSTRAINT-6810, and REQ-CL-6810.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from scripts.experiments import experiment_6810_v595_contract_manifest_preflight as exp


REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def roadmap() -> dict[str, object]:
    """Load the checked-in execution contract once for focused tests."""

    return exp.load_yaml_mapping(REPO / exp.ROADMAP_PATH)


def test_req_6810_owned_specs_name_complete_contract_boundaries() -> None:
    """REQ-AGENTIC-6810-1 and sibling anchors own all four V595 contracts."""

    for contract, owner in exp.CONTRACT_OWNERS.items():
        spec_text = (REPO / owner.spec_path).read_text(encoding="utf-8")
        section = exp.requirement_section(spec_text, owner.requirement)
        assert owner.implementing_task in section
        assert owner.artifact_path in section
        assert owner.gate_field in section
        assert owner.authority_boundary in section
        assert owner.failure_behavior in section
        assert contract in exp.CONTRACT_OWNERS


def test_req_6810_design_and_manifest_have_exact_v595_identity(
    roadmap: dict[str, object],
) -> None:
    """REQ-AGENTIC-6810-1 requires four phases and Exp6810 through Exp6823."""

    design = exp.parse_design(REPO / exp.DESIGN_PATH)
    rows = exp.audit_task_contracts(roadmap, design)
    assert design.phase_numbers == (1, 2, 3, 4)
    assert design.task_numbers == tuple(range(6810, 6824))
    assert [row["experiment_number"] for row in rows] == list(range(6810, 6824))
    assert len({row["task_id"] for row in rows}) == 14
    assert len({row["deliverable"] for row in rows}) == 14
    assert all(row["passed"] for row in rows)
    assert rows[-1]["task_id"] == "exp6823-v595-branch-disposition"
    assert rows[-1]["ungated"] is True


def test_req_6810_gates_resolve_to_verbatim_nonpositive_fields(
    roadmap: dict[str, object],
) -> None:
    """REQ-CONSTRAINT-6810 resolves each gate without scientific positivity."""

    rows = exp.audit_gate_resolution(roadmap, exp.retired_experiment_ids(REPO))
    assert rows
    assert all(row["upstream_exists"] for row in rows)
    assert all(row["producer_declares_field_verbatim"] for row in rows)
    assert all(row["positivity_gate"] is False for row in rows)
    assert all(row["upstream_retired"] is False for row in rows)
    assert all(row["passed"] for row in rows)


def test_req_6810_models_priors_rows_and_prompt_tails_are_complete(
    roadmap: dict[str, object],
) -> None:
    """REQ-CL-6810 keeps task metadata complete and fail-closed."""

    models = exp.audit_model_specs(roadmap)
    priors = exp.audit_prior_failures(roadmap)
    prompts = exp.audit_prompt_contracts(roadmap)
    assert all(row["passed"] for row in models)
    assert all(row["passed"] for row in priors)
    assert all(row["passed"] for row in prompts)
    assert all(row["per_unit_rows"] for row in prompts)


def test_req_6810_mutations_fail_the_specific_contract(
    roadmap: dict[str, object],
) -> None:
    """REQ-CONSTRAINT-6810 rejects duplicate, positive, retired, and vague contracts."""

    duplicate = deepcopy(roadmap)
    duplicate["tasks"][1]["id"] = duplicate["tasks"][0]["id"]
    duplicate["tasks"][1]["deliverable"] = duplicate["tasks"][0]["deliverable"]
    task_rows = exp.audit_task_contracts(duplicate, exp.parse_design(REPO / exp.DESIGN_PATH))
    assert any(not row["unique_task_id"] for row in task_rows)
    assert any(not row["unique_deliverable"] for row in task_rows)

    positive = deepcopy(roadmap)
    positive["tasks"][1]["gated_on"][0]["artifact_field"] = "acceptance_gate_positive"
    gate_rows = exp.audit_gate_resolution(positive, set())
    assert gate_rows[0]["positivity_gate"] is True
    assert gate_rows[0]["passed"] is False

    retired = deepcopy(roadmap)
    retired["tasks"][1]["gated_on"][0]["upstream"] = "exp2091"
    retired_rows = exp.audit_gate_resolution(retired, {"exp2091"})
    assert retired_rows[0]["upstream_retired"] is True
    assert retired_rows[0]["passed"] is False

    missing_prior = deepcopy(roadmap)
    missing_prior["tasks"][0]["prior_failures"] = []
    assert exp.audit_prior_failures(missing_prior)[0]["passed"] is False

    bad_model = deepcopy(roadmap)
    bad_model["tasks"][2]["prompt"] = bad_model["tasks"][2]["prompt"].replace(
        "unsloth/Qwen3.6-35B-A3B-GGUF", "vendor/legacy-model"
    ).replace("unsloth/gemma-4-31B-it-GGUF", "vendor/legacy-model").replace(
        "unsloth/gemma-4-26B-A4B-it-GGUF", "vendor/legacy-model"
    )
    assert exp.audit_model_specs(bad_model)[2]["passed"] is False

    bad_tail = deepcopy(roadmap)
    bad_tail["tasks"][0]["prompt"] += "\nContinue with unrelated work."
    bad_tail["tasks"][0]["per_unit_rows"] = False
    assert exp.audit_prompt_contracts(bad_tail)[0]["passed"] is False


def test_req_6810_ready_artifact_has_one_task_and_owner_row(
    roadmap: dict[str, object],
) -> None:
    """REQ-AGENTIC-6810-2 emits a row for every task and requirement owner."""

    artifact = exp.build_artifact(REPO, "20260831", 1.25, roadmap=roadmap)
    assert exp.validate_artifact(artifact) == []
    assert artifact["task_count"] == 14
    assert artifact["roadmap_identity"] == {
        "milestone": "2026.08.595",
        "first_experiment": 6810,
        "last_experiment": 6823,
    }
    assert len(artifact["rows"]) == 14 + len(exp.CONTRACT_OWNERS)
    assert artifact["v595_contract_map_ready"] is True
    assert artifact["verdict_class"] == "null"
    assert str(artifact["honest_verdict"]).startswith("complete:")
    assert artifact["gate_check_summary"] == []
    assert set(artifact["field_principles"]) == set(artifact)
    assert artifact["reproducibility_checksum"] == exp.payload_checksum(artifact)


def test_req_6810_blocked_identity_and_artifact_tampering_fail_closed(
    roadmap: dict[str, object],
) -> None:
    """REQ-AGENTIC-6810-1 blocks a wrong milestone and detects changed output."""

    wrong = deepcopy(roadmap)
    wrong["milestone"] = "2026.08.594"
    artifact = exp.build_artifact(REPO, "20260831", 0.5, roadmap=wrong)
    assert artifact["v595_contract_map_ready"] is False
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == "complete_blocked_v595_contract_manifest_preflight"
    assert artifact["gate_check_summary"]
    assert exp.validate_artifact(artifact) == []

    changed = deepcopy(artifact)
    changed["task_count"] = 14
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(changed)


def test_req_6810_cli_writes_only_the_requested_atomic_artifact(
    tmp_path: Path,
) -> None:
    """REQ-CL-6810 keeps test execution away from tracked result state."""

    output = tmp_path / "nested" / "preflight.json"
    assert exp.main(
        ["--date", "20260831", "--project-root", str(REPO), "--output", str(output)]
    ) == 0
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["v595_contract_map_ready"] is True
    assert exp.main(["--validate", "--output", str(output)]) == 0

    malformed = yaml.safe_dump({"milestone": "wrong", "tasks": []})
    (tmp_path / "roadmap.yaml").write_text(malformed, encoding="utf-8")
    with pytest.raises(ValueError, match="top-level roadmap"):
        exp.audit_task_contracts([], exp.parse_design(REPO / exp.DESIGN_PATH))


def test_req_6810_defensive_inputs_and_nonready_contracts_fail_closed(
    tmp_path: Path,
    roadmap: dict[str, object],
) -> None:
    """REQ-CONSTRAINT-6810 rejects malformed source and incomplete contract shapes."""

    sequence_yaml = tmp_path / "sequence.yaml"
    sequence_yaml.write_text("- not\n- a\n- roadmap\n", encoding="utf-8")
    with pytest.raises(ValueError, match="top-level roadmap"):
        exp.load_yaml_mapping(sequence_yaml)

    assert exp.requirement_section("## REQ-OTHER: unrelated\n", "REQ-MISSING") == ""
    with pytest.raises(ValueError, match="tasks must be a list"):
        exp.audit_task_contracts(
            {"milestone": "2026.08.595", "tasks": None},
            exp.parse_design(REPO / exp.DESIGN_PATH),
        )
    assert exp._required_fields_block("No artifact field section.") == ""

    fake_repo = tmp_path / "fake-repo"
    (fake_repo / "ops").mkdir(parents=True)
    (fake_repo / "ops/exclusion_manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "retired": ["invalid-row", {"experiment_id": 2091}],
                "retired_experiments": [],
                "retired_extras": [],
            }
        ),
        encoding="utf-8",
    )
    assert "exp2091" in exp.retired_experiment_ids(fake_repo)

    incomplete = deepcopy(roadmap)
    incomplete["tasks"][0]["prompt"] += "\nUncontracted tail."
    artifact = exp.build_artifact(REPO, "20260831", 0.25, roadmap=incomplete)
    assert artifact["v595_contract_map_ready"] is False
    assert artifact["gate_check_summary"]
    assert exp.validate_artifact(artifact) == []


def test_req_6810_artifact_validator_names_each_invalid_contract(
    roadmap: dict[str, object],
) -> None:
    """REQ-AGENTIC-6810-1 keeps ready and blocked artifact failures explicit."""

    ready = exp.build_artifact(REPO, "20260831", 0.25, roadmap=roadmap)
    missing = deepcopy(ready)
    missing.pop("schema")
    errors = exp.validate_artifact(missing)
    assert "missing_required_fields:schema" in errors
    assert "field_principles_do_not_cover_artifact" in errors

    invalid_ready = deepcopy(ready)
    invalid_ready.update(
        {
            "verdict_class": "unknown",
            "verifier_is_oracle": True,
            "task_count": 0,
            "roadmap_identity": {},
            "gate_check_summary": [{"passed": False}],
            "honest_verdict": "not terminal",
        }
    )
    errors = exp.validate_artifact(invalid_ready)
    assert "verdict_class_outside_closed_enum" in errors
    assert "verifier_is_oracle_must_be_false" in errors
    assert "ready_artifact_row_or_task_count_mismatch" in errors
    assert "ready_artifact_roadmap_identity_mismatch" in errors
    assert "ready_artifact_has_failed_gate_summary" in errors
    assert "ready_artifact_verdict_is_not_terminal" in errors

    wrong = deepcopy(roadmap)
    wrong["milestone"] = "2026.08.594"
    invalid_blocked = exp.build_artifact(REPO, "20260831", 0.25, roadmap=wrong)
    invalid_blocked.update(
        {
            "verdict_class": "null",
            "gate_check_summary": [],
            "honest_verdict": "complete: wrong blocked shape",
        }
    )
    errors = exp.validate_artifact(invalid_blocked)
    assert "blocked_artifact_verdict_class_mismatch" in errors
    assert "blocked_artifact_missing_gate_summary" in errors
    assert "blocked_artifact_honest_verdict_mismatch" in errors


def test_req_6810_cli_refuses_to_write_invalid_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CL-6810 makes validation failure terminal before atomic output."""

    output = tmp_path / "must-not-exist.json"
    monkeypatch.setattr(exp, "validate_artifact", lambda _artifact: ["forced_error"])
    assert exp.main(
        ["--date", "20260831", "--project-root", str(REPO), "--output", str(output)]
    ) == 1
    assert not output.exists()
