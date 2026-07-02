"""Tests for Exp 5135 V471 source/scope audit.

Spec refs: REQ-REPORT-5135, SCENARIO-REPORT-5135,
SCENARIO-REPORT-5135-BLOCKED-SCOPE.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from carnot import experiment_5135_v471_source_scope_audit as mod
from scripts import experiment_5135_v471_source_scope_audit as script_mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-reporting/spec.md"


def _spec_section() -> str:
    spec = SPEC_PATH.read_text(encoding="utf-8")
    start = spec.index("### REQ-REPORT-5135")
    end = spec.index("### REQ-REPORT-5110", start)
    return spec[start:end]


def _artifact() -> dict[str, object]:
    return mod.build_artifact(
        root=REPO,
        duration_s=1.25,
        run_date="20260702",
        tests_run=["tests/python/test_experiment_5135_v471_source_scope_audit.py"],
    )


def _docs_with_selected_roadmap(tasks: list[dict[str, object]]) -> dict[str, object]:
    docs = mod.load_input_documents(REPO)
    loaded = yaml.safe_load(str(docs["selected_roadmap_text"]))
    loaded["tasks"] = tasks
    roadmap_text = yaml.safe_dump(loaded, sort_keys=False)
    return docs | {
        "selected_roadmap_text": roadmap_text,
        "active_roadmap_text": roadmap_text,
    }


def test_req_report_5135_spec_declares_source_scope_gate_contract() -> None:
    """REQ-REPORT-5135: OpenSpec declares the V471 audit contract."""

    section = _spec_section()

    assert "REQ-REPORT-5135" in section
    assert "SCENARIO-REPORT-5135" in section
    assert "SCENARIO-REPORT-5135-BLOCKED-SCOPE" in section
    assert mod.EXPERIMENT_ID in section
    assert str(mod.RESULT_RELATIVE_PATH) in section
    assert "V471-PLANNER-REFERENCES" in section
    assert "MODEL_SPECS" in section
    assert "unsloth/Qwen3.6-35B-A3B-GGUF" in section
    assert "structured `gated_on`" in section
    assert "FoVer selector" in section
    assert "ops/exclusion_manifest.yaml" in section
    assert "scripts/research_conductor.py" in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_req_report_5135_artifact_maps_sources_and_preserves_scope() -> None:
    """REQ-REPORT-5135: artifact maps every V471 task and blocks doomed reruns."""

    artifact = _artifact()

    mod.validate_artifact(artifact)
    assert mod.REQUIRED_ARTIFACT_FIELDS.issubset(artifact)
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["milestone"] == mod.MILESTONE
    assert artifact["honest_verdict"] == mod.COMPLETE_VERDICT
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["v471_reference_block_found"] is True
    assert artifact["fover_same_scope_rerun_found"] is False
    assert artifact["sota_model_discipline_ok"] is True
    assert artifact["structured_gates_ok"] is True
    assert artifact["exclusion_manifest_conflicts"] == []
    assert artifact["conductor_modified"] is False
    assert artifact["tests_run"] == ["tests/python/test_experiment_5135_v471_source_scope_audit.py"]
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES

    assert set(artifact["task_source_map"]) == mod.REQUIRED_TASK_IDS
    for task_id, row in artifact["task_source_map"].items():
        assert task_id in mod.REQUIRED_TASK_IDS
        assert row["task_title"]
        assert row["motivation_type"] in {"fresh_source", "local_continuation", "capstone_aggregation"}
        assert row["sources_or_artifacts"]

    source_ids = {
        source_id
        for row in artifact["task_source_map"].values()
        for source_id in row["sources_or_artifacts"]
    }
    assert {
        "openskill_verifier_anchors",
        "k2v_verifiable_data_synthesis",
        "symbolic_kan_certificate_residuals",
        "solver_verified_formulation_generation_selection",
        "reward_guided_energy_based_decoding",
        "partitioned_probabilistic_computing_telemetry",
    }.issubset(source_ids)
    assert "results/experiment_5133_capstone_v470.json" in source_ids

    roadmap = artifact["roadmap_parse_evidence"]
    assert roadmap["selected_roadmap"] == "research-roadmap.yaml"
    assert roadmap["research_roadmap_next"]["exists"] is False
    assert roadmap["active_roadmap"]["milestone"] == mod.MILESTONE


def test_scenario_report_5135_sota_tasks_require_model_specs_and_mandated_gguf() -> None:
    """SCENARIO-REPORT-5135: LLM-backed tasks carry MODEL_SPECS and GGUF discipline."""

    details = _artifact()["sota_model_discipline_details"]

    assert {row["task_id"] for row in details} == mod.LLM_BACKED_TASK_IDS
    for row in details:
        assert row["needs_llm_inference"] is True
        assert row["model_specs_required_field_present"] is True
        assert set(row["mandated_ggufs"]) == mod.MANDATED_GGUFS
        assert row["task_or_global_mandated_gguf_found"] is True
        assert row["ok"] is True


def test_scenario_report_5135_structured_gate_details_cover_required_dependencies() -> None:
    """SCENARIO-REPORT-5135: condition-dependent tasks have conductor-readable gates."""

    artifact = _artifact()
    details = artifact["structured_gate_details"]

    assert {row["task_id"] for row in details} == set(mod.REQUIRED_STRUCTURED_GATES)
    for row in details:
        expected = mod.REQUIRED_STRUCTURED_GATES[row["task_id"]]
        assert row["ok"] is True
        assert row["gate_found"] is True
        assert row["expected_gate"]["upstream"] == expected["upstream"]
        assert row["expected_gate"]["artifact_field"] == expected["artifact_field"]
        assert row["expected_gate"]["op"] == expected["op"]
        assert row["expected_gate"]["value"] == expected["value"]
    assert artifact["conductor_gate_support"]["evaluate_gates_available"] is True


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda artifact: {key: value for key, value in artifact.items() if key != "task_source_map"},
            "required field",
        ),
        (
            lambda artifact: artifact | {"v471_reference_block_found": False},
            "reference block",
        ),
        (
            lambda artifact: artifact
            | {
                "task_source_map": {
                    key: value for key, value in artifact["task_source_map"].items() if key != "exp5136"
                }
            },
            "task_source_map",
        ),
        (
            lambda artifact: artifact | {"fover_same_scope_rerun_found": True},
            "FoVer",
        ),
        (
            lambda artifact: artifact | {"sota_model_discipline_ok": False},
            "SOTA",
        ),
        (
            lambda artifact: artifact | {"structured_gates_ok": False},
            "structured gate",
        ),
        (
            lambda artifact: artifact | {"conductor_modified": True},
            "conductor_modified",
        ),
        (
            lambda artifact: artifact | {"field_principles": {}},
            "field_principle",
        ),
        (
            lambda artifact: artifact | {"experiment_id": "bad"},
            "experiment_id",
        ),
        (
            lambda artifact: artifact | {"milestone": "2026.07.470"},
            "milestone",
        ),
        (
            lambda artifact: artifact | {"honest_verdict": "done"},
            "honest_verdict",
        ),
        (
            lambda artifact: artifact | {"inference_substrate": "live_llm"},
            "inference_substrate",
        ),
        (
            lambda artifact: artifact | {"duration_s": 0.0},
            "duration_s",
        ),
        (
            lambda artifact: artifact
            | {
                "task_source_map": artifact["task_source_map"]
                | {"exp5136": artifact["task_source_map"]["exp5136"] | {"sources_or_artifacts": []}}
            },
            "invalid row",
        ),
        (
            lambda artifact: artifact | {"tests_run": []},
            "tests_run",
        ),
        (
            lambda artifact: artifact
            | {
                "exclusion_manifest_conflicts": [
                    {
                        "task_id": "exp5137",
                        "severity": "hard_block",
                        "reason": "synthetic retired-scope conflict",
                    }
                ]
            },
            "exclusion",
        ),
    ],
)
def test_validator_rejects_scope_and_schema_failures_for_scenario_report_5135(
    mutate: object,
    message: str,
) -> None:
    """SCENARIO-REPORT-5135-BLOCKED-SCOPE: invalid audit claims fail closed."""

    bad_artifact = mutate(copy.deepcopy(_artifact()))

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(bad_artifact)


def test_blocked_task_map_sota_gate_and_exclusion_states_validate_for_scenario_report_5135() -> None:
    """SCENARIO-REPORT-5135-BLOCKED-SCOPE: blocked verdict branches are valid artifacts."""

    docs = mod.load_input_documents(REPO)
    loaded = yaml.safe_load(str(docs["selected_roadmap_text"]))
    missing_task_artifact = mod.build_artifact_from_documents(
        documents=_docs_with_selected_roadmap(loaded["tasks"][:-1]),
        duration_s=0.5,
        run_date="20260702",
        tests_run=["focused"],
    )
    assert missing_task_artifact["honest_verdict"] == mod.BLOCKED_TASK_MAP_VERDICT

    altered_tasks = copy.deepcopy(loaded["tasks"])
    for task in altered_tasks:
        if str(task["id"]).startswith("exp5136"):
            task["prompt"] = str(task["prompt"]).replace("MODEL_SPECS", "MODEL_SPEC_MISSING")
    sota_artifact = mod.build_artifact_from_documents(
        documents=_docs_with_selected_roadmap(altered_tasks),
        duration_s=0.5,
        run_date="20260702",
        tests_run=["focused"],
    )
    assert sota_artifact["honest_verdict"] == mod.BLOCKED_SOTA_VERDICT

    gate_tasks = copy.deepcopy(loaded["tasks"])
    for task in gate_tasks:
        if str(task["id"]).startswith("exp5137"):
            task.pop("gated_on", None)
    gate_artifact = mod.build_artifact_from_documents(
        documents=_docs_with_selected_roadmap(gate_tasks),
        duration_s=0.5,
        run_date="20260702",
        tests_run=["focused"],
    )
    assert gate_artifact["honest_verdict"] == mod.BLOCKED_STRUCTURED_GATES_VERDICT

    exclusion_artifact = mod.build_artifact_from_documents(
        documents=docs
        | {
            "exclusion_manifest_text": "retired_extras:\n- blocked_patterns:\n  - solver-verified formulation selector\n"
        },
        duration_s=0.5,
        run_date="20260702",
        tests_run=["focused"],
    )
    assert exclusion_artifact["honest_verdict"] == mod.BLOCKED_EXCLUSION_VERDICT
    assert exclusion_artifact["exclusion_manifest_conflicts"]


def test_low_level_parsers_cover_yaml_manifest_and_negated_fover_edges_for_req_report_5135() -> None:
    """REQ-REPORT-5135: parser helpers expose malformed YAML and retired-scope conflicts."""

    bad_yaml = mod.parse_roadmap_yaml("tasks: [", path="bad.yaml", exists=True)
    assert bad_yaml["parses"] is False
    assert bad_yaml["milestone"] == "yaml_error"

    conflicts = mod.build_exclusion_manifest_conflicts(
        [
            {"id": "not-an-exp", "title": "noop", "prompt": ""},
            {"id": "exp5136-clean", "title": "Retired id", "prompt": ""},
            {"id": "exp5137-selector", "title": "Solver-verified formulation selector", "prompt": ""},
            {
                "id": "exp5138-negated",
                "title": "No FoVer in-domain pool",
                "prompt": "This task must not use a FoVer in-domain candidate-selection pool.",
            },
        ],
        """
retired:
- experiment_id: 5136
  reason: synthetic
retired_extras:
- blocked_patterns:
  - solver-verified formulation selector
  - fover in-domain candidate-selection pool
""",
    )
    assert [row["task_id"] for row in conflicts] == ["exp5136", "exp5137"]


def test_missing_reference_block_builds_blocked_artifact_for_scenario_report_5135() -> None:
    """SCENARIO-REPORT-5135-BLOCKED-SCOPE: missing V471 block is terminal blocked."""

    docs = mod.load_input_documents(REPO)
    artifact = mod.build_artifact_from_documents(
        documents=docs | {"research_references": "# no V471 references here\n"},
        duration_s=0.5,
        run_date="20260702",
        tests_run=["focused"],
    )

    assert artifact["v471_reference_block_found"] is False
    assert artifact["honest_verdict"] == mod.BLOCKED_REFERENCE_VERDICT
    mod.validate_artifact(artifact)


def test_fover_same_scope_rerun_builds_blocked_artifact_for_scenario_report_5135() -> None:
    """SCENARIO-REPORT-5135-BLOCKED-SCOPE: same-scope FoVer rerun is preserved."""

    docs = mod.load_input_documents(REPO)
    bad_roadmap = docs["selected_roadmap_text"].replace(
        "Build a receipt-backed structured reasoning pool v2 over exact-checkable non-FoVer tasks.",
        "Build an in-domain FoVer candidate-selection pool with the same selector audit.",
        1,
    )
    artifact = mod.build_artifact_from_documents(
        documents=docs | {"selected_roadmap_text": bad_roadmap},
        duration_s=0.5,
        run_date="20260702",
        tests_run=["focused"],
    )

    assert artifact["fover_same_scope_rerun_found"] is True
    assert artifact["honest_verdict"] == mod.BLOCKED_FOVER_VERDICT
    assert artifact["fover_rerun_findings"]
    mod.validate_artifact(artifact)


def test_script_entrypoint_writes_valid_artifact_for_req_report_5135(tmp_path: Path) -> None:
    """REQ-REPORT-5135: script entrypoint writes the same validated artifact."""

    output = tmp_path / "experiment_5135_v471_source_scope_audit.json"

    path = script_mod.main(
        root=REPO,
        output=output,
        date="20260702",
        duration_s=0.75,
        tests_run=["script-entrypoint"],
    )

    assert path == output
    artifact = json.loads(path.read_text(encoding="utf-8"))
    mod.validate_artifact(artifact)
    assert artifact["tests_run"] == ["script-entrypoint"]
