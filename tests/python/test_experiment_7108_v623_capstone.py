"""Tests for REQ-REPORT-7108 and SCENARIO-REPORT-7108-* contracts."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from carnot import experiment_7108_v623_capstone as mod


ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "openspec/capabilities/research-reporting/spec.md"

CONTRACT_ROWS = (
    ("exp7097-v623-contract-preflight", "V623 Markdown and YAML task-contract preflight", "results/experiment_7097_v623_contract_preflight.json", []),
    ("exp7098-v623-execution-sota-ingestion", "V623 execution-time SOTA ingestion and claim-boundary audit", "results/experiment_7098_v623_sota_ingestion.json", []),
    ("exp7099-adapter-withheld-live-path-preflight", "Adapter-withheld ARC live-path preflight", "results/experiment_7099_v623_adapter_withheld_preflight.json", []),
    ("exp7100-adapter-withheld-arc-loo-measurement", "Mandatory adapter-withheld ARC leave-one-game-out measurement", "results/experiment_7100_v623_adapter_withheld_loo.json", [("exp7099-adapter-withheld-live-path-preflight", "adapter_withheld_live_path_ready_score")]),
    ("exp7101-adapter-withheld-arc-cold-audit", "Independent adapter-withheld ARC provenance and leakage audit", "results/experiment_7101_v623_adapter_withheld_cold_audit.json", [("exp7100-adapter-withheld-arc-loo-measurement", "adapter_withheld_loo_complete_score")]),
    ("exp7102-feasibility-projected-action-energy", "Exact feasibility projection and analytic ARC action-energy comparison", "results/experiment_7102_v623_feasibility_action_energy.json", [("exp7100-adapter-withheld-arc-loo-measurement", "adapter_withheld_loo_complete_score"), ("exp7101-adapter-withheld-arc-cold-audit", "adapter_withheld_audit_ready_score")]),
    ("exp7103-adapter-withheld-energy-live-ab", "Adapter-withheld feasibility-energy live A/B", "results/experiment_7103_v623_adapter_withheld_energy_live_ab.json", [("exp7102-feasibility-projected-action-energy", "projected_action_energy_comparison_complete_score")]),
    ("exp7104-degree16-action-energy-portability", "Degree-16 action-energy software portability receipt", "results/experiment_7104_v623_degree16_action_energy_portability.json", [("exp7102-feasibility-projected-action-energy", "projected_action_energy_comparison_complete_score")]),
    ("exp7105-sealed-exact-constraint-stream", "Sealed 144-event exact constraint stream", "results/experiment_7105_v623_exact_constraint_stream.json", []),
    ("exp7106-delayed-commit-procedural-memory-csl", "Delayed-commit procedural-memory continuous self-learning A/B", "results/experiment_7106_v623_procedural_memory_csl.json", [("exp7105-sealed-exact-constraint-stream", "exact_constraint_stream_ready_score")]),
    ("exp7107-continual-memory-cold-audit", "Fresh-process continual-memory retention and rollback audit", "results/experiment_7107_v623_continual_memory_cold_audit.json", [("exp7106-delayed-commit-procedural-memory-csl", "procedural_memory_comparison_complete_score")]),
    ("exp7108-v623-capstone", "V623 independent evidence matrix and branch disposition", "results/experiment_7108_v623_capstone.json", []),
)


def markdown_contract(rows: tuple = CONTRACT_ROWS) -> str:
    """Build a separate five-column fixture without using production parser data."""

    lines = [
        "# V623 fixture",
        "",
        "**Milestone:** `2026.09.623`",
        "",
        "## Exact Task Contract",
        "",
        "| Order | Full task ID | Title | Deliverable | Structured gate |",
        "|---:|---|---|---|---|",
    ]
    for order, (task_id, title, deliverable, gates) in enumerate(rows, 1):
        cell = "; ".join(f"`{owner}.{field} == 1`" for owner, field in gates) or "none"
        lines.append(f"| {order} | `{task_id}` | {title} | `{deliverable}` | {cell} |")
    return "\n".join(lines)


def yaml_contract(rows: tuple = CONTRACT_ROWS) -> dict:
    """Build active-YAML values independently from the Markdown fixture."""

    tasks = []
    for task_id, title, deliverable, gates in rows:
        task = {
            "id": task_id,
            "title": title,
            "deliverable": deliverable,
            "milestone": "2026.09.623",
            "per_unit_rows": True,
            "prior_failures": [],
        }
        if gates:
            task["gated_on"] = [
                {"upstream": owner, "artifact_field": field, "op": "==", "value": 1}
                for owner, field in gates
            ]
        tasks.append(task)
    return {"milestone": "2026.09.623", "tasks": tasks}


def minimal_artifact(number: int, *, verdict: str = "null") -> dict:
    """Make one compact present artifact for defensive unit checks."""

    artifact = {
        "experiment_id": number,
        "run_date": "20260907",
        "inference_substrate": "fixture aggregation",
        "inference_substrate_class": "aggregation",
        "execution_venue": "host",
        "preconditions_checked": [],
        "source_artifact_hashes": {},
        "rows": [{"unit": "one", "value": 0}],
        "gate_check_summary": {"passed": True, "failed_check": None, "expected_value": 1, "observed_value": 1},
        "verifier_is_oracle": False,
        "verdict_class": verdict,
        "honest_verdict": f"complete_{verdict}_fixture",
        "reproducibility_checksum": "",
    }
    artifact["field_principles"] = {
        key: f"The {key} field keeps this fixture independently checkable."
        for key in artifact
        if key not in mod.UNPRINCIPLED_METADATA_FIELDS
    }
    return artifact


def test_spec_and_contract_define_exactly_twelve_rows() -> None:
    """REQ-REPORT-7108 and SCENARIO-REPORT-7108-CONTRACT freeze both views."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("### REQ-REPORT-7108") :]
    for marker in (
        "SCENARIO-REPORT-7108-CONTRACT",
        "SCENARIO-REPORT-7108-ARTIFACTS",
        "SCENARIO-REPORT-7108-GATES",
        "SCENARIO-REPORT-7108-HEADLINES",
        "SCENARIO-REPORT-7108-ARC",
        "SCENARIO-REPORT-7108-BOUNDARIES",
        "SCENARIO-REPORT-7108-TRANSACTIONS",
        "SCENARIO-REPORT-7108-DISPOSITION",
        "SCENARIO-REPORT-7108-ARTIFACT",
    ):
        assert marker in section

    markdown_rows = mod.parse_markdown_contract(markdown_contract())
    yaml_rows = mod.parse_yaml_contract(yaml_contract())
    parity = mod.contract_parity(markdown_rows, yaml_rows)
    assert [row["id"] for row in markdown_rows] == [row[0] for row in CONTRACT_ROWS]
    assert len(yaml_rows) == len(parity) == 12
    assert all(row["passed"] for row in parity)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "reordered", "title", "deliverable", "gate"])
def test_contract_mutations_are_visible(mutation: str) -> None:
    """SCENARIO-REPORT-7108-CONTRACT rejects every five-column mismatch."""

    value = yaml_contract()
    if mutation == "missing":
        value["tasks"].pop()
    elif mutation == "duplicate":
        value["tasks"][1] = deepcopy(value["tasks"][0])
    elif mutation == "reordered":
        value["tasks"][0], value["tasks"][1] = value["tasks"][1], value["tasks"][0]
    elif mutation == "title":
        value["tasks"][0]["title"] = "changed"
    elif mutation == "deliverable":
        value["tasks"][0]["deliverable"] = "results/changed.json"
    else:
        value["tasks"][3]["gated_on"][0]["artifact_field"] = "nested.field"
    parity = mod.contract_parity(
        mod.parse_markdown_contract(markdown_contract()), mod.parse_yaml_contract(value)
    )
    assert len(parity) == 12
    assert not all(row["passed"] for row in parity)


def test_missing_and_duplicate_artifacts_use_exact_deliverables(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7108-ARTIFACTS separates absence from duplicate IDs."""

    tasks = mod.parse_yaml_contract(yaml_contract())[:2]
    results = tmp_path / "results"
    results.mkdir()
    exact = results / Path(tasks[0]["deliverable"]).name
    exact.write_text("{}", encoding="utf-8")
    (results / "experiment_7097_duplicate.json").write_text("{}", encoding="utf-8")
    rows = mod.discover_artifacts(tmp_path, tasks)
    assert rows[0]["exact_path_present"] is True
    assert rows[0]["duplicate_artifacts"] is True
    assert rows[1]["exact_path_present"] is False
    assert rows[1]["candidate_count"] == 0


def test_stale_milestone_and_invalid_verdict_fail_identity() -> None:
    """SCENARIO-REPORT-7108-ARTIFACTS rejects stale or invalid identities."""

    task = mod.parse_yaml_contract(yaml_contract())[8]
    artifact = minimal_artifact(7105)
    artifact["milestone"] = "2026.09.622"
    identity = mod.artifact_identity_row(task, artifact, "20260907")
    verdict = mod.verdict_consistency_row(task, {**artifact, "verdict_class": "maybe"})
    assert identity["passed"] is False
    assert verdict["passed"] is False


def test_hash_mutation_and_malformed_principles_fail(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7108-ARTIFACTS checks bytes and scientific reasons."""

    source = tmp_path / "source.txt"
    source.write_text("before", encoding="utf-8")
    expected = "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest()
    artifact = minimal_artifact(7105)
    artifact["source_artifact_hashes"] = {str(source): expected}
    assert all(row["passed"] for row in mod.source_hash_rows(7105, artifact, tmp_path))
    source.write_text("after", encoding="utf-8")
    assert not all(row["passed"] for row in mod.source_hash_rows(7105, artifact, tmp_path))
    artifact["field_principles"].pop("rows")
    assert mod.field_principle_row("exp7105", artifact)["passed"] is False


def test_absent_per_unit_rows_and_gate_field_mismatch_fail() -> None:
    """SCENARIO-REPORT-7108-GATES and HEADLINES require exact top-level rows."""

    tasks = mod.parse_yaml_contract(yaml_contract())
    artifact = minimal_artifact(7100)
    artifact["rows"] = []
    assert mod.per_unit_presence_row(tasks[3], artifact)["passed"] is False
    upstream = minimal_artifact(7099)
    upstream["nested"] = {"adapter_withheld_live_path_ready_score": 1}
    gates = mod.replay_gates(tasks, {tasks[2]["id"]: upstream})
    target = next(row for row in gates if row["consumer"] == tasks[3]["id"])
    assert target["observed_value"] is None
    assert target["passed"] is False


def test_row_headline_contradiction_and_all_zero_completion() -> None:
    """SCENARIO-REPORT-7108-HEADLINES keeps row truth separate from value."""

    rows = [
        {"event_id": f"e{i}", "group_id": f"g{i % 12}", "constraint_family": f"f{i % 4}"}
        for i in range(144)
    ]
    artifact = minimal_artifact(7105, verdict="circular_positive")
    artifact.update(
        {
            "rows": rows,
            "event_count": 143,
            "group_count": 12,
            "family_count": 4,
            "exact_constraint_stream_ready_score": 1,
            "witness_replay_rows": [{"passed": True} for _ in rows],
            "verifier_is_oracle": True,
        }
    )
    recomputed = mod.recompute_headlines(7105, artifact)
    assert any(row["field"] == "event_count" and not row["passed"] for row in recomputed)
    complete_null = minimal_artifact(7100)
    complete_null.update(
        {"adapter_withheld_loo_complete_score": 1, "adapter_withheld_any_level_score": 0}
    )
    assert mod.classify_disposition(7100, complete_null, clean=True, missing=False) == "retain_null"


def test_circularity_and_arc_inflation_fail() -> None:
    """SCENARIO-REPORT-7108-ARC and BOUNDARIES prevent solve inflation."""

    task = mod.parse_yaml_contract(yaml_contract())[8]
    circular = minimal_artifact(7105, verdict="positive")
    circular["verifier_is_oracle"] = True
    assert mod.verdict_consistency_row(task, circular)["passed"] is False

    arc = minimal_artifact(7100, verdict="positive")
    arc.update(
        {
            "solve_provenance": "development_proxy",
            "offline_reproduced": False,
            "arc_registry_delta": 1,
            "per_game_results": [
                {
                    "game_id": "r11l",
                    "transition": {"level_before": 0, "level_after": 1},
                    "forbidden_read_rows": [{"attempted": True, "passed": False}],
                }
            ],
        }
    )
    provenance, registry_rows, recomputed = mod.arc_boundary_rows(7100, arc, {"games": [{"game": "r11l", "levels_reproduced": 6}]})
    assert provenance[0]["passed"] is False
    assert registry_rows[0]["registry_levels"] == 6
    assert recomputed[0]["counted_level_delta"] == 0


def test_hardware_overclaim_fails_host_boundary() -> None:
    """SCENARIO-REPORT-7108-BOUNDARIES keeps host placement off hardware."""

    artifact = minimal_artifact(7104, verdict="positive")
    artifact.update({"hardware_execution_claimed": True, "z1_power_w": 0.1})
    row = mod.hardware_claim_row(7104, artifact)
    assert row["passed"] is False
    assert set(row["forbidden_claim_fields"]) == {"hardware_execution_claimed", "z1_power_w"}


def test_current_matrix_is_complete_and_external_blocks_are_terminal(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7108-DISPOSITION accepts blocked external branches."""

    artifact = mod.build_artifact(ROOT, "20260907", tmp_path / "capstone.json")
    dispositions = {row["experiment_id"]: row["disposition"] for row in artifact["task_disposition_rows"]}
    assert len(artifact["rows"]) == len(dispositions) == 12
    assert artifact["expected_task_count"] == artifact["observed_task_count"] == 12
    assert dispositions[7100] == dispositions[7101] == "blocked"
    assert dispositions[7102] == dispositions[7103] == dispositions[7104] == "blocked"
    assert dispositions[7105] == "retain_circular"
    assert dispositions[7106] == "promote"
    assert dispositions[7108] == "promote"
    assert artifact["v623_evidence_matrix_complete_score"] == 1
    assert artifact["verdict_class"] == "positive"
    assert artifact["honest_verdict"].startswith("complete_")
    assert mod.validate_artifact(artifact) == []


def test_memory_rows_and_transactions_recompute_from_current_evidence(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7108-TRANSACTIONS checks deltas, retention, and audit parity."""

    artifact = mod.build_artifact(ROOT, "20260907", tmp_path / "capstone.json")
    memory = next(row for row in artifact["self_learning_rows"] if row["experiment_id"] == 7106)
    transaction = next(row for row in artifact["transaction_audit_rows"] if row["experiment_id"] == 7106)
    cold = next(row for row in artifact["transaction_audit_rows"] if row["experiment_id"] == 7107)
    assert memory["comparison_complete_recomputed"] == 1
    assert memory["value_ready_recomputed"] == 1
    assert memory["hard_group_regression_count"] == 0
    assert transaction["delayed_commits_after_feedback"] is True
    assert transaction["delayed_commits_atomic"] is True
    assert cold["producer_auditor_parity"] is True


def test_missing_capstone_precondition_is_schema_complete(tmp_path: Path) -> None:
    """REQ-REPORT-7108 blocks only when its own required inputs are absent."""

    output = tmp_path / "results" / "capstone.json"
    output.parent.mkdir()
    artifact = mod.build_artifact(tmp_path, "20260907", output)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["inference_substrate_class"] == "blocked_no_run"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["gate_check_summary"]["failed_check"] == "active_roadmap_readable"
    assert mod.validate_artifact(artifact) == []


def test_partial_caused_only_by_external_block_is_invalid(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7108-DISPOSITION reserves partial for capstone work."""

    artifact = mod.build_artifact(ROOT, "20260907", tmp_path / "capstone.json")
    artifact["verdict_class"] = "partial"
    artifact["honest_verdict"] = "complete_partial_external_tasks_missing"
    artifact["reproducibility_checksum"] = mod.reproducibility_checksum(artifact)
    assert "partial_for_external_block" in mod.validate_artifact(artifact)


def test_command_writes_only_the_requested_temporary_path(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7108-ARTIFACT runs the public command end to end."""

    output = tmp_path / "experiment_7108_v623_capstone.json"
    assert mod.main(["--date", "20260907", "--root", str(ROOT), "--output", str(output)]) == 0
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["experiment_id"] == 7108
    assert artifact["v623_evidence_matrix_complete_score"] == 1
    assert mod.validate_artifact(artifact) == []


def test_defensive_parsers_cover_malformed_external_data(tmp_path: Path) -> None:
    """REQ-REPORT-7108 treats malformed external data as evidence, not a crash."""

    assert mod.parse_yaml_contract({"tasks": "wrong"}) == []
    assert mod.parse_yaml_contract({"tasks": [None]}) == []
    assert mod.parse_markdown_contract("no contract here") == []
    malformed = """## Exact Task Contract
| 1 | `exp1-x` | title | result | `bad gate` |
| 2 | too | many | cells | in | row |
"""
    assert mod.parse_markdown_contract(malformed)[0]["gates"] == [{"malformed": "bad gate"}]
    scalar_contract = """## Exact Task Contract
| 1 | `exp1-x` | title | result | `exp1-x.ready == true`; `exp1-x.ratio >= 0.5`; `exp1-x.name != alpha` |
"""
    gates = mod.parse_markdown_contract(scalar_contract)[0]["gates"]
    assert [gate["value"] for gate in gates] == [True, 0.5, "alpha"]

    artifact = minimal_artifact(7105)
    artifact["field_principles"] = None
    assert mod.field_principle_row("exp7105", artifact)["passed"] is False
    artifact = minimal_artifact(7105)
    artifact["field_principles"]["rows"] = "short"
    assert mod.field_principle_row("exp7105", artifact)["malformed_fields"] == ["rows"]
    artifact["source_artifact_hashes"] = {str(tmp_path / "absent"): "sha256:missing"}
    assert mod.source_hash_rows(7105, artifact, tmp_path)[0]["observed_hash"] is None

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("key: [", encoding="utf-8")
    assert mod._load_json(bad_json) is None
    assert mod._load_json(tmp_path / "missing.json") is None
    assert mod._load_yaml(bad_yaml) is None
    assert mod._load_yaml(tmp_path / "missing.yaml") is None


def test_defensive_recomputation_and_dispositions_cover_closed_classes() -> None:
    """REQ-REPORT-7108 recomputes numeric rows and every terminal class."""

    assert mod._headline("delta", 0.5, 0.5000001, 7102)["passed"] is True
    generic = minimal_artifact(7102, verdict="positive")
    generic.update({"rows": [{"passed": True}], "comparison_complete_score": 1})
    assert mod.recompute_headlines(7102, generic)[0]["passed"] is True
    assert mod._upstream_checksum(7100, generic) is None
    registry = {"games": [{"game": "x", "levels_reproduced": [1, 2]}]}
    arc = {"solve_provenance": "development_proxy", "arc_registry_delta": 0, "offline_reproduced": False, "per_game_results": [{"game_id": "x", "transition": {}}]}
    _, joined, _ = mod.arc_boundary_rows(7100, arc, registry)
    assert joined[0]["registry_levels"] == 2
    assert mod.classify_disposition(7100, minimal_artifact(7100, verdict="blocked"), clean=True, missing=False) == "blocked"
    assert mod.classify_disposition(7100, minimal_artifact(7100, verdict="partial"), clean=True, missing=False) == "partial_own_work"
    assert mod.classify_disposition(7100, minimal_artifact(7100, verdict="disqualified"), clean=True, missing=False) == "disqualified"


def test_ungated_missing_artifact_gets_exact_path_diagnostic(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7108-ARTIFACTS records an ungated missing producer."""

    for relative in (
        "research-roadmap.yaml",
        "openspec/change-proposals/research-roadmap-vNEXT.md",
        "ops/exclusion_manifest.yaml",
        "ops/arc_solve_registry.yaml",
    ):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    output = tmp_path / "results/capstone.json"
    output.parent.mkdir()
    artifact = mod.build_artifact(tmp_path, "20260907", output)
    diagnostic = next(row for row in artifact["blocked_diagnostic_rows"] if row["experiment_id"] == 7097)
    assert diagnostic["failed_check"] == "exact_deliverable_present"


def test_validator_reports_each_independent_semantic_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-7108-ARTIFACT rejects each malformed capstone surface."""

    good = mod.build_artifact(ROOT, "20260907", tmp_path / "capstone.json")
    missing = deepcopy(good)
    missing.pop("rows")
    assert mod.validate_artifact(missing)[0].startswith("missing_required_fields:")

    malformed = deepcopy(good)
    malformed["field_principles"] = {}
    malformed["inference_substrate_class"] = "wrong"
    malformed["execution_venue"] = "hardware"
    malformed["verifier_is_oracle"] = True
    malformed["verdict_class"] = "wrong"
    malformed["honest_verdict"] = "unfinished"
    errors = mod.validate_artifact(malformed)
    assert {"field_principles_incomplete", "inference_substrate_class_invalid", "execution_venue_invalid", "verifier_is_oracle_invalid", "verdict_class_invalid", "honest_verdict_prefix_invalid"} <= set(errors)

    blocked = mod.build_artifact(tmp_path / "absent-root", "20260907", tmp_path / "capstone.json")
    blocked["gate_check_summary"] = {"passed": False}
    blocked["reproducibility_checksum"] = mod.reproducibility_checksum(blocked)
    assert "blocked_gate_summary_invalid" in mod.validate_artifact(blocked)

    inconsistent = deepcopy(good)
    inconsistent["task_disposition_rows"] = inconsistent["task_disposition_rows"][:-1]
    inconsistent["rows"] = []
    inconsistent["observed_task_count"] = 11
    inconsistent["v623_evidence_matrix_complete_score"] = 1
    inconsistent["task_disposition_rows"][0]["disposition"] = "partial_own_work"
    inconsistent["reproducibility_checksum"] = mod.reproducibility_checksum(inconsistent)
    errors = mod.validate_artifact(inconsistent)
    assert {"task_disposition_rows_invalid", "rows_disposition_mismatch", "task_count_invalid", "matrix_complete_score_mismatch", "partial_disposition_class_mismatch"} <= set(errors)

    corrupt = deepcopy(good)
    corrupt["random_seed"] += 1
    assert "reproducibility_checksum_mismatch" in mod.validate_artifact(corrupt)

    monkeypatch.setattr(mod, "build_artifact", lambda *args: {"invalid": True})
    assert mod.main(["--date", "20260907", "--root", str(tmp_path)]) == 1
