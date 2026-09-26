"""REQ-REPORT-7726 and REQ-HARNESS-7726 contract regression checks."""

from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest
import yaml

from carnot import experiment_7726_v673_contract_methods as subject


ROOT = Path(__file__).resolve().parents[2]
DESIGN = (ROOT / "openspec/change-proposals/research-roadmap-vNEXT.md").read_text()
ROADMAP = yaml.safe_load((ROOT / "research-roadmap.yaml").read_text())


def test_three_independent_contracts_and_mutations() -> None:
    """SCENARIO-REPORT-7726-CONTRACT: all fields and private defects matter."""

    result = subject.compare_contract(DESIGN, ROADMAP)
    assert result["passed"]
    assert len(result["rows"]) == 13
    assert all(row["matched"] and row["raw_denominator"] > 0 for row in result["rows"])
    for mutation in ("delete", "reorder", "stale", "producer_field", "model"):
        private = subject.mutate_roadmap(ROADMAP, mutation)
        assert private != ROADMAP
        assert not subject.compare_contract(DESIGN, private)["passed"], mutation


def test_missing_independent_content_and_wrong_gate() -> None:
    """SCENARIO-REPORT-7726-CUSTODY: no source is reconstructed from YAML."""

    assert (
        "design_table_missing"
        in subject.compare_contract(
            DESIGN.replace("## Exact Task Contract", "## Removed Contract"), ROADMAP
        )["errors"]
    )
    assert (
        "design_json_missing"
        in subject.compare_contract(DESIGN.replace("```json", "```removed", 1), ROADMAP)["errors"]
    )
    machine, table, gates, errors = subject.parse_design(DESIGN)
    assert not errors and len(machine) == len(table) == 13 and len(gates) == 7
    changed = deepcopy(ROADMAP)
    changed["tasks"][4]["gated_on"][0]["artifact_field"] = "wrong_producer_field"
    assert not subject.compare_contract(DESIGN, changed)["passed"]


def test_authority_resolution(tmp_path: Path) -> None:
    """SCENARIO-HARNESS-7726-SELECTED: V673 staged takes priority when present."""

    active = tmp_path / "research-roadmap.yaml"
    active.write_text(yaml.safe_dump(ROADMAP))
    selected, _, candidates = subject.resolve_authority(tmp_path)
    assert selected == active and candidates[0]["exists"] is False
    staged = tmp_path / "research-roadmap-next.yaml"
    staged.write_text(yaml.safe_dump(ROADMAP))
    assert subject.resolve_authority(tmp_path)[0] == staged
    stale = deepcopy(ROADMAP)
    stale["milestone"] = "2026.09.672"
    staged.write_text(yaml.safe_dump(stale))
    assert subject.resolve_authority(tmp_path)[0] == active
    active.unlink()
    with pytest.raises(ValueError, match="matching V673 authority"):
        subject.resolve_authority(tmp_path)


def test_malformed_machine_block() -> None:
    """SCENARIO-REPORT-7726-CUSTODY: malformed design is never reconstructed."""

    wrong = DESIGN.replace('"milestone": "2026.09.673"', '"milestone": "2026.09.672"', 1)
    assert "design_json_invalid" in subject.parse_design(wrong)[3]
    broken = DESIGN.replace('"milestone": "2026.09.673"', '"milestone": ', 1)
    assert "design_json_invalid" in subject.parse_design(broken)[3]


def test_raw_replay_detects_changed_rows_and_input_bytes(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7726-TERMINAL: a candidate replays from raw rows."""

    result = subject.compare_contract(DESIGN, ROADMAP)
    raw = tmp_path / "rows.json"
    raw.write_text(json.dumps(result["rows"]))
    assert subject.reduce_raw(raw, DESIGN, ROADMAP) == result
    altered = json.loads(raw.read_text())
    altered[0]["matched"] = False
    raw.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="raw rows differ"):
        subject.reduce_raw(raw, DESIGN, ROADMAP)


def test_blocked_operand_has_full_custody() -> None:
    """SCENARIO-REPORT-7726-CUSTODY: external absence is terminal blocked."""

    row = subject.operand("required_input", "exp7713", "missing.json", "exists", True, False)
    assert row == {
        "check": "required_input",
        "upstream": "exp7713",
        "artifact_path": "missing.json",
        "field": "exists",
        "operator": "==",
        "expected": True,
        "observed": False,
        "passed": False,
    }


def test_candidate_schema_and_cold_custody(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-7726-TERMINAL: required fields and byte replay bind."""

    comparison = subject.compare_contract(DESIGN, ROADMAP)
    raw = tmp_path / "rows.json"
    raw.write_text(json.dumps(comparison["rows"]))
    authority = ROOT / "research-roadmap.yaml"
    value = subject.build_artifact(ROOT, authority, comparison, [], raw, [], 0.0)
    for field in (
        "honest_verdict",
        "verdict_class",
        "flagged_adversarial",
        "gate_check_summary",
        "acceptance_gate_results",
        "rows",
        "sample_size_budget",
        "claim_scope",
        "inference_substrate",
        "inference_substrate_class",
        "MODEL_SPECS",
        "planned_MODEL_SPECS",
        "model_specs",
        "model_invoked",
        "execution_venue",
        "phase_spans",
        "random_seed",
        "reproducibility_checksum",
        "source_artifact_hashes",
        "preconditions_checked",
        "validation_receipts",
        "verifier_is_oracle",
        "field_principles",
        "contract_ready_score",
        "method_map_path",
    ):
        assert field in value, field
    assert value["acceptance_gate_results"]["brier_score"] is None
    assert value["claim_scope"]["fresh_generalization_eligible"] is False
    assert subject.validate_candidate(value, ROOT, raw)
    damaged = deepcopy(value)
    damaged["rows"][0]["matched"] = False
    assert not subject.validate_candidate(damaged, ROOT, raw)
    damaged["reproducibility_checksum"] = subject.checksum(damaged)
    assert not subject.validate_candidate(damaged, ROOT, raw)
    for role, field, replacement in (
        ("planned_output_not_input", "sha256", "wrong"),
        ("current_input", "exists", False),
        ("current_input", "sha256", "wrong"),
    ):
        changed = deepcopy(value)
        next(row for row in changed["source_artifact_hashes"] if row["role"] == role)[field] = (
            replacement
        )
        changed["reproducibility_checksum"] = subject.checksum(changed)
        assert not subject.validate_candidate(changed, ROOT, raw)
    assert not subject.validate_candidate(value, ROOT, tmp_path / "missing-rows.json")
    missing_method = tmp_path / "absent-method.md"
    monkeypatch.setattr(subject, "METHOD", missing_method)
    blocked = subject.build_artifact(
        ROOT, ROOT / "research-roadmap.yaml", comparison, [], raw, [], 0.0
    )
    assert blocked["verdict_class"] == "blocked"
    assert any(row["artifact_path"] == str(missing_method) for row in blocked["gate_check_summary"])


@pytest.mark.parametrize("adversarial_failure", [False, True])
def test_run_and_terminal_reader_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, adversarial_failure: bool
) -> None:
    """SCENARIO-REPORT-7726-TERMINAL: real builder and replay reach publication."""

    for relative in (
        "research-roadmap.yaml",
        str(subject.DESIGN),
        str(subject.METHOD),
        "research-references.md",
        "research-studying.md",
        "ops/exclusion_manifest.yaml",
        "openspec/capabilities/research-reporting/spec.md",
        "openspec/capabilities/research-harnesses/spec.md",
        str(subject.MODULE),
        str(subject.CLI),
        str(subject.TEST),
        "results/experiment_7713_v672_contract_methods.json",
        "results/experiment_7725_v672_capstone.json",
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    monkeypatch.setattr(subject, "ROOT", tmp_path)
    fake = subject.CommandSpec("fake_validation", ("true",), "private")
    monkeypatch.setattr(subject, "validation_plan", lambda *_: [fake])

    def fake_commands(
        _root: Path, commands: list[subject.CommandSpec], **_kw: object
    ) -> list[dict]:
        return [
            {
                "name": item.name,
                "passed": not (adversarial_failure and item.name == "adversarial_verify"),
                "exit_code": int(adversarial_failure and item.name == "adversarial_verify"),
                "command_argv": list(item.argv),
                "log_sha256": "sha256:fixture",
            }
            for item in commands
        ]

    monkeypatch.setattr(subject, "run_commands", fake_commands)
    value = subject.run_experiment(tmp_path, subject.RUN_DATE, subject.RESULT)
    assert (tmp_path / subject.RESULT).is_file()
    assert value["contract_ready_score"] == int(not adversarial_failure)
    assert value["flagged_adversarial"] == adversarial_failure
    assert len(value["contract_mutation_rows"]) == 5
    raw = next((tmp_path / subject.RAW).glob("run-*/rows.json"))
    candidate = tmp_path / subject.RESULT
    assert subject.main(["--cold-validate", str(candidate), "--raw", str(raw)]) == 0
    assert subject.main(["--independent-reduce", str(candidate), "--raw", str(raw)]) == 0
    assert subject.main(["--cold-validate", "missing.json", "--raw", str(raw)]) == 1
    assert subject.validate_candidate(value, tmp_path, raw)
    assert {item.name for item in subject.terminal_plan(tmp_path, candidate, raw)} == {
        "cold_cli_replay",
        "cold_independent_reduction",
        "adversarial_verify",
        "verdict_row_consistency_strict",
    }
    if not adversarial_failure:
        original = subject.validate_candidate
        monkeypatch.setattr(subject, "validate_candidate", lambda *_: False)
        with pytest.raises(RuntimeError, match="raw candidate"):
            subject.run_experiment(tmp_path, subject.RUN_DATE, subject.RESULT)
        calls = iter((True, False))
        monkeypatch.setattr(subject, "validate_candidate", lambda *_: next(calls))
        with pytest.raises(RuntimeError, match="terminal candidate"):
            subject.run_experiment(tmp_path, subject.RUN_DATE, subject.RESULT)
        monkeypatch.setattr(subject, "validate_candidate", original)

        def changed_exact(
            _root: Path, commands: list[subject.CommandSpec], **kw: object
        ) -> list[dict]:
            rows = fake_commands(_root, commands, **kw)
            if Path(str(kw["log_dir"])).name == "exact_terminal_logs":
                rows[0]["passed"] = False
            return rows

        monkeypatch.setattr(subject, "run_commands", changed_exact)
        with pytest.raises(RuntimeError, match="reader outcomes changed"):
            subject.run_experiment(tmp_path, subject.RUN_DATE, subject.RESULT)


def test_validation_plan_and_invalid_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-HARNESS-7726: scoped and selected-byte readers are frozen."""

    plan = subject.validation_plan(ROOT, ROOT / "research-roadmap.yaml", tmp_path)
    names = {item.name for item in plan}
    assert {
        "focused_pytest",
        "changed_module_coverage_report",
        "roadmap_schema",
        "exclusion_manifest",
        "roadmap_gate_audit",
        "arc_orphan_solver",
        "prompt_path",
    } <= names
    assert (tmp_path / "basetemp").is_dir()
    with pytest.raises(ValueError, match="root, date"):
        subject.run_experiment(ROOT, "invalid", subject.RESULT)
    with pytest.raises(ValueError, match="unknown mutation"):
        subject.mutate_roadmap(ROADMAP, "unknown")
    monkeypatch.setattr(subject, "run_experiment", lambda *_: None)
    assert subject.main(["--date", subject.RUN_DATE]) == 0
