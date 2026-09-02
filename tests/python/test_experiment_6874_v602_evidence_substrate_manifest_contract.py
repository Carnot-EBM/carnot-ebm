"""Tests for the append-only V602 evidence contract.

Spec refs: REQ-REPORT-6874 and SCENARIO-REPORT-6874-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from carnot import experiment_6874_v602_evidence_substrate_manifest_contract as mod


REPO_ROOT = Path(__file__).resolve().parents[2]


def _roadmap_task(number: int, *, field: str = "ready_score") -> dict[str, object]:
    """Build one small normalized task row for contract tests."""

    return {
        "order": number - 6873,
        "number": number,
        "task_id": f"exp{number}-task",
        "title": f"Task {number}",
        "deliverable": f"results/experiment_{number}_task.json",
        "gates": (
            []
            if number == 6874
            else [
                {
                    "upstream": f"exp{number - 1}-task",
                    "artifact_field": field,
                    "op": "==",
                    "value": 1,
                }
            ]
        ),
        "prior_failures": [],
        "run_command": f"run experiment {number}",
    }


def test_req_report_6874_owns_fields_and_requested_scenarios() -> None:
    """REQ-REPORT-6874 names the complete artifact and failure contract."""

    text = (REPO_ROOT / mod.SPEC_PATH).read_text(encoding="utf-8")
    section = text[text.index("### REQ-REPORT-6874") :]
    assert all(field in section for field in mod.REQUIRED_ARTIFACT_FIELDS)
    for marker in (
        "SCENARIO-REPORT-6874-MISSING-V601",
        "SCENARIO-REPORT-6874-MISSING-CONDUCTOR",
        "SCENARIO-REPORT-6874-WRAPPED-FIELD",
        "SCENARIO-REPORT-6874-STALE-ARTIFACT",
        "SCENARIO-REPORT-6874-DOCUMENT-ONLY",
        "SCENARIO-REPORT-6874-YAML-ONLY",
        "SCENARIO-REPORT-6874-GATE-TYPO",
        "SCENARIO-REPORT-6874-PRIOR-SUBFIELD",
        "SCENARIO-REPORT-6874-RETIRED-DEPENDENCY",
        "SCENARIO-REPORT-6874-NO-LLM",
        mod.INFERENCE_SUBSTRATE,
        mod.OUTPUT_PATH.as_posix(),
    ):
        assert marker in section


def test_wrapped_field_reads_only_explicit_principle_wrapper() -> None:
    """SCENARIO-REPORT-6874-WRAPPED-FIELD preserves ordinary dictionaries."""

    wrapped = {"principle": "The scalar has a reason.", "value": 1}
    ordinary = {"value": 1, "checks": []}
    assert mod.unwrap_principle(wrapped) == 1
    assert mod.unwrap_principle(ordinary) is ordinary
    assert mod.unwrap_principle("plain") == "plain"


def test_missing_and_stale_v601_artifacts_remain_explicit() -> None:
    """SCENARIO-REPORT-6874-MISSING-V601 and -STALE-ARTIFACT fail closed."""

    tasks = [_roadmap_task(6865)]
    rows = mod.build_v601_terminal_rows(tasks, {}, "", "20260902")
    assert rows[0]["artifact_present"] is False
    assert rows[0]["artifact_sha256"] is None
    assert rows[0]["artifact_fresh"] is False

    stale = {
        "experiment_id": 6865,
        "run_date": "2026-09-01",
        "status": "complete",
        "honest_verdict": "complete_old_result",
        "verdict_class": "positive",
    }
    check = mod.artifact_freshness(stale, 6865, "20260902")
    assert check["passed"] is False
    assert check["expected"] == {"experiment_id": 6865, "run_date": "2026-09-02"}
    assert check["observed"]["run_date"] == "2026-09-01"

    stale["run_date"] = "2026-09-02"
    stale["status"] = "in_progress"
    assert mod.artifact_freshness(stale, 6865, "20260902")["passed"] is False


def test_missing_conductor_row_does_not_hide_present_artifact() -> None:
    """SCENARIO-REPORT-6874-MISSING-CONDUCTOR keeps source states separate."""

    task = _roadmap_task(6865)
    payload = {
        "experiment_id": 6865,
        "run_date": "2026-09-02",
        "status": "complete",
        "honest_verdict": "complete_result",
        "verdict_class": "positive",
        "gate_check_summary": {"passed": True},
    }
    rows = mod.build_v601_terminal_rows([task], {6865: payload}, "", "20260902")
    assert rows[0]["artifact_present"] is True
    assert rows[0]["conductor_terminal_row_present"] is False
    assert rows[0]["source_complete"] is False


def test_design_and_yaml_only_tasks_and_gate_typo_fail_parity() -> None:
    """SCENARIO-REPORT-6874-DOCUMENT-ONLY, -YAML-ONLY, and -GATE-TYPO."""

    design = [_roadmap_task(6874), _roadmap_task(6875, field="ready_score")]
    roadmap = [_roadmap_task(6874), _roadmap_task(6876)]
    rows = mod.compare_task_contracts(design, roadmap)
    assert [row["presence"] for row in rows] == ["both", "document_only", "yaml_only"]
    assert rows[1]["passed"] is False
    assert rows[2]["passed"] is False

    typo = deepcopy(design)
    typo[1]["gates"][0]["artifact_field"] = "raedy_score"
    gate_rows = mod.build_gate_contract_rows(design, typo)
    mismatch = next(row for row in gate_rows if row["downstream_number"] == 6875)
    assert mismatch["expected"]["artifact_field"] == "ready_score"
    assert mismatch["observed"]["artifact_field"] == "raedy_score"
    assert mismatch["passed"] is False


def test_design_parser_reads_tasks_deliverables_gates_and_optional_priors() -> None:
    """REQ-REPORT-6874 parses document claims instead of copying YAML claims."""

    text = """# Roadmap
### Exp6874: First task
Body.

**Deliverable:** `results/experiment_6874_first_task.json`

```yaml
prior_failures:
  - experiment_id: exp6000-old
    verdict: complete_null_old
    addressed_by: A changed method.
    retire_if_same_verdict: true
```

### Exp6875: Second task
Body.

**Deliverable:** `results/experiment_6875_second_task.json`

## Dependency Graph

| Downstream task | Upstream field | Condition |
|---|---|---|
| Exp6875 | `exp6874.first_ready_score` | `== 1` |
"""
    rows = mod.parse_design_tasks(text, 6874, 6875)
    assert [row["task_id"] for row in rows] == ["exp6874-first-task", "exp6875-second-task"]
    assert rows[0]["prior_failures"][0]["experiment_id"] == "exp6000-old"
    assert rows[1]["gates"] == [
        {
            "upstream": "exp6874-first-task",
            "artifact_field": "first_ready_score",
            "op": "==",
            "value": 1,
        }
    ]
    with pytest.raises(ValueError, match="missing deliverable"):
        mod.parse_design_tasks("### Exp6874: Missing\nBody", 6874, 6874)


def test_roadmap_parser_preserves_order_and_rejects_malformed_tasks() -> None:
    """REQ-REPORT-6874 reads executable order, gates, and prior blocks."""

    document = {
        "milestone": "2026.09.602",
        "tasks": [
            {
                "id": "exp6874-first-task",
                "title": "First task",
                "deliverable": "results/experiment_6874_first_task.json",
                "prompt": "Run command: run first",
                "gated_on": [],
                "prior_failures": [],
            }
        ],
    }
    rows = mod.parse_roadmap_tasks(document)
    assert rows[0]["order"] == 1
    assert rows[0]["run_command"] == "run first"
    with pytest.raises(ValueError, match="tasks list"):
        mod.parse_roadmap_tasks({"tasks": {}})
    with pytest.raises(ValueError, match="malformed roadmap task"):
        mod.parse_roadmap_tasks({"tasks": ["bad"]})


def test_prior_failure_missing_subfield_and_changed_verdict_fail() -> None:
    """SCENARIO-REPORT-6874-PRIOR-SUBFIELD checks exact primary verdicts."""

    task = _roadmap_task(6874)
    task["prior_failures"] = [
        {
            "experiment_id": "exp6000-old",
            "verdict": "changed_verdict",
            "retire_if_same_verdict": True,
        }
    ]
    rows = mod.build_prior_failure_contract_rows(
        [task], {6000: {"honest_verdict": "complete_null_old"}}
    )
    assert rows[0]["missing_subfields"] == ["addressed_by"]
    assert rows[0]["verdict_exact"] is False
    assert rows[0]["passed"] is False

    task["prior_failures"] = [None]
    malformed = mod.build_prior_failure_contract_rows([task], {})
    assert malformed[0]["missing_subfields"] == [
        "experiment_id",
        "verdict",
        "addressed_by",
        "retire_if_same_verdict",
    ]


def test_retired_dependency_is_detected_by_normalized_experiment_id() -> None:
    """SCENARIO-REPORT-6874-RETIRED-DEPENDENCY rejects a dead gate edge."""

    manifest = {
        "retired": [{"experiment_id": 6000}],
        "retired_extras": [
            {
                "experiment_ids": ["exp6001-old"],
                "un_retired_experiment_ids": ["exp6001-old"],
            }
        ],
    }
    retired = mod.retired_experiment_ids(manifest)
    assert retired == {6000}
    task = _roadmap_task(6875)
    task["gates"][0]["upstream"] = "exp6000-old"
    rows = mod.build_retired_dependency_rows([task], retired)
    assert rows == [
        {
            "downstream_task_id": "exp6875-task",
            "upstream_task_id": "exp6000-old",
            "upstream_experiment_id": 6000,
            "retired": True,
            "passed": False,
        }
    ]


def test_short_artifact_requires_mechanical_no_llm_proof() -> None:
    """SCENARIO-REPORT-6874-NO-LLM refuses prose-only substrate aliases."""

    safe_source = """from pathlib import Path
def run():
    return Path('source.json').read_text()
"""
    payload = {
        "inference_substrate": "arbitrary_prose_no_llm",
        "generated_answer_count": 0,
        "token_likelihood_call_count": 0,
        "rows": [{"source": "one"}],
    }
    safe = mod.inspect_no_llm_evidence(safe_source, payload, "python reducer.py")
    assert safe["mechanically_no_new_llm_inference"] is True
    assert safe["eligible"] is True
    assert safe["model_call_count"] == 0
    assert safe["gpu_receipt_count"] == 0
    assert safe["source_row_count"] == 1

    risky = mod.inspect_no_llm_evidence(
        "from llama_cpp import Llama\ndef run():\n    return Llama(model_path='x')()\n",
        payload,
        "python live.py",
    )
    assert risky["mechanically_no_new_llm_inference"] is False
    assert risky["eligible"] is False
    assert "llama_cpp" in risky["risk_markers"]

    tokenizer_only = mod.inspect_no_llm_evidence(
        "from llama_cpp import Llama\ndef run():\n    return Llama(model_path='x', vocab_only=True).tokenize(b'x')\n",
        payload,
        "python tokenizer.py",
    )
    assert tokenizer_only["mechanically_no_new_llm_inference"] is True

    unknown_count = deepcopy(payload)
    unknown_count.pop("generated_answer_count")
    unknown_count.pop("token_likelihood_call_count")
    assert mod.inspect_no_llm_evidence(safe_source, unknown_count, "python reducer.py")[
        "eligible"
    ] is True


def test_activation_copy_receipt_proves_copy_then_delete() -> None:
    """REQ-REPORT-6874 binds the staged, activated, and deleted Git objects."""

    outputs = {
        ("rev-parse", "stage"): "stage-full",
        ("rev-parse", "activate"): "activate-full",
        ("rev-parse", "delete"): "delete-full",
        ("rev-parse", "stage:research-roadmap-next.yaml"): "same-blob",
        ("rev-parse", "activate^:research-roadmap-next.yaml"): "same-blob",
        ("rev-parse", "activate:research-roadmap.yaml"): "same-blob",
        ("rev-parse", "delete^:research-roadmap-next.yaml"): "same-blob",
        ("rev-parse", "HEAD:research-roadmap.yaml"): "same-blob",
        ("cat-file", "-e", "delete:research-roadmap-next.yaml"): None,
        ("merge-base", "--is-ancestor", "stage", "activate"): "",
        ("merge-base", "--is-ancestor", "activate", "delete"): "",
    }

    def fake_git(_root: Path, args: tuple[str, ...], allow_failure: bool = False) -> str | None:
        assert allow_failure is (
            args[:2] == ("cat-file", "-e") or args[:2] == ("merge-base", "--is-ancestor")
        )
        return outputs[args]

    receipt = mod.build_activation_copy_receipt(
        Path("."),
        stage_commit="stage",
        activation_commit="activate",
        deletion_commit="delete",
        run_git=fake_git,
    )
    assert receipt["copy_equal"] is True
    assert receipt["deleted_after_copy"] is True
    assert receipt["passed"] is True

    def broken_git(_root: Path, _args: tuple[str, ...], allow_failure: bool = False) -> str | None:
        del allow_failure
        raise RuntimeError("missing git evidence")

    blocked = mod.build_activation_copy_receipt(Path("."), run_git=broken_git)
    assert blocked["passed"] is False
    assert blocked["error"] == "missing git evidence"


def test_real_v601_and_v602_sources_produce_valid_blocked_contract() -> None:
    """REQ-REPORT-6874 preserves all evidence and the real four-versus-eleven mismatch."""

    artifact = mod.build_artifact(REPO_ROOT, "20260902")
    assert mod.validate_artifact(artifact) == []
    assert len(artifact["v601_terminal_task_rows"]) == 9
    assert all(row["source_complete"] for row in artifact["v601_terminal_task_rows"])
    assert artifact["fixed_sequence_semantic_branch_closed"] is True
    assert artifact["reliability_update_branch_closed"] is True
    assert artifact["v602_activation_copy_receipt"]["passed"] is True
    assert artifact["v602_evidence_contract_ready_score"] == 0
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == (
        "complete_blocked_v602_evidence_substrate_manifest_contract"
    )
    parity = artifact["v602_document_yaml_parity_rows"]
    assert len(parity) == 11
    assert sum(row["presence"] == "both" for row in parity) == 4
    assert sum(row["presence"] == "document_only" for row in parity) == 7
    assert [row["number"] for row in artifact["v601_unexecuted_design_task_rows"]] == [
        6874,
        6875,
        6876,
        6877,
    ]
    assert all(
        row["execution_state"] == "never_executed"
        for row in artifact["v601_unexecuted_design_task_rows"]
    )
    assert artifact["gate_check_summary"]["failed_check"] == "v602_document_yaml_parity"
    assert set(mod.REQUIRED_ARTIFACT_FIELDS).issubset(artifact["field_principles"])
    gate_fields = {
        row["artifact_field"] for row in artifact["v602_gate_contract_rows"]
    }
    assert gate_fields.issubset(artifact["field_principles"])


def test_stored_and_fresh_adversarial_disagreements_are_preserved() -> None:
    """REQ-REPORT-6874 does not overwrite a stored verifier disposition."""

    payloads = {
        6865: {
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "OLD", "severity": "critical"}],
        }
    }
    reports = {6865: {"flags": [{"kind": "NEW", "severity": "warn", "detail": "new"}]}}
    rows = mod.build_stored_vs_fresh_adversarial_rows([_roadmap_task(6865)], payloads, reports)
    assert rows[0]["stored_flag_kinds"] == ["OLD"]
    assert rows[0]["fresh_flag_kinds"] == ["NEW"]
    assert rows[0]["disagreement"] is True


def test_validator_and_checksum_reject_tampering() -> None:
    """REQ-REPORT-6874 rejects missing fields, false readiness, and checksum drift."""

    artifact = mod.build_artifact(REPO_ROOT, "20260902")

    def errors_for(**changes: object) -> list[str]:
        changed = deepcopy(artifact)
        changed.update(changes)
        changed["reproducibility_checksum"] = mod.reproducibility_checksum(changed)
        return mod.validate_artifact(changed)

    missing = deepcopy(artifact)
    missing.pop("rows")
    missing["reproducibility_checksum"] = mod.reproducibility_checksum(missing)
    assert "missing_required_fields:rows" in mod.validate_artifact(missing)
    assert "invalid_inference_substrate" in errors_for(inference_substrate="wrong")
    assert "verifier_is_oracle_must_be_false" in errors_for(verifier_is_oracle=True)
    assert "invalid_verdict_class" in errors_for(verdict_class="unknown")
    assert "honest_verdict_not_terminal" in errors_for(honest_verdict="blocked")
    assert "readiness_recomputation_mismatch" in errors_for(v602_evidence_contract_ready_score=1)
    assert "field_principles_missing" in errors_for(field_principles={})

    checksum = deepcopy(artifact)
    checksum["status"] = "tampered"
    assert "reproducibility_checksum_mismatch" in mod.validate_artifact(checksum)


def test_source_readers_and_cli_write_only_the_requested_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-6874 keeps malformed input and output behavior deterministic."""

    malformed = tmp_path / "malformed.json"
    malformed.write_text("[]", encoding="utf-8")
    assert mod.read_json_object(malformed) == (None, "json_object_required")
    missing = tmp_path / "missing.json"
    assert mod.read_json_object(missing)[1] == "missing"
    malformed.write_text("{", encoding="utf-8")
    assert str(mod.read_json_object(malformed)[1]).startswith("invalid_json:")

    output = tmp_path / "nested" / "artifact.json"
    assert mod.main(["--date", "20260902", "--output", str(output)]) == 0
    written = json.loads(output.read_text(encoding="utf-8"))
    assert mod.validate_artifact(written) == []

    invalid = deepcopy(written)
    invalid["honest_verdict"] = "bad"
    monkeypatch.setattr(mod, "build_artifact", lambda *_args, **_kwargs: invalid)
    assert mod.main(["--date", "20260902", "--output", str(tmp_path / "bad.json")]) == 1

    with pytest.raises(SystemExit):
        mod.main(["--date", "bad-date", "--output", str(tmp_path / "date.json")])


def test_yaml_source_is_parseable_and_exact_v602_milestone_is_required() -> None:
    """REQ-REPORT-6874 validates YAML shape and the activated milestone."""

    document = yaml.safe_load((REPO_ROOT / mod.ROADMAP_PATH).read_text(encoding="utf-8"))
    assert document["milestone"] == mod.V602_MILESTONE
    assert mod.parse_roadmap_tasks(document)
    assert mod.milestone_check(document)["passed"] is True
    assert mod.milestone_check({"milestone": "2026.09.601"})["passed"] is False


def test_req_report_6874_helper_failure_edges_are_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-6874 fails closed on malformed or absent source shapes."""

    assert mod.sha256_path(tmp_path / "absent") is None
    assert mod._derive_verdict_class({"honest_verdict": "complete_null_result"}) == "null"
    assert mod._derive_verdict_class({}) == "partial"
    assert mod._gate_result({"gates_evaluated": [None]}) == {"passed": True, "rows": []}
    assert mod._gate_result({"gate_check_summary": "missing"}) == {
        "passed": None,
        "summary": "missing",
    }
    assert mod._extract_run_command("") is None
    assert mod._parse_condition("invalid condition") == ("invalid condition", None)

    with pytest.raises(ValueError, match="malformed roadmap task"):
        mod.parse_roadmap_tasks({"tasks": [{}]})

    design = """### Exp6874: In range
**Deliverable:** `results/experiment_6874_in_range.json`

| Downstream task | Upstream field | Condition |
|---|---|---|
| Exp6875 | `exp6874.ready_score` | `== 1` |
"""
    assert mod.parse_design_tasks(design, 6874, 6874)[0]["gates"] == []
    invalid_source = mod.inspect_no_llm_evidence("def broken(:", {}, "python broken.py")
    assert invalid_source["risk_markers"] == ["invalid_python"]
    assert mod.retired_experiment_ids({"retired": [None]}) == set()

    assert mod._run_git(tmp_path, ("rev-parse", "HEAD"), True) is None
    with pytest.raises(RuntimeError):
        mod._run_git(tmp_path, ("rev-parse", "HEAD"), False)

    monkeypatch.setattr(mod.importlib.util, "spec_from_file_location", lambda *_args: None)
    with pytest.raises(RuntimeError, match="import failed"):
        mod._load_verifier(REPO_ROOT)


def test_req_report_6874_yaml_and_fresh_report_failure_edges(tmp_path: Path) -> None:
    """REQ-REPORT-6874 reports missing and malformed YAML and deliverables."""

    missing = tmp_path / "missing.yaml"
    assert mod._load_yaml_mapping(missing) == (None, "missing")
    malformed = tmp_path / "malformed.yaml"
    malformed.write_text("[", encoding="utf-8")
    assert str(mod._load_yaml_mapping(malformed)[1]).startswith("invalid_yaml:")
    malformed.write_text("[]", encoding="utf-8")
    assert mod._load_yaml_mapping(malformed) == (None, "yaml_mapping_required")

    mod._VERIFY_CACHE.clear()
    assert mod._fresh_reports(tmp_path, [_roadmap_task(6865)], lambda _path: {}) == {}


def test_req_report_6874_builder_preserves_source_and_verifier_failures(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-6874 emits a blocked manifest when primary evidence cannot load."""

    def broken_git(
        _root: Path, _args: tuple[str, ...], allow_failure: bool = False
    ) -> str | None:
        del allow_failure
        raise RuntimeError("missing durable receipt")

    blocked = mod.build_artifact(tmp_path, "20260902", verifier=lambda _path: {}, run_git=broken_git)
    assert blocked["verdict_class"] == "blocked"
    source_errors = blocked["preconditions_checked"]["checks"][0]["observed"]
    assert any(
        row["source"] == "v601_git_sources" for row in source_errors
    )

    def nonmapping_git(
        _root: Path, _args: tuple[str, ...], allow_failure: bool = False
    ) -> str | None:
        del allow_failure
        return "[]"

    nonmapping = mod.build_artifact(
        tmp_path, "20260902", verifier=lambda _path: {}, run_git=nonmapping_git
    )
    failures = nonmapping["preconditions_checked"]["failed_checks"]
    assert any("not a mapping" in str(row["observed"]) for row in failures)

    tasks = [_roadmap_task(6865), _roadmap_task(6866)]
    present = tmp_path / str(tasks[1]["deliverable"])
    present.parent.mkdir(parents=True, exist_ok=True)
    present.write_text(
        json.dumps(
            {
                "experiment_id": 6866,
                "run_date": "2026-09-02",
                "status": "complete",
                "honest_verdict": "complete_result",
                "verdict_class": "positive",
            }
        ),
        encoding="utf-8",
    )

    roadmap_text = yaml.safe_dump(
        {
            "milestone": mod.V601_MILESTONE,
            "tasks": [
                {
                    "id": task["task_id"],
                    "title": task["title"],
                    "deliverable": task["deliverable"],
                    "gated_on": [],
                    "prior_failures": [],
                    "prompt": "",
                }
                for task in tasks
            ],
        }
    )

    def fixture_git(
        _root: Path, args: tuple[str, ...], allow_failure: bool = False
    ) -> str | None:
        del allow_failure
        if args[:1] == ("show",) and args[1].endswith(":research-roadmap.yaml"):
            return roadmap_text
        if args[:1] in {("show",), ("ls-tree",)}:
            return ""
        raise RuntimeError("activation receipt unavailable")

    def failing_verifier(_path: Path) -> dict[str, object]:
        raise RuntimeError("verifier failed")

    mod._VERIFY_CACHE.clear()
    source_failures = mod.build_artifact(
        tmp_path, "20260902", verifier=failing_verifier, run_git=fixture_git
    )
    source_checks = source_failures["preconditions_checked"]["checks"][0]["observed"]
    assert {row["error"] for row in source_checks} >= {"missing", "verifier failed"}


def test_req_report_6874_positive_and_validator_ready_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-6874 keeps the positive terminal shape coupled to readiness."""

    monkeypatch.setattr(mod, "_ready_from_artifact", lambda _artifact: True)
    artifact = mod.build_artifact(REPO_ROOT, "20260902", verifier=lambda _path: {"flags": []})
    assert artifact["status"] == "complete"
    assert artifact["verdict_class"] == "positive"

    artifact["verdict_class"] = "blocked"
    artifact["reproducibility_checksum"] = mod.reproducibility_checksum(artifact)
    assert "ready_verdict_class_mismatch" in mod.validate_artifact(artifact)
