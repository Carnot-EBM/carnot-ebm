"""Focused tests for the V598 independent capstone.

Spec refs: REQ-RESEARCH-6847,
SCENARIO-RESEARCH-6847-MISSING-ARTIFACTS,
SCENARIO-RESEARCH-6847-HASH-AND-GATE-REPLAY,
SCENARIO-RESEARCH-6847-CLOSED-VERDICTS,
SCENARIO-RESEARCH-6847-BRANCH-INDEPENDENCE, and
SCENARIO-RESEARCH-6847-RETIREMENT.
"""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest
import yaml


REPO = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO / "python/carnot/experiment_6847_v598_independent_capstone.py"
SPEC = importlib.util.spec_from_file_location("exp6847_under_test", MODULE_PATH)
assert SPEC and SPEC.loader
exp = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = exp
SPEC.loader.exec_module(exp)


@pytest.fixture(scope="module")
def current_inputs() -> tuple[list[dict], dict[str, dict]]:
    """Load the current V598 roadmap and all available artifacts."""

    planned = exp.load_planned_tasks(REPO)
    sources = exp.load_source_artifacts(REPO, planned)
    return planned, sources


def _replace_source_payload(
    sources: dict[str, dict], task_id: str, payload: dict[str, object]
) -> dict[str, dict]:
    """Replace one payload without copying the large sealed-memory artifact."""

    changed = dict(sources)
    changed[task_id] = dict(changed[task_id])
    changed[task_id]["payload"] = payload
    return changed


def test_req_research_6847_spec_declares_terminal_contract() -> None:
    """REQ-RESEARCH-6847: the spec owns the artifact contract."""

    section = (
        (REPO / exp.REPORT_SPEC_PATH).read_text(encoding="utf-8").split("REQ-RESEARCH-6847", 1)[1]
    )
    anchors = set(exp.spec_anchors(section))

    assert {
        "REQ-RESEARCH-6847",
        "SCENARIO-RESEARCH-6847-MISSING-ARTIFACTS",
        "SCENARIO-RESEARCH-6847-HASH-AND-GATE-REPLAY",
        "SCENARIO-RESEARCH-6847-CLOSED-VERDICTS",
        "SCENARIO-RESEARCH-6847-BRANCH-INDEPENDENCE",
        "SCENARIO-RESEARCH-6847-RETIREMENT",
    } <= anchors
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert field in section
    assert exp.INFERENCE_SUBSTRATE in section
    assert exp.RESULT_PATH.as_posix() in section


def test_scenario_research_6847_missing_artifacts_do_not_block_capstone(
    current_inputs: tuple[list[dict], dict[str, dict]],
) -> None:
    """SCENARIO-RESEARCH-6847-MISSING-ARTIFACTS keeps Exp6838 visible."""

    planned, sources = current_inputs
    inventory = exp.build_artifact_inventory(REPO, planned, sources)
    artifact = exp.build_artifact(
        REPO,
        run_date="20260901",
        duration_s=0.25,
        planned=planned,
        sources=sources,
    )

    by_task = {row["task_id"]: row for row in inventory}
    assert [row["task_id"] for row in planned] == list(exp.EXPECTED_TASK_IDS)
    assert by_task["exp6838"]["artifact_state"] == "missing"
    assert by_task["exp6838"]["verdict_class"] == "blocked"
    assert by_task[exp.CAPSTONE_TASK_ID]["artifact_state"] == "current_synthesis"
    assert artifact["preconditions_checked"]["active_roadmap_present"] is True
    assert artifact["preconditions_checked"]["prompt_next_roadmap_present"] is False
    assert artifact["typed_compatibility_disposition"]["verdict_class"] == "blocked"
    assert artifact["v598_disposition_complete_score"] == 1
    assert artifact["honest_verdict"].startswith("complete_")
    assert exp.validate_artifact(artifact) == []


def test_scenario_research_6847_recomputes_gates_and_detects_bad_hashes(
    current_inputs: tuple[list[dict], dict[str, dict]],
) -> None:
    """SCENARIO-RESEARCH-6847-HASH-AND-GATE-REPLAY catches hash drift."""

    planned, sources = current_inputs
    payload = deepcopy(sources["exp6836"]["payload"])
    payload["source_artifact_hashes"]["exp6835"]["file_sha256"] = "sha256:bad"
    changed = _replace_source_payload(sources, "exp6836", payload)

    rows = exp.build_rows(REPO, planned, changed)
    by_criterion = {row["criterion_id"]: row for row in rows}

    assert by_criterion["exp6836.source_hash.exp6835"]["status"] == "failed"
    assert by_criterion["exp6836.source_hash.exp6835"]["verdict_class"] == "disqualified"
    assert by_criterion["exp6837.obligation_compatibility_stream_ready_score"]["status"] == "failed"
    assert (
        by_criterion["exp6837.obligation_compatibility_stream_ready_score"]["observed_value"] == 0
    )
    assert by_criterion["exp6842.continuous_self_learning_ready_score"]["observed_value"] == 0


def test_scenario_research_6847_detects_producer_aggregate_disagreement(
    current_inputs: tuple[list[dict], dict[str, dict]],
) -> None:
    """SCENARIO-RESEARCH-6847-HASH-AND-GATE-REPLAY does not trust aggregates."""

    planned, sources = current_inputs
    payload = deepcopy(sources["exp6846"]["payload"])
    payload["exact_agreement_results"]["agreement_rate"] = 0.5
    changed = _replace_source_payload(sources, "exp6846", payload)

    rows = exp.build_rows(REPO, planned, changed)
    row = {row["criterion_id"]: row for row in rows}["exp6846.exact_agreement_rate"]

    assert row["observed_value"] == 1.0
    assert row["producer_value"] == 0.5
    assert row["producer_agrees"] is False
    assert row["status"] == "producer_disagreement"
    assert row["verdict_class"] == "partial"


def test_scenario_research_6847_closed_classes_and_circularity_are_preserved(
    current_inputs: tuple[list[dict], dict[str, dict]],
) -> None:
    """SCENARIO-RESEARCH-6847-CLOSED-VERDICTS keeps circularity explicit."""

    planned, sources = current_inputs
    payload = deepcopy(sources["exp6836"]["payload"])
    payload["verdict_class"] = "positive"
    payload["verifier_is_oracle"] = True
    changed = _replace_source_payload(sources, "exp6836", payload)

    inventory = exp.build_artifact_inventory(REPO, planned, changed)
    classes = {row["verdict_class"] for row in inventory}

    assert classes <= exp.CLOSED_VERDICT_CLASSES
    assert {row["task_id"]: row for row in inventory}["exp6836"]["verdict_class"] == (
        "circular_positive"
    )
    assert exp.classify_source_record({"artifact_state": "invalid", "payload": None}) == (
        "disqualified"
    )
    assert exp.classify_source_record({"artifact_state": "missing", "payload": None}) == ("blocked")


def test_scenario_research_6847_branch_independence_and_retirement(
    current_inputs: tuple[list[dict], dict[str, dict]],
) -> None:
    """SCENARIO-RESEARCH-6847-BRANCH-INDEPENDENCE keeps branches local."""

    planned, sources = current_inputs
    artifact = exp.build_artifact(
        REPO,
        run_date="20260901",
        duration_s=0.25,
        planned=planned,
        sources=sources,
    )

    assert artifact["arc_shadow_monitor_disposition"]["ready_score"] == 1
    assert artifact["tool_gap_disposition"]["verdict_class"] == "blocked"
    assert artifact["supervisor_credit_disposition"]["verdict_class"] == "blocked"
    assert artifact["continuous_self_learning_disposition"]["verdict_class"] == "null"
    assert artifact["learning_kernel_disposition"]["ready_score"] == 1
    assert artifact["typed_compatibility_disposition"]["blocking_criteria"] == [
        "exp6837.obligation_compatibility_stream_ready_score",
        "exp6838.artifact_present",
    ]
    assert {row["disposition_id"] for row in artifact["next_milestone_recommendation"]} == {
        "typed_compatibility",
        "learning_kernel",
        "continuous_self_learning",
        "supervisor_credit",
        "tool_gap",
        "arc_shadow_monitor",
    }
    assert not any("policy benefit" in claim for claim in artifact["allowed_claims"])
    assert any("level solve" in claim for claim in artifact["forbidden_claims"])

    retirement = artifact["prior_failure_retirement_decisions"]
    assert retirement
    assert all("retirement_recommended" in row for row in retirement)
    assert any(row["task_id"] == "exp6838" for row in retirement)


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("verdict_class", "positive"),
        ("verifier_is_oracle", True),
        ("rows", []),
        ("artifact_inventory", []),
        ("field_principles", {}),
        ("source_artifact_hashes", {}),
        ("v598_disposition_complete_score", 0),
        ("reproducibility_checksum", "sha256:bad"),
    ],
)
def test_req_research_6847_validator_fails_closed(
    current_inputs: tuple[list[dict], dict[str, dict]],
    field: str,
    bad: object,
) -> None:
    """REQ-RESEARCH-6847 validates the terminal artifact shape."""

    planned, sources = current_inputs
    artifact = exp.build_artifact(
        REPO,
        run_date="20260901",
        duration_s=0.25,
        planned=planned,
        sources=sources,
    )
    changed = deepcopy(artifact)
    changed[field] = bad
    if field != "reproducibility_checksum":
        changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)

    assert exp.validate_artifact(changed)


def _design_text(
    ids: tuple[str, ...] = exp.EXPECTED_TASK_IDS, milestone: str = exp.MILESTONE
) -> str:
    lines = [f"**Milestone:** `{milestone}`"]
    for task_id in ids:
        lines.extend(
            [
                f"### Exp {task_id.removeprefix('exp')}: Synthetic {task_id}",
                f"**Deliverable:** `{exp.TASK_PATHS[task_id]}`",
            ]
        )
    return "\n".join(lines)


def _manifest(deliverable_override: tuple[str, str] | None = None) -> dict[str, object]:
    tasks = []
    for full_id in exp.FULL_TASK_IDS:
        task_id = exp.short_task_id(full_id)
        deliverable = exp.TASK_PATHS[task_id]
        if deliverable_override and deliverable_override[0] == task_id:
            deliverable = deliverable_override[1]
        tasks.append(
            {
                "id": full_id,
                "title": f"Synthetic {task_id}",
                "deliverable": deliverable,
                "prior_failures": [],
            }
        )
    return {
        "milestone": exp.MILESTONE,
        "milestone_doc": exp.DESIGN_PATH.as_posix(),
        "tasks": tasks,
    }


def _write_plan_root(root: Path, manifest: object, design: str) -> None:
    (root / exp.ACTIVE_ROADMAP_PATH).write_text(yaml.safe_dump(manifest), encoding="utf-8")
    (root / exp.DESIGN_PATH.parent).mkdir(parents=True, exist_ok=True)
    (root / exp.DESIGN_PATH).write_text(design, encoding="utf-8")


def _resign(artifact: dict[str, object]) -> dict[str, object]:
    artifact["reproducibility_checksum"] = exp.reproducibility_checksum(artifact)
    return artifact


def test_req_research_6847_defensive_manifest_and_cli_edges(tmp_path: Path) -> None:
    """REQ-RESEARCH-6847 fails closed for malformed manifests and wrappers."""

    with pytest.raises(ValueError, match="invalid V598 task id"):
        exp.short_task_id("bad-task")
    with pytest.raises(ValueError, match="invalid V598 task id"):
        exp.short_task_id("exp0000")
    assert (
        exp._next_deliverable(
            ["### Exp 6847: Present", "**Deliverable:** `results/example.json`"], 0
        )
        == "results/example.json"
    )
    with pytest.raises(ValueError, match="deliverable missing"):
        exp._next_deliverable(["### Exp 6847: Missing", "### Exp 6848: Next"], 0)
    with pytest.raises(ValueError, match="milestone missing"):
        exp.parse_design_tasks("# no milestone")
    assert exp.sha256_file(tmp_path / "absent.json") is None

    _write_plan_root(tmp_path, [], _design_text())
    with pytest.raises(ValueError, match="mapping with tasks"):
        exp.load_planned_tasks(tmp_path)
    bad_milestone = _manifest()
    bad_milestone["milestone"] = "2026.09.999"
    _write_plan_root(tmp_path, bad_milestone, _design_text())
    with pytest.raises(ValueError, match="expected V598 roadmap"):
        exp.load_planned_tasks(tmp_path)
    bad_doc = _manifest()
    bad_doc["milestone_doc"] = "wrong.md"
    _write_plan_root(tmp_path, bad_doc, _design_text())
    with pytest.raises(ValueError, match="expected V598 design milestone document path"):
        exp.load_planned_tasks(tmp_path)
    _write_plan_root(tmp_path, _manifest(), _design_text(milestone="2026.09.999"))
    with pytest.raises(ValueError, match="expected V598 design"):
        exp.load_planned_tasks(tmp_path)
    _write_plan_root(tmp_path, _manifest(), _design_text(exp.EXPECTED_TASK_IDS[:-1]))
    with pytest.raises(ValueError, match="Exp6835 through Exp6847"):
        exp.load_planned_tasks(tmp_path)
    bad_manifest = _manifest()
    bad_manifest["tasks"] = bad_manifest["tasks"][:-1]
    _write_plan_root(tmp_path, bad_manifest, _design_text())
    with pytest.raises(ValueError, match="exact 13 tasks"):
        exp.load_planned_tasks(tmp_path)
    bad_manifest = _manifest()
    bad_manifest["tasks"][0] = []
    _write_plan_root(tmp_path, bad_manifest, _design_text())
    with pytest.raises(ValueError, match="task must be a mapping"):
        exp.load_planned_tasks(tmp_path)
    bad_manifest = _manifest()
    bad_manifest["tasks"][0], bad_manifest["tasks"][1] = (
        bad_manifest["tasks"][1],
        bad_manifest["tasks"][0],
    )
    _write_plan_root(tmp_path, bad_manifest, _design_text())
    with pytest.raises(ValueError, match="ordered Exp6835 through Exp6847"):
        exp.load_planned_tasks(tmp_path)
    _write_plan_root(tmp_path, _manifest(("exp6835", "results/wrong.json")), _design_text())
    with pytest.raises(ValueError, match="deliverable mismatch"):
        exp.load_planned_tasks(tmp_path)
    design_with_bad_deliverable = _design_text().replace(
        exp.TASK_PATHS["exp6835"], "results/wrong.json", 1
    )
    _write_plan_root(tmp_path, _manifest(), design_with_bad_deliverable)
    with pytest.raises(ValueError, match="deliverable mismatch"):
        exp.load_planned_tasks(tmp_path)

    planned = [
        {"task_id": "exp6835", "path": "missing.json"},
        {"task_id": "exp6836", "path": "bad.json"},
        {"task_id": exp.CAPSTONE_TASK_ID, "path": exp.RESULT_PATH.as_posix()},
    ]
    (tmp_path / "bad.json").write_text("[]", encoding="utf-8")
    sources = exp.load_source_artifacts(tmp_path, planned)
    assert sources["exp6835"]["artifact_state"] == "missing"
    assert sources["exp6836"]["artifact_state"] == "invalid"
    assert sources[exp.CAPSTONE_TASK_ID]["artifact_state"] == "current_synthesis"
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    sources = exp.load_source_artifacts(tmp_path, planned)
    assert sources["exp6836"]["artifact_state"] == "invalid"

    target = tmp_path / "artifact.json"
    assert exp.main(["--repo-root", str(REPO), "--output", str(target)]) == 0
    artifact = json.loads(target.read_text(encoding="utf-8"))
    assert exp.validate_artifact(artifact) == []
    assert exp.main(["--validate", "--output", str(target)]) == 0
    target.write_text(json.dumps({"bad": "payload"}), encoding="utf-8")
    assert exp.main(["--validate", "--output", str(target)]) == 1
    assert exp.main(["--validate", "--output", str(tmp_path / "missing.json")]) == 1

    wrapper_path = REPO / "scripts/experiments/experiment_6847_v598_independent_capstone.py"
    saved_path = list(sys.path)
    saved_carnot = sys.modules.get("carnot")
    saved_module = sys.modules.get("carnot.experiment_6847_v598_independent_capstone")
    fake_carnot = types.ModuleType("carnot")
    fake_carnot.__path__ = []
    fake_module = types.ModuleType("carnot.experiment_6847_v598_independent_capstone")
    fake_module.main = exp.main
    sys.modules["carnot"] = fake_carnot
    sys.modules["carnot.experiment_6847_v598_independent_capstone"] = fake_module
    for path in (REPO, REPO / "python"):
        while str(path) in sys.path:
            sys.path.remove(str(path))
    spec = importlib.util.spec_from_file_location("exp6847_wrapper", wrapper_path)
    assert spec and spec.loader
    wrapper = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(wrapper)
    finally:
        sys.path = saved_path
        if saved_carnot is None:
            sys.modules.pop("carnot", None)
        else:
            sys.modules["carnot"] = saved_carnot
        if saved_module is None:
            sys.modules.pop("carnot.experiment_6847_v598_independent_capstone", None)
        else:
            sys.modules["carnot.experiment_6847_v598_independent_capstone"] = saved_module
    assert wrapper.main(["--repo-root", str(REPO), "--output", str(tmp_path / "wrap.json")]) == 0


def test_req_research_6847_defensive_helper_edges(
    current_inputs: tuple[list[dict], dict[str, dict]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-RESEARCH-6847 covers closed fallback and validation branches."""

    assert exp.classify_source_record({"artifact_state": "present", "payload": None}) == (
        "disqualified"
    )
    for text, expected in (
        ("complete_circular_positive_self_oracle", "circular_positive"),
        ("complete_disqualified_bad_hash", "disqualified"),
        ("blocked_gate_check_failed", "blocked"),
        ("partial_incomplete_rows", "partial"),
        ("success_positive_ready", "positive"),
        ("complete_terminal_null", "null"),
    ):
        assert (
            exp.classify_source_record(
                {
                    "artifact_state": "present",
                    "payload": {"status": text, "honest_verdict": text},
                }
            )
            == expected
        )

    assert exp._source_hash_receipts({"source_artifact_hashes": "not-a-map"}) == []
    assert exp._source_hash_receipts({"source_artifact_hashes": {"exp6835": "abc"}}) == [
        {
            "source_id": "exp6835",
            "path": exp.TASK_PATHS["exp6835"],
            "declared_sha256": "sha256:abc",
        }
    ]
    assert exp._path_get({"a": 1}, "a.b") is None
    assert exp._compare_gate(2, ">=", 1) is True
    assert exp._compare_gate(1, "<=", 2) is True
    with pytest.raises(ValueError, match="unsupported roadmap gate"):
        exp._compare_gate(1, "!=", 1)
    assert exp._producer_score({"flag": True}, "flag") == 1
    assert exp._bool_path({"a": 1}, "a", "b") is False
    assert exp._gate_summary_rows("exp6835", {}) == []
    summary_rows = exp._gate_summary_rows(
        "exp6835",
        {
            "gate_check_summary": {
                "checks": [
                    "bad",
                    {"check": "ok", "expected": True, "observed": True, "passed": True},
                ],
                "readiness_gates": {
                    "bad": "skip",
                    "not_ok": {"observed": {"value": 0}, "passed": False},
                },
            }
        },
    )
    assert {row["criterion_id"] for row in summary_rows} == {
        "exp6835.gate.ok",
        "exp6835.readiness_gate.not_ok",
    }
    assert (
        exp._recompute_6845("exp6845", {"obligation_ledger": [], "request_receipt_joins": {}})[1][
            "observed_value"
        ]
        == 0
    )
    assert exp._status_for_blocking(["exp6835.block"], ready_score=1) == "blocked"

    planned, sources = current_inputs
    artifact = exp.build_artifact(
        REPO,
        run_date="20260901",
        duration_s=0.25,
        planned=planned,
        sources=sources,
    )

    changed = deepcopy(artifact)
    changed["field_principles"].pop("schema")
    assert "field_principles" in " ".join(exp.validate_artifact(_resign(changed)))

    changed = deepcopy(artifact)
    changed["artifact_inventory"][0]["verdict_class"] = "bad"
    assert "inventory row" in " ".join(exp.validate_artifact(_resign(changed)))

    changed = deepcopy(artifact)
    changed["rows"][0] = "bad"
    assert "rows must contain mappings" in " ".join(exp.validate_artifact(_resign(changed)))

    changed = deepcopy(artifact)
    changed["rows"][0]["verdict_class"] = "bad"
    assert "criterion row" in " ".join(exp.validate_artifact(_resign(changed)))

    changed = deepcopy(artifact)
    changed["rows"][0]["status"] = "producer_disagreement"
    changed["rows"][0]["producer_agrees"] = True
    assert "producer disagreement" in " ".join(exp.validate_artifact(_resign(changed)))

    changed = deepcopy(artifact)
    changed["learning_kernel_disposition"]["verdict_class"] = "bad"
    assert "learning_kernel_disposition" in " ".join(exp.validate_artifact(_resign(changed)))

    changed = deepcopy(artifact)
    changed["learning_kernel_disposition"]["verdict_class"] = "blocked"
    changed["learning_kernel_disposition"]["blocking_criteria"] = []
    assert "blocked without" in " ".join(exp.validate_artifact(_resign(changed)))

    changed = deepcopy(artifact)
    changed["allowed_claims"] = ["policy benefit overclaim"]
    assert "policy benefit" in " ".join(exp.validate_artifact(_resign(changed)))

    monkeypatch.setattr(exp, "load_planned_tasks", lambda repo_root: [])
    monkeypatch.setattr(exp, "load_source_artifacts", lambda repo_root, planned: {})
    monkeypatch.setattr(exp, "build_artifact", lambda *args, **kwargs: {"bad": "payload"})
    monkeypatch.setattr(exp, "validate_artifact", lambda artifact: ["forced error"])
    assert exp._build_and_write(REPO, REPO / "results/unused-6847-test.json", "20260901") == 1
