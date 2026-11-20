"""REQ-REPORT-9084-V775 / SCENARIO-REPORT-9084-V775: exposed methods stay honest."""

from copy import deepcopy
import json
import os
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import human_methods_9084 as d
from carnot.reporting import human_methods_runner_9084 as runner
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.v709_execution import child


def records() -> list[dict[str, Any]]:
    """Three sibling positions let controls distinguish question and answer identity."""
    return [
        dict(
            item_id=f"q{q}-{g}",
            qid=f"q{q}",
            dataset="private",
            model=g,
            question=f"Question {q}?",
            answer_text=f"Answer {g}",
            ref_str="context",
            annotator1_label=str(q % 2),
            annotator2_label="-1" if q else "1",
        )
        for q in range(3)
        for g in d.old.GENERATORS
    ]


def partition(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    """A small fixed split exercises the same parser without creating study evidence."""
    groups = sorted({d.old.text_hash(d.old.normalized(r["question"])) for r in rows})
    return dict(fit=groups[:1], tune=groups[1:2], test=groups[2:])


def test_protocol() -> None:
    """REQ-REPORT-9084-V775: fixed design cannot respond to observed performance."""
    p = d.protocol()
    assert p["embedding"]["token_limits"] == dict(question=256, answer_text=768)
    assert p["embedding"]["pooling"] == "mean_last_layer_nonpadding"
    assert p["projection"]["seed"] == 775084
    assert p["projection"]["dimensions"] == 27
    assert len(p["arms"]) == 6
    assert p["fit"]["l2"] == 0.01
    assert p["calibration"]["temperatures"] == [0.5, 1, 2, 4]
    assert p["uncertainty"]["draws"] == 10000
    assert p["uncertainty"]["seed"] == 775092
    assert p["uncertainty"]["rerank_half_coverage_each_draw"] is True


def test_parser_and_feature_isolation() -> None:
    """SCENARIO-REPORT-9084-V775: all forbidden cells are inert at the learner boundary."""
    rows = records()
    reduced = d.inventory(rows, partition(rows))
    assert len(reduced["rows"]) == 9
    assert reduced["question_count"] == 3
    assert reduced["annotator2_observed_count"] == 3
    assert reduced["annotator2_missing_count"] == 6
    assert reduced["disagreement_count"] == 3
    assert {r["role"] for r in reduced["rows"]} == {"TRAIN", "CAL", "TEST"}
    original = d.feature_view(rows[0])
    for field in d.PROHIBITED:
        assert d.feature_view(dict(rows[0], **{field: "prohibited mutation"})) == original
    assert d.label("-1") is None
    assert d.label("0") == 0 and d.label("1") == 1
    with pytest.raises(ValueError, match="label_encoding"):
        d.label("2")
    with pytest.raises(ValueError, match="text_fields"):
        d.feature_view(dict(question=17, answer_text="a"))


@pytest.mark.parametrize(
    "mutation",
    ["duplicate", "missing_sibling", "label", "partition", "alias", "generator", "empty"],
)
def test_parser_rejects_incompatible_inputs(mutation: str) -> None:
    """SCENARIO-REPORT-9084-V775: incompatible identity blocks instead of repartitioning."""
    rows = records()
    roles = partition(rows)
    if mutation == "duplicate":
        rows.append(deepcopy(rows[0]))
    elif mutation == "missing_sibling":
        rows.pop()
    elif mutation == "label":
        rows[0]["annotator1_label"] = "unknown"
    elif mutation == "partition":
        roles["test"].append(roles["fit"][0])
    elif mutation == "alias":
        rows[0]["qid"] = "other"
    elif mutation == "generator":
        rows[0]["model"] = "other"
    else:
        rows[0]["question"] = ""
    with pytest.raises(ValueError):
        d.inventory(rows, roles)


def test_actual_retained_sources_and_missing_root(tmp_path: Path) -> None:
    """REQ-REPORT-9084-V775: original bytes inventory fully even when producer custody blocks."""
    value = d.inspect(d.ROOT)
    assert value["question_count"] == 300 and len(value["rows"]) == 900
    assert value["annotator2_observed_count"] == 300
    assert value["annotator2_missing_count"] == 600
    assert value["release_commit"] == d.old.COMMIT
    assert value["historical_failures"]
    assert "effective_independent_groups" not in value
    assert d.inspect(tmp_path)["gate_check_summary"][0]["observed"] == "missing_file"


def test_build_and_rehashed_tamper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-9084-V775: failed owned checks disqualify and replay distrusts summaries."""
    ref = runner.measure(tmp_path / "missing", tmp_path / "raw", "20261115", {})
    value = runner.build(ref, {})
    assert value["verdict_class"] == "disqualified"
    assert value["human_methods_ready_score"] == 0
    monkeypatch.setattr(runner, "receipts_pass", lambda _s, _v: True)
    value = runner.build(ref, {})
    assert value["verdict_class"] == "blocked"
    assert value["MODEL_SPECS"] == [] and value["current_llm_invocations"] == 0
    path = tmp_path / (d.NAME + ".json")
    atomic_json(path, value)
    assert runner.replay(path)["passed"]
    value["human_methods_ready_score"] = 1
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(path, value)
    with pytest.raises(ValueError, match="reduction_drift"):
        runner.replay(path)


def test_actual_foreign_cli(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9084-V775: real CLI replays and rejects tampering without PYTHONPATH."""
    value = runner.build(runner.measure(tmp_path / "missing", tmp_path / "raw", "20261115", {}), {})
    path = tmp_path / (d.NAME + ".json")
    atomic_json(path, value)
    prefix = ["env", "-u", "PYTHONPATH", "-C", str(tmp_path), str(d.ROOT / ".venv/bin/python")]
    config = os.environ.get("COVERAGE_RCFILE")
    if config:
        prefix += ["-m", "coverage", "run", "--rcfile=" + config]
    prefix += [str(d.ROOT / d.CLI)]
    assert child("cold", prefix + ["--cold-replay", str(path)], tmp_path / "logs")["passed"]
    value["completed_count"] = 999
    atomic_json(path, value)
    assert child("tamper", prefix + ["--cold-replay", str(path)], tmp_path / "logs", expected=1)[
        "passed"
    ]
    assert child("date", prefix + ["--date", "bad"], tmp_path / "logs", expected=1)["passed"]


def private_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Private copies exercise a valid producer binding without repairing live historical bytes."""
    root = tmp_path / "inputs"
    pins = {}
    for name in d.PINS:
        value = dict(
            honest_verdict="complete_null_private_control",
            verdict_class="null",
            exposure_scope="private fixture metadata, never a study result",
            release_criterion_ready_score=1,
            methods_ready_score=1,
            learning_audit_ready_score=1,
        )
        if "8389" in name:
            original = json.loads((d.ROOT / "results" / (name + ".json")).read_bytes())
            value["manifest_reference"] = original["manifest_reference"]
            value["code_config_hashes"] = [
                d.ref(Path(r["path"])) for r in original["code_config_hashes"]
            ]
        path = root / "results" / (name + ".json")
        atomic_json(path, value)
        pins[name] = d.ref(path)["sha256"].split(":")[1]
    path = root / "openspec/change-proposals/v726-methods-protocol.json"
    path.parent.mkdir(parents=True)
    path.write_bytes((d.ROOT / "openspec/change-proposals/v726-methods-protocol.json").read_bytes())
    monkeypatch.setattr(d, "PINS", pins)
    return root


def test_positive_control_and_exact_gate_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-9084-V775: readiness follows valid design, while missing and zero differ."""
    root = private_tree(tmp_path, monkeypatch)
    assert d.inspect(root)["blocker"] is None
    reference = runner.measure(root, tmp_path / "raw", "20261115", {})
    monkeypatch.setattr(runner, "receipts_pass", lambda _s, _v: True)
    value = runner.build(reference, {})
    assert value["human_methods_ready_score"] == 1
    assert value["verdict_class"] == "positive"
    assert value["independent_eval_ready_score"] == value["independent_generalization_score"] == 0
    path = root / "results/experiment_8403_v724_released_question_identity.json"
    original = json.loads(path.read_bytes())
    for operand in ["missing_field", 0]:
        changed = deepcopy(original)
        if operand == "missing_field":
            changed.pop("release_criterion_ready_score")
        else:
            changed["release_criterion_ready_score"] = operand
        atomic_json(path, changed)
        d.PINS[path.stem] = d.ref(path)["sha256"].split(":")[1]
        gate = next(
            c
            for c in d.inspect(root)["gate_check_summary"]
            if c["artifact_field"] == "release_criterion_ready_score"
        )
        assert gate["observed"] == operand and gate["passed"] is False


@pytest.mark.parametrize("mutation", ["lineage", "roster", "bytes", "protocol"])
def test_authority_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str) -> None:
    """SCENARIO-REPORT-9084-V775: new metadata cannot authorize changed original bytes."""
    root = private_tree(tmp_path, monkeypatch)
    path = root / "results/experiment_8389_v723_human_label_custody.json"
    value = json.loads(path.read_bytes())
    manifest = json.loads(Path(value["manifest_reference"]["path"]).read_bytes())
    if mutation == "lineage":
        manifest["release_commit"] = "newer"
    elif mutation == "roster":
        manifest["release_files"].pop("README.md")
    elif mutation == "bytes":
        manifest["release_files"]["README.md"]["sha256"] = "changed"
    else:
        (root / "openspec/change-proposals/v726-methods-protocol.json").write_text("{}")
    manifest_path = tmp_path / "manifest.json"
    atomic_json(manifest_path, manifest)
    value["manifest_reference"] = d.ref(manifest_path)
    atomic_json(path, value)
    d.PINS[path.stem] = d.ref(path)["sha256"].split(":")[1]
    assert d.inspect(root)["blocker"] == "source_authority"


def test_replay_operand_protocol_and_source_mutations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-9084-V775: summaries cannot authorize altered primitives or protocols."""
    reference = runner.measure(tmp_path / "missing", tmp_path / "raw", "20261115", {})
    value = runner.build(reference, {})
    path = tmp_path / (d.NAME + ".json")
    changed = deepcopy(value)
    changed["source_references"][0]["sha256"] = "changed"
    atomic_json(path, changed)
    with pytest.raises(ValueError, match="operand_hash"):
        runner.replay(path)
    atomic_json(path, value)
    monkeypatch.setattr(d, "protocol", lambda: {})
    with pytest.raises(ValueError, match="protocol_drift"):
        runner.replay(path)
    monkeypatch.undo()
    monkeypatch.setattr(
        d,
        "inspect",
        lambda _r: dict(
            work=json.loads(Path(reference["path"]).read_bytes())["reduction"],
            gate_check_summary=[],
        ),
    )
    with pytest.raises(ValueError, match="source_reduction_drift"):
        runner.replay(path)


def test_storage_scope_validation_and_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-9084-V775: private storage and owned receipts precede publication."""
    scope = runner.scope_manifest([])
    assert scope["owned_paths"] == d.OWNED
    assert not runner.receipts_pass(scope, {})
    raw = tmp_path / "not_private"
    raw.mkdir(mode=0o755)
    reference = runner.measure(tmp_path / "missing", raw, "20261115", {})
    assert (
        json.loads(Path(reference["path"]).read_bytes())["reduction"]["blocker"]
        == "private_storage"
    )
    output = tmp_path / (d.NAME + ".json")
    args = [
        "--root",
        str(tmp_path / "missing"),
        "--output",
        str(output),
        "--scope",
        str(tmp_path / "absent"),
        "--receipts",
        str(tmp_path / "absent"),
    ]
    assert runner.main(["--date", "bad"]) == 1
    assert runner.main(["--cold-replay", str(tmp_path / "absent")]) == 1
    assert runner.main(args) == 1 and not output.exists()
    monkeypatch.setattr(runner, "receipts_pass", lambda _s, _v: True)
    monkeypatch.setattr(runner, "validate", lambda _p, _l: dict(passed=True))
    assert runner.main(args) == 0
    assert runner.main(["--cold-replay", str(output)]) == 0
    scope_path, validation_path = tmp_path / "scope.json", tmp_path / "validation.json"
    atomic_json(scope_path, scope)
    atomic_json(validation_path, {})
    assert (
        runner.main(args[:-4] + ["--scope", str(scope_path), "--receipts", str(validation_path)])
        == 0
    )


def test_terminal_scanners(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9084-V775: unchanged scanners and real cold CLI inspect actual bytes."""
    value = runner.build(runner.measure(tmp_path / "missing", tmp_path / "raw", "20261115", {}), {})
    path = tmp_path / (d.NAME + ".json")
    atomic_json(path, value)
    assert runner.validate(path, tmp_path / "validators")["passed"]


def test_terminal_inventory_count_controls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-9084-V775: unchanged scanners accept actual inventory counts, including coincident A2 support."""
    reference = runner.measure(d.ROOT, tmp_path / "raw", "20261115", {})
    monkeypatch.setattr(runner, "receipts_pass", lambda _s, _v: True)
    value = runner.build(reference, {})
    from scripts.adversarial_verify import verify_artifact

    path = tmp_path / (d.NAME + ".json")
    atomic_json(path, value)
    report = verify_artifact(path)
    assert not report["flags"]
    assert value["sample_size_budget"]["effective_independent_group_count"] == 300
