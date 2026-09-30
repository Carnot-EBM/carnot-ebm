"""REQ-REPORT-7952-V689: current custody, primitive reduction and publication."""

import gzip
import json
from pathlib import Path

import pytest
import yaml

from carnot.reporting import v689_capstone as cap
from carnot.reporting import v689_capstone_validation as validation
from scripts.experiments import experiment_7952_v689_capstone as cli

ROOT = Path(__file__).resolve().parents[2]


def fixture(root: Path, complete: bool = False) -> tuple[Path, Path]:
    """Use frozen activation bytes so later roadmap changes cannot alter tests."""
    root.mkdir(parents=True, exist_ok=True)
    design, active = root / "design.md", root / "research-roadmap.yaml"
    for path, name in ((design, "design.md"), (active, "active.yaml")):
        path.write_bytes(gzip.decompress((ROOT / f"tests/fixtures/v689/{name}.gz").read_bytes()))
    (root / "research-roadmap-next.yaml").write_bytes(active.read_bytes())
    if complete:
        tasks = yaml.safe_load(active.read_bytes())["tasks"]
        for task in tasks[:-1]:
            gates = {
                g["artifact_field"]: g["value"]
                for t in tasks
                for g in t.get("gated_on", [])
                if g["upstream"] == task["id"] and g["op"] == "=="
            }
            producer(root, task, **gates)
    return design, active


def producer(root: Path, task: dict, **extra: object) -> Path:
    """Fixtures establish mechanical behavior without claiming natural benefit."""
    path = root / task["deliverable"]
    cap.atomic_json(
        path,
        {
            "experiment_id": int(task["id"][3:7]),
            "task_id": task["id"],
            "milestone": "2026.09.689",
            "run_date": "20260930",
            "MODEL_SPECS": task["MODEL_SPECS"],
            "honest_verdict": "complete_null_fixture",
            "verdict_class": "null",
            "flagged_adversarial": False,
            "rows": [],
            **extra,
        },
    )
    return path


def test_current_dispositions_and_cold_replay(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7952-CUSTODY: absence never borrows conductor evidence."""
    design, active = fixture(tmp_path)
    tasks = yaml.safe_load(active.read_bytes())["tasks"]
    producer(tmp_path, tasks[0], verdict_class="blocked")
    producer(tmp_path, tasks[1], verdict_class="disqualified")
    producer(tmp_path, tasks[4])
    cap.atomic_json(
        tmp_path / "results/experiment_7943_energy_fit.json",
        dict(
            blocked_at_layer="conductor_pre_gate",
            gates_evaluated=[
                dict(
                    upstream=tasks[2]["id"],
                    artifact_path="actual-primary.json",
                    artifact_sha256="sha256:old",
                    artifact_field="energy_fit_ready_score",
                    op="==",
                    expected=1,
                    actual=0,
                    passed=False,
                )
            ],
        ),
    )
    value = cap.build_candidate(tmp_path, design, active, "20260930")
    assert len(value["rows"]) == len(value["independent_reduction_rows"]) == 13
    assert [value["rows"][i]["status"] for i in (0, 1, 2, 3, 4, 12)] == [
        "blocked",
        "disqualified",
        "absent",
        "skipped",
        "null",
        "self_administrative",
    ]
    assert value["verdict_class"] == "blocked"
    assert len(value["gap_decisions"]) == 3
    assert value["sample_size_budget"]["completed"] == 13
    assert cap.cold_replay(value, tmp_path, design, active) == []
    assert any(r["decision"] == "retire_unchanged_scope" for r in value["retirement_decisions"])
    raw_path = Path(value["independent_reduction_rows"][0]["primitive_rows_path"])
    old_raw = raw_path.read_bytes()
    raw_path.write_text("{}")
    assert "source_bytes_changed" in cap.cold_replay(value, tmp_path, design, active)
    raw_path.write_bytes(old_raw)
    value["sample_size_budget"]["completed"] = 99
    assert "sample_size_budget_changed" in cap.cold_replay(value, tmp_path, design, active)
    producer(tmp_path, tasks[4], rows=[None])
    assert "source_bytes_changed" in cap.cold_replay(value, tmp_path, design, active)
    value["G1"] = True
    assert "publication_operands_changed" in cap.cold_replay(value, tmp_path, design, active)
    with pytest.raises(ValueError, match="date"):
        cap.build_candidate(tmp_path, design, active, "20260929")


def test_complete_null_and_invalid_authority(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7952-CUSTODY: a valid null can open readiness."""
    design, active = fixture(tmp_path, True)
    value = cap.build_candidate(tmp_path, design, active, "20260930")
    assert value["verdict_class"] == "null"
    assert all(r["eligible"] for r in value["rows"][:-1])
    assert cap.cold_replay(value, tmp_path, design, active) == []
    task = yaml.safe_load(active.read_bytes())["tasks"][0]
    producer(tmp_path, task, task_id="wrong", rows=[None])
    assert (
        cap.build_candidate(tmp_path, design, active, "20260930")["rows"][0]["status"]
        == "disqualified"
    )
    active.write_text("invalid: [")
    assert not cap.build_candidate(tmp_path, design, active, "20260930")["activation_confirmed"]
    active.write_text("milestone: 2026.09.689\ntasks: []\n")
    assert not cap.build_candidate(tmp_path, design, active, "20260930")["activation_confirmed"]


def test_primitive_roles_groups_and_timing() -> None:
    """SCENARIO-REPORT-7952-CUSTODY: preserve role and seed denominators."""
    rows = [
        dict(
            family_id="f",
            source_cluster_id="g",
            role="evaluation",
            status="completed",
            seed=s,
            arm=a,
            probability=p,
            label=1,
            cost=c,
            latency_ms=2,
        )
        for s in (1, 2)
        for a, p, c in (("head", 0.8, 0.2), ("control", 0.5, 0.5))
    ]
    value = cap.reduce_primitives(rows)
    assert value["by_role"]["evaluation"]["brier_by_arm"]["head"] == pytest.approx(0.04)
    assert value["source_groups"] == value["independent_families"] == 1
    assert value["seed_count"] == 2
    assert value["service_timing_ms"]["head"] == 2
    with pytest.raises(ValueError, match="future_label"):
        cap.reduce_primitives([{**rows[0], "feature_fields": ["future_label"]}])
    with pytest.raises(ValueError, match="score_stress"):
        cap.reduce_primitives([{**rows[0], "score_draw_seed": 1, "stress_draw_seed": 1}])
    action = cap.reduce_primitives(
        [
            dict(
                family_id="f",
                arm="head",
                status="completed",
                label=1,
                probability=0.9,
                action="accept",
                started_monotonic_ns=1000,
                ended_monotonic_ns=2001000,
            )
        ]
    )
    assert action["cost_by_arm"]["head"] == 5
    assert action["service_timing_ms"]["head"] == 2
    assert action["unsupported_false_accepts_by_arm"]["head"] == 1


def test_manifest_dates_and_owned_includes(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7952-QUALIFICATION: freeze exact current includes."""
    frozen = validation.manifest(tmp_path)
    by_name = {r["name"]: r for r in frozen["commands"]}
    for name in ("e2e016_fixture", "e2e016_replay"):
        argv = by_name[name]["argv"]
        assert argv[argv.index("--date") + 1] == "20260929"
    assert frozen["coverage_includes"] == list(validation.OWNED)
    assert len(frozen["dependency_hashes"]) > len(validation.OWNED)


def test_real_publication_result_and_rejections(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7952-QUALIFICATION: parse the actual read-only command."""
    spec = dict(
        name="publication_gate",
        argv=[str(ROOT / ".venv/bin/python"), "scripts/publication_gate.py", "--json"],
        expected_exit=0,
        deadline_s=60,
    )
    receipt = validation.run_check(ROOT, spec, tmp_path, tmp_path / "sealed")
    value = {}
    validation.publication_result(value, receipt)
    assert receipt["passed"]
    assert value["paper_ready"] == all(value[k] for k in cap.GATES)
    Path(receipt["log_path"]).write_text("invalid")
    validation.publication_result({}, receipt)
    assert not receipt["passed"]
    failed = dict(passed=False)
    empty = {}
    validation.publication_result(empty, failed)
    assert empty == {}


@pytest.mark.parametrize("fail", (False, True, "replay"))
def test_qualification_and_primary_readers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail: bool | str
) -> None:
    """SCENARIO-REPORT-7952-PUBLICATION: owned failures cannot open readiness."""
    design, active = fixture(tmp_path)
    frozen = validation.manifest(tmp_path / "prepare")
    publication = dict(gates={k: {"pass": False} for k in cap.GATES})

    def child(root: Path, spec: dict, private: Path, durable: Path) -> dict:
        log = durable / (spec["name"] + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        report = dict(flagged_count=0)
        content = json.dumps(publication if spec["name"] == "publication_gate" else report)
        if spec.get("expected_exit") != 0:
            content += str(spec.get("failure_reason", ""))
        log.write_text(content)
        if spec["name"] == "coverage_json":
            cap.atomic_json(
                private / "coverage.json",
                dict(
                    files={
                        name: dict(
                            summary=dict(num_statements=1, covered_lines=1, missing_lines=0),
                            missing_lines=[],
                        )
                        for name in validation.OWNED
                    }
                ),
            )
        return {
            **spec,
            "passed": not (fail and spec["name"] == "unit"),
            "actual_exit": int(fail and spec["name"] == "unit"),
            "log_path": str(log),
            "log_sha256": cap.sha256_file(log),
        }

    monkeypatch.setattr(validation, "run_check", child)
    monkeypatch.setattr(validation, "manifest", lambda private: frozen)
    if fail == "replay":
        monkeypatch.setattr(cap, "cold_replay", lambda *args: ["claim_changed"])
    output = tmp_path / "results/experiment_7952_v689_capstone.json"
    assert validation.qualify(tmp_path, design, active, "20260930", output) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == ("disqualified" if fail else "blocked")
    assert value["capstone_execution_ready_score"] == int(not fail)
    resolution = json.loads(Path(value["primary_resolution_receipt"]["path"]).read_bytes())
    assert resolution["gate_sha256"] == resolution["document_sha256"] == cap.sha256_file(output)
    sidecar = json.loads(Path(value["terminal_validation_sidecar_path"]).read_bytes())
    assert sidecar["candidate_sha256"] == cap.sha256_file(output)


@pytest.mark.parametrize("flags", ([0], [1, 0, 0], [1, 1, 1], [1, 0, 1]))
def test_terminal_changes_are_checked_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flags: list[int]
) -> None:
    """SCENARIO-REPORT-7952-PUBLICATION: recheck final verdict bytes."""
    design, active = fixture(tmp_path)
    value = cap.build_candidate(tmp_path, design, active, "20260930")
    value["capstone_execution_ready_score"] = 1
    calls = []

    def child(root: Path, spec: dict, private: Path, durable: Path) -> dict:
        log = tmp_path / f"{len(calls)}.log"
        flag = flags[min(len(calls) // 2, len(flags) - 1)]
        log.write_text(json.dumps(dict(flagged_count=flag)))
        calls.append(spec)
        return {**spec, "passed": True, "actual_exit": 0, "log_path": str(log)}

    monkeypatch.setattr(validation, "run_check", child)
    output = tmp_path / "results/experiment_7952_v689_capstone.json"
    if flags == [1, 0, 1]:
        with pytest.raises(ValueError, match="unstable"):
            validation.terminal(value, output, tmp_path / "private", tmp_path / "durable")
        return
    validation.terminal(value, output, tmp_path / "private", tmp_path / "durable")
    actual = json.loads(output.read_bytes())
    assert actual["flagged_adversarial"] == bool(flags[-1])
    assert actual["capstone_execution_ready_score"] == int(not flags[0])


def test_cli_routes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-7952-QUALIFICATION: small CLI exposes all owned routes."""
    design, active = fixture(tmp_path)
    output = tmp_path / "results/experiment_7952_v689_capstone.json"
    base = [
        "--date",
        "20260930",
        "--root",
        str(tmp_path),
        "--design",
        str(design),
        "--output",
        str(output),
    ]
    assert cli.main([*base, "--evidence-only"]) == 0
    assert cli.main([*base, "--cold-replay", str(output)]) == 0
    monkeypatch.setattr(validation, "terminal", lambda *args: None)
    assert cli.main([*base, "--terminal-recheck", str(output)]) == 0
    monkeypatch.setattr(validation, "qualify", lambda *args: 0)
    assert cli.main(base) == 0
    with pytest.raises(SystemExit):
        cli.main([])


def test_invalid_terminal_report_and_reader_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7952-PUBLICATION: invalid attestation disqualifies evidence."""
    design, active = fixture(tmp_path)
    value = cap.build_candidate(tmp_path, design, active, "20260930")

    def child(root: Path, spec: dict, private: Path, durable: Path) -> dict:
        log = tmp_path / (spec["name"] + ".log")
        log.write_text('{"flagged_count":"invalid"}')
        return {**spec, "passed": True, "actual_exit": 0, "log_path": str(log)}

    monkeypatch.setattr(validation, "run_check", child)
    output = tmp_path / "results/experiment_7952_v689_capstone.json"
    validation.terminal(value, output, tmp_path / "private", tmp_path / "durable")
    assert json.loads(output.read_text())["verdict_class"] == "disqualified"
    monkeypatch.setattr(
        validation,
        "reader_receipt",
        lambda *args, **kwargs: dict(gate_sha256="drift", document_sha256="drift"),
    )
    with pytest.raises(ValueError, match="primary_reader_drift"):
        validation.terminal(value, output, tmp_path / "private", tmp_path / "durable")


def test_sentence_reconstruction_and_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7952-CUSTODY: sentence events keep original human targets."""
    calls = []
    monkeypatch.setattr(
        cap.sentence,
        "reconstruct",
        lambda value: calls.append(value) or {"label_counts": {"0": 59, "1": 3}},
    )
    value = {"rows": [], "sentence_public_manifest": {"path": "original"}}
    assert cap.sentence_reduction(value)["label_counts"]["1"] == 3
    assert calls == [value]
    assert cap.sentence_reduction({"rows": []}) == {"status": "no_sentence_transport"}
    monkeypatch.setattr(
        cap.sentence,
        "reconstruct",
        lambda value: (_ for _ in ()).throw(ValueError("evaluator_drift")),
    )
    with pytest.raises(ValueError, match="evaluator_drift"):
        cap.sentence_reduction(value)


def test_target_roles_masks_and_restart() -> None:
    """SCENARIO-REPORT-7952-CUSTODY: repeats cannot inflate source independence."""
    rows = [
        dict(
            family_id="f",
            source_cluster_id="g",
            role="evaluation",
            arm=a,
            status="completed",
            seed=s,
            probability=p,
            label=1,
            action="escalate",
            known_mask=[1, 0],
            restart_identity="same",
            service_span_complete=True,
            started_monotonic_ns=1000,
            ended_monotonic_ns=2001000,
        )
        for s in (1, 2)
        for a, p in (("constrained_set", 0.8), ("source_erased_constrained_set", 0.5))
    ]
    reduced = cap.reduce_primitives(rows)
    assert reduced["source_groups"] == 1
    assert reduced["abstention_by_arm"]["constrained_set"] == 1
    assert (
        reduced["paired_comparisons"]["constrained_set__source_erased_constrained_set"][
            "independent_source_groups"
        ]
        == 1
    )
    assert reduced["mask_denominators"] == {"1,0": 4}
    assert reduced["complete_service_spans"] == 4
    for extra, reason in (
        ({"known_mask": [2]}, "invalid_mask"),
        ({"restart_identity": "other"}, "restart_identity"),
        ({"ended_monotonic_ns": 0}, "negative_service_span"),
    ):
        with pytest.raises(ValueError, match=reason):
            cap.reduce_primitives([*rows, {**rows[0], **extra}])
    assert cap.reduce_primitives([])["paired_comparisons"] == {}


def test_unknown_masks_and_exact_failed_operands(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7952-CUSTODY: unknown labels and blocked operands stay explicit."""
    value = cap.reduce_primitives([dict(family_id="f", known_mask=[-1, 0, 1], status="excluded")])
    assert value["mask_denominators"] == {"-1,0,1": 1}
    design, active = fixture(tmp_path)
    tasks = yaml.safe_load(active.read_bytes())["tasks"]
    producer(
        tmp_path,
        tasks[0],
        verdict_class="blocked",
        gate_check_summary=[
            dict(
                upstream_id="external",
                path="original.json",
                hash="sha256:original",
                field="class_1",
                op=">=",
                expected=8,
                observed=3,
            )
        ],
    )
    result = cap.build_candidate(tmp_path, design, active, "20260930")
    assert all(
        set(
            (
                "upstream_id",
                "artifact_path",
                "artifact_hash",
                "artifact_field",
                "op",
                "expected",
                "observed",
            )
        )
        <= set(g)
        for g in result["gate_check_summary"]
    )
    g = next(g for g in result["gate_check_summary"] if g["upstream_id"] == "external")
    assert g["artifact_field"] == "class_1" and g["artifact_hash"] == "sha256:original"


def test_frozen_static_and_consumer_closure(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7952-QUALIFICATION: exact strict checks bind every consumer."""
    frozen = validation.manifest(tmp_path)
    commands = {row["name"]: row for row in frozen["commands"]}
    assert commands["mypy"]["argv"][1:] == ["--strict", *validation.OWNED]
    for name in ("consumers_e2e018", "spec"):
        assert all(
            arg in frozen["dependency_hashes"]
            for arg in commands[name]["argv"]
            if arg.startswith("tests/") and arg.endswith(".py")
        )
