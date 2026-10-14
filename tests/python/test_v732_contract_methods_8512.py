"""REQ-REPORT-8512, REQ-VERIFY-8512: complete authority and sealed methods."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest
import yaml

from carnot.reporting import v732_contract_methods as e
from carnot.reporting import v732_contract_runner as r
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import child


@pytest.fixture
def full_root(tmp_path):
    """An explicit oracle design fixture exercises admission without repairing live authority."""
    root = tmp_path / "authority"
    (root / e.DESIGN).parent.mkdir(parents=True)
    for name in [e.DESIGN, e.ACTIVE]:
        (root / name).write_bytes((e.ROOT / name).read_bytes())
    plan = yaml.safe_load((root / e.ACTIVE).read_bytes())
    tasks = plan["tasks"]
    text = (root / e.DESIGN).read_text().split("## Exact task contract")[0]
    text += "\n## Exact task contract\n\nCanonical complete-task SHA-256: `" + e.TASKS_PIN + "`\n"
    for i, t in enumerate(tasks):
        text += f"| {i + 1} | {t['id']} | {t['title']} | {t['phase']} | {t['deliverable']} |\n"
    text += "<!-- V732_TASK_CONTRACT_START -->\n```json\n" + json.dumps(plan)
    text += "\n```\n<!-- V732_TASK_CONTRACT_END -->\n"
    (root / e.DESIGN).write_text(text)
    return root


@pytest.mark.parametrize(
    "mode",
    [
        "valid",
        "count",
        "order",
        "id",
        "title",
        "phase",
        "deliverable",
        "model",
        "substrate",
        "gate",
        "prompt",
        "retirement",
        "end",
        "truncated",
        "missing",
        "rehash",
        "stage",
    ],
)
def test_authority(full_root, tmp_path, mode):
    """SCENARIO-REPORT-8512-AUTHORITY: every independent authority operand binds."""
    active, design = full_root / e.ACTIVE, full_root / e.DESIGN
    plan = yaml.safe_load(active.read_bytes())
    task = plan["tasks"][0]
    if mode in {"id", "title", "deliverable", "model", "prompt"}:
        task[mode] += " changed"
    if mode == "phase":
        task[mode] = 4
    if mode == "substrate":
        task["inference_substrate_class"] = "model_bounded_generation"
    if mode == "gate":
        plan["tasks"][1]["gated_on"][0]["value"] = 0
    if mode == "retirement":
        task["prior_failures"][0]["retire_if_same_verdict"] = False
    if mode == "count":
        plan["tasks"].pop()
    if mode == "order":
        plan["tasks"][:2] = plan["tasks"][1::-1]
    if mode == "end":
        design.write_text(design.read_text().replace("<!-- V732_TASK_CONTRACT_END -->", ""))
    if mode == "truncated":
        design.write_text(design.read_text().split("## Exact task contract")[0])
    if mode == "missing":
        design.unlink()
    if mode == "rehash":
        task["prompt"] += " forged"
        design.write_text(
            design.read_text()
            .split("<!-- V732_TASK_CONTRACT_START")[0]
            .replace(e.TASKS_PIN, e.tasks_digest(plan["tasks"]))
            + "<!-- V732_TASK_CONTRACT_START -->\n```json\n"
            + json.dumps(plan)
            + "\n```\n<!-- V732_TASK_CONTRACT_END -->\n"
        )
    if mode == "stage":
        staged = deepcopy(plan)
        staged["tasks"][0]["prompt"] += " changed"
        (full_root / e.STAGED).write_text(yaml.safe_dump(staged))
    active.write_text(yaml.safe_dump(plan))
    result = e.authority(full_root, tmp_path / "snapshots")
    assert result["activated"] == (mode == "valid")
    assert len(result["contract_rows"]) == 14
    assert all(row["effective_independent_groups"] == 0 for row in result["contract_rows"])
    if mode != "valid":
        assert result["gate_check_summary"]


def test_protocol():
    """SCENARIO-REPORT-8512-METHODS: scientific operands survive authority repair."""
    p = e.protocol()
    old = e.read(p["inherited_protocol_reference"])
    assert all(p[k] == old[k] for k in e.INHERITED)
    assert p["scalar_fit"]["coefficients"] == 13
    assert p["append_cost"]["bootstrap_seed"] == 7298480
    assert p["evidence_canary"]["calls"] == 16
    assert len(p["evidence_canary"]["views"]) == 4
    assert "successful" in p["validation_repairs"]["panel_registry_rejection"]
    assert p["canonical_tasks_sha256"] == e.TASKS_PIN
    assert p["observed_active_tasks_sha256"] == e.TASKS_PIN


@pytest.mark.parametrize(
    "mode", ["note", "inherit", "basis", "fixture", "view", "brier", "contrast"]
)
def test_protocol_rejection(monkeypatch, mode):
    """SCENARIO-REPORT-8512-METHODS: repaired envelopes cannot move frozen methods."""
    original_read, original_hash = e.read, e.sha256_file
    if mode == "note":
        monkeypatch.setattr(e, "sha256_file", lambda _: "changed")
    elif mode == "basis":
        monkeypatch.setattr(
            e,
            "sha256_file",
            lambda p: "changed" if p.name == "sparse_energy_7996.py" else original_hash(p),
        )
    else:

        def changed(ref):
            value = deepcopy(original_read(ref))
            if ref["path"] == str(e.ROOT / e.PROTOCOL):
                if mode == "inherit":
                    value["append_protocol"]["fsync_before_ack"] = False
                elif mode == "fixture":
                    value["validation_repairs"]["panel_registry_rejection"] = "omitted"
                elif mode == "view":
                    value["evidence_canary"]["calls"] = 12
                elif mode == "brier":
                    value["identity_brier"]["steps"] = 199
                elif mode == "contrast":
                    value["primary_contrasts"]["bootstrap_seed"] = 1
            return value

        monkeypatch.setattr(e, "read", changed)
    with pytest.raises(ValueError):
        e.protocol()


def fake_history(absent=None):
    """One varied family shows readiness gates do not silently depend on each other."""
    return dict(
        families=[
            dict(experiment_id=n, qualified=n != absent, gate_check_summary=[]) for n in e.FIELDS
        ],
        references=[],
        receipts=[],
    )


@pytest.mark.parametrize("absent", [None, 8430, 8433, 8493])
def test_family_boundaries(full_root, tmp_path, monkeypatch, absent):
    """SCENARIO-REPORT-8512-METHODS: optional and historical absences stay independent."""
    monkeypatch.setattr(e, "historical", lambda root, raw: fake_history(absent))
    raw = tmp_path / "raw"
    work = e.measure(full_root, raw)
    receipt = child("real", [sys.executable, "-c", "print('genuine')"], raw / "logs")
    value = e.build(work, [receipt], raw, tmp_path / (e.NAME + ".json"))
    assert value["methods_ready_score"] == value["current_contract_ready_score"] == 1
    assert value["source_inputs_ready_score"] == int(absent != 8430)
    assert value["learning_inputs_ready_score"] == int(absent not in {8430, 8433})
    assert value["service_inputs_ready_score"] == int(absent != 8493)
    assert value["runtime_repair_ready_score"] == 0
    assert not any(value["model_invocation_counts"].values())
    assert set(value) <= set(value["field_principles"])
    assert value["future_dependencies"][0]["experiment_id"] == 8513
    assert e.build(work, [], raw, tmp_path / (e.NAME + ".json"))["verdict_class"] == "disqualified"


def test_missing_and_deadline(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8512-CONTROLS: missing authority and expired work remain visible."""
    monkeypatch.setattr(e, "historical", lambda root, raw: fake_history())
    work = e.measure(tmp_path / "missing", tmp_path / "raw")
    value = e.build(work, [dict(passed=True)], tmp_path / "raw", tmp_path / (e.NAME + ".json"))
    assert value["verdict_class"] == "blocked"
    assert value["methods_ready_score"] == 0
    clock = iter([0, 3601_000_000_000])
    monkeypatch.setattr(e, "measurement_clock", lambda: next(clock))
    work = e.measure(tmp_path / "missing", tmp_path / "late")
    assert any(g["artifact_field"] == "measurement_cap_s" for g in work["failures"])


@pytest.mark.parametrize("mode", ["valid", "protocol", "authority", "family", "source", "receipt"])
def test_rehashed_primitives(full_root, tmp_path, monkeypatch, mode):
    """SCENARIO-VERIFY-8512-CONTROLS: independent operands reject self-consistent forgery."""
    monkeypatch.setattr(e, "historical", lambda root, raw: fake_history())
    raw = tmp_path / "raw"
    work = e.measure(full_root, raw)
    receipt = child("real", [sys.executable, "-c", "print('genuine')"], raw / "logs")
    if mode == "protocol":
        work["protocol"]["partition"]["reseeding"] = True
    if mode == "authority":
        work["contract"]["contract_rows"][0]["checks"]["full_object"] = False
    if mode == "family":
        work["history"]["families"][0]["qualified"] = False
    atomic_json(raw / "measurement.json", work)
    value = e.build(work, [receipt], raw, tmp_path / (e.NAME + ".json"))
    if mode == "source":
        value["code_config_hashes"][0]["sha256"] = "changed"
    if mode == "receipt":
        value["validation_receipts"][0]["duration_s"] += 1
    value["reproducibility_checksum"] = e.checksum(value)
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, value)
    assert e.replay(candidate) == (mode == "valid")


def test_real_private_cli(tmp_path):
    """SCENARIO-VERIFY-8512-CONTROLS: real publication qualifies current and historical inputs."""
    output = tmp_path / (e.NAME + ".json")
    base = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    cases = [
        (["--date", "20261014", "--private-control", "--output", str(output)], 0),
        (["--cold-replay", str(output)], 0),
        (["--cold-replay", str(tmp_path / "missing")], 1),
        (["--deliberate-error"], 7),
        (["--date", "20261012"], 2),
        (["--private-control"], 2),
        (["--history-request", str(tmp_path / "missing")], 2),
        (
            [
                "--private-control",
                "--root",
                str(tmp_path / "missing"),
                "--output",
                str(tmp_path / "absent" / (e.NAME + ".json")),
            ],
            0,
        ),
    ]
    for i, (args, expected) in enumerate(cases):
        receipt = child(f"cli-{i}", base + args, tmp_path / "logs", expected=expected, deadline=600)
        assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "circular_positive"
    assert value["methods_ready_score"] == value["current_contract_ready_score"] == 1
    assert value["source_inputs_ready_score"] == value["learning_inputs_ready_score"] == 1
    assert value["service_inputs_ready_score"] == 1
    assert value["required_checks_passed"]
    value["source_inputs_ready_score"] = 0
    value["reproducibility_checksum"] = e.checksum(value)
    atomic_json(output, value)
    assert not e.replay(output)


def test_manifest(tmp_path):
    """SCENARIO-VERIFY-8512-CONTROLS: historical shards preserve every collected assertion."""
    plan = r.manifest(tmp_path / "plan")
    for collection in (
        p for p in plan if p["name"].startswith("collect_") and e.TEST not in p["argv"]
    ):
        suite = collection["argv"][-1]
        actual = [
            node
            for p in plan
            if p["name"].startswith("historical_")
            for node in p["collected_node_ids"]
            if node.startswith(suite + "::")
        ]
        assert actual == collection["collected_node_ids"]
        assert len(actual) == len(set(actual))


def test_preserved_v730():
    """SCENARIO-REPORT-8512-AUTHORITY: historical design stays byte-identical under its original authority."""
    p = e.protocol()
    ref = p["preserved_v730_design_reference"]
    assert e.sha256_file(Path(ref["path"])) == ref["sha256"]
    assert "<!-- V730_TASK_CONTRACT_END -->" in Path(ref["path"]).read_text()


def test_strongest_baseline_registry():
    """SCENARIO-REPORT-8512-METHODS: strongest controls cannot drift during later fitting."""
    p = e.protocol()
    fit, audit = p["identity_brier"], p["primary_contrasts"]
    assert (fit["answers"], fit["questions"]) == (900, 300)
    assert fit["question_counts"] == [150, 50, 100]
    assert {k: len(v) for k, v in p["partition"]["roles"].items()} == {
        "fit": 150,
        "tune": 50,
        "test": 100,
    }
    assert (fit["full_inputs"], fit["full_coefficients"], fit["scalar_coefficients"]) == (
        9,
        109,
        13,
    )
    assert fit["steps"] == 200 and fit["full_batch"]
    assert fit["learning_rate"] == 0.01 and fit["l2"] == 0.001 and fit["temperature"] == 1
    assert fit["seeds"] == [17, 29, 43, 71, 101]
    assert fit["permutation_seeds"] == {"fit": 725, "tune": 7252, "test": 7251}
    assert fit["alpha_grid"] == [0, 0.25, 0.5, 0.75, 1]
    assert fit["alpha_selection"] == "tune A1 Brier only" and fit["alpha_ties"] == "smaller"
    assert fit["alpha_zero"] == "return exact raw bytes"
    assert (
        fit["bias_regularization"]
        == "L2 applies to every coefficient including bias, as in the shipped basis"
    )
    assert fit["test_target_reads_before_prediction_seal"] is False
    assert len(audit["contrasts"]) == 3
    assert audit["bootstrap_draws"] == 10000 and audit["bootstrap_seed"] == 7318502
    assert audit["confidence_percent"] == 98.3333333333
    assert audit["positive_class_ceiling"] == "circular_positive"
    assert not p["strict_import_repair"]["accepted_old_speed_benefit"]
    assert p["publication_qualification"]["never_clear_inherited_flags"]
    assert (
        p["failed_v730_qualification"]["observed_primary_sha256"]
        != p["failed_v730_qualification"]["bound_primary_sha256"]
    )
    assert p["ledger_lag"]["context_last_milestone"] == "2026.10.729"
    assert p["ledger_lag"]["observed_last_milestone"] == "2026.10.730"


def test_live_missing_contract(tmp_path):
    """SCENARIO-REPORT-8512-AUTHORITY: the observed queue never repairs a truncated design."""
    actual = e.authority(e.ROOT, tmp_path / "current")
    assert actual["observed_count"] == 14 and actual["activated"]
    assert actual["canonical_tasks_sha256"] == e.TASKS_PIN
    assert all(row["matched"] for row in actual["contract_rows"])


def test_preserved_failed_sidecar(tmp_path):
    """SCENARIO-REPORT-8512-EXPOSURE: the old flag and stale seal remain falsifiable."""
    from carnot.reporting.primary_publication import read_bound_sidecar

    p = e.protocol()
    failed = p["failed_v730_qualification"]
    old = e.read(failed["primary_reference"])
    assert old["flagged_adversarial"] is True
    primary = tmp_path / "results" / "experiment_8489_original.json"
    sidecar = primary.parent / "raw" / primary.stem / "validators" / "old.json"
    sidecar.parent.mkdir(parents=True)
    primary.write_bytes(Path(failed["primary_reference"]["path"]).read_bytes())
    sidecar.write_bytes(Path(failed["original_bound_sidecar_reference"]["path"]).read_bytes())
    with pytest.raises(ValueError, match="stale_primary_hash"):
        read_bound_sidecar(primary, sidecar)
    assert p["retirement_ledger"]["V717"]["H1"] == -0.00390625
    assert p["exposure_ledger"]["ARC"]["resolved_outcomes"] == 0


@pytest.mark.parametrize("mode", ["history", "task_map", "diagnostics", "activation"])
def test_new_independent_seals(full_root, tmp_path, monkeypatch, mode):
    """SCENARIO-REPORT-8512-METHODS: rehashed new registrations retain independent seals."""
    original_read, original_hash = e.read, e.sha256_file
    if mode == "activation":
        ref = e.protocol()["v732_activation_reference"]
        monkeypatch.setattr(
            e, "sha256_file", lambda p: "changed" if str(p) == ref["path"] else original_hash(p)
        )
        result = e.authority(full_root, tmp_path / "snapshots")
        assert not result["activated"]
        return
    if mode == "history":
        ref = e.protocol()["preserved_v731_design_reference"]
        monkeypatch.setattr(
            e,
            "sha256_file",
            lambda p: "changed" if str(p) == ref["source_path"] else original_hash(p),
        )
    else:

        def changed(ref):
            value = deepcopy(original_read(ref))
            if ref["path"] == str(e.ROOT / e.PROTOCOL):
                if mode == "task_map":
                    value["task_map_v732"][0]["prompt"] += " forged"
                else:
                    value["optimization_diagnostics"]["test_target_access"] = True
            return value

        monkeypatch.setattr(e, "read", changed)
    with pytest.raises(ValueError):
        e.protocol()


def test_new_registrations_and_scheduling():
    """SCENARIO-REPORT-8512-EXPOSURE: unmeasured science and censoring remain explicit."""
    p = e.protocol()
    assert p["optimization_diagnostics"]["steps"] == [0, 25, 50, 100, 200]
    assert not p["optimization_diagnostics"]["test_target_access"]
    assert p["arc_mutation_receipts"]["episodes"] == 16
    assert p["arc_mutation_receipts"]["refinement_min_resolved_per_shared_cell"] == 5
    assert all(not row["changed"] for row in p["protocol_delta"]["numerical_sections"].values())
    slots = p["v731_scheduling_dispositions"]
    assert [s["experiment_id"] for s in slots if s["disposition"] == "pre_gate_receipt"] == [
        8500,
        8503,
        8506,
    ]
    assert [s["experiment_id"] for s in slots if s["disposition"] == "cascade_skip"] == [
        8501,
        8502,
        8504,
        8505,
        8507,
        8508,
    ]
    assert all(
        s["primary_reference"]["path"] for s in slots if s["disposition"] == "pre_gate_receipt"
    )


def test_runner_boundaries(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8512-CONTROLS: worker routes and restoration retain process boundaries."""
    monkeypatch.setattr(r.history, "worker", lambda a, b: 3)
    assert (
        r.main(
            ["--history-request", str(tmp_path / "in"), "--history-output", str(tmp_path / "out")]
        )
        == 3
    )
    monkeypatch.setattr(r.shipped.shipped, "run", lambda *a, **kw: 0)
    assert r.main(["--private-control", "--output", str(tmp_path / (e.NAME + ".json"))]) == 0
    monkeypatch.setattr(
        r.shipped.shipped, "run", lambda *a, **kw: (_ for _ in ()).throw(ValueError("intentional"))
    )
    with pytest.raises(ValueError, match="intentional"):
        r.main(["--output", str(tmp_path / (e.NAME + ".json"))])
    calls = []
    monkeypatch.setattr(r, "child", lambda *a, **kw: calls.append(kw) or {})
    r.bounded_child("plain", [], tmp_path)
    r.bounded_child("cold", ["--cold-replay"], tmp_path)
    assert calls == [{}, {"deadline": 600}]


@pytest.mark.parametrize("mode", ["valid", "pin", "authority", "readiness"])
def test_v730_family_consumer(tmp_path, monkeypatch, mode):
    """SCENARIO-REPORT-8512-METHODS: original append qualification keeps its own field and authority."""
    h = r.history
    p = e.protocol()
    authority = []
    for ref in p["v730_authority_references"]:
        saved = h.cap.freeze(Path(ref["path"]), tmp_path / "authority")
        saved["source_path"] = ref["source_path"]
        authority.append(saved)
    primary = h.cap.freeze(
        next((e.ROOT / "results").glob("experiment_8492_*.json")), tmp_path / "primary"
    )
    args = dict(number=8492, primary=primary, authority=authority)
    monkeypatch.setattr(h.cap, "outcome", lambda *a: dict(eligible=mode != "readiness"))
    if mode == "pin":
        primary["sha256"] = "changed"
    if mode == "authority":
        authority[0]["sha256"] = "changed"
    row = h.qualify(args, tmp_path / "scratch")
    assert row["qualified"] == (mode == "valid")
    assert row["field"] == "append_protocol_ready_score"
    if mode != "valid":
        assert row["gate_check_summary"][0]["observed"] == (1 if mode == "readiness" else None)


def test_historical_worker_and_failed_child(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8512-CONTROLS: missing and failed historical work never grants readiness."""
    h = r.history
    monkeypatch.setattr(h.old, "qualify", lambda *a: dict(qualified=True))
    request, output = tmp_path / "request.json", tmp_path / "output.json"
    atomic_json(request, dict(number=8430, primary={}))
    assert h.worker(request, output) == 0
    assert json.loads(output.read_bytes())["qualified"]
    monkeypatch.setattr(
        h.old, "historical", lambda *a: dict(families=[], receipts=[], references=[])
    )
    monkeypatch.setattr(h, "child", lambda *a, **kw: dict(passed=False, actual_exit=-9))
    missing_root = tmp_path / "missing"
    (missing_root / "results").mkdir(parents=True)
    result = h.historical(missing_root, tmp_path / "raw", e.CLI, e.protocol())
    assert len(result["families"]) == 2
    assert not any(row["qualified"] for row in result["families"])
    assert all(row["gate_check_summary"][0]["observed"] == -9 for row in result["families"])


def test_killed_historical_reader_disqualifies(full_root, tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8512-CONTROLS: killed replay is a validation failure, not absent science."""
    raw = tmp_path / "raw"
    receipt = child("real", [sys.executable, "-c", "print('genuine')"], raw / "logs")
    history = fake_history()
    history["receipts"] = [
        dict(receipt, passed=False, normal_exit=False, actual_exit=-9, exit_code=-9)
    ]
    monkeypatch.setattr(e, "historical", lambda *a: history)
    work = e.measure(full_root, raw)
    value = e.build(work, [receipt], raw, tmp_path / (e.NAME + ".json"))
    assert value["verdict_class"] == "disqualified"
    assert not value["required_checks_passed"]
    assert value["methods_ready_score"] == value["service_inputs_ready_score"] == 0
