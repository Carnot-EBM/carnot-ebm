"""REQ-REPORT-9099-V776 / SCENARIO-REPORT-9099-V776: causal human mechanics."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit

from carnot.memory import human_online_9099 as learner
from carnot.reporting import online_methods_9099 as methods
from carnot.reporting.current_work_receipt import atomic_json


def test_rosters_and_frozen_bounds() -> None:
    """REQ-REPORT-9099-V776: original IDs, rather than group digests, fix chronology."""
    source = json.loads((methods.ROOT / methods.AUTHORITY).read_bytes())
    roster = methods.rosters(source["rows"])
    original = sorted(
        {r["qid"] for r in source["rows"] if r["role"] == "TRAIN"},
        key=lambda q: hashlib.sha256(("v776-online" + q).encode()).hexdigest(),
    )
    assert roster["initial_group_roster"] == original[:100]
    assert sum(roster["stream_roster"], []) == original[100:]
    assert len(roster["guard_roster"]) == 50 and len(roster["retention_roster"]) == 100
    assert set(roster["donor_map"].values()) <= set(original[:100])
    assert methods.protocol()["benefit_bounds"]["tail"] == 0.05 / 6
    assert len(methods.protocol()["benefit_bounds"]["bounds"]) == 6
    with pytest.raises(ValueError, match="original_roster"):
        methods.rosters(source["rows"][:-1])


def test_single_vector_gradient_and_validation() -> None:
    """REQ-REPORT-9099-V776: the intercept is not anchored; accumulator is elementwise."""
    theta, initial, acc, phi = (np.array(v, dtype=float) for v in ([2, 2], [0, 0], [0, 0], [1, 3]))
    result, total = learner.step(theta, acc, initial, phi, 1, False)
    gradient = (expit(theta @ phi) - 1) * phi + np.array([0, 0.02])
    gradient /= max(1, np.linalg.norm(gradient))
    np.testing.assert_array_equal(total, gradient**2)
    np.testing.assert_array_equal(result, theta - 0.05 * gradient / np.sqrt(total + 1e-8))
    bias, total = learner.step(theta, acc, initial, phi, 0, True)
    assert bias[1] == theta[1] and total[1] == 0
    for operand in [np.zeros((3, 2)), [float("nan"), 0]]:
        with pytest.raises(ValueError, match="update_operands"):
            learner.step(operand, acc, initial, phi, 0, False)
    for label in [2, True]:
        with pytest.raises(ValueError, match="update_operands"):
            learner.step(theta, acc, initial, phi, label, False)
    with pytest.raises(ValueError, match="update_operands"):
        learner.step(theta, [-1, 0], initial, phi, 0, False)


def test_fitting_refits_only_initialization() -> None:
    """REQ-REPORT-9099-V776: initialization is an actual converged proper-loss fit."""
    x = np.random.default_rng(776099).normal(size=(300, 6))
    labels = np.tile([0, 1], 150)
    fit = learner.fit_initial(x, labels)
    assert fit["fit_rows"] == 300
    assert len(fit["initial"]["adaptive_spline"]) == 217
    assert len(fit["initial"]["adaptive_linear"]) == 28
    assert all(r["gradient_inf"] <= 1e-6 for r in fit["fits"].values())
    changed = learner.transform(x * 100, fit["transform"])
    assert changed.shape == (300, 27) and np.max(np.abs(changed)) <= 1
    with pytest.raises(ValueError, match="initialization_operands"):
        learner.fit_initial(x[:299], labels[:299])


def test_durable_release_rollback_and_duplicate(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9099-V776: rejected feedback is consumed exactly once."""
    request, labels = methods.fixture("positive")
    label_path = tmp_path / "labels.json"
    atomic_json(label_path, labels)
    state = learner.drive(request, tmp_path / "store", label_path)
    assert state["pending"] == [] and state["acknowledgments"] == list(range(5))
    assert len(state["rows"]) == 75 and len(state["retention_rows"]) == 1500
    assert state["coefficients"]["adaptive_spline"] != request["initial"]["adaptive_spline"]
    before = deepcopy(state)
    assert not learner.feedback(state, request, 0, labels["labels"][0])
    assert state == before
    with pytest.raises(PermissionError, match="retention"):
        learner.release(state, request, label_path, 0, role="retention")
    with pytest.raises(ValueError, match="prediction_seal"):
        learner.release(learner.initial(request), request, label_path, 0)
    bad = deepcopy(request)
    bad["guard_labels"] = [0] * 150
    rejected = learner.drive(bad, tmp_path / "reject", label_path)
    assert all(
        not r["passed"] for r in rejected["rollback_checks"] if r["arm"] != "donor_label_spline"
    )
    assert rejected["coefficients"]["adaptive_spline"] == bad["initial"]["adaptive_spline"]
    assert not np.any(rejected["accumulators"]["adaptive_spline"])
    assert rejected["acknowledgments"] == list(range(5))


def test_future_and_sibling_truth_cannot_change_sealed_predictions(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9099-V776: current siblings and future labels stay behind the seal."""
    request, labels = methods.fixture("positive")
    p = tmp_path / "labels.json"
    atomic_json(p, labels)
    a = learner.drive(request, tmp_path / "a", p)
    labels["labels"][1:] = [[0, 0, 0]] * 4
    atomic_json(p, labels)
    b = learner.drive(request, tmp_path / "b", p)
    assert a["rows"][:45] == b["rows"][:45]
    assert [r for r in a["rows"] if r["arm"] == "donor_label_spline"] == [
        r for r in b["rows"] if r["arm"] == "donor_label_spline"
    ]
    assert [r for r in a["rows"] if r["arm"] == "frozen_spline"] == [
        r for r in b["rows"] if r["arm"] == "frozen_spline"
    ]


def test_real_hard_exits_and_fixture_reduction(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9099-V776: exit73 checkpoints and resumed states are actual child outputs."""
    result = methods.evaluate(tmp_path)
    assert len(result["rows"]) == 150
    assert result["exact_resume"]
    assert all(r["passed"] for r in result["causal_controls"])
    assert all(r["exit_code"] == 73 and r["passed"] for r in result["hard_exit_receipts"])
    assert (
        result["fixture_metrics"]["positive"]["adaptive_spline"]
        < result["fixture_metrics"]["positive"]["frozen_spline"]
    )
    assert methods.reduce(tmp_path) == result
    state = tmp_path / "positive" / "store" / "state.json"
    original_state = state.read_bytes()
    changed = json.loads(state.read_bytes())
    changed["state"]["acknowledgments"] = []
    from carnot.reporting.current_work_receipt import canonical_hash

    changed["sha256"] = canonical_hash(changed["state"])
    atomic_json(state, changed)
    with pytest.raises(ValueError, match="trajectory_drift"):
        methods.reduce(tmp_path)
    state.write_bytes(original_state)
    request_path = tmp_path / "positive/request.json"
    original_request = request_path.read_bytes()
    atomic_json(request_path, {})
    with pytest.raises(ValueError, match="fixture_operand_drift"):
        methods.reduce(tmp_path)
    request_path.write_bytes(original_request)
    log = Path(result["hard_exit_receipts"][0]["stdout_path"])
    log.write_bytes(log.read_bytes() + b"tampered\n")
    with pytest.raises(ValueError, match="hard_exit_log_drift"):
        methods.reduce(tmp_path)


def test_authentication_missing_field_zero_and_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-9099-V776: absence, missing fields and observed zero remain distinct."""
    result = methods.authenticate(methods.ROOT)
    assert not result["blocker"]
    assert methods.authenticate(tmp_path)["gate_check_summary"][0]["observed"] == "missing_file"
    path = tmp_path / methods.CONTRACT
    path.parent.mkdir(parents=True, exist_ok=True)
    for payload, observed in [({}, "missing_field"), ({"contract_ready_score": 0}, 0)]:
        atomic_json(path, payload)
        monkeypatch.setattr(
            methods,
            "PINS",
            {methods.CONTRACT: (methods.ref(path)["sha256"], "contract_ready_score", 1)},
        )
        failed = methods.authenticate(tmp_path)
        assert failed["gate_check_summary"][1]["observed"] == observed
        assert failed["blocker"] == "upstream_field"
    monkeypatch.setattr(
        methods, "PINS", {methods.CONTRACT: ("sha256:bad", "contract_ready_score", 1)}
    )
    assert methods.authenticate(tmp_path)["blocker"] == "primary_bytes"


def test_bad_operands_and_checkpoint_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-9099-V776: invalid operands and incompatible states cannot be resumed."""
    request, labels = methods.fixture("no_signal")
    p = tmp_path / "labels.json"
    atomic_json(p, labels)
    for key, value in [
        ("online_qids", ["x"] * 5),
        ("donor_labels", [[2] * 3] * 5),
        ("guard_labels", [0] * 149),
    ]:
        bad = deepcopy(request)
        bad[key] = value
        with pytest.raises(ValueError):
            learner.drive(bad, tmp_path / "bad", p)
    for name, value in [("initial", [0]), ("online_design", [[[0] * 217] * 3] * 5)]:
        bad = deepcopy(request)
        if name == "initial":
            bad[name]["adaptive_spline"] = value
        else:
            bad[name]["spline"] = value
        with pytest.raises(ValueError):
            learner.checked_request(bad)
    state = learner.drive(request, tmp_path / "store", p)
    with pytest.raises(ValueError, match="duplicate_feedback_drift"):
        learner.feedback(state, request, 0, [1 - y for y in labels["labels"][0]])
    with pytest.raises(ValueError, match="feedback_order"):
        learner.feedback(learner.initial(request), request, 0, [1, 1, 1])
    changed = deepcopy(request)
    changed["condition"] = "different"
    with pytest.raises(ValueError, match="resume_state"):
        learner.drive(changed, tmp_path / "store", p)
    with pytest.raises(ValueError, match="transform_operands"):
        learner.transform([[float("nan")]], {"mean": [0]})
    monkeypatch.setattr(
        learner, "minimize", lambda *a, **k: type("BadFit", (), {"x": np.full(217, np.nan)})()
    )
    with pytest.raises(ValueError, match="initialization_nonconvergence"):
        learner.fit_initial(np.zeros((300, 2)), np.tile([0, 1], 150))


def test_report_readiness_replay_and_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-9099-V776: scientific bounds stay unmeasured even when mechanics qualify."""
    from carnot.reporting import online_methods_runner_9099 as runner
    from carnot.reporting.current_work_receipt import canonical_hash

    assert runner.scope_manifest([])["owned_paths"] == methods.OWNED
    assert not runner.receipts_pass({}, {})
    reference = runner.measure(methods.ROOT, tmp_path / "work", "20261115", {})
    value = runner.build(reference, {})
    assert value["verdict_class"] == "disqualified" and value["online_methods_ready_score"] == 0
    monkeypatch.setattr(runner, "receipts_pass", lambda *a: True)
    value = runner.build(reference, {})
    assert value["verdict_class"] == "null" and value["online_methods_ready_score"] == 1
    assert value["donor_observed_label_pair_count"] == 150
    assert value["donor_changed_label_pair_count"] == sum(
        r["numerator"] for r in value["donor_label_change_rows"]
    )
    from scripts.adversarial_verify import check_tautology

    flags = []
    check_tautology(value, flags)
    assert not any(flag.severity == "critical" for flag in flags)
    assert value["learning_signal_score"] == value["independent_generalization_score"] == 0
    output = tmp_path / (methods.NAME + ".json")
    atomic_json(output, value)
    assert runner.replay(output)["passed"]
    value["initial_group_roster"] = list(reversed(value["initial_group_roster"]))
    atomic_json(output, value)
    with pytest.raises(ValueError, match="reduction_drift"):
        runner.replay(output)
    value = runner.build(reference, {})
    value["rows"][0]["value"] += 1
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(output, value)
    with pytest.raises(ValueError, match="reduction_drift"):
        runner.replay(output)
    missing = runner.measure(tmp_path / "absent", tmp_path / "blocked", "20261115", {})
    blocked = runner.build(missing, {})
    assert blocked["verdict_class"] == "blocked" and blocked["censored_count"] == 150
    private_work = tmp_path / "work_reference.json"
    atomic_json(private_work, missing)
    monkeypatch.setattr(runner, "validate", lambda *a: dict(passed=True))
    assert runner.main(["--work", str(private_work), "--output", str(output)]) == 0
    assert runner.main(["--cold-replay", str(output)]) == 0
    assert runner.main(["--date", "bad"]) == 1
    monkeypatch.setattr(runner, "receipts_pass", lambda *a: False)
    before = output.read_bytes()
    assert runner.main(["--work", str(private_work), "--output", str(output)]) == 1
    assert output.read_bytes() == before


def test_replay_rejects_rehashed_roster_and_primitive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-9099-V776: cascading reference hashes cannot replace original frozen rosters."""
    from carnot.reporting import online_methods_runner_9099 as runner

    reference = runner.measure(methods.ROOT, tmp_path / "work", "20261115", {})
    monkeypatch.setattr(runner, "receipts_pass", lambda *a: True)
    original = json.loads(Path(reference["path"]).read_bytes())
    output = tmp_path / (methods.NAME + ".json")
    for kind, message in [
        ("roster", "roster_drift"),
        ("source", "source_reduction_drift"),
        ("result", "fixture_reduction_drift"),
        ("current", "current_work_drift"),
        ("protocol", "protocol_drift"),
        ("donor", "donor_change_drift"),
    ]:
        work = deepcopy(original)
        if kind in ("roster", "protocol", "current"):
            key = {
                "roster": "roster_reference",
                "protocol": "protocol_reference",
                "current": "current_work_reference",
            }[kind]
            payload = json.loads(Path(work[key]["path"]).read_bytes())
            if kind == "roster":
                payload["initial_group_roster"].reverse()
            elif kind == "current":
                payload["invocation_counts"]["model_loads_completed"] = 1
            else:
                payload["version"] = "forged"
            changed = tmp_path / (kind + ".json")
            atomic_json(changed, payload)
            work[key] = methods.ref(changed)
        elif kind == "source":
            work["evidence"]["roster"]["initial_group_roster"].reverse()
        elif kind == "donor":
            work["donor_changes"]["numerator"] += 1
        else:
            work["result"]["rows"][0]["numerator"] += 1
        candidate = tmp_path / "forged_work.json"
        atomic_json(candidate, work)
        value = runner.build(methods.ref(candidate), {})
        atomic_json(output, value)
        with pytest.raises(ValueError, match=message):
            runner.replay(output)


def test_foreign_cli_and_scanners(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9099-V776: actual cold replay outside checkout rejects rehashed totals."""
    import os
    from carnot.reporting import online_methods_runner_9099 as runner
    from carnot.reporting.current_work_receipt import canonical_hash
    from carnot.reporting.v709_execution import child

    reference = runner.measure(tmp_path / "missing", tmp_path / "work", "20261115", {})
    value = runner.build(reference, {})
    path = tmp_path / (methods.NAME + ".json")
    atomic_json(path, value)
    assert runner.validate(path, tmp_path / "scanners")["passed"]
    prefix = [
        "env",
        "-u",
        "PYTHONPATH",
        "-C",
        str(tmp_path),
        str(methods.ROOT / ".venv/bin/python"),
    ]
    if os.environ.get("COVERAGE_RCFILE"):
        prefix += ["-m", "coverage", "run", "--rcfile=" + os.environ["COVERAGE_RCFILE"]]
    prefix += [str(methods.ROOT / methods.CLI)]
    assert child("valid", prefix + ["--cold-replay", str(path)], tmp_path / "foreign")["passed"]
    value["completed_count"] = 150
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(path, value)
    assert child("tamper", prefix + ["--cold-replay", str(path)], tmp_path / "foreign", expected=1)[
        "passed"
    ]
    assert child("date", prefix + ["--date", "bad"], tmp_path / "foreign", expected=1)["passed"]


def test_donors_are_known_initialization_siblings() -> None:
    """REQ-REPORT-9099-V776: a future qid or substituted donor label cannot enter updates."""
    request, _ = methods.fixture("positive")
    assert request["donor_labels"] == [request["initial_labels"][q] for q in request["donor_qids"]]
    bad = deepcopy(request)
    bad["donor_qids"][0] = bad["online_qids"][0]
    with pytest.raises(ValueError, match="donor_causality"):
        learner.checked_request(bad)
    bad = deepcopy(request)
    bad["initial_labels"]["private-init-0"] = [0, 0]
    bad["donor_labels"] = [[0, 0]] * 5
    with pytest.raises(ValueError, match="donor_labels"):
        learner.checked_request(bad)


def test_donor_change_fraction_uses_original_positions() -> None:
    """REQ-REPORT-9099-V776: label-change diagnostics retain actual denominators after map freeze."""
    source = json.loads((methods.ROOT / methods.AUTHORITY).read_bytes())
    roster = methods.rosters(source["rows"])
    changes = methods.donor_changes(source["rows"], roster)
    assert len(changes["rows"]) == 50
    assert changes["denominator"] == 150
    assert changes["numerator"] == sum(r["numerator"] for r in changes["rows"])
    assert changes["fraction"] == changes["numerator"] / 150
    request, _ = methods.fixture("positive")
    bad = deepcopy(request)
    bad["donor_labels"][0] = [1, 1, 1]
    with pytest.raises(ValueError, match="donor_causality"):
        learner.checked_request(bad)


def test_private_storage_and_sealed_label_refusal(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9099-V776: broad storage permissions and invalid released targets fail closed."""
    from carnot.reporting import online_methods_runner_9099 as runner

    raw = tmp_path / "public"
    raw.mkdir(mode=0o755)
    reference = runner.measure(tmp_path / "absent", raw, "20261115", {})
    value = runner.build(reference, {})
    assert value["gate_check_summary"][0]["check"] == "private_storage"
    output = tmp_path / (methods.NAME + ".json")
    atomic_json(output, value)
    assert runner.replay(output)["passed"]
    changed = deepcopy(value)
    changed["raw_shard_hashes"][-1]["sha256"] = "sha256:bad"
    atomic_json(output, changed)
    with pytest.raises(ValueError, match="operand_hash"):
        runner.replay(output)
    request, _ = methods.fixture("positive")
    state = learner.initial(request)
    state.update(query=2, phase="release", pending=[0, 1])
    state["rows"] = learner.predictions(state, request, "online", 0) + learner.predictions(
        state, request, "online", 1
    )
    labels = tmp_path / "labels.json"
    atomic_json(labels, dict(labels=[[True, 1, 1]] * 5))
    with pytest.raises(ValueError, match="feedback_labels"):
        learner.release(state, request, labels, 0)
