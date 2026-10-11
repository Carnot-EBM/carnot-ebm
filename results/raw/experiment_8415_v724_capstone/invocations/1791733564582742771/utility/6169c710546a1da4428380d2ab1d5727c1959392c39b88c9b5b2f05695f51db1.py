"""REQ-VERIFY-8350 / REQ-REPORT-8350: seal barriers and independent costs."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
from typing import Any

import pytest

from carnot.verify import static_benefit_audit_8350 as k
from carnot.reporting import static_benefit_audit_8350 as e
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


def panel() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Constructed oracle rows test accounting, never replace natural observations."""
    predictions, targets = [], []
    for i in range(128):
        y = i % 2
        targets.append(dict(slot=i + 1, unit_id=str(i), source_cluster_id=str(i), y=y))
        for arm in k.ARMS:
            p = (0.9 if y else 0.1) if arm in ("spline34", "sigmoid34") else 1 - y * 1.0
            predictions.append(
                dict(
                    slot=i + 1,
                    unit_id=str(i),
                    source_cluster_id=str(i),
                    arm=arm,
                    p=p,
                    action=k.fitted.action(p),
                )
            )
    return predictions, targets


def test_independent_costs_and_frozen_bootstrap() -> None:
    """SCENARIO-VERIFY-8350-REDUCTION: all128 clusters, exact frozen thresholds."""
    p, t = panel()
    result = k.reduce(p, t, "RBF34", dict(passed=True, geometry_qualified=False))
    assert result["h1_development_signal_score"] == 1
    assert result["bootstrap_summary"]["all_intended"]["mean_gain"] == 1
    assert result["bootstrap_summary"]["all_intended"]["lower_one_sided_975"] == 1
    assert result["bootstrap_summary"]["all_intended"]["requested_draws"] == 10000
    assert result["bootstrap_summary"]["all_intended"]["random_seed"] == 7178311
    assert result["qualified_count"] == 128
    assert len(result["paired_cost_rows"]) == 128
    assert len(result["arm_results"]) == 6
    assert result["typed_action_control"]["verdict_class"] == "circular_positive"
    for row in p:
        row["p"] = None
        row["action"] = "escalate"
    empty = k.reduce(p, t, "RBF34", dict(passed=False, geometry_qualified=False))
    assert empty["science_disposition"] == "null_insufficient_support"
    assert empty["arm_results"][0]["all_intended"]["cost_mean"] == 0.5
    assert empty["bootstrap_summary"]["complete_case"]["valid_draws"] == 0


def test_unknown_targets_keep_joint_bounds_and_support() -> None:
    """REQ-VERIFY-8350: a missing label does not prove a correct decision."""
    p, t = panel()
    t[0]["y"] = None
    result = k.reduce(p, t, "RBF34", dict(passed=True, geometry_qualified=False))
    assert result["qualified_count"] == 127
    assert result["missing_bounds"][0]["gain_lower"] == -1
    assert result["missing_bounds"][0]["gain_upper"] == 1
    assert result["paired_cost_rows"][0]["spline_cost"] is None
    assert result["bootstrap_summary"]["all_intended"]["mean_gain"] is None
    for row in p:
        if row["arm"] not in ("spline34", "sigmoid34"):
            row["p"] = next(
                r["p"] for r in p if r["slot"] == row["slot"] and r["arm"] == "spline34"
            )
            row["action"] = k.fitted.action(row["p"])
    null = k.reduce(p, t, "linear6", dict(passed=False, geometry_qualified=False))
    assert null["h1_development_signal_score"] == 0
    assert null["science_disposition"] == "null_insufficient_support"
    with pytest.raises(ValueError):
        k.reduce(p[:-1], t, "RBF34", {})
    with pytest.raises(ValueError):
        k.reduce(p + [p[0]], t, "RBF34", {})
    with pytest.raises(ValueError):
        k.reduce(p, t[:-1], "RBF34", {})
    t[0]["source_cluster_id"] = "foreign"
    with pytest.raises(ValueError, match="source_target_identity"):
        k.reduce(p, t, "RBF34", {})


def test_diagnostic_never_scores_labels() -> None:
    """SCENARIO-VERIFY-8350-BARRIER: first96 is predictor-only."""
    p, _ = panel()
    rows = k.diagnostic(p)
    assert len(rows) == 96
    assert sum(rows[0]["action_counts"].values()) == 6
    assert "cost" not in json.dumps(rows)
    assert len(k.diagnostic([])) == 96


def test_actual_natural_barrier_replay(tmp_path: Path) -> None:
    """REQ-REPORT-8350: real sealed operands and rehashed semantic rejection."""
    work = e.measure(e.ROOT, tmp_path / "raw")
    assert not work["failures"]
    assert work["retention_seal_check"]["passed"]
    assert work["full_h1_measured"]
    assert work["label_access_log"][0]["all_seals_authenticated_before_access"]
    value = e.build(work, tmp_path / "raw", [dict(name="test_control", passed=True)])
    path = tmp_path / "candidate.json"
    atomic_json(path, value)
    assert e.replay(path)
    forged = deepcopy(work)
    for row in forged["predictions"]:
        if row["slot"] == 1 and row["arm"] in ("spline34", "sigmoid34"):
            row["p"] = 0.79
            row["action"] = k.fitted.action(row["p"])
    forged["reduction"] = k.reduce(
        forged["predictions"], forged["targets"], forged["comparator"], forged["optimizer"]
    )
    primitive_path = Path(forged["raw_refs"][0]["path"])
    primitive = json.loads(primitive_path.read_bytes())
    primitive["predictions"] = forged["predictions"]
    atomic_json(primitive_path, primitive)
    forged["raw_refs"][0]["sha256"] = sha256_file(primitive_path)
    atomic_json(tmp_path / "raw" / "measurement.json", forged)
    value = e.build(forged, tmp_path / "raw", [dict(name="test_control", passed=True)])
    atomic_json(path, value)
    assert not e.replay(path)
    result = subprocess.run(
        [str(e.ROOT / ".venv/bin/python"), "-u", str(e.ROOT / e.CLI), "--cold-replay", str(path)],
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 1
    assert not e.replay(tmp_path / "absent.json")


def test_replay_custody_and_authority_failures(tmp_path: Path) -> None:
    """REQ-REPORT-8350: changed logs, code, gates and clocks cannot authenticate."""
    raw = tmp_path / "raw"
    work = e.measure(tmp_path / "missing", raw)
    candidate = tmp_path / "candidate.json"
    receipts = [dict(name="control", passed=True)]
    value = e.build(work, raw, receipts)
    atomic_json(candidate, value)
    value["measurement_reference"]["sha256"] = "changed"
    atomic_json(candidate, value)
    assert not e.replay(candidate)
    value = e.build(work, raw, receipts)
    bad = deepcopy(work)
    bad["code_refs"][0]["sha256"] = "changed"
    atomic_json(raw / "measurement.json", bad)
    value["measurement_reference"]["sha256"] = sha256_file(raw / "measurement.json")
    atomic_json(candidate, value)
    assert not e.replay(candidate)
    atomic_json(raw / "measurement.json", work)
    log = tmp_path / "log"
    log.write_text("changed")
    value = e.build(work, raw, [dict(passed=True, log_path=str(log), log_sha256="wrong")])
    atomic_json(candidate, value)
    assert not e.replay(candidate)
    for mutation in ("gate", "authority", "clock"):
        bad = deepcopy(work)
        if mutation == "gate":
            bad["gates"][0]["observed"] = "forged"
        elif mutation == "authority":
            bad["authority"]["activated"] = True
        else:
            bad["label_access_log"] = [
                dict(
                    seals_authenticated_wall_ns=10,
                    access_wall_ns=9,
                    all_seals_authenticated_before_access=True,
                    learner_feedback_count=0,
                )
            ]
            primitive = json.loads(Path(bad["raw_refs"][0]["path"]).read_bytes())
            primitive["label_access_log"] = bad["label_access_log"]
            atomic_json(Path(bad["raw_refs"][0]["path"]), primitive)
            bad["raw_refs"][0]["sha256"] = sha256_file(Path(bad["raw_refs"][0]["path"]))
        atomic_json(raw / "measurement.json", bad)
        atomic_json(candidate, e.build(bad, raw, receipts))
        assert not e.replay(candidate)
    assert e.build(work, raw, [dict(passed=False)])["verdict_class"] == "disqualified"


def test_absent_windows_leave_every_reserved_target_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-8350-BARRIER: natural first96 predictions remain diagnostics."""
    original = e.primary

    def missing_windows(
        root: Path, name: str, field: str, work: dict[str, Any], raw: Path
    ) -> dict[str, Any]:
        value = original(root, name, field, work, raw)
        if name == e.LEARNER:
            value = dict(value, retention_prediction_seals=[])
        return value

    monkeypatch.setattr(e, "primary", missing_windows)
    raw = tmp_path / "raw"
    work = e.measure(e.ROOT, raw)
    assert not work["full_h1_measured"]
    assert work["targets"] == [] and work["label_access_log"] == []
    assert len(work["stream_diagnostic"]) == 96
    assert sum(sum(r["action_counts"].values()) for r in work["stream_diagnostic"]) == 576
    assert all("8351" not in r["path"] for r in work["refs"])
    value = e.build(work, raw, [dict(passed=True)])
    assert value["verdict_class"] == "blocked"
    assert value["static_audit_ready_score"] == 0
    assert value["bootstrap_summary"] == {} and value["missing_bounds"] == []
    assert work["failures"][0]["artifact_field"] == "four_window_seal_count"
    assert work["failures"][0]["observed"] == 0


def test_malformed_retention_and_owned_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-VERIFY-8350: changed seal contents block; an owned reducer error disqualifies."""
    learner = json.loads((e.ROOT / e.LEARNER).read_bytes())
    pred = json.loads((e.ROOT / e.inputs.PRED).read_bytes())
    manifest = json.loads(Path(pred["prediction_manifest"]["path"]).read_bytes())
    bundle = json.loads(Path(manifest["input_reference"]["path"]).read_bytes())
    for kind in ("count", "target", "window", "head", "issue"):
        bad = deepcopy(learner)
        if kind in ("count", "target"):
            operand = json.loads(Path(bad["retention_prediction_seals"][0]["path"]).read_bytes())
            if kind == "count":
                operand["rows"].pop()
            else:
                operand["rows"][0]["targets_opened"] = True
            path = tmp_path / (kind + ".json")
            atomic_json(path, operand)
            bad["retention_prediction_seals"][0] = e.reference(path)
        elif kind == "window":
            bad["retention_prediction_seals"][1] = bad["retention_prediction_seals"][0]
        elif kind == "head":
            bad["heads_sha256"] = "foreign"
        else:
            bad["issued_rows"].pop()
        with pytest.raises(ValueError):
            e.retention(
                bad,
                bundle,
                pred["prediction_rows"],
                dict(gates=[], failures=[], refs=[]),
                tmp_path / kind,
            )

    def broken(*args: Any) -> dict[str, Any]:
        raise ValueError("deliberate_owned_reducer_error")

    monkeypatch.setattr(k, "reduce", broken)
    raw = tmp_path / "owned"
    work = e.measure(e.ROOT, raw)
    assert work["owned_failure"] and work["label_access_log"]
    assert e.build(work, raw, [dict(passed=True)])["verdict_class"] == "disqualified"


def test_real_cli_missing_inputs(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8350-CLI: actual child and failure publication."""
    cli = e.ROOT / e.CLI
    out = tmp_path / "results" / (e.NAME + ".json")
    command = [str(e.ROOT / ".venv/bin/python"), "-u", str(cli)]
    result = subprocess.run(
        command + ["--private-run", "--root", str(tmp_path), "--output", str(out)],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(out.read_bytes())
    assert value["verdict_class"] == "blocked"
    assert value["static_audit_ready_score"] == 0
    assert value["label_access_log"] == []
    assert (
        subprocess.run(
            command + ["--cold-replay", str(out)], capture_output=True, timeout=60
        ).returncode
        == 0
    )
    assert (
        subprocess.run(command + ["--date", "wrong"], capture_output=True, timeout=60).returncode
        == 2
    )
