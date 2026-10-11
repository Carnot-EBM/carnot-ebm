"""REQ-VERIFY-8351 / REQ-REPORT-8351: original evidence and causal reductions."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
from typing import Any

import pytest

from carnot.reporting import learning_retention_audit_8351 as e
from carnot.verify import learning_retention_audit_8351 as k
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


def panel() -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Oracle controls exercise thresholds without replacing natural evidence."""
    state: dict[str, Any] = dict(
        issued=[], retention=[], updates=[dict(arm="online_sparse", reason="applied")]
    )
    targets = [
        dict(slot=s, unit_id=str(s), source_cluster_id=str(s), y=s % 2) for s in range(1, 129)
    ]
    for t in targets:
        for arm in k.ARMS:
            p = (0.9 if t["y"] else 0.1) if arm != "frozen_spline" else float(1 - t["y"])
            row = dict(
                slot=t["slot"],
                unit_id=t["unit_id"],
                source_cluster_id=t["source_cluster_id"],
                arm=arm,
                p=p,
                action=k.action(p),
            )
            if t["slot"] <= 96:
                state["issued"].append(row)
            else:
                state["retention"].extend(dict(row, window=w) for w in k.WINDOWS)
    support = dict(
        passed=True,
        reachable_count=88,
        update_budget_control=dict(passed=True),
        scalar_update=dict(recomputed=True),
    )
    return state, targets, support


def test_fixed_later_slots_and_all_windows() -> None:
    """SCENARIO-VERIFY-8351-REDUCE: fixed 88 slots and all four windows."""
    state, targets, support = panel()
    result = k.reduce(state, targets, support)
    assert len(result["paired_cost_rows"]) == 88
    assert len(result["retention_rows"]) == 640
    assert result["block_bootstrap_summary"]["requested_draws"] == 10000
    assert result["block_bootstrap_summary"]["block_length"] == 8
    assert result["block_bootstrap_summary"]["random_seed"] == 7178312
    assert result["block_bootstrap_summary"]["mean_gain"] == 1
    assert result["block_bootstrap_summary"]["lower_one_sided_975"] == 1
    assert result["h2_development_signal_score"] == 1
    assert len(result["arm_results"]) == 5
    assert [r["window"] for r in result["retention_window_bounds"]] == k.WINDOWS
    assert result["typed_action_control"]["passed"]
    assert result["typed_action_control"]["verdict_class"] == "circular_positive"


def test_null_support_and_unreachable_budget() -> None:
    """REQ-VERIFY-8351: an update-budget limit narrows a null retirement."""
    state, targets, support = panel()
    for row in state["issued"] + state["retention"]:
        row["p"], row["action"] = 0.5, "escalate"
    assert (
        k.reduce(state, targets, support)["science_disposition"] == "null_delayed_decision_benefit"
    )
    support["reachable_count"] = 0
    assert (
        k.reduce(state, targets, support)["science_disposition"] == "null_update_budget_no_headroom"
    )
    for row in state["issued"]:
        row["p"], row["action"] = None, "escalate"
    result = k.reduce(state, targets, support)
    assert result["science_disposition"] == "null_insufficient_support"
    assert result["arm_results"][0]["cost_mean"] == 0.5
    assert result["qualified_count"] == 0
    assert result["block_bootstrap_summary"]["mean_gain"] == 0


def test_missing_labels_identity_and_parity() -> None:
    """REQ-VERIFY-8351: unknown labels remain bounded and never replaced."""
    state, targets, support = panel()
    targets[8]["y"] = None
    targets[96]["y"] = None
    result = k.reduce(state, targets, support)
    assert result["qualified_count"] == 87
    assert len(result["missing_bounds"]) == 1
    assert result["block_bootstrap_summary"]["mean_gain"] is None
    assert result["retention_window_bounds"][0]["missing_slots"] == [97]
    assert result["retention_window_bounds"][0]["qualified_count"] == 31
    state["issued"][0]["action"] = "invalid"
    with pytest.raises(ValueError):
        k.reduce(state, targets, support)
    state, targets, support = panel()
    targets[8]["source_cluster_id"] = targets[9]["source_cluster_id"]
    with pytest.raises(ValueError):
        k.reduce(state, targets, support)
    state, targets, support = panel()
    state["retention"].pop()
    with pytest.raises(ValueError):
        k.reduce(state, targets, support)


@pytest.fixture(scope="module")
def natural(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict[str, Any], Path]:
    """SCENARIO-VERIFY-8351-CAUSAL: use sealed repository observations."""
    raw = tmp_path_factory.mktemp("8351-natural") / "raw"
    return e.measure(e.ROOT, raw), raw


def test_natural_independent_reconstruction(natural: tuple[dict[str, Any], Path]) -> None:
    """SCENARIO-VERIFY-8351-CAUSAL: causal primitives precede evaluator access."""
    work, _ = natural
    assert not work["failures"]
    assert work["checks"]["passed"]
    assert work["checks"]["scalar_update"]["recomputed"]
    assert work["checks"]["scalar_update"]["deliberate_error_rejected"]
    assert work["checks"]["future_label_invariance"]
    assert work["checks"]["duplicate_stale_rejection"]
    assert work["checks"]["source_separation"]
    assert work["checks"]["source_role_separation"]["passed"]
    assert work["checks"]["feedback_rejection_controls"] == ["duplicate", "stale"]
    assert work["checks"]["before_update_substitution"]["passed"]
    assert work["checks"]["before_update_substitution"]["later_distinct_source_count"] > 0
    assert work["checks"]["restart_parity"]
    assert work["label_access_log"][0]["all_seals_authenticated_before_access"]
    assert work["label_access_log"][0]["learner_feedback_count"] == 0
    assert work["prior_evaluator_exposure"]["observed"]
    result = k.reduce(work["state"], work["targets"], work["checks"])
    assert len(result["paired_cost_rows"]) == 88
    assert result["retention_window_bounds"][-1]["qualified_count"] == 23
    assert result["retention_window_bounds"][-1]["class_support"] == {"0": 9, "1": 14}
    assert len(result["retention_window_bounds"][-1]["missing_slots"]) == 9
    bad = deepcopy(work["state"])
    bad["issued"][0]["p"] += 0.1
    with pytest.raises(ValueError):
        k.reconstruct(work["bundle"], bad, work["events"], work["checkpoints"], work["future"])


def test_build_replay_and_rehashed_tamper(
    natural: tuple[dict[str, Any], Path], tmp_path: Path
) -> None:
    """SCENARIO-REPORT-8351-CLI: fresh reduction rejects invented readiness."""
    work, raw = natural
    receipts = [dict(name="control", passed=True)]
    value = e.build(work, raw, receipts)
    assert value["learning_audit_ready_score"] == 1
    assert value["independent_generalization_score"] == 0
    assert value["MODEL_SPECS"] == []
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, value)
    assert e.replay(candidate)
    for key in ("learning_audit_ready_score", "h2_development_signal_score"):
        bad = dict(value, **{key: 99})
        bad.pop("reproducibility_checksum")
        bad["reproducibility_checksum"] = canonical_hash(bad)
        atomic_json(candidate, bad)
        assert not e.replay(candidate)
    assert e.build(work, raw, [])["verdict_class"] == "disqualified"
    badwork = dict(work, owned_failure=True)
    assert e.build(badwork, raw, receipts)["learning_audit_ready_score"] == 0
    for p in k.ARMS:
        assert p in str(value["arm_results"])
    assert not e.replay(tmp_path / "missing.json")
    assert not e.replay(e.ROOT / "pyproject.toml")


def test_custody_and_full_primitive_rehash_rejected(
    natural: tuple[dict[str, Any], Path], tmp_path: Path
) -> None:
    """REQ-REPORT-8351: rehashed measurements cannot supersede pinned inputs."""
    original, _ = natural
    raw = tmp_path / "raw"
    raw.mkdir()
    work = deepcopy(original)
    atomic_json(raw / "measurement.json", work)
    candidate = tmp_path / "candidate.json"
    receipts = [dict(passed=True)]
    for field in ("refs", "code_refs", "raw_refs"):
        bad = deepcopy(work)
        bad[field][0]["sha256"] = "changed"
        atomic_json(raw / "measurement.json", bad)
        atomic_json(candidate, e.build(bad, raw, receipts))
        assert not e.replay(candidate)
    bad = deepcopy(work)
    bad["state"]["issued"][0]["p"] += 0.1
    atomic_json(raw / "measurement.json", bad)
    atomic_json(candidate, e.build(bad, raw, receipts))
    assert not e.replay(candidate)
    atomic_json(raw / "measurement.json", work)
    value = e.build(work, raw, receipts)
    value["measurement_reference"]["sha256"] = "bad"
    atomic_json(candidate, value)
    assert not e.replay(candidate)
    log = tmp_path / "log"
    log.write_text("actual")
    atomic_json(
        candidate,
        e.build(work, raw, [dict(passed=True, stdout_path=str(log), stdout_sha256="bad")]),
    )
    assert not e.replay(candidate)
    bad = deepcopy(work)
    bad["label_access_log"][0]["learner_feedback_count"] = 1
    atomic_json(raw / "measurement.json", bad)
    atomic_json(candidate, e.build(bad, raw, receipts))
    assert not e.replay(candidate)


def test_external_absence_and_actual_private_cli(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8351-CLI: no missing input becomes synthetic science."""
    raw = tmp_path / "raw"
    work = e.measure(tmp_path / "absent", raw)
    assert work["failures"]
    assert work["label_access_log"] == []
    assert work["failures"][0]["observed"] is None
    value = e.build(work, raw, [dict(passed=True)])
    assert value["verdict_class"] == "blocked"
    assert value["censored_count"] == 88
    candidate = tmp_path / "blocked.json"
    atomic_json(candidate, value)
    assert e.replay(candidate)
    output = tmp_path / "experiment_8351_private.json"
    assert e.main(["--private", "--root", str(tmp_path / "absent"), "--output", str(output)]) == 0
    assert json.loads(output.read_bytes())["verdict_class"] == "blocked"
    command = [str(e.ROOT / ".venv/bin/python"), "-u", str(e.ROOT / e.CLI)]
    result = subprocess.run(
        command + ["--cold-replay", str(output)], capture_output=True, timeout=120
    )
    assert result.returncode == 0
    result = subprocess.run(
        command + ["--private", "--output", str(e.ROOT / "results/experiment_8351_private.json")],
        capture_output=True,
        timeout=20,
    )
    assert result.returncode == 2
    assert e.main(["--cold-replay", str(tmp_path / "missing.json")]) == 1


def test_actual_private_success_cli(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8351-CLI: real child replay and typed finding controls."""
    output = tmp_path / "experiment_8351_natural.json"
    assert e.main(["--private", "--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["required_checks_passed"]
    assert value["learning_audit_ready_score"] == 1
    assert e.replay(output)
    bad = dict(value, retention_window_bounds=[])
    bad.pop("reproducibility_checksum")
    bad["reproducibility_checksum"] = canonical_hash(bad)
    atomic_json(tmp_path / "rehashed.json", bad)
    assert not e.replay(tmp_path / "rehashed.json")


def test_budget_and_scalar_failures(natural: tuple[dict[str, Any], Path]) -> None:
    """REQ-VERIFY-8351: scalar updates and source identity have negative controls."""
    work, _ = natural
    for field in ("updates", "retention"):
        state = deepcopy(work["state"])
        row = next(r for r in state[field] if r["arm"] == "online_sparse")
        if field == "updates":
            row["coefficients"][2] += 0.1
        else:
            row["p"] += 0.1
        with pytest.raises(ValueError):
            k.reconstruct(
                work["bundle"], state, work["events"], work["checkpoints"], work["future"]
            )
    bad = deepcopy(work["bundle"])
    bad["slots"][0]["source_cluster_id"] = bad["slots"][1]["source_cluster_id"]
    with pytest.raises(ValueError):
        k.reconstruct(bad, work["state"], work["events"], work["checkpoints"], work["future"])


@pytest.mark.parametrize(
    "mutation", ["roster", "release", "shuffle", "checkpoint", "retention", "journal", "issue"]
)
def test_primitive_negative_controls(natural: tuple[dict[str, Any], Path], mutation: str) -> None:
    """SCENARIO-VERIFY-8351-CAUSAL: each causal boundary rejects changed evidence."""
    work, _ = natural
    state, events, checkpoints = (
        deepcopy(work["state"]),
        deepcopy(work["events"]),
        deepcopy(work["checkpoints"]),
    )
    if mutation == "roster":
        state["issued"].pop(0)
    elif mutation == "release":
        state["releases"][0]["label_slot"] += 1
    elif mutation == "shuffle":
        next(r for r in state["updates"] if r["arm"] == "shuffled_due_feedback")["used_y"] = 2
    elif mutation == "checkpoint":
        checkpoints["32"]["updates"].pop()
    elif mutation == "issue":
        state["issued"][0]["action"] = "invalid"
    elif mutation == "retention":
        state["retention"][0]["action"] = "invalid"
    else:
        events[-1]["state_hash"] = "changed"
    with pytest.raises(ValueError):
        k.reconstruct(work["bundle"], state, events, checkpoints, work["future"])


def test_empty_retention_and_false_coverage(
    tmp_path: Path, natural: tuple[dict[str, Any], Path]
) -> None:
    """REQ-REPORT-8351: empty support and uncovered code cannot claim readiness."""
    state, targets, support = panel()
    for row in state["retention"]:
        row["p"], row["action"] = None, "escalate"
    result = k.reduce(state, targets, support)
    assert result["retention_window_bounds"][0]["qualified_count"] == 0
    assert result["h2_development_signal_score"] == 0
    work, raw = natural
    coverage = tmp_path / "coverage.json"
    atomic_json(coverage, dict(files={p: dict(summary=dict(missing_lines=1)) for p in e.OWNED}))
    bad = dict(work, owned_coverage_reference=e.reference(coverage))
    assert e.build(bad, raw, [dict(passed=True)])["verdict_class"] == "disqualified"
    assert e.manifest(tmp_path, tmp_path / "candidate.json")["no_full_repository_suite"]


def test_idempotent_snapshot_tamper(tmp_path: Path) -> None:
    """REQ-REPORT-8351: repeated byte custody does not permit changed snapshot bytes."""
    source = tmp_path / "source.json"
    atomic_json(source, dict(value=1))
    work: dict[str, Any] = dict(gates=[], failures=[], refs=[])
    raw = tmp_path / "raw"
    ref = e.reference(source)
    e.bind(work, ref, raw)
    saved = raw / "inputs" / (ref["sha256"][7:] + "-" + source.name)
    saved.chmod(0o600)
    saved.write_text("{}")
    with pytest.raises(ValueError):
        e.bind(work, ref, raw)


def test_authority_and_configuration_rehash(
    natural: tuple[dict[str, Any], Path], tmp_path: Path
) -> None:
    """REQ-REPORT-8351: authenticated authority and fixed configuration cannot drift."""
    work, raw = natural
    candidate = tmp_path / "candidate.json"
    receipts = [dict(passed=True)]
    try:
        bad = deepcopy(work)
        bad["authority"]["canonical_tasks_sha256"] = "invented"
        atomic_json(raw / "measurement.json", bad)
        atomic_json(candidate, e.build(bad, raw, receipts))
        assert not e.replay(candidate)
        bad = deepcopy(work)
        config = tmp_path / "changed_config.json"
        atomic_json(config, dict(k.CONFIG, seed=1))
        bad["raw_refs"][1] = e.reference(config)
        atomic_json(raw / "measurement.json", bad)
        atomic_json(candidate, e.build(bad, raw, receipts))
        assert not e.replay(candidate)
    finally:
        atomic_json(raw / "measurement.json", work)
