"""REQ-REPORT-9153-V780 / SCENARIO-REPORT-9153-V780: costs and ceilings differ."""

from copy import deepcopy
import json
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any

import pytest

from carnot.reporting import validation_schedule_9139 as old
from carnot.reporting import validation_schedule_9153 as s
from carnot.reporting.current_work_receipt import atomic_json


def legacy() -> dict[str, Any]:
    h = dict(
        nodes=["a.py::test_a", "b.py::test_b"],
        events=[
            dict(nodeid="a.py::test_a", when=p, outcome="passed", duration=0.001)
            for p in ["setup", "call", "teardown"]
        ],
        required_command_union=[dict(name="check", argv=["true"], deadline=4800)],
        original_assertion_references=[],
    )
    return old.schedule(h)


def test_fast_costs_do_not_sum_watchdogs() -> None:
    plan = s.schedule(legacy(), [])
    assert plan["historical_ceiling_sum_s"] == legacy()["budget_equation"]["projected_total_s"]
    assert plan["budget_equation"]["estimated_remaining_s"] is None
    assert plan["budget_equation"]["feasible"] is None
    assert plan["cost_model_rows"][-1]["kill_timeout_s"] == 4800
    assert plan["measured_cost_rows"][0]["measured_elapsed_s"] == 0.003
    assert len(plan["unknown_cost_rows"]) == 3
    assert plan["budget_equation"]["verification_reservations"] == 1
    measured = {r["unit_id"]: 0.01 for r in plan["cost_model_rows"]}
    fast = s.schedule(legacy(), [], measured)
    assert fast["budget_equation"]["estimated_remaining_s"] == 0.03
    assert fast["budget_equation"]["feasible"] is True
    measured["command:check"] = 3400
    assert s.schedule(legacy(), [], measured)["budget_equation"]["feasible"] is False
    with pytest.raises(ValueError, match="duplicate_node"):
        s.schedule(legacy(), ["a.py::test_a"])


@pytest.fixture(scope="module")
def actual(tmp_path_factory: Any) -> dict[str, Any]:
    raw = tmp_path_factory.mktemp("v780-children")
    module = "tests/python/test_forward_evidence_9111.py"
    nodes = [module + "::test_no_load_and_historical_scope", module + "::test_forward_event_ledger"]
    plan = old.schedule(
        dict(
            nodes=nodes,
            events=[],
            required_command_union=[],
            original_assertion_references=[old.ref(old.ROOT / module)],
        )
    )
    plan = s.schedule(plan, [])
    return s.execute(plan, raw, "v780-test", [old.ref(old.ROOT / module)], time.monotonic() + 120)


def test_actual_child_and_normal_consumption(actual: dict[str, Any], tmp_path: Path) -> None:
    assert old.consume(actual, "v780-test")["completed_count"] == 2
    bundle = tmp_path / "bundle.json"
    atomic_json(bundle, actual)
    argv = s.r.foreign_argv(
        old.prior.old.CLI, "--schedule-v779", str(bundle), "--schedule-invocation", "v780-test"
    )
    receipt = old.prior.run_check("normal", argv, tmp_path, 120)
    assert receipt["passed"]
    assert json.loads((tmp_path / "consumed.json").read_bytes())["rerun_commands"] == []


@pytest.mark.parametrize("change", ["foreign", "missing", "duplicate", "code", "events"])
def test_original_rejections(actual: dict[str, Any], change: str) -> None:
    bundle = deepcopy(actual)
    if change == "foreign":
        bundle["invocation_id"] = "foreign"
    elif change == "missing":
        bundle["partitions"] = []
    elif change == "duplicate":
        bundle["partitions"] *= 2
    elif change == "code":
        bundle["code_config_hashes"][0]["sha256"] = "bad"
    else:
        bundle["partitions"][0]["events_reference"]["sha256"] = "bad"
    with pytest.raises(ValueError):
        old.consume(bundle, "v780-test")


def test_deadline_and_measured_timeout(tmp_path: Path) -> None:
    source = tmp_path / "test_wait.py"
    source.write_text("import time\ndef test_wait():\n    time.sleep(2)\n")
    plan = old.schedule(
        dict(
            nodes=[str(source) + "::test_wait"],
            events=[],
            required_command_union=[],
            original_assertion_references=[],
        )
    )
    plan = s.schedule(plan, [])
    expired = s.execute(plan, tmp_path / "expired", "expired", [], time.monotonic() - 1)
    assert (
        not expired["partitions"] and expired["remaining_nodes"] == plan["original_node_manifest"]
    )
    plan["execution_graph"][0]["budget_s"] = 0.1
    timed = s.execute(plan, tmp_path / "timeout", "timeout", [], time.monotonic() + 120)
    receipt = timed["partitions"][0]["receipt"]
    assert receipt["timed_out"] and 0.1 <= receipt["duration_s"] < 2
    assert timed["remaining_nodes"] == plan["original_node_manifest"]
    with pytest.raises(ValueError, match="partition_exit"):
        old.consume(timed, "timeout")
    plan["verification_graph"] = [dict(name="tiny", argv=["true"], deadline=4800)]
    checked = s.execute(
        dict(plan, execution_graph=[], original_node_manifest=[]),
        tmp_path / "checks",
        "checks",
        [],
        time.monotonic() + 30,
    )
    assert checked["checks"][0]["receipt"]["passed"]
    with pytest.raises(ValueError, match="duplicate_command"):
        s.execute(
            dict(plan, verification_graph=plan["verification_graph"] * 2),
            tmp_path / "duplicate",
            "d",
            [],
            time.monotonic() + 30,
        )


@pytest.mark.parametrize("operand", ["missing_file", "missing_field", 0, True])
def test_gate_operands(tmp_path: Path, monkeypatch: Any, operand: Any) -> None:
    name = next(iter(s.INPUTS))
    if operand != "missing_file":
        path = tmp_path / name
        atomic_json(path, {} if operand == "missing_field" else dict(contract_ready_score=operand))
        monkeypatch.setitem(s.INPUTS, name, (old.ref(path)["sha256"], "contract_ready_score", 1))
    work = s.authenticate(tmp_path, tmp_path / "raw")
    assert work["checks"][-1]["observed"] == operand
    assert not work["checks"][-1]["passed"]


def test_authentication_scope(tmp_path: Path, monkeypatch: Any) -> None:
    work = s.authenticate(s.ROOT, tmp_path / "auth")
    assert all(c["passed"] for c in work["checks"])
    assert len(work["schedule"]["original_node_manifest"]) == 674
    assert len(work["schedule"]["historical_original_node_manifest"]) == 621
    assert work["schedule"]["historical_ceiling_sum_s"] == 27880.142389201996
    prior = old.authenticate

    def changed(root: Path, raw: Path) -> dict[str, Any]:
        result = prior(root, raw)
        result["checks"].append(dict(passed=False))
        return result

    monkeypatch.setattr(old, "authenticate", changed)
    assert "schedule" not in s.authenticate(s.ROOT, tmp_path / "failed")

    def drift(root: Path, raw: Path) -> dict[str, Any]:
        result = prior(root, raw)
        result["schedule"]["units"] = "drift"
        return result

    monkeypatch.setattr(old, "authenticate", drift)
    with pytest.raises(ValueError, match="historical_scope_drift"):
        s.authenticate(s.ROOT, tmp_path / "drift")


def test_historical_projection_tamper() -> None:
    plan = legacy()
    plan["budget_equation"]["projected_total_s"] += 1
    with pytest.raises(ValueError, match="historical_ceiling_drift"):
        s.schedule(plan, [])


def test_blocked_replay_and_private_cli(tmp_path: Path) -> None:
    output = tmp_path / (s.NAME + ".json")
    args = ["--input-root", str(tmp_path / "absent"), "--output", str(output)]
    assert old.prior.run_check("refusal", s.r.foreign_argv(s.CLI, *args), tmp_path / "logs")[
        "passed"
    ]
    value = json.loads(output.read_bytes())
    assert value["honest_verdict"].startswith("complete_blocked_")
    assert value["current_work_receipts"][0]["current_run_id"] == "exp9153"
    assert value["current_llm_invocations"] == value["generated_tokens"] == 0
    assert old.prior.run_check(
        "cold", s.r.foreign_argv(s.CLI, "--replay", str(output)), tmp_path / "logs"
    )["passed"]
    value["schedule_ready_score"] = 1
    atomic_json(output, value)
    assert old.prior.run_check(
        "tamper", s.r.foreign_argv(s.CLI, "--replay", str(output)), tmp_path / "logs", expected=1
    )["passed"]
    value["reproducibility_checksum"] = "bad"
    atomic_json(output, value)
    with pytest.raises(ValueError, match="replay_hash"):
        s.replay(output)
    assert s.main(["--date", "bad"]) == 1
    assert s.main(["--replay", str(tmp_path / "absent")]) == 1


def test_plan_replay_and_readiness(
    tmp_path: Path, actual: dict[str, Any], monkeypatch: Any
) -> None:
    work: dict[str, Any] = dict(
        checks=[],
        references=[],
        receipts=[],
        code_config_hashes=[],
        legacy_schedule=legacy(),
        additions=[],
        schedule=s.schedule(legacy(), []),
    )
    assert not s.qualified(work)
    path = tmp_path / "candidate.json"
    atomic_json(path, s.build(work, "20261116", 0, 1))
    assert s.replay(path)["passed"]
    value = json.loads(path.read_bytes())
    value["replay_input"]["work"]["schedule"]["units"] = "drift"
    value["reproducibility_checksum"] = s.canonical_hash(value["replay_input"])
    atomic_json(path, value)
    with pytest.raises(ValueError, match="schedule_drift"):
        s.replay(path)
    work["bundle"] = actual
    work["rows"] = s.r.probe_rows(actual)
    work["consumed_reference"] = old.ref(Path(actual["root"]) / "consumed.json")
    old.orchestration(Path(actual["root"]) / "bundle.json", "v780-test")
    work["consumed_reference"] = old.ref(Path(actual["root"]) / "consumed.json")
    monkeypatch.setattr(s, "qualified", lambda w: True)
    ready = s.build(work, "20261116", 0, 1)
    assert ready["schedule_ready_score"] == 1
    assert ready["model_work_authorized"] is False
    assert ready["sample_size_budget"]["effective_independent_groups"] == 0


def test_verification_receipt_and_coverage_controls(
    tmp_path: Path, actual: dict[str, Any], monkeypatch: Any
) -> None:
    class Tracer:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def combine(self, **kwargs: Any) -> None:
            pass

        def save(self) -> None:
            (tmp_path / ".coverage").write_text("test coverage control")

        def analysis2(self, path: str) -> tuple[Any, ...]:
            return path, [1], [], [], ""

        def set_option(self, *args: Any) -> None:
            pass

        def load(self) -> None:
            pass

    monkeypatch.setattr(s, "Coverage", Tracer)
    monkeypatch.setattr(s, "commands", lambda p: [("tiny", ["true"])])
    work: dict[str, Any] = dict(receipts=[], bundle=actual)
    old.orchestration(Path(actual["root"]) / "bundle.json", "v780-test")
    work.update(
        consumed_reference=old.ref(Path(actual["root"]) / "consumed.json"),
        isolated_child_receipts=[dict(passed=True)],
    )
    s.verify(work, tmp_path, time.monotonic() + 30)
    assert s.qualified(work)
    changed = deepcopy(work)
    changed["receipts"] = []
    assert not s.qualified(changed)
    changed = deepcopy(work)
    changed["receipts"][0]["passed"] = False
    assert not s.qualified(changed)
    changed = deepcopy(work)
    changed["coverage"]["files"][0]["missing"] = [1]
    assert not s.qualified(changed)
    changed = deepcopy(work)
    changed["consumed_reference"]["sha256"] = "bad"
    assert not s.qualified(changed)
    s.verify(dict(receipts=[]), tmp_path, time.monotonic() - 1)
    monkeypatch.setattr(s, "commands", lambda p: [("tiny", ["false"])])
    failed = dict(work, receipts=[])
    s.verify(failed, tmp_path, time.monotonic() + 30)
    assert not s.qualified(failed)


def test_main_normal_flow_without_recursive_suite(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.delenv("CARNOT_9153_REPOSITORY_HEALTH_REFERENCE", raising=False)
    monkeypatch.setattr(
        s,
        "Coverage",
        lambda **kw: SimpleNamespace(start=lambda: None, stop=lambda: None, save=lambda: None),
    )
    work = dict(
        checks=[],
        references=[],
        receipts=[],
        code_config_hashes=[],
        legacy_schedule=legacy(),
        additions=[],
        schedule=s.schedule(legacy(), []),
    )
    monkeypatch.setattr(s, "authenticate", lambda *a: deepcopy(work))
    monkeypatch.setattr(s, "measure", lambda *a: None)
    monkeypatch.setattr(s, "verify", lambda *a: None)
    monkeypatch.setattr(old.prior, "run_check", lambda *a, **kw: dict(passed=True))
    monkeypatch.setattr(s, "publish_primary", lambda p, v, check: atomic_json(p, v))
    output = tmp_path / (s.NAME + ".json")
    assert s.main(["--output", str(output)]) == 0
    assert s.main(["--replay", str(output)]) == 0
    assert s.terminal(output, tmp_path)["passed"]


def test_rehashed_costs_need_actual_lifetime(tmp_path: Path) -> None:
    work: dict[str, Any] = dict(
        checks=[],
        references=[],
        receipts=[],
        code_config_hashes=[],
        legacy_schedule=legacy(),
        additions=[],
        schedule=s.schedule(legacy(), []),
    )
    value = s.build(work, "20261116", 0, 1)
    saved = value["replay_input"]["work"]
    measured = {r["unit_id"]: 0.01 for r in saved["schedule"]["cost_model_rows"]}
    saved["schedule"] = s.schedule(legacy(), [], measured)
    value["reproducibility_checksum"] = s.canonical_hash(value["replay_input"])
    path = tmp_path / "tampered.json"
    atomic_json(path, value)
    with pytest.raises(ValueError, match="schedule_drift"):
        s.replay(path)


def test_exhaustion_does_not_launch_children(tmp_path: Path, monkeypatch: Any) -> None:
    plan = s.schedule(legacy(), [])
    work = dict(
        schedule=plan, legacy_schedule=legacy(), additions=[], code_config_hashes=[], receipts=[]
    )
    s.measure(work, tmp_path / "measurement", time.monotonic() - 1)
    assert work["measurement_error"] == "measurement_deadline_exhausted"
    assert not work["bundle"]["partitions"]
    bad = deepcopy(plan)
    bad["execution_graph"] *= 2
    with pytest.raises(ValueError, match="partition_union"):
        s.execute(bad, tmp_path / "bad", "bad", [], time.monotonic() + 1)
    empty = dict(plan, execution_graph=[], original_node_manifest=[])
    bundle = s.execute(empty, tmp_path / "empty", "empty", [], time.monotonic() - 1)
    assert not bundle["checks"]


def test_authenticated_scope_replay(tmp_path: Path) -> None:
    work = s.authenticate(s.ROOT, tmp_path / "auth")
    value = s.build(work, "20261116", 0, 1)
    path = tmp_path / "candidate.json"
    atomic_json(path, value)
    assert s.replay(path)["passed"]
    value["replay_input"]["work"]["additions"].pop()
    value["reproducibility_checksum"] = s.canonical_hash(value["replay_input"])
    atomic_json(path, value)
    with pytest.raises(ValueError, match="historical_scope_drift"):
        s.replay(path)


def test_measurement_phases_and_bundle_replay(
    tmp_path: Path, actual: dict[str, Any], monkeypatch: Any
) -> None:
    plan = actual["schedule"]
    h = old.schedule(
        dict(
            nodes=plan["original_node_manifest"],
            events=[],
            required_command_union=[],
            original_assertion_references=plan["original_assertion_hashes"],
        )
    )
    work: dict[str, Any] = dict(
        checks=[],
        references=[],
        receipts=[],
        code_config_hashes=[],
        legacy_schedule=h,
        additions=[],
        schedule=s.schedule(h, []),
    )
    receipt = old.prior.run_check("measurement_control", ["true"], tmp_path / "receipts")

    def child(*args: Any) -> dict[str, Any]:
        atomic_json(args[1] / "bundle.json", actual)
        old.orchestration(args[1] / "bundle.json", "v780-test")
        return deepcopy(actual)

    monkeypatch.setattr(s, "execute", child)
    monkeypatch.setattr(old.prior, "run_check", lambda *a, **kw: receipt)
    s.measure(work, tmp_path / "measurement", time.monotonic() + 120)
    assert work["checks"][-1]["observed"] == "unknown"
    value = s.build(work, "20261116", 0, 1)
    path = tmp_path / "measured.json"
    atomic_json(path, value)
    assert s.replay(path)["passed"]
    value["replay_input"]["work"]["rows"] = []
    value["reproducibility_checksum"] = s.canonical_hash(value["replay_input"])
    atomic_json(path, value)
    with pytest.raises(ValueError, match="bundle_drift"):
        s.replay(path)
    clock = iter([0, 0, 10])
    monkeypatch.setattr(s, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    s.measure(work, tmp_path / "late", 5)
    assert work["measurement_error"] == "measurement_deadline_exhausted"


def test_affected_failure_disqualifies_even_with_external_block() -> None:
    work = dict(
        checks=[dict(check="slow_child_causal_trace", passed=False)],
        references=[],
        receipts=[],
        code_config_hashes=[],
        verification_root="/tmp/absent9153",
    )
    value = s.build(work, "20261116", 0, 1)
    assert value["verdict_class"] == "disqualified"
    assert value["schedule_ready_score"] == 0


def test_probe_identity_is_unique_to_invocation(tmp_path: Path, monkeypatch: Any) -> None:
    plan = s.schedule(legacy(), [])
    work = dict(
        schedule=plan, legacy_schedule=legacy(), additions=[], code_config_hashes=[], receipts=[]
    )
    real = s.execute
    seen = []

    def capture(plan: Any, raw: Path, invocation: str, hashes: Any, stop: float) -> Any:
        seen.append(invocation)
        return real(plan, raw, invocation, hashes, stop)

    monkeypatch.setattr(s, "execute", capture)
    for identity in ["run-a", "run-b"]:
        s.measure(deepcopy(work), tmp_path / identity / "measurement", time.monotonic() - 1)
    assert seen == ["run-a", "run-b"]


def test_retained_repository_health_is_authenticated(tmp_path: Path) -> None:
    receipt = old.prior.run_check("repository_health", ["true"], tmp_path / "logs")
    reference = old.ref(tmp_path / "logs/repository_health/repository_health.receipt.json")
    assert s.repository_health(reference) == receipt
    with pytest.raises(ValueError, match="reference_drift"):
        s.repository_health(dict(reference, sha256="bad"))
    changed = dict(receipt, name="foreign")
    path = tmp_path / "foreign.json"
    atomic_json(path, changed)
    with pytest.raises(ValueError, match="foreign_repository_health"):
        s.repository_health(old.ref(path))


def test_retained_health_main_does_not_repeat_repository_suite(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """SCENARIO-REPORT-9153-V780: authenticate retained health without another suite."""
    health = old.prior.run_check("repository_health", ["true"], tmp_path / "health")
    reference = old.ref(Path(health["stdout_path"]).parent / "repository_health.receipt.json")
    monkeypatch.setenv("CARNOT_9153_REPOSITORY_HEALTH_REFERENCE", json.dumps(reference))
    monkeypatch.setattr(
        s,
        "Coverage",
        lambda **kw: SimpleNamespace(start=lambda: None, stop=lambda: None, save=lambda: None),
    )
    monkeypatch.setattr(
        s,
        "authenticate",
        lambda *a: dict(checks=[], references=[], receipts=[], code_config_hashes=[]),
    )
    monkeypatch.setattr(s, "measure", lambda *a: None)
    monkeypatch.setattr(s, "verify", lambda *a: None)
    monkeypatch.setattr(old.prior, "run_check", lambda *a, **kw: pytest.fail("repeated suite"))
    monkeypatch.setattr(s, "publish_primary", lambda p, v, check: atomic_json(p, v))
    output = tmp_path / (s.NAME + ".json")
    assert s.main(["--output", str(output)]) == 0
    work = json.loads(output.read_bytes())["replay_input"]["work"]
    assert work["repository_health"] == health
    assert work["repository_health_scope"] == "retained_unrelated_repository_health"
