"""REQ-REPORT-8929: health reuse authenticates evidence without suppressing owned work."""

import pytest

from carnot.reporting import v763_capstone as m
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.experiment_7303_validation_scope import CommandSpec


@pytest.mark.parametrize("fault", [None, "missing", "duplicate", "command", "log", "bytes"])
def test_validation_namespace_authenticates_health(monkeypatch, tmp_path, fault):
    """SCENARIO-REPORT-8929-CONTRACT: the real wrapper rejects corrupt health custody."""
    log = tmp_path / "health.log"
    log.write_text("original failed health")
    old = dict(
        name="global_python_health",
        scope="repository_health",
        command_argv=["pytest", "tests/python", "-q"],
        passed=False,
        log_path=log.name,
        log_sha256=m.reference(log)["sha256"],
    )
    receipts = [old]
    if fault == "missing":
        receipts = []
    elif fault == "duplicate":
        receipts = [old, old]
    elif fault == "command":
        old["command_argv"] = ["unrelated"]
    elif fault == "log":
        log.unlink()
    path = tmp_path / "validation.json"
    atomic_json(path, dict(receipts=receipts))
    monkeypatch.setattr(m, "HEALTH_REFERENCE", m.reference(path))
    if fault == "bytes":
        atomic_json(path, dict(receipts=[]))
    calls = []

    def run(root, commands, **kwargs):
        calls.extend(commands)
        return [dict(name=s.name, scope=s.scope, passed=True) for s in commands]

    monkeypatch.setattr(m, "run_commands", run)
    owned = CommandSpec("global_python_health", ("owned-check",), "owned")
    plan = [owned, CommandSpec(old["name"], ("pytest", "tests/python", "-q"), old["scope"])]
    if fault:
        with pytest.raises(ValueError):
            m.checker.run_commands(tmp_path, plan)
        assert calls == []
    else:
        result = m.checker.run_commands(tmp_path, plan)
        assert calls == [owned]
        assert result[0]["passed"] is True
        assert result[1] == dict(old, reused_from=m.HEALTH_REFERENCE)
