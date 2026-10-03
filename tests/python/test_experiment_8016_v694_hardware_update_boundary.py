"""SCENARIO-REPORT-8016-SEAL: private script routes and publication consumers."""

import json
from pathlib import Path
import runpy
import sys

import pytest

from carnot import experiment_8016_v694_hardware_update_boundary as e
from carnot.reporting import hardware_update_8016 as h
from carnot.reporting.current_work_receipt import atomic_json


def test_real_cli_routes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-8016: script-path entry, freeze, replay and rejection execute."""
    fixture = tmp_path / "fixture.json"
    atomic_json(fixture, h.fixture())
    sealed = tmp_path / "sealed.json"
    atomic_json(sealed, dict(scope="circular_private_fixture"))
    plan = h.fixture()
    plan["cited_upstream_artifacts"] = [dict(path=str(sealed), sha256=e.sha256_file(sealed))]
    atomic_json(fixture, plan)
    output = tmp_path / "results" / (e.NAME + ".json")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            e.OWNED[-1],
            "--fixture-input",
            str(fixture),
            "--validation-worker",
            "--output",
            str(output),
        ],
    )
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(e.ROOT / e.OWNED[-1]), run_name="__main__")
    assert exit_info.value.code == 0
    assert e.main(["--cold-replay", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "circular_positive"
    assert all(
        Path(r["path"]).is_relative_to(output.parent / "raw") for r in value["raw_shard_hashes"]
    )
    value["quantized_update_ready_score"] = 0
    atomic_json(output, value)
    assert e.main(["--cold-replay", str(output)]) == 1
    assert e.main(["--date", "20261001"]) == 1
    assert (
        e.main(
            [
                "--root",
                str(tmp_path / "absent"),
                "--validation-worker",
                "--output",
                str(tmp_path / "blocked" / output.name),
            ]
        )
        == 0
    )


@pytest.mark.parametrize("passed", [True, False])
def test_owned_checks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, passed: bool) -> None:
    """REQ-REPORT-8016: owned check failure closes readiness."""
    fixture = tmp_path / "fixture.json"
    atomic_json(fixture, h.fixture())
    monkeypatch.setattr(
        e,
        "validate",
        lambda *args: (
            [dict(name="owned", required=True, passed=passed)],
            {p: dict(num_statements=1, covered_lines=1) for p in e.OWNED},
        ),
    )
    out = tmp_path / (e.NAME + ".json")
    assert e.main(["--fixture-input", str(fixture), "--output", str(out)]) == 0
    assert json.loads(out.read_bytes())["verdict_class"] == (
        "circular_positive" if passed else "disqualified"
    )


def test_validation_reduction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-8016-SEAL: command receipts retain exact coverage counts."""
    specs = e.commands(tmp_path / "raw", tmp_path)
    assert (tmp_path / "pytest").is_dir()
    monkeypatch.setattr(
        e,
        "run_commands",
        lambda *args, **kwargs: [
            dict(name="owned", scope="owned", log_path="relative.log"),
            dict(
                name="full_pytest", scope="repository_health", log_path=str(tmp_path / "health.log")
            ),
        ],
    )
    receipts, counts = e.validate(specs, tmp_path / "raw", tmp_path)
    assert counts == {}
    assert receipts[0]["required"] and not receipts[1]["required"]
    atomic_json(
        tmp_path / "coverage.json",
        dict(files={e.OWNED[0]: dict(summary=dict(num_statements=2, covered_lines=2))}),
    )
    assert e.validate(specs, tmp_path / "raw", tmp_path)[1][e.OWNED[0]]["covered_lines"] == 2
    assert all(r.name != "full_pytest" for r in e.commands(tmp_path / "raw", tmp_path))
    monkeypatch.setattr(
        e,
        "run_commands",
        lambda *args, **kwargs: [dict(name="owned", scope="owned", log_path="relative.log")],
    )
    reused, _ = e.validate(specs, tmp_path / "raw", tmp_path)
    assert reused[-1]["reused_diagnostic"]


@pytest.mark.parametrize("fault", ["reader", "final", "code_drift"])
def test_final_guards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str) -> None:
    """SCENARIO-REPORT-8016-SEAL: publication or code drift cannot report success."""
    source = tmp_path / "fixture.json"
    atomic_json(source, h.fixture())
    out = tmp_path / (e.NAME + ".json")
    if fault == "reader":
        monkeypatch.setattr(e, "reader_receipt", lambda *args, **kwargs: dict(passed=False))
    elif fault == "final":
        original = e.terminal_check
        calls = 0

        def check(path: Path) -> dict:
            nonlocal calls
            calls += 1
            return original(path) if calls == 1 else dict(passed=False)

        monkeypatch.setattr(e, "terminal_check", check)
    else:
        original_hash = e.sha256_file
        calls_hash = 0

        def changed(path: Path) -> str:
            nonlocal calls_hash
            if path == e.ROOT / e.OWNED[0]:
                calls_hash += 1
                if calls_hash > 1:
                    return "sha256:changed"
            return original_hash(path)

        monkeypatch.setattr(e, "sha256_file", changed)
    assert (
        e.main(["--fixture-input", str(source), "--validation-worker", "--output", str(out)]) == 1
    )
