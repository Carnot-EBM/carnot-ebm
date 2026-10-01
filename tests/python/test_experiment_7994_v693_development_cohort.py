"""REQ-REPORT-7994: real private CLI, evaluator lineage and cold replay."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from carnot import experiment_7994_v693_development_cohort as e
from carnot.reporting.current_work_receipt import atomic_json
from test_development_cohort_7994 import training


def fixture(tmp_path):
    training(tmp_path)
    original = tmp_path / "original.json"
    atomic_json(original, dict(request_rows=[]))
    source = tmp_path / "results/experiment_7980_v692_evidence_features.json"
    atomic_json(
        source,
        dict(
            experiment_id=7980,
            task_id="exp7980-evidence-features",
            run_date="20261001",
            feature_views_ready_score=1,
            flagged_adversarial=False,
            verdict_class="null",
            rows=[],
            original_public=e.d.c.reference(original),
            exposure_audit=dict(discovered_exclusions=[]),
        ),
    )
    return source


def cli(args, cwd):
    argv = [sys.executable, str(e.ROOT / e.OWNED[-1]), *map(str, args)]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    config = os.environ.get("CARNOT_7994_COVERAGE_CONFIG")
    if config:
        argv[1:1] = ["-m", "coverage", "run", "--rcfile=" + config]
    return subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True, timeout=60)


def test_private_cli_success_blocked_replay(tmp_path):
    fixture(tmp_path)
    output = tmp_path / "out" / (e.NAME + ".json")
    result = cli(["--fixture-root", tmp_path, "--output", output], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    v = json.loads(output.read_text())
    assert all(span["end_s"] <= v["duration_s"] for span in v["phase_spans"])
    assert v["verdict_class"] == "circular_positive"
    assert v["cohort_ready_score"] == 0
    assert len(v["stream_slot_rows"]) == 256
    assert set(v["public_role_manifests"]) == {"calibration", "stream", "retention"}
    terminal = json.loads(Path(v["terminal_validation_sidecar_path"]).read_text())
    validation = json.loads(Path(terminal["sidecar_path"]).read_text())["report"]
    assert Path(validation["receipts"][0]["command_argv"][-1]).parent.name.startswith(
        "carnot-7994-terminal-"
    )
    assert cli(["--cold-replay", output], tmp_path).returncode == 0
    changed = copy.deepcopy(v)
    changed["stream_slot_rows"][0]["reveal_at_slot"] = 0
    atomic_json(output, changed)
    assert cli(["--cold-replay", output], tmp_path).returncode == 1
    other = tmp_path / "absent"
    blocked = other / "out" / (e.NAME + ".json")
    assert cli(["--root", other, "--private-worker", "--output", blocked], tmp_path).returncode == 0
    assert json.loads(blocked.read_text())["verdict_class"] == "blocked"
    assert cli(["--cold-replay", blocked], tmp_path).returncode == 0
    assert (
        cli(
            ["--fixture-root", tmp_path, "--output", e.ROOT / "results" / (e.NAME + ".json")],
            tmp_path,
        ).returncode
        == 1
    )
    assert cli(["--date", "20260930"], tmp_path).returncode == 2


def test_evaluator_controls_and_lineage(tmp_path):
    fixture(tmp_path)
    plan = e.d.authenticate(tmp_path, fixture=True)
    raw = tmp_path / "raw"
    value = e.measure(plan, tmp_path, raw)
    assert value["sample_size_budget"]["completed"] == 384
    e.replay(value)
    assert value["positive_control_results"]["passed"]
    sources, responses = e.d.public_training(tmp_path)
    roster, public = e.d.select(sources[:2], responses[:2], set(), set())
    seal = e.d.seal(raw / "small", roster, public)
    responses_path = tmp_path / "data/ragtruth/response.jsonl"
    rows = [json.loads(line) for line in responses_path.read_text().splitlines()[:-1]]
    rows[0]["labels"] = [
        dict(
            start=0,
            end=2,
            text="XX",
            label_type="unsupported",
            implicit_true=False,
            due_to_null=False,
            meta=None,
        )
    ]
    responses_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary = e.d.evaluate(tmp_path, seal, raw / "small/evaluator")
    assert summary["rows"][0]["status"] == "excluded" or summary["rows"][1]["status"] == "excluded"
    assert all("y" not in r for r in summary["rows"])
    rows[1]["response"] = ""
    responses_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    sources, responses = e.d.public_training(tmp_path)
    roster, public = e.d.select(sources[:2], responses[:2], set(), set())
    empty_seal = e.d.seal(raw / "empty", roster, public)
    empty_summary = e.d.evaluate(tmp_path, empty_seal, raw / "empty/evaluator")
    assert any(r["failure"] == "empty_answer" for r in empty_summary["rows"])
    with patch.object(e.d, "check_roles", side_effect=ValueError("duplicate_role")):
        with pytest.raises(ValueError, match="duplicate_role"):
            e.d.seal(raw / "bad", roster, public)
    manifest = e.freeze_commands(raw, tmp_path / "scratch")
    assert manifest["artifact_guard_enabled"]
    v = e.base([])
    e.apply_validation(v, [dict(name="owned", passed=False, exit_code=1)])
    assert v["verdict_class"] == "disqualified" and v["cohort_ready_score"] == 0


def test_terminal_failure_paths_and_owned_receipts(tmp_path):
    """REQ-REPORT-7994: owned errors cannot masquerade as external blocking."""
    fixture(tmp_path)
    plan = e.d.authenticate(tmp_path, fixture=True)
    with patch.object(e, "run_commands", return_value=[dict(passed=False)]):
        with pytest.raises(ValueError, match="evaluator_child_failed"):
            e.measure(plan, tmp_path, tmp_path / "failed")
    training_root = tmp_path / "short"
    training(training_root, 2)
    short = e.measure(plan, training_root, tmp_path / "short_raw")
    assert short["verdict_class"] == "blocked" and short["selection_shortfalls"]["stream"] == 256
    short["rows_hash"] = "changed"
    with pytest.raises(ValueError, match="rows_drift"):
        e.replay(short)
    short["rows_hash"] = e.canonical_hash(short["rows"])
    short["rows"][0]["failure"] = "changed"
    short["rows_hash"] = e.canonical_hash(short["rows"])
    with pytest.raises(ValueError, match="evaluator_rows_drift"):
        e.replay(short)
    with (
        patch.object(e, "reader_receipt", return_value=dict(passed=False)),
        patch.object(e, "publish_primary", return_value={}),
    ):
        with pytest.raises(ValueError, match="primary_reader_failure"):
            e.publish(tmp_path / "reader" / (e.NAME + ".json"), e.base([]))
    assert e.main(["--evaluate", str(tmp_path / "not_a_seal")]) == 1
    assert e.main(["--output", str(tmp_path / "wrong.json")]) == 1
    saved_freeze = e.freeze_commands
    for complete in [False, True]:

        def frozen(raw, scratch):
            manifest = saved_freeze(raw, scratch)
            manifest["commands"] = []
            if complete:
                atomic_json(
                    Path(manifest["coverage_json"]),
                    dict(
                        files={
                            p: dict(summary=dict(num_statements=1, missing_lines=0))
                            for p in e.OWNED
                        }
                    ),
                )
            return manifest

        output = tmp_path / str(complete) / (e.NAME + ".json")
        with patch.object(e, "freeze_commands", side_effect=frozen):
            assert e.main(["--root", str(tmp_path / "missing"), "--output", str(output)]) == 0
        assert json.loads(output.read_text())["verdict_class"] == (
            "blocked" if complete else "disqualified"
        )
    v = e.base([])
    e.apply_validation(v, [dict(name="repository_health", passed=False, exit_code=2)])
    assert v["verdict_class"] == "null"
