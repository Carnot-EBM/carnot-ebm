"""Private producer and consumer checks for REQ-REPORT-7942."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot import experiment_7942_v689_sentence_labels as producer
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar, reader_receipt


def fixture(path):
    """Give every source its own cluster while retaining both target classes."""
    public, responses, evaluators, roles, sources = [], [], [], [], []
    for i in range(64):
        source = f"Source {i}." if i != 0 else "Café source."
        answer = f"Answer {i}."
        family = f"opaque-{i}"
        public.append(
            dict(
                family_id=family,
                source_bytes=source.encode().hex(),
                answer_bytes=answer.encode().hex(),
            )
        )
        annotations = (
            []
            if i % 2 == 0
            else [
                dict(
                    start=0,
                    end=6,
                    text="Answer",
                    implicit_true=False,
                    due_to_null=False,
                    label_type="human",
                    meta="fixture",
                )
            ]
        )
        responses.append(
            dict(
                id=str(i),
                source_id=str(i),
                response=answer,
                quality="good",
                labels=annotations,
                model="historical",
            )
        )
        evaluators.append(dict(family_id=family, response_id=str(i), role="evaluation"))
        roles.append(
            dict(
                family_id=family,
                role="evaluation",
                status="completed",
                source_cluster_id=producer.labels.digest(source.encode()),
            )
        )
        sources.append(dict(source_id=str(i), source_info=source))
    data = dict(
        public=public, responses=responses, evaluators=evaluators, roles=roles, sources=sources
    )
    atomic_json(path, data)
    return data


def test_private_success_cold_replay_and_actual_readers(tmp_path):
    """SCENARIO-REPORT-7942-TERMINAL: newer sidecars do not hide the primary."""
    input_path = tmp_path / "inputs.json"
    fixture(input_path)
    output = tmp_path / "success" / (producer.NAME + ".json")
    assert (
        producer.main(
            ["--date", "20260930", "--fixture-input", str(input_path), "--output", str(output)]
        )
        == 0
    )
    value = json.loads(output.read_text())
    assert value["verdict_class"] == "circular_positive"
    assert value["sentence_labels_ready_score"] == 0
    assert value["model_specs"] == []
    assert producer.reconstruct(value)["label_counts"] == {"0": 32, "1": 32}
    sidecar = Path(value["terminal_validation_sidecar_path"])
    assert read_bound_sidecar(output, sidecar)
    os.utime(sidecar, ns=(output.stat().st_mtime_ns + 10**9,) * 2)
    actual = reader_receipt(
        producer.TASK, output.parent, field="sentence_labels_ready_score", expected=0
    )
    assert actual["passed"] and actual["gate_sha256"] == sha256_file(output)
    replay_dir = tmp_path / "replay"
    assert (
        producer.main(
            [
                "--date",
                "20260930",
                "--cold-replay",
                str(output),
                "--output",
                str(replay_dir / (producer.NAME + ".json")),
            ]
        )
        == 0
    )
    changed = deepcopy(value)
    changed["label_counts"]["1"] += 1
    with pytest.raises(ValueError, match="reduction_drift"):
        producer.reconstruct(changed)
    changed = deepcopy(value)
    changed["sentence_public_manifest"]["sha256"] = "bad"
    with pytest.raises(ValueError, match="hash_drift"):
        producer.reconstruct(changed)


def test_real_cli_success_negative_blocked_and_replay(tmp_path):
    """REQ-REPORT-7942: subprocess routes have distinct publication directories."""
    input_path = tmp_path / "inputs.json"
    fixture(input_path)
    cli = "scripts/experiments/experiment_7942_v689_sentence_labels.py"
    prefix = [sys.executable, "-u"]
    if os.environ.get("CARNOT_7942_COVERAGE_FILE"):
        prefix += [
            "-m",
            "coverage",
            "run",
            "--parallel-mode",
            "--data-file=" + os.environ["CARNOT_7942_COVERAGE_FILE"],
            "--include=" + producer.INCLUDE,
        ]
    base = prefix + [cli, "--date", "20260930"]
    output = tmp_path / "cli-success" / (producer.NAME + ".json")
    commands = [
        (base + ["--fixture-input", str(input_path), "--output", str(output)], 0, "published"),
        (
            base
            + [
                "--cold-replay",
                str(output),
                "--output",
                str(tmp_path / "cli-replay" / (producer.NAME + ".json")),
            ],
            0,
            "replay_passed",
        ),
        (
            base
            + [
                "--root",
                str(tmp_path / "missing"),
                "--output",
                str(tmp_path / "cli-blocked" / (producer.NAME + ".json")),
            ],
            0,
            "published",
        ),
        (
            prefix
            + [
                cli,
                "--date",
                "20260929",
                "--output",
                str(tmp_path / "cli-negative" / (producer.NAME + ".json")),
            ],
            2,
            "20260930",
        ),
    ]
    for argv, expected, reason in commands:
        child = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, "PYTHONPATH": "python:."},
            timeout=60,
        )
        assert child.returncode == expected, child.stdout + child.stderr
        assert reason in child.stdout + child.stderr
    value = json.loads((tmp_path / "cli-blocked" / (producer.NAME + ".json")).read_text())
    assert value["verdict_class"] == "blocked" and value["gate_check_summary"]
    data = json.loads(output.read_text())
    data["label_counts"]["0"] = 999
    bad = tmp_path / "cli-negative-replay" / "bad.json"
    atomic_json(bad, data)
    child = subprocess.run(
        base
        + [
            "--cold-replay",
            str(bad),
            "--output",
            str(tmp_path / "cli-negative-replay" / (producer.NAME + ".json")),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert child.returncode == 1 and "reduction_drift" in child.stdout + child.stderr


@pytest.mark.parametrize("tamper", [False, True])
def test_direct_cold_replay_without_repository_pythonpath(tmp_path, tamper):
    """SCENARIO-REPORT-7942-DIRECT-CLI: replay works outside the checkout."""
    input_path = tmp_path / "inputs.json"
    fixture(input_path)
    value = producer.build_fixture(input_path, tmp_path / "primitive")
    if tamper:
        value["label_counts"]["0"] += 1
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, value)
    prefix = [sys.executable, "-u"]
    if os.environ.get("CARNOT_7942_COVERAGE_FILE"):
        prefix += [
            "-m",
            "coverage",
            "run",
            "--parallel-mode",
            "--data-file=" + os.environ["CARNOT_7942_COVERAGE_FILE"],
            "--include=" + producer.INCLUDE,
        ]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    child = subprocess.run(
        prefix
        + [
            str(producer.ROOT / "scripts/experiments" / (producer.NAME + ".py")),
            "--cold-replay",
            str(candidate),
            "--output",
            str(tmp_path / "replay" / (producer.NAME + ".json")),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert child.returncode == (1 if tamper else 0), child.stdout + child.stderr
    assert ("reduction_drift" if tamper else "replay_passed") in child.stdout + child.stderr


def test_authentication_terminal_errors_and_required_failure(tmp_path, monkeypatch):
    """REQ-REPORT-7942: external custody and owned checks have distinct outcomes."""
    failures, _ = producer.authenticate(tmp_path)
    assert failures and failures[0]["observed"] is None
    root = Path(__file__).resolve().parents[2]
    failures, upstream = producer.authenticate(root)
    assert failures == [] and upstream["source_boundary_ready_score"] == 1
    copied = tmp_path / "results/experiment_7892_v685_source_boundary.json"
    atomic_json(copied, {"task_id": "wrong"})
    failures, _ = producer.authenticate(tmp_path)
    assert failures[0]["field"] == "sha256"
    input_path = tmp_path / "inputs.json"
    data = fixture(input_path)
    data["responses"][0]["labels"] = None
    atomic_json(input_path, data)
    assert (
        producer.main(
            [
                "--fixture-input",
                str(input_path),
                "--output",
                str(tmp_path / "negative" / (producer.NAME + ".json")),
            ]
        )
        == 1
    )
    with pytest.raises(SystemExit) as error:
        producer.main(["--date", "20260929"])
    assert error.value.code == 2
    checks = [{"name": "owned", "passed": False, "exit_code": 1}]
    value = producer.base([], [])
    value.update(sentence_labels_ready_score=1, verdict_class="null")
    producer.apply_validation(value, checks)
    assert value["verdict_class"] == "disqualified" and value["sentence_labels_ready_score"] == 0
    assert producer.apply_validation(producer.base([], []), [{"passed": True}]) is None
    monkeypatch.setattr(
        producer,
        "run_commands",
        lambda *args, **kwargs: [
            dict(name=spec.name, exit_code=0, passed=True, output_tail="", log_path="unused")
            for spec in args[1]
        ],
    )
    manifest = producer.freeze_commands(tmp_path / "validation")
    assert manifest["coverage_includes"] == producer.INCLUDE
    assert all(c["deadline_s"] <= 600 for c in manifest["commands"])
    assert producer.execute_commands(manifest, tmp_path / "logs")


def test_validation_coverage_workspace_combines_frozen_cli(tmp_path):
    """SCENARIO-REPORT-7942-COVERAGE-WORKSPACE: guarded children save real data."""
    from coverage import CoverageData

    raw = tmp_path / "raw"
    manifest = producer.freeze_commands(raw)
    coverage_file = Path(manifest["coverage_file"])
    workspace = coverage_file.parent
    assert workspace != raw
    assert workspace.is_dir() and not workspace.is_relative_to(producer.ROOT / "results")
    fixture(raw / "validation_input.json")
    commands = {item["name"]: item for item in manifest["commands"]}
    for name in ("real_fixture_cli", "coverage_combine"):
        child = subprocess.run(
            commands[name]["argv"],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        assert child.returncode == 0, child.stdout + child.stderr
    assert commands["coverage_combine"]["argv"][-1] == str(workspace)
    assert coverage_file.is_file() and coverage_file.stat().st_size > 0
    data = CoverageData(basename=str(coverage_file))
    data.read()
    assert {str(producer.ROOT / path) for path in producer.OWNED} <= data.measured_files()


def test_fixture_cold_reconstruction_offsets_features_and_manifest_tamper(tmp_path):
    """REQ-REPORT-7942: cold reduction starts at primitive rows and exact inputs."""
    path = tmp_path / "input.json"
    fixture(path)
    raw = tmp_path / "raw"
    value = producer.build_fixture(path, raw)
    assert producer.reconstruct(value)
    for key in ("sentence_evaluator_manifest", "sentence_cohort_manifest"):
        changed = deepcopy(value)
        changed[key]["sha256"] = "changed"
        with pytest.raises(ValueError, match="hash_drift"):
            producer.reconstruct(changed)
    offset_path = Path(value["offset_join_rows"]["path"])
    old = offset_path.read_bytes()
    offset_path.write_text("[]")
    with pytest.raises(ValueError, match="hash_drift"):
        producer.reconstruct(value)
    offset_path.write_bytes(old)
    # Relabeling primitive targets without their exact input authority fails.
    target = Path(value["sentence_evaluator_manifest"]["path"])
    rows = producer.read_jsonl(target)
    rows[0]["y"] = 1
    producer.write_jsonl(target, rows)
    value["sentence_evaluator_manifest"]["sha256"] = sha256_file(target)
    with pytest.raises(ValueError, match="evaluator_drift"):
        producer.reconstruct(value)


def test_terminal_validator_and_full_result_separation(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7942-TERMINAL: final validators bind only checked bytes."""
    candidate = tmp_path / "candidate.json"
    value = producer.base([], [])
    atomic_json(candidate, value)
    assert producer.terminal_check(candidate)["passed"]
    monkeypatch.setattr(
        producer,
        "run_commands",
        lambda *args, **kwargs: [
            dict(
                name="adversarial",
                passed=False,
                exit_code=1,
                output_tail='{"flagged_count":1}',
                log_path="fixture",
            )
        ],
    )
    report = producer.terminal_check(candidate)
    assert not report["passed"] and report["flagged_adversarial"] is True


def test_live_validation_flow_and_external_label_failure(tmp_path, monkeypatch):
    """REQ-REPORT-7942: owned checks fail closed without recursing into pytest."""
    path = tmp_path / "input.json"
    fixture(path)
    private = producer.build_fixture(path, tmp_path / "fixture-raw")
    private.update(verdict_class="null", honest_verdict="complete_null_transport")
    monkeypatch.setattr(producer, "build_live", lambda root, raw: deepcopy(private))

    def measured_fixture(manifest, logs):
        atomic_json(
            logs.parent / "coverage.json",
            {"files": {"fixture.py": {"summary": {"num_statements": 1, "covered_lines": 1}}}},
        )
        return [{"passed": True}]

    monkeypatch.setattr(producer, "execute_commands", measured_fixture)
    output = tmp_path / "live-flow" / (producer.NAME + ".json")
    assert producer.main(["--output", str(output)]) == 0
    assert (
        json.loads(output.read_text())["coverage_statement_counts"]["fixture.py"]["covered_lines"]
        == 1
    )
    monkeypatch.setattr(producer, "execute_commands", lambda *args: [{"passed": False}])
    output = tmp_path / "owned-failure" / (producer.NAME + ".json")
    assert producer.main(["--output", str(output)]) == 0
    assert json.loads(output.read_text())["verdict_class"] == "disqualified"

    def unavailable(root, raw):
        raise ValueError("external_offset_mismatch")

    monkeypatch.setattr(producer, "build_live", unavailable)
    output = tmp_path / "external-label-failure" / (producer.NAME + ".json")
    assert producer.main(["--output", str(output)]) == 0
    assert json.loads(output.read_text())["verdict_class"] == "blocked"


def test_real_cached_originals_and_cold_reconstruction(tmp_path, monkeypatch):
    """REQ-VERIFY-7942: all 640 families retain the pinned roles and public limits."""
    value = producer.build_live(producer.ROOT, tmp_path / "cached-corpus")
    assert len(value["rows"]) == 640
    assert value["role_counts"]["evaluation"] == 64
    assert value["source_cluster_counts"]["all_roles"] == 640
    assert all(row["passed"] for row in value["mutation_rows"])
    assert producer.reconstruct(value)["sample_size_budget"]["intended"] == 64
    failed, upstream = producer.authenticate(producer.ROOT)
    assert not failed
    upstream["role_counts"] = {}
    monkeypatch.setattr(producer, "authenticate", lambda root: ([], upstream))
    with pytest.raises(ValueError, match="original_role_roster"):
        producer.build_live(producer.ROOT, tmp_path / "bad-roster")
    monkeypatch.setattr(producer, "authenticate", lambda root: ([{"field": "missing"}], {}))
    with pytest.raises(ValueError, match="cold_custody"):
        producer.reconstruct(value)


def test_public_feature_tampering_and_terminal_reclassification(tmp_path, monkeypatch):
    """REQ-REPORT-7942: changes to saved public views or features cannot pass replay."""
    path = tmp_path / "input.json"
    fixture(path)
    value = producer.build_fixture(path, tmp_path / "primitive")
    cohort = Path(value["sentence_cohort_manifest"]["path"])
    original = cohort.read_bytes()
    data = json.loads(original)
    data["boundaries"][0]["intervals"] = [[0, 1]]
    atomic_json(cohort, data)
    value["sentence_cohort_manifest"]["sha256"] = sha256_file(cohort)
    with pytest.raises(ValueError, match="public_reconstruction_drift"):
        producer.reconstruct(value)
    cohort.write_bytes(original)
    value["sentence_cohort_manifest"]["sha256"] = sha256_file(cohort)
    feature = Path(value["feature_manifest"]["path"])
    data = producer.read_jsonl(feature)
    data[0]["features"][0] += 1
    producer.write_jsonl(feature, data)
    value["feature_manifest"]["sha256"] = sha256_file(feature)
    with pytest.raises(ValueError, match="feature_drift"):
        producer.reconstruct(value)
    attempts = []

    def terminal(candidate):
        attempts.append(sha256_file(candidate))
        return dict(passed=len(attempts) > 1, flagged_adversarial=False)

    monkeypatch.setattr(producer, "terminal_check", terminal)
    output = tmp_path / "terminal-recheck" / (producer.NAME + ".json")
    assert producer.main(["--fixture-input", str(path), "--output", str(output)]) == 0
    assert attempts[0] != attempts[1] == attempts[2]
    assert json.loads(output.read_text())["verdict_class"] == "disqualified"
    monkeypatch.setattr(producer, "reader_receipt", lambda *args, **kwargs: {"passed": False})
    assert (
        producer.main(
            [
                "--fixture-input",
                str(path),
                "--output",
                str(tmp_path / "bad-reader" / (producer.NAME + ".json")),
            ]
        )
        == 1
    )


def test_unsafe_readiness_and_code_receipt_drift(tmp_path):
    """REQ-REPORT-7942: readiness cannot exceed independently reconstructed custody."""
    path = tmp_path / "input.json"
    fixture(path)
    value = producer.build_fixture(path, tmp_path / "raw")
    value["sentence_labels_ready_score"] = 1
    with pytest.raises(ValueError, match="unsafe_readiness"):
        producer.reconstruct(value)
    empty = producer.base([], [])
    empty["sentence_labels_ready_score"] = 1
    with pytest.raises(ValueError, match="unsafe_readiness"):
        producer.reconstruct(empty)
    value["sentence_labels_ready_score"] = 0
    value["code_config_hashes"] = [{"path": str(path), "sha256": "wrong"}]
    with pytest.raises(ValueError, match="hash_drift"):
        producer.reconstruct(value)
