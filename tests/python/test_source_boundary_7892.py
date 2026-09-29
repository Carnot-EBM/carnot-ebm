"""REQ-REPORT-7892-V685: current source custody and producer identity."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

from coverage import CoverageData
import pytest

from carnot.reporting import source_boundary_7892 as boundary
from scripts.experiments import experiment_7892_v685_source_boundary as producer


def _families() -> tuple[list[dict], list[dict]]:
    rows = []
    public = []
    for role, count in boundary.ORIGINAL_ROLES.items():
        for index in range(count):
            family = f"{role}-{index}"
            source = f"Source {family}.".encode()
            answer = f"Answer {family}.".encode()
            rows.append(
                {
                    "family_id": family,
                    "role": role,
                    "source_group": f"group-{family}",
                    "source_sha256": f"source-{family}",
                    "label_provenance": {"human_label": index % 2, "annotation_byte_offsets": []},
                    "status": "completed",
                }
            )
            public.append(
                {"family_id": family, "source_bytes": source.hex(), "answer_bytes": answer.hex()}
            )
    return rows, public


def test_role_split_and_label_isolation() -> None:
    """SCENARIO-REPORT-7892-CUSTODY: labels cannot choose policy role."""
    rows, public = _families()
    qualified, evaluators, counts = boundary.qualify(rows, public)
    assert counts == boundary.EIGHT_ROLES
    assert len(qualified) == len(evaluators) == 640
    assert {row["label_scope"] for row in evaluators} == {"response"}
    assert all("human_label" not in row for row in qualified)
    changed = deepcopy(rows)
    for row in changed:
        row["label_provenance"]["human_label"] ^= 1
    assert [row["role"] for row in boundary.qualify(changed, public)[0]] == [
        row["role"] for row in qualified
    ]
    assert [row["human_label"] for row in boundary.qualify(changed, public)[1]] != [
        row["human_label"] for row in evaluators
    ]


def test_policy_role_count_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-7892-CUSTODY: a corrupt policy splitter cannot qualify."""
    rows, public = _families()
    monkeypatch.setattr(
        boundary.source_boundary_7880,
        "policy_subroles",
        lambda records: {
            row["family_id"]: "policy_design" for row in records if row["role"] == "policy"
        },
    )
    with pytest.raises(ValueError, match="policy_role_count"):
        boundary.qualify(rows, public)


@pytest.mark.parametrize(
    "fault", ["missing", "role", "duplicate", "family", "label", "public_label"]
)
def test_qualification_rejects_drift(fault: str) -> None:
    """SCENARIO-REPORT-7892-CUSTODY: no absent or changed role/label is filled."""
    rows, public = _families()
    if fault == "missing":
        rows.pop()
        public.pop()
    elif fault == "role":
        rows[0]["role"] = "evaluation"
    elif fault == "duplicate":
        rows[1]["source_group"] = rows[0]["source_group"]
    elif fault == "family":
        rows[1]["family_id"] = rows[0]["family_id"]
    elif fault == "label":
        rows[0]["label_provenance"]["human_label"] = None
    else:
        public[0]["label"] = 1
    with pytest.raises(ValueError):
        boundary.qualify(rows, public)


def test_identity_and_budget() -> None:
    """SCENARIO-REPORT-7892-TERMINAL: old producer identity cannot publish."""
    artifact = {
        "experiment_id": 7892,
        "task_id": "exp7892-source-boundary",
        "milestone": "2026.09.685",
        "execution_venue": "host",
    }
    boundary.require_identity(artifact)
    for key in artifact:
        with pytest.raises(ValueError, match="identity"):
            boundary.require_identity({**artifact, key: "wrong"})
    assert (
        boundary.budget(
            [{"family_id": "a", "status": "completed"}, {"family_id": "b", "status": "failed"}], 2
        )["failed"]
        == 1
    )


@pytest.mark.parametrize("fault", ["none", "missing", "malformed", "label", "replay"])
def test_private_real_cli(tmp_path: Path, fault: str) -> None:
    """SCENARIO-REPORT-7892-VALIDATION: real public CLI and cold replay."""
    public = tmp_path / "space path" / "public.jsonl"
    public.parent.mkdir()
    feature = public.with_name("feature.jsonl")
    row = {"family_id": "fixture", "source_bytes": b"A. B.".hex(), "answer_bytes": b"A.".hex()}
    if fault == "malformed":
        row["source_bytes"] = "bad hex"
    if fault == "label":
        row["human_label"] = 1
    if fault != "missing":
        public.write_text(json.dumps(row) + "\n")
    command = [
        sys.executable,
        "-u",
        str(producer.__file__),
        "--fixture-public",
        str(public),
        "--fixture-output",
        str(feature),
    ]
    if fault == "replay":
        assert subprocess.run(command, capture_output=True, check=False).returncode == 0
        command.append("--cold-replay")
    result = subprocess.run(command, capture_output=True, check=False)
    assert (result.returncode == 0) == (fault in {"none", "replay"})
    if fault in {"none", "replay"}:
        assert feature.is_file()


def test_private_custody_and_cold_feature_replay(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-REPORT-7892-CUSTODY: 640 private joins replay without labels."""
    rows, public = _families()
    monkeypatch.setattr(
        producer.acquisition, "acquire", lambda _manifest, _progress: (rows, public)
    )
    monkeypatch.setattr(producer.prior, "LICENSE", tmp_path / "license.json")
    producer.prior.LICENSE.write_text("{}")
    artifact = producer.base(0.0, [], [])
    producer.custody(
        artifact,
        {"development_manifest_path": str(producer.prior.LICENSE)},
        {
            "license": "MIT",
            "repository": "https://example.test",
            "commit": "revision",
            "label_authority": "human",
        },
        tmp_path / "raw",
        0.0,
    )
    assert artifact["role_counts"] == boundary.EIGHT_ROLES
    assert artifact["sample_size_budget"]["independent"] == 640
    assert not any("human_label" in row for row in artifact["rows"])
    for index in range(2):
        producer.source_projection.replay_file(
            tmp_path / f"raw/public_{index}.jsonl", tmp_path / f"raw/features_{index}.jsonl"
        )
    assert len(artifact["public_shards"]) == len(artifact["evaluator_shards"]) == 2
    assert len(artifact["feature_shards"]) == 2
    assert all(
        Path(item["path"]).stat().st_size < 50 * 1024 * 1024 for item in artifact["feature_shards"]
    )
    cohort = Path(artifact["cohort_manifest_path"])
    value = json.loads(cohort.read_text())
    value["rows"] = []
    cohort.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="cohort_hash_collision"):
        producer.custody(
            artifact,
            {"development_manifest_path": str(producer.prior.LICENSE)},
            {
                "license": "MIT",
                "repository": "https://example.test",
                "commit": "revision",
                "label_authority": "human",
            },
            tmp_path / "raw",
            0.0,
        )
    public[0]["answer_bytes"] = ""
    producer.custody(
        artifact,
        {"development_manifest_path": str(producer.prior.LICENSE)},
        {
            "license": "MIT",
            "repository": "https://example.test",
            "commit": "revision",
            "label_authority": "human",
        },
        tmp_path / "raw_exclusion",
        0.0,
    )
    assert artifact["rows"][0]["status"] == "excluded"


def test_custody_rejects_oversized_raw_shard(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-REPORT-7892-CUSTODY: a shard at the 50 MiB limit fails closed."""
    rows, public = _families()
    monkeypatch.setattr(producer.acquisition, "acquire", lambda *_: (rows, public))
    original_write = producer.source_projection.write_jsonl

    def write_oversized(path: Path, part: list[dict]) -> None:
        if path.name == "public_0.jsonl":
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as stream:
                stream.truncate(50 * 1024 * 1024)
        else:
            original_write(path, part)

    monkeypatch.setattr(producer.source_projection, "write_jsonl", write_oversized)
    with pytest.raises(ValueError, match="raw_shard_too_large:.*public_0.jsonl"):
        producer.custody(producer.base(0.0, [], []), {}, {}, tmp_path / "raw", 0.0)


def test_frozen_scope_and_expected_nonzero_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-REPORT-7892-VALIDATION: a nonzero exit needs its frozen reason."""
    scope = producer.freeze(tmp_path, tmp_path / "raw")
    names = {item["name"] for item in scope["commands"]}
    assert {
        "coverage_unit",
        "coverage_cli_success",
        "coverage_cli_failure",
        "coverage_cold_replay",
        "coverage_historical_cli",
        "cold_feature_replay_0",
        "cold_feature_replay_1",
        "e2e_015",
        "e2e_016_fixture",
        "e2e_016_replay",
    } <= names
    assert all(
        "--date" in item["argv"] for item in scope["commands"] if item["name"].startswith("e2e_016")
    )
    log = tmp_path / "child.log"
    log.write_text("ValueError: wrong_reason\n")
    monkeypatch.setattr(
        producer,
        "run_commands",
        lambda *_a, **_k: [
            {"log_path": str(log), "exit_code": 1, "timed_out": False, "passed": False}
        ],
    )
    spec = {
        "name": "negative",
        "argv": ["false"],
        "classification": "required",
        "timeout_s": 5,
        "expected_exit": "nonzero",
        "expected_reason": "FileNotFoundError",
    }
    assert not producer.child(spec, 0, tmp_path / "raw", tmp_path, 0.0)["passed"]
    log.write_text("FileNotFoundError: missing\n")
    receipt = producer.child(spec, 1, tmp_path / "raw", tmp_path, 0.0)
    assert receipt["passed"]
    Path(receipt["log_path"]).write_text("changed sealed bytes")
    with pytest.raises(ValueError, match="sealed_log_collision"):
        producer.child(spec, 1, tmp_path / "raw", tmp_path, 0.0)


def test_custody_rejects_import_alias(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """SCENARIO-REPORT-7892-CUSTODY: a nonqualified import fails before shards."""
    rows, public = _families()
    monkeypatch.setattr(producer.acquisition, "acquire", lambda *_: (rows, public))
    monkeypatch.setattr(producer.acquisition, "qualified_imports", lambda: {"bad": "alias"})
    with pytest.raises(ValueError, match="resolved_imports_invalid"):
        producer.custody(producer.base(0.0, [], []), {}, {}, tmp_path / "raw", 0.0)


def test_validation_failure_and_terminal_recheck(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-REPORT-7892-TERMINAL: failed required and flagged bytes are rechecked."""
    artifact = producer.base(0.0, [], [])
    scope = producer.freeze(tmp_path, tmp_path / "raw")
    owned = [str((producer.ROOT / path).resolve()) for path in scope["owned_files"]]
    for name in ("unit", "success", "failure", "replay", "historical"):
        data = CoverageData(basename=str(tmp_path / f"{name}.coverage"))
        data.add_lines({owned[0]: {1}})
        data.write()
    calls = []

    def fake_child(spec: dict, index: int, raw: Path, private: Path, start: float) -> dict:
        calls.append(spec["name"])
        if spec["name"] == "coverage_report":
            (private / "coverage.json").write_text(
                json.dumps(
                    {
                        "files": {
                            name: {"summary": {"num_statements": 1, "missing_lines": 0}}
                            for name in scope["owned_files"]
                        }
                    }
                )
            )
        return {
            "name": spec["name"],
            "passed": spec["name"] != "ruff_check",
            "exit_code": 1 if spec["name"] == "ruff_check" else 0,
            "timed_out": False,
            "log_path": str(tmp_path / "fake.log"),
            "output_tail": '{"flagged_count": 0}',
        }

    monkeypatch.setattr(producer, "child", fake_child)
    producer.validate(artifact, scope, tmp_path / "raw", tmp_path, 0.0)
    assert artifact["gate_check_summary"]
    assert "coverage_combine" not in calls
    artifact["flagged_adversarial"] = False
    artifact["rows"] = [{"family_id": "one", "status": "completed", "role": "fit"}]
    checks = iter([{"flagged_count": 1}, {"flagged_count": 0}])

    def terminal_child(spec: dict, index: int, raw: Path, private: Path, start: float) -> dict:
        return {
            "name": spec["name"],
            "passed": True,
            "exit_code": 0,
            "log_path": str(tmp_path / "fake.log"),
            "output_tail": json.dumps(next(checks)) if "adversarial" in spec["name"] else "",
        }

    monkeypatch.setattr(producer, "child", terminal_child)
    result = producer.terminal(artifact, tmp_path / "result.json", tmp_path / "raw", tmp_path, 0.0)
    assert result["verdict_class"] == "disqualified"
    assert result["source_boundary_ready_score"] == 0
    assert (
        producer.sha256_file(tmp_path / "result.json")
        == json.loads((tmp_path / "raw/terminal_validation_receipts.json").read_text())[
            "candidate_sha256"
        ]
    )


def test_dated_private_orchestration(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """SCENARIO-REPORT-7892-VALIDATION: dated paths reach each owned phase."""
    history = tmp_path / "history.json"
    history.write_text("{}")
    monkeypatch.setattr(producer.prior, "OUTPUT", history)
    monkeypatch.setattr(producer, "terminal", lambda artifact, *_: artifact)
    with pytest.raises(ValueError, match="run_date_mismatch"):
        producer.run_experiment("wrong", tmp_path / "out.json", tmp_path / "raw")
    monkeypatch.setattr(
        producer.prior,
        "preflight",
        lambda _start: (
            None,
            [],
            [producer.operand("exp7810", tmp_path / "missing", "sha256", "present", None)],
            {},
        ),
    )
    blocked = producer.run_experiment("20260929", tmp_path / "out.json", tmp_path / "raw")
    assert blocked["inference_substrate_class"] == "blocked_no_run"
    history.unlink()
    absent = producer.run_experiment("20260929", tmp_path / "out.json", tmp_path / "raw")
    assert absent["gate_check_summary"][-1]["upstream_id"] == "exp7880"
    history.write_text("{}")
    monkeypatch.setattr(
        producer.prior, "preflight", lambda _start: ({"schema": "fixture"}, [], [], {})
    )
    monkeypatch.setattr(
        producer, "custody", lambda *_: (_ for _ in ()).throw(ValueError("family_count"))
    )
    blocked = producer.run_experiment("20260929", tmp_path / "out.json", tmp_path / "raw")
    assert blocked["gate_check_summary"][-1]["artifact_field"] == "source_family_count"
    monkeypatch.setattr(producer, "custody", lambda *_: None)
    monkeypatch.setattr(producer, "validate", lambda artifact, *_: artifact.update(validated=True))
    complete = producer.run_experiment("20260929", tmp_path / "out.json", tmp_path / "raw")
    assert complete["validated"]
    monkeypatch.setattr(producer, "run_experiment", lambda *_: complete)
    assert (
        producer.main(
            [
                "--date",
                "20260929",
                "--output",
                str(tmp_path / "out.json"),
                "--raw-root",
                str(tmp_path / "raw"),
            ]
        )
        == 0
    )
    with pytest.raises(SystemExit):
        producer.main([])


def test_flagged_candidate_cannot_be_ready(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """SCENARIO-REPORT-7892-TERMINAL: a flagged pass still closes readiness."""
    artifact = producer.base(0.0, [], [])
    artifact["rows"] = [{"family_id": "one", "role": "fit", "status": "completed"}]
    calls = []

    def flagged_child(spec: dict, index: int, raw: Path, private: Path, start: float) -> dict:
        calls.append(spec["name"])
        return {
            "name": spec["name"],
            "passed": True,
            "exit_code": 0,
            "log_path": str(tmp_path / "fake.log"),
            "output_tail": '{"flagged_count": 1}' if len(calls) == 1 else '{"flagged_count": 0}',
        }

    monkeypatch.setattr(producer, "child", flagged_child)
    result = producer.terminal(artifact, tmp_path / "out.json", tmp_path / "raw", tmp_path, 0.0)
    assert result["flagged_adversarial"]
    assert result["source_boundary_ready_score"] == 0


def test_blocked_and_rejected_terminal_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-REPORT-7892-TERMINAL: external block and final rejection remain terminal."""
    artifact = producer.base(0.0, [], [])
    artifact["inference_substrate_class"] = "blocked_no_run"

    def passing(spec: dict, index: int, raw: Path, private: Path, start: float) -> dict:
        return {
            "name": spec["name"],
            "passed": True,
            "exit_code": 0,
            "log_path": str(tmp_path / "fake.log"),
            "output_tail": '{"flagged_count": 0}',
        }

    monkeypatch.setattr(producer, "child", passing)
    blocked = producer.terminal(
        artifact, tmp_path / "blocked.json", tmp_path / "blocked_raw", tmp_path, 0.0
    )
    assert blocked["verdict_class"] == "blocked"
    calls = []

    def rejected(spec: dict, index: int, raw: Path, private: Path, start: float) -> dict:
        calls.append(spec["name"])
        return {
            "name": spec["name"],
            "passed": len(calls) not in (2, 4),
            "exit_code": 1 if len(calls) in (2, 4) else 0,
            "log_path": str(tmp_path / "fake.log"),
            "output_tail": "malformed",
        }

    monkeypatch.setattr(producer, "child", rejected)
    with pytest.raises(ValueError, match="final_candidate_verification_failed"):
        producer.terminal(
            producer.base(0.0, [], []),
            tmp_path / "rejected.json",
            tmp_path / "rejected_raw",
            tmp_path,
            0.0,
        )
    assert not (tmp_path / "rejected.json").exists()


def test_validation_combines_measured_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-REPORT-7892-VALIDATION: completed shards gate the full report."""
    artifact = producer.base(0.0, [], [])
    scope = producer.freeze(tmp_path, tmp_path / "raw")
    owned = [str((producer.ROOT / path).resolve()) for path in scope["owned_files"]]
    with pytest.raises(ValueError, match="coverage_missing"):
        producer.coverage_counts(tmp_path, scope)
    empty = CoverageData(basename=str(tmp_path / "unit.coverage"))
    empty.add_lines({str(tmp_path / "unrelated.py"): {1}})
    empty.write()
    with pytest.raises(ValueError, match="coverage_empty"):
        producer.coverage_counts(tmp_path, scope)
    for name in ("unit", "success", "failure", "replay", "historical"):
        data = CoverageData(basename=str(tmp_path / f"{name}.coverage"))
        data.add_lines({owned[0]: {1}})
        data.write()

    def fake_child(spec: dict, index: int, raw: Path, private: Path, start: float) -> dict:
        if spec["name"] == "coverage_report":
            (private / "coverage.json").write_text(
                json.dumps(
                    {
                        "files": {
                            name: {"summary": {"num_statements": 1, "missing_lines": 0}}
                            for name in scope["owned_files"]
                        }
                    }
                )
            )
        return {
            "name": spec["name"],
            "passed": True,
            "exit_code": 0,
            "timed_out": False,
            "log_path": str(tmp_path / "fake.log"),
        }

    monkeypatch.setattr(producer, "child", fake_child)
    producer.validate(artifact, scope, tmp_path / "raw", tmp_path, 0.0)
    assert not artifact["gate_check_summary"]
    assert artifact["coverage_statement_counts"]["combined"][owned[0]]["missing_lines"] == 0
    assert any(item["name"] == "coverage_report" for item in artifact["validation_receipts"])

    def failed_report(spec: dict, index: int, raw: Path, private: Path, start: float) -> dict:
        receipt = fake_child(spec, index, raw, private, start)
        if spec["name"] == "coverage_report":
            receipt["passed"] = False
            receipt["exit_code"] = 1
            (private / "coverage.json").write_text(json.dumps({"files": {}}))
        return receipt

    monkeypatch.setattr(producer, "child", failed_report)
    rejected = producer.base(0.0, [], [])
    producer.validate(rejected, scope, tmp_path / "raw", tmp_path, 0.0)
    assert {item["artifact_field"] for item in rejected["gate_check_summary"]} >= {
        "coverage_report",
        f"coverage:{scope['owned_files'][0]}",
    }

    def absent_report(spec: dict, index: int, raw: Path, private: Path, start: float) -> dict:
        return {
            "name": spec["name"],
            "passed": True,
            "exit_code": 0,
            "timed_out": False,
            "log_path": str(tmp_path / "fake.log"),
        }

    (tmp_path / "coverage.json").unlink()
    monkeypatch.setattr(producer, "child", absent_report)
    absent = producer.base(0.0, [], [])
    producer.validate(absent, scope, tmp_path / "raw", tmp_path, 0.0)
    assert absent["gate_check_summary"][-1]["artifact_field"] == "coverage_report"


def test_validation_rejects_missing_coverage_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-REPORT-7892-VALIDATION: exit zero without data cannot qualify."""
    scope = producer.freeze(tmp_path, tmp_path / "raw")
    artifact = producer.base(0.0, [], [])
    monkeypatch.setattr(
        producer,
        "child",
        lambda spec, *_: {
            "name": spec["name"],
            "passed": True,
            "exit_code": 0,
            "timed_out": False,
            "log_path": str(tmp_path / "fake.log"),
        },
    )
    producer.validate(artifact, scope, tmp_path / "raw", tmp_path, 0.0)
    assert artifact["gate_check_summary"][-1]["artifact_field"] == "coverage_statement_counts"
