"""Private regression cases for REQ-VERIFY-7930-V688."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.reporting.current_work_receipt import (
    atomic_json,
    sha256_file,
    build_current_work_receipt,
)
from carnot.verify import energy_fit_7930 as current


def qualification(tmp_path: Path) -> tuple[Path, Path]:
    """A byte-bound primary lets mutations test custody without changing history."""
    upstream = tmp_path / "source.json"
    atomic_json(upstream, {"experiment_id": 7892})
    dependency = tmp_path / "runtime.py"
    dependency.write_text("qualified bytes\n")
    primary = tmp_path / "runtime.json"
    sidecar = tmp_path / "runtime.validators.json"
    atomic_json(
        primary,
        {
            "experiment_id": 7916,
            "training_runtime_ready_score": 1,
            "flagged_adversarial": False,
            "verdict_class": "circular_positive",
            "honest_verdict": "complete_circular_positive_training_fixture",
            "terminal_validation_sidecar_path": str(sidecar),
            "training_dependency_hashes": {str(dependency): sha256_file(dependency)},
            "source_artifact_hashes": [
                {"role": "upstream", "path": str(upstream), "sha256": sha256_file(upstream)}
            ],
            "historical_required_failures": [{"name": "historical_failure"}],
            "current_work_receipt": build_current_work_receipt(
                run_id="private-qualification",
                owner_pid=1,
                events=[],
                inference_substrate="aggregation_from_upstream_artifacts",
                inference_substrate_details={},
                inference_substrate_class="no_model_load",
                execution_venue="host",
                started_monotonic_ns=0,
                ended_monotonic_ns=1,
                phase_spans=[],
            ),
        },
    )
    atomic_json(sidecar, {"candidate_sha256": sha256_file(primary), "receipts": [{"passed": True}]})
    return primary, upstream


def test_exact_primary_and_dependency(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7930-CUSTODY: newer attestations do not supply readiness."""
    primary, upstream = qualification(tmp_path)
    sources, dependencies, artifact = current.authenticate(primary, upstream)
    assert sources[0]["sha256"] == sha256_file(primary)
    assert dependencies and artifact["training_runtime_ready_score"] == 1
    value = json.loads(primary.read_text())
    value["training_runtime_ready_score"] = 0
    atomic_json(primary, value)
    with pytest.raises(current.custody.InputBlocked, match="training_runtime_ready_score"):
        current.authenticate(primary, upstream)


@pytest.mark.parametrize("mutation", ["missing", "sidecar", "dependency", "source", "terminal"])
def test_custody_mutations(tmp_path: Path, mutation: str) -> None:
    """SCENARIO-VERIFY-7930-CUSTODY: each failure keeps its exact external operand."""
    primary, upstream = qualification(tmp_path)
    value = json.loads(primary.read_text())
    if mutation == "missing":
        primary.unlink()
    elif mutation == "sidecar":
        atomic_json(Path(value["terminal_validation_sidecar_path"]), {"candidate_sha256": "stale"})
    elif mutation == "dependency":
        (tmp_path / "runtime.py").write_text("changed")
    elif mutation == "source":
        upstream.write_text("changed")
    else:
        value["honest_verdict"] = "partial_training"
        atomic_json(primary, value)
    with pytest.raises(current.custody.InputBlocked) as error:
        current.authenticate(primary, upstream)
    row = error.value.operands[0]
    assert row["upstream_id"] == "exp7916-training-qualification"
    assert {
        "artifact_path",
        "artifact_sha256",
        "artifact_field",
        "op",
        "expected",
        "observed",
    } <= row.keys()


def test_current_checkpoint_binding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-7930-FIT: complete identity prevents historical checkpoint reuse."""
    records = [{"id": "fit", "role": "fit"}, {"id": "tune", "role": "tune"}]
    monkeypatch.setattr(current, "ARMS", ("local_set",))
    monkeypatch.setattr(current, "SEEDS", (67801,))
    calls = []

    def fit(*args: object) -> dict[str, object]:
        calls.append(args)
        return {
            "parameter_count": 266,
            "curve": [{"epoch": n} for n in range(16)],
            "temperature": 1.0,
            "params": {"w": [0.0]},
        }

    monkeypatch.setattr(current.natural_training, "fit", fit)
    monkeypatch.setattr(current.training_runtime, "save", lambda p, h: atomic_json(p, h))
    manifest, path = current.fit_heads(records, tmp_path, "identity-one", 3000)
    assert len(manifest) == 1 and len(calls) == 1
    current.fit_heads(records, tmp_path, "identity-one", 3000)
    assert len(calls) == 1
    current.fit_heads(records, tmp_path, "identity-two", 3000)
    assert len(calls) == 2 and json.loads(path.read_text())["dependency_hash"] == "identity-two"
    with pytest.raises(TimeoutError):
        current.fit_heads(records, tmp_path, "identity-three", -1)


def test_primary_layout(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7930-TERMINAL: real readers ignore a newer nested sidecar."""
    value = {
        "experiment_id": 7930,
        "task_id": "exp7930-energy-fit",
        "honest_verdict": "complete_null_energy_fit",
        "verdict_class": "null",
        "flagged_adversarial": False,
        "energy_fit_ready_score": 1,
    }
    output = tmp_path / "experiment_7930_fixture.json"
    receipt = current.publication.publish_primary(output, value, lambda _: {"passed": True})
    Path(receipt["sidecar_path"]).touch()
    selected = current.publication.reader_receipt(
        "exp7930-energy-fit", tmp_path, field="energy_fit_ready_score"
    )
    assert selected["passed"] and selected["gate_sha256"] == sha256_file(output)


def test_cli_rejects_date() -> None:
    """SCENARIO-VERIFY-7930-TERMINAL: executing producer keeps the current date."""
    from scripts.experiments import experiment_7930_v688_energy_fit as cli

    with pytest.raises(SystemExit) as error:
        cli.main(["--date", "20260929"])
    assert error.value.code == 2


def public_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Small complete bytes test custody and exclusions without the development corpus."""
    roles = {"fit": 1, "tune": 1, "evaluation": 2}
    monkeypatch.setattr(current.prior, "ROLE_BUDGET", roles)
    public, evaluator, members = [], [], []
    for index, role in enumerate(("fit", "tune", "evaluation", "evaluation")):
        source = b"Lumen has 12. Orion is present."
        answer = b"Lumen has 13." if index < 3 else b"Lumen has 13. " * 17
        family = str(index)
        public.append(
            {"family_id": family, "source_bytes": source.hex(), "answer_bytes": answer.hex()}
        )
        evaluator.append(
            {
                "family_id": family,
                "role": role,
                "human_label": 1,
                "observed": 1,
                "annotation_byte_offsets": [[0, len(answer)]],
            }
        )
        members.append(
            {
                "family_id": family,
                "role": role,
                "source_cluster_id": current.evidence_views.digest(source),
                "exclusion_reasons": [] if index < 3 else ["answer_units_over_budget"],
            }
        )
    p, e, f = (tmp_path / name for name in ("public.jsonl", "evaluator.jsonl", "features.jsonl"))
    p.write_text("".join(json.dumps(r) + "\n" for r in public))
    e.write_text("".join(json.dumps(r) + "\n" for r in evaluator))
    f.write_text("{}\n")
    cohort = tmp_path / "cohort.json"
    atomic_json(cohort, {"rows": members})
    upstream = tmp_path / "upstream.json"
    atomic_json(
        upstream,
        {
            "experiment_id": 7892,
            "source_boundary_ready_score": 1,
            "flagged_adversarial": False,
            "verdict_class": "circular_positive",
            "honest_verdict": "complete_circular_positive_source_boundary",
            "cohort_manifest_path": str(cohort),
            "cohort_manifest_sha256": sha256_file(cohort),
            "public_shards": [{"path": str(p), "sha256": sha256_file(p)}],
            "evaluator_shards": [{"path": str(e), "sha256": sha256_file(e)}],
            "feature_shards": [{"path": str(f), "sha256": sha256_file(f)}],
        },
    )
    return upstream


def test_public_exclusions_and_label_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-7930-CUSTODY: labels outside fit/tune stay unopened before fitting."""
    upstream = public_fixture(tmp_path, monkeypatch)
    records, excluded, sources = current.public_records(upstream)
    assert len(records) == 3 and len(excluded) == 1 and sources
    assert all("label" not in row and "known" not in row for row in records)
    rows = current.attach_labels(records, upstream, {"fit", "tune"})
    assert len(rows) == 2 and all(row["known"] == [0] for row in rows)
    artifact = json.loads(upstream.read_text())
    e = Path(artifact["evaluator_shards"][0]["path"])
    labels = [json.loads(line) for line in e.read_text().splitlines()]
    labels[2].pop("human_label")
    e.write_text("".join(json.dumps(r) + "\n" for r in labels))
    artifact["evaluator_shards"][0]["sha256"] = sha256_file(e)
    atomic_json(upstream, artifact)
    assert len(current.attach_labels(records, upstream, {"fit", "tune"})) == 2
    with pytest.raises(current.custody.InputBlocked):
        current.attach_labels(records, upstream, {"evaluation"})


@pytest.mark.parametrize(
    "mutation",
    ["gate", "terminal", "cohort", "columns", "exclusions", "roles", "shard", "missing_cohort"],
)
def test_public_custody_mutations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    """SCENARIO-VERIFY-7930-CUSTODY: source qualification cannot hide changed bytes or roles."""
    upstream = public_fixture(tmp_path, monkeypatch)
    a = json.loads(upstream.read_text())
    cohort = Path(a["cohort_manifest_path"])
    if mutation == "gate":
        a["source_boundary_ready_score"] = 0
    elif mutation == "terminal":
        a["verdict_class"] = "blocked"
    elif mutation == "cohort":
        cohort.write_text("{}")
    elif mutation == "missing_cohort":
        cohort.unlink()
    elif mutation in ("exclusions", "roles"):
        value = json.loads(cohort.read_text())
        value["rows"][0]["exclusion_reasons" if mutation == "exclusions" else "role"] = (
            ["changed"] if mutation == "exclusions" else "evaluation"
        )
        atomic_json(cohort, value)
        a["cohort_manifest_sha256"] = sha256_file(cohort)
    elif mutation == "columns":
        p = Path(a["public_shards"][0]["path"])
        rows = [json.loads(line) for line in p.read_text().splitlines()]
        rows[0]["human_label"] = 1
        p.write_text("".join(json.dumps(r) + "\n" for r in rows))
        a["public_shards"][0]["sha256"] = sha256_file(p)
    else:
        Path(a["evaluator_shards"][0]["path"]).write_text("changed")
    atomic_json(upstream, a)
    with pytest.raises(current.custody.InputBlocked):
        current.public_records(upstream)


def test_malformed_primary_is_external_block(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7930-CUSTODY: invalid external JSON is not an owned failure."""
    primary, upstream = qualification(tmp_path)
    primary.write_text("invalid JSON")
    with pytest.raises(current.custody.InputBlocked, match="valid_json"):
        current.authenticate(primary, upstream)


def test_qualified_runtime_receipt_drift(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7930-CUSTODY: the primary runtime ledger must still reduce correctly."""
    primary, upstream = qualification(tmp_path)
    value = json.loads(primary.read_text())
    value["current_work_receipt"]["event_count"] = 5
    atomic_json(primary, value)
    atomic_json(
        Path(value["terminal_validation_sidecar_path"]),
        {"candidate_sha256": sha256_file(primary), "receipts": [{"passed": True}]},
    )
    with pytest.raises(current.custody.InputBlocked, match="runtime_receipt"):
        current.authenticate(primary, upstream)


def test_fit_heartbeat_and_invalid_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-7930-FIT: progress persists during compute and invalid heads never seal."""
    from types import SimpleNamespace

    class Event:
        def __init__(self) -> None:
            self.calls = 0

        def wait(self, seconds: float) -> bool:
            self.calls += 1
            return self.calls > 1

        def set(self) -> None:
            pass

    class Thread:
        def __init__(self, target: object, **kwargs: object) -> None:
            self.target = target

        def start(self) -> None:
            self.target()

        def join(self, **kwargs: object) -> None:
            pass

    monkeypatch.setattr(current, "threading", SimpleNamespace(Event=Event, Thread=Thread))
    monkeypatch.setattr(
        current.natural_training, "fit", lambda *a: {"parameter_count": 5000, "curve": []}
    )
    with pytest.raises(ValueError, match="head budget"):
        current.fit_heads([], tmp_path, "current", 3000)


def test_blocked_replay_drift(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7930-TERMINAL: blocked replay retains named operands and zero readiness."""
    path = tmp_path / "blocked.json"
    atomic_json(
        path,
        {
            "verdict_class": "blocked",
            "energy_fit_ready_score": 0,
            "gate_check_summary": ["missing"],
        },
    )
    current.replay(path)
    atomic_json(
        path, {"verdict_class": "blocked", "energy_fit_ready_score": 1, "gate_check_summary": []}
    )
    with pytest.raises(ValueError, match="blocked gate drift"):
        current.replay(path)


def test_cold_reconstruction_rejects_changed_probabilities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-7930-TERMINAL: stored probabilities must reproduce from public bytes."""
    upstream = public_fixture(tmp_path, monkeypatch)
    public, _, sources = current.public_records(upstream)
    records = current.attach_labels(public, upstream, {"fit", "tune"})
    head = current.natural_training.fit([records[0]], [records[1]], "local_set", 67801, 0.01, 1)
    checkpoint = tmp_path / "head.json"
    current.training_runtime.save(checkpoint, head)
    spec = {
        "arm": "local_set",
        "seed": 67801,
        "path": str(checkpoint),
        "sha256": sha256_file(checkpoint),
    }
    manifest = tmp_path / "heads.json"
    atomic_json(manifest, {"checkpoints": [spec]})
    q = current.library(upstream, tmp_path / "candidate.json")
    prediction, _ = q.score(records, [spec], tmp_path)
    rows = [json.loads(line) for line in prediction.read_text().splitlines()]
    value = {
        "verdict_class": "null",
        "upstream_path": str(upstream),
        "source_artifact_hashes": sources,
        "rows": rows,
        "prediction_rows_path": str(prediction),
        "prediction_rows_sha256": sha256_file(prediction),
        "checkpoint_manifest_path": str(manifest),
    }
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, value)
    current.replay(candidate)
    value["source_artifact_hashes"] = [{"path": str(upstream), "sha256": "stale"}]
    atomic_json(candidate, value)
    with pytest.raises(ValueError, match="source hash drift"):
        current.replay(candidate)
    value["source_artifact_hashes"] = sources
    rows[0]["probability"] = 0.01
    rows[0]["loss"] = q._loss(0.01, rows[0]["label"])
    prediction.write_text("".join(json.dumps(row) + "\n" for row in rows))
    value["prediction_rows_sha256"] = sha256_file(prediction)
    atomic_json(candidate, value)
    with pytest.raises(ValueError, match="cold probability drift"):
        current.replay(candidate)


def test_cli_ready_and_cold_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-7930-TERMINAL: readiness assertion and cold mismatch produce real exits."""
    from scripts.experiments import experiment_7930_v688_energy_fit as cli

    output = tmp_path / "experiment_7930_private.json"

    def produce(*args: object) -> int:
        atomic_json(output, {"energy_fit_ready_score": 1})
        return 0

    monkeypatch.setattr(cli.run, "produce", produce)
    assert cli.main(["--date", "20260930", "--output", str(output), "--assert-ready"]) == 0
    monkeypatch.setattr(cli.core, "replay", lambda *a: None)
    assert cli.main(["--date", "20260930", "--cold-replay", str(output)]) == 0

    def failure(*args: object) -> None:
        raise ValueError("changed")

    monkeypatch.setattr(cli.core, "replay", failure)
    assert cli.main(["--date", "20260930", "--cold-replay", str(output)]) == 2
