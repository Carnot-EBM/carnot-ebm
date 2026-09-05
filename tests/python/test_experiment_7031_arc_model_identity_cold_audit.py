"""Tests for REQ-ARC-7031 and its independent identity mutation audit."""

from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from carnot import experiment_7031_arc_model_identity_cold_audit as experiment
from carnot.agentic import arc_eval_provenance as provenance
from carnot.agentic.arc_eval_provenance import build_arc_model_identity_receipt


ROOT = Path(__file__).resolve().parents[2]
HF_ID = "audit-owner/audit-model-GGUF"
REVISION = "7031freshrevision"
FILENAME = "fresh-audit-model.gguf"


def _direct_snapshot_fixture(tmp_path: Path) -> tuple[dict[str, str], Path]:
    """Create new bytes that do not depend on the earlier experiment fixtures."""

    requested = (
        tmp_path
        / "hub"
        / "models--audit-owner--audit-model-GGUF"
        / "snapshots"
        / REVISION
        / FILENAME
    )
    requested.parent.mkdir(parents=True)
    payload = b"direct regular GGUF fixture for the cold audit\n"
    requested.write_bytes(payload)
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    return (
        {
            "model_path": str(requested),
            "model_filename": FILENAME,
            "hf_id": HF_ID,
            "revision": REVISION,
            "model_file_hash": digest,
        },
        requested,
    )


@pytest.fixture(scope="module")
def ready_artifact(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """Run the required isolated process once for all positive artifact checks."""

    base = tmp_path_factory.mktemp("exp7031-ready")
    return experiment.build_artifact(
        ROOT,
        execution_date="20260905",
        output_path=base / "result.json",
        work_dir=base / "work",
    )


def test_req_arc_7031_spec_exists_before_implementation() -> None:
    """REQ-ARC-7031 names the cold process, mutation, and regression gates."""

    text = (ROOT / "openspec/capabilities/arc-agi/spec.md").read_text(encoding="utf-8")
    assert "REQ-ARC-7031" in text
    for scenario in (
        "SCENARIO-ARC-7031-COLD-SNAPSHOT-ROUND-TRIP",
        "SCENARIO-ARC-7031-ONE-FACTOR-MUTATIONS",
        "SCENARIO-ARC-7031-DIRECT-AND-LEGACY-REGRESSIONS",
        "SCENARIO-ARC-7031-UPSTREAM-DRIFT-BLOCKS",
        "SCENARIO-ARC-7031-LIVE-CONSUMER-REACHABILITY",
    ):
        assert scenario in text


def test_direct_regular_snapshot_gguf_remains_accepted(tmp_path: Path) -> None:
    """SCENARIO-ARC-7031-DIRECT-AND-LEGACY-REGRESSIONS keeps direct files valid."""

    spec, requested = _direct_snapshot_fixture(tmp_path)
    receipt = build_arc_model_identity_receipt(
        selected_model_spec=spec,
        observed_server_model_path=str(requested),
    )

    assert receipt["requested_model_path"] == str(requested)
    assert receipt["observed_server_model_path"] == str(requested)
    assert receipt["resolved_model_path"] == str(requested)
    assert receipt["model_file_hash"] == spec["model_file_hash"]


def test_shared_identity_bridge_defensive_edges_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-7031 covers malformed facts and filesystem failures in the shared bridge."""

    spec, requested = _direct_snapshot_fixture(tmp_path / "base")
    assert provenance.huggingface_snapshot_revision("relative.gguf", HF_ID) is None
    with pytest.raises(TypeError, match="selected_model_spec"):
        build_arc_model_identity_receipt(  # type: ignore[arg-type]
            selected_model_spec=[], observed_server_model_path=str(requested)
        )

    invalid_specs = (
        ({**spec, "model_path": "relative.gguf"}, "requested_model_path must be absolute"),
        ({**spec, "model_filename": "nested/model.bin"}, "requested_model_filename"),
        ({**spec, "hf_id": "not-a-repository"}, "requested_hf_id"),
        ({**spec, "revision": "nested/revision"}, "requested_revision"),
        ({**spec, "model_file_hash": "not-a-hash"}, "model_file_hash"),
    )
    for invalid_spec, expected in invalid_specs:
        with pytest.raises(ValueError, match=expected):
            build_arc_model_identity_receipt(
                selected_model_spec=invalid_spec,
                observed_server_model_path=str(requested),
            )

    with pytest.raises(ValueError, match="observed_server_model_path is missing"):
        build_arc_model_identity_receipt(
            selected_model_spec=spec,
            observed_server_model_path=str(tmp_path / "missing.gguf"),
        )

    original_resolve = Path.resolve
    with monkeypatch.context() as patch:

        def fail_resolution(path: Path, *args: object, **kwargs: object) -> Path:
            if path == requested:
                raise OSError("resolution denied")
            return original_resolve(path, *args, **kwargs)

        patch.setattr(Path, "resolve", fail_resolution)
        with pytest.raises(ValueError, match="path resolution failed"):
            build_arc_model_identity_receipt(
                selected_model_spec=spec,
                observed_server_model_path=str(requested),
            )

    original_is_symlink = Path.is_symlink
    original_stat = Path.stat
    identity_stat_started = False
    with monkeypatch.context() as patch:

        def track_identity_stat(path: Path) -> bool:
            nonlocal identity_stat_started
            result = original_is_symlink(path)
            if path == requested:
                identity_stat_started = True
            return result

        def fail_identity_stat(path: Path, *args: object, **kwargs: object) -> os.stat_result:
            if path == requested and identity_stat_started:
                raise OSError("metadata denied")
            return original_stat(path, *args, **kwargs)

        patch.setattr(Path, "is_symlink", track_identity_stat)
        patch.setattr(Path, "stat", fail_identity_stat)
        with pytest.raises(ValueError, match="path metadata failed"):
            build_arc_model_identity_receipt(
                selected_model_spec=spec,
                observed_server_model_path=str(requested),
            )

    other = requested.with_name("other.gguf")
    other.write_bytes(requested.read_bytes())
    with pytest.raises(ValueError, match="direct snapshot GGUF"):
        build_arc_model_identity_receipt(
            selected_model_spec=spec,
            observed_server_model_path=str(other),
        )

    payload = b"content whose digest is not the blob basename\n"
    model_root = tmp_path / "bad-blob" / "hub" / "models--audit-owner--audit-model-GGUF"
    blob = model_root / "blobs" / ("0" * 64)
    blob.parent.mkdir(parents=True)
    blob.write_bytes(payload)
    linked = model_root / "snapshots" / REVISION / FILENAME
    linked.parent.mkdir(parents=True)
    linked.symlink_to(Path("../../blobs") / blob.name)
    bad_blob_spec = {
        "model_path": str(linked),
        "model_filename": FILENAME,
        "hf_id": HF_ID,
        "revision": REVISION,
        "model_file_hash": "sha256:" + hashlib.sha256(payload).hexdigest(),
    }
    with pytest.raises(ValueError, match="resolved blob basename"):
        build_arc_model_identity_receipt(
            selected_model_spec=bad_blob_spec,
            observed_server_model_path=str(blob),
        )


def test_current_and_legacy_schema_defensive_edges_are_covered(tmp_path: Path) -> None:
    """REQ-ARC-7031 rejects v2 contradictions without changing valid v1 semantics."""

    class OfflinePolicy:
        proposer = None

    no_llm = provenance.build_arc_eval_provenance_for_policy(
        OfflinePolicy(),
        counters_before={},
        counters_after={},
        envelope={},
        solve_provenance="development_proxy",
        factory_path=Path(__file__),
        repo_root=ROOT,
    )
    no_llm["schema_version"] = provenance.ARC_EVAL_PROVENANCE_SCHEMA_VERSION_V2
    no_llm.update(dict.fromkeys(provenance.ARC_MODEL_IDENTITY_KEYS, provenance.NOT_APPLICABLE))
    no_llm["provenance_hash"] = provenance.compute_arc_eval_provenance_hash(no_llm)
    assert provenance.validate_arc_eval_provenance(no_llm).valid

    invalid_no_llm = dict(no_llm, requested_model_path="/tmp/claimed.gguf")
    invalid_no_llm["provenance_hash"] = provenance.compute_arc_eval_provenance_hash(invalid_no_llm)
    assert any(
        "no-LLM identity field" in error
        for error in provenance.validate_arc_eval_provenance(invalid_no_llm).errors
    )

    spec, requested = _direct_snapshot_fixture(tmp_path / "current")
    identity = build_arc_model_identity_receipt(
        selected_model_spec=spec,
        observed_server_model_path=str(requested),
    )
    current = provenance.build_arc_eval_provenance(
        replace(
            experiment._legacy_input(
                provenance,
                model_repository=HF_ID,
                model_filename=identity["requested_model_filename"],
                model_hash=identity["model_file_hash"],
            ),
            schema_version=provenance.ARC_EVAL_PROVENANCE_SCHEMA_VERSION_V2,
            **identity,
        )
    )
    contradictions = (
        ("resolved_model_path", str(tmp_path / "alias.gguf"), "shared model identity receipt"),
        ("model_repository", "other/repository", "model_repository contradicts"),
        ("model_filename", "other.gguf", "model_filename contradicts"),
        ("model_hash", "sha256:" + "0" * 64, "model_hash contradicts"),
    )
    for field, value, expected in contradictions:
        mutated = dict(current, **{field: value})
        mutated["provenance_hash"] = provenance.compute_arc_eval_provenance_hash(mutated)
        assert any(
            expected in error for error in provenance.validate_arc_eval_provenance(mutated).errors
        )

    legacy_with_identity = experiment._legacy_input(
        provenance,
        requested_model_path=str(requested),
    )
    with pytest.raises(ValueError, match="legacy provenance input"):
        provenance.build_arc_eval_provenance(legacy_with_identity)


def test_policy_builder_validates_supplied_and_observed_identity(
    tmp_path: Path,
) -> None:
    """REQ-ARC-7031 covers both production routes through the shared identity builder."""

    spec, requested = _direct_snapshot_fixture(tmp_path / "identity")
    identity = build_arc_model_identity_receipt(
        selected_model_spec=spec,
        observed_server_model_path=str(requested),
    )
    server = tmp_path / "llama-server"
    server.write_bytes(b"server")

    class Proposer:
        model_path = str(requested)
        model_filename = FILENAME
        model_repository = HF_ID
        n_ctx = 4096
        observed_server_n_ctx = 4096
        port = 17031
        generator_server_path = str(server)
        last_launch_argv = (str(server), "-m", str(requested), "--port", "17031")

        @staticmethod
        def _url() -> str:
            return "http://127.0.0.1:17031"

    class Policy:
        proposer = Proposer()

    call = {
        "counters_before": {"requests": 0, "completions": 0, "errors": 0},
        "counters_after": {"requests": 1, "completions": 1, "errors": 0},
        "envelope": {
            "generator_cuda_gpu_requested": "0",
            "gpus_held": [
                {
                    "index": 0,
                    "gpu_uuid": "GPU-70310000-0000-0000-0000-000000000001",
                    "gpu_model": "NVIDIA RTX 3090",
                }
            ],
        },
        "solve_provenance": "live_agent_self_discovery",
        "factory_path": Path(__file__),
        "repo_root": ROOT,
        "lease": {
            "lease_id": "lease-7031",
            "lease_hash": "sha256:" + "4" * 64,
            "lease_issued_at": "2026-09-05T00:00:00+00:00",
            "lease_expires_at": "2026-09-05T02:00:00+00:00",
        },
        "lease_checked_at": "2026-09-05T01:00:00+00:00",
    }
    supplied = provenance.build_arc_eval_provenance_for_policy(
        Policy(), model_identity_receipt=identity, **call
    )
    assert supplied["schema_version"] == provenance.ARC_EVAL_PROVENANCE_SCHEMA_VERSION_V2

    contradictory = dict(identity, resolved_model_path=str(tmp_path / "alias.gguf"))
    with pytest.raises(ValueError, match="model_identity_receipt contradicts"):
        provenance.build_arc_eval_provenance_for_policy(
            Policy(), model_identity_receipt=contradictory, **call
        )

    class ObservedProposer(Proposer):
        requested_model_path = str(requested)
        requested_model_filename = None
        model_revision = REVISION
        observed_server_model_path = None

        @staticmethod
        def observed_model_path() -> str:
            return str(requested)

    class ObservedPolicy:
        proposer = ObservedProposer()

    observed = provenance.build_arc_eval_provenance_for_policy(ObservedPolicy(), **call)
    assert observed["requested_model_filename"] == FILENAME
    assert observed["schema_version"] == provenance.ARC_EVAL_PROVENANCE_SCHEMA_VERSION_V2

    class UnreadableObservationProposer(Proposer):
        observed_server_model_path = None

        @staticmethod
        def observed_model_path() -> str:
            raise RuntimeError("observation unavailable")

    class UnreadableObservationPolicy:
        proposer = UnreadableObservationProposer()

    legacy = provenance.build_arc_eval_provenance_for_policy(UnreadableObservationPolicy(), **call)
    assert legacy["schema_version"] == provenance.ARC_EVAL_PROVENANCE_SCHEMA_VERSION


def test_cold_process_accepts_positives_and_rejects_every_mutation(
    ready_artifact: dict[str, object],
) -> None:
    """SCENARIO-ARC-7031-ONE-FACTOR-MUTATIONS records each exact rejection."""

    assert experiment.validate_artifact(ready_artifact) == []
    assert ready_artifact["arc_model_identity_audit_ready_score"] == 1
    assert ready_artifact["verdict_class"] == "positive"
    assert str(ready_artifact["honest_verdict"]).startswith("complete_positive_")
    assert all(row["accepted"] for row in ready_artifact["positive_identity_rows"])
    assert all(row["accepted"] for row in ready_artifact["legacy_regression_rows"])

    groups = {
        "alias_mutation_rows": {"requested_filename", "observed_path"},
        "hash_mutation_rows": {"content_hash", "same_size_different_bytes"},
        "hub_revision_mutation_rows": {"repository", "revision"},
        "path_type_rows": {"broken_link", "path_type", "ambiguous_hard_link"},
        "stale_server_rows": {"stale_server_identity"},
    }
    observed_cases: set[str] = set()
    for field, expected_cases in groups.items():
        rows = ready_artifact[field]
        assert {row["case"] for row in rows} == expected_cases
        assert all(row["accepted"] is False for row in rows)
        assert all(
            str(row["reject_reason"]).startswith("invalid ARC model identity:") for row in rows
        )
        assert all(row["changed_factor"] == row["case"] for row in rows)
        observed_cases.update(row["case"] for row in rows)
    assert observed_cases == {
        "content_hash",
        "repository",
        "revision",
        "requested_filename",
        "observed_path",
        "broken_link",
        "path_type",
        "stale_server_identity",
        "same_size_different_bytes",
        "ambiguous_hard_link",
    }


def test_full_receipt_round_trips_in_a_fresh_restricted_process(
    ready_artifact: dict[str, object],
) -> None:
    """SCENARIO-ARC-7031-COLD-SNAPSHOT-ROUND-TRIP proves process isolation."""

    process = ready_artifact["fresh_process_rows"][0]
    assert process["pid"] != process["parent_pid"]
    assert process["isolated_python"] is True
    assert process["private_work_dir"] is True
    assert process["minimal_environment"] is True, json.dumps(process, sort_keys=True)
    assert process["fixture_origin"] == "generated_by_exp7031_worker"
    assert process["round_trip_valid"] is True
    assert process["returncode"] == 0
    assert ready_artifact["command_receipt_rows"][0]["returncode"] == 0
    assert "-I" in ready_artifact["command_receipt_rows"][0]["argv"]


def test_live_belief_consumer_reaches_only_the_shared_validator(
    ready_artifact: dict[str, object],
) -> None:
    """SCENARIO-ARC-7031-LIVE-CONSUMER-REACHABILITY rejects a local copy."""

    rows = ready_artifact["consumer_reachability_rows"]
    assert {row["symbol"] for row in rows} == {
        "build_arc_model_identity_receipt",
        "validate_arc_evaluation_row",
        "validate_arc_eval_provenance",
    }
    assert all(row["shared_function_identity"] is True for row in rows)
    assert all(row["shared_source_file"] is True for row in rows)
    assert all(row["local_copy_present"] is False for row in rows)


def test_exp7030_artifact_and_every_cited_source_hash_are_recomputed(
    ready_artifact: dict[str, object],
) -> None:
    """REQ-ARC-7031 binds its decision to the complete upstream hash set."""

    upstream = json.loads((ROOT / experiment.UPSTREAM_RELATIVE_PATH).read_text(encoding="utf-8"))
    hashes = ready_artifact["source_artifact_hashes"]
    assert hashes[experiment.UPSTREAM_RELATIVE_PATH.as_posix()] == experiment.sha256_file(
        ROOT / experiment.UPSTREAM_RELATIVE_PATH
    )
    for relative, expected in upstream["source_artifact_hashes"].items():
        assert hashes[relative] == expected == experiment.sha256_file(ROOT / relative)
    assert ready_artifact["upstream_gate_rows"][0]["observed_value"] == 1
    assert (
        ready_artifact["cited_upstream_artifacts"][0]["artifact_hash"]
        == hashes[experiment.UPSTREAM_RELATIVE_PATH.as_posix()]
    )


def test_upstream_source_drift_is_a_terminal_block(tmp_path: Path) -> None:
    """SCENARIO-ARC-7031-UPSTREAM-DRIFT-BLOCKS keeps drift out of partial results."""

    upstream_path = ROOT / experiment.UPSTREAM_RELATIVE_PATH
    upstream = json.loads(upstream_path.read_text(encoding="utf-8"))
    fake_root = tmp_path / "fake-root"
    fake_upstream = fake_root / experiment.UPSTREAM_RELATIVE_PATH
    fake_upstream.parent.mkdir(parents=True)
    shutil.copy2(upstream_path, fake_upstream)
    for relative in upstream["source_artifact_hashes"]:
        target = fake_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    drifted_relative = next(iter(upstream["source_artifact_hashes"]))
    with (fake_root / drifted_relative).open("ab") as handle:
        handle.write(b"\n7031 drift fixture\n")

    artifact = experiment.build_artifact(
        fake_root,
        execution_date="20260905",
        output_path=fake_root / experiment.RESULT_RELATIVE_PATH,
        work_dir=fake_root / "private-work",
    )

    summary = artifact["gate_check_summary"]
    assert artifact["arc_model_identity_audit_ready_score"] == 0
    assert artifact["verdict_class"] == "blocked"
    assert str(artifact["honest_verdict"]).startswith("blocked_arc_model_identity_cold_audit:")
    assert summary["failed_check"] == f"exp7030_source_hash:{drifted_relative}"
    assert summary["expected_value"] == upstream["source_artifact_hashes"][drifted_relative]
    assert summary["observed_value"] == experiment.sha256_file(fake_root / drifted_relative)
    assert artifact["fresh_process_rows"] == []
    assert experiment.validate_artifact(artifact) == []

    forged_score = copy.deepcopy(artifact)
    forged_score["arc_model_identity_audit_ready_score"] = 1
    forged_score["reproducibility_checksum"] = experiment.artifact_checksum(forged_score)
    assert "blocked_ready_score_invalid" in experiment.validate_artifact(forged_score)

    forged_summary = copy.deepcopy(artifact)
    forged_summary["gate_check_summary"] = {}
    forged_summary["reproducibility_checksum"] = experiment.artifact_checksum(forged_summary)
    assert "blocked_gate_check_summary_invalid" in experiment.validate_artifact(forged_summary)

    forged_verdict = copy.deepcopy(artifact)
    forged_verdict["honest_verdict"] = "blocked_with_no_exact_gate"
    forged_verdict["reproducibility_checksum"] = experiment.artifact_checksum(forged_verdict)
    assert "blocked_verdict_invalid" in experiment.validate_artifact(forged_verdict)


def test_malformed_inputs_and_import_failures_remain_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-ARC-7031-UPSTREAM-DRIFT-BLOCKS also covers unreadable evidence."""

    missing = tmp_path / "missing.json"
    malformed = tmp_path / "malformed.json"
    nonobject = tmp_path / "nonobject.json"
    malformed.write_text("{", encoding="utf-8")
    nonobject.write_text("[]", encoding="utf-8")
    assert experiment._load_json(missing) == {}
    assert experiment._load_json(malformed) == {}
    assert experiment._load_json(nonobject) == {}

    monkeypatch.setattr(
        experiment.importlib,
        "import_module",
        lambda _name: (_ for _ in ()).throw(ImportError("cold import denied")),
    )
    work = tmp_path / "import-work"
    work.mkdir()
    output = tmp_path / "result.json"
    checks, _evidence, _hashes, _commands = experiment.collect_preconditions(ROOT, output, work)
    row = next(item for item in checks if item["check"] == "exp7030_artifact_valid")
    assert row["observed_value"] == ["ImportError: cold import denied"]


def test_malformed_worker_output_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-7031 does not treat an unparseable child receipt as audit evidence."""

    completed = subprocess.CompletedProcess(["python"], 1, stdout="not-json", stderr="failure")
    receipt = {"returncode": 1, "terminal": True}
    monkeypatch.setattr(
        experiment,
        "_isolated_command",
        lambda *_args, **_kwargs: (completed, receipt),
    )
    payload, observed_receipt = experiment._run_worker(tmp_path)
    assert payload == {}
    assert observed_receipt == receipt


def test_worker_cli_parser_dispatches_only_explicit_worker_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-7031 keeps the private worker protocol explicit."""

    monkeypatch.setattr(experiment, "_worker_main", lambda args: 17)
    assert experiment.main(["--worker", "--work-dir", str(tmp_path)]) == 17
    with pytest.raises(SystemExit):
        experiment.main(["--work-dir", str(tmp_path)])


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (lambda value: value.pop("rows"), "required_fields_missing"),
        (lambda value: value.__setitem__("field_principles", {}), "field_principles"),
        (lambda value: value.__setitem__("inference_substrate", "live_llm"), "inference_substrate"),
        (lambda value: value.__setitem__("duration_s", -1), "duration_s"),
        (lambda value: value.__setitem__("verifier_is_oracle", True), "verifier_is_oracle"),
        (
            lambda value: value.__setitem__("arc_model_identity_audit_ready_score", True),
            "ready_score",
        ),
        (lambda value: value["positive_identity_rows"].clear(), "positive_identity"),
        (lambda value: value["alias_mutation_rows"][0].__setitem__("accepted", True), "mutation"),
        (lambda value: value["legacy_regression_rows"].clear(), "legacy_regression"),
        (lambda value: value["fresh_process_rows"].clear(), "fresh_process"),
        (lambda value: value["consumer_reachability_rows"].clear(), "consumer_reachability"),
        (lambda value: value["command_receipt_rows"].clear(), "command_receipt"),
        (lambda value: value.__setitem__("gate_check_summary", {}), "gate_check_summary"),
        (lambda value: value.__setitem__("verdict_class", "partial"), "positive_verdict"),
        (lambda value: value.__setitem__("reproducibility_checksum", "bad"), "checksum"),
    ],
)
def test_artifact_validator_rejects_forged_positive_evidence(
    ready_artifact: dict[str, object], mutation, error: str
) -> None:
    """REQ-ARC-7031 recomputes readiness instead of trusting its score."""

    artifact = copy.deepcopy(ready_artifact)
    mutation(artifact)
    if error != "checksum":
        artifact["reproducibility_checksum"] = experiment.artifact_checksum(artifact)
    assert any(error in item for item in experiment.validate_artifact(artifact))


def test_worker_and_writer_defensive_paths(tmp_path: Path) -> None:
    """REQ-ARC-7031 keeps malformed worker output and artifacts from publication."""

    assert experiment.validate_artifact([]) == ["artifact_object_required"]
    with pytest.raises(ValueError, match="invalid Exp7031 artifact"):
        experiment.write_artifact(tmp_path / "bad.json", {})

    worker = experiment.execute_worker(tmp_path / "worker")
    assert worker["positive_identity_rows"]
    assert len(worker["alias_mutation_rows"]) == 2
    assert len(worker["hash_mutation_rows"]) == 2
    assert len(worker["hub_revision_mutation_rows"]) == 2
    assert len(worker["path_type_rows"]) == 3
    assert len(worker["stale_server_rows"]) == 1


def test_run_writes_one_stable_terminal_artifact(tmp_path: Path) -> None:
    """REQ-ARC-7031 publishes only after the cold audit and validation pass."""

    output = tmp_path / "exp7031.json"
    artifact = experiment.run(ROOT, execution_date="20260905", output_path=output)

    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert experiment.validate_artifact(artifact) == []
    assert artifact["arc_model_identity_audit_ready_score"] == 1
    assert os.stat(output).st_size > 0
