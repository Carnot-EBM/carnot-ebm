"""REQ-ARC-7010: forward ARC evaluation provenance fails closed."""

from __future__ import annotations

import argparse
import copy
import json
import runpy
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from carnot import experiment_7010_arc_eval_provenance_contract as experiment
from carnot.agentic import arc_eval_provenance as provenance
from carnot.agentic.arc_eval_provenance import (
    ARC_EVAL_PROVENANCE_REQUIRED_KEYS,
    ARC_EVAL_PROVENANCE_SCHEMA_VERSION,
    CONSUMER_REQUIRED_KEYS,
    NO_LLM_INFERENCE_SUBSTRATE,
    NOT_APPLICABLE,
    PRODUCER_REQUIRED_KEYS,
    SOLVE_PROVENANCE_VALUES,
    ArcEvalProvenanceInput,
    build_arc_eval_provenance,
    compute_arc_eval_provenance_hash,
    validate_arc_eval_provenance,
    validate_arc_evaluation_row,
)
from carnot.experiment_7010_arc_eval_provenance_contract import (
    REQUIRED_ARTIFACT_FIELDS,
    build_artifact,
    validate_artifact,
)

REPO = Path(__file__).resolve().parents[2]


def _digest(marker: str) -> str:
    return "sha256:" + marker * 64


def _live_input(**changes: object) -> ArcEvalProvenanceInput:
    source = ArcEvalProvenanceInput(
        inference_substrate="local_gguf_cuda",
        gpu_uuid="GPU-00000000-0000-0000-0000-000000000001",
        gpu_model="NVIDIA RTX 3090",
        cuda_device=1,
        model_repository="unsloth/Qwen3.8-27B-GGUF",
        model_filename="Qwen3.8-27B-Q4_K_M.gguf",
        model_hash=_digest("1"),
        n_ctx=98304,
        server_binary="/opt/llama.cpp/llama-server",
        server_binary_hash=_digest("2"),
        server_command_hash=_digest("3"),
        endpoint="http://127.0.0.1:8919",
        port=8919,
        lease_id="lease-7010",
        lease_hash=_digest("4"),
        lease_issued_at="2026-09-05T00:00:00+00:00",
        lease_expires_at="2026-09-05T02:00:00+00:00",
        lease_checked_at="2026-09-05T01:00:00+00:00",
        request_count=3,
        completion_count=2,
        error_count=1,
        policy_hash=_digest("5"),
        factory_hash=_digest("6"),
        git_commit="7" * 40,
        solve_provenance="live_agent_self_discovery",
    )
    return replace(source, **changes)


def _rehash(record: dict[str, object]) -> dict[str, object]:
    record["provenance_hash"] = compute_arc_eval_provenance_hash(record)
    return record


def test_live_fixture_round_trips_under_one_schema() -> None:
    """SCENARIO-ARC-7010-LIVE-ROUND-TRIP."""
    record = build_arc_eval_provenance(_live_input())
    decision = validate_arc_eval_provenance(json.loads(json.dumps(record)))

    assert decision.valid and decision.headline_eligible
    assert tuple(record) == ARC_EVAL_PROVENANCE_REQUIRED_KEYS
    assert PRODUCER_REQUIRED_KEYS is ARC_EVAL_PROVENANCE_REQUIRED_KEYS
    assert CONSUMER_REQUIRED_KEYS is ARC_EVAL_PROVENANCE_REQUIRED_KEYS
    assert record["schema_version"] == ARC_EVAL_PROVENANCE_SCHEMA_VERSION
    assert record["provenance_hash"] == compute_arc_eval_provenance_hash(record)


@pytest.mark.parametrize("mode", ["absent", "null", "malformed", "aliased", "extra"])
def test_shape_defects_fail_closed(mode: str) -> None:
    """SCENARIO-ARC-7010-REJECTION-MATRIX: no compatibility aliases."""
    record = build_arc_eval_provenance(_live_input())
    if mode == "absent":
        del record["gpu_uuid"]
    elif mode == "null":
        record["gpu_uuid"] = None
    elif mode == "malformed":
        record["gpu_uuid"] = "cuda:1"
    elif mode == "aliased":
        record["gpu_id"] = record.pop("gpu_uuid")
    else:
        record["model_path"] = "/tmp/model.gguf"
    _rehash(record)

    decision = validate_arc_eval_provenance(record)
    assert not decision.valid
    assert not decision.headline_eligible


@pytest.mark.parametrize(
    ("changes", "error_fragment"),
    [
        ({"completion_count": 1}, "counter"),
        ({"endpoint": "http://127.0.0.1:8920"}, "port"),
        ({"lease_checked_at": "2026-09-05T03:00:00+00:00"}, "lease"),
    ],
)
def test_partial_requests_port_reuse_and_stale_leases_reject(
    changes: dict[str, object], error_fragment: str
) -> None:
    """SCENARIO-ARC-7010-REJECTION-MATRIX: contradictions are ineligible."""
    record = build_arc_eval_provenance(_live_input())
    record.update(changes)
    _rehash(record)

    decision = validate_arc_eval_provenance(record)
    assert not decision.valid
    assert any(error_fragment in error for error in decision.errors)


def test_no_llm_row_is_explicit_and_cannot_claim_cuda_or_gguf() -> None:
    """SCENARIO-ARC-7010-NO-LLM-IS-EXPLICIT."""
    na = NOT_APPLICABLE
    no_llm = ArcEvalProvenanceInput(
        inference_substrate=NO_LLM_INFERENCE_SUBSTRATE,
        gpu_uuid=na,
        gpu_model=na,
        cuda_device=na,
        model_repository=na,
        model_filename=na,
        model_hash=na,
        n_ctx=na,
        server_binary=na,
        server_binary_hash=na,
        server_command_hash=na,
        endpoint=na,
        port=na,
        lease_id=na,
        lease_hash=na,
        lease_issued_at=na,
        lease_expires_at=na,
        lease_checked_at=na,
        request_count=0,
        completion_count=0,
        error_count=0,
        policy_hash=_digest("8"),
        factory_hash=_digest("9"),
        git_commit="a" * 40,
        solve_provenance="development_proxy",
    )
    accepted = build_arc_eval_provenance(no_llm)
    assert validate_arc_eval_provenance(accepted).valid

    for field, claim in (("cuda_device", 0), ("model_filename", "claimed.gguf")):
        mutated = copy.deepcopy(accepted)
        mutated[field] = claim
        _rehash(mutated)
        assert not validate_arc_eval_provenance(mutated).valid


@pytest.mark.parametrize("solve_provenance", sorted(SOLVE_PROVENANCE_VALUES))
def test_all_solve_provenance_values_are_accepted(solve_provenance: str) -> None:
    """SCENARIO-ARC-7010-SOLVE-PROVENANCE-ENUM."""
    record = build_arc_eval_provenance(_live_input(solve_provenance=solve_provenance))
    row = {"solve_provenance": solve_provenance, "arc_eval_provenance": record}
    assert validate_arc_evaluation_row(row).headline_eligible


def test_row_solve_provenance_is_required_and_must_match() -> None:
    """REQ-ARC-7010 requires solve provenance on every evaluation row."""
    record = build_arc_eval_provenance(_live_input())
    assert not validate_arc_evaluation_row({"arc_eval_provenance": record}).valid
    row = {"solve_provenance": "outer_loop_re", "arc_eval_provenance": record}
    assert not validate_arc_evaluation_row(row).valid


def test_provenance_hash_is_stable_in_a_fresh_process() -> None:
    """SCENARIO-ARC-7010-LIVE-ROUND-TRIP: canonical bytes cross processes."""
    record = build_arc_eval_provenance(_live_input())
    code = (
        "import json,sys; from carnot.agentic.arc_eval_provenance import "
        "compute_arc_eval_provenance_hash as h; print(h(json.loads(sys.stdin.read())))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO,
        input=json.dumps(record),
        text=True,
        capture_output=True,
        check=True,
    )
    assert completed.stdout.strip() == record["provenance_hash"]


def test_leaderboard_producer_and_dashboard_consumer_are_wired() -> None:
    """REQ-ARC-7010 keeps producer and headline consumer on the shared validator."""
    producer = (REPO / "scripts" / "arc_leaderboard_eval.py").read_text()
    consumer = (REPO / "scripts" / "outer_loop_dashboard.py").read_text()
    assert "build_arc_eval_provenance_for_policy" in producer
    assert '"solve_provenance": _solve_provenance' in producer
    assert "validate_arc_evaluation_row" in consumer
    assert '"arc_eval_provenance"' in consumer


def test_experiment_artifact_proves_every_required_field_rejects(tmp_path: Path) -> None:
    """REQ-ARC-7010 emits the deterministic acceptance and field rejection evidence."""
    artifact = build_artifact(REPO, execution_date="20260905")

    assert validate_artifact(artifact) == []
    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["arc_eval_provenance_contract_ready_score"] == 1
    assert artifact["historical_artifacts_modified"] is False
    assert artifact["inference_substrate"] == "deterministic_arc_provenance_contract_no_llm"
    assert {row["field"] for row in artifact["missing_field_rows"]} == set(
        ARC_EVAL_PROVENANCE_REQUIRED_KEYS
    )
    assert all(not row["accepted"] for row in artifact["rejected_fixture_rows"])
    assert all("solve_provenance" in row for row in artifact["rows"])
    assert artifact["honest_verdict"].startswith("positive_")


def test_policy_boundary_builds_live_and_no_llm_records(tmp_path: Path) -> None:
    """REQ-ARC-7010 producer observations pass through the typed builder."""

    class Proposer:
        n_completion_calls = 3
        n_completion_ok = 2
        model_repository = "owner/repository"
        model_filename = "model.gguf"
        n_ctx = 98304
        observed_server_n_ctx = 98304
        port = 8919
        generator_server_path = ""

        def __init__(self, model_path: Path, server_path: Path) -> None:
            self.model_path = str(model_path)
            self.last_launch_argv = (str(server_path), "-m", str(model_path), "--port", "8919")

        def _url(self) -> str:
            return "http://127.0.0.1:8919"

    class Policy:
        def __init__(self, proposer: object | None) -> None:
            self.proposer = proposer

    model = tmp_path / "model.gguf"
    server = tmp_path / "llama-server"
    model.write_bytes(b"model")
    server.write_bytes(b"server")
    policy = Policy(Proposer(model, server))
    lease = {
        "lease_id": "lease-test",
        "lease_hash": _digest("b"),
        "lease_issued_at": "2026-09-05T00:00:00+00:00",
        "lease_expires_at": "2026-09-05T02:00:00+00:00",
    }
    live = provenance.build_arc_eval_provenance_for_policy(
        policy,
        counters_before={"requests": 0, "completions": 0, "errors": 0},
        counters_after={"requests": 3, "completions": 2, "errors": 1},
        envelope={
            "generator_cuda_gpu_requested": "1",
            "gpus_held": [{"index": 1, "gpu_uuid": _live_input().gpu_uuid, "gpu_model": "RTX"}],
        },
        solve_provenance="live_agent_self_discovery",
        factory_path=Path(__file__),
        repo_root=REPO,
        lease=lease,
        lease_checked_at="2026-09-05T01:00:00+00:00",
    )
    assert validate_arc_eval_provenance(live).valid
    assert live["model_hash"] != live["server_binary_hash"]
    assert provenance.evaluation_counters(policy) == {
        "requests": 3,
        "completions": 2,
        "errors": 1,
    }

    no_llm = provenance.build_arc_eval_provenance_for_policy(
        Policy(None),
        counters_before={},
        counters_after={},
        envelope={},
        solve_provenance="development_proxy",
        factory_path=Path(__file__),
        repo_root=REPO,
    )
    assert no_llm["inference_substrate"] == NO_LLM_INFERENCE_SUBSTRATE
    assert no_llm["gpu_uuid"] == NOT_APPLICABLE


def test_validator_and_producer_defensive_paths_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-ARC-7010-REJECTION-MATRIX covers unreadable and invalid state."""
    assert not validate_arc_eval_provenance([]).valid
    assert not validate_arc_evaluation_row([]).valid
    with pytest.raises(TypeError):
        build_arc_eval_provenance({})  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        build_arc_eval_provenance(_live_input(gpu_uuid="bad"))

    broken = build_arc_eval_provenance(_live_input())
    broken["gpu_model"] = "different"
    assert any("provenance_hash" in error for error in validate_arc_eval_provenance(broken).errors)
    broken = build_arc_eval_provenance(_live_input())
    broken["endpoint"] = "http://[invalid"
    _rehash(broken)
    assert not validate_arc_eval_provenance(broken).valid
    broken = build_arc_eval_provenance(_live_input())
    broken["lease_checked_at"] = "2026-09-05T01:00:00"
    _rehash(broken)
    assert not validate_arc_eval_provenance(broken).valid

    no_llm = build_arc_eval_provenance(_no_llm_fixture())
    no_llm.update({"request_count": 1, "completion_count": 1})
    _rehash(no_llm)
    assert any("no-LLM counters" in error for error in validate_arc_eval_provenance(no_llm).errors)
    assert provenance._sha256_file(tmp_path / "absent") is None
    monkeypatch.setattr(
        provenance.inspect, "getsourcefile", lambda _value: (_ for _ in ()).throw(OSError())
    )
    assert provenance._policy_path(object()) is None
    monkeypatch.setenv("CARNOT_ARC_EVAL_LEASE_JSON", "not-json")
    assert provenance._explicit_lease(None) == {}
    monkeypatch.setenv("CARNOT_ARC_EVAL_LEASE_JSON", '{"lease_id":"from-env"}')
    assert provenance._explicit_lease(None)["lease_id"] == "from-env"

    model = tmp_path / "observed.gguf"
    server = tmp_path / "llama-server"
    model.write_bytes(b"model")
    server.write_bytes(b"server")

    class BrokenProposer:
        n_completion_calls = 1
        n_completion_ok = 0
        model_path = str(model)
        model_filename = "contradictory.gguf"
        model_repository = "owner/repository"
        n_ctx = 98304
        port = 8919
        generator_server_path = ""
        last_launch_argv = (str(server), "-m", str(model), "--port", "8919")

        def _url(self) -> str:
            raise RuntimeError("endpoint unreadable")

    class BrokenPolicy:
        proposer = BrokenProposer()

    with pytest.raises(ValueError) as invalid:
        provenance.build_arc_eval_provenance_for_policy(
            BrokenPolicy(),
            counters_before={"requests": 0, "completions": 0, "errors": 0},
            counters_after={"requests": 1, "completions": 0, "errors": 1},
            envelope={
                "generator_cuda_gpu_requested": "1",
                "gpus_held": [{"index": 1, "gpu_uuid": _live_input().gpu_uuid, "gpu_model": "RTX"}],
            },
            solve_provenance="live_agent_self_discovery",
            factory_path=Path(__file__),
            repo_root=REPO,
            lease={
                "lease_id": "lease-test",
                "lease_hash": _digest("f"),
                "lease_issued_at": "2026-09-05T00:00:00+00:00",
                "lease_expires_at": "2026-09-05T02:00:00+00:00",
            },
            lease_checked_at="2026-09-05T01:00:00+00:00",
        )
    assert "model_filename" in str(invalid.value)
    assert "endpoint" in str(invalid.value)


def _no_llm_fixture() -> ArcEvalProvenanceInput:
    na = NOT_APPLICABLE
    return ArcEvalProvenanceInput(
        inference_substrate=NO_LLM_INFERENCE_SUBSTRATE,
        gpu_uuid=na,
        gpu_model=na,
        cuda_device=na,
        model_repository=na,
        model_filename=na,
        model_hash=na,
        n_ctx=na,
        server_binary=na,
        server_binary_hash=na,
        server_command_hash=na,
        endpoint=na,
        port=na,
        lease_id=na,
        lease_hash=na,
        lease_issued_at=na,
        lease_expires_at=na,
        lease_checked_at=na,
        request_count=0,
        completion_count=0,
        error_count=0,
        policy_hash=_digest("c"),
        factory_hash=_digest("d"),
        git_commit="e" * 40,
        solve_provenance="development_proxy",
    )


def test_artifact_validator_rejects_each_gate_corruption(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    """REQ-ARC-7010 artifact verification cannot trust its stored ready score."""
    clean = build_artifact(REPO, execution_date="20260905")
    assert validate_artifact(None) == ["artifact must be an object"]
    cases = (
        ("missing", lambda value: value.pop("rows"), "missing artifact field"),
        (
            "principle",
            lambda value: value["field_principles"].pop("rows"),
            "missing field principles",
        ),
        (
            "accepted",
            lambda value: value["accepted_fixture_rows"][0]["record"].update({"gpu_uuid": "bad"}),
            "accepted fixture rejected",
        ),
        (
            "rejected",
            lambda value: value["rejected_fixture_rows"][0].update(
                {"record": clean["accepted_fixture_rows"][0]["record"], "accepted": True}
            ),
            "rejection fixture accepted",
        ),
        ("row", lambda value: value["rows"][0].pop("solve_provenance"), "evaluation row"),
        (
            "ready",
            lambda value: value.update({"arc_eval_provenance_contract_ready_score": 0}),
            "ready score",
        ),
        ("class", lambda value: value.update({"verdict_class": "blocked"}), "verdict_class"),
        ("prefix", lambda value: value.update({"honest_verdict": "blocked_wrong"}), "prefix"),
        ("oracle", lambda value: value.update({"verifier_is_oracle": True}), "verifier_is_oracle"),
        (
            "historical",
            lambda value: value.update({"historical_artifacts_modified": True}),
            "historical artifacts",
        ),
        ("checksum", lambda value: value.update({"random_seed": 0}), "checksum"),
    )
    for _name, mutate, expected in cases:
        corrupted = copy.deepcopy(clean)
        mutate(corrupted)
        assert any(expected in error for error in validate_artifact(corrupted))

    monkeypatch.setattr(experiment, "build_artifact", lambda *_args, **_kwargs: clean)
    monkeypatch.setattr(experiment, "validate_artifact", lambda _artifact: ["bad"])
    with pytest.raises(ValueError):
        experiment.write_artifact(tmp_path, execution_date="20260905")
    monkeypatch.setattr(experiment, "validate_artifact", lambda _artifact: [])
    out = experiment.write_artifact(tmp_path, execution_date="20260905")
    assert json.loads(out.read_text())["random_seed"] == experiment.RANDOM_SEED
    monkeypatch.setattr(experiment, "write_artifact", lambda *_args, **_kwargs: out)
    assert experiment.main(["--date", "20260905"]) == 0
    assert str(out) in capsys.readouterr().out


def test_package_module_entrypoint_delegates_to_main(monkeypatch) -> None:
    """REQ-ARC-7010 keeps the package's direct-execution boundary covered."""

    class StopAtArgumentParseError(RuntimeError):
        pass

    class Parser:
        def add_argument(self, *_args, **_kwargs) -> None:
            return None

        def parse_args(self, _argv=None):
            raise StopAtArgumentParseError

    monkeypatch.setattr(argparse, "ArgumentParser", Parser)
    with pytest.raises(StopAtArgumentParseError):
        runpy.run_path(experiment.__file__, run_name="__main__")
