"""REQ-REPORT-7920-V687; REQ-VERIFY-7920-V687; SCENARIO-REPORT-7920-*.

Private fixtures check accounting and reject changed custody before inference.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_7920_v687_qwen_sufficiency as exp
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.verify import context_sufficiency_7854 as context
from carnot.verify import source_interventions as source


def family(number: int = 0) -> dict[str, Any]:
    text = f"First {number}. Second item. Third item. Fourth item. Fifth item."
    answer = "Third item. More text."
    return dict(
        family_id=str(number),
        source_group=str(number),
        complete_source=text,
        complete_response=answer,
        source_sha256=source.digest(text.encode()),
        response_sha256=source.digest(answer.encode()),
        eligible=True,
    )


class Runtime:
    """Expose deterministic replies without claiming a pretrained model call."""

    def count(self, text: str) -> int:
        return len(text.split())

    def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.loads(payload["messages"][1]["content"])
        return dict(
            model=source.MODEL_ID,
            choices=[
                dict(
                    finish_reason="stop",
                    message=dict(
                        content=json.dumps(
                            dict(
                                unsupported_probability=0.3,
                                source_sentence_id=body["visible_source_sentence_ids"][-1],
                            )
                        )
                    ),
                )
            ],
            usage=dict(completion_tokens=20, prompt_tokens=50),
        )


def test_capture_and_cold_reduce(tmp_path: Path) -> None:
    rows = exp.measure([family()], context.freeze_protocol(seed=67801), Runtime(), tmp_path)
    assert len(rows) == 4 and all(r["intended"] for r in rows)
    assert rows[0]["completed"] and all(r["source_byte_fidelity"] for r in rows if r["started"])
    result = exp.reduce_rows(rows)
    assert result["sample_size_budget"]["intended"] == 4
    assert result["natural_brier"] is None and result["semantic_sensitivity"]["interval"] is None
    artifact = dict(result, rows=rows, raw_response_shards=[])
    path = tmp_path / "candidate.json"
    atomic_json(path, artifact)
    assert exp.replay(path) == result
    artifact["sample_size_budget"]["completed"] = 999
    atomic_json(path, artifact)
    with pytest.raises(ValueError, match="reduction_drift"):
        exp.replay(path)


@pytest.mark.parametrize("kind", ["excluded", "overlength", "invalid", "budget", "timeout"])
def test_capture_failure_denominators(tmp_path: Path, kind: str) -> None:
    row = family()
    runtime = Runtime()
    if kind == "excluded":
        row["eligible"] = False
    if kind == "overlength":
        runtime.count = lambda _: 9000
    if kind == "invalid":
        runtime.generate = lambda _: dict(model="wrong")
    if kind == "timeout":

        def timeout(_: Any) -> Any:
            raise TimeoutError()

        runtime.generate = timeout
    rows = exp.measure(
        [row],
        context.freeze_protocol(seed=67801),
        runtime,
        tmp_path,
        latest_launch_s=0 if kind == "budget" else 2400,
    )
    assert len(rows) == 4 and not all(r["completed"] for r in rows)
    assert exp.reduce_rows(rows)["sample_size_budget"]["independent"] == 0


def test_source_cluster_interval() -> None:
    rows = [
        dict(
            family_id=str(i),
            source_group=str(i),
            arm=arm,
            intended=True,
            eligible=True,
            started=True,
            completed=True,
            failed=False,
            censored=False,
            excluded=False,
            probability=0.2 if arm == "witness_neighbors" else 0.4,
            source_byte_fidelity=True,
            syntax_valid=True,
        )
        for i in range(32)
        for arm in context.ARMS
    ]
    reduced = exp.reduce_rows(rows)
    assert reduced["sample_size_budget"]["independent"] == 32
    assert reduced["semantic_sensitivity"]["mean_neighbors_minus_filler"] == pytest.approx(-0.2)
    assert reduced["semantic_sensitivity"]["interval"] == pytest.approx([-0.2, -0.2])
    rows[0]["source_group"] = "1"
    assert exp.reduce_rows(rows)["sample_size_budget"]["independent"] == 31


def test_authority_and_freeze(tmp_path: Path) -> None:
    public = tmp_path / "public.jsonl"
    public.write_text(
        "\n".join(
            json.dumps(
                dict(
                    family_id=str(i),
                    source_bytes=family(i)["complete_source"].encode().hex(),
                    answer_bytes=family(i)["complete_response"].encode().hex(),
                )
            )
            for i in range(50)
        )
    )
    cohort = tmp_path / "cohort.json"
    atomic_json(
        cohort,
        dict(
            rows=[
                dict(family_id=str(i), source_group=f"{i:02d}", role="evaluation")
                for i in range(50)
            ]
        ),
    )
    authority = dict(
        public_shards=[dict(path=str(public), sha256=sha256_file(public))],
        cohort_manifest_path=str(cohort),
        cohort_manifest_sha256=sha256_file(cohort),
    )
    rows = exp.freeze_families(authority)
    assert len(rows) == 48 and rows[0]["source_group"] == "00"
    public.write_text("drift")
    with pytest.raises(ValueError, match="custody"):
        exp.freeze_families(authority)


def test_authentication_missing_and_changed(tmp_path: Path) -> None:
    checks, authorities = exp.authenticate(tmp_path)
    assert authorities == {} and all(not r["passed"] for r in checks)
    for name, score in [
        ("7917_v687_intervention_qualification", "intervention_protocol_ready_score"),
        ("7892_v685_source_boundary", "source_boundary_ready_score"),
    ]:
        atomic_json(
            tmp_path / "results" / f"experiment_{name}.json",
            dict(verdict_class="disqualified", flagged_adversarial=True, **{score: 0}),
        )
    checks, authorities = exp.authenticate(tmp_path)
    assert len(authorities) == 2 and any(not r["passed"] for r in checks)


def test_manifest_and_dates(tmp_path: Path) -> None:
    plan = exp.command_manifest(tmp_path)
    for item in plan["commands"]:
        if item["name"].startswith("e2e_016"):
            assert item["argv"][item["argv"].index("--date") + 1] == "20260929"
    with pytest.raises(ValueError, match="run_date_mismatch"):
        exp.main(["--date", "20260929"])


def test_replay_shard_drift(tmp_path: Path) -> None:
    shard = tmp_path / "rows.json"
    atomic_json(shard, dict(rows=[]))
    path = tmp_path / "candidate.json"
    atomic_json(
        path,
        dict(
            exp.reduce_rows([]),
            rows=[],
            raw_response_shards=[dict(path=str(shard), sha256="changed")],
        ),
    )
    with pytest.raises(ValueError, match="shard_drift"):
        exp.replay(path)


def test_runtime_tokens_generation_and_identity(monkeypatch: Any, tmp_path: Path) -> None:
    from carnot.inference import qwen_sufficiency_7920 as live

    runtime = live.QwenRuntime(tmp_path / "model.gguf", tmp_path, 0)

    class Worker:
        receipt = dict(owned_by_task=True)

        def post_json(self, path: str, body: Any, timeout_s: float) -> Any:
            if path == "/apply-template":
                return dict(prompt="rendered")
            if path == "/tokenize":
                return dict(tokens=[1, 2, 3])
            return Runtime().generate(body)

    runtime.worker = Worker()
    assert runtime.count("plain") == 3
    assert runtime.count(json.dumps([dict(role="user", content="x")])) == 3
    assert (
        runtime.generate(
            context.build_request(
                family(), "full_source", None, None, context.freeze_protocol(seed=67801), len
            )
        )["model"]
        == source.MODEL_ID
    )
    assert runtime.receipts[0]["output_tokens"] == 20
    assert live.offload_layers("offloaded 65/65 layers to GPU") == [65, 65]
    assert live.offload_layers("CPU only") == [0, 0]


def test_bounded_heartbeat_success_failure_and_timeout() -> None:
    from carnot.inference.qwen_sufficiency_7920 import bounded

    assert bounded(lambda: 4, 1, interval_s=0.001) == 4

    def fail() -> Any:
        raise ValueError("owned_error")

    with pytest.raises(ValueError, match="owned_error"):
        bounded(fail, 1)
    import time

    with pytest.raises(TimeoutError):
        bounded(lambda: time.sleep(0.05), 0.001, interval_s=0.001)


def test_runtime_load_cleanup(monkeypatch: Any, tmp_path: Path) -> None:
    from carnot.inference import qwen_sufficiency_7920 as live

    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")

    class Worker:
        receipt = dict(owned_by_task=True)

        def __init__(self, **kwargs: Any) -> None:
            pass

        def launch(self) -> Any:
            return self.receipt

        def wait_for_health(self, _: Any) -> Any:
            return dict(ok=True)

        def cleanup(self) -> Any:
            return dict(leak_free=True)

    monkeypatch.setattr(live, "OwnedLlamaCppProcess", Worker)
    monkeypatch.setattr(
        live,
        "get_json",
        lambda _: dict(
            model_path=str(model),
            chat_template="template",
            default_generation_settings=dict(n_ctx=8192),
        ),
    )
    monkeypatch.setattr(live, "offload_layers", lambda _: [65, 65])
    runtime = live.QwenRuntime(model, tmp_path, 0)
    runtime.log.write_text("offloaded 65/65 layers to GPU")
    receipt = runtime.load()
    assert receipt["authenticated"] and receipt["tokenizer"] == "embedded_GGUF"
    assert runtime.close()["leak_free"]
    monkeypatch.setattr(live, "get_json", lambda _: dict(model_path="wrong"))
    with pytest.raises(RuntimeError, match="identity"):
        runtime.load()
    monkeypatch.setattr(Worker, "wait_for_health", lambda *_: dict(ok=False))
    with pytest.raises(RuntimeError, match="load"):
        runtime.load()
    runtime.worker = None
    assert runtime.close()["leak_free"]


def test_no_run_and_terminal_recheck(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.setattr(exp, "terminal", lambda *args: (True, False, dict(candidate_sha256="x")))
    artifact = exp.run(tmp_path, tmp_path / "scratch", tmp_path / "result.json")
    assert artifact["verdict_class"] == "blocked" and artifact["MODEL_SPECS"] == []
    assert artifact["model_invocation_counts"]["generation_calls_attempted"] == 0
    assert artifact["qwen_measurement_ready_score"] == 0
    monkeypatch.setattr(exp, "terminal", lambda *args: (False, True, dict(candidate_sha256="x")))
    with pytest.raises(ValueError, match="terminal_revalidation_failed"):
        exp.run(tmp_path, tmp_path / "other", tmp_path / "unpublished.json")
    assert not (tmp_path / "unpublished.json").exists()


def test_primitive_schema_and_owned_failure(tmp_path: Path) -> None:
    artifact = exp.build_artifact([], [], {}, {}, [], {}, [], time.monotonic_ns())
    assert artifact["verdict_class"] == "null"
    failed = dict(name="required", scope="required", passed=False, command_argv=["false"])
    artifact = exp.build_artifact([], [], {}, {}, [failed], {}, [], time.monotonic_ns())
    assert artifact["verdict_class"] == "disqualified"
    assert set(artifact) <= set(artifact["field_principles"])


@pytest.mark.parametrize("mode", ["success", "missing", "capacity", "lease", "load"])
def test_owned_runtime_gate_paths(monkeypatch: Any, tmp_path: Path, mode: str) -> None:
    from carnot import gpu_lease_phase_journal as leases
    from carnot.inference import qwen_sufficiency_7920 as live
    from carnot.inference import sota_models

    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    monkeypatch.setattr(
        sota_models,
        "cached_current_model",
        lambda: None if mode == "missing" else dict(model_path=str(model), hf_id=source.MODEL_ID),
    )
    snapshot = tmp_path / "gpu.log"
    snapshot.write_text("0, uuid, 400, 23000\n" if mode != "capacity" else "")
    monkeypatch.setattr(exp, "execute", lambda *_: [dict(log_path=str(snapshot), passed=True)])
    monkeypatch.setattr(exp, "gpu_memory", lambda *_: (400, dict(passed=True)))

    class Lease:
        document = dict(phase="preflight")

        def owner_receipt(self) -> Any:
            return dict(owner="fixture")

        def transition(self, phase: str, **_: Any) -> None:
            self.document["phase"] = phase

        def release(self) -> Any:
            return dict(released=True)

    def acquire(**_: Any) -> Any:
        if mode == "lease":
            raise RuntimeError("device_busy")
        return Lease()

    monkeypatch.setattr(leases.GpuLease, "acquire", acquire)

    class Worker(Runtime):
        receipts: list[Any] = []

        def __init__(self, *args: Any) -> None:
            self.log = tmp_path / "server.log"
            self.log.write_text("fixture")

        def load(self) -> Any:
            if mode == "load":
                raise RuntimeError("wrong_model")
            return dict(authenticated=True, tokenizer="embedded_GGUF")

        def close(self) -> Any:
            return dict(leak_free=True)

    monkeypatch.setattr(live, "QwenRuntime", Worker)
    identity, rows, checks = exp.live_capture(
        [family()], context.freeze_protocol(seed=67801), tmp_path, tmp_path / "raw"
    )
    assert len(rows) == (4 if mode == "success" else 0)
    assert bool(identity.get("authenticated")) == (mode == "success")
    assert all(c["passed"] for c in checks) == (mode == "success")
    if mode == "success":
        assert identity["gpu_lease"]["release"]["released"]


def test_full_orchestration_and_terminal_hook(monkeypatch: Any, tmp_path: Path) -> None:
    from carnot import experiment_7893_v685_intervention_protocol as previous

    monkeypatch.setattr(
        previous, "validate_terminal", lambda *_: (True, False, dict(candidate_sha256="unit"))
    )
    assert exp.terminal(tmp_path / "unused", tmp_path, 0)[0]
    monkeypatch.setattr(exp, "authenticate", lambda _: ([], dict(**{"7892": {}, "7917": {}})))
    monkeypatch.setattr(exp, "freeze_families", lambda _: [family()])
    captured = exp.measure(
        [family()], context.freeze_protocol(seed=67801), Runtime(), tmp_path / "capture"
    )
    monkeypatch.setattr(
        exp,
        "live_capture",
        lambda *_: (dict(authenticated=True, load_attempted=True), captured, []),
    )
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    atomic_json(
        scratch / "coverage.json",
        dict(
            files={
                exp.MODULE: dict(summary=dict(num_statements=1, covered_lines=1, missing_lines=0))
            }
        ),
    )
    monkeypatch.setattr(
        exp,
        "execute",
        lambda *_: [dict(name="fixture", scope="required", passed=True, command_argv=["fixture"])],
    )
    artifact = exp.run(tmp_path, scratch, tmp_path / "result.json")
    assert artifact["qwen_measurement_ready_score"] == 1
    assert artifact["MODEL_SPECS"] == [source.MODEL_ID]
    assert exp.replay(tmp_path / "result.json")["sample_size_budget"]["completed"] == 4


def test_exact_command_sealing(monkeypatch: Any, tmp_path: Path) -> None:
    log = tmp_path / "child.log"
    log.write_text("expected_reason")
    monkeypatch.setattr(
        exp,
        "run_commands",
        lambda *_args, **_kwargs: [
            dict(
                log_path=str(log),
                log_sha256=sha256_file(log),
                exit_code=1,
                timed_out=False,
                name="rejection",
            )
        ],
    )
    plan = dict(
        commands=[
            dict(
                name="rejection",
                argv=["fixture"],
                scope="required",
                timeout_s=5,
                expected_exit_code=1,
                expected_failure_reason="expected_reason",
            )
        ]
    )
    receipt = exp.execute(plan, tmp_path, tmp_path / "sealed")[0]
    assert receipt["passed"] and Path(receipt["log_path"]).read_bytes() == log.read_bytes()


def test_runtime_get_json(monkeypatch: Any) -> None:
    from carnot.inference import qwen_sufficiency_7920 as live

    class Response:
        def __enter__(self) -> Any:
            return self

        def __exit__(self, *_: Any) -> None:
            pass

        def read(self) -> bytes:
            return b'{"model_path":"fixture"}'

    monkeypatch.setattr(live, "urlopen", lambda *_args, **_kwargs: Response())
    assert live.get_json("http://fixture")["model_path"] == "fixture"


def test_gpu_memory_receipt(monkeypatch: Any, tmp_path: Path) -> None:
    log = tmp_path / "gpu.log"
    log.write_text("18001\n")
    monkeypatch.setattr(exp, "execute", lambda *_: [dict(log_path=str(log), passed=True)])
    value, receipt = exp.gpu_memory(0, tmp_path, tmp_path)
    assert value == 18001 and receipt["passed"]


def test_qualified_custody_references(tmp_path: Path) -> None:
    dependency = tmp_path / "dependency.txt"
    dependency.write_text("frozen")
    manifest = tmp_path / "manifest.json"
    atomic_json(manifest, dict(source_hashes={str(dependency): sha256_file(dependency)}))
    for number, name, score in [
        (7917, "intervention_qualification", "intervention_protocol_ready_score"),
        (7892, "source_boundary", "source_boundary_ready_score"),
    ]:
        atomic_json(
            tmp_path
            / "results"
            / f"experiment_{number}_v{687 if number == 7917 else 685}_{name}.json",
            dict(
                experiment_id=number,
                verdict_class="circular_positive",
                flagged_adversarial=False,
                **{score: 1},
                source_artifact_hashes={
                    "dependency": dict(path=str(dependency), sha256=sha256_file(dependency))
                },
                validation_command_manifest_path=str(manifest),
                protocol_manifest_path=str(manifest),
                protocol_manifest_sha256=sha256_file(manifest),
            ),
        )
    checks, authorities = exp.authenticate(tmp_path)
    assert len(authorities) == 2 and all(c["passed"] for c in checks)
    dependency.write_text("changed")
    assert any(not c["passed"] for c in exp.authenticate(tmp_path)[0])


def test_failed_e2e_and_successful_terminal_recheck(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.setattr(exp, "authenticate", lambda _: ([], {"7892": {}, "7917": {}}))
    monkeypatch.setattr(exp, "freeze_families", lambda _: [family()])
    monkeypatch.setattr(
        exp,
        "execute",
        lambda *_: [dict(name="failed", scope="required", passed=False, command_argv=["fixture"])],
    )
    calls = []

    def terminal(*_: Any) -> Any:
        calls.append(1)
        return len(calls) > 1, len(calls) == 1, dict(candidate_sha256="fixture")

    monkeypatch.setattr(exp, "terminal", terminal)
    artifact = exp.run(tmp_path, tmp_path / "scratch", tmp_path / "result.json")
    assert len(calls) == 2 and artifact["verdict_class"] == "disqualified"
    assert artifact["qwen_measurement_ready_score"] == 0


def test_historical_wrong_date_rejects_before_fixture(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7920-CUSTODY: preserve the historical producer date."""
    import os
    import subprocess

    output = tmp_path / "wrong-date.json"
    child = subprocess.run(
        [
            str(exp.ROOT / ".venv/bin/python"),
            "scripts/experiments/experiment_7868_v683_intervention_protocol.py",
            "--date",
            "20260930",
            "--fixture-e2e",
            str(output),
        ],
        cwd=exp.ROOT,
        env={**os.environ, "PYTHONPATH": "python:."},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert child.returncode == 1 and "run_date_mismatch" in child.stderr
    assert not output.exists()


def test_custody_hex_bytes_are_decoded_before_sentence_selection(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7920-CAPTURE: source bytes are hexadecimal custody fields."""
    item = family()
    public = tmp_path / "public.jsonl"
    cohort = tmp_path / "cohort.json"
    public.write_text(
        json.dumps(
            dict(
                family_id="0",
                source_bytes=item["complete_source"].encode().hex(),
                answer_bytes=item["complete_response"].encode().hex(),
            )
        )
        + "\n"
    )
    atomic_json(cohort, dict(rows=[dict(family_id="0", source_group="0", role="evaluation")]))
    authority = dict(
        public_shards=[dict(path=str(public), sha256=sha256_file(public))],
        cohort_manifest_path=str(cohort),
        cohort_manifest_sha256=sha256_file(cohort),
    )
    frozen = exp.freeze_families(authority)
    assert frozen[0]["complete_source"] == item["complete_source"]
    assert frozen[0]["complete_response"] == item["complete_response"] and frozen[0]["eligible"]
