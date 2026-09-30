"""REQ-VERIFY-7932-V688; REQ-REPORT-7932-V688.

SCENARIO-VERIFY-7932-VIEWS, SCENARIO-VERIFY-7932-REDUCE and
SCENARIO-REPORT-7932-VALIDATION keep transport success distinct from truth.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_7932_v688_qwen_completion as exp
from carnot.verify import qwen_completion_7932 as protocol
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.verify import source_interventions as source


def family(number: int = 0) -> dict[str, Any]:
    text = (
        "Other words. Other words. Target fact. Other words. Other words. Other words. Other words."
    )
    answer = "Target fact. More text."
    return dict(
        family_id=str(number),
        source_group=str(number),
        complete_source=text,
        complete_response=answer,
        source_sha256=source.digest(text.encode()),
        response_sha256=source.digest(answer.encode()),
        eligible=True,
    )


def response(
    content: str = '{"unsupported_probability":0.3,"source_sentence_id":null}',
    finish: str = "stop",
    model: str = source.MODEL_ID,
    tokens: int = 20,
) -> dict[str, Any]:
    return dict(
        model=model,
        choices=[dict(finish_reason=finish, message=dict(content=content))],
        usage=dict(completion_tokens=tokens, prompt_tokens=100),
    )


class Runtime:
    """Keep fixture replies separate from any claim of pretrained inference."""

    def count(self, text: str) -> int:
        return len(text.split())

    def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        return response()


def test_frozen_views_and_identical_decoder_messages() -> None:
    views = protocol.freeze_views([family()], Runtime().count)
    item = views[0]
    assert item["eligible"] and item["witness_sentence_id"] == 2
    assert item["filler_sentence_ids"] == [4, 5]
    assert item["neighbor_added_tokens"] == item["filler_tokens"]
    for view in protocol.VIEWS:
        plain = protocol.payload(item, view, "plain")
        grammar = protocol.payload(item, view, "grammar")
        assert plain["messages"] == grammar["messages"]
        assert "grammar" not in plain and "response_format" not in plain
        assert grammar["grammar"] == protocol.GRAMMAR
        assert plain["max_tokens"] == 96 and plain["seed"] == 67801
        body = json.loads(plain["messages"][1]["content"])
        assert body["original_answer"] == "Target fact."
    tied = family()
    tied["complete_source"] = "Target fact. Target fact. Other words. Other words. Other words."
    tied["source_sha256"] = source.digest(tied["complete_source"].encode())
    assert protocol.freeze_views([tied], Runtime().count)[0]["witness_sentence_id"] == 0
    assert protocol.freeze_config()["output_token_limit"] == 36864


@pytest.mark.parametrize(
    "mode", ["ineligible", "filler", "context", "source_drift", "answer_drift"]
)
def test_view_exclusions(mode: str) -> None:
    row = family()
    count = Runtime().count
    if mode == "ineligible":
        row["eligible"] = False
    if mode == "filler":
        row["complete_source"] = "Target fact."
        row["source_sha256"] = source.digest(row["complete_source"].encode())
    if mode == "context":
        count = lambda _: 8192
    if mode.endswith("drift"):
        row["source_sha256" if mode == "source_drift" else "response_sha256"] = "wrong"
        with pytest.raises(ValueError, match="custody_drift"):
            protocol.freeze_views([row], count)
        return
    item = protocol.freeze_views([row], count)[0]
    assert not item["eligible"] and item["exclusion_reason"]


@pytest.mark.parametrize(
    "body,finish,model,tokens,status,syntax",
    [
        (
            '{"unsupported_probability":0.2,"source_sentence_id":2}',
            "stop",
            source.MODEL_ID,
            20,
            "completed",
            True,
        ),
        (
            '{"unsupported_probability":0,"source_sentence_id":null}',
            "stop",
            source.MODEL_ID,
            20,
            "completed",
            True,
        ),
        (
            '{"unsupported_probability":0.2,"source_sentence_id":99}',
            "stop",
            source.MODEL_ID,
            20,
            "invalid_citation",
            True,
        ),
        (
            '{"unsupported_probability":true,"source_sentence_id":null}',
            "stop",
            source.MODEL_ID,
            20,
            "invalid_schema",
            True,
        ),
        (
            '{"unsupported_probability":2,"source_sentence_id":null}',
            "stop",
            source.MODEL_ID,
            20,
            "invalid_schema",
            True,
        ),
        (
            '{"unsupported_probability":NaN,"source_sentence_id":null}',
            "stop",
            source.MODEL_ID,
            20,
            "invalid_syntax",
            False,
        ),
        (
            '{"unsupported_probability":0,"unsupported_probability":1,"source_sentence_id":null}',
            "stop",
            source.MODEL_ID,
            20,
            "invalid_syntax",
            False,
        ),
        ("{}", "stop", source.MODEL_ID, 20, "invalid_schema", True),
        ("[]", "stop", source.MODEL_ID, 20, "invalid_schema", True),
        (
            '{"unsupported_probability":0,"source_sentence_id":true}',
            "stop",
            source.MODEL_ID,
            20,
            "invalid_schema",
            True,
        ),
        ("no JSON", "stop", source.MODEL_ID, 20, "invalid_syntax", False),
        ("{}", "length", source.MODEL_ID, 96, "token_limit", True),
        ("{}", "stop", "wrong", 20, "wrong_model", True),
        ("{}", "stop", source.MODEL_ID, 97, "token_limit", True),
    ],
)
def test_reply_validation(
    body: str, finish: str, model: str, tokens: int, status: str, syntax: bool
) -> None:
    parsed = protocol.parse_response(response(body, finish, model, tokens), [2])
    assert parsed["status"] == status and parsed["syntax_valid"] is syntax
    assert parsed["completed"] == (status == "completed")


def test_malformed_envelope() -> None:
    assert protocol.parse_response({}, [])["status"] == "invalid_envelope"
    with pytest.raises(ValueError, match="decoder"):
        protocol.payload({}, "full_source", "fallback")


def test_capture_checkpoint_and_invalid_output_retention(tmp_path: Path) -> None:
    runtime = Runtime()
    runtime.generate = lambda p: response() if "grammar" in p else response("broken")
    views = protocol.freeze_views([family()], runtime.count)
    rows = protocol.capture(views, runtime, tmp_path)
    assert len(rows) == 8 and sum(r["completed"] for r in rows) == 4
    assert all(r["started"] for r in rows)
    assert json.loads((tmp_path / "checkpoint.json").read_text())["config_hash"]
    reduced = protocol.reduce_rows(rows)
    assert reduced["complete_family_fraction"]["grammar"] == 1 / 48
    assert reduced["complete_family_fraction"]["plain"] == 0
    assert reduced["sentence_label_eligibility"]["eligible"] == 0
    assert reduced["natural_brier"] is None and reduced["natural_cost"] is None


@pytest.mark.parametrize("mode", ["excluded", "budget", "timeout", "transport", "token_budget"])
def test_capture_failures_keep_denominator(tmp_path: Path, mode: str) -> None:
    runtime = Runtime()
    views = protocol.freeze_views([family()], runtime.count)
    if mode == "excluded":
        views[0]["eligible"] = False
    if mode in {"timeout", "transport"}:

        def fail(_: Any) -> Any:
            raise TimeoutError() if mode == "timeout" else OSError("transport")

        runtime.generate = fail
    rows = protocol.capture(
        views,
        runtime,
        tmp_path,
        latest_launch_s=0 if mode == "budget" else 2400,
        token_budget=0 if mode == "token_budget" else 36864,
    )
    assert len(rows) == 8 and not any(r["completed"] for r in rows)
    if mode == "timeout":
        assert all(r["censored"] for r in rows)
    reduced = protocol.reduce_rows(rows)
    assert reduced["sample_size_budget"]["intended"] == 384
    assert reduced["sample_size_budget"]["accounted"] == 8


def test_paired_bootstrap_discordance_and_secondary(tmp_path: Path) -> None:
    rows = protocol.capture(
        protocol.freeze_views([family(i) for i in range(48)], Runtime().count), Runtime(), tmp_path
    )
    for row in rows:
        row["probability"] = 0.2 if row["view"] == "witness_neighbors" else 0.4
        if int(row["family_id"]) < 8 and row["decoder"] == "plain":
            row["completed"] = False
            row["failed"] = True
    reduced = protocol.reduce_rows(rows)
    assert reduced["paired_completion_delta"]["estimate"] == pytest.approx(8 / 48)
    assert reduced["paired_completion_delta"]["protocol_benefit"]
    assert reduced["paired_completion_delta"]["exact_p"] == pytest.approx(2 / 256)
    sensitivity = reduced["semantic_sensitivity"]
    assert sensitivity["complete_in_both"] == 40
    assert sensitivity["arms"]["grammar"]["estimate"] == pytest.approx(-0.2)
    assert sensitivity["arms"]["plain"]["ci95"] == pytest.approx([-0.2, -0.2])
    rows[0]["source_group"] = "wrong"
    with pytest.raises(ValueError, match="group_drift"):
        protocol.reduce_rows(rows)


def test_replay_custody_and_aggregate_drift(tmp_path: Path) -> None:
    rows = protocol.capture(protocol.freeze_views([family()], Runtime().count), Runtime(), tmp_path)
    shard = tmp_path / "rows.json"
    atomic_json(shard, dict(rows=rows))
    artifact = dict(
        protocol.reduce_rows(rows),
        rows=rows,
        raw_response_shards=[dict(path=str(shard), sha256=sha256_file(shard))],
    )
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, artifact)
    assert exp.replay(candidate)["sample_size_budget"]["completed"] == 8
    artifact["paired_completion_delta"]["estimate"] = 99
    atomic_json(candidate, artifact)
    with pytest.raises(ValueError, match="reduction_drift"):
        exp.replay(candidate)
    shard.write_text("drift")
    with pytest.raises(ValueError, match="shard_drift"):
        exp.replay(candidate)


def test_manifest_dates_and_coverage_scope(tmp_path: Path) -> None:
    plan = exp.command_manifest(tmp_path)
    assert "qwen_completion_7932.py" in plan["coverage_include"]
    assert "qwen_sufficiency_7920.py" not in plan["coverage_include"]
    assert plan["commands"][0]["name"] == "model_free_checks"
    for item in plan["commands"]:
        if item["name"].startswith("e2e_016"):
            assert item["argv"][item["argv"].index("--date") + 1] == "20260929"
    with pytest.raises(ValueError, match="run_date_mismatch"):
        exp.main(["--date", "20260929"])


def test_authenticate_missing(tmp_path: Path) -> None:
    checks, authorities = exp.authenticate(tmp_path)
    assert not authorities and all(not r["passed"] for r in checks)


@pytest.mark.parametrize(
    "mode",
    ["success", "missing", "capacity", "lease", "load", "cleanup", "metadata", "metadata_error"],
)
def test_owned_runtime_paths(monkeypatch: Any, tmp_path: Path, mode: str) -> None:
    from carnot import gpu_lease_phase_journal as leases
    from carnot.inference import sota_models

    model = tmp_path / "Q4_K_M.gguf"
    model.write_bytes(b"GGUF")

    def metadata(_: Any) -> Any:
        if mode == "metadata_error":
            raise ValueError("malformed_gguf")
        return dict(
            quantization="Q8_0" if mode == "metadata" else "Q4_K_M",
            tokenizer_metadata=dict(token_count=256, chat_template_present=True),
        )

    monkeypatch.setattr(exp, "read_gguf_metadata", metadata)
    monkeypatch.setattr(
        sota_models,
        "cached_current_model",
        lambda: None if mode == "missing" else dict(model_path=str(model), hf_id=source.MODEL_ID),
    )
    log = tmp_path / "gpu.log"
    log.write_text("0, uuid, 4, 24000\n" if mode != "capacity" else "")
    monkeypatch.setattr(exp.prior, "execute", lambda *_: [dict(log_path=str(log), passed=True)])
    monkeypatch.setattr(exp.prior, "gpu_memory", lambda *_: (4, {}))

    class Lease:
        document = dict(phase="preflight")

        def owner_receipt(self) -> Any:
            return {}

        def transition(self, phase: str, **_: Any) -> None:
            self.document["phase"] = phase

        def release(self) -> Any:
            return dict(released=True)

    def acquire(**_: Any) -> Any:
        if mode == "lease":
            raise RuntimeError("busy")
        return Lease()

    monkeypatch.setattr(leases.GpuLease, "acquire", acquire)

    class Worker(Runtime):
        receipts: list[Any] = []

        def __init__(self, *args: Any) -> None:
            self.log = tmp_path / "server.log"
            self.log.write_text("fixture")

        def load(self) -> Any:
            if mode == "load":
                raise RuntimeError("load")
            return dict(authenticated=True)

        def close(self) -> Any:
            return dict(leak_free=mode != "cleanup")

    monkeypatch.setattr(exp, "QwenRuntime", Worker)
    identity, rows, checks, manifest = exp.live_capture([family()], tmp_path, tmp_path / "raw")
    assert bool(rows) == (mode in {"success", "cleanup"})
    assert all(c["passed"] for c in checks) == (mode == "success")
    if mode == "success":
        assert identity["gpu_lease"]["release"]["released"] and manifest[0]["eligible"]


@pytest.mark.parametrize("ok", [True, False])
def test_cpu_grammar_fixture(monkeypatch: Any, tmp_path: Path, ok: bool) -> None:
    model = tmp_path / "smoke.gguf"
    model.write_bytes(b"GGUF")

    class Worker:
        receipt: dict[str, Any] = {}

        def __init__(self, **kwargs: Any) -> None:
            kwargs["log_path"].write_text("CPU grammar fixture")

        def launch(self) -> Any:
            return {}

        def wait_for_health(self, _: Any) -> Any:
            return dict(ok=ok)

        def post_json(self, path: str, payload: Any, _: Any) -> Any:
            assert payload["grammar"]
            return dict(
                content='{"unsupported_probability":0,"source_sentence_id":null}',
                stop=True,
                tokens_predicted=8,
            )

        def cleanup(self) -> Any:
            return dict(leak_free=True)

    monkeypatch.setattr(exp, "OwnedLlamaCppProcess", Worker)
    receipt = exp.cpu_grammar_fixture(tmp_path, tmp_path / "raw", model=model)
    assert receipt["passed"] is ok and receipt["cleanup"]["leak_free"]


def test_artifact_readiness_and_verdicts(tmp_path: Path) -> None:
    rows = protocol.capture(protocol.freeze_views([family()], Runtime().count), Runtime(), tmp_path)
    identity = dict(authenticated=True, load_attempted=True)
    result = exp.build_artifact(rows, [], {}, identity, [], {}, [], 0)
    assert result["verdict_class"] == "null" and result["qwen_measurement_ready_score"] == 0
    checks = [exp.prior.operand("upstream", tmp_path / "missing", "exists", True, False)]
    assert exp.build_artifact([], checks, {}, {}, [], {}, [], 0)["verdict_class"] == "blocked"
    failed = dict(passed=False, scope="required", command_argv=["false"])
    assert (
        exp.build_artifact([], [], {}, {}, [failed], {}, [], 0)["verdict_class"] == "disqualified"
    )
    assert set(result) <= set(result["field_principles"])


def test_orchestration_and_terminal_recheck(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.setattr(exp, "authenticate", lambda _: ([], {"7892": {}}))
    monkeypatch.setattr(exp.prior, "freeze_families", lambda _: [family()])
    monkeypatch.setattr(exp, "cpu_grammar_fixture", lambda *_: dict(passed=True))
    rows = protocol.capture(protocol.freeze_views([family()], Runtime().count), Runtime(), tmp_path)
    monkeypatch.setattr(
        exp,
        "live_capture",
        lambda *_: (dict(authenticated=True, load_attempted=True), rows, [], []),
    )
    monkeypatch.setattr(exp.prior, "execute", lambda *_: [])
    calls = []

    def terminal(*_: Any) -> Any:
        calls.append(1)
        return len(calls) > 1, len(calls) == 1, dict(candidate_sha256="fixture", reports=[])

    monkeypatch.setattr(exp.prior, "terminal", terminal)
    result = exp.run(tmp_path, tmp_path / "scratch", tmp_path / "experiment_7932_fixture.json")
    assert result["verdict_class"] == "disqualified" and len(calls) == 3
    monkeypatch.setattr(exp.prior, "terminal", lambda *_: (False, True, {}))
    with pytest.raises(ValueError, match="terminal_revalidation_failed"):
        exp.run(tmp_path, tmp_path / "scratch2", tmp_path / "experiment_7932_fixture2.json")


def test_authentication_read_error_and_missing_cpu_fixture(
    monkeypatch: Any, tmp_path: Path
) -> None:
    def fail(_: Any) -> Any:
        raise ValueError("malformed_authority")

    monkeypatch.setattr(exp.prior, "authenticate", fail)
    checks, authorities = exp.authenticate(tmp_path)
    assert not authorities and not checks[0]["passed"]
    assert not exp.cpu_grammar_fixture(tmp_path, tmp_path, model=tmp_path / "absent")["passed"]


def test_primitive_replay_rejections(tmp_path: Path) -> None:
    rows = protocol.capture(protocol.freeze_views([family()], Runtime().count), Runtime(), tmp_path)
    candidate = tmp_path / "candidate.json"
    shard = tmp_path / "rows.json"
    atomic_json(shard, dict(rows=[]))
    base = dict(protocol.reduce_rows(rows), rows=rows, raw_response_shards=[])
    atomic_json(
        candidate,
        dict(base, raw_response_shards=[dict(path=str(shard), sha256=sha256_file(shard))]),
    )
    with pytest.raises(ValueError, match="primitive_drift"):
        exp.replay(candidate)
    rows[0]["request_sha256"] = "wrong"
    atomic_json(candidate, base)
    with pytest.raises(ValueError, match="primitive_hash_drift"):
        exp.replay(candidate)
    rows[0]["request_sha256"] = source.digest(rows[0]["request_bytes"].encode())
    rows[0]["probability"] = 1
    atomic_json(candidate, base)
    with pytest.raises(ValueError, match="parse_drift"):
        exp.replay(candidate)
    with pytest.raises(ValueError, match="duplicate_cell"):
        protocol.reduce_rows([rows[0], rows[0]])


def test_current_orchestration_coverage_and_reader_failure(
    monkeypatch: Any, tmp_path: Path
) -> None:
    monkeypatch.setattr(exp, "authenticate", lambda _: ([], {"7892": {}}))
    monkeypatch.setattr(exp.prior, "freeze_families", lambda _: [family()])
    monkeypatch.setattr(exp, "cpu_grammar_fixture", lambda *_: dict(passed=True))
    views = protocol.freeze_views([family()], Runtime().count)
    rows = protocol.capture(views, Runtime(), tmp_path)
    monkeypatch.setattr(
        exp,
        "live_capture",
        lambda *_: (dict(authenticated=True, load_attempted=True), rows, [], views),
    )

    def execute(*args: Any) -> Any:
        scratch = args[1]
        atomic_json(args[2] / "view_manifest.json", dict(rows=views))
        atomic_json(
            scratch / "coverage.json",
            dict(
                files={
                    exp.MODULE: dict(
                        summary=dict(num_statements=1, covered_lines=1, missing_lines=0)
                    )
                }
            ),
        )
        return [dict(passed=True, scope="required", command_argv=["fixture"])]

    monkeypatch.setattr(exp.prior, "execute", execute)
    monkeypatch.setattr(exp.prior, "terminal", lambda *_: (True, False, dict(reports=[])))
    result = exp.run(tmp_path, tmp_path / "scratch", tmp_path / "experiment_7932_fixture.json")
    assert result["coverage_statement_counts"][exp.MODULE]["covered"] == 1
    monkeypatch.setattr(
        exp,
        "reader_receipt",
        lambda *_args, **_kwargs: dict(
            gate_path="wrong", document_path="wrong", gate_sha256="wrong", document_sha256="wrong"
        ),
    )
    with pytest.raises(ValueError, match="primary_resolution_failed"):
        exp.run(tmp_path, tmp_path / "scratch2", tmp_path / "experiment_7932_fixture.json")


def test_main_fixture_replay_and_external_block(tmp_path: Path) -> None:
    path = tmp_path / "fixture.json"
    assert exp.main(["--fixture-e2e", str(path)]) == 0
    assert exp.main(["--cold-replay", str(path)]) == 0
    output = tmp_path / "experiment_7932_private_blocked.json"
    assert exp.main(["--root", str(tmp_path / "absent"), "--output", str(output)]) == 0
    artifact = json.loads(output.read_text())
    assert artifact["verdict_class"] == "blocked" and not artifact["MODEL_SPECS"]
    resolution = json.loads(Path(artifact["primary_resolution_receipt"]["path"]).read_text())
    assert resolution["identity_passed"] and resolution["gate_sha256"] == sha256_file(output)
    assert resolution["gates"][0]["actual"] == 0


def test_frozen_view_replay_binding(tmp_path: Path) -> None:
    views = protocol.freeze_views([family()], Runtime().count)
    rows = protocol.capture(views, Runtime(), tmp_path)
    manifest = tmp_path / "view_manifest.json"
    atomic_json(manifest, dict(rows=views))
    artifact = dict(
        protocol.reduce_rows(rows),
        rows=rows,
        raw_response_shards=[],
        view_manifest=dict(path=str(manifest), sha256=sha256_file(manifest)),
    )
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, artifact)
    assert exp.replay(candidate)
    rows[0]["request_bytes"] = rows[0]["request_bytes"].replace("Target fact.", "Edited fact.")
    rows[0]["request_sha256"] = source.digest(rows[0]["request_bytes"].encode())
    atomic_json(candidate, artifact)
    with pytest.raises(ValueError, match="frozen_request_drift"):
        exp.replay(candidate)
    manifest.write_text("drift")
    with pytest.raises(ValueError, match="view_manifest_drift"):
        exp.replay(candidate)


def test_decoder_pair_order_and_frozen_plan_binding(tmp_path: Path) -> None:
    rows = protocol.capture(protocol.freeze_views([family()], Runtime().count), Runtime(), tmp_path)
    manifest = tmp_path / "commands.json"
    atomic_json(manifest, dict(commands=[]))
    artifact = dict(
        protocol.reduce_rows(rows),
        rows=rows,
        raw_response_shards=[],
        validation_command_manifest_path=str(manifest),
        validation_command_manifest_sha256=sha256_file(manifest),
    )
    candidate = tmp_path / "candidate.json"
    rows[0]["order_position"] = 1 - rows[0]["order_position"]
    atomic_json(candidate, artifact)
    with pytest.raises(ValueError, match="decoder_order_drift"):
        exp.replay(candidate)
    manifest.write_text("drift")
    with pytest.raises(ValueError, match="command_manifest_drift"):
        exp.replay(candidate)
