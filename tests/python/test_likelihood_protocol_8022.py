"""REQ-REPORT-8022: private arithmetic and public isolation precede runtime work."""

import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from carnot import experiment_8022_v695_likelihood_protocol as e
from carnot.inference import fixed_answer_likelihood_8022 as s
from carnot.reporting.current_work_receipt import atomic_json


class Tokens:
    """Byte tokens expose boundary mistakes without substituting science models."""

    def tokenize(self, text, add_bos=True, special=False):
        return ([256] if add_bos else []) + list(text)

    def detokenize(self, tokens, **kwargs):
        return bytes(t for t in tokens if t != 256)

    def render(self, source, question):
        return b"<user>" + question + b"\n" + source + b"<assistant>\n"


def public(n=2):
    return {
        role: [
            dict(
                family_id=f"{role}-{i}",
                source_bytes=f"{role} {i} alpha beta".encode().hex(),
                answer_bytes=b"alpha".hex(),
            )
            for i in range(n)
        ]
        for role in s.ROLES
    }


def test_float64_full_vocabulary_and_alignment():
    """SCENARIO-REPORT-8022-TOKENS: independent arithmetic detects shifts and overflow."""
    logits = np.array([[1000, 999, 998], [0, 1, 2], [9, 0, 0]], dtype=np.float32)
    value = s.target_likelihood(logits, [2, 1, 2], 1)
    expected = [
        999 - (1000 + math.log(1 + math.exp(-1) + math.exp(-2))),
        2 - math.log(1 + math.e + math.exp(2)),
    ]
    assert value["target_logprobs"] == pytest.approx(expected, abs=1e-12)
    assert value["mean_nll"] == pytest.approx(-sum(expected) / 2)
    assert value["vocabulary_size"] == 3
    assert value["normalization_max_error"] < 1e-12
    for matrix, tokens, boundary in [
        (logits, [2], 0),
        (logits, [2], 1),
        (logits, [2, 9], 1),
        (np.full((3, 3), np.nan), [2, 1], 1),
    ]:
        with pytest.raises(ValueError):
            s.target_likelihood(matrix, tokens, boundary)


def test_views_offsets_removal_and_source_roles():
    """SCENARIO-REPORT-8022-PANEL: fixed bytes, stable lexical removal and no labels."""
    tokenizer = Tokens()
    item = s.prepare(public()["fit"][0], tokenizer)
    assert item["views"]["full"] == item["views"]["duplicate"]
    assert item["answer_bytes"] == public()["fit"][0]["answer_bytes"]
    assert item["removal_mask"]["selected_chunk_index"] == 0
    assert item["views"]["no_source"]["source_bytes"] == ""
    assert item["response_token_offsets"][-1] == [4, 5]
    manifest = s.freeze_panel(public(), tokenizer, dict(fit=1, tune=1, evaluation=1))
    assert manifest["counts"] == dict(fit=1, tune=1, evaluation=1)
    assert len(manifest["rows"]) == 3
    assert manifest["labels_opened"] is False
    bad = public()
    bad["fit"][0]["y"] = 1
    with pytest.raises(ValueError, match="public_fields"):
        s.freeze_panel(bad, tokenizer)
    bad = public()
    bad["tune"][0]["source_bytes"] = bad["fit"][0]["source_bytes"]
    with pytest.raises(ValueError, match="source_role_overlap"):
        s.freeze_panel(bad, tokenizer)
    bad = public()
    bad["fit"][0]["answer_bytes"] = (b"x" * 385).hex()
    manifest = s.freeze_panel(bad, tokenizer, dict(fit=1, tune=1, evaluation=1))
    assert manifest["exclusions"][0]["reason"] == "answer_token_budget"
    assert manifest["rows"][0]["family_id"] == "fit-1"
    assert s.freeze_panel(public(0), tokenizer)["complete"] is False


def test_teacher_forcing_only_and_duplicate():
    """SCENARIO-REPORT-8022-TOKENS: only eval supplies scores; generation is impossible."""

    class Runtime(Tokens):
        n_tokens = 0
        scores = None

        def reset(self):
            self.n_tokens = 0

        def eval(self, tokens):
            self.n_tokens = len(tokens)
            self.scores = np.zeros((len(tokens), 257), dtype=np.float32)

        def generate(self, *args, **kwargs):
            pytest.fail("generation forbidden")

    runtime = Runtime()
    value = s.qualify(runtime)
    assert value["passed"] and value["forward_pass_counts"] == 8
    assert value["generated_tokens"] == 0
    assert all(r["mean_nll"] == pytest.approx(math.log(257)) for r in value["rows"])
    assert value["duplicate_max_difference"] <= 1e-6


def test_durable_forward_failure_and_alignment(tmp_path):
    """SCENARIO-REPORT-8022-SEAL: a failed forward retains its original attempted slot."""

    class Broken(Tokens):
        n_tokens = 1

        def reset(self):
            pass

        def eval(self, tokens):
            pass

    with pytest.raises(ValueError, match="runtime_token_alignment"):
        s.qualify(Broken(), tmp_path)
    row = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert row["status"] == "failed" and row["generated_tokens"] == 0


def test_roundtrip_context_boundary_and_ties():
    """SCENARIO-REPORT-8022-PANEL: no hidden clipping and a stable removal tie-break."""
    row = public()["fit"][0]
    with pytest.raises(ValueError, match="public_fields"):
        s.prepare(dict(row, y=1), Tokens())
    bad = dict(row, source_bytes="")
    with pytest.raises(ValueError, match="incomplete_context"):
        s.prepare(bad, Tokens())

    class Changed(Tokens):
        def detokenize(self, tokens, **kwargs):
            return b"wrong bytes"

    with pytest.raises(ValueError, match="token_byte_roundtrip"):
        s.prepare(row, Changed())

    class Merged(Tokens):
        def tokenize(self, text, **kwargs):
            result = super().tokenize(text, **kwargs)
            return result[:-1] if text.endswith(b"<assistant>\nalpha") else result

    with pytest.raises(ValueError, match="prompt_answer_boundary"):
        s.prepare(row, Merged())
    with pytest.raises(ValueError, match="input_token_budget"):
        s.prepare(dict(row, source_bytes=(b"s" * 6001).hex()), Tokens())
    item = s.prepare(dict(row, source_bytes=(b"alpha " + b"x" * 250 + b"alpha").hex()), Tokens())
    assert item["removal_mask"]["selected_chunk_index"] == 0


def test_frozen_public_slots_keep_null_metrics():
    """SCENARIO-REPORT-8022-PANEL: frozen views do not invent public measurements."""
    value = s.freeze_panel(public(), Tokens(), dict(fit=1, tune=1, evaluation=1))
    assert len(value["slots"]) == 12
    assert all(r["mean_nll"] is None and r["denominator"] == 5 for r in value["slots"])
    assert all(r["status"] == "frozen_unscored" for r in value["slots"])


def test_upstream_missing_is_terminal():
    """REQ-REPORT-8022: no upstream fit readiness can gate this independent branch."""
    plan = e.authenticate(e.ROOT)
    assert all(c["passed"] for c in plan["checks"])
    assert set(plan["public"]) == set(s.ROLES)
    plan = e.authenticate(Path("/tmp/carnot-8022-nonexistent"))
    assert any(not c["passed"] for c in plan["checks"])


def cli(args, tmp_path):
    argv = [sys.executable, str(e.ROOT / e.CLI), *map(str, args)]
    env = dict(os.environ, PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    config = env.get("CARNOT_8022_COVERAGE_CONFIG")
    if config:
        argv[1:1] = ["-m", "coverage", "run", "--rcfile=" + config]
    return subprocess.run(argv, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=90)


def test_private_cli_publication_and_cold_drift(tmp_path):
    """SCENARIO-REPORT-8022-SEAL: scripted private bytes cannot claim live readiness."""
    fixture = tmp_path / "fixture.json"
    atomic_json(fixture, dict(plan=dict(checks=[], references=[], public=public()), result={}))
    output = tmp_path / "results" / (e.NAME + ".json")
    run = cli(["--fixture", fixture, "--output", output], tmp_path)
    assert run.returncode == 0, run.stdout + run.stderr
    value = json.loads(output.read_text())
    assert value["verdict_class"] == "blocked"
    assert value["likelihood_protocol_ready_score"] == value["generated_tokens"] == 0
    assert cli(["--cold-replay", output], tmp_path).returncode == 0
    bad = copy.deepcopy(value)
    bad["generated_tokens"] = 1
    atomic_json(output, bad)
    assert cli(["--cold-replay", output], tmp_path).returncode == 1
    assert cli(["--fixture", fixture], tmp_path).returncode == 1
    assert cli(["--date", "20261001"], tmp_path).returncode == 2
    assert cli(["--runtime-child", tmp_path / "missing"], tmp_path).returncode == 1


def test_runtime_worker_owned_and_failure_gates(tmp_path, monkeypatch):
    """REQ-REPORT-8022: mock API exercises lease custody, not model evidence."""
    import llama_cpp
    from llama_cpp import llama_cpp as native
    from llama_cpp import llama_chat_format

    r = e.runtime
    model_path = tmp_path / "fixture.gguf"
    model_path.write_bytes(b"private model fixture")
    source = tmp_path / "public.json"
    atomic_json(source, public())
    monkeypatch.setattr(
        r, "cached_current_model", lambda: dict(hf_id=r.MODEL, model_path=str(model_path))
    )
    monkeypatch.setattr(r, "bounded", lambda fn, seconds: fn())
    monkeypatch.setattr(native, "llama_supports_gpu_offload", lambda: True)
    monkeypatch.setattr(r.s, "freeze_panel", lambda public, runtime: dict(complete=True))

    class Lease:
        document = {"phase": "preflight"}

        def owner_receipt(self):
            return dict(private=True)

        def transition(self, phase, **kwargs):
            self.document = dict(phase=phase)

        def release(self):
            return dict(released=True)

    monkeypatch.setattr(r.GpuLease, "acquire", lambda **kwargs: Lease())

    def commands(root, specs, **kwargs):
        name = specs[0].name
        return [
            dict(
                output_tail="0, private-GPU, 4, 24000"
                if name == "capacity"
                else "16000"
                if name == "resident"
                else "4",
                passed=True,
            )
        ]

    monkeypatch.setattr(r, "run_commands", commands)

    class Model(Tokens):
        metadata = {"tokenizer.chat_template": "private"}
        _logits_all = True
        n_tokens = 0
        scores = np.zeros((200, 257))

        def __init__(self, **kwargs):
            pass

        def token_eos(self):
            return 2

        def token_bos(self):
            return 1

        def n_vocab(self):
            return 257

        def reset(self):
            self.n_tokens = 0

        def eval(self, tokens):
            self.n_tokens = len(tokens)

        def close(self):
            pass

    class Formatter:
        def __init__(self, **kwargs):
            pass

        def __call__(self, **kwargs):
            from types import SimpleNamespace

            return SimpleNamespace(prompt=kwargs["messages"][0]["content"] + "<assistant>\n")

    monkeypatch.setattr(llama_chat_format, "Jinja2ChatFormatter", Formatter)
    monkeypatch.setattr(llama_cpp, "Llama", Model)
    for mode in ["success", "missing", "capacity", "load_error", "logits"]:
        output = tmp_path / mode / "capture.json"
        monkeypatch.setattr(
            r,
            "cached_current_model",
            lambda: None if mode == "missing" else dict(hf_id=r.MODEL, model_path=str(model_path)),
        )
        monkeypatch.setattr(native, "llama_supports_gpu_offload", lambda: mode != "capacity")

        def load(**kwargs):
            if mode == "load_error":
                raise RuntimeError("private_loader_failure")
            item = Model()
            item._logits_all = mode != "logits"
            return item

        monkeypatch.setattr(llama_cpp, "Llama", load)
        r.worker(source, output)
        result = json.loads(output.read_text())
        if mode == "success":
            assert result["qualification"]["forward_pass_counts"] == 8
            assert result["model_invocation_counts"]["model_loads_completed"] == 2
            assert result["cleanup"]["lease_released"]
        else:
            assert result["checks"] and not result["checks"][0]["passed"]
        assert result["model_invocation_counts"]["generation_calls_attempted"] == 0


def test_main_owned_checks_child_and_terminal_failures(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8022-SEAL: real orchestration branches retain failures."""
    frozen = s.freeze_panel(public(), Tokens(), dict(fit=1, tune=1, evaluation=1))
    result = dict(
        panel=frozen,
        qualification=dict(passed=True, forward_pass_counts=8, rows=[], scored_tokens=8),
        duration_s=3,
        model_invocation_counts=dict(
            e.ZERO_INVOCATION_COUNTS, model_loads_attempted=1, model_loads_completed=1
        ),
    )
    monkeypatch.setattr(
        e, "authenticate", lambda root: dict(checks=[], references=[], public=public())
    )
    monkeypatch.setattr(
        e, "commands", lambda scratch: [e.CommandSpec("private_owned", ("private",), "owned", 1)]
    )
    monkeypatch.setattr(e, "terminal", lambda path: dict(passed=True))
    monkeypatch.setattr(e, "reader_receipt", lambda *args, **kwargs: dict(passed=True))
    mode = "normal"

    def run(root, specs, **kwargs):
        if specs[0].name == "teacher_forced_preflight":
            target = Path(specs[0].argv[-1])
            if mode != "missing_capture":
                atomic_json(target, result)
            return [
                dict(
                    passed=mode != "missing_capture",
                    scope="runtime",
                    exit_code=int(mode == "missing_capture"),
                    log_path=str(tmp_path / "log"),
                    log_sha256="private",
                )
            ]
        if specs[0].name == "private_owned":
            config = Path(kwargs["extra_env"]["CARNOT_8022_COVERAGE_CONFIG"])
            atomic_json(
                config.parent / "coverage.json",
                dict(totals=dict(num_statements=1, covered_lines=1, percent_covered=100)),
            )
        return [dict(passed=mode != "cold_error", scope=specs[0].scope, exit_code=0)]

    monkeypatch.setattr(e, "run_commands", run)
    for mode in ["normal", "missing_capture", "cold_error", "reader_error"]:
        monkeypatch.setattr(
            e, "reader_receipt", lambda *args, **kwargs: dict(passed=mode != "reader_error")
        )
        output = tmp_path / mode / "results" / (e.NAME + ".json")
        code = e.main(["--output", str(output)])
        assert code == (1 if mode in {"cold_error", "reader_error"} else 0)
        if output.exists():
            value = json.loads(output.read_text())
            assert value["coverage_statement_counts"]["percent_covered"] == 100
            if mode == "normal":
                from scripts.adversarial_verify import _classify_current_task_inference_claim

                assert value["model_invocation_counts"]["generation"]["attempted"] == 0
                assert _classify_current_task_inference_claim(value)["state"] == "live_inference"
                assert e.main(["--output", str(output)]) == 0
                raw = output.parent / "raw" / output.stem
                atomic_json(raw / "public.json", {"changed": True})
                assert e.main(["--output", str(output)]) == 1
    monkeypatch.setattr(e.runtime, "worker", lambda *args: None)
    source = tmp_path / "public.json"
    atomic_json(source, public())
    assert e.main(["--runtime-child", str(source)]) == 0


def test_live_missing_inputs_cli_and_custody(tmp_path):
    """REQ-REPORT-8022: missing external bytes terminate; changed bytes reject replay."""
    output = tmp_path / "results" / (e.NAME + ".json")
    run = cli(["--root", tmp_path / "missing", "--output", output], tmp_path)
    assert run.returncode == 0, run.stdout + run.stderr
    value = json.loads(output.read_text())
    assert value["verdict_class"] == "blocked" and value["gate_check_summary"]
    e.replay(value)
    raw = Path(value["checkpoint_references"][0]["path"]).parent
    (raw / "public.json").write_text("changed")
    with pytest.raises(ValueError, match="hash:"):
        e.replay(value)


def test_tiny_fixture_error_and_duplicate_rejection(monkeypatch):
    """SCENARIO-REPORT-8022-TOKENS: normalization failures cannot earn readiness."""
    original = s.target_likelihood
    monkeypatch.setattr(s, "target_likelihood", lambda *args: dict(target_logprobs=[999]))
    with pytest.raises(ValueError, match="independent_normalization_fixture"):
        s.qualify(Tokens())
    monkeypatch.setattr(s, "target_likelihood", original)
