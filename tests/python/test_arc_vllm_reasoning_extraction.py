"""Regression coverage for scored vLLM answer-channel extraction.

All tests are CPU-only and exercise the real request normalization or extraction seam.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

from carnot.agentic import arc_executable_world_model as wm


REQ = "REQ-ARC-WMTE-10011"
REAL_CODE = """```python
def engine(grid, action, data=None):
    return grid
```"""
DRAFT_CODE = """```python
def engine(grid, action, data=None):
    return None
```"""


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = json.dumps(payload).encode()

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False


def _chat(
    monkeypatch: pytest.MonkeyPatch,
    message: dict,
    *,
    repeat_penalty: float | None = None,
    vllm: bool = True,
    finish_reason: str = "stop",
    think_mode: bool = False,
) -> tuple[wm.LocalGGUFProposer, dict, str, dict]:
    if vllm:
        monkeypatch.setenv("CARNOT_ARC_LLM_BACKEND", "vllm")
    else:
        monkeypatch.delenv("CARNOT_ARC_LLM_BACKEND", raising=False)
    captured: dict = {}
    raw = {
        "choices": [{"message": message, "finish_reason": finish_reason}],
        "usage": {"completion_tokens": 17},
    }

    def _urlopen(request, **_kwargs):
        captured.update(json.loads(request.data))
        return _FakeResponse(raw)

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
    proposer = wm.LocalGGUFProposer()
    normalized, extraction = proposer._chat_complete_request(
        "prompt",
        max_tokens=128,
        temperature=0.2,
        stop=None,
        repeat_penalty=repeat_penalty,
        repeat_last_n=256,
        think_mode=think_mode,
    )
    return proposer, normalized, extraction, captured


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _install_vllm_source(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, source: str) -> Path:
    package = tmp_path / "vllm"
    reasoning = package / "reasoning"
    reasoning.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    init = reasoning / "__init__.py"
    init.write_text(source, encoding="utf-8")
    spec = SimpleNamespace(
        origin=str(package / "__init__.py"),
        submodule_search_locations=[str(package)],
    )
    monkeypatch.setattr("importlib.util.find_spec", lambda name: spec if name == "vllm" else None)
    return init


def test_vllm_content_extracts_answer_after_last_think_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-ARC-WMTE-10011-ANSWER: the final fence wins over a reasoning draft."""
    content = f"analysis</think>stale draft\n{DRAFT_CODE}\n</think>\n{REAL_CODE}"
    proposer, normalized, extraction, _ = _chat(monkeypatch, {"content": content})

    assert "return grid" in wm._extract_python(extraction)
    assert "return None" not in wm._extract_python(extraction)
    assert normalized["content"] == content
    assert proposer.channel_totals["chars_think_stripped"] == len(content) - len(extraction)


def test_vllm_unclosed_think_has_empty_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-ARC-WMTE-10011-UNCLOSED: real untagged truncation yields no code."""
    content = f"drafting\n{DRAFT_CODE}"
    _proposer, normalized, extraction, _ = _chat(
        monkeypatch,
        {"content": content},
        finish_reason="length",
        think_mode=True,
    )

    assert extraction == ""
    assert wm._extract_python(extraction) == ""
    assert normalized["content"] == content


def test_vllm_parser_split_answer_does_not_require_a_closing_tag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-ARC-WMTE-10011: parser-split final content is already an answer channel."""
    _proposer, _normalized, extraction, _ = _chat(
        monkeypatch,
        {"content": REAL_CODE, "reasoning": "private reasoning"},
        think_mode=True,
    )

    assert extraction == REAL_CODE


def test_vllm_stray_think_after_answer_keeps_preceding_engine() -> None:
    """REQ-ARC-WMTE-10011: a later stray opener cannot erase an earlier valid answer."""
    original = REAL_CODE + "\n<think>second reasoning segment"
    answer, stripped = wm._vllm_answer_text(original)

    assert wm._extract_python(answer) == wm._extract_python(REAL_CODE)
    assert stripped == len(original) - len(REAL_CODE + "\n")


def test_vllm_no_think_tags_is_byte_identical() -> None:
    """SCENARIO-ARC-WMTE-10011-UNCLOSED: untagged content is byte-identical."""
    original = "\x00  leading\n" + REAL_CODE + "\ntrailing \udcff"
    answer, stripped = wm._vllm_answer_text(original)

    assert answer == original
    assert answer.encode("utf-8", "surrogatepass") == original.encode("utf-8", "surrogatepass")
    assert stripped == 0


@pytest.mark.parametrize("field", ["reasoning", "reasoning_content"])
def test_vllm_reads_both_reasoning_field_names(monkeypatch: pytest.MonkeyPatch, field: str) -> None:
    """SCENARIO-ARC-WMTE-10011-CHANNELS: both compatible reasoning fields are read."""
    proposer, normalized, extraction, _ = _chat(
        monkeypatch, {"content": REAL_CODE, field: "private chain"}
    )

    assert proposer.last_reasoning_content == "private chain"
    assert "private chain" in normalized["content"]
    assert extraction == REAL_CODE


def test_registered_qwen3_parser_reaches_launch_argv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-ARC-WMTE-10011-PARSER: confirmed qwen3 registration enables the flag."""
    source = '_REASONING_PARSERS_TO_REGISTER = {"qwen3": ("module", "Parser")}\n'
    init = _install_vllm_source(monkeypatch, tmp_path, source)
    proposer = wm.LocalGGUFProposer(port=9123)

    argv = proposer._build_vllm_launch_argv("/model")

    index = argv.index("--reasoning-parser")
    assert argv[index + 1] == "qwen3"
    assert str(init) in proposer.last_vllm_reasoning_parser_decision
    assert list(proposer.last_launch_argv) == argv
    monkeypatch.setattr(proposer, "_healthy", lambda: False)
    assert (
        proposer.liveness_witness()["generator_vllm_reasoning_parser_decision"]
        == proposer.last_vllm_reasoning_parser_decision
    )


def test_unregistered_qwen3_parser_is_omitted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-ARC-WMTE-10011-PARSER: an absent registration omits the risky flag."""
    _install_vllm_source(
        monkeypatch,
        tmp_path,
        '_REASONING_PARSERS_TO_REGISTER = {"other": ("module", "Parser")}\n',
    )
    proposer = wm.LocalGGUFProposer(port=9123)

    argv = proposer._build_vllm_launch_argv("/model")

    assert "--reasoning-parser" not in argv
    assert "not_registered" in proposer.last_vllm_reasoning_parser_decision
    assert list(proposer.last_launch_argv) == argv


def test_reasoning_parser_check_failure_is_fail_safe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-ARC-WMTE-10011-PARSER: inspection failure omits the flag and records why."""
    monkeypatch.setattr(
        "importlib.util.find_spec", lambda _name: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    proposer = wm.LocalGGUFProposer(port=9123)

    argv = proposer._build_vllm_launch_argv(str(tmp_path / "model"))

    assert "--reasoning-parser" not in argv
    assert "check_failed:RuntimeError" in proposer.last_vllm_reasoning_parser_decision
    assert list(proposer.last_launch_argv) == argv


def test_vllm_repetition_penalty_unset_keeps_payload_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-ARC-WMTE-10011-SAMPLING: unset means no vLLM sampling intervention."""
    monkeypatch.delenv("CARNOT_ARC_VLLM_REPETITION_PENALTY", raising=False)
    monkeypatch.delenv("CARNOT_ARC_GENERATOR_SEED", raising=False)
    monkeypatch.delenv("CARNOT_ARC_INDUCE_THINKING_BUDGET", raising=False)
    proposer, _normalized, _extraction, payload = _chat(
        monkeypatch, {"content": REAL_CODE}, repeat_penalty=1.1
    )

    assert payload == {
        "messages": [{"role": "user", "content": "prompt"}],
        "max_tokens": 128,
        "temperature": 0.2,
        "cache_prompt": True,
        "repeat_penalty": 1.1,
        "repeat_last_n": 256,
    }
    assert proposer.vllm_request_sampling_receipts[-1] == {
        "repetition_penalty": None,
        "reason": "unset",
    }


def test_vllm_repetition_penalty_opt_in_reaches_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-ARC-WMTE-10011-SAMPLING: a configured float is sent and recorded."""
    monkeypatch.setenv("CARNOT_ARC_VLLM_REPETITION_PENALTY", "1.125")
    proposer, _normalized, _extraction, payload = _chat(
        monkeypatch, {"content": REAL_CODE}, repeat_penalty=1.1
    )

    assert payload["repetition_penalty"] == pytest.approx(1.125)
    assert payload["repeat_penalty"] == 1.1 and payload["repeat_last_n"] == 256
    assert proposer.vllm_request_sampling_receipts[-1] == {
        "repetition_penalty": 1.125,
        "reason": "env",
    }
    monkeypatch.setattr(proposer, "_healthy", lambda: False)
    assert proposer.liveness_witness()["generator_vllm_request_sampling_receipts"][-1] == {
        "repetition_penalty": 1.125,
        "reason": "env",
    }


def test_vllm_sampling_receipt_is_bounded_with_request_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-ARC-WMTE-10011-SAMPLING: repeated calls do not grow witness rows."""
    monkeypatch.setenv("CARNOT_ARC_LLM_BACKEND", "vllm")
    monkeypatch.delenv("CARNOT_ARC_VLLM_REPETITION_PENALTY", raising=False)
    raw = {
        "choices": [{"message": {"content": REAL_CODE}, "finish_reason": "stop"}],
        "usage": {"completion_tokens": 17},
    }
    monkeypatch.setattr(urllib.request, "urlopen", lambda *_args, **_kwargs: _FakeResponse(raw))
    proposer = wm.LocalGGUFProposer()

    for _ in range(40):
        proposer._chat_complete_request("prompt", max_tokens=128, temperature=0.2, stop=None)

    assert proposer.vllm_request_sampling_receipts == [
        {"repetition_penalty": None, "reason": "unset"}
    ]
    assert proposer.vllm_request_count == 40
    monkeypatch.setattr(proposer, "_healthy", lambda: False)
    witness = proposer.liveness_witness()
    assert witness["generator_vllm_request_count"] == 40
    assert len(witness["generator_vllm_request_sampling_receipts"]) == 1


def test_vllm_raw_completion_receives_opt_in_repetition_penalty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-ARC-WMTE-10011-SAMPLING: think-OFF raw calls honor the same opt-in."""
    monkeypatch.setenv("CARNOT_ARC_VLLM_REPETITION_PENALTY", "1.125")
    captured: dict = {}
    raw = {
        "choices": [{"text": "answer", "finish_reason": "stop"}],
        "usage": {"completion_tokens": 3},
    }

    def _urlopen(request, **_kwargs):
        captured.update(json.loads(request.data))
        return _FakeResponse(raw)

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
    proposer = wm.LocalGGUFProposer()

    proposer._vllm_raw_completion({"prompt": "prompt", "n_predict": 8, "temperature": 0.0})

    assert captured["repetition_penalty"] == pytest.approx(1.125)
    assert proposer.vllm_request_count == 1
    assert proposer.vllm_request_sampling_receipts == [
        {"repetition_penalty": 1.125, "reason": "env"}
    ]


@pytest.mark.parametrize("bad", ["nope", "nan", "inf", "0", "-1"])
def test_vllm_repetition_penalty_invalid_value_is_omitted(
    monkeypatch: pytest.MonkeyPatch, bad: str
) -> None:
    """REQ-ARC-WMTE-10011: an invalid penalty cannot break or alter a scored request."""
    monkeypatch.setenv("CARNOT_ARC_VLLM_REPETITION_PENALTY", bad)
    proposer, _normalized, _extraction, payload = _chat(monkeypatch, {"content": REAL_CODE})

    assert "repetition_penalty" not in payload
    assert proposer.vllm_request_sampling_receipts[-1]["repetition_penalty"] is None
    assert proposer.vllm_request_sampling_receipts[-1]["reason"].startswith("invalid:")


def test_vllm_blocks_reasoning_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-ARC-WMTE-10011-CHANNELS: vLLM never extracts a reasoning draft."""
    monkeypatch.setenv("CARNOT_ARC_CHAT_EMPTY_CONTENT_FALLBACK", "1")
    _proposer, _normalized, extraction, _ = _chat(
        monkeypatch,
        {"content": "", "reasoning": DRAFT_CODE},
        think_mode=True,
    )

    assert extraction == ""


def test_vllm_blocks_llamacpp_assistant_prefill_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-ARC-WMTE-10011-CHANNELS: vLLM never sends the incompatible retry."""
    monkeypatch.setenv("CARNOT_ARC_CHAT_FORCE_ANSWER_CONTINUATION", "1")
    _proposer, _normalized, extraction, payload = _chat(
        monkeypatch,
        {"content": "", "reasoning": "still planning"},
        think_mode=True,
    )

    assert extraction == ""
    assert payload["messages"] == [{"role": "user", "content": "prompt"}]


def test_reused_vllm_server_inherits_kernel_launch_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-ARC-WMTE-10011-PARSER: reuse keeps the probe's decision and argv."""
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    argv = ["python", "-m", "vllm.entrypoints.openai.api_server", "--reasoning-parser", "qwen3"]
    decision = "registered:qwen3 source=/mounted/vllm/reasoning/__init__.py"
    monkeypatch.setenv("CARNOT_ARC_VLLM_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("CARNOT_ARC_VLLM_REUSED_REASONING_PARSER_DECISION", decision)
    monkeypatch.setenv("CARNOT_ARC_VLLM_REUSED_LAUNCH_ARGV", json.dumps(argv))
    proposer = wm.LocalGGUFProposer()
    monkeypatch.setattr(proposer, "_vllm_healthy", lambda: True)
    monkeypatch.setattr(proposer, "_vllm_reusable", lambda _model_dir: True)

    assert proposer._ensure_vllm_server() is True
    assert proposer.last_vllm_reasoning_parser_decision == decision
    assert list(proposer.last_launch_argv) == argv


def test_raw_vllm_completion_sanitizes_before_extracting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-ARC-WMTE-10011-ANSWER: raw vLLM completion uses the same answer rule."""
    monkeypatch.setenv("CARNOT_ARC_LLM_BACKEND", "vllm")
    monkeypatch.setenv("CARNOT_ARC_INDUCE_THINK", "0")
    proposer = wm.LocalGGUFProposer(tries=1, use_chat_template=False)
    content = f"draft\n{DRAFT_CODE}\n</think>\n{REAL_CODE}"
    monkeypatch.setattr(proposer, "_ensure_server", lambda: True)
    monkeypatch.setattr(proposer, "observed_n_ctx", lambda: None)
    monkeypatch.setattr(
        proposer,
        "_vllm_raw_completion",
        lambda _payload: {
            "content": content,
            "stop_type": "eos",
            "truncated": False,
            "timings": {"predicted_n": 20},
        },
    )

    ok, code = proposer.generate("prompt", required=("engine",), tries=1)

    assert ok is True and "return grid" in code and "return None" not in code
    assert proposer.last_raw_completion == content
    assert proposer.channel_totals["chars_think_stripped"] == len(content) - len(f"\n{REAL_CODE}")


def test_llamacpp_chat_extraction_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-ARC-WMTE-10011: llama.cpp still passes tagged content through unchanged."""
    monkeypatch.delenv("CARNOT_ARC_LLM_BACKEND", raising=False)
    content = f"draft\n{DRAFT_CODE}\n</think>\n{REAL_CODE}"
    _proposer, normalized, extraction, _ = _chat(
        monkeypatch, {"content": content, "reasoning": "vLLM-only alias"}, vllm=False
    )

    assert normalized["content"] == content
    assert extraction == content
    assert "return None" in wm._extract_python(extraction)


def test_saved_pilot_calls_match_llamacpp_engine_hashes(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-10011-PILOT: all eight saved calls recover the final engine."""
    assert tmp_path.is_dir()  # pytest owns every writable location used by this regression.
    root = Path(__file__).resolve().parents[2]
    calls = sorted(
        (root / "results/raw/experiment_10010_b2_think_on_pilot/calls").glob("*/*/call*.json")
    )
    assert len(calls) == 8

    for path in calls:
        saved = json.loads(path.read_text(encoding="utf-8"))
        reasoning = saved["reasoning_content"]
        final = saved["final_content"]
        vllm_content = f"{reasoning}\n</think>\n\n{final}"
        fixed = wm._extract_python(wm._vllm_answer_text(vllm_content)[0])
        llamacpp = wm._extract_python(final)
        assert _sha(fixed) == _sha(llamacpp), path
