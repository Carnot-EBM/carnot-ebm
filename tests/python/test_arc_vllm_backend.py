"""REQ-ARC-WMTE-6510: the vLLM backend, a SECOND generator backend gated default-off.

WHY IT EXISTS. Measured on the scored Kaggle Blackwell card at production request shape
(22k prompt / 8192 generated, ignore_eos, every finish `length`): vLLM with native SM120 cutlass
FP4 and fp8 KV reached 651.8 tok/s aggregate at k=32, against 228.3 for the best llama.cpp
config and ~52 for what shipped. Single-stream is a wash (56.1 vs 52.2) -- the entire win is
continuous batching plus native FP4 at concurrency.

WHY IT IS A SECOND BACKEND AND NOT A REPLACEMENT. The dev 3090s are sm_86 and cannot execute
NVFP4 at all, so llama.cpp remains the local and conductor path. That is also why the ~60
existing test files that pin llama.cpp specifics stay valid unchanged: they pin the path that is
still the default.

Scenarios: SCENARIO-ARC-WMTE-6510-DEFAULTOFF (only the exact opt-in switches backends),
SCENARIO-ARC-WMTE-6510-RESHAPE (vLLM replies are reshaped to llama.cpp's response contract so
truncation detection is unchanged), SCENARIO-ARC-WMTE-6510-REUSE (a running server is refused,
never adopted, when it serves the wrong model or too small a pool),
SCENARIO-ARC-WMTE-6510-NOMTP (speculation is not silently believed on this backend).
"""

from __future__ import annotations

import json

import pytest

from carnot.agentic import arc_executable_world_model as wm


# SCENARIO-ARC-WMTE-6510-DEFAULTOFF
@pytest.mark.parametrize(
    "value,expected",
    [
        (None, False),
        ("", False),
        ("llamacpp", False),
        ("typo", False),
        ("vllm", True),
        ("VLLM", True),
    ],
)
def test_backend_switch_is_exact_opt_in(monkeypatch, value, expected) -> None:
    """A malformed env var must resolve to llama.cpp. The opposite default would let a typo
    silently migrate the scored generator to a backend the local cards cannot even run."""
    monkeypatch.delenv("CARNOT_ARC_LLM_BACKEND", raising=False)
    if value is not None:
        monkeypatch.setenv("CARNOT_ARC_LLM_BACKEND", value)
    assert wm._vllm_backend_active() is expected


def test_max_seqs_default_and_floor(monkeypatch) -> None:
    """24 is chosen from the measurement, not taste: fp8 KV fits ~22 concurrent sessions at
    capped production length in the ~61 GB KV budget, and the scaling curve is already bending by
    k=32 (16->32 gained only 1.35x). Malformed values fall back rather than raising."""
    monkeypatch.delenv("CARNOT_ARC_VLLM_MAX_SEQS", raising=False)
    assert wm._vllm_max_seqs() == 24
    monkeypatch.setenv("CARNOT_ARC_VLLM_MAX_SEQS", "8")
    assert wm._vllm_max_seqs() == 8
    monkeypatch.setenv("CARNOT_ARC_VLLM_MAX_SEQS", "0")
    assert wm._vllm_max_seqs() == 1
    monkeypatch.setenv("CARNOT_ARC_VLLM_MAX_SEQS", "garbage")
    assert wm._vllm_max_seqs() == 24


def test_model_dir_requires_a_real_config(monkeypatch, tmp_path) -> None:
    """The env pin is honoured only if it actually looks like a model dir. A stale or wrong pin
    must degrade to None so the caller records a server failure, rather than launching vLLM
    against a directory with no weights in it."""
    monkeypatch.setenv("CARNOT_ARC_VLLM_MODEL_DIR", str(tmp_path))
    assert wm._resolve_vllm_model_dir() is None  # no config.json
    (tmp_path / "config.json").write_text("{}")
    assert wm._resolve_vllm_model_dir() == str(tmp_path)


# SCENARIO-ARC-WMTE-6510-RESHAPE
def test_vllm_reply_is_reshaped_to_the_llamacpp_contract(monkeypatch) -> None:
    """`_record_completion_diagnostics` reads content/stop_type/truncated/timings.predicted_n.
    vLLM returns choices[0].text/finish_reason/usage.completion_tokens. If the translation drops
    any of those, truncation detection silently stops working -- the exact class of silent
    degradation this file's neighbours were written to catch."""
    p = wm.LocalGGUFProposer.__new__(wm.LocalGGUFProposer)
    p.port, p.timeout, p.max_tokens = 9, 30, 4096

    captured = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(
                {
                    "choices": [{"text": "ENGINE", "finish_reason": "length"}],
                    "usage": {"completion_tokens": 4096},
                }
            ).encode()

    import urllib.request

    def _fake(req, *a, **kw):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode())
        return _Resp()

    monkeypatch.setattr(urllib.request, "urlopen", _fake)
    out = p._vllm_raw_completion({"prompt": "P", "n_predict": 4096, "temperature": 0.2})

    assert captured["url"].endswith("/v1/completions"), "must not POST llama.cpp's /completion"
    assert captured["body"]["max_tokens"] == 4096, "n_predict must map to max_tokens"
    assert out["content"] == "ENGINE"
    assert out["stop_type"] == "limit", "finish_reason=length is llama.cpp's 'limit'"
    assert out["timings"]["predicted_n"] == 4096, "generated-token count feeds _limit_diagnostic"


def test_stop_maps_to_eos_not_limit(monkeypatch) -> None:
    """The two stop reasons drive DIFFERENT diagnostics: 'limit' means the cap truncated the
    answer, 'eos' means the model finished. Collapsing them would make every clean completion
    look truncated."""
    p = wm.LocalGGUFProposer.__new__(wm.LocalGGUFProposer)
    p.port, p.timeout, p.max_tokens = 9, 30, 4096

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(
                {
                    "choices": [{"text": "ok", "finish_reason": "stop"}],
                    "usage": {"completion_tokens": 12},
                }
            ).encode()

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: _Resp())
    assert p._vllm_raw_completion({"prompt": "P"})["stop_type"] == "eos"


# SCENARIO-ARC-WMTE-6510-REUSE
def _proposer_with_models_reply(monkeypatch, payload):
    p = wm.LocalGGUFProposer.__new__(wm.LocalGGUFProposer)
    p.port, p.timeout, p.max_tokens = 9, 30, 4096
    p.reuse_model_check = p.reuse_n_ctx_check = ""
    p.observed_server_n_ctx = None

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(payload).encode()

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: _Resp())
    return p


def test_reuse_refuses_a_server_running_the_wrong_model(monkeypatch) -> None:
    """Same policy as the llama.cpp `_reusable`: REFUSE and relaunch, never adopt. A stale server
    from a previous run satisfying a bare health check is how a run silently induces with the
    wrong model while its witness reports the intended one."""
    p = _proposer_with_models_reply(
        monkeypatch, {"data": [{"id": "/other/model", "max_model_len": 999999}]}
    )
    assert p._vllm_reusable("/want/model") is False
    assert "refused_wrong_model" in p.reuse_model_check


def test_reuse_refuses_a_smaller_pool_but_accepts_larger(monkeypatch) -> None:
    need = wm._INDUCE_WORST_CASE_PROMPT_TOKENS + 4096 + 2048
    p = _proposer_with_models_reply(monkeypatch, {"data": [{"id": "m", "max_model_len": need - 1}]})
    assert p._vllm_reusable("/want/model") is False
    assert "refused_smaller_pool" in p.reuse_n_ctx_check

    p2 = _proposer_with_models_reply(
        monkeypatch, {"data": [{"id": "m", "max_model_len": need + 10_000}]}
    )
    assert p2._vllm_reusable("/want/model") is True
    assert p2.reuse_n_ctx_check == "larger_ok"


def test_reuse_fails_open_when_v1_models_is_unreachable(monkeypatch) -> None:
    """Mirrors the llama.cpp path's /props-unreadable behaviour: a server that cannot answer the
    identity query is reused with a recorded warning rather than bricking the run."""
    p = wm.LocalGGUFProposer.__new__(wm.LocalGGUFProposer)
    p.port, p.timeout, p.max_tokens = 9, 30, 4096
    p.reuse_model_check = p.reuse_n_ctx_check = ""
    import urllib.request

    def _boom(*a, **kw):
        raise OSError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    assert p._vllm_reusable("/want/model") is True
    assert p.reuse_model_check == "unobserved_v1_models_unreachable"
