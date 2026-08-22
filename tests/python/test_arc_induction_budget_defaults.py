"""REQ-ARC-WMTE-6620 / REQ-ARC-WMTE-6630: the induce budget moves with the generator pin.

Origin: the 2026-08-21 supervisor A/B baseline. Five public games, budget 2000, and ZERO
world models -- every induce response decoded exactly 4096 tokens (the 9B-era default cap)
against a generator whose median induction is 62,490 tokens. The scored kernel pins the
correct values via env, so Kaggle worked while every env-less local run failed 100% of
inductions, silently.

No GPU, no server, no instantiation of LocalGGUFProposer (its __post_init__ runs a
VRAM-dependent offload fit): the wiring is pinned through dataclass field factories and
the pure resolvers.

Spec refs: REQ-ARC-WMTE-6620 (SCENARIO-ARC-WMTE-6620-1..3), REQ-ARC-WMTE-6630
(SCENARIO-ARC-WMTE-6630-1..2).
"""

from __future__ import annotations

import dataclasses
import inspect
import re
from pathlib import Path
from types import SimpleNamespace

from carnot.agentic.arc_executable_world_model import (
    ARC_LIVE_GENERATOR_INDUCE_MAX_TOKENS_DEFAULT,
    ARC_LIVE_GENERATOR_INDUCE_TIMEOUT_FLOOR_S,
    _INDUCE_WORST_CASE_PROMPT_TOKENS,
    LocalGGUFProposer,
    _default_induce_timeout_s,
    _induce_max_tokens_default,
    _pool_clamped_n_predict,
)
from carnot.agentic.arc_llm_reinduction import _model_specs

_REPO_ROOT = Path(__file__).resolve().parents[2]


# ---- SCENARIO-ARC-WMTE-6620-1: defaults follow the pin, env still wins -----------------


def test_max_tokens_default_is_the_generator_pin(monkeypatch) -> None:
    """Unset env resolves to the pin constant -- 131072, not the 9B-era 4096."""

    monkeypatch.delenv("CARNOT_ARC_INDUCE_MAX_TOKENS", raising=False)
    assert _induce_max_tokens_default() == ARC_LIVE_GENERATOR_INDUCE_MAX_TOKENS_DEFAULT
    assert ARC_LIVE_GENERATOR_INDUCE_MAX_TOKENS_DEFAULT == 131072


def test_max_tokens_env_override_wins(monkeypatch) -> None:
    monkeypatch.setenv("CARNOT_ARC_INDUCE_MAX_TOKENS", "8192")
    assert _induce_max_tokens_default() == 8192


def test_max_tokens_malformed_env_falls_back(monkeypatch) -> None:
    """A malformed env var must degrade to the pin default, never crash a live episode."""

    monkeypatch.setenv("CARNOT_ARC_INDUCE_MAX_TOKENS", "not-a-number")
    assert _induce_max_tokens_default() == ARC_LIVE_GENERATOR_INDUCE_MAX_TOKENS_DEFAULT


def test_dataclass_defaults_are_the_shared_resolvers() -> None:
    """Both construction sites and the lever harness OMIT max_tokens/timeout, so the
    dataclass factories ARE the single default. Pin their identity: re-introducing a
    literal here re-opens the drift REQ-ARC-FCP-5699-35 records."""

    fields = {f.name: f for f in dataclasses.fields(LocalGGUFProposer)}
    assert fields["max_tokens"].default_factory is _induce_max_tokens_default
    assert fields["timeout"].default_factory is _default_induce_timeout_s


def test_timeout_floor_is_the_pin_constant(monkeypatch) -> None:
    """Floor raised 600 -> 2400 for the Qwen3.8 pin; env override untouched."""

    monkeypatch.delenv("CARNOT_ARC_INDUCE_TIMEOUT", raising=False)
    assert ARC_LIVE_GENERATOR_INDUCE_TIMEOUT_FLOOR_S == 2400
    assert _default_induce_timeout_s() >= ARC_LIVE_GENERATOR_INDUCE_TIMEOUT_FLOOR_S
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TIMEOUT", "77")
    assert _default_induce_timeout_s() == 77


# ---- SCENARIO-ARC-WMTE-6620-2: kernel parity is pinned ----------------------------------


def test_scored_kernel_env_pins_equal_the_constants() -> None:
    """The scored kernel's setdefault literals and the pin constants must stay equal, or
    the scored and local configurations drift apart again -- the exact shape of the
    original incident, pointed the other way."""

    kernel = (_REPO_ROOT / "scripts" / "kaggle" / "submission_kernel" / "main.py").read_text()
    m = re.search(r'setdefault\("CARNOT_ARC_INDUCE_MAX_TOKENS", "(\d+)"\)', kernel)
    assert m is not None, "kernel no longer pins CARNOT_ARC_INDUCE_MAX_TOKENS"
    assert int(m.group(1)) == ARC_LIVE_GENERATOR_INDUCE_MAX_TOKENS_DEFAULT
    t = re.search(r'setdefault\("CARNOT_ARC_INDUCE_TIMEOUT", "(\d+)"\)', kernel)
    assert t is not None, "kernel no longer pins CARNOT_ARC_INDUCE_TIMEOUT"
    assert int(t.group(1)) == ARC_LIVE_GENERATOR_INDUCE_TIMEOUT_FLOOR_S


# ---- SCENARIO-ARC-WMTE-6620-3: pool clamp ------------------------------------------------


def test_pool_clamp_fits_the_local_pool() -> None:
    """131072 against the local 106,496-cell pool clamps to the single-stream room.
    llama-server admission counts prompt + n_predict, so the unclamped request is refused
    outright (HTTP 500), not truncated."""

    assert _pool_clamped_n_predict(131072, 106496) == 106496 - _INDUCE_WORST_CASE_PROMPT_TOKENS
    assert _pool_clamped_n_predict(131072, 106496) == 84144


def test_pool_clamp_is_a_noop_on_large_or_unknown_pools() -> None:
    assert _pool_clamped_n_predict(131072, 614400) == 131072  # scored llama.cpp pool
    assert _pool_clamped_n_predict(131072, None) == 131072  # vLLM serves no /props
    assert _pool_clamped_n_predict(4096, 106496) == 4096  # small budgets pass through


def test_pool_clamp_degenerate_pool_passes_through() -> None:
    """A pool smaller than the worst-case prompt, or one leaving only a sliver of room,
    passes through unclamped: the server's loud admission refusal beats a "successful"
    few-token completion that looks like a healthy-but-terse model (exp5866's mode C)."""

    assert _pool_clamped_n_predict(131072, 10000) == 131072  # pool < worst-case prompt
    assert _pool_clamped_n_predict(131072, _INDUCE_WORST_CASE_PROMPT_TOKENS + 8) == 131072
    assert (  # exactly at the floor: clamp applies
        _pool_clamped_n_predict(131072, _INDUCE_WORST_CASE_PROMPT_TOKENS + 1024) == 1024
    )


def test_generate_and_complete_text_send_the_clamped_budget() -> None:
    """Wiring pin: both request builders route their n_predict through the clamp. Deleting
    either clamp line (mutation) turns exactly one of these assertions red."""

    gen_src = inspect.getsource(LocalGGUFProposer.generate)
    assert "_pool_clamped_n_predict(int(self.max_tokens), self.observed_n_ctx())" in gen_src
    assert '"n_predict": _n_predict' in gen_src
    assert 'max_tokens=_payload["n_predict"]' in gen_src  # think-mode chat branch
    # Whitespace-insensitive: ruff may wrap the call across lines without changing meaning.
    ct_nows = "".join(inspect.getsource(LocalGGUFProposer.complete_text).split())
    assert (
        "_pool_clamped_n_predict(int(max_tokensorself.max_tokens),self.observed_n_ctx())" in ct_nows
    )
    assert '"n_predict":_n_predict' in ct_nows


# ---- REQ-ARC-WMTE-6630: model label derives from the weights actually loaded ------------


def test_label_names_the_configured_weights_file_not_the_repo_pin() -> None:
    """SCENARIO-ARC-WMTE-6630-1: a frozen 9B repo pin under a 27B path override must be
    labelled by the 27B file. The old form produced
    'Qwen3.5-9B-MTP GGUF (.../Qwen3.8-27B-Q4_K_M.gguf)' -- a label contradicting its path."""

    proposer = SimpleNamespace(
        repo_substr="Qwen3.5-9B-MTP",
        model_path="/models/Qwen3.8-27B-Q4_K_M.gguf",
    )
    assert _model_specs(proposer) == "Qwen3.8-27B-Q4_K_M GGUF (/models/Qwen3.8-27B-Q4_K_M.gguf)"


def test_label_prefers_the_running_servers_own_report() -> None:
    """SCENARIO-ARC-WMTE-6630-2: observed_model_path() is the only channel that catches a
    stale server on the port; when readable it wins over the configured path."""

    proposer = SimpleNamespace(
        repo_substr="Qwen3.5-9B-MTP",
        model_path="/models/Qwen3.8-27B-Q4_K_M.gguf",
        observed_model_path=lambda: "/srv/other-model.gguf",
    )
    assert _model_specs(proposer) == "other-model GGUF (/srv/other-model.gguf)"


def test_label_falls_back_when_the_observer_raises_or_is_empty() -> None:
    def _boom() -> str:
        raise RuntimeError("props unreachable")

    proposer = SimpleNamespace(repo_substr="pin", model_path="/m.gguf", observed_model_path=_boom)
    assert _model_specs(proposer) == "m GGUF (/m.gguf)"
    bare = SimpleNamespace(repo_substr="pin-only", observed_model_path=lambda: None)
    assert _model_specs(bare) == "pin-only"


def test_label_explicit_model_specs_attribute_keeps_precedence() -> None:
    proposer = SimpleNamespace(
        model_specs="explicit-label",
        model_path="/m.gguf",
        observed_model_path=lambda: "/srv/other.gguf",
    )
    assert _model_specs(proposer) == "explicit-label"
