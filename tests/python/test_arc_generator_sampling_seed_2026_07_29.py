"""REQ-ARC-WMTE-6046: the generator's sampler seed is settable, and OFF by default.

THE INCIDENT. Every generation `LocalGGUFProposer` issues goes out at
``temperature = 0.2 + 0.1*attempt`` -- nonzero -- with NO ``seed`` field. `llama-server` treats an
absent seed as -1, "pick a fresh random one", so two runs of IDENTICAL code on the identical game
with the identical harness `seed` produce different LLM output. The harness `seed` seeds
`random`/`numpy` in the driver; it never reached the server's sampler.

Measured cost: comparing two runs that share treatment, seed, model file and game
(`results/arc_engine_retention_20260729/cells` `ret1` vs
`results/arc_heldout_31b_vs_9b_20260728/cells` `31b`), 2 of 5 cells diverge under identical code --
40%. That floor is at least as large as any treatment effect measured on this path, which is why
three A/B measurements on 2026-07-29 returned uninformative nulls.

Live-server verification (2026-07-29, gemma-4-31B-it on port 8171, T=0.4, a deliberately
high-entropy prompt): 4 identical UNSEEDED requests produced 3 distinct completions; 4 identical
SEEDED requests produced 1. These tests cover the plumbing and the default; they do not re-run the
live check, which needs a loaded 21 GiB server.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest

from carnot.agentic.arc_executable_world_model import LocalGGUFProposer


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts from "operator has not opted in", the shipped state."""
    monkeypatch.delenv("CARNOT_ARC_GENERATOR_SEED", raising=False)


# ---- SCENARIO-ARC-WMTE-6046-1: default OFF means byte-identical to before -----------------
def test_unset_returns_none_so_the_payload_is_unchanged() -> None:
    """The whole safety argument rests on this: no env var, no behaviour change."""
    assert LocalGGUFProposer.sampling_seed(0) is None
    assert LocalGGUFProposer.sampling_seed(7) is None


def test_empty_string_is_treated_as_unset() -> None:
    """`CARNOT_ARC_GENERATOR_SEED=` in a systemd drop-in must not mean "seed 0"."""
    os.environ["CARNOT_ARC_GENERATOR_SEED"] = ""
    assert LocalGGUFProposer.sampling_seed(0) is None
    os.environ["CARNOT_ARC_GENERATOR_SEED"] = "   "
    assert LocalGGUFProposer.sampling_seed(0) is None


def test_malformed_value_falls_back_to_none_rather_than_raising() -> None:
    """A typo'd env var must not take down a live episode mid-run."""
    os.environ["CARNOT_ARC_GENERATOR_SEED"] = "not-an-int"
    assert LocalGGUFProposer.sampling_seed(0) is None
    os.environ["CARNOT_ARC_GENERATOR_SEED"] = "3.5"
    assert LocalGGUFProposer.sampling_seed(0) is None


# ---- SCENARIO-ARC-WMTE-6046-2: the seed varies with the attempt --------------------------
def test_seed_varies_with_attempt_so_the_retry_ladder_still_explores() -> None:
    """A single fixed seed would flatten the `0.2 + 0.1*attempt` diversity ladder.

    The ladder exists so a failed induction is retried with MORE diversity. Pinning one seed
    across attempts would make attempt 2 far more correlated with attempt 1 than intended, so the
    seed must be per-attempt while the RUN as a whole stays reproducible.
    """
    os.environ["CARNOT_ARC_GENERATOR_SEED"] = "7"
    seeds = [LocalGGUFProposer.sampling_seed(a) for a in range(4)]
    assert seeds == [7000, 7001, 7002, 7003]
    assert len(set(seeds)) == 4  # every attempt distinct
    # And reproducible: the same base + attempt always gives the same seed.
    assert LocalGGUFProposer.sampling_seed(2) == 7002


def test_distinct_base_seeds_do_not_collide_across_attempts() -> None:
    """base*1000 leaves room for far more attempts than the ladder will ever run.

    A naive `base + attempt` would make base=7 attempt=1 collide with base=8 attempt=0, silently
    aliasing two runs that a comparison is trying to keep apart.
    """
    os.environ["CARNOT_ARC_GENERATOR_SEED"] = "7"
    a = {LocalGGUFProposer.sampling_seed(i) for i in range(50)}
    os.environ["CARNOT_ARC_GENERATOR_SEED"] = "8"
    b = {LocalGGUFProposer.sampling_seed(i) for i in range(50)}
    assert not (a & b)


def test_negative_base_is_accepted_because_minus_one_is_llama_cpp_for_random() -> None:
    """-1 is llama.cpp's own "random" sentinel, so it must remain expressible."""
    os.environ["CARNOT_ARC_GENERATOR_SEED"] = "-1"
    assert LocalGGUFProposer.sampling_seed(0) == -1000


# ---- SCENARIO-ARC-WMTE-6046-3: every generation path honours it ---------------------------
class _CapturedRequest:
    """Records the payload of each outbound request instead of sending it."""

    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    def urlopen(self, req: Any, timeout: float | None = None) -> Any:  # noqa: ANN401
        self.payloads.append(json.loads(req.data.decode()))

        class _Resp:
            def __enter__(_s) -> Any:  # noqa: ANN001, N805
                return _s

            def __exit__(_s, *a: Any) -> bool:  # noqa: ANN001, N805
                return False

            def read(_s) -> bytes:  # noqa: ANN001, N805
                return b'{"content": "ok", "choices": [{"message": {"content": "ok"}}]}'

        return _Resp()


def _proposer_with_captured_http(monkeypatch: pytest.MonkeyPatch) -> tuple:
    """A proposer whose server checks pass and whose HTTP is captured, so no GPU is needed.

    `complete_text` is the narrowest of the three generation paths -- it does not gate on code
    extraction -- which makes it the cheapest place to assert the payload shape. The other two
    paths are asserted by source contract in the companion test below.
    """
    import urllib.request

    cap = _CapturedRequest()
    prop = LocalGGUFProposer(repo_substr="gemma-4-31B-it", port=65500)
    monkeypatch.setattr(prop, "_ensure_server", lambda: True)
    monkeypatch.setattr(urllib.request, "urlopen", cap.urlopen)
    # No need to patch json.load: the fake response exposes read(), which is all `json.load`
    # asks of it. Patching a stdlib function the code under test shares would be a broader
    # intervention than the test needs.
    return prop, cap


def test_completion_payload_omits_seed_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    prop, cap = _proposer_with_captured_http(monkeypatch)
    prop.complete_text("hello", max_tokens=8)
    assert cap.payloads, "no request was captured"
    assert "seed" not in cap.payloads[0]


def test_completion_payload_carries_the_seed_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    os.environ["CARNOT_ARC_GENERATOR_SEED"] = "11"
    prop, cap = _proposer_with_captured_http(monkeypatch)
    prop.complete_text("hello", max_tokens=8)
    assert cap.payloads[0]["seed"] == 11000


def test_all_three_generation_paths_consult_sampling_seed() -> None:
    """Source contract: no route out of this class may stay unseeded while another is seeded.

    An artifact that declares a seeded run while one of its generation paths silently sampled at
    random is worse than an unseeded run -- it is an unseeded run wearing a reproducibility claim.
    A test that only covered `complete_text` would have permitted exactly that, so the count of
    call sites is pinned here.
    """
    import inspect

    from carnot.agentic import arc_executable_world_model as e3

    src = inspect.getsource(e3.LocalGGUFProposer)
    # One definition plus one call per generation path: /completion in generate(),
    # /v1/chat/completions in _chat_complete_request(), and complete_text().
    assert src.count("self.sampling_seed(") == 3, (
        "a generation path was added or removed without wiring the sampler seed"
    )
    # `payload["seed"]` is a substring of `_payload["seed"]`, so this one count covers both
    # spellings and equals the number of assignment sites. (A first draft summed the two counts
    # and double-counted the underscore spelling -- 4 where the truth was 3.)
    assert src.count('payload["seed"]') == 3


def test_the_chat_template_route_gets_the_per_attempt_seed_too() -> None:
    """`_chat_complete_request` must receive `attempt`, not silently reuse the base seed.

    Qwen3.6 / ThinkingCap take the chat-template route. If that route reused one seed across the
    whole retry ladder, the ladder's diversity would collapse on exactly the models that use it,
    while the /completion route kept its per-attempt seeds -- two different behaviours under one
    reproducibility claim.
    """
    import inspect

    from carnot.agentic import arc_executable_world_model as e3

    sig = inspect.signature(e3.LocalGGUFProposer._chat_complete_request)
    assert "attempt" in sig.parameters
    assert sig.parameters["attempt"].default == 0  # additive: existing callers are unaffected

    src = inspect.getsource(e3.LocalGGUFProposer)
    # The ladder in generate() must actually pass it through, not just accept it.
    assert "attempt=attempt," in src
    assert "self.sampling_seed(attempt)" in src


def test_seed_zero_is_honoured_and_not_read_as_unset() -> None:
    """The classic falsy-zero bug: `CARNOT_ARC_GENERATOR_SEED=0` must NOT mean "off".

    An operator who pins seed 0 is asking for determinism. A truthiness test on the parsed int
    (`if not base: return None`) would silently give them the nondeterministic path while their
    artifact claimed a seeded run -- the worst of both. The guard is on the RAW STRING being
    empty, not on the parsed value, precisely so 0 survives.
    """
    os.environ["CARNOT_ARC_GENERATOR_SEED"] = "0"
    assert LocalGGUFProposer.sampling_seed(0) == 0
    assert LocalGGUFProposer.sampling_seed(3) == 3
    assert LocalGGUFProposer.sampling_seed(0) is not None


def test_surrounding_whitespace_on_a_real_value_is_tolerated() -> None:
    """A systemd drop-in or a shell export can easily leave a stray space."""
    os.environ["CARNOT_ARC_GENERATOR_SEED"] = " 12 "
    assert LocalGGUFProposer.sampling_seed(1) == 12001
