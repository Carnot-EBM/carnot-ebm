"""The live ARC generator is gemma-4-31B-it, at EVERY site on the live/induction path.

REQ-ARC-WMTE-6021 / SCENARIO-ARC-WMTE-6021-ONE-GENERATOR-PIN

WHY THIS FILE EXISTS. Operator directive 2026-07-28: "We must use Gemma-4-31B and stop using
Qwen-3.5-9B and Qwen-3.6-27B." The Qwen3.5-9B pin existed for exactly one reason -- an assumed
16 GB Kaggle VRAM ceiling that a 5.9 GB Q4 model fits -- and the operator has declared that
ceiling void. The head-to-head that drove it (13 games x 3 replicates, Q4_K_M both sides,
n_ctx 32768, one model per card):

                   induce_ok    fail-as-zero    survivor-mean    nonzero cells
    gemma-4-31B      38/39          0.3843          0.3944            23
    Qwen3.6-27B      21/39          0.0627          0.1164             4

    matched per-game tally 11-0-2, two-sided sign p = 0.00098 (the minimum reachable at 11
    discordant pairs). Dominant driver is LOADABILITY: the 27B failed to emit an importable
    world_model.py on 18 of 39 attempts, so fail-as-zero is the honest column and the survivor
    mean is survivorship-biased.

WHAT THIS FILE PROTECTS, beyond "the string changed". A generator swap has three failure modes
that are all silent, and each gets an assertion here:

  1. PARTIAL REVERT. The pin used to be a string literal repeated at six sites. Reverting one of
     them (or adding a seventh) does not raise -- it loads a SECOND model on a SECOND port. At
     5.9 GB that was waste; at 18.3 GB it is an OOM. So the tests assert every live site reads
     the SAME canonical constant, not merely that each says "gemma".
  2. MTP LEFT ON. gemma-4-31B-it is dense and declares no `nextn_predict_layers`. The live sites
     defaulted `CARNOT_ARC_MTP` to "1", which makes `_ensure_server()` emit
     `--spec-type draft-mtp --model-draft <the same 18.3 GB file>` -- the weights twice, ~36.6 GB,
     a guaranteed cudaMalloc failure that costs 180 s and ends in an LLM-OFF run still reporting
     itself as the LLM-on scored path.

     ^^^ PARTIALLY CORRECTED 2026-07-28 (same day, measured); preserved per never-prune because
     the CONCLUSION -- never pass the main weights as the draft -- is still exactly right and is
     now enforced in `_ensure_server()`. The PREMISE is wrong: gemma-4-31B-it DOES have MTP. Its
     head is a SEPARATE 491 MiB GGUF (`mtp-gemma-4-31B-it-Q8_0.gguf`, arch `gemma4-assistant`)
     rather than heads embedded in the main file, which is why none was found there. Enabling MTP
     with the real head costs +1290 MiB at the shipped n_ctx 81920 -- not a second copy of the
     weights -- and buys a measured 1.398x decode. The local default stays "0" because a 24 GB
     card must offload ~14 FFN blocks to fit it and that costs more than MTP returns; the SCORED
     default (`ARC_LIVE_GENERATOR_MTP_SCORED_DEFAULT`) is "1" because the 96 GB Kaggle card needs
     no offload at all. See tests/python/test_arc_generator_migration_defects.py.
  3. `/no_think` LEFT IN. That is a Qwen3 hybrid-thinking control token. Gemma-4 has no such
     token and consumes it as literal prompt text -- a dead channel that looks like a feature.

These tests are written to FAIL if anyone re-pins any live site back to a Qwen model.
"""

from __future__ import annotations


import pytest

# MODULE-LEVEL imports, deliberately. `arc_competition_agent` costs ~590 MB to import, and this
# repo's pytest memory watchdog (tests/python/conftest.py) attributes any growth between a test's
# setup and teardown to that test. Importing inside a fixture therefore fails every test in the
# file with a spurious "Memory leak: +587MB"; importing at collection time -- which is what every
# other agent test here does -- puts the cost outside the watchdog's measurement window.
from carnot.agentic import arc_competition_agent as agent  # noqa: E402
from carnot.agentic import arc_executable_world_model as wm  # noqa: E402
from carnot.agentic import arc_ige_cell_selector as sel_mod  # noqa: E402

# Any of these appearing as a live generator pin is a regression, by name.
RETIRED_GENERATOR_MARKERS = ("Qwen3.5-9B", "Qwen3.6-27B", "Qwen3.5-9B-MTP", "ThinkingCap-27B")


@pytest.fixture()
def mods(monkeypatch):
    """Clean generator-related env so an ambient override cannot mask a real regression."""
    monkeypatch.delenv("CARNOT_ARC_MTP", raising=False)
    monkeypatch.delenv("CARNOT_ARC_GGUF_PATH", raising=False)
    monkeypatch.delenv("CARNOT_ARC_INDUCE_N_CTX", raising=False)
    monkeypatch.delenv("CARNOT_ARC_FFN_CPU_LAYERS", raising=False)
    return agent, wm


def test_the_canonical_pin_is_qwen38_27b(mods) -> None:
    """Updated 2026-07-31: the pin moved from Q4_K_M to the QAT quant of the SAME model.

    Not a generator change -- gemma-4-31B-it either way, so every retired-marker assertion
    below still applies unchanged. Only the quantisation moved, and only on non-quality
    grounds: a 20-game x 3-trial head-to-head found the two INDISTINGUISHABLE (mean-B 6-6-8,
    sign test p = 1.0), so the switch was made for ~1 GB less VRAM and a matching QAT MTP
    drafter. See tests/python/test_arc_live_generator_qat_pairing.py.

    The repo substring is deliberately the QAT-specific one: "gemma-4-31B-it" matches BOTH
    cache directories and would resolve ambiguously.
    """
    agent, wm = mods
    assert wm.ARC_LIVE_GENERATOR_REPO_SUBSTR == "Qwen3.8-27B"
    assert wm.ARC_LIVE_GENERATOR_MODEL_ID == "unsloth/Qwen3.8-27B-GGUF"
    assert wm.ARC_LIVE_GENERATOR_MODEL_FILENAME == "Qwen3.8-27B-Q4_K_M.gguf"
    for marker in RETIRED_GENERATOR_MARKERS:
        assert marker not in wm.ARC_LIVE_GENERATOR_REPO_SUBSTR
        assert marker not in wm.ARC_LIVE_GENERATOR_MODEL_ID
    # The agent module re-exports the SAME object, not a copy that can drift.
    assert agent.ARC_LIVE_GENERATOR_REPO_SUBSTR is wm.ARC_LIVE_GENERATOR_REPO_SUBSTR


def test_mtp_defaults_off_because_gemma_4_31b_has_no_mtp_heads(mods) -> None:
    """Failure mode 2. Left at "1" this double-loads 18.3 GB of weights and OOMs."""
    _agent, wm = mods
    assert wm.ARC_LIVE_GENERATOR_MTP_DEFAULT == "0"


def test_no_think_prefix_is_empty_because_it_is_a_qwen_token(mods) -> None:
    """Failure mode 3: a control token the model does not have becomes literal prompt text."""
    _agent, wm = mods
    assert wm.ARC_LIVE_GENERATOR_NO_THINK_PREFIX == ""


def test_the_induction_proposer_uses_the_pin(mods) -> None:
    """`E3AgentPolicy._proposer()` is the scored induction generator -- the single most
    load-bearing construction site in the live path."""
    agent, wm = mods
    pol = agent.E3AgentPolicy("pintest", proposer=None, value_head=lambda _frame: 0.0)
    prop = pol._proposer()
    assert isinstance(prop, wm.LocalGGUFProposer)
    assert prop.repo_substr == wm.ARC_LIVE_GENERATOR_REPO_SUBSTR
    assert prop.mtp is False
    assert prop.no_think_prefix == ""
    assert prop.kv_quant == "q8_0"


def test_the_sge_candidate_router_uses_the_same_pin(mods) -> None:
    """`_load_sge_candidate_router()`'s own docstring promises a config IDENTICAL to
    `_proposer()`'s, because both default to port 8919 and rely on server reuse to share ONE warm
    model. A divergence does not fail loudly -- it relaunches on a fresh port and loads the model
    a second time, which at 18.3 GB is an OOM rather than a waste."""
    agent, wm = mods
    router = agent._load_sge_candidate_router("g50t")
    completer = router.proposer.completer
    assert isinstance(completer, wm.LocalGGUFProposer)
    assert completer.repo_substr == wm.ARC_LIVE_GENERATOR_REPO_SUBSTR
    assert completer.mtp is False
    assert completer.no_think_prefix == ""

    # ...and the two sites must genuinely AGREE, not merely each be gemma.
    pol = agent.E3AgentPolicy("pintest", proposer=None, value_head=lambda _frame: 0.0)
    prop = pol._proposer()
    for attr in ("repo_substr", "mtp", "kv_quant", "no_think_prefix", "n_ctx"):
        assert getattr(completer, attr) == getattr(prop, attr), attr


def test_submitted_agent_config_declares_the_new_generator(mods) -> None:
    agent, wm = mods
    frozen = agent.SUBMITTED_AGENT_CONFIG["frozen_generator"]
    assert frozen["model_id"] == wm.ARC_LIVE_GENERATOR_MODEL_ID
    assert frozen["repo_substr"] == wm.ARC_LIVE_GENERATOR_REPO_SUBSTR
    assert frozen["model_filename"] == wm.ARC_LIVE_GENERATOR_MODEL_FILENAME
    # SUPERSEDED 2026-07-28 (same day, measured); the old assertions were
    #     assert frozen["mtp"] is False
    #     assert frozen["spec_type"] is None
    # and they encoded a premise that turned out to be false: that gemma-4-31B-it has no MTP.
    # It does -- via a SEPARATE 491 MiB head (`mtp-gemma-4-31B-it-Q8_0.gguf`, arch
    # `gemma4-assistant`), not heads embedded in the main GGUF, which is why none was found there.
    #
    # `SUBMITTED_AGENT_CONFIG` describes the SCORED launch, so it tracks the SCORED constant. The
    # local default stays "0" and is correctly DIFFERENT: on a 24 GB dev card MTP-on forces ~14 FFN
    # blocks to system RAM to fit, and that offload costs more decode than MTP's measured 1.398x
    # returns. On the 96 GB Kaggle card no offload is needed, so it is a pure win.
    scored_on = wm.ARC_LIVE_GENERATOR_MTP_SCORED_DEFAULT != "0"
    assert frozen["mtp"] is scored_on
    assert frozen["spec_type"] == ("draft-mtp" if scored_on else None)
    # The head is a distinct FILE, so the config must name it distinctly. Recording only "mtp: True"
    # is what would let `--model-draft` drift back onto the main weights -- a configuration
    # llama.cpp accepts, warns about, and then serves with speculation silently disabled.
    assert frozen["mtp_head_filename"] == wm.ARC_LIVE_GENERATOR_MTP_HEAD_FILENAME
    assert frozen["mtp_head_arch"] == wm.ARC_LIVE_GENERATOR_MTP_HEAD_ARCH
    assert frozen["mtp_head_filename"] != frozen["model_filename"]
    assert frozen["no_think_prefix"] == ""
    for marker in RETIRED_GENERATOR_MARKERS:
        assert marker not in frozen["model_id"]
        assert marker not in frozen["model_filename"]


def test_submitted_config_records_both_operator_dataset_uploads(mods) -> None:
    """The Kaggle kernel attaches the model as a DATASET, and only the operator can create one.

    HISTORY, kept because it is the reason this test exists: `kaggle_dataset_uploaded` was False
    and this test asserted it, because the 18.3 GB gemma weights dataset did not exist yet and the
    flag was the mechanism stopping that dependency from being forgotten between here and a
    submission. The operator uploaded it on 2026-07-28, along with the 491 MB MTP head as a SECOND
    dataset, so the assertion flips from "records the outstanding upload" to "records that both
    landed". The flags stay in the config either way -- they are the record the readiness gate
    reads, not a one-time to-do.
    """
    # 2026-08-16 the generator moved to Qwen3.8-27B and this briefly returned to the FIRST role the
    # docstring describes -- recording an outstanding upload. 2026-08-17 it landed: the private
    # `carnot-qwen38-27b-gguf` dataset holds 17,106,775,008 bytes, verified byte-identical to the
    # local GGUF once Kaggle finished indexing. So the assertion flips back to "records that it
    # landed", the same arc gemma went through on 2026-07-28.
    agent, _wm = mods
    frozen = agent.SUBMITTED_AGENT_CONFIG["frozen_generator"]
    assert frozen["kaggle_dataset_slug"] == "iancblenke/carnot-qwen38-27b-gguf"
    assert frozen["kaggle_dataset_uploaded"] is True
    # No MTP draft head ships for Qwen3.8-27B, so there is no second dataset. That is a measured
    # cost of the swap (gemma's head bought roughly 1.8x decode), not a missing upload -- so the
    # slug is None and the uploaded flag is False, and they must not disagree with each other.
    assert frozen["kaggle_mtp_head_dataset_slug"] is None
    assert frozen["kaggle_mtp_head_dataset_uploaded"] is False
    assert frozen["kaggle_mtp_head_dataset_slug"] != frozen["kaggle_dataset_slug"]


def test_the_ige_cell_selector_has_no_generator_literal_of_its_own(mods) -> None:
    """It used to carry the pin as a string literal in TWO places (the ctor default and
    `coerce_ige_cell_selector`'s), which is precisely how a switch lands in one and not the other.
    Now neither holds a literal: an unset value resolves to the canonical pin at build time."""
    _agent, wm = mods
    sel = sel_mod.IGECellSelector()
    assert sel.repo_substr == ""  # "" == not overridden
    prop = sel._get_proposer()
    assert prop.repo_substr == wm.ARC_LIVE_GENERATOR_REPO_SUBSTR
    assert prop.mtp is False
    assert prop.no_think_prefix == ""

    coerced = sel_mod.coerce_ige_cell_selector({"enabled": True})
    assert coerced is not None
    assert coerced._get_proposer().repo_substr == wm.ARC_LIVE_GENERATOR_REPO_SUBSTR

    # An EXPLICIT override still wins -- this is a default, not a hardcode.
    assert (
        sel_mod.IGECellSelector(repo_substr="some-other-model")._get_proposer().repo_substr
        == "some-other-model"
    )


def test_no_live_path_module_still_pins_a_retired_qwen_generator() -> None:
    """A source-level sweep, because the per-site assertions above can only cover sites we
    remembered. Historical PROSE about what the 9B measured is legitimate (never-prune) and is
    excluded by looking only at lines that are actually a generator pin -- a `repo_substr=` or
    `_resolve_gguf(` argument, or an assignment to one of the canonical constants.

    THE FILE LIST IS THE WHOLE POINT, and it was wrong when this test shipped. It listed four
    `python/carnot/agentic/` modules and stopped there, which is exactly the set the switch had
    already been applied to -- so the sweep could only ever confirm files known clean. Meanwhile
    `scripts/arc_loop_solve.py` -- one of the TWO canonical live entrypoints per CLAUDE.md's ARC
    Live-Path Reachability Discipline -- still pinned `repo_substr="Qwen3.5-9B-MTP", mtp=True` at
    line 352, and this test passed green over it. A sweep whose corpus is "the files we already
    fixed" is not a safety net; it is a second copy of the per-site assertions.

    THE PATTERN LIST likewise had a hole. It matched only per-SITE literals
    (`repo_substr=`/`_resolve_gguf(`), so reverting the canonical constants themselves --
    `ARC_LIVE_GENERATOR_REPO_SUBSTR = "Qwen3.5-9B-MTP"`, the single edit that re-pins every site
    at once -- was invisible to it. Verified by mutation: that revert killed 7 of the 9 tests in
    this file and this one still PASSED. Now the constant assignments are matched too, so the
    sweep catches the cheapest possible regression as well as the most obscure one.
    """
    import pathlib
    import re

    repo = pathlib.Path(__file__).resolve().parents[2]
    live = [
        "python/carnot/agentic/arc_competition_agent.py",
        "python/carnot/agentic/arc_executable_world_model.py",
        "python/carnot/agentic/arc_ige_cell_selector.py",
        "python/carnot/agentic/arc_llm_guided_solve.py",
        # The two live entrypoints named in CLAUDE.md. `arc_competition_agent.py` (above) is the
        # scored one; these two are the offline development twin and the Kaggle kernel that wraps
        # the scored one. Both construct generators, both were missed by the first sweep.
        "scripts/arc_loop_solve.py",
        "scripts/kaggle/submission_kernel/main.py",
    ]
    pin = re.compile(
        r"""(repo_substr\s*=\s*|_resolve_gguf\(\s*|ARC_LIVE_GENERATOR_[A-Z_]+\s*=\s*)["']([^"']*)["']"""
    )
    offenders = []
    for rel in live:
        for lineno, line in enumerate(
            (repo / rel).read_text(encoding="utf-8").splitlines(), start=1
        ):
            for _pfx, value in pin.findall(line):
                if any(marker in value for marker in RETIRED_GENERATOR_MARKERS):
                    offenders.append(f"{rel}:{lineno}: {line.strip()}")
    assert not offenders, "retired Qwen generator pinned on the live path:\n" + "\n".join(offenders)
