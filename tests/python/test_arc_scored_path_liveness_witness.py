"""Tests for the SCORED-PATH generator-liveness witness (2026-07-27).

REQ-ARC-GEN-LIVENESS-2: a dead generator MUST NOT be able to produce a scored-path record
that claims `llm_enabled: True` and passes `scripts/arc_llm_on_liveness_lint.py`.

The sibling file `test_arc_llm_on_liveness_lint.py` proves the GATE fires on the eight
recorded K=4 dead cells. That gate was blind to the scored path for a structural reason:
the scored path emitted NO row at all. `arc_competition_agent.py` had zero `print` and zero
`logging` calls across 6290 lines, and each game's `induction_attempts` list died with its
thread, so a scored run whose generator died completed 400 actions, exited 0, and left
nothing behind that could distinguish "the LLM tier ran and did not help" from "the LLM
tier was dead the whole time". These tests pin the two halves that close it:

  SCENARIO-WITNESS-EXISTS   the policy emits a row carrying the liveness PRIMITIVES, and
                            `Agent.cleanup()` (the framework's once-per-game hook) writes
                            it to stderr AND to disk.
  SCENARIO-GATE-REFUSES     that row, taken from a genuinely dead generator, is REFUSED by
                            the existing lint -- i.e. the emitter and the gate actually
                            meet, rather than each being individually plausible.
  SCENARIO-ORIGIN-REPLAY    the eight REAL recorded dead cells, re-expressed in the new
                            witness shape (which adds `llm.calls`), still FAIL. A new field
                            that accidentally exempted the origin incident would be worse
                            than no field.
  SCENARIO-NO-OVER-FIRE     a game that never stalled into induction makes zero generator
                            calls; that is WARN, not FAIL, because "never asked" is not the
                            same defect as "asked and got nothing".
  SCENARIO-NEVER-CRASHES    the witness can never break the run. An unsolved level scores 0
                            whether or not the agent aborts, so aborting cannot GAIN score,
                            and swarm.py runs every game in ONE process -- an exception
                            escaping cleanup() could zero the whole eval.

WHY THE DEAD-GENERATOR CASE USES A REAL LocalGGUFProposer AND A REAL FAILING CALL rather
than a hand-written row: the whole point is that the counters are populated by the actual
swallow path (`except Exception -> return False, msg`). A test that hand-builds
`{"errors": 2}` would pass even if the production code never incremented anything -- which
is exactly the state the corpus was in before this change (877 stat blocks with an `errors`
key, zero non-zero, because the only incrementing branch required an exception to
propagate and none ever did).
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys

import pytest

# Constructing a real E3AgentPolicy pulls in the agentic stack (jax + torch + numpy), a
# ~680MB ONE-TIME import footprint that the RSS watchdog reads as a per-test leak. Same
# situation, same marker, same reasoning as test_arc_gateway_card_ground_truth.py: these
# tests do not leak, the import is genuinely that big -- and the whole point of this file is
# that the witness is taken from the REAL policy object rather than a hand-built dict, so
# skipping the construction to dodge the watchdog would gut the test.
pytestmark = pytest.mark.memory_watchdog_skip

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_LINT = os.path.join(_REPO, "scripts", "arc_llm_on_liveness_lint.py")
_ROWS_DIR = os.path.join(_REPO, "results", "llm_on_contention_rows_20260726")


def _load_lint():
    spec = importlib.util.spec_from_file_location("arc_llm_on_liveness_lint", _LINT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lint = _load_lint()


def _policy(game: str = "zz00"):
    from carnot.agentic.arc_competition_agent import E3AgentPolicy

    return E3AgentPolicy(game)


def _dead_proposer():
    """A REAL LocalGGUFProposer whose server can never be reached.

    `_ensure_server` is stubbed to False rather than pointing at a closed port because the
    real method would spend up to 600s trying to LAUNCH a server. The stub reproduces the
    exact branch a dead/unreachable generator takes; the counters, the message and the
    return contract are all the production ones."""
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

    p = LocalGGUFProposer(port=1)
    p._ensure_server = lambda: False  # type: ignore[method-assign]
    return p


# --- SCENARIO-WITNESS-EXISTS + SCENARIO-GATE-REFUSES -----------------------------------


def test_dead_generator_witness_is_refused_by_the_gate() -> None:
    """The end-to-end contract: real swallowed failures -> a row -> the lint REFUSES it."""
    policy = _policy()
    policy.proposer = _dead_proposer()

    ok, msg = policy.proposer.generate("induce something")
    assert ok is False, "the dead-generator path must still return False (control flow unchanged)"
    ok2, _ = policy.proposer.complete_text("pick a cell")
    assert ok2 is False

    row = policy.generator_liveness_witness()
    assert row["llm_enabled"] is True, "the row must still CLAIM the LLM tier -- that is the point"
    assert row["llm"]["calls"] == 2, row["llm"]
    assert row["llm"]["responses"] == 0, row["llm"]
    assert row["llm"]["errors"] == 2, (
        "the errors channel was structurally dead before this change (877 blocks, zero "
        f"non-zero); it must now count real swallowed failures, got {row['llm']}"
    )
    assert row["generator_healthy_after"] is False
    assert row["llm_on_row_valid"] is False

    findings = lint.check_row(row)
    codes = {f["code"] for f in findings}
    assert any(f["severity"] == "FAIL" for f in findings), (
        f"THE GATE DID NOT REFUSE A DEAD-GENERATOR SCORED ROW; codes={codes or 'NONE'}"
    )
    assert {"DEAD_GENERATOR", "NO_COMPLETIONS"} <= codes, codes


def test_witness_keeps_the_server_diagnostic_string() -> None:
    """exp5866 finding 4: the one informative string was thrown away. It must be in the row."""
    policy = _policy()
    policy.proposer = _dead_proposer()
    policy.proposer.generate("x")
    row = policy.generator_liveness_witness()
    diags = row.get("generator_server_failure_diagnostics")
    assert diags and isinstance(diags, list), row
    assert "llama-server failed" in diags[0], diags


def test_witness_records_the_configured_pool_and_budget() -> None:
    """The row must say WHICH n_ctx/max_tokens produced it, or a future concurrency fault
    cannot be attributed to a configuration from the record alone -- the gap that made this
    fault take three sessions to characterise."""
    policy = _policy()
    policy.proposer = _dead_proposer()
    policy.proposer.generate("x")
    row = policy.generator_liveness_witness()
    assert row["generator_n_ctx"] == policy.proposer.n_ctx
    assert row["generator_max_tokens"] == policy.proposer.max_tokens


def test_witness_before_any_induction_says_so_explicitly() -> None:
    """A lazily-unbuilt proposer means the tier was never reached. The row must SAY that
    rather than omit the liveness fields -- an absent witness reads as a clean null."""
    policy = _policy()
    assert policy.proposer is None
    row = policy.generator_liveness_witness()
    assert row["generator_constructed"] is False
    assert row["llm"] == {"calls": 0, "responses": 0, "errors": 0, "content_failures": 0}
    assert row["generator_healthy_after"] is None
    assert row["llm_on_row_valid"] is False


# --- SCENARIO-NO-OVER-FIRE --------------------------------------------------------------


def test_never_engaged_row_is_warn_not_fail() -> None:
    """Zero calls -> WARN LLM_TIER_NEVER_ENGAGED. Failing this shape would flag every scored
    game that never stalled into induction, which trains the operator to ignore the gate."""
    policy = _policy()
    row = policy.generator_liveness_witness()
    findings = lint.check_row(row)
    assert [f["code"] for f in findings] == ["LLM_TIER_NEVER_ENGAGED"], findings
    assert findings[0]["severity"] == "WARN"
    assert not any(f["severity"] == "FAIL" for f in findings)


def test_healthy_generator_row_is_clean() -> None:
    """The control: a witness from a generator that answered must produce NO findings, or
    the gate would refuse every honest scored row and be turned off."""
    policy = _policy()
    p = _dead_proposer()
    p._ensure_server = lambda: True  # type: ignore[method-assign]
    p._healthy = lambda: True  # type: ignore[method-assign]
    p.n_completion_calls, p.n_completion_ok = 7, 7
    policy.proposer = p
    row = policy.generator_liveness_witness()
    assert row["llm_on_row_valid"] is True
    assert not lint.check_row(row), lint.check_row(row)


# --- SCENARIO-ORIGIN-REPLAY -------------------------------------------------------------


def test_the_eight_recorded_dead_cells_still_fail_in_the_new_witness_shape() -> None:
    """Read the REAL eight cells off disk, re-express each in the new witness shape (adding
    the `llm.calls` field this change introduces), and assert the gate still fires.

    A new field that accidentally exempted the origin incident -- e.g. by making
    NO_COMPLETIONS conditional in a way that also swallowed `calls`-absent rows -- would be
    a regression of exactly the kind this project has shipped before."""
    import glob

    paths = sorted(glob.glob(os.path.join(_ROWS_DIR, "cells", "cell_K4_*_b400.json")))
    if not paths:
        pytest.skip(f"origin corpus absent: {_ROWS_DIR}")
    assert len(paths) == 8, f"expected the 8 recorded K=4 cells, found {len(paths)}"
    for path in paths:
        with open(path) as fh:
            cell = json.load(fh)
        old = cell.get("row") or cell
        responses = int((old.get("llm") or {}).get("responses") or 0)
        witness = {
            "game": old.get("game"),
            "actions": old.get("actions"),
            "llm_enabled": True,
            # the new shape: calls >= responses, and the failures are now COUNTED
            "llm": {
                "calls": max(1, responses + 1),
                "responses": responses,
                "errors": 1,
                "content_failures": 0,
            },
            "generator_healthy_after": old.get("generator_healthy_after"),
            "llm_on_row_valid": False,
        }
        codes = {f["code"] for f in lint.check_row(witness)}
        assert "DEAD_GENERATOR" in codes, (
            f"{os.path.basename(path)} exempted by the new shape: {codes}"
        )


def test_zeroing_calls_cannot_dodge_the_gate() -> None:
    """The anti-gaming branch for the new field: a row that zeroes `calls` to earn the WARN
    while still recording server errors is self-contradictory and FAILS."""
    codes = {
        f["code"]
        for f in lint.check_row(
            {
                "llm_enabled": True,
                "game": "liar",
                "actions": 400,
                "llm": {"calls": 0, "responses": 0, "errors": 5},
                "generator_healthy_after": True,
                "llm_on_row_valid": True,
            }
        )
    }
    assert "WITNESS_SELF_CONTRADICTORY" in codes, codes
    assert "VALID_STAMP_WRONG" in codes, codes


def test_partial_failures_are_warn_and_do_not_make_the_stamp_a_lie() -> None:
    """A run with 8 completions and 2 failures is partially degraded, not invalid. It must
    WARN, and the WARN alone must not manufacture a VALID_STAMP_WRONG."""
    findings = lint.check_row(
        {
            "llm_enabled": True,
            "game": "partial",
            "actions": 400,
            "llm": {"calls": 10, "responses": 8, "errors": 2},
            "generator_healthy_after": True,
            "llm_on_row_valid": True,
        }
    )
    codes = {f["code"] for f in findings}
    assert codes == {"GENERATOR_PARTIAL_FAILURES"}, codes
    assert not any(f["severity"] == "FAIL" for f in findings)


# --- SCENARIO-NEVER-CRASHES (the cleanup() adapter) ------------------------------------


class _FakeBase:
    """Stand-in for the framework's Agent base, with the cleanup() contract it really has."""

    MAX_ACTIONS = 80

    def __init__(self, *a, **k) -> None:
        self.game_id = "zz00"
        self.action_counter = 0
        self.cleaned = False

    @property
    def levels_completed(self) -> int:
        return 0

    def cleanup(self, scorecard=None) -> None:
        self.cleaned = True


def _agent_with(policy, tmp_path, monkeypatch):
    from carnot.agentic.arc_competition_agent import make_carnot_agent

    monkeypatch.setenv("CARNOT_ARC_LIVENESS_DIR", str(tmp_path))
    cls = make_carnot_agent(_FakeBase)
    agent = object.__new__(cls)  # bypass the real __init__ (disk/jax/model load)
    _FakeBase.__init__(agent)
    agent.action_counter = 396
    agent._policy = policy
    return agent


def test_cleanup_writes_a_lintable_row_and_a_stderr_line(tmp_path, monkeypatch, capsys) -> None:
    policy = _policy()
    policy.proposer = _dead_proposer()
    policy.proposer.generate("x")
    agent = _agent_with(policy, tmp_path, monkeypatch)

    agent.cleanup()

    assert agent.cleaned is True, "super().cleanup() must still run"
    err = capsys.readouterr().err
    assert "LLM LIVENESS" in err, err
    assert "healthy_after=False" in err, err

    written = sorted(tmp_path.glob("llm_liveness_*.json"))
    assert len(written) == 1, [p.name for p in written]
    row = json.loads(written[0].read_text())
    assert row["actions"] == 396
    codes = {f["code"] for f in lint.check_row(row)}
    assert "DEAD_GENERATOR" in codes, codes


def test_cleanup_still_calls_super_when_the_witness_explodes(tmp_path, monkeypatch) -> None:
    """The witness must never be able to zero a game. swarm.py runs every game in ONE
    process, so an exception escaping here could take down the whole eval."""

    class _Exploding:
        def generator_liveness_witness(self):
            raise RuntimeError("boom")

    agent = _agent_with(_Exploding(), tmp_path, monkeypatch)
    agent.cleanup()  # must NOT raise
    assert agent.cleaned is True


def test_cleanup_survives_an_unwritable_output_dir(tmp_path, monkeypatch, capsys) -> None:
    """The stderr channel must survive a read-only filesystem -- that is why there are two
    channels rather than only the JSON row."""
    policy = _policy()
    policy.proposer = _dead_proposer()
    policy.proposer.generate("x")
    blocked = tmp_path / "nope"
    blocked.write_text("i am a file, not a directory")
    agent = _agent_with(policy, blocked, monkeypatch)
    agent.cleanup()
    assert agent.cleaned is True
    assert "LLM LIVENESS" in capsys.readouterr().err


# --- the record fixes (exp5866 finding 4) ---------------------------------------------


def test_http_failure_description_includes_the_response_body() -> None:
    """`repr(HTTPError)` is only the generic reason phrase. The body carries the actual
    diagnosis -- "Context size has been exceeded." for the 500, and for the 400 literally
    the fix ("try increasing it"). Both were read and discarded."""
    import io
    import urllib.error

    from carnot.agentic.arc_executable_world_model import _describe_http_failure

    err = urllib.error.HTTPError(
        "http://127.0.0.1:8919/completion",
        500,
        "Internal Server Error",
        {},  # type: ignore[arg-type]
        io.BytesIO(b'{"error":{"message":"Context size has been exceeded."}}'),
    )
    described = _describe_http_failure(err)
    assert "500" in described
    assert "Context size has been exceeded." in described, described

    # a non-HTTP exception degrades to the plain repr rather than raising
    assert "ValueError" in _describe_http_failure(ValueError("plain"))


def test_pool_truncation_is_named_differently_from_the_budget_limit() -> None:
    """The two faults that both report stop_type == 'limit' must not share one message.

    The old text said "HIT n_predict=<max> OUTPUT LIMIT" for both, so exp5866's mode C -- a
    completion cut off at 630 of a 4096-token budget because the prompt had eaten the shared
    pool -- was indistinguishable from a model that genuinely used its whole budget. The
    prescriptions are OPPOSITE: mode C needs a bigger -c, and a bigger max_tokens makes it
    worse."""
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

    p = LocalGGUFProposer(max_tokens=4096, n_ctx=16384)

    p._record_completion_diagnostics({"stop_type": "limit", "timings": {"predicted_n": 4096}})
    budget = p._limit_diagnostic()
    assert "n_predict=4096 OUTPUT LIMIT" in budget, budget
    assert "SHARED CONTEXT POOL" not in budget

    p._record_completion_diagnostics({"stop_type": "limit", "timings": {"predicted_n": 630}})
    pool = p._limit_diagnostic()
    assert "TRUNCATED BY SHARED CONTEXT POOL" in pool, pool
    assert "630" in pool and "16384" in pool, pool
    assert "CARNOT_ARC_INDUCE_N_CTX" in pool, pool

    # no timings at all -> must fall back to the old message, never crash
    p._record_completion_diagnostics({"stop_type": "limit"})
    assert "OUTPUT LIMIT" in p._limit_diagnostic()


def test_content_failure_is_not_counted_as_a_liveness_failure() -> None:
    """A server that answers with unusable code is ALIVE. Conflating the two would make a
    terse model read as a dead generator and defeat the gate's whole purpose."""
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

    p = LocalGGUFProposer(port=1)
    p._ensure_server = lambda: True  # type: ignore[method-assign]
    p._healthy = lambda: True  # type: ignore[method-assign]
    # the server answers, but with no `def engine` -> content failure, 3 tries
    p._chat_complete_request = lambda *a, **k: ({"content": "no code here"}, "no code here")  # type: ignore[method-assign]
    p.use_chat_template = True
    ok, _ = p.generate("x", tries=2)
    assert ok is False
    assert p.n_server_failures == 0, "an answering server must not be counted as a failure"
    assert p.n_content_failures == 1
    assert p.liveness_witness()["llm"]["errors"] == 0


# --- the submission-kernel pre-flight -------------------------------------------------


def _kernel_agent_src() -> str:
    """Extract the `my_agent.py` source the submission kernel AUTHORS at runtime.

    It lives as a string literal inside main.py, so nothing imports or executes it during
    normal testing -- which is why a hardcoded probe config could sit in it undetected."""
    import ast

    path = os.path.join(_REPO, "scripts", "kaggle", "submission_kernel", "main.py")
    with open(path) as fh:
        src = fh.read()
    ast.parse(src)  # the kernel itself must be valid Python
    for node in ast.walk(ast.parse(src)):
        if (
            isinstance(node, ast.Assign)
            and getattr(node.targets[0], "id", "") == "AGENT_SRC"
            and isinstance(node.value, ast.Constant)
        ):
            agent_src = node.value.value
            ast.parse(agent_src)  # ...and so must what it writes to disk
            return agent_src
    pytest.fail("AGENT_SRC not found in the submission kernel")


def test_kernel_probe_reads_the_shipped_context_size_instead_of_a_literal() -> None:
    """The probe printed "ctx=16384" and launched with `-c "16384"` as hardcoded strings. If
    the agent's own default moved, the probe would validate a configuration the agent never
    uses -- and report it HEALTHY. That is the measure-one-thing-ship-another shape of the
    0.08 incident, reproduced inside the diagnostic."""
    agent_src = _kernel_agent_src()
    assert "_default_induce_n_ctx" in agent_src, (
        "the kernel probe must read the shipped n_ctx default, not repeat a literal"
    )
    assert '"-c", "16384"' not in agent_src and "'-c', '16384'" not in agent_src


def test_kernel_probe_exercises_concurrency_not_just_health() -> None:
    """A /health check is a concurrency-1 probe, and concurrency 1 is exactly where this fault
    is invisible. The probe must issue SIMULTANEOUS full-budget requests."""
    agent_src = _kernel_agent_src()
    assert "ThreadPoolExecutor" in agent_src, "the probe is still single-threaded"
    assert "LLM CONCURRENCY" in agent_src, "the probe result must be greppable in the eval log"
    assert "CARNOT_ARC_INDUCE_N_CTX" in agent_src, (
        "a failed concurrency probe must name the lever that fixes it"
    )
    # the probe must use the agent's OWN completion budget: it is (prompt + n_predict) x K
    # that has to fit the pool, so probing with a smaller n_predict would pass where the
    # agent fails.
    assert "CARNOT_ARC_INDUCE_MAX_TOKENS" in agent_src
    assert '"n_predict": _maxtok' in agent_src


def test_kernel_probe_uses_the_slot_count_not_a_smaller_k() -> None:
    """K=2 IS A PASSING REGION FOR A K=4 REQUIREMENT -- measured, not argued.

    Admission needs n_ctx >= K*(prompt + n_predict). The probe's own synthetic prompt measures
    17238 tokens through the model's tokenizer, so at the shipped n_ctx=81920/max_tokens=4096:
      K=2 -> 2*(17238+4096) = 42668 <= 81920  PASS
      K=4 -> 4*(17238+4096) = 85336 >  81920  FAIL
    Directly measured on an RTX 3090 (2026-07-27, mtp-off): 4/4 HTTP 500 "Context size has been
    exceeded", per-slot n_tokens 20469..20493 at release == exactly 81920/4. So the first
    hardened probe would have printed "LLM CONCURRENCY OK" for a configuration that fails at the
    K the eval actually produces -- a pass that could not have failed, in the gate itself.

    The probe must therefore derive K from the same module that sizes the pool."""
    agent_src = _kernel_agent_src()
    assert "max_workers=2" not in agent_src, (
        "the probe is pinned at K=2 again; that is a passing region for the K=4 requirement"
    )
    assert "_LLAMA_SERVER_DEFAULT_SLOTS" in agent_src, (
        "the probe must read the slot count from the module that sizes the pool, so the two "
        "cannot disagree"
    )
    assert "max_workers=_K" in agent_src


def test_kernel_probe_reads_the_body_and_the_stop_taxonomy_not_just_http_status() -> None:
    """The fix investigation's own load-bearing result: an HTTP-status-only gate PASSES
    `--parallel 1` 4/4 at lower VRAM while it generates 648/650/184/648 tokens against a 4096
    budget -- mode C, silent truncation, the exact defect under investigation. So a status-only
    gate would have shipped the bug. The probe must compare generated tokens to the budget."""
    agent_src = _kernel_agent_src()
    assert '"stop": ["\\n"]' not in agent_src, (
        "the newline stop halts generation after ~1 token, so the probe cannot observe "
        "truncation even in principle"
    )
    assert "predicted_n" in agent_src, "the probe must read how many tokens were generated"
    assert "pool_exhaustion_truncations" in agent_src
    assert "LLM CONCURRENCY SILENTLY TRUNCATED" in agent_src, (
        "mode C must have its own greppable verdict; folding it into the generic FAIL message "
        "loses the distinction whose prescriptions are OPPOSITE (raise n_ctx vs raise max_tokens)"
    )
    # the 500 body says "Context size has been exceeded." and the 400 body says "try increasing
    # it" -- the informative string that was thrown away unread twice.
    assert "_ex.read()" in agent_src, "the probe must read the server's own error body"


def test_kernel_probe_reads_props_to_validate_the_slot_assumption_on_the_real_binary() -> None:
    """The sizing rests on llama.cpp defaulting to 4 slots when --parallel is absent. That was
    read from THIS repo's local build's source; the scored run uses a different bundled binary.
    One GET on /props settles it on the binary that actually runs."""
    agent_src = _kernel_agent_src()
    assert "/props" in agent_src
    assert "total_slots" in agent_src
    assert "LLM SLOT COUNT MISMATCH" in agent_src, (
        "a slot count other than the one the pool was sized for must be loud, not silent"
    )


def test_kernel_probe_prompt_is_sized_to_the_pool_it_validates() -> None:
    """A probe prompt LARGER than the pool admits tests the wrong thing: it fails for a reason
    the eval will never hit and would push an operator to raise n_ctx needlessly. The prompt is
    trimmed against the server's own /tokenize to the same measured worst case the pool is
    sized for -- read, not eyeballed."""
    agent_src = _kernel_agent_src()
    assert "_INDUCE_WORST_CASE_PROMPT_TOKENS" in agent_src
    assert "/tokenize" in agent_src, "the prompt size must be READ from the server, not assumed"


def test_kernel_preflight_prints_the_actual_gpu() -> None:
    """machine_shape 'NvidiaL4' is an unvalidated free-form string (kagglesdk ships no enum for
    it), and the only Kaggle nvidia-smi read this project holds is a P100 16GB from a DIFFERENT
    kernel. One line in the scored log finally replaces the inference with a measurement."""
    agent_src = _kernel_agent_src()
    assert "LLM GPU HARDWARE" in agent_src
    assert "memory.total" in agent_src


def test_kernel_probe_cannot_crash_the_submission() -> None:
    """Every probe branch stays inside a try. A probe that raised would zero the whole
    submission -- strictly worse than the silent degradation it is meant to expose."""
    agent_src = _kernel_agent_src()
    assert "LLM CONCURRENCY PROBE ERROR (non-fatal)" in agent_src
    assert "LLM PROBE ERROR (non-fatal, agent continues with LLM env set)" in agent_src


# --- the induce-exception record ------------------------------------------------------


def test_induction_exception_is_recorded_with_its_repr(monkeypatch) -> None:
    """The outermost induce handler wrote the bare string "exception" and discarded the
    type/message/traceback -- which is why making generate() raise would have been LESS
    informative than its current return: the raise would land HERE and be erased.

    `_fit_dsl_model` is the raise site because it is called UNCONDITIONALLY inside the
    outermost try (arc_competition_agent.py:5607) and has no inner handler of its own, so an
    exception from it reaches exactly the handler under test."""
    policy = _policy()
    monkeypatch.delenv("CARNOT_ARC_DISABLE_INDUCTION", raising=False)
    monkeypatch.setattr(policy, "_active_transitions", lambda: [object()])
    monkeypatch.setattr(policy, "_active_dsl_transitions", lambda: [])

    def boom(*a, **k):
        raise RuntimeError("induction blew up inside the try")

    monkeypatch.setattr(policy, "_fit_dsl_model", boom)

    policy._induce_and_plan()  # must not raise -- control flow is unchanged

    attempt = policy.induction_attempts[-1]
    assert attempt["skipped"] == "exception", attempt
    assert attempt.get("exception", "").startswith("RuntimeError("), (
        "the outermost handler still discards the exception -- the record fix is not live: "
        f"{attempt}"
    )
    assert "induction blew up inside the try" in attempt["exception"], attempt
    assert sys.exc_info()[0] is None, "the exception must not propagate"


# --- 2026-07-27 ADVERSARIAL REVIEW: the witness must MEASURE, not echo -------------------
# Findings 1 and 11 of the review of commit 776161963. Both are the SAME defect class the
# witness was built to remove, re-appearing one layer up: a field that reports what the
# caller INTENDED rather than what was actually the case. Each test below replays the
# finding's own reproduction, so a regression fires on the real incident and not on a
# stylised version of it.


class _PropsServerProposer:
    """Test double for a RUNNING llama-server whose /props reports `served`.

    Subclasses the real LocalGGUFProposer so every method under test is the production one;
    only the two network reads are replaced. Using the real class matters because the whole
    finding is that a method returned self-state instead of reading the wire -- a hand-built
    stub would pass whether or not the production code was fixed."""

    @staticmethod
    def make(served: int, *, declared: int, port: int = 65500):
        from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

        p = LocalGGUFProposer(port=port, n_ctx=declared)
        p._healthy = lambda: True  # type: ignore[method-assign]
        p.server_props = lambda: {  # type: ignore[method-assign]
            "default_generation_settings": {"n_ctx": served},
            "total_slots": 4,
        }
        return p


def test_witness_publishes_the_OBSERVED_n_ctx_not_the_declared_one() -> None:
    """FINDING 1, origin replay. On the dev box, port 8919 (this class's DEFAULT port) was
    serving n_ctx=16384 from the previous evening's launch while `LocalGGUFProposer().n_ctx`
    was 81920. The witness published `int(self.n_ctx)` -- 81920 -- so a run reusing the
    faulty server SELF-CERTIFIED AS FIXED. The witness must report 16384."""
    p = _PropsServerProposer.make(16384, declared=81920)
    row = p.liveness_witness()
    assert row["generator_n_ctx"] == 16384, (
        "the witness is still echoing the DECLARED n_ctx; a stale faulty server would "
        f"report itself as fixed: {row}"
    )
    assert row["generator_n_ctx_declared"] == 81920, row
    assert row["generator_n_ctx_source"] == "server_props_observed", row
    assert row["generator_total_slots_observed"] == 4, row


def test_witness_marks_an_unobservable_n_ctx_as_declared_only() -> None:
    """The fallback must be LABELLED, not silently indistinguishable from a measurement --
    otherwise a build that does not serve /props re-opens the same gap invisibly."""
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

    p = LocalGGUFProposer(port=65501, n_ctx=81920)
    p._healthy = lambda: True  # type: ignore[method-assign]
    p.server_props = lambda: {}  # type: ignore[method-assign]
    row = p.liveness_witness()
    assert row["generator_n_ctx"] == 81920, row
    assert row["generator_n_ctx_source"] == "declared_only", (
        f"an unobserved value must say so; unlabelled it reads as evidence: {row}"
    )


def test_ensure_server_refuses_to_reuse_a_smaller_pool() -> None:
    """FINDING 1, the behavioural half. `_ensure_server()` returned True on a bare /health
    check, so the n_ctx fix was a silent NO-OP against any server already on the port."""
    p = _PropsServerProposer.make(16384, declared=81920)
    assert p._reusable() is False, (
        "a 16384 server is being reused by a proposer that needs 81920 -- this is the "
        f"silent no-op: {p.reuse_n_ctx_check}"
    )
    assert "refused_smaller_pool" in p.reuse_n_ctx_check
    assert p.observed_server_n_ctx == 16384


def test_ensure_server_does_reuse_an_equal_or_larger_pool() -> None:
    """The no-over-fire control. Refusing a pool that is big enough would relaunch a server
    on every call -- a 180s stall per game and a new way to end up LLM-off."""
    assert _PropsServerProposer.make(81920, declared=81920)._reusable() is True
    bigger = _PropsServerProposer.make(131072, declared=81920)
    assert bigger._reusable() is True
    assert bigger.reuse_n_ctx_check == "larger_ok"


def test_stub_proposer_row_is_not_an_llm_on_claim() -> None:
    """FINDING 11, origin replay. The 2026-07-27 first-win measurement's 100 LLM-OFF control
    cells ran `_NoOpProposer` and every row carried `llm_enabled: True`, so the lint reported
    "174 rows, 174 claiming llm_enabled=True" over a corpus where 100 had no generator at
    all. A consumer keying on `llm_enabled` alone reads a NoOp arm as an LLM-on arm."""

    class _NoOpProposer:  # the shape experiment_4605 installs: no liveness channel at all
        def induce(self, *a, **k):
            return False, "disabled_exp4605_parity"

    policy = _policy()
    policy.proposer = _NoOpProposer()
    row = policy.generator_liveness_witness()

    assert row["generator_is_stub"] is True, row
    assert row["llm_tier_operational"] is False, (
        "a stub arm still reports an operational LLM tier -- the silent misclassification "
        f"is live: {row}"
    )
    assert row["llm_on_row_valid"] is False, row
    assert lint.check_row(row) == [], (
        "an honest LLM-OFF control row must make no LLM-on claim for the gate to judge; "
        f"got {lint.check_row(row)}"
    )


def test_a_REAL_but_dead_generator_is_still_operational_and_still_refused() -> None:
    """The mutation proof for finding 11's fix. If `llm_tier_operational` were computed from
    'did the generator answer' instead of 'was a real generator installed', a dead generator
    would mark itself non-operational and the lint would stop refusing it -- turning a fix
    for a control-arm mislabel into a hole in the gate that matters."""
    policy = _policy()
    policy.proposer = _dead_proposer()
    policy.proposer.generate("x")
    row = policy.generator_liveness_witness()

    assert row["generator_is_stub"] is False, row
    assert row["llm_tier_operational"] is True, (
        f"a REAL generator that died must still claim the tier, or the gate cannot refuse it: {row}"
    )
    codes = {f["code"] for f in lint.check_row(row)}
    assert {"DEAD_GENERATOR", "NO_COMPLETIONS"} <= codes, codes


def test_a_port_relaunch_does_not_poison_the_liveness_gate() -> None:
    """The mutation proof for finding 1's fix. The refusal-to-reuse path has to record
    ITSELF somewhere, and the obvious place -- `_note_server_failure` -- would increment
    `n_server_failures`, flipping `llm_on_row_valid` to False for a run whose generator
    then worked perfectly. That would over-fire the one gate that has to stay trustworthy,
    trading a silent false-negative for a noisy false-positive. The refusal belongs on its
    own channel."""
    p = _PropsServerProposer.make(16384, declared=81920, port=65502)
    assert p._reusable() is False
    # Simulate exactly what _ensure_server does on the refusal branch.
    p.reuse_refusals.append("port 65502 already serves n_ctx=16384 < required 81920")
    p.n_completion_calls, p.n_completion_ok = 5, 5

    row = p.liveness_witness()
    assert row["llm"]["errors"] == 0, (
        "the port relaunch was counted as a GENERATOR failure; a healthy run would now fail "
        f"llm_on_row_valid for a configuration event: {row['llm']}"
    )
    assert row["generator_reuse_refusals"], "the refusal must still be recorded somewhere"
    assert "refused_smaller_pool" in row["generator_reuse_n_ctx_check"], row


def test_a_false_operational_stamp_cannot_hide_a_row_with_real_generator_traffic() -> None:
    """The escape-hatch check for finding 11's fix. Honouring `llm_tier_operational: false`
    unconditionally would let any row opt out of the gate with one field -- which is the
    trust-a-derived-value failure this lint's own design note warns about. A row that
    disclaims the tier while its counters record real calls is judged by the primitives."""
    lying = {
        "game": "zz00",
        "llm_enabled": True,
        "llm_tier_operational": False,  # the stamp says "no LLM tier here"
        "llm": {"calls": 6, "responses": 0, "errors": 6},  # ...the counters disagree
        "generator_healthy_after": False,
        "llm_on_row_valid": False,
    }
    codes = {f["code"] for f in lint.check_row(lying)}
    assert "DEAD_GENERATOR" in codes, (
        f"a false llm_tier_operational stamp silenced the gate on a dead generator: {codes}"
    )

    honest_stub = {
        "game": "zz00",
        "llm_enabled": True,
        "llm_tier_operational": False,
        "generator_is_stub": True,
        "llm": {"calls": -1, "responses": -1, "errors": -1},
        "generator_healthy_after": None,
        "llm_on_row_valid": False,
    }
    assert lint.check_row(honest_stub) == [], lint.check_row(honest_stub)
