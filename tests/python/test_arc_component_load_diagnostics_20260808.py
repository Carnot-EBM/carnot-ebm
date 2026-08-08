"""Spec: REQ-ARC-WMTE-6230.

Regression tests for the silent component-degradation witness gap.

docs/research-notes/live-agent-adversarial-review-2026-08-08.md, "Gaps" section, major
finding 3:

  "Declared-ON search components (candidate router, frame-change scorer, and therefore the
  action-effect prior) degrade to None through bare `except Exception` fallbacks with no
  counter, no stderr line, and no witness field, while SUBMITTED_AGENT_CONFIG declares them
  enabled ... Fix: record load outcomes (`*_loaded` booleans plus the swallowed repr) in the
  generator-liveness witness; print one greppable line per fallback."

THE FIX. `_load_submitted_candidate_router` and `_load_submitted_frame_change_scorer` set a
per-call diagnostic dict on a THREAD-LOCAL side channel (`_component_load_diagnostics`) and
print one greppable line ("CANDIDATE ROUTER LOAD ..." / "FRAME-CHANGE SCORER LOAD FAILED
...") on every fallback or failure. `E3AgentPolicy.__init__` reads its own thread's slot
immediately after calling each loader (same thread, no race) and stores the result on
`self._candidate_router_load_diagnostics` / `self._frame_change_scorer_load_diagnostics`,
which `generator_liveness_witness()` then surfaces unconditionally as
`candidate_router_loaded` / `candidate_router_load_diagnostics` /
`frame_change_scorer_loaded` / `frame_change_scorer_load_diagnostics`.

A THREAD LOCAL rather than a plain module-level dict, deliberately: the scored swarm runs one
thread per game (scripts/kaggle/submission_kernel/main.py), and a bare global would let two
concurrently-constructing games' diagnostics interleave and read back each other's data.
TestThreadLocalIsolation below is the regression test for that specific failure mode.

BOTH LOADER FUNCTIONS' RETURN TYPE IS UNCHANGED (still `Any | None`, never a tuple). A first
attempt at this fix changed the return type to `tuple[Any | None, dict]`; a repo-wide grep
before committing found 10+ other callers -- experiments, tests, and the production module
`arc_reactive_verifier_filter.py` -- that all depend on the bare object-or-None shape, so
that attempt was reverted in favour of the side-channel design tested here.
TestNoSignatureChange is the regression guard for that specific mistake not recurring.
"""

from __future__ import annotations

import threading

from carnot.agentic import arc_competition_agent as agent
from carnot.agentic.arc_competition_agent import E3AgentPolicy


class TestCandidateRouterDiagnostics:
    def test_default_path_loads_discriminative_router_and_reports_loaded(self, capsys):
        agent._component_load_diagnostics.__dict__.clear()
        router = agent._load_submitted_candidate_router(game_id="zz01")
        assert router is not None
        diag = agent._component_load_diagnostics.candidate_router
        assert diag["loaded"] is True
        assert diag["discriminative_loaded"] is True
        assert diag["sge_attempted"] is False
        captured = capsys.readouterr()
        assert "CANDIDATE ROUTER LOAD" not in captured.out

    def test_sge_not_requested_by_default(self):
        assert agent._sge_candidate_router_requested() is False

    def test_sge_failure_falls_through_and_reports_sge_error(self, capsys, monkeypatch):
        monkeypatch.setattr(agent, "_sge_candidate_router_requested", lambda: True)
        monkeypatch.setattr(
            agent,
            "_load_sge_candidate_router",
            lambda game_id: (_ for _ in ()).throw(RuntimeError("sge boom")),
        )
        agent._component_load_diagnostics.__dict__.clear()
        router = agent._load_submitted_candidate_router(game_id="zz02")
        assert router is not None  # falls through to the discriminative router
        diag = agent._component_load_diagnostics.candidate_router
        assert diag["sge_attempted"] is True
        assert "sge boom" in diag["sge_error"]
        assert diag["discriminative_loaded"] is True
        assert diag["loaded"] is True
        captured = capsys.readouterr()
        assert "CANDIDATE ROUTER LOAD FALLBACK (game=zz02)" in captured.out
        assert "sge boom" in captured.out

    def test_both_paths_failing_reports_unrouted_and_greppable_line(self, capsys, monkeypatch):
        monkeypatch.setattr(
            agent.arc_discriminative_router,
            "load_online_click_target_router",
            lambda root=None: (_ for _ in ()).throw(RuntimeError("disc boom")),
        )
        agent._component_load_diagnostics.__dict__.clear()
        router = agent._load_submitted_candidate_router(game_id="zz03")
        assert router is None
        diag = agent._component_load_diagnostics.candidate_router
        assert diag["loaded"] is False
        assert diag["sge_attempted"] is False
        assert "disc boom" in diag["discriminative_error"]
        captured = capsys.readouterr()
        assert "CANDIDATE ROUTER LOAD FAILED (game=zz03)" in captured.out
        assert "UNROUTED" in captured.out
        assert "disc boom" in captured.out


class TestFrameChangeScorerDiagnostics:
    def test_disabled_reports_disabled_no_print(self, capsys, monkeypatch):
        monkeypatch.setattr(agent, "SUBMITTED_FRAME_CHANGE_PREDICTOR_ENABLED", False)
        agent._component_load_diagnostics.__dict__.clear()
        scorer = agent._load_submitted_frame_change_scorer()
        assert scorer is None
        diag = agent._component_load_diagnostics.frame_change_scorer
        assert diag == {"enabled": False, "loaded": False, "error": None}
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_enabled_but_loader_returns_none_reports_failure(self, capsys, monkeypatch):
        monkeypatch.setattr(agent, "SUBMITTED_FRAME_CHANGE_PREDICTOR_ENABLED", True)
        monkeypatch.setattr(agent, "load_live_action_effect_scorer", lambda root=None: None)
        agent._component_load_diagnostics.__dict__.clear()
        scorer = agent._load_submitted_frame_change_scorer()
        assert scorer is None
        diag = agent._component_load_diagnostics.frame_change_scorer
        assert diag["enabled"] is True
        assert diag["loaded"] is False
        assert diag["error"] is None
        captured = capsys.readouterr()
        assert "FRAME-CHANGE SCORER LOAD FAILED" in captured.out
        assert "inert" in captured.out

    def test_enabled_but_loader_raises_reports_repr(self, capsys, monkeypatch):
        monkeypatch.setattr(agent, "SUBMITTED_FRAME_CHANGE_PREDICTOR_ENABLED", True)
        monkeypatch.setattr(
            agent,
            "load_live_action_effect_scorer",
            lambda root=None: (_ for _ in ()).throw(ValueError("scorer boom")),
        )
        agent._component_load_diagnostics.__dict__.clear()
        scorer = agent._load_submitted_frame_change_scorer()
        assert scorer is None
        diag = agent._component_load_diagnostics.frame_change_scorer
        assert diag["loaded"] is False
        assert "scorer boom" in diag["error"]
        captured = capsys.readouterr()
        assert "FRAME-CHANGE SCORER LOAD FAILED" in captured.out
        assert "scorer boom" in captured.out

    def test_enabled_and_succeeds_reports_loaded_true(self):
        agent._component_load_diagnostics.__dict__.clear()
        scorer = agent._load_submitted_frame_change_scorer()
        assert scorer is not None
        diag = agent._component_load_diagnostics.frame_change_scorer
        assert diag == {"enabled": True, "loaded": True, "error": None}


class TestInitWiring:
    def test_default_path_populates_diagnostics_from_the_loaders(self):
        p = E3AgentPolicy("lp85", proposer=object(), target_levels=1, value_head=None)
        assert p._candidate_router_load_diagnostics.get("loaded") is True
        assert p._frame_change_scorer_load_diagnostics.get("loaded") is True

    def test_caller_supplied_router_skips_the_loader(self):
        class FakeRouter:
            pass

        p = E3AgentPolicy(
            "lp85",
            proposer=object(),
            target_levels=1,
            value_head=None,
            candidate_router=FakeRouter(),
        )
        assert p._candidate_router_load_diagnostics == {"loaded": True, "caller_supplied": True}

    def test_caller_supplied_none_router_records_not_loaded(self):
        p = E3AgentPolicy(
            "lp85",
            proposer=object(),
            target_levels=1,
            value_head=None,
            candidate_router=None,
        )
        assert p._candidate_router_load_diagnostics == {
            "loaded": False,
            "caller_supplied": True,
        }

    def test_caller_supplied_frame_change_scorer_skips_the_loader(self):
        class FakeScorer:
            pass

        p = E3AgentPolicy(
            "lp85",
            proposer=object(),
            target_levels=1,
            value_head=None,
            frame_change_scorer=FakeScorer(),
        )
        assert p._frame_change_scorer_load_diagnostics == {
            "loaded": True,
            "caller_supplied": True,
        }


class TestWitnessFields:
    def test_witness_reports_diagnostics_on_the_proposer_none_early_return_path(self):
        p = E3AgentPolicy("lp85", proposer=None, target_levels=1, value_head=None)
        w = p.generator_liveness_witness()
        assert w["candidate_router_loaded"] == p._candidate_router_load_diagnostics.get("loaded")
        assert w["candidate_router_load_diagnostics"] == p._candidate_router_load_diagnostics
        assert w["frame_change_scorer_loaded"] == p._frame_change_scorer_load_diagnostics.get(
            "loaded"
        )
        assert w["frame_change_scorer_load_diagnostics"] == p._frame_change_scorer_load_diagnostics

    def test_witness_reports_diagnostics_on_the_proposer_present_path(self):
        p = E3AgentPolicy("lp85", proposer=object(), target_levels=1, value_head=None)
        w = p.generator_liveness_witness()
        assert w["candidate_router_loaded"] == p._candidate_router_load_diagnostics.get("loaded")
        assert w["frame_change_scorer_loaded"] == (
            p._frame_change_scorer_load_diagnostics.get("loaded")
        )

    def test_witness_reflects_a_degraded_router_as_false(self, monkeypatch):
        monkeypatch.setattr(
            agent.arc_discriminative_router,
            "load_online_click_target_router",
            lambda root=None: (_ for _ in ()).throw(RuntimeError("disc boom")),
        )
        p = E3AgentPolicy("lp85", proposer=object(), target_levels=1, value_head=None)
        w = p.generator_liveness_witness()
        assert w["candidate_router_loaded"] is False
        assert "disc boom" in w["candidate_router_load_diagnostics"]["discriminative_error"]


class TestNoSignatureChange:
    def test_candidate_router_loader_still_returns_bare_object_or_none(self):
        result = agent._load_submitted_candidate_router(game_id="zz04")
        assert result is None or not isinstance(result, tuple)

    def test_frame_change_scorer_loader_still_returns_bare_object_or_none(self):
        result = agent._load_submitted_frame_change_scorer()
        assert result is None or not isinstance(result, tuple)


class TestThreadLocalIsolation:
    def test_component_load_diagnostics_is_a_real_thread_local(self):
        # The regression this guards: swapping `_component_load_diagnostics` back to a plain
        # module-level dict (the naive design a future edit might "simplify" to) would still
        # pass every single-threaded test above while silently reintroducing the exact
        # cross-game race the review's swarm concurrency (one thread per game) requires this
        # to avoid. `threading.local()` gives each thread its own attribute namespace; a
        # plain dict does not.
        assert isinstance(agent._component_load_diagnostics, threading.local)

    def test_two_threads_calling_the_candidate_router_loader_do_not_cross_contaminate(
        self, monkeypatch
    ):
        # Deterministic, race-free variant: no shared mutable global is monkeypatched from
        # inside either thread. `_sge_candidate_router_requested` is patched ONCE, before
        # either thread starts, to a value both threads share safely; the per-thread outcome
        # is instead driven purely by the `game_id` argument each thread passes, which is
        # thread-local by construction (a normal Python argument, not shared state).
        monkeypatch.setattr(agent, "_sge_candidate_router_requested", lambda: True)

        def fake_sge_loader(game_id):
            if game_id == "thread_b":
                raise RuntimeError(f"sge boom for {game_id}")
            return object()

        monkeypatch.setattr(agent, "_load_sge_candidate_router", fake_sge_loader)

        results: dict[str, dict] = {}
        barrier = threading.Barrier(2)

        def run(game_id: str) -> None:
            barrier.wait(timeout=5)  # maximize the chance of genuine interleaving
            agent._load_submitted_candidate_router(game_id=game_id)
            results[game_id] = dict(agent._component_load_diagnostics.candidate_router)

        ta = threading.Thread(target=run, args=("thread_a",))
        tb = threading.Thread(target=run, args=("thread_b",))
        ta.start()
        tb.start()
        ta.join(timeout=5)
        tb.join(timeout=5)

        # thread_a's SGE call succeeds outright; thread_b's SGE call fails and falls through
        # to the (real, unpatched) discriminative router. If the diagnostic were shared
        # across threads instead of thread-local, one thread's read could pick up the
        # other's shape (e.g. thread_a showing a populated sge_error, or thread_b showing
        # sge_loaded=True) depending on which thread's write landed last.
        assert results["thread_a"]["sge_loaded"] is True
        assert results["thread_a"]["sge_error"] is None
        assert results["thread_b"]["sge_loaded"] is False
        assert "sge boom for thread_b" in results["thread_b"]["sge_error"]
        assert results["thread_b"]["discriminative_loaded"] is True
