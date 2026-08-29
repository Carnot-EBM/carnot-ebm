"""Gate-cascade settlement: deterministic gate failures retire at once.

Spec: REQ-CONDUCTOR-GATECASCADE-1, SCENARIO-CONDUCTOR-GATECASCADE-1-A..F
(openspec/capabilities/research-harnesses/spec.md).

Incident (2026-08-29, milestone 2026.08.589): exp6755 finished honestly
with environment_grammar_targetable_rows=21. exp6756's gate demanded
>=24. The value is frozen in a finished artifact, so no retry can pass,
yet the conductor re-evaluated the gate 3 times (2 min apart), retired
exp6756, then let the exp6757->exp6760 chain burn 3 GATE_BLOCK rows PER
LINK before settling (~18 min of spin, 16 log rows, runnable tasks
exp6761+ delayed).

The fix has three rules, each tested at the entry point the conductor
actually calls (evaluate_gates / pick_next_task), not at a mirror:
  1. evaluate_gates marks a failure deterministic (summary prefix
     "gate-unsat(final): ") when a FAILED gate read a FINISHED upstream
     artifact.
  2. pick_next_task retires a task on the FIRST terminal-marked
     GATE_BLOCK row instead of after MAX_FAILURES_PER_TASK rows.
  3. pick_next_task closes retirement transitively over gated_on edges,
     so a dead chain settles in one iteration.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import conductor_gates  # noqa: E402
import research_conductor  # noqa: E402

# ---------------------------------------------------------------------------
# The incident's exact gate input, reproduced hermetically. Values copied
# from results/experiment_6755_lossless_gguf_output_reparse.json and the
# exp6756 gated_on block in the 2026.08.589 roadmap.
# ---------------------------------------------------------------------------

EXP6755_ARTIFACT = {
    "experiment": 6755,
    "status": "complete",
    "honest_verdict": (
        "complete: lossless transport replay recovered 216/216 rows; "
        "11/216 exact-valid is a separate semantic outcome"
    ),
    "transport_reparse_ready": True,
    "environment_grammar_targetable_rows": 21,
}

EXP6756_TASK = {
    "id": "exp6756-environment-indexed-proof-grammar-fixture",
    "title": "Environment-indexed proof grammar runtime fixture",
    "gated_on": [
        {
            "upstream": "exp6755-lossless-gguf-output-reparse",
            "artifact_field": "transport_reparse_ready",
            "op": "==",
            "value": True,
        },
        {
            "upstream": "exp6755-lossless-gguf-output-reparse",
            "artifact_field": "environment_grammar_targetable_rows",
            "op": ">=",
            "value": 24,
        },
    ],
}


def _write_artifact(results_dir: Path, exp_num: int, payload: dict) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"experiment_{exp_num}_fixture.json").write_text(json.dumps(payload))


# ---------------------------------------------------------------------------
# Rule 1 — evaluate_gates marks deterministic failures.
# ---------------------------------------------------------------------------


def test_incident_input_marks_deterministic(tmp_path):
    """SCENARIO-CONDUCTOR-GATECASCADE-1-A: the exact exp6756/exp6755 input
    produces a terminal-marked summary. This is the incident regression."""
    _write_artifact(tmp_path, 6755, EXP6755_ARTIFACT)
    check = conductor_gates.evaluate_gates(EXP6756_TASK, results_dir=tmp_path)
    assert not check.passed
    assert check.summary.startswith(conductor_gates.GATE_UNSAT_FINAL_PREFIX)
    assert conductor_gates.gate_failure_is_deterministic(check)
    # The observed value must still be visible for diagnosis.
    assert "environment_grammar_targetable_rows" in check.summary


def test_healthy_roadmap_gets_no_marker(tmp_path):
    """SCENARIO-CONDUCTOR-GATECASCADE-1-B: passing gates never carry the
    marker. Threshold met -> normal satisfied summary."""
    passing = dict(EXP6755_ARTIFACT, environment_grammar_targetable_rows=30)
    _write_artifact(tmp_path, 6755, passing)
    check = conductor_gates.evaluate_gates(EXP6756_TASK, results_dir=tmp_path)
    assert check.passed
    assert check.summary == "2 gate(s) satisfied"
    assert not conductor_gates.gate_failure_is_deterministic(check)


@pytest.mark.parametrize(
    "upstream_state",
    [
        {"status": "running"},
        {"status": "blocked", "honest_verdict": "blocked_gate_check_failed"},
        {"status": "failed", "honest_verdict": "partial_bootstrap_only"},
    ],
)
def test_unfinished_upstream_stays_transient(tmp_path, upstream_state):
    """SCENARIO-CONDUCTOR-GATECASCADE-1-C: a failure against a bootstrap /
    blocked / failed upstream artifact keeps the retry budget. The upstream
    may re-run and rewrite the value, so no marker."""
    payload = dict(EXP6755_ARTIFACT, **upstream_state)
    payload["environment_grammar_targetable_rows"] = 0
    if "honest_verdict" not in upstream_state:
        payload.pop("honest_verdict", None)
    _write_artifact(tmp_path, 6755, payload)
    check = conductor_gates.evaluate_gates(EXP6756_TASK, results_dir=tmp_path)
    assert not check.passed
    assert not check.summary.startswith(conductor_gates.GATE_UNSAT_FINAL_PREFIX)
    assert not conductor_gates.gate_failure_is_deterministic(check)


def test_missing_upstream_artifact_stays_transient(tmp_path):
    """SCENARIO-CONDUCTOR-GATECASCADE-1-C (missing-file arm): no artifact on
    disk means the upstream has not run; the failure is not terminal."""
    check = conductor_gates.evaluate_gates(EXP6756_TASK, results_dir=tmp_path)
    assert not check.passed
    assert not check.summary.startswith(conductor_gates.GATE_UNSAT_FINAL_PREFIX)
    assert not conductor_gates.gate_failure_is_deterministic(check)


def test_missing_field_on_finished_artifact_is_deterministic(tmp_path):
    """A FINISHED artifact that lacks the gated field will never grow it.
    That failure is terminal too."""
    payload = dict(EXP6755_ARTIFACT)
    payload.pop("environment_grammar_targetable_rows")
    _write_artifact(tmp_path, 6755, payload)
    check = conductor_gates.evaluate_gates(EXP6756_TASK, results_dir=tmp_path)
    assert not check.passed
    assert check.summary.startswith(conductor_gates.GATE_UNSAT_FINAL_PREFIX)


def test_status_alone_marks_final(tmp_path):
    """The status arm of _upstream_is_final stands on its own: a finished
    status with no honest_verdict still makes the failure terminal."""
    payload = dict(EXP6755_ARTIFACT)
    payload.pop("honest_verdict")
    _write_artifact(tmp_path, 6755, payload)
    check = conductor_gates.evaluate_gates(EXP6756_TASK, results_dir=tmp_path)
    assert check.summary.startswith(conductor_gates.GATE_UNSAT_FINAL_PREFIX)


def test_verdict_prefix_alone_marks_final(tmp_path):
    """The verdict arm stands on its own: a terminal-prefix honest_verdict
    with no status field still makes the failure terminal."""
    payload = dict(EXP6755_ARTIFACT)
    payload.pop("status")
    payload["honest_verdict"] = "passed_reparse_confirmed_216_rows"
    _write_artifact(tmp_path, 6755, payload)
    check = conductor_gates.evaluate_gates(EXP6756_TASK, results_dir=tmp_path)
    assert check.summary.startswith(conductor_gates.GATE_UNSAT_FINAL_PREFIX)


def test_prefix_constants_match_across_modules():
    """The conductor's log-row matcher and the gate module's summary prefix
    are separate constants (no cross-import at module load). They must agree
    or the marker becomes decorative."""
    assert conductor_gates.GATE_UNSAT_FINAL_PREFIX.startswith(
        research_conductor._GATE_BLOCK_TERMINAL_DETAIL_PREFIXES[0]
    )
    # log_step truncates details to 80 chars; the marker must survive that.
    assert len(conductor_gates.GATE_UNSAT_FINAL_PREFIX) < 80
    # _upstream_is_final lowercases the verdict before comparing, so every
    # prefix must stay lowercase or the arm silently never fires.
    assert all(p == p.lower() for p in conductor_gates._FINAL_VERDICT_PREFIXES)


def test_research_step_logs_summary_verbatim():
    """Call-site guard: research_step must log gate_check.summary itself.
    If the call site is replaced with a constant, the marker never reaches
    the log and rule 2 goes dead while unit tests stay green."""
    source = (SCRIPTS_DIR / "research_conductor.py").read_text()
    assert 'log_step(task["title"], "GATE_BLOCK", gate_check.summary)' in source


# ---------------------------------------------------------------------------
# Rules 2 + 3 — pick_next_task, driven at the real entry point.
# ---------------------------------------------------------------------------


def _row(title: str, status: str, details: str) -> str:
    """Mirror log_step's exact row format, truncations included."""
    return f"| 2026-08-29 20:47 UTC | {title[:50]} | {status} | {details[:80]} |"


TITLE_6755 = "Lossless GGUF output boundary and 216-row reparse"
TITLE_6756 = "Environment-indexed proof grammar runtime fixture"

ACTIVATION_ROW = _row("Milestone 2026.08.589 activated", "OK", "13 tasks queued")
EXP6755_OK_ROW = _row(TITLE_6755, "OK", "106 passed, 1 warning in 237.95s (0:03:57)")
# The row the fixed conductor writes for the incident (marker + truncation).
EXP6756_TERMINAL_ROW = _row(
    TITLE_6756,
    "GATE_BLOCK",
    "gate-unsat(final): 1 of 2 gate(s) failed; first failure: "
    "exp6755-lossless-gguf-output-reparse.environment_grammar_targetable_rows",
)
# The pre-fix row shape (no marker) for the transient contrast case.
EXP6756_PLAIN_ROW = _row(
    TITLE_6756,
    "GATE_BLOCK",
    "1 of 2 gate(s) failed; first failure: "
    "exp6755-lossless-gguf-output-reparse.environment_grammar_targetable_rows",
)


def _incident_tasks() -> list[dict]:
    """The 2026.08.589 chain shape: done root, doomed link, two dependents,
    one runnable ungated task."""
    return [
        {
            "id": "exp6755-lossless-gguf-output-reparse",
            "title": TITLE_6755,
        },
        EXP6756_TASK,
        {
            "id": "exp6757-dccd-environment-grammar-ab",
            "title": "Three-model DCCD environment-grammar A/B",
            "gated_on": [
                {
                    "upstream": "exp6756-environment-indexed-proof-grammar-fixture",
                    "artifact_field": "dynamic_proof_grammar_ready",
                    "op": "==",
                    "value": True,
                }
            ],
        },
        {
            "id": "exp6758-proof-transport-independent-audit",
            "title": "Independent proof-transport and support audit",
            "gated_on": [
                {
                    "upstream": "exp6757-dccd-environment-grammar-ab",
                    "artifact_field": "proof_transport_ok",
                    "op": "==",
                    "value": True,
                }
            ],
        },
        {
            "id": "exp6761-procedural-memory-stream",
            "title": "Procedural memory stream",
        },
    ]


@pytest.fixture()
def conductor_env(tmp_path, monkeypatch):
    """Pin the conductor's filesystem seams to tmp_path and install the
    incident task list. Everything else runs the real code."""
    log_file = tmp_path / "conductor-log.md"
    log_file.write_text("| Timestamp | Task | Status | Details |\n|---|---|---|---|\n")
    monkeypatch.setattr(research_conductor, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(research_conductor, "CONDUCTOR_LOG", log_file)
    monkeypatch.setattr(research_conductor, "RESEARCH_TASKS", _incident_tasks())
    monkeypatch.setattr(research_conductor, "_ensure_tasks_loaded", lambda: None)
    return log_file


def test_one_terminal_row_retires_and_settles_chain(conductor_env):
    """SCENARIO-CONDUCTOR-GATECASCADE-1-D: after ONE terminal-marked
    GATE_BLOCK row, the doomed task and its whole downstream chain are
    skipped and the next runnable task is returned in the SAME iteration."""
    log = "\n".join([ACTIVATION_ROW, EXP6755_OK_ROW, EXP6756_TERMINAL_ROW])
    picked = research_conductor.pick_next_task(log)
    assert picked is not None
    assert picked["id"] == "exp6761-procedural-memory-stream"
    # The dependents were pre-emptively skipped with the terminal reason,
    # so the NEXT iteration retires them from the log without rescanning.
    logged = conductor_env.read_text()
    assert "Pre-emptive skip: upstream retired" in logged
    assert "Three-model DCCD environment-grammar A/B" in logged
    assert "Independent proof-transport and support audit" in logged


def test_plain_gate_block_keeps_retry_budget(conductor_env):
    """SCENARIO-CONDUCTOR-GATECASCADE-1-E: one UNMARKED gate-block row does
    NOT retire the task — a transient block keeps its 3 tries (the
    pre-2026-08-29 semantics, still correct for unfinished upstreams)."""
    log = "\n".join([ACTIVATION_ROW, EXP6755_OK_ROW, EXP6756_PLAIN_ROW])
    picked = research_conductor.pick_next_task(log)
    assert picked is not None
    assert picked["id"] == "exp6756-environment-indexed-proof-grammar-fixture"


def test_three_plain_gate_blocks_still_retire(conductor_env):
    """SCENARIO-CONDUCTOR-GATECASCADE-1-F: MAX_FAILURES_PER_TASK unmarked
    rows retire the task exactly as before (the 2026-04-29 rule holds)."""
    log = "\n".join([ACTIVATION_ROW, EXP6755_OK_ROW] + [EXP6756_PLAIN_ROW] * 3)
    picked = research_conductor.pick_next_task(log)
    assert picked is not None
    assert picked["id"] == "exp6761-procedural-memory-stream"


def test_preemptive_skip_row_silences_repeat_logging(conductor_env):
    """A task already skip-logged once for a retired upstream is retired by
    that row alone. Later iterations stay silent, instead of re-logging the
    same skip every 2 minutes for the rest of the milestone."""
    fail_row = _row(TITLE_6756, "FAIL", "Codex CLI error: Hard wall-clock cap after 4800s")
    skip_row = _row(
        "Three-model DCCD environment-grammar A/B",
        "GATE_BLOCK",
        "Pre-emptive skip: upstream retired (exp6756-environment-indexed-proof-grammar-fixture)",
    )
    log = "\n".join([ACTIVATION_ROW, EXP6755_OK_ROW] + [fail_row] * 3 + [skip_row])
    picked = research_conductor.pick_next_task(log)
    assert picked is not None
    assert picked["id"] == "exp6761-procedural-memory-stream"
    # exp6757 must NOT be skip-logged again — its one existing row retired it.
    logged = conductor_env.read_text()
    assert "Three-model DCCD environment-grammar A/B" not in logged


def test_closure_settles_deep_chain_in_one_call(conductor_env, monkeypatch):
    """SCENARIO-CONDUCTOR-GATECASCADE-1-D (depth arm): with the chain root
    retired by three plain FAILs, one pick_next_task call must skip EVERY
    transitive dependent, not just the direct one. Pre-fix, the second-level
    dependent was returned and burned its own 3 gate evaluations."""
    fail_row = _row(TITLE_6756, "FAIL", "Codex CLI error: Hard wall-clock cap after 4800s")
    log = "\n".join([ACTIVATION_ROW, EXP6755_OK_ROW] + [fail_row] * 3)
    picked = research_conductor.pick_next_task(log)
    assert picked is not None
    # exp6757 (direct) AND exp6758 (transitive) are both skipped.
    assert picked["id"] == "exp6761-procedural-memory-stream"
