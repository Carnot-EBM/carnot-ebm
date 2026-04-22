"""Tests for Experiment 738 — Step-Level JEPAProbe + Tier 2 Cross-Session Memory.

Coverage targets:
- test_gated_blocked_path: gate fail → correct artifact written, status="gated_blocked"  (REQ-VER-038)
- test_step_extraction_produces_one_tensor_per_segment: extract_step_states returns one hidden
  state per CoT segment  (REQ-VER-038-1, REQ-VER-038-2)
- test_pool_states_produces_correct_shape: pool_states max-pools to (hidden_dim,)  (REQ-VER-038-3)
- test_pool_states_is_elementwise_max: pool output >= every input element  (REQ-VER-038-3)
- test_session_memory_persist_writes_correct_schema: persist() writes valid JSON with
  schema "carnot.session_memory_relay.v1"  (REQ-FR11-005-1)
- test_session_memory_load_replays_templates: load() calls replay_template so templates
  are immediately active in S2  (REQ-FR11-005)
- test_session_memory_load_missing_file_returns_zero: load() returns 0 on missing file  (REQ-FR11-005-3)
- test_replay_template_activates_in_library: replay_template sets count >= min_frequency  (REQ-FR11-005-4)
- test_three_session_precision_monotone: 3-session simulation has templates_replayed_in_s2 > 0  (REQ-FR11-006)
"""

from __future__ import annotations

import json
import pathlib
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# test_gated_blocked_path  (REQ-VER-038)
# ---------------------------------------------------------------------------


def test_gated_blocked_path(tmp_path):
    """When Exp 734 gate is NOT operational, Exp 738 writes gated_blocked artifact and stops.

    Spec: REQ-VER-038 (gate enforcement), SCENARIO-VER-047
    """
    import sys

    scripts_dir = str(pathlib.Path(__file__).resolve().parents[2] / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    deliverable = tmp_path / "results" / "experiment_738_step_probe_tier2_memory.json"
    (tmp_path / "results").mkdir(parents=True, exist_ok=True)

    # Write a gate file that does NOT pass.
    gate_path = tmp_path / "results" / "experiment_734_fr11_tier21_relay.json"
    gate_path.write_text(json.dumps({"honest_verdict": "something_else"}))

    # Patch _GATE_PATH and _REPO_ROOT in the experiment module.
    import importlib
    import experiment_738_step_probe_tier2_memory as exp738  # noqa: PLC0415

    with (
        patch.object(exp738, "_GATE_PATH", gate_path),
        patch.object(exp738, "_REPO_ROOT", tmp_path),
        patch.object(exp738, "_DELIVERABLE", str(deliverable.relative_to(tmp_path))),
    ):
        exp738.main()

    assert deliverable.exists(), "Blocked artifact must be written"
    data = json.loads(deliverable.read_text())
    assert data["status"] == "gated_blocked"
    assert data["honest_verdict"] == "gated_blocked_fr11_relay_not_operational"
    assert data["gate_source"] == "exp734"
    assert "schema" in data


# ---------------------------------------------------------------------------
# test_step_extraction_produces_one_tensor_per_segment  (REQ-VER-038-1, REQ-VER-038-2)
# ---------------------------------------------------------------------------


def test_step_extraction_produces_one_tensor_per_segment():
    """extract_step_states returns one (1024,) array per detected CoT boundary segment.

    Spec: REQ-VER-038-1, REQ-VER-038-2, SCENARIO-VER-047
    """
    from carnot.samplers.jepa_reasoner_probe import JEPAReasonerProbe

    probe = JEPAReasonerProbe(model_name="synthetic", layer_index=16, device="cpu")

    # Inject a mock model/tokenizer so no real weights are loaded.
    # We need extract_hidden_state to return a deterministic (1024,) array.
    def fake_extract(text):
        # Return a deterministic array based on text length to make each call distinct.
        rng = np.random.default_rng(len(text))
        return rng.standard_normal(1024).astype(np.float32)

    # Patch extract_hidden_state on the instance.
    probe.extract_hidden_state = fake_extract  # type: ignore[method-assign]

    # 3-sentence CoT text should produce 3 segments.
    text = "First step is done. Second step completes. Third step follows."
    states = probe.extract_step_states(text)

    assert len(states) == 3, f"Expected 3 states for 3 sentences, got {len(states)}"
    for i, s in enumerate(states):
        assert s.shape == (1024,), f"State {i} shape should be (1024,), got {s.shape}"


def test_step_extraction_fallback_on_no_boundaries():
    """extract_step_states returns exactly 1 state when text has no CoT boundaries.

    Spec: REQ-VER-038-2
    """
    from carnot.samplers.jepa_reasoner_probe import JEPAReasonerProbe

    probe = JEPAReasonerProbe(model_name="synthetic", layer_index=16, device="cpu")

    def fake_extract(text):
        return np.zeros(1024, dtype=np.float32)

    probe.extract_hidden_state = fake_extract  # type: ignore[method-assign]

    states = probe.extract_step_states("no separators here")
    assert len(states) == 1, "Single segment → single state"


# ---------------------------------------------------------------------------
# test_pool_states_produces_correct_shape  (REQ-VER-038-3)
# ---------------------------------------------------------------------------


def test_pool_states_produces_correct_shape():
    """pool_states returns shape (hidden_dim,) regardless of n_steps.

    Spec: REQ-VER-038-3
    """
    from carnot.samplers.jepa_reasoner_probe import JEPAReasonerProbe

    rng = np.random.default_rng(7)
    states = [rng.standard_normal(1024).astype(np.float32) for _ in range(5)]
    pooled = JEPAReasonerProbe.pool_states(states)
    assert pooled.shape == (1024,), f"pool_states shape should be (1024,), got {pooled.shape}"


def test_pool_states_is_elementwise_max():
    """pool_states output >= every element of every input state (element-wise max).

    Spec: REQ-VER-038-3
    """
    from carnot.samplers.jepa_reasoner_probe import JEPAReasonerProbe

    rng = np.random.default_rng(13)
    states = [rng.standard_normal(1024).astype(np.float32) for _ in range(4)]
    pooled = JEPAReasonerProbe.pool_states(states)
    for s in states:
        assert np.all(pooled >= s - 1e-6), "pooled must be >= each individual state element-wise"


# ---------------------------------------------------------------------------
# test_session_memory_persist_writes_correct_schema  (REQ-FR11-005-1)
# ---------------------------------------------------------------------------


def test_session_memory_persist_writes_correct_schema():
    """SessionMemory.persist() writes a JSON file with schema 'carnot.session_memory_relay.v1'.

    Spec: REQ-FR11-005, REQ-FR11-005-1, SCENARIO-FR11-005
    """
    from carnot.pipeline.session_memory import SessionMemory
    from carnot.pipeline.fr11_event_bus import ViolationEvent
    from carnot.pipeline.constraint_template_library import ConstraintTemplateLibrary
    from datetime import datetime, timezone

    with tempfile.TemporaryDirectory() as tmp_dir:
        mem = SessionMemory(storage_dir=tmp_dir, model_id="test-model")
        lib = ConstraintTemplateLibrary()
        lib.register_builtin_templates()

        # Fire 6 carry_check violations to cross min_frequency=5.
        for i in range(6):
            ev = ViolationEvent(
                query_id=f"q_{i}",
                step_index=0,
                energy_score=0.5,
                probe_confidence=0.8,
                constraint_type="carry_check",
                question_domain="arithmetic",
                timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            )
            mem.on_violation(ev, lib)

        persist_path = str(pathlib.Path(tmp_dir) / "relay.json")
        mem.persist(persist_path)

        assert pathlib.Path(persist_path).exists(), "persist() must write the file"
        payload = json.loads(pathlib.Path(persist_path).read_text())

        assert payload["schema"] == "carnot.session_memory_relay.v1"
        assert "violations_by_type" in payload
        assert "template_keys" in payload
        # carry_check should be in template_keys (6 violations >= min_frequency=5).
        assert "carry_check" in payload["template_keys"]


# ---------------------------------------------------------------------------
# test_session_memory_load_replays_templates  (REQ-FR11-005)
# ---------------------------------------------------------------------------


def test_session_memory_load_replays_templates():
    """SessionMemory.load() calls replay_template() for each key, making templates active in S2.

    Spec: REQ-FR11-005, SCENARIO-FR11-005
    """
    from carnot.pipeline.session_memory import SessionMemory
    from carnot.pipeline.fr11_event_bus import ViolationEvent
    from carnot.pipeline.constraint_template_library import ConstraintTemplateLibrary
    from datetime import datetime, timezone

    with tempfile.TemporaryDirectory() as tmp_dir:
        persist_path = str(pathlib.Path(tmp_dir) / "relay.json")

        # --- Session 1: accumulate and persist ---
        lib_s1 = ConstraintTemplateLibrary()
        lib_s1.register_builtin_templates()
        mem_s1 = SessionMemory(storage_dir=tmp_dir + "/s1", model_id="test-model")
        for i in range(6):
            ev = ViolationEvent(
                query_id=f"q_{i}",
                step_index=0,
                energy_score=0.5,
                probe_confidence=0.8,
                constraint_type="carry_check",
                question_domain="arithmetic",
                timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            )
            mem_s1.on_violation(ev, lib_s1)
        mem_s1.persist(persist_path)

        # --- Session 2: load and verify template is active ---
        lib_s2 = ConstraintTemplateLibrary()
        lib_s2.register_builtin_templates()
        mem_s2 = SessionMemory(storage_dir=tmp_dir + "/s2", model_id="test-model")
        n_replayed = mem_s2.load_relay(persist_path, lib_s2)

        assert n_replayed > 0, "At least one template must be replayed"
        active = lib_s2.get_active_templates("test-model")
        active_keys = {t.pattern_key for t in active}
        assert "carry_check" in active_keys, "carry_check must be active after replay"


# ---------------------------------------------------------------------------
# test_session_memory_load_missing_file_returns_zero  (REQ-FR11-005-3)
# ---------------------------------------------------------------------------


def test_session_memory_load_missing_file_returns_zero():
    """SessionMemory.load() returns 0 silently when the file doesn't exist.

    Spec: REQ-FR11-005-3
    """
    from carnot.pipeline.session_memory import SessionMemory
    from carnot.pipeline.constraint_template_library import ConstraintTemplateLibrary

    with tempfile.TemporaryDirectory() as tmp_dir:
        mem = SessionMemory(storage_dir=tmp_dir, model_id="test-model")
        lib = ConstraintTemplateLibrary()
        lib.register_builtin_templates()
        # File does not exist — must not raise.
        result = mem.load_relay("/nonexistent/path/relay.json", lib)
        assert result == 0


# ---------------------------------------------------------------------------
# test_replay_template_activates_in_library  (REQ-FR11-005-4)
# ---------------------------------------------------------------------------


def test_replay_template_activates_in_library():
    """ConstraintTemplateLibrary.replay_template() sets count >= min_frequency for the template.

    Spec: REQ-FR11-005-4
    """
    from carnot.pipeline.constraint_template_library import ConstraintTemplateLibrary

    lib = ConstraintTemplateLibrary()
    lib.register_builtin_templates()

    # carry_check has min_frequency=5; verify it's not active before replay.
    active_before = lib.get_active_templates("my-model")
    assert all(t.pattern_key != "carry_check" for t in active_before), (
        "carry_check should not be active before any replay"
    )

    lib.replay_template("carry_check", "my-model")
    active_after = lib.get_active_templates("my-model")
    active_keys = {t.pattern_key for t in active_after}
    assert "carry_check" in active_keys, "carry_check must be active after replay_template()"


# ---------------------------------------------------------------------------
# test_three_session_precision_monotone  (REQ-FR11-006)
# ---------------------------------------------------------------------------


def test_three_session_precision_monotone():
    """3-session simulation: templates_replayed_in_s2 > 0 (FR-11 Tier 2 relay functional).

    Spec: REQ-FR11-006, REQ-FR11-006-1, SCENARIO-FR11-006
    """
    import sys

    scripts_dir = str(pathlib.Path(__file__).resolve().parents[2] / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    import experiment_738_step_probe_tier2_memory as exp738  # noqa: PLC0415

    result = exp738._run_three_session_simulation()

    assert result["templates_replayed_in_s2"] > 0, (
        "At least one template from S1 must be replayed in S2 "
        f"(got templates_replayed_in_s2={result['templates_replayed_in_s2']})"
    )
    assert result["fr11_tier2_relay_functional"] is True
    assert result["persist_file_written"] is True
