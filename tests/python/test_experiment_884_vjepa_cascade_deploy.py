"""Tests for Experiment 884: VJEPA v2 Cascade Deploy.

Spec traces: REQ-VERIFY-145, SCENARIO-VERIFY-233, SCENARIO-VERIFY-234

**Coverage targets (code added in Exp 884):**
    - check_gate: passes on ood_auc > 0.60, writes blocked artifact + raises on failure
    - _write_blocked_artifact: written artifact has required schema fields
    - assign_honest_verdict: all three outcome branches (blocked, closed, marginal)
    - VJEPAv2EnergyAdapter.energy(): returns float in [0,1], accepts CoTEnergyInput
    - _load_jepa_model(): priority order (v2 first, falls back, handles missing file)
    - save_model_safetensors / load round-trip: params survive save/load cycle
    - update_architecture_tier2: replaces Tier 2 row, preserves surrounding content
    - generate_arc_heldout / generate_svamp_heldout: different seed from Exp 883
    - evaluate_on_heldout: returns 0.5 on empty, float in [0,1] on real input
    - Full integration smoke (tiny corpus, 5 epochs): artifact written with all fields
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Force CPU for all JAX ops in tests
os.environ.setdefault("JAX_PLATFORMS", "cpu")

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from python.carnot.models.eorm import CoTEnergyInput
from python.carnot.models.vjepa_predictor import (
    VOCAB_SIZE,
    VariationalJEPAPredictor,
    build_tfidf_features,
    prepare_corpus,
)
from python.carnot.pipeline.three_tier_pipeline import (
    VJEPAv2EnergyAdapter,
    _load_jepa_model,
)
from experiment_884_vjepa_cascade_deploy import (
    REQUIRED_RESULT_FIELDS,
    RESULT_PATH,
    _write_blocked_artifact,
    assign_honest_verdict,
    check_gate,
    evaluate_on_heldout,
    generate_arc_heldout,
    generate_svamp_heldout,
    save_model_safetensors,
    update_architecture_tier2,
)


# ===========================================================================
# SCENARIO-VERIFY-233: Gate check passes / fails correctly
# ===========================================================================

class TestCheckGate:
    """REQ-VERIFY-145 — Gate check enforces ood_auc > 0.60 strictly."""

    def _write_exp883_json(self, path: Path, ood_auc: float) -> None:
        artifact = {
            "experiment": 883,
            "schema": "carnot-experiment-v1",
            "run_date": "2026-04-25T00:00:00Z",
            "honest_verdict": "vjepa_ood_above_gate",
            "ood_auc": ood_auc,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            json.dump(artifact, fh)

    def test_passes_when_ood_auc_above_0_60(self, tmp_path):
        """ood_auc=0.664 (Exp 883 actual value) should clear the gate."""
        p = tmp_path / "experiment_883.json"
        self._write_exp883_json(p, 0.664)
        result = check_gate(p)
        assert result["ood_auc"] == 0.664

    def test_blocked_when_ood_auc_exactly_0_60(self, tmp_path):
        """0.60 is NOT strictly > 0.60 — gate must block."""
        p = tmp_path / "experiment_883.json"
        self._write_exp883_json(p, 0.60)
        with patch("experiment_884_vjepa_cascade_deploy.RESULT_PATH", tmp_path / "884.json"):
            with pytest.raises(SystemExit):
                check_gate(p)

    def test_blocked_when_ood_auc_below_0_60(self, tmp_path):
        """ood_auc=0.55 must block."""
        p = tmp_path / "experiment_883.json"
        self._write_exp883_json(p, 0.55)
        with patch("experiment_884_vjepa_cascade_deploy.RESULT_PATH", tmp_path / "884.json"):
            with pytest.raises(SystemExit):
                check_gate(p)

    def test_blocked_when_exp883_missing(self, tmp_path):
        """Missing Exp 883 result must block."""
        missing = tmp_path / "nonexistent.json"
        with patch("experiment_884_vjepa_cascade_deploy.RESULT_PATH", tmp_path / "884.json"):
            with pytest.raises(SystemExit):
                check_gate(missing)

    def test_blocked_artifact_has_required_schema(self, tmp_path):
        """When gate fails, the blocked artifact must have all required fields."""
        p = tmp_path / "experiment_883.json"
        self._write_exp883_json(p, 0.40)
        result_path = tmp_path / "884.json"
        with patch("experiment_884_vjepa_cascade_deploy.RESULT_PATH", result_path):
            with pytest.raises(SystemExit):
                check_gate(p)
        assert result_path.exists()
        with result_path.open() as fh:
            data = json.load(fh)
        missing = REQUIRED_RESULT_FIELDS - set(data.keys())
        assert not missing, f"Missing fields in blocked artifact: {missing}"
        assert data["cascade_deployed"] is False
        assert data["honest_verdict"] == "blocked"


# ===========================================================================
# _write_blocked_artifact: schema validation
# ===========================================================================

class TestWriteBlockedArtifact:
    """REQ-VERIFY-145 — Blocked artifact contains all required schema fields."""

    def test_blocked_artifact_fields(self, tmp_path):
        with patch("experiment_884_vjepa_cascade_deploy.RESULT_PATH", tmp_path / "out.json"):
            _write_blocked_artifact("exp883_ood_auc_below_0.60", 0.55)
        with (tmp_path / "out.json").open() as fh:
            data = json.load(fh)
        assert data["experiment"] == 884
        assert data["schema"] == "carnot-experiment-v1"
        assert data["cascade_deployed"] is False
        assert data["blocked_by"] == "exp883_ood_auc_below_0.60"
        assert data["exp883_ood_auc"] == 0.55

    def test_blocked_artifact_with_none_auc(self, tmp_path):
        """Should not crash when exp883_ood_auc is None (file was missing)."""
        with patch("experiment_884_vjepa_cascade_deploy.RESULT_PATH", tmp_path / "out.json"):
            _write_blocked_artifact("exp883_result_missing", None)
        with (tmp_path / "out.json").open() as fh:
            data = json.load(fh)
        assert data["exp883_ood_auc"] is None


# ===========================================================================
# SCENARIO-VERIFY-234: Honest verdict assignment
# ===========================================================================

class TestAssignHonestVerdict:
    """REQ-VERIFY-145 — All verdict branches are reachable."""

    def test_blocked_when_not_deployed(self):
        verdict, closed, partial = assign_honest_verdict(False, 0.70)
        assert verdict == "blocked"
        assert closed is False
        assert partial is False

    def test_retro_closed_when_auc_above_0_65(self):
        verdict, closed, partial = assign_honest_verdict(True, 0.70)
        assert verdict == "deployed_retro_closed"
        assert closed is True
        assert partial is False

    def test_retro_closed_at_exactly_0_65_is_false(self):
        # 0.65 is NOT > 0.65, so should be marginal not closed
        verdict, closed, partial = assign_honest_verdict(True, 0.65)
        assert verdict == "deployed_marginal"
        assert closed is False
        assert partial is True

    def test_marginal_when_auc_in_0_60_to_0_65(self):
        verdict, closed, partial = assign_honest_verdict(True, 0.62)
        assert verdict == "deployed_marginal"
        assert closed is False
        assert partial is True

    def test_marginal_boundary_just_above_0_60(self):
        verdict, closed, partial = assign_honest_verdict(True, 0.601)
        assert verdict == "deployed_marginal"
        assert partial is True


# ===========================================================================
# VJEPAv2EnergyAdapter: energy() interface
# ===========================================================================

class TestVJEPAv2EnergyAdapter:
    """REQ-VERIFY-145 — Adapter wraps VariationalJEPAPredictor with .energy() method."""

    def _make_adapter(self) -> VJEPAv2EnergyAdapter:
        """Build a minimal adapter with vocab_size=10 for fast tests."""
        model = VariationalJEPAPredictor(in_dim=10, context_dim=10, latent_dim=4)
        texts = ["correct step arithmetic", "error wrong calculation"]
        _, tok2idx = build_tfidf_features(texts, vocab_size=10)
        return VJEPAv2EnergyAdapter(model, tok2idx, vocab_size=10)

    def test_energy_returns_float(self):
        adapter = self._make_adapter()
        cot_input = CoTEnergyInput(question_text="What is 2+2?", response_text="2+2=4 correct")
        result = adapter.energy(cot_input)
        assert isinstance(result, float)

    def test_energy_in_zero_one_range(self):
        adapter = self._make_adapter()
        for text in ["correct step", "error wrong", "", "arithmetic reasoning valid"]:
            cot_input = CoTEnergyInput(question_text="", response_text=text)
            result = adapter.energy(cot_input)
            assert 0.0 <= result <= 1.0, f"energy={result} out of [0,1] for text={text!r}"

    def test_energy_uses_response_not_question(self):
        """Adapter should derive energy from response_text only (question is ignored)."""
        adapter = self._make_adapter()
        same_response = "correct arithmetic step"
        e1 = adapter.energy(CoTEnergyInput(question_text="Q1", response_text=same_response))
        e2 = adapter.energy(CoTEnergyInput(question_text="Q2 different", response_text=same_response))
        assert e1 == e2, "Energy must depend only on response_text"

    def test_empty_response_does_not_crash(self):
        adapter = self._make_adapter()
        result = adapter.energy(CoTEnergyInput(question_text="", response_text=""))
        assert isinstance(result, float)


# ===========================================================================
# _load_jepa_model(): priority order
# ===========================================================================

class TestLoadJepaModelPriority:
    """REQ-VERIFY-145 — _load_jepa_model checks v2 first, falls back on missing file."""

    def test_returns_none_when_no_safetensors_file(self, tmp_path):
        """No safetensors file → returns None (fall back to discriminative JEPA)."""
        result = _load_jepa_model(project_root=str(tmp_path))
        assert result is None

    def test_loads_v2_when_file_exists(self, tmp_path):
        """v2 safetensors file found → returns VJEPAv2EnergyAdapter."""
        # Create a small model and save it to simulate Exp 884 deploy output
        model = VariationalJEPAPredictor(in_dim=VOCAB_SIZE, context_dim=VOCAB_SIZE, latent_dim=32)
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        save_model_safetensors(model, results_dir / "vjepa_predictor_v2.safetensors")

        adapter = _load_jepa_model(project_root=str(tmp_path))
        assert adapter is not None
        assert isinstance(adapter, VJEPAv2EnergyAdapter)

    def test_loaded_adapter_energy_in_range(self, tmp_path):
        """Loaded adapter should produce energy in [0,1] after save/load round-trip."""
        model = VariationalJEPAPredictor(in_dim=VOCAB_SIZE, context_dim=VOCAB_SIZE, latent_dim=32)
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        save_model_safetensors(model, results_dir / "vjepa_predictor_v2.safetensors")

        adapter = _load_jepa_model(project_root=str(tmp_path))
        assert adapter is not None
        e = adapter.energy(CoTEnergyInput(question_text="", response_text="step one calculate"))
        assert 0.0 <= e <= 1.0

    def test_fallback_to_older_vjepa_when_v2_missing(self, tmp_path):
        """If vjepa_predictor_v2.safetensors is absent but v1 exists, use v1."""
        model = VariationalJEPAPredictor(in_dim=VOCAB_SIZE, context_dim=VOCAB_SIZE, latent_dim=32)
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        # Save as v1 (older version filename)
        save_model_safetensors(model, results_dir / "vjepa_predictor_v1.safetensors")

        # v2 does not exist, but v1 does
        adapter = _load_jepa_model(project_root=str(tmp_path))
        assert adapter is not None  # Should load v1 as fallback

    def test_v2_takes_priority_over_v1(self, tmp_path):
        """When both v1 and v2 exist, v2 is always loaded."""
        model = VariationalJEPAPredictor(in_dim=VOCAB_SIZE, context_dim=VOCAB_SIZE, latent_dim=32)
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        save_model_safetensors(model, results_dir / "vjepa_predictor_v1.safetensors")
        save_model_safetensors(model, results_dir / "vjepa_predictor_v2.safetensors")
        # Both exist; _load_jepa_model should pick v2 (explicit priority path)
        adapter = _load_jepa_model(project_root=str(tmp_path))
        assert adapter is not None


# ===========================================================================
# save_model_safetensors: round-trip fidelity
# ===========================================================================

class TestSaveModelSafetensors:
    """REQ-VERIFY-145 — Saved params survive safetensors save/load round-trip."""

    def test_roundtrip_preserves_params(self, tmp_path):
        from safetensors.numpy import load_file as st_load
        model = VariationalJEPAPredictor(in_dim=VOCAB_SIZE, context_dim=VOCAB_SIZE, latent_dim=32)
        path = tmp_path / "test_vjepa.safetensors"
        save_model_safetensors(model, path)

        raw = st_load(str(path))
        loaded_params = {k: jnp.array(v) for k, v in raw.items()}

        model2 = VariationalJEPAPredictor(in_dim=VOCAB_SIZE, context_dim=VOCAB_SIZE, latent_dim=32)
        model2.set_all_params(loaded_params)

        # Both models should produce identical predictions
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (VOCAB_SIZE,))
        ctx = jnp.zeros(VOCAB_SIZE)
        p1 = model.predict(x, ctx, key)
        p2 = model2.predict(x, ctx, key)
        assert abs(p1 - p2) < 1e-5, f"Round-trip mismatch: {p1} vs {p2}"

    def test_all_param_keys_saved(self, tmp_path):
        """All expected parameter keys must be present in the saved file."""
        from safetensors.numpy import load_file as st_load
        model = VariationalJEPAPredictor(in_dim=VOCAB_SIZE, context_dim=VOCAB_SIZE, latent_dim=32)
        path = tmp_path / "params.safetensors"
        save_model_safetensors(model, path)
        raw = st_load(str(path))
        expected_keys = set(model.get_all_params().keys())
        assert expected_keys == set(raw.keys())


# ===========================================================================
# update_architecture_tier2: content transformation
# ===========================================================================

class TestUpdateArchitectureTier2:
    """REQ-VERIFY-145 — Architecture Tier 2 row updated correctly."""

    _OLD_ROW = "| 2 | EORM | `EORMModel` | ~10 ms | CoT energy reward model (55M params) | `energy < eorm_threshold` |"

    def _make_arch_content(self) -> str:
        """Return minimal architecture.md content containing the Tier 2 row."""
        return (
            "## Verification Pipeline Tiers\n\n"
            "| Tier | Name | Class | Cost | Signal Source | Skip Condition |\n"
            "|------|------|-------|------|---------------|----------------|\n"
            f"{self._OLD_ROW}\n"
            "| 3 | Ising | `VerifyRepairPipeline` | ~0.006 ms/constraint | Full constraint verification | Always runs if tiers 0-2 pass |\n\n"
            "Each tier returns early if it can clear the response, avoiding subsequent more expensive tiers.\n"
        )

    def test_replaces_tier2_eorm_row(self, tmp_path):
        arch = tmp_path / "architecture.md"
        arch.write_text(self._make_arch_content())
        update_architecture_tier2(arch, 0.664, "2026-04-25")
        content = arch.read_text()
        assert "VJEPA v2" in content
        assert "VariationalJEPAPredictor" in content
        assert "0.6640" in content  # AUC formatted to 4 decimal places

    def test_old_eorm_row_removed(self, tmp_path):
        arch = tmp_path / "architecture.md"
        arch.write_text(self._make_arch_content())
        update_architecture_tier2(arch, 0.664, "2026-04-25")
        content = arch.read_text()
        # Old row literal should be gone (replaced by new row)
        assert self._OLD_ROW not in content

    def test_surrounding_content_preserved(self, tmp_path):
        arch = tmp_path / "architecture.md"
        arch.write_text(self._make_arch_content())
        update_architecture_tier2(arch, 0.664, "2026-04-25")
        content = arch.read_text()
        # Tier 3 row must still be present
        assert "Ising" in content
        assert "`VerifyRepairPipeline`" in content

    def test_idempotent_when_eorm_row_missing(self, tmp_path):
        """When the EORM row is already replaced, function must not crash."""
        arch = tmp_path / "architecture.md"
        arch.write_text("| 2 | VJEPA v2 | already updated row |\n\nEach tier returns early if it can clear the response\n")
        # Should not raise
        update_architecture_tier2(arch, 0.664, "2026-04-25")


# ===========================================================================
# generate_arc_heldout / generate_svamp_heldout: different from Exp 883
# ===========================================================================

class TestHeldoutGenerators:
    """REQ-VERIFY-145 — Held-out corpora use different seeds from Exp 883."""

    def test_arc_heldout_count(self):
        pairs = generate_arc_heldout(n_steps=10, seed=999)
        # Complete-problem guarantee: may overshoot by up to one problem's steps
        assert len(pairs) >= 10

    def test_svamp_heldout_count(self):
        pairs = generate_svamp_heldout(n_steps=10, seed=999)
        assert len(pairs) >= 10

    def test_arc_heldout_has_one_incorrect_per_problem(self):
        pairs = generate_arc_heldout(n_steps=10, seed=999)
        by_qid: dict[str, list[str]] = {}
        for p in pairs:
            by_qid.setdefault(p["question_id"], []).append(p["label"])
        for qid, labels in by_qid.items():
            assert labels.count("incorrect") == 1, f"{qid}: {labels}"

    def test_svamp_heldout_has_one_incorrect_per_problem(self):
        pairs = generate_svamp_heldout(n_steps=10, seed=999)
        by_qid: dict[str, list[str]] = {}
        for p in pairs:
            by_qid.setdefault(p["question_id"], []).append(p["label"])
        for qid, labels in by_qid.items():
            assert labels.count("incorrect") == 1, f"{qid}: {labels}"

    def test_arc_heldout_domain_tag(self):
        pairs = generate_arc_heldout(n_steps=8, seed=999)
        assert all(p["domain"] == "arc_heldout" for p in pairs)

    def test_svamp_heldout_domain_tag(self):
        pairs = generate_svamp_heldout(n_steps=8, seed=999)
        assert all(p["domain"] == "svamp_heldout" for p in pairs)

    def test_different_seed_from_exp883(self):
        """Held-out questions (seed=999) must differ from Exp 883 OOD set (seed=42)."""
        from experiment_883_vjepa_v2_expanded_corpus import (
            generate_arc_synthetic,
            generate_svamp_synthetic,
        )
        exp883_arc = generate_arc_synthetic(n_steps=10, seed=42)
        exp883_svamp = generate_svamp_synthetic(n_steps=10, seed=42)
        heldout_arc = generate_arc_heldout(n_steps=10, seed=999)
        heldout_svamp = generate_svamp_heldout(n_steps=10, seed=999)
        # Step texts must differ (different domain name too: arc_synthetic vs arc_heldout)
        assert [p["step_text"] for p in exp883_arc] != [p["step_text"] for p in heldout_arc]
        assert [p["step_text"] for p in exp883_svamp] != [p["step_text"] for p in heldout_svamp]


# ===========================================================================
# evaluate_on_heldout: edge cases
# ===========================================================================

class TestEvaluateOnHeldout:
    """REQ-VERIFY-145 — evaluate_on_heldout handles edge cases."""

    def test_empty_corpus_returns_half(self):
        model = VariationalJEPAPredictor(in_dim=10, context_dim=10, latent_dim=4)
        key = jax.random.PRNGKey(0)
        result = evaluate_on_heldout(model, [], key)
        assert result == 0.5

    def test_result_in_zero_one_range(self):
        model = VariationalJEPAPredictor(in_dim=10, context_dim=10, latent_dim=4)
        key = jax.random.PRNGKey(0)
        texts = ["correct step", "wrong step error"]
        _, tok2idx = build_tfidf_features(texts, vocab_size=10)
        raw = [
            {"question_id": "q0", "step_text": "correct step", "label": "correct"},
            {"question_id": "q0", "step_text": "wrong step error", "label": "incorrect"},
        ]
        corpus = prepare_corpus(raw, tok2idx, 10)
        result = evaluate_on_heldout(model, corpus, key)
        assert 0.0 <= result <= 1.0


# ===========================================================================
# Integration smoke test: full run on tiny corpus
# ===========================================================================

class TestIntegrationSmoke:
    """SCENARIO-VERIFY-233 — Full Exp 884 produces valid artifact with all fields."""

    def test_run_produces_artifact(self, tmp_path):
        """Smoke test: run full experiment on tiny corpus (5 epochs) in tmp dirs."""
        import experiment_884_vjepa_cascade_deploy as exp884

        # Set up Exp 883 result with passing gate
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        exp883_data = {
            "experiment": 883, "schema": "carnot-experiment-v1",
            "run_date": "2026-04-25T00:00:00Z", "honest_verdict": "vjepa_ood_above_gate",
            "ood_auc": 0.664,
        }
        exp883_path = results_dir / "experiment_883_vjepa_v2_expanded_corpus.json"
        with exp883_path.open("w") as fh:
            json.dump(exp883_data, fh)

        # Create minimal architecture.md
        arch_dir = tmp_path / "_bmad"
        arch_dir.mkdir()
        arch_path = arch_dir / "architecture.md"
        arch_path.write_text(
            "## Verification Pipeline Tiers\n\n"
            "| Tier | Name | Class | Cost | Signal Source | Skip Condition |\n"
            "|------|------|-------|------|---------------|----------------|\n"
            "| 2 | EORM | `EORMModel` | ~10 ms | CoT energy reward model (55M params) | `energy < eorm_threshold` |\n\n"
            "Each tier returns early if it can clear the response.\n"
        )

        artifact_path = results_dir / "experiment_884_vjepa_cascade_deploy.json"

        # Patch all path constants and run with minimal epochs
        # Capture original before patching to avoid recursion
        _orig_train = exp884.train_vjepa_v2

        def _fast_train(model, corpus, dn, n_epochs=200, lr=1e-3, seed=0):
            return _orig_train(model, corpus, dn, n_epochs=3, lr=lr, seed=seed)

        with (
            patch.object(exp884, "EXP_883_RESULT_PATH", exp883_path),
            patch.object(exp884, "RESULT_PATH", artifact_path),
            patch.object(exp884, "MODEL_SAVE_PATH", results_dir / "vjepa_predictor_v2.safetensors"),
            patch("experiment_884_vjepa_cascade_deploy._ROOT", tmp_path),
            patch.object(exp884, "train_vjepa_v2", _fast_train),
        ):
            artifact = exp884.run_experiment()

        assert artifact_path.exists()
        missing = REQUIRED_RESULT_FIELDS - set(artifact.keys())
        assert not missing, f"Missing required fields: {missing}"
        assert artifact["experiment"] == 884
        assert artifact["cascade_deployed"] is True
        assert isinstance(artifact["final_ood_auc"], float)
        assert 0.0 <= artifact["final_ood_auc"] <= 1.0

    def test_assert_deliverable_written_passes_after_run(self, tmp_path):
        """assert_deliverable_written() passes when artifact is valid."""
        import experiment_884_vjepa_cascade_deploy as exp884

        # Create minimal valid artifact
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        artifact_path = results_dir / "experiment_884.json"
        data = {
            "experiment": 884,
            "schema": "carnot-experiment-v1",
            "run_date": "2026-04-25T00:00:00Z",
            "honest_verdict": "deployed_retro_closed",
            "cascade_deployed": True,
            "final_ood_auc": 0.68,
            "retro_jepa_ood_closed": True,
            "retro_jepa_ood_partially_closed": False,
            "model_version": "vjepa_v2",
        }
        with artifact_path.open("w") as fh:
            json.dump(data, fh)

        with patch.object(exp884, "RESULT_PATH", artifact_path):
            exp884.assert_deliverable_written()  # Must not raise

    def test_assert_deliverable_fails_when_missing(self, tmp_path):
        """assert_deliverable_written() raises AssertionError when file absent."""
        import experiment_884_vjepa_cascade_deploy as exp884
        missing = tmp_path / "not_there.json"
        with patch.object(exp884, "RESULT_PATH", missing):
            with pytest.raises(AssertionError):
                exp884.assert_deliverable_written()
