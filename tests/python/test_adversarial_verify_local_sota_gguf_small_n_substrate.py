"""Tests for adversarial_verify.py's live_llm_inference_local_gguf_sota substrate category.

Origin: 2026-07-05. Two .481 tasks both declared this substrate and were flagged
DURATION_TOO_SHORT under the generic 60s live_llm_inference floor, though both are honest,
non-fabricated null/negative results:

- exp5262 (results/experiment_5262_solver_grounded_constraint_extraction_v481.json): 4 short
  structured-JSON constraint-generation calls over a bounded fixture set (real llama.cpp
  `raw_output` text confirmed present), 56.3s total -- correctly concluded
  constraint_validity_rate=0.25 vs. a 0.5 baseline, no useful oracle-distinct signal.
- exp5263 (results/experiment_5263_neuron_attention_energy_hallucination_probe_v481.json): 6
  single-forward-pass logit/logprob extraction probes (no generation) over a 6-fixture panel,
  23.7s total -- correctly concluded a null logit-energy signal_delta=0.0048.

Both are genuinely lighter than the full multi-hundred-token generation runs the 60s floor is
calibrated for (a handful of short generation calls, or a handful of single-forward-pass
probes), but still real model-load-and-invoke work -- not fabricated. Mirrors the same pattern
as this project's four prior substrate-category additions this week: add a genuine new category
with its own calibrated floor rather than force-fitting into an existing one.

Spec refs: none (operational lint fix, no OpenSpec capability).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import scripts.adversarial_verify as av


def _write_payload(tmp_path: Path, payload: dict[str, Any]) -> Path:
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _report_for_payload(tmp_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return av.verify_artifact(_write_payload(tmp_path, payload))


def _flag_kinds(report: dict[str, Any]) -> set[str]:
    return {flag["kind"] for flag in report["flags"]}


class TestIsLocalSotaGgufSmallN:
    def test_recognizes_canonical_substrate_value(self) -> None:
        d = {"inference_substrate": "live_llm_inference_local_gguf_sota"}
        assert av._is_local_sota_gguf_small_n(d) is True

    def test_does_not_match_generic_live_llm_inference(self) -> None:
        d = {"inference_substrate": "live_llm_inference"}
        assert av._is_local_sota_gguf_small_n(d) is False

    def test_does_not_match_missing_field(self) -> None:
        assert av._is_local_sota_gguf_small_n({}) is False


class TestDurationFloorForArtifact:
    def test_returns_the_10s_floor(self) -> None:
        d = {"inference_substrate": "live_llm_inference_local_gguf_sota"}
        floor = av.duration_floor_for_artifact(d)
        assert floor == {
            "substrate": "live_llm_inference_local_gguf_sota",
            "min_duration_s": 10.0,
            "reason": "local_sota_gguf_small_n",
        }


class TestExp5262And5263IncidentReproduction:
    def test_exp5262_56s_constraint_extraction_no_longer_flags(self, tmp_path: Path) -> None:
        payload = {
            "experiment": "exp5262_repro",
            "honest_verdict": "complete: solver-grounded extraction produced no useful signal",
            "inference_substrate": "live_llm_inference_local_gguf_sota",
            "duration_s": 56.308939,
            "model_specs": {"flagship_dense": {"hf_id": "unsloth/gemma-4-31B-it-GGUF"}},
        }
        report = _report_for_payload(tmp_path, payload)
        assert "DURATION_TOO_SHORT" not in _flag_kinds(report)

    def test_exp5263_24s_logit_probe_no_longer_flags(self, tmp_path: Path) -> None:
        payload = {
            "experiment": "exp5263_repro",
            "honest_verdict": "complete: null logit-energy delta=0.0048",
            "inference_substrate": "live_llm_inference_local_gguf_sota",
            "duration_s": 23.660199,
            "model_specs": {"flagship_dense": {"hf_id": "unsloth/gemma-4-31B-it-GGUF"}},
        }
        report = _report_for_payload(tmp_path, payload)
        assert "DURATION_TOO_SHORT" not in _flag_kinds(report)

    def test_genuinely_too_fast_claim_is_still_caught(self, tmp_path: Path) -> None:
        payload = {
            "experiment": "exp_fabricated_small_n_gguf",
            "honest_verdict": "complete: fake fast result",
            "inference_substrate": "live_llm_inference_local_gguf_sota",
            "duration_s": 0.5,
            "model_specs": {"flagship_dense": {"hf_id": "unsloth/gemma-4-31B-it-GGUF"}},
        }
        report = _report_for_payload(tmp_path, payload)
        assert "DURATION_TOO_SHORT" in _flag_kinds(report)

    def test_full_generation_claim_still_needs_the_60s_floor(self, tmp_path: Path) -> None:
        """Regression guard: declaring the new substrate must not become a loophole for
        artifacts that actually claim full generative inference."""
        payload = {
            "experiment": "exp_full_generation",
            "honest_verdict": "complete: full generation run",
            "inference_substrate": "live_llm_inference",
            "duration_s": 15.0,
            "model_specs": {"gguf_path": "/some/path/model.gguf"},
        }
        report = _report_for_payload(tmp_path, payload)
        assert "DURATION_TOO_SHORT" in _flag_kinds(report)

    def test_methodology_still_requires_model_specs(self, tmp_path: Path) -> None:
        """This substrate is genuinely LLM inference (unlike the no-LLM substrates) --
        model_specs/target_model must still be required, not exempted. Includes a
        compute-bound marker string (a GGUF mention outside model_specs) so the
        methodology check actually fires -- without any marker present, it early-returns
        regardless of substrate, per check_methodology_present's own gate."""
        payload = {
            "experiment": "exp_missing_model_specs",
            "honest_verdict": "complete: pilot ran on a local GGUF model",
            "inference_substrate": "live_llm_inference_local_gguf_sota",
            "duration_s": 30.0,
            "random_seed": 1,
            "reproducibility_checksum": "abc123",
        }
        report = _report_for_payload(tmp_path, payload)
        assert "METHODOLOGY_MISSING" in _flag_kinds(report)

    def test_uppercase_model_specs_key_is_recognized(self, tmp_path: Path) -> None:
        """exp5262/exp5263 (and 22 other corpus artifacts as of 2026-07-05) declare the
        field as uppercase MODEL_SPECS (a Python-constant-style name) rather than
        lowercase model_specs -- both must be recognized as satisfying the requirement."""
        payload = {
            "experiment": "exp_uppercase_model_specs",
            "honest_verdict": "complete: pilot ran on a local GGUF model",
            "inference_substrate": "live_llm_inference_local_gguf_sota",
            "duration_s": 30.0,
            "random_seed": 1,
            "reproducibility_checksum": "abc123",
            "MODEL_SPECS": {"flagship_dense": {"hf_id": "unsloth/gemma-4-31B-it-GGUF"}},
        }
        report = _report_for_payload(tmp_path, payload)
        flags_by_kind = {f["kind"]: f for f in report["flags"]}
        assert "METHODOLOGY_MISSING" not in flags_by_kind or "model_specs" not in flags_by_kind[
            "METHODOLOGY_MISSING"
        ].get("detail", "")
