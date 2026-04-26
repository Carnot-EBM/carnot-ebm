"""Tests for Experiment 882 — Live Cascade v7: Gemma4-E4B-it + Full Cascade.

Traces to:
    REQ-BENCH-015  (live cascade benchmark)
    SCENARIO-BENCH-034 (cascade with single inference model)

Why these tests exist:
    The experiment has five critical paths that must be covered by CPU-only,
    mock-backed tests so CI never requires real GPU hardware:

    1. Gate check — missing preflight artifact or live_env_fixed!=True yields blocked.
    2. CARNOT_FORCE_LIVE missing — yields blocked artifact.
    3. Model load failure — writes blocked artifact.
    4. _run_cascade() — tests all branches (simulation, live_gpu, cascade clear,
       Tier 3 repair, StreamingCoT advisory).
    5. _compute_metrics() — accuracy, signed_improvement, cascade_skip_rate,
       cascade_tiers_active, honest_verdict for each verdict branch.
    6. Answer extraction helpers — _extract_final_answer and _answers_match.
    7. Happy-path main() — all REQUIRED_RESULT_FIELDS present in written artifact.

All GPU calls, ThreeTierPipeline, and VerifyRepairPipeline are mocked so the
entire suite runs in < 5 s on any CPU-only CI machine.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path wiring
# ---------------------------------------------------------------------------
_REPO = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "python"))
sys.path.insert(0, str(_REPO / "scripts"))

import scripts.experiment_882_live_cascade_v7_gemma4 as exp882  # noqa: E402
from scripts.experiment_template import REQUIRED_RESULT_FIELDS  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_preflight(results_dir: Path, *, live_env_fixed: bool) -> None:
    """Write a minimal Exp 855 preflight artifact."""
    p = results_dir / "experiment_855_preflight_v15.json"
    p.write_text(json.dumps({"live_env_fixed": live_env_fixed, "status": "success"}))


def _mock_tmpl_factory(tmp_path: Path) -> MagicMock:
    """Return a MagicMock ExperimentTemplate that writes the deliverable when build_result called.

    Why set assert_deliverable_written explicitly: MagicMock intercepts calls to
    any attribute starting with 'assert_' and treats them as built-in mock
    assertion helpers, raising AttributeError if the call signature doesn't match.
    Setting the attribute explicitly as a plain MagicMock bypasses that interception.
    """
    deliverable_path = tmp_path / exp882.DELIVERABLE

    def _build_result(payload: dict, **kwargs: Any) -> dict:
        base = {
            "status": kwargs.get("status", "success"),
            "experiment": exp882.EXP_ID,
            "title": exp882.TITLE,
            "schema": "carnot.experiment.v1",
            "run_date": "20260425",
            "started_at": "2026-04-25T00:00:00Z",
            "finished_at": "2026-04-25T00:00:01Z",
            "duration_s": 1.0,
        }
        base.update(payload)
        return base

    mock = MagicMock()
    mock.build_result.side_effect = _build_result
    mock.setup.return_value = None
    mock.checkpoint_save.return_value = None
    # Explicitly set to bypass MagicMock's assert_* interception.
    mock.assert_deliverable_written = MagicMock(return_value=None)
    return mock


# ---------------------------------------------------------------------------
# Gate check — CARNOT_FORCE_LIVE missing
# ---------------------------------------------------------------------------


class TestGateForceLiveMissing:
    """REQ-BENCH-015: gate must block when CARNOT_FORCE_LIVE is absent."""

    def test_blocked_when_force_live_not_set(self, tmp_path: Path) -> None:
        """Missing CARNOT_FORCE_LIVE must produce a blocked artifact."""
        (tmp_path / "results").mkdir(parents=True)
        _write_preflight(tmp_path / "results", live_env_fixed=True)
        deliverable = tmp_path / exp882.DELIVERABLE

        mock_tmpl = _mock_tmpl_factory(tmp_path)

        env_without = {k: v for k, v in os.environ.items() if k != "CARNOT_FORCE_LIVE"}
        with (
            patch.object(exp882, "_REPO_ROOT", tmp_path),
            patch(
                "scripts.experiment_882_live_cascade_v7_gemma4.ExperimentTemplate",
                return_value=mock_tmpl,
            ),
            patch.dict("os.environ", env_without, clear=True),
        ):
            exp882.main()

        assert deliverable.exists()
        data = json.loads(deliverable.read_text())
        assert data["status"] == "blocked"
        assert data["honest_verdict"] == "blocked"


# ---------------------------------------------------------------------------
# Gate check — preflight artifact missing
# ---------------------------------------------------------------------------


class TestGatePrefLightMissing:
    """REQ-BENCH-015: gate blocks when Exp 855 artifact is absent."""

    def test_blocked_when_preflight_missing(self, tmp_path: Path) -> None:
        """No preflight artifact must yield blocked."""
        (tmp_path / "results").mkdir(parents=True)
        deliverable = tmp_path / exp882.DELIVERABLE

        mock_tmpl = _mock_tmpl_factory(tmp_path)

        with (
            patch.object(exp882, "_REPO_ROOT", tmp_path),
            patch(
                "scripts.experiment_882_live_cascade_v7_gemma4.ExperimentTemplate",
                return_value=mock_tmpl,
            ),
            patch.dict("os.environ", {"CARNOT_FORCE_LIVE": "1"}),
        ):
            exp882.main()

        assert deliverable.exists()
        data = json.loads(deliverable.read_text())
        assert data["status"] == "blocked"


# ---------------------------------------------------------------------------
# Gate check — live_env_fixed False
# ---------------------------------------------------------------------------


class TestGateLiveEnvFalse:
    """REQ-BENCH-015: gate blocks when live_env_fixed != True."""

    def test_blocked_when_live_env_not_fixed(self, tmp_path: Path) -> None:
        """live_env_fixed=False in preflight must yield blocked."""
        (tmp_path / "results").mkdir(parents=True)
        _write_preflight(tmp_path / "results", live_env_fixed=False)
        deliverable = tmp_path / exp882.DELIVERABLE

        mock_tmpl = _mock_tmpl_factory(tmp_path)

        with (
            patch.object(exp882, "_REPO_ROOT", tmp_path),
            patch(
                "scripts.experiment_882_live_cascade_v7_gemma4.ExperimentTemplate",
                return_value=mock_tmpl,
            ),
            patch.dict("os.environ", {"CARNOT_FORCE_LIVE": "1"}),
        ):
            exp882.main()

        assert deliverable.exists()
        data = json.loads(deliverable.read_text())
        assert data["status"] == "blocked"


# ---------------------------------------------------------------------------
# Model load failure
# ---------------------------------------------------------------------------


class TestModelLoadFailure:
    """REQ-BENCH-015: model load failure must produce blocked artifact."""

    def test_blocked_on_model_load_exception(self, tmp_path: Path) -> None:
        """If AutoModelForCausalLM raises, artifact must be blocked."""
        (tmp_path / "results").mkdir(parents=True)
        _write_preflight(tmp_path / "results", live_env_fixed=True)
        deliverable = tmp_path / exp882.DELIVERABLE

        mock_tmpl = _mock_tmpl_factory(tmp_path)

        import torch

        def _raise(*a, **kw):  # noqa: ANN001
            raise RuntimeError("Out of memory")

        with (
            patch.object(exp882, "_REPO_ROOT", tmp_path),
            patch(
                "scripts.experiment_882_live_cascade_v7_gemma4.ExperimentTemplate",
                return_value=mock_tmpl,
            ),
            patch.dict("os.environ", {"CARNOT_FORCE_LIVE": "1"}),
            patch("transformers.AutoTokenizer.from_pretrained", side_effect=_raise),
        ):
            exp882.main()

        assert deliverable.exists()
        data = json.loads(deliverable.read_text())
        assert data["status"] == "blocked"
        assert data["honest_verdict"] == "blocked"
        assert "model_load_error" in data


# ---------------------------------------------------------------------------
# _extract_final_answer
# ---------------------------------------------------------------------------


class TestExtractFinalAnswer:
    """Unit tests for _extract_final_answer()."""

    def test_plain_integer(self) -> None:
        """'8' extracts to '8'."""
        assert exp882._extract_final_answer("8") == "8"

    def test_dollar_amount(self) -> None:
        """'$12' is extracted from dollar-prefixed text."""
        assert exp882._extract_final_answer("$12") == "$12"

    def test_dollar_with_decimal(self) -> None:
        """'$164.50' is extracted correctly."""
        assert exp882._extract_final_answer("The answer is $164.50 total.") == "$164.50"

    def test_fraction(self) -> None:
        """'1/3' is extracted from fraction-containing text."""
        assert exp882._extract_final_answer("1/3 of the tank") == "1/3"

    def test_decimal_number(self) -> None:
        """'0.5' is extracted from decimal text."""
        assert exp882._extract_final_answer("Probability: 0.5") == "0.5"

    def test_leading_minus(self) -> None:
        """Negative number '-5' is extracted."""
        assert exp882._extract_final_answer("-5 degrees") == "-5"

    def test_fallback_first_token(self) -> None:
        """Non-numeric text returns first whitespace-separated token."""
        result = exp882._extract_final_answer("hello world")
        assert result == "hello"

    def test_empty_string_returns_empty(self) -> None:
        """Empty string returns empty string (fallback)."""
        result = exp882._extract_final_answer("")
        assert result == ""


# ---------------------------------------------------------------------------
# _answers_match
# ---------------------------------------------------------------------------


class TestAnswersMatch:
    """Unit tests for _answers_match()."""

    def test_identical_strings(self) -> None:
        """Identical strings are equal."""
        assert exp882._answers_match("8", "8") is True

    def test_dollar_spacing_normalised(self) -> None:
        """'$ 12' and '$12' are considered equal."""
        assert exp882._answers_match("$ 12", "$12") is True

    def test_case_insensitive(self) -> None:
        """Comparison is case-insensitive."""
        assert exp882._answers_match("Yes", "yes") is True

    def test_different_answers(self) -> None:
        """Different numeric values are not equal."""
        assert exp882._answers_match("8", "9") is False

    def test_fraction_equality(self) -> None:
        """'1/3' matches '1/3'."""
        assert exp882._answers_match("1/3", "1/3") is True


# ---------------------------------------------------------------------------
# _run_cascade — simulation path
# ---------------------------------------------------------------------------


class TestRunCascadeSimulation:
    """_run_cascade() with inference_mode=simulation_fallback."""

    def _make_problem(self, idx: int = 5) -> dict[str, Any]:
        return {"id": f"gsm8k_{idx}", "question": "What is 2+2?", "answer": "40"}

    def test_returns_required_keys(self) -> None:
        """Result must contain all documented keys. REQ-BENCH-015."""
        p = self._make_problem()
        result = exp882._run_cascade(p, None, None, None, None, "simulation_fallback", None)
        for key in (
            "id",
            "tier_exited_at",
            "was_correct_baseline",
            "was_correct_carnot",
            "repaired",
            "latency_ms",
            "streaming_cot_unstable",
        ):
            assert key in result, f"Missing key: {key}"

    def test_no_crash_on_none_pipelines(self) -> None:
        """Must not raise when model and pipelines are all None."""
        p = self._make_problem()
        exp882._run_cascade(p, None, None, None, None, "simulation_fallback", None)

    def test_tier_exited_at_none_in_simulation(self) -> None:
        """Simulation path must leave tier_exited_at as None (no real tiers)."""
        p = self._make_problem()
        result = exp882._run_cascade(p, None, None, None, None, "simulation_fallback", None)
        assert result["tier_exited_at"] is None

    def test_streaming_cot_none_when_no_detector(self) -> None:
        """streaming_cot_unstable is None when no detector wired."""
        p = self._make_problem()
        result = exp882._run_cascade(p, None, None, None, None, "simulation_fallback", None)
        assert result["streaming_cot_unstable"] is None


# ---------------------------------------------------------------------------
# _run_cascade — live GPU path with mocked ThreeTierPipeline
# ---------------------------------------------------------------------------


class TestRunCascadeLiveGPU:
    """SCENARIO-BENCH-034: live_gpu path with mocked pipelines."""

    def _make_problem(self, idx: int = 5) -> dict[str, Any]:
        return {"id": f"gsm8k_{idx}", "question": "What is 5+3?", "answer": "8"}

    def _make_mock_model(self, answer: str = "8") -> tuple[MagicMock, MagicMock]:
        """Return (mock_model, mock_tokenizer) that produce a fixed answer.

        The tokenizer mock returns a MagicMock so that .to(device) is callable
        (a real dict has no .to() method but the code calls inputs.to(model.device)).
        input_ids shape attribute is set so shape[1] indexing works.
        """
        import torch

        mock_tok = MagicMock()
        mock_tok.eos_token_id = 1
        # inputs = tokenizer(prompt, ...) — must support .to(device)
        mock_inputs = MagicMock()
        mock_inputs.__getitem__ = lambda self, k: torch.zeros(1, 5, dtype=torch.long)
        input_ids = torch.zeros(1, 5, dtype=torch.long)
        mock_inputs.input_ids = input_ids
        # inputs["input_ids"].shape[1] == 5 — used to slice new tokens
        mock_inputs.__getitem__ = MagicMock(return_value=input_ids)
        mock_inputs.to.return_value = mock_inputs
        mock_tok.return_value = mock_inputs
        mock_tok.decode.return_value = answer

        mock_model = MagicMock()
        mock_model.device = "cpu"
        # model.generate → output_ids; new tokens = output_ids[0][5:]
        mock_model.generate.return_value = [torch.zeros(6, dtype=torch.long)]

        return mock_model, mock_tok

    def test_cascade_clears_eorm_tier(self) -> None:
        """When ThreeTierPipeline.verify() returns tier_used='eorm', tier_exited_at=2."""
        p = self._make_problem()
        mock_model, mock_tok = self._make_mock_model("8")

        mock_three_tier = MagicMock()
        mock_three_tier.verify.return_value = (True, "eorm", 0.3)

        result = exp882._run_cascade(
            p, mock_model, mock_tok, mock_three_tier, None, "live_gpu", None
        )
        assert result["tier_exited_at"] == 2

    def test_cascade_clears_sink_probe(self) -> None:
        """tier_used='sink_probe' maps to tier_exited_at=1."""
        p = self._make_problem()
        mock_model, mock_tok = self._make_mock_model("8")

        mock_three_tier = MagicMock()
        mock_three_tier.verify.return_value = (True, "sink_probe", 0.8)

        result = exp882._run_cascade(
            p, mock_model, mock_tok, mock_three_tier, None, "live_gpu", None
        )
        assert result["tier_exited_at"] == 1

    def test_cascade_clears_nup_probe(self) -> None:
        """tier_used='nup_probe_v4' maps to tier_exited_at=0."""
        p = self._make_problem()
        mock_model, mock_tok = self._make_mock_model("8")

        mock_three_tier = MagicMock()
        mock_three_tier.verify.return_value = (True, "nup_probe_v4", -0.5)

        result = exp882._run_cascade(
            p, mock_model, mock_tok, mock_three_tier, None, "live_gpu", None
        )
        assert result["tier_exited_at"] == 0

    def test_tier3_repair_invoked_when_cascade_misses(self) -> None:
        """When cascade does not clear (tier_used='ising'), VerifyRepairPipeline is called."""
        p = self._make_problem()
        mock_model, mock_tok = self._make_mock_model("WRONG")

        mock_three_tier = MagicMock()
        mock_three_tier.verify.return_value = (False, "ising", 0.9)

        mock_repair_result = MagicMock()
        mock_repair_result.repaired_response = "8"
        mock_vrp = MagicMock()
        mock_vrp.verify_and_repair.return_value = mock_repair_result

        result = exp882._run_cascade(
            p, mock_model, mock_tok, mock_three_tier, mock_vrp, "live_gpu", None
        )
        assert result["repaired"] is True
        assert result["was_correct_carnot"] is True

    def test_no_repair_when_vrp_not_available(self) -> None:
        """When verify_repair is None, cascade miss means no repair."""
        p = self._make_problem()
        mock_model, mock_tok = self._make_mock_model("WRONG")

        mock_three_tier = MagicMock()
        mock_three_tier.verify.return_value = (False, "ising", 0.9)

        result = exp882._run_cascade(
            p, mock_model, mock_tok, mock_three_tier, None, "live_gpu", None
        )
        assert result["repaired"] is False
        assert result["tier_exited_at"] is None

    def test_three_tier_exception_handled_gracefully(self) -> None:
        """If ThreeTierPipeline.verify() raises, the cascade should not crash."""
        p = self._make_problem()
        mock_model, mock_tok = self._make_mock_model("8")

        mock_three_tier = MagicMock()
        mock_three_tier.verify.side_effect = RuntimeError("EORM model not loaded")

        result = exp882._run_cascade(
            p, mock_model, mock_tok, mock_three_tier, None, "live_gpu", None
        )
        assert result["tier_exited_at"] is None
        assert "id" in result

    def test_streaming_cot_advisory_recorded(self) -> None:
        """StreamingCoT result is stored in streaming_cot_unstable key."""
        p = {"id": "gsm8k_5", "question": "What is 5+3?", "answer": "8"}
        mock_model, mock_tok = self._make_mock_model("8")

        mock_three_tier = MagicMock()
        mock_three_tier.verify.return_value = (True, "eorm", 0.3)

        mock_detect_result = MagicMock()
        mock_detect_result.is_streaming_unstable = True
        mock_detector = MagicMock()
        mock_detector.detect.return_value = mock_detect_result

        result = exp882._run_cascade(
            p, mock_model, mock_tok, mock_three_tier, None, "live_gpu", mock_detector
        )
        assert result["streaming_cot_unstable"] is True


# ---------------------------------------------------------------------------
# _compute_metrics
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    """Unit tests for _compute_metrics() covering all verdict branches."""

    def _make_results(
        self,
        n: int,
        *,
        base_correct: int,
        carnot_correct: int,
        skipped: int,
        repaired: int = 0,
    ) -> list[dict[str, Any]]:
        """Build a synthetic per_question list for metric testing."""
        results = []
        for i in range(n):
            tee = 1 if i < skipped else None
            rep = i >= skipped and i < (skipped + repaired)
            results.append(
                {
                    "id": f"gsm8k_{i}",
                    "tier_exited_at": tee,
                    "was_correct_baseline": i < base_correct,
                    "was_correct_carnot": i < carnot_correct,
                    "repaired": rep,
                    "latency_ms": 10.0,
                    "streaming_cot_unstable": None,
                }
            )
        return results

    def test_simulation_fallback_verdict(self) -> None:
        """simulation_fallback when inference_mode != live_gpu. REQ-BENCH-015."""
        data = self._make_results(10, base_correct=7, carnot_correct=8, skipped=5)
        m = exp882._compute_metrics(data, "simulation_fallback")
        assert m["honest_verdict"] == "simulation_fallback"

    def test_positive_improvement_verdict(self) -> None:
        """positive_improvement when signed_improvement > 0 and live_gpu."""
        data = self._make_results(10, base_correct=7, carnot_correct=8, skipped=0, repaired=1)
        m = exp882._compute_metrics(data, "live_gpu")
        assert m["signed_improvement"] > 0
        # cascade_tiers_active < 3 (only Tier 3 fires), so positive_improvement
        assert m["honest_verdict"] == "positive_improvement"

    def test_live_no_improvement_verdict(self) -> None:
        """live_no_improvement when live_gpu but signed_improvement <= 0."""
        data = self._make_results(10, base_correct=7, carnot_correct=7, skipped=0)
        m = exp882._compute_metrics(data, "live_gpu")
        assert m["signed_improvement"] == 0.0
        assert m["honest_verdict"] == "live_no_improvement"

    def test_cascade_running_verdict(self) -> None:
        """cascade_running when >= 3 tiers active regardless of improvement."""
        # Make results where tiers 0, 1, 2 all fire
        results = []
        tier_map = [0, 1, 2, None, None]
        for i in range(10):
            tee = tier_map[i % 5]
            results.append(
                {
                    "id": f"gsm8k_{i}",
                    "tier_exited_at": tee,
                    "was_correct_baseline": True,
                    "was_correct_carnot": True,
                    "repaired": tee is None and i % 5 == 4,
                    "latency_ms": 5.0,
                    "streaming_cot_unstable": None,
                }
            )
        m = exp882._compute_metrics(results, "live_gpu")
        assert m["cascade_tiers_active"] >= 3
        assert m["honest_verdict"] == "cascade_running"

    def test_empty_input_returns_blocked(self) -> None:
        """Empty per_question list returns honest_verdict='blocked'."""
        m = exp882._compute_metrics([], "live_gpu")
        assert m["honest_verdict"] == "blocked"

    def test_accuracy_values_correct(self) -> None:
        """baseline_accuracy and carnot_accuracy are correctly computed fractions."""
        data = self._make_results(10, base_correct=6, carnot_correct=8, skipped=0)
        m = exp882._compute_metrics(data, "live_gpu")
        assert m["baseline_accuracy"] == pytest.approx(0.6, abs=0.001)
        assert m["carnot_accuracy"] == pytest.approx(0.8, abs=0.001)

    def test_cascade_skip_rate(self) -> None:
        """cascade_skip_rate = skipped / n."""
        data = self._make_results(10, base_correct=7, carnot_correct=7, skipped=4)
        m = exp882._compute_metrics(data, "live_gpu")
        assert m["cascade_skip_rate"] == pytest.approx(0.4, abs=0.001)


# ---------------------------------------------------------------------------
# Problem corpus integrity
# ---------------------------------------------------------------------------


class TestProblemCorpus:
    """Validate the GSM8K problem corpus shape. REQ-BENCH-015."""

    def test_exactly_50_problems(self) -> None:
        """Corpus must have exactly 50 problems."""
        assert len(exp882._GSM8K_PROBLEMS) == 50

    def test_all_problems_have_required_keys(self) -> None:
        """Every problem must have 'id', 'question', and 'answer'."""
        for p in exp882._GSM8K_PROBLEMS:
            assert "id" in p
            assert "question" in p
            assert "answer" in p

    def test_ids_are_unique(self) -> None:
        """All problem ids must be distinct."""
        ids = [p["id"] for p in exp882._GSM8K_PROBLEMS]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Happy-path main() — REQUIRED_RESULT_FIELDS in artifact
# ---------------------------------------------------------------------------


class TestMainHappyPath:
    """SCENARIO-BENCH-034: happy-path main() writes a valid artifact."""

    def test_required_fields_in_artifact(self, tmp_path: Path) -> None:
        """All REQUIRED_RESULT_FIELDS must be present in the output artifact."""
        (tmp_path / "results").mkdir(parents=True)
        _write_preflight(tmp_path / "results", live_env_fixed=True)
        deliverable = tmp_path / exp882.DELIVERABLE

        mock_tmpl = _mock_tmpl_factory(tmp_path)

        import torch

        mock_tok = MagicMock()
        mock_tok.eos_token_id = 1
        mock_tok.decode.return_value = "8"
        mock_inputs = MagicMock()
        input_ids = torch.zeros(1, 3, dtype=torch.long)
        mock_inputs.__getitem__ = MagicMock(return_value=input_ids)
        mock_inputs.to.return_value = mock_inputs
        mock_tok.return_value = mock_inputs

        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.generate.return_value = [torch.zeros(4, dtype=torch.long)]

        mock_three_tier = MagicMock()
        mock_three_tier.verify.return_value = (True, "eorm", 0.2)

        with (
            patch.object(exp882, "_REPO_ROOT", tmp_path),
            patch(
                "scripts.experiment_882_live_cascade_v7_gemma4.ExperimentTemplate",
                return_value=mock_tmpl,
            ),
            patch.dict("os.environ", {"CARNOT_FORCE_LIVE": "1"}, clear=False),
            patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tok),
            patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_model),
            patch(
                "carnot.pipeline.three_tier_pipeline.ThreeTierPipeline",
                return_value=mock_three_tier,
            ),
            patch(
                "carnot.pipeline.verify_repair.VerifyRepairPipeline",
                return_value=MagicMock(),
            ),
        ):
            exp882.main()

        assert deliverable.exists(), "Deliverable must be written"
        data = json.loads(deliverable.read_text())
        for field in REQUIRED_RESULT_FIELDS:
            assert field in data, f"REQUIRED_RESULT_FIELDS missing: {field}"

    def test_cascade_skip_rate_present(self, tmp_path: Path) -> None:
        """cascade_skip_rate must be present in the artifact. REQ-BENCH-015."""
        (tmp_path / "results").mkdir(parents=True)
        _write_preflight(tmp_path / "results", live_env_fixed=True)
        deliverable = tmp_path / exp882.DELIVERABLE

        mock_tmpl = _mock_tmpl_factory(tmp_path)

        import torch

        mock_tok = MagicMock()
        mock_tok.eos_token_id = 1
        mock_tok.decode.return_value = "8"
        mock_inputs = MagicMock()
        input_ids = torch.zeros(1, 3, dtype=torch.long)
        mock_inputs.__getitem__ = MagicMock(return_value=input_ids)
        mock_inputs.to.return_value = mock_inputs
        mock_tok.return_value = mock_inputs

        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.generate.return_value = [torch.zeros(4, dtype=torch.long)]

        mock_three_tier = MagicMock()
        mock_three_tier.verify.return_value = (True, "eorm", 0.2)

        with (
            patch.object(exp882, "_REPO_ROOT", tmp_path),
            patch(
                "scripts.experiment_882_live_cascade_v7_gemma4.ExperimentTemplate",
                return_value=mock_tmpl,
            ),
            patch.dict("os.environ", {"CARNOT_FORCE_LIVE": "1"}, clear=False),
            patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tok),
            patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_model),
            patch(
                "carnot.pipeline.three_tier_pipeline.ThreeTierPipeline",
                return_value=mock_three_tier,
            ),
            patch(
                "carnot.pipeline.verify_repair.VerifyRepairPipeline",
                return_value=MagicMock(),
            ),
        ):
            exp882.main()

        data = json.loads(deliverable.read_text())
        assert "cascade_skip_rate" in data
        assert "inference_mode" in data
        assert data["inference_mode"] == "live_gpu"
