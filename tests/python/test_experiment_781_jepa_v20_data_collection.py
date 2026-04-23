"""Tests for Exp 781: JEPA v20 live data collection (fover_labeled_steps_live_v2.json).

Coverage targets
----------------
_is_truthy_env():
  - Returns False for absent var
  - Returns False for '0', 'false', 'False', ''
  - Returns True for '1', 'true', 'True', 'yes'

_load_gsm8k_questions():
  - Returns exactly n questions
  - Each question has 'question' and 'question_id' keys
  - Synthetic fallback fires when datasets raises ImportError

_load_checkpoint() / _save_checkpoint():
  - Returns None when no checkpoint file exists
  - Returns parsed data when checkpoint file is present
  - Save creates the file; re-load retrieves same data

run_experiment():
  - blocked_no_live_gpu verdict when CARNOT_FORCE_LIVE is unset/falsy
  - kill_gpu_zombies() is called before setup_gpu() when live_gpu
  - labeled pairs written to fover_labeled_steps_live_v2.json (not the v1 file)
  - honest_verdict='real_data_collected_sufficient' requires n_labeled >= 80
    AND inference_mode='live_gpu'
  - honest_verdict='real_data_collected_insufficient' for 20 <= n_labeled < 80
  - honest_verdict='real_data_below_threshold' for n_labeled < 20
  - all REQUIRED_RESULT_FIELDS present in artifact
  - GPU unhealthy → blocked artifact written

Spec: REQ-LEARN-048, REQ-LEARN-049, SCENARIO-LEARN-092, SCENARIO-LEARN-093
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _d in [str(_REPO_ROOT / "python"), str(_REPO_ROOT / "scripts")]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from scripts.experiment_template import REQUIRED_RESULT_FIELDS  # noqa: E402

# Import helpers under test directly (not the script entrypoint, which triggers
# apply_env_autofix at module level and may touch GPU hardware).
import importlib  # noqa: E402

_mod = importlib.import_module("experiment_781_jepa_v20_data_collection")

_is_truthy_env = _mod._is_truthy_env
_load_gsm8k_questions = _mod._load_gsm8k_questions
_load_checkpoint = _mod._load_checkpoint
_save_checkpoint = _mod._save_checkpoint
run_experiment = _mod.run_experiment


# ---------------------------------------------------------------------------
# _is_truthy_env
# ---------------------------------------------------------------------------


class TestIsTruthyEnv:
    """Unit tests for _is_truthy_env().

    Spec: REQ-LEARN-049
    """

    def test_absent_returns_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Absent env var → False."""
        monkeypatch.delenv("CARNOT_FORCE_LIVE", raising=False)
        assert _is_truthy_env("CARNOT_FORCE_LIVE") is False

    @pytest.mark.parametrize("val", ["0", "false", "False", ""])
    def test_falsy_values_return_false(self, monkeypatch: pytest.MonkeyPatch, val: str) -> None:
        """Falsy string values → False.

        Spec: REQ-LEARN-049 (honest reporting requires gate to actually block when
        CARNOT_FORCE_LIVE is falsy, not just absent).
        """
        monkeypatch.setenv("CARNOT_FORCE_LIVE", val)
        assert _is_truthy_env("CARNOT_FORCE_LIVE") is False

    @pytest.mark.parametrize("val", ["1", "true", "True", "yes"])
    def test_truthy_values_return_true(self, monkeypatch: pytest.MonkeyPatch, val: str) -> None:
        """Truthy string values → True."""
        monkeypatch.setenv("CARNOT_FORCE_LIVE", val)
        assert _is_truthy_env("CARNOT_FORCE_LIVE") is True


# ---------------------------------------------------------------------------
# _load_gsm8k_questions
# ---------------------------------------------------------------------------


class TestLoadGsm8kQuestions:
    """Unit tests for _load_gsm8k_questions().

    Spec: REQ-LEARN-048
    """

    def test_returns_n_questions_synthetic_fallback(self) -> None:
        """When datasets is unavailable, returns exactly n synthetic questions.

        Verifies that the fallback path produces the right count without
        requiring the HuggingFace datasets library.
        """
        with patch.dict(sys.modules, {"datasets": None}):
            qs = _load_gsm8k_questions(10, seed=9999)
        assert len(qs) == 10

    def test_each_question_has_required_keys(self) -> None:
        """Every returned question has 'question' and 'question_id' keys."""
        with patch.dict(sys.modules, {"datasets": None}):
            qs = _load_gsm8k_questions(5, seed=9999)
        for q in qs:
            assert "question" in q
            assert "question_id" in q

    def test_question_ids_are_unique(self) -> None:
        """All question_ids are distinct (no duplicates in the sample)."""
        with patch.dict(sys.modules, {"datasets": None}):
            qs = _load_gsm8k_questions(20, seed=9999)
        ids = [q["question_id"] for q in qs]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# _load_checkpoint / _save_checkpoint
# ---------------------------------------------------------------------------


class TestCheckpoint:
    """Unit tests for checkpoint helpers.

    Spec: REQ-LEARN-048 (checkpoint ensures no work is lost on partial runs)
    """

    def test_load_returns_none_when_absent(self, tmp_path: Path) -> None:
        """No checkpoint file → None returned."""
        with patch.object(_mod, "CHECKPOINT_PATH", str(tmp_path / "no_ckpt.json")):
            result = _load_checkpoint()
        assert result is None

    def test_save_then_load_round_trip(self, tmp_path: Path) -> None:
        """Save checkpoint, then load it back and get the same data."""
        ckpt_path = str(tmp_path / "exp781_ckpt.json")
        responses = [{"question_id": "q0", "question": "2+2?", "response": "4"}]
        with patch.object(_mod, "CHECKPOINT_PATH", ckpt_path):
            _save_checkpoint(responses)
            loaded = _load_checkpoint()
        assert loaded is not None
        assert loaded["responses"] == responses


# ---------------------------------------------------------------------------
# run_experiment — blocked path
# ---------------------------------------------------------------------------


class TestRunExperimentBlocked:
    """Tests for the 'blocked_no_live_gpu' code path.

    Spec: REQ-LEARN-048, REQ-LEARN-049, SCENARIO-LEARN-092
    """

    def _make_tmpl(self, tmp_path: Path) -> MagicMock:
        """Build a minimal mock ExperimentTemplate that satisfies run_experiment."""
        tmpl = MagicMock()
        deliverable = tmp_path / "exp781.json"
        tmpl._output_path = deliverable
        tmpl.build_result.side_effect = lambda data, **kw: {
            **{f: "x" for f in REQUIRED_RESULT_FIELDS},
            **data,
            **kw,
        }
        return tmpl

    def test_blocked_when_carnot_force_live_absent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CARNOT_FORCE_LIVE absent → honest_verdict='blocked_no_live_gpu'.

        Spec: REQ-LEARN-049 (must report honest verdict including blocked state)
        """
        monkeypatch.delenv("CARNOT_FORCE_LIVE", raising=False)
        tmpl = self._make_tmpl(tmp_path)
        artifact = run_experiment(tmpl)
        assert artifact["honest_verdict"] == "blocked_no_live_gpu"
        assert artifact["inference_mode"] == "blocked_no_live_gpu"

    def test_blocked_when_carnot_force_live_zero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CARNOT_FORCE_LIVE='0' → honest_verdict='blocked_no_live_gpu'.

        Spec: REQ-LEARN-049
        """
        monkeypatch.setenv("CARNOT_FORCE_LIVE", "0")
        tmpl = self._make_tmpl(tmp_path)
        artifact = run_experiment(tmpl)
        assert artifact["honest_verdict"] == "blocked_no_live_gpu"

    def test_blocked_artifact_written_to_disk(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Blocked artifact is written to tmpl._output_path."""
        monkeypatch.delenv("CARNOT_FORCE_LIVE", raising=False)
        tmpl = self._make_tmpl(tmp_path)
        run_experiment(tmpl)
        assert (tmp_path / "exp781.json").exists()

    def test_blocked_artifact_has_required_fields(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Blocked artifact contains all REQUIRED_RESULT_FIELDS.

        Spec: REQ-LEARN-049
        """
        monkeypatch.delenv("CARNOT_FORCE_LIVE", raising=False)
        tmpl = self._make_tmpl(tmp_path)
        artifact = run_experiment(tmpl)
        for field in REQUIRED_RESULT_FIELDS:
            assert field in artifact, f"Missing required field: {field}"


# ---------------------------------------------------------------------------
# run_experiment — live path: call order and file isolation
# ---------------------------------------------------------------------------


class TestRunExperimentLive:
    """Tests for the live_gpu code path.

    Spec: REQ-LEARN-048 (kill_gpu_zombies before setup_gpu; separate v2 file)
    Spec: REQ-LEARN-049 (honest verdict tiers based on n_labeled)
    Spec: SCENARIO-LEARN-092 (call order: env_autofix → kill_gpu_zombies → setup_gpu)
    Spec: SCENARIO-LEARN-093 (labeled pairs to v2 file, not v1 file)
    """

    def _make_live_tmpl(self, tmp_path: Path, n_labeled: int) -> tuple[MagicMock, list[str]]:
        """Build a mock tmpl and a call-order log for live path tests.

        Returns (tmpl, call_log) — call_log is mutated by the mocks to record order.
        """
        call_log: list[str] = []

        def _mock_setup_gpu(specs: list) -> dict:
            call_log.append("setup_gpu")
            return {"all_healthy": True, "models": []}

        tmpl = MagicMock()
        deliverable = tmp_path / "exp781.json"
        tmpl._output_path = deliverable
        tmpl.setup_gpu.side_effect = _mock_setup_gpu
        tmpl.build_result.side_effect = lambda data, **kw: {
            **{f: "x" for f in REQUIRED_RESULT_FIELDS},
            **data,
            **kw,
        }

        return tmpl, call_log

    def _mock_kill_gpu_zombies(self, call_log: list[str]) -> MagicMock:
        """Return a mock kill_gpu_zombies that appends to call_log."""
        mock = MagicMock()
        mock.return_value = MagicMock(
            honest_verdict="no_zombies_found",
            pids_killed=[],
            vram_freed_mb=0.0,
        )

        def _side_effect(gpu_index: int = 0) -> MagicMock:
            call_log.append("kill_gpu_zombies")
            return mock.return_value

        mock.side_effect = _side_effect
        return mock

    def _make_annotated_pairs(self, n_labeled: int) -> tuple[list, list]:
        """Build minimal mock annotation output that produces n_labeled pairs."""
        responses = [{"question_id": f"q{i}", "question": f"Q{i}", "response": f"R{i}"} for i in range(5)]
        pairs = [
            {"question_id": f"q{i}", "step_text": f"step{i}", "label": "correct", "confidence": 1.0}
            for i in range(n_labeled)
        ]
        return responses, pairs

    def test_kill_gpu_zombies_called_before_setup_gpu(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """kill_gpu_zombies() is called BEFORE setup_gpu().

        This is the critical ordering requirement from RETRO-028:
        zombie VRAM must be freed before the model load request is issued.

        Spec: REQ-LEARN-048, SCENARIO-LEARN-092
        """
        monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")
        tmpl, call_log = self._make_live_tmpl(tmp_path, n_labeled=80)
        kill_mock = self._mock_kill_gpu_zombies(call_log)

        responses, pairs = self._make_annotated_pairs(80)

        mock_ann_inst = MagicMock()
        mock_ann_inst.annotate_corpus.return_value = [[]]
        mock_ann_inst.to_training_pairs.return_value = pairs
        mock_ann_cls = MagicMock(return_value=mock_ann_inst)

        mock_pipeline_inst = MagicMock()
        mock_pipeline_inst.verify_and_repair.return_value = MagicMock(final_response="4")
        mock_pipeline_cls = MagicMock(return_value=mock_pipeline_inst)

        with (
            patch.object(_mod, "kill_gpu_zombies", kill_mock),
            patch.object(_mod, "_load_gsm8k_questions", return_value=responses),
            patch.object(_mod, "_load_checkpoint", return_value=None),
            patch.object(_mod, "_save_checkpoint"),
            patch("carnot.pipeline.verify_repair.VerifyRepairPipeline", mock_pipeline_cls),
            patch.object(_mod, "FOVERAnnotator", mock_ann_cls),
            patch.object(_mod, "LABELED_STEPS_V2_PATH", str(tmp_path / "v2.json")),
        ):
            run_experiment(tmpl)

        # kill_gpu_zombies must appear before setup_gpu in call_log
        assert "kill_gpu_zombies" in call_log
        assert "setup_gpu" in call_log
        assert call_log.index("kill_gpu_zombies") < call_log.index("setup_gpu"), (
            "kill_gpu_zombies must be called BEFORE setup_gpu (REQ-LEARN-048)"
        )

    def test_labeled_pairs_written_to_v2_not_v1(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Labeled pairs written to v2 file; v1 file not created/modified.

        Spec: REQ-LEARN-048, SCENARIO-LEARN-093
        """
        monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")
        tmpl, call_log = self._make_live_tmpl(tmp_path, n_labeled=80)
        kill_mock = self._mock_kill_gpu_zombies(call_log)

        v2_path = tmp_path / "fover_labeled_steps_live_v2.json"
        v1_path = tmp_path / "fover_labeled_steps_live.json"
        responses, pairs = self._make_annotated_pairs(80)

        mock_ann_inst = MagicMock()
        mock_ann_inst.annotate_corpus.return_value = [[]]
        mock_ann_inst.to_training_pairs.return_value = pairs
        mock_ann_cls = MagicMock(return_value=mock_ann_inst)

        mock_pipeline_inst = MagicMock()
        mock_pipeline_inst.verify_and_repair.return_value = MagicMock(final_response="4")
        mock_pipeline_cls = MagicMock(return_value=mock_pipeline_inst)

        with (
            patch.object(_mod, "kill_gpu_zombies", kill_mock),
            patch.object(_mod, "_load_gsm8k_questions", return_value=responses),
            patch.object(_mod, "_load_checkpoint", return_value=None),
            patch.object(_mod, "_save_checkpoint"),
            patch("carnot.pipeline.verify_repair.VerifyRepairPipeline", mock_pipeline_cls),
            patch.object(_mod, "FOVERAnnotator", mock_ann_cls),
            patch.object(_mod, "LABELED_STEPS_V2_PATH", str(v2_path)),
        ):
            run_experiment(tmpl)

        assert v2_path.exists(), "fover_labeled_steps_live_v2.json must be written"
        assert not v1_path.exists(), (
            "fover_labeled_steps_live.json (v1 baseline) must NOT be touched (REQ-LEARN-048)"
        )
        written = json.loads(v2_path.read_text())
        assert isinstance(written, list)
        assert len(written) == 80

    @pytest.mark.parametrize(
        "n_labeled,expected_verdict",
        [
            (80, "real_data_collected_sufficient"),
            (100, "real_data_collected_sufficient"),
            (79, "real_data_collected_insufficient"),
            (20, "real_data_collected_insufficient"),
            (19, "real_data_below_threshold"),
            (0, "real_data_below_threshold"),
        ],
    )
    def test_honest_verdict_tiers(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        n_labeled: int,
        expected_verdict: str,
    ) -> None:
        """Honest verdict is determined solely by n_labeled (REQ-LEARN-049).

        Spec: REQ-LEARN-049, SCENARIO-LEARN-093
        """
        monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")
        tmpl, call_log = self._make_live_tmpl(tmp_path, n_labeled=n_labeled)
        kill_mock = self._mock_kill_gpu_zombies(call_log)

        responses, pairs = self._make_annotated_pairs(n_labeled)

        mock_ann_inst = MagicMock()
        mock_ann_inst.annotate_corpus.return_value = [[]]
        mock_ann_inst.to_training_pairs.return_value = pairs
        mock_ann_cls = MagicMock(return_value=mock_ann_inst)

        mock_pipeline_inst = MagicMock()
        mock_pipeline_inst.verify_and_repair.return_value = MagicMock(final_response="4")
        mock_pipeline_cls = MagicMock(return_value=mock_pipeline_inst)

        with (
            patch.object(_mod, "kill_gpu_zombies", kill_mock),
            patch.object(_mod, "_load_gsm8k_questions", return_value=responses),
            patch.object(_mod, "_load_checkpoint", return_value=None),
            patch.object(_mod, "_save_checkpoint"),
            patch("carnot.pipeline.verify_repair.VerifyRepairPipeline", mock_pipeline_cls),
            patch.object(_mod, "FOVERAnnotator", mock_ann_cls),
            patch.object(_mod, "LABELED_STEPS_V2_PATH", str(tmp_path / "v2.json")),
        ):
            artifact = run_experiment(tmpl)

        assert artifact["honest_verdict"] == expected_verdict, (
            f"n_labeled={n_labeled} should give '{expected_verdict}', "
            f"got '{artifact['honest_verdict']}'"
        )

    def test_sufficient_verdict_requires_live_gpu(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """'real_data_collected_sufficient' requires inference_mode='live_gpu' (REQ-LEARN-049).

        A non-live run with n_labeled >= 80 is impossible by design (the blocked
        path returns before annotation), but we confirm the blocked path yields
        'blocked_no_live_gpu', not 'real_data_collected_sufficient'.
        """
        monkeypatch.setenv("CARNOT_FORCE_LIVE", "0")
        tmpl = MagicMock()
        deliverable = tmp_path / "exp781.json"
        tmpl._output_path = deliverable
        tmpl.build_result.side_effect = lambda data, **kw: {
            **{f: "x" for f in REQUIRED_RESULT_FIELDS},
            **data,
            **kw,
        }
        artifact = run_experiment(tmpl)
        assert artifact["honest_verdict"] != "real_data_collected_sufficient"
        assert artifact["inference_mode"] != "live_gpu"

    def test_gpu_unhealthy_returns_blocked(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GPU setup failure → blocked artifact, not a crash.

        Spec: REQ-LEARN-048
        """
        monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")
        call_log: list[str] = []
        kill_mock = self._mock_kill_gpu_zombies(call_log)

        tmpl = MagicMock()
        deliverable = tmp_path / "exp781.json"
        tmpl._output_path = deliverable
        tmpl.setup_gpu.return_value = {"all_healthy": False, "models": [{"error": "OOM"}]}
        tmpl.build_result.side_effect = lambda data, **kw: {
            **{f: "x" for f in REQUIRED_RESULT_FIELDS},
            **data,
            **kw,
        }

        with patch.object(_mod, "kill_gpu_zombies", kill_mock):
            artifact = run_experiment(tmpl)

        assert artifact["honest_verdict"] == "blocked_no_live_gpu"
        assert (tmp_path / "exp781.json").exists()
