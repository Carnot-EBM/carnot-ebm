"""Tests for Experiment 330: Live HuggingFace publish with Exp 328 live-GPU benchmarks.

Wraps Exp 317 publish pipeline and embeds live-GPU benchmark results from Exp 328
into all 16+ HuggingFace model READMEs.

Spec: REQ-PUBLISH-004, SCENARIO-PUBLISH-007, SCENARIO-PUBLISH-008
Run date: 20260415
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_exp328_results() -> dict:
    """Build a minimal Exp 328 live result fixture (matches real file structure)."""
    return {
        "experiment": 328,
        "schema": "carnot.live_fullscale_benchmark.v1",
        "inference_mode": "live_gpu",
        "status": "success",
        "run_date": "20260415",
        "baseline_deviation": {
            "Qwen3.5-0.8B": {
                "baseline_accuracy": 0.2375,
                "published_baseline": 0.25,
                "deviation": -0.0125,
                "within_tolerance": True,
            },
            "Gemma4-E4B-it": {
                "baseline_accuracy": 0.225,
                "published_baseline": 0.8,
                "deviation": -0.575,
                "within_tolerance": False,
            },
        },
        "first_live_run_evidence": {
            "timestamp": "2026-04-15T03:05:43Z",
            "inference_mode": "live_gpu",
            "Qwen3.5-0.8B_baseline_all_accuracy": 0.275,
            "Qwen3.5-0.8B_baseline_all_ci": "[0.234, 0.321]",
            "Gemma4-E4B-it_baseline_all_accuracy": 0.263,
            "Gemma4-E4B-it_baseline_all_ci": "[0.222, 0.308]",
            "note": "Console-captured evidence only",
        },
        "benchmark_n_gsm8k": 80,
        "benchmark_n_humaneval": 5,
    }


def _make_mock_hf_api(existing_readme: str = "") -> MagicMock:
    """Build a mock HfApi for dependency injection."""
    api = MagicMock()
    api.whoami.return_value = {"name": "ianblenke", "type": "user"}
    if existing_readme:
        api.hf_hub_download.return_value = existing_readme
    else:
        api.hf_hub_download.side_effect = Exception("404 README not found")
    return api


# ---------------------------------------------------------------------------
# load_publish_results — REQ-PUBLISH-004
# ---------------------------------------------------------------------------


class TestLoadPublishResults:
    """Validate load_publish_results() schema enforcement."""

    def test_loads_valid_json(self, tmp_path: Path) -> None:
        """load_publish_results() must return a dict from a valid JSON file."""
        from scripts.experiment_330_hf_live_publish import load_publish_results

        data = {"experiment": 330, "status": "success", "n_models_updated": 17}
        p = tmp_path / "result.json"
        p.write_text(json.dumps(data))
        result = load_publish_results(p)
        assert isinstance(result, dict)
        assert result["experiment"] == 330

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        """load_publish_results() must raise FileNotFoundError when path absent."""
        from scripts.experiment_330_hf_live_publish import load_publish_results

        with pytest.raises(FileNotFoundError):
            load_publish_results(tmp_path / "nonexistent.json")

    def test_raises_on_invalid_json(self, tmp_path: Path) -> None:
        """load_publish_results() must raise ValueError when JSON is malformed."""
        from scripts.experiment_330_hf_live_publish import load_publish_results

        p = tmp_path / "bad.json"
        p.write_text("not json {{{")
        with pytest.raises(ValueError, match="JSON"):
            load_publish_results(p)

    def test_raises_on_missing_experiment_key(self, tmp_path: Path) -> None:
        """load_publish_results() must raise ValueError when 'experiment' key absent."""
        from scripts.experiment_330_hf_live_publish import load_publish_results

        p = tmp_path / "no_exp.json"
        p.write_text(json.dumps({"status": "success"}))
        with pytest.raises(ValueError, match="experiment"):
            load_publish_results(p)

    def test_raises_on_missing_status_key(self, tmp_path: Path) -> None:
        """load_publish_results() must raise ValueError when 'status' key absent."""
        from scripts.experiment_330_hf_live_publish import load_publish_results

        p = tmp_path / "no_status.json"
        p.write_text(json.dumps({"experiment": 330}))
        with pytest.raises(ValueError, match="status"):
            load_publish_results(p)


# ---------------------------------------------------------------------------
# validate_live_publish — REQ-PUBLISH-004
# ---------------------------------------------------------------------------


class TestValidateLivePublish:
    """Validate validate_live_publish() raises on non-success artifacts."""

    def test_passes_on_success(self) -> None:
        """validate_live_publish() must not raise when status == 'success'."""
        from scripts.experiment_330_hf_live_publish import validate_live_publish

        validate_live_publish({"experiment": 330, "status": "success"})

    def test_raises_on_blocked(self) -> None:
        """validate_live_publish() must raise ValueError when status == 'blocked'."""
        from scripts.experiment_330_hf_live_publish import validate_live_publish

        with pytest.raises(ValueError, match="blocked"):
            validate_live_publish({"experiment": 330, "status": "blocked"})

    def test_raises_on_error(self) -> None:
        """validate_live_publish() must raise ValueError when status == 'error'."""
        from scripts.experiment_330_hf_live_publish import validate_live_publish

        with pytest.raises(ValueError, match="error"):
            validate_live_publish({"experiment": 330, "status": "error"})

    def test_raises_on_arbitrary_non_success(self) -> None:
        """validate_live_publish() raises for any status that isn't 'success'."""
        from scripts.experiment_330_hf_live_publish import validate_live_publish

        with pytest.raises(ValueError):
            validate_live_publish({"experiment": 330, "status": "pending"})


# ---------------------------------------------------------------------------
# adapt_exp328_to_per_variant — REQ-PUBLISH-004
# ---------------------------------------------------------------------------


class TestAdaptExp328ToPerVariant:
    """Verify Exp 328 live results are adapted to per_variant_results format."""

    def test_produces_all_variant(self) -> None:
        """Adapter must produce a per_variant_results['all'] dict."""
        from scripts.experiment_330_hf_live_publish import adapt_exp328_to_per_variant

        exp328 = _make_exp328_results()
        result = adapt_exp328_to_per_variant(exp328)
        assert "per_variant_results" in result
        assert "all" in result["per_variant_results"]

    def test_models_present_in_all_variant(self) -> None:
        """Both Qwen and Gemma model entries must appear in 'all' variant."""
        from scripts.experiment_330_hf_live_publish import adapt_exp328_to_per_variant

        exp328 = _make_exp328_results()
        result = adapt_exp328_to_per_variant(exp328)
        all_variant = result["per_variant_results"]["all"]
        assert "Qwen3.5-0.8B" in all_variant
        assert "Gemma4-E4B-it" in all_variant

    def test_accuracy_values_from_first_live_run_evidence(self) -> None:
        """Accuracy values must come from first_live_run_evidence (most authoritative)."""
        from scripts.experiment_330_hf_live_publish import adapt_exp328_to_per_variant

        exp328 = _make_exp328_results()
        result = adapt_exp328_to_per_variant(exp328)
        qwen = result["per_variant_results"]["all"]["Qwen3.5-0.8B"]
        # first_live_run_evidence has 0.275 for Qwen3.5-0.8B
        assert abs(qwen["accuracy"] - 0.275) < 1e-6

    def test_inference_mode_is_live_gpu(self) -> None:
        """Adapted result must have inference_mode == 'live_gpu'."""
        from scripts.experiment_330_hf_live_publish import adapt_exp328_to_per_variant

        exp328 = _make_exp328_results()
        result = adapt_exp328_to_per_variant(exp328)
        assert result.get("inference_mode") == "live_gpu"

    def test_n_gsm8k_carried_through(self) -> None:
        """n_gsm8k from Exp 328 must appear in adapted result."""
        from scripts.experiment_330_hf_live_publish import adapt_exp328_to_per_variant

        exp328 = _make_exp328_results()
        result = adapt_exp328_to_per_variant(exp328)
        assert result.get("n_gsm8k") == 80

    def test_returns_none_when_exp328_missing_evidence(self) -> None:
        """Returns None when first_live_run_evidence is missing (no data to embed)."""
        from scripts.experiment_330_hf_live_publish import adapt_exp328_to_per_variant

        result = adapt_exp328_to_per_variant({})
        assert result is None


# ---------------------------------------------------------------------------
# Blocked artifact — SCENARIO-PUBLISH-008
# ---------------------------------------------------------------------------


class TestBlockedArtifact330:
    """Verify blocked artifact schema when credentials are absent."""

    def _get_blocked(self, tmp_path: Path) -> dict:
        from scripts.experiment_330_hf_live_publish import run_experiment_330

        mock_api = MagicMock()
        mock_api.whoami.side_effect = Exception("Not authenticated")
        with (
            patch("subprocess.run", side_effect=FileNotFoundError),
            patch(
                "scripts.experiment_330_hf_live_publish._make_hf_api_330",
                return_value=mock_api,
            ),
            patch(
                "scripts.experiment_317_hf_publish._make_hf_api",
                return_value=mock_api,
            ),
        ):
            return run_experiment_330(
                dry_run=True,
                results_path=tmp_path / "exp330.json",
                hf_api=mock_api,
            )

    def test_status_is_blocked(self, tmp_path: Path) -> None:
        """Blocked artifact must have status == 'blocked'."""
        result = self._get_blocked(tmp_path)
        assert result.get("status") == "blocked"

    def test_experiment_is_330(self, tmp_path: Path) -> None:
        """Blocked artifact must have experiment == 330."""
        result = self._get_blocked(tmp_path)
        assert result.get("experiment") == 330

    def test_next_action_contains_login(self, tmp_path: Path) -> None:
        """Blocked artifact must include next_action with huggingface-cli login."""
        result = self._get_blocked(tmp_path)
        action = result.get("next_action", "")
        assert "huggingface-cli login" in action

    def test_n_models_updated_is_zero(self, tmp_path: Path) -> None:
        """Blocked artifact must have n_models_updated == 0."""
        result = self._get_blocked(tmp_path)
        assert result.get("n_models_updated") == 0

    def test_written_to_disk(self, tmp_path: Path) -> None:
        """Blocked artifact must be written to results_path."""
        results_file = tmp_path / "exp330_blocked.json"
        from scripts.experiment_330_hf_live_publish import run_experiment_330

        mock_api = MagicMock()
        mock_api.whoami.side_effect = Exception("Not authenticated")
        with (
            patch("subprocess.run", side_effect=FileNotFoundError),
            patch(
                "scripts.experiment_330_hf_live_publish._make_hf_api_330",
                return_value=mock_api,
            ),
            patch(
                "scripts.experiment_317_hf_publish._make_hf_api",
                return_value=mock_api,
            ),
        ):
            run_experiment_330(dry_run=True, results_path=results_file, hf_api=mock_api)
        assert results_file.exists()
        data = json.loads(results_file.read_text())
        assert data["status"] == "blocked"


# ---------------------------------------------------------------------------
# Artifact schema (dry-run success) — REQ-PUBLISH-004
# ---------------------------------------------------------------------------


class TestRunExperiment330Schema:
    """Verify exp330 artifact schema in dry-run success mode."""

    def _get_dry_run_result(self, tmp_path: Path, exp328_data: dict | None = None) -> dict:
        from scripts.experiment_330_hf_live_publish import run_experiment_330

        mock_api = _make_mock_hf_api(existing_readme="")

        exp328_path = tmp_path / "experiment_328_live_fullscale_results.json"
        if exp328_data is not None:
            exp328_path.write_text(json.dumps(exp328_data))

        with (
            patch("subprocess.run", side_effect=FileNotFoundError),
            patch(
                "scripts.experiment_330_hf_live_publish._make_hf_api_330",
                return_value=mock_api,
            ),
            patch(
                "scripts.experiment_317_hf_publish._make_hf_api",
                return_value=mock_api,
            ),
        ):
            return run_experiment_330(
                dry_run=True,
                results_path=tmp_path / "exp330.json",
                exp328_results_path=exp328_path,
                hf_api=mock_api,
            )

    def test_experiment_is_330(self, tmp_path: Path) -> None:
        """Artifact must have experiment == 330."""
        result = self._get_dry_run_result(tmp_path, _make_exp328_results())
        assert result.get("experiment") == 330

    def test_schema_field(self, tmp_path: Path) -> None:
        """Artifact must have schema == 'carnot.hf_publish.v1'."""
        result = self._get_dry_run_result(tmp_path, _make_exp328_results())
        assert result.get("schema") == "carnot.hf_publish.v1"

    def test_status_success(self, tmp_path: Path) -> None:
        """Artifact must have status == 'success' in dry-run pass."""
        result = self._get_dry_run_result(tmp_path, _make_exp328_results())
        assert result.get("status") == "success"

    def test_n_models_updated_present(self, tmp_path: Path) -> None:
        """Artifact must include n_models_updated."""
        result = self._get_dry_run_result(tmp_path, _make_exp328_results())
        assert "n_models_updated" in result

    def test_fcv_updated_present(self, tmp_path: Path) -> None:
        """Artifact must include fcv_updated bool."""
        result = self._get_dry_run_result(tmp_path, _make_exp328_results())
        assert "fcv_updated" in result

    def test_joint_placeholder_created_present(self, tmp_path: Path) -> None:
        """Artifact must include joint_placeholder_created bool."""
        result = self._get_dry_run_result(tmp_path, _make_exp328_results())
        assert "joint_placeholder_created" in result

    def test_live_benchmark_embedded_true_when_exp328_present(
        self, tmp_path: Path
    ) -> None:
        """live_benchmark_embedded must be True when Exp 328 results are available."""
        result = self._get_dry_run_result(tmp_path, _make_exp328_results())
        assert result.get("live_benchmark_embedded") is True

    def test_live_benchmark_embedded_false_when_exp328_absent(
        self, tmp_path: Path
    ) -> None:
        """live_benchmark_embedded must be False when Exp 328 results are absent."""
        result = self._get_dry_run_result(tmp_path, exp328_data=None)
        assert result.get("live_benchmark_embedded") is False

    def test_exp328_baseline_accuracy_per_model_present(self, tmp_path: Path) -> None:
        """Artifact must include exp328_baseline_accuracy dict per model."""
        result = self._get_dry_run_result(tmp_path, _make_exp328_results())
        bm = result.get("exp328_baseline_accuracy", {})
        assert "Qwen3.5-0.8B" in bm
        assert "Gemma4-E4B-it" in bm

    def test_n_idempotency_checks_passed_present(self, tmp_path: Path) -> None:
        """Artifact must include n_idempotency_checks_passed."""
        result = self._get_dry_run_result(tmp_path, _make_exp328_results())
        assert "n_idempotency_checks_passed" in result

    def test_written_to_disk(self, tmp_path: Path) -> None:
        """Artifact must be written to results_path."""
        results_file = tmp_path / "exp330_disk.json"
        from scripts.experiment_330_hf_live_publish import run_experiment_330

        mock_api = _make_mock_hf_api()

        exp328_path = tmp_path / "experiment_328_live_fullscale_results.json"
        exp328_path.write_text(json.dumps(_make_exp328_results()))

        with (
            patch("subprocess.run", side_effect=FileNotFoundError),
            patch(
                "scripts.experiment_330_hf_live_publish._make_hf_api_330",
                return_value=mock_api,
            ),
            patch(
                "scripts.experiment_317_hf_publish._make_hf_api",
                return_value=mock_api,
            ),
        ):
            run_experiment_330(
                dry_run=True,
                results_path=results_file,
                exp328_results_path=exp328_path,
                hf_api=mock_api,
            )
        assert results_file.exists()
        data = json.loads(results_file.read_text())
        assert data["experiment"] == 330


# ---------------------------------------------------------------------------
# n_models_updated >= 16 when credentials present — REQ-PUBLISH-004
# ---------------------------------------------------------------------------


class TestModelsUpdatedCount:
    """n_models_updated must be >= 16 when credentials OK and no existing sentinel."""

    def test_n_models_updated_at_least_16(self, tmp_path: Path) -> None:
        """At least 16 per-token EBM repos + FCV + joint = 18 total updated."""
        from scripts.experiment_330_hf_live_publish import run_experiment_330

        mock_api = _make_mock_hf_api(existing_readme="")

        exp328_path = tmp_path / "experiment_328_live_fullscale_results.json"
        exp328_path.write_text(json.dumps(_make_exp328_results()))

        with (
            patch("subprocess.run", side_effect=FileNotFoundError),
            patch(
                "scripts.experiment_330_hf_live_publish._make_hf_api_330",
                return_value=mock_api,
            ),
            patch(
                "scripts.experiment_317_hf_publish._make_hf_api",
                return_value=mock_api,
            ),
        ):
            result = run_experiment_330(
                dry_run=True,
                results_path=tmp_path / "exp330.json",
                exp328_results_path=exp328_path,
                hf_api=mock_api,
            )
        assert result.get("n_models_updated", 0) >= 16


# ---------------------------------------------------------------------------
# Idempotency — SCENARIO-PUBLISH-007
# ---------------------------------------------------------------------------


class TestIdempotency330:
    """Running exp330 twice must not create duplicate sentinel entries."""

    def test_all_idempotent_on_second_run(self, tmp_path: Path) -> None:
        """When all READMEs already have the Phase 1 sentinel, n_models_updated == 0."""
        from scripts.experiment_317_hf_publish import _PHASE1_SENTINEL
        from scripts.experiment_330_hf_live_publish import run_experiment_330

        already_patched = f"{_PHASE1_SENTINEL}\nAlready patched."

        def _side_effect(repo_id: str, filename: str, repo_type: str) -> str:
            if repo_id == "Carnot-EBM/carnot-formal-claim-verifier-v1":
                return "<!-- carnot-exp317-exp316-results -->\nPatched."
            if repo_id == "Carnot-EBM/carnot-joint-constraint-v1":
                return "RESEARCH PROTOTYPE — weights not published\nDone."
            return already_patched

        mock_api = MagicMock()
        mock_api.whoami.return_value = {"name": "ianblenke"}
        mock_api.hf_hub_download.side_effect = _side_effect

        exp328_path = tmp_path / "experiment_328_live_fullscale_results.json"
        exp328_path.write_text(json.dumps(_make_exp328_results()))

        with (
            patch("subprocess.run", side_effect=FileNotFoundError),
            patch(
                "scripts.experiment_330_hf_live_publish._make_hf_api_330",
                return_value=mock_api,
            ),
            patch(
                "scripts.experiment_317_hf_publish._make_hf_api",
                return_value=mock_api,
            ),
        ):
            result = run_experiment_330(
                dry_run=False,
                results_path=tmp_path / "exp330.json",
                exp328_results_path=exp328_path,
                hf_api=mock_api,
            )

        assert result.get("n_models_updated") == 0
        mock_api.upload_file.assert_not_called()


# ---------------------------------------------------------------------------
# Results JSON on-disk schema (if file exists)
# ---------------------------------------------------------------------------


class TestResultsJsonSchema330:
    """Validate results/experiment_330_hf_publish_results.json when it exists."""

    @pytest.fixture
    def results(self) -> dict:
        results_path = (
            Path(__file__).parent.parent.parent
            / "results"
            / "experiment_330_hf_publish_results.json"
        )
        if not results_path.exists():
            pytest.skip("experiment_330_hf_publish_results.json not yet generated")
        return json.loads(results_path.read_text())

    def test_experiment_is_330(self, results: dict) -> None:
        assert results.get("experiment") == 330

    def test_schema_is_correct(self, results: dict) -> None:
        assert results.get("schema") == "carnot.hf_publish.v1"

    def test_has_status(self, results: dict) -> None:
        assert "status" in results

    def test_has_n_models_updated(self, results: dict) -> None:
        assert "n_models_updated" in results

    def test_has_fcv_updated(self, results: dict) -> None:
        assert "fcv_updated" in results

    def test_has_joint_placeholder_created(self, results: dict) -> None:
        assert "joint_placeholder_created" in results

    def test_has_live_benchmark_embedded(self, results: dict) -> None:
        assert "live_benchmark_embedded" in results

    def test_blocked_has_next_action(self, results: dict) -> None:
        if results.get("status") != "blocked":
            pytest.skip("Not blocked")
        assert "huggingface-cli login" in results.get("next_action", "")

    def test_success_n_models_at_least_16(self, results: dict) -> None:
        if results.get("status") != "success":
            pytest.skip("Not successful")
        assert results.get("n_models_updated", 0) + results.get("n_models_skipped", 0) >= 16
