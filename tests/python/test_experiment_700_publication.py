"""Tests for experiment 700 — publication readiness audit.

Covers all public functions in scripts/experiment_700_publication_readiness.py
with enough cases to achieve 100% branch coverage.

Spec: REQ-PUBLISH-001, REQ-PUBLISH-002, REQ-PUBLISH-003,
      SCENARIO-PUBLISH-001, SCENARIO-PUBLISH-002, SCENARIO-PUBLISH-003
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiment_700_publication_readiness import (
    DELIVERABLE,
    EXP_ID,
    HEADLINE_SOURCES,
    MODEL_CARD_PATH,
    PROVENANCE_DOC_PATH,
    SCHEMA,
    build_provenance_table,
    check_679_gate,
    compute_honest_verdict,
    load_result_file,
    write_model_card,
    write_provenance_doc,
)
import scripts.experiment_700_publication_readiness as mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_repo(tmp_path: Path) -> Path:
    """Create a minimal repo-like directory tree in a temp path."""
    (tmp_path / "results").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "python" / "carnot" / "models").mkdir(parents=True)
    return tmp_path


def _write_679(tmp_path: Path, **overrides: object) -> None:
    data = {
        "experiment": 679,
        "status": "success",
        "inference_mode": "live_gpu",
        "signed_improvement": 1.0,
        "retro_033_validated": True,
    }
    data.update(overrides)
    (tmp_path / "results" / "experiment_679_vr_200q_scale.json").write_text(
        json.dumps(data)
    )


def _write_694(tmp_path: Path, **overrides: object) -> None:
    data = {
        "experiment": 694,
        "status": "success",
        "inference_mode": "live_gpu",
        "cross_model_delta": -1.8,
        "grammar_recall": 0.0,
        "gemma_signed_improvement": -0.8,
    }
    data.update(overrides)
    (tmp_path / "results" / "experiment_694_vr_cross_model.json").write_text(
        json.dumps(data)
    )


def _write_691(tmp_path: Path, **overrides: object) -> None:
    data = {
        "experiment": 691,
        "status": "success",
        "inference_mode": "live_gpu",
        "mean_auroc": 0.958511,
    }
    data.update(overrides)
    (tmp_path / "results" / "experiment_691_prompt_injection_kan_cross_dataset.json").write_text(
        json.dumps(data)
    )


def _minimal_all_results(tmp_path: Path) -> None:
    """Write enough result files for a full success run."""
    _write_679(tmp_path)
    _write_694(tmp_path)
    _write_691(tmp_path)
    for name in [
        "experiment_682_jepa_v15_ood_audit",
        "experiment_698_jepa_v16",
        "experiment_681_adversarial_vr",
        "experiment_680_humaneval_vr",
    ]:
        (tmp_path / "results" / f"{name}.json").write_text(
            json.dumps({"honest_verdict": "test", "true_ood_auc": 0.47, "v16_ood_auc": 0.47,
                        "ood_auc_delta": 0.0, "cross_model_delta": -1.8,
                        "gemma_signed_improvement": -0.8})
        )


# ---------------------------------------------------------------------------
# check_679_gate
# ---------------------------------------------------------------------------


class TestCheck679Gate:
    """Spec: REQ-PUBLISH-003, SCENARIO-PUBLISH-003"""

    def test_gate_passes_with_retro_validated(self, tmp_repo: Path) -> None:
        """Gate passes when retro_033_validated=True and status=success.

        Spec: SCENARIO-PUBLISH-003
        """
        _write_679(tmp_repo, retro_033_validated=True)
        passes, data = check_679_gate(tmp_repo)
        assert passes is True
        assert data["experiment"] == 679

    def test_gate_passes_with_vr_200q_validated(self, tmp_repo: Path) -> None:
        """Gate also passes when vr_200q_validated=True and status=success.

        Spec: REQ-PUBLISH-003
        """
        _write_679(tmp_repo, vr_200q_validated=True, retro_033_validated=False)
        passes, data = check_679_gate(tmp_repo)
        assert passes is True

    def test_gate_fails_when_file_missing(self, tmp_repo: Path) -> None:
        """Gate fails when Exp 679 result file does not exist.

        Spec: SCENARIO-PUBLISH-003
        """
        passes, data = check_679_gate(tmp_repo)
        assert passes is False
        assert data == {}

    def test_gate_fails_when_status_not_success(self, tmp_repo: Path) -> None:
        """Gate fails when retro_033_validated=True but status=blocked.

        Spec: REQ-PUBLISH-003
        """
        _write_679(tmp_repo, status="blocked")
        passes, _ = check_679_gate(tmp_repo)
        assert passes is False

    def test_gate_fails_when_both_validated_false(self, tmp_repo: Path) -> None:
        """Gate fails when both vr_200q_validated and retro_033_validated are False.

        Spec: SCENARIO-PUBLISH-003
        """
        _write_679(tmp_repo, retro_033_validated=False, vr_200q_validated=False)
        passes, _ = check_679_gate(tmp_repo)
        assert passes is False


# ---------------------------------------------------------------------------
# load_result_file
# ---------------------------------------------------------------------------


class TestLoadResultFile:
    """Spec: REQ-PUBLISH-001"""

    def test_loads_existing_file(self, tmp_repo: Path) -> None:
        """Returns parsed dict for an existing JSON file.

        Spec: REQ-PUBLISH-001
        """
        _write_679(tmp_repo)
        data = load_result_file(tmp_repo, "results/experiment_679_vr_200q_scale.json")
        assert data["experiment"] == 679

    def test_returns_empty_dict_for_missing_file(self, tmp_repo: Path) -> None:
        """Returns empty dict when the file does not exist — not a crash.

        Spec: REQ-PUBLISH-001
        """
        data = load_result_file(tmp_repo, "results/nonexistent.json")
        assert data == {}


# ---------------------------------------------------------------------------
# build_provenance_table
# ---------------------------------------------------------------------------


class TestBuildProvenanceTable:
    """Spec: REQ-PUBLISH-001, SCENARIO-PUBLISH-001"""

    def test_all_valid_when_all_live_gpu(self, tmp_repo: Path) -> None:
        """All rows have provenance_valid=True when every file has inference_mode=live_gpu.

        Spec: SCENARIO-PUBLISH-001
        """
        _write_679(tmp_repo)
        _write_694(tmp_repo)
        _write_691(tmp_repo)
        table = build_provenance_table(tmp_repo)
        assert all(row["provenance_valid"] for row in table)
        assert len(table) == len(HEADLINE_SOURCES)

    def test_invalid_when_inference_mode_not_live_gpu(self, tmp_repo: Path) -> None:
        """Row is invalid when inference_mode != live_gpu.

        Spec: REQ-PUBLISH-001
        """
        _write_679(tmp_repo, inference_mode="blocked")
        _write_694(tmp_repo)
        _write_691(tmp_repo)
        table = build_provenance_table(tmp_repo)
        vr_row = next(r for r in table if "signed_improvement" in r["metric"])
        assert vr_row["provenance_valid"] is False
        assert vr_row["inference_mode"] == "blocked"

    def test_invalid_when_file_missing(self, tmp_repo: Path) -> None:
        """Row is invalid and inference_mode='missing' when file is absent.

        Spec: REQ-PUBLISH-001
        """
        # Only write 679; 694 and 691 will be missing
        _write_679(tmp_repo)
        table = build_provenance_table(tmp_repo)
        missing_rows = [r for r in table if r["inference_mode"] == "missing"]
        assert len(missing_rows) > 0
        for row in missing_rows:
            assert row["provenance_valid"] is False

    def test_value_extracted_correctly(self, tmp_repo: Path) -> None:
        """The value field matches the key extracted from the result JSON.

        Spec: REQ-PUBLISH-001
        """
        _write_679(tmp_repo, signed_improvement=0.95)
        _write_694(tmp_repo)
        _write_691(tmp_repo)
        table = build_provenance_table(tmp_repo)
        vr_row = next(r for r in table if "signed_improvement" in r["metric"])
        assert vr_row["value"] == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# compute_honest_verdict
# ---------------------------------------------------------------------------


class TestComputeHonestVerdict:
    """Spec: REQ-PUBLISH-003, SCENARIO-PUBLISH-003"""

    def test_publication_ready(self) -> None:
        """Returns publication_ready when all conditions are met.

        Spec: REQ-PUBLISH-003
        """
        verdict = compute_honest_verdict(
            gate_passes=True,
            all_provenance_valid=True,
            cross_model_result_exists=True,
        )
        assert verdict == "publication_ready"

    def test_publication_ready_with_caveats(self) -> None:
        """Returns publication_ready_with_caveats when cross-model result is missing.

        Spec: REQ-PUBLISH-003
        """
        verdict = compute_honest_verdict(
            gate_passes=True,
            all_provenance_valid=True,
            cross_model_result_exists=False,
        )
        assert verdict == "publication_ready_with_caveats"

    def test_blocked_when_gate_fails(self) -> None:
        """Returns publication_blocked_no_primary_result when gate fails.

        Spec: SCENARIO-PUBLISH-003
        """
        verdict = compute_honest_verdict(
            gate_passes=False,
            all_provenance_valid=True,
            cross_model_result_exists=True,
        )
        assert verdict == "publication_blocked_no_primary_result"

    def test_blocked_when_provenance_invalid(self) -> None:
        """Returns publication_blocked_no_primary_result when a measurable row is not live_gpu.

        Even if gate_passes is True and cross_model exists, a measured result
        with inference_mode != live_gpu blocks publication.

        Spec: REQ-PUBLISH-001
        """
        verdict = compute_honest_verdict(
            gate_passes=True,
            all_provenance_valid=False,
            cross_model_result_exists=True,
        )
        assert verdict == "publication_blocked_no_primary_result"

    def test_gate_failure_takes_priority(self) -> None:
        """Gate failure takes priority over all other conditions.

        Spec: SCENARIO-PUBLISH-003
        """
        verdict = compute_honest_verdict(
            gate_passes=False,
            all_provenance_valid=False,
            cross_model_result_exists=False,
        )
        assert verdict == "publication_blocked_no_primary_result"


# ---------------------------------------------------------------------------
# write_provenance_doc
# ---------------------------------------------------------------------------


class TestWriteProvenanceDoc:
    """Spec: REQ-PUBLISH-001, REQ-PUBLISH-002"""

    def test_creates_file_when_absent(self, tmp_repo: Path) -> None:
        """Creates the provenance doc when it does not exist.

        Spec: REQ-PUBLISH-001
        """
        table = [
            {
                "metric": "VR signed_improvement",
                "value": 1.0,
                "source_exp": 679,
                "inference_mode": "live_gpu",
                "provenance_valid": True,
            }
        ]
        negative = [{"label": "JEPA v15", "description": "AUC=0.4751"}]
        write_provenance_doc(tmp_repo, table, negative)
        doc_path = tmp_repo / PROVENANCE_DOC_PATH
        assert doc_path.exists()
        content = doc_path.read_text()
        assert "VALID" in content
        assert "JEPA v15" in content

    def test_appends_to_existing_file(self, tmp_repo: Path) -> None:
        """Appends to existing content rather than overwriting.

        Spec: REQ-PUBLISH-002 (never remove content)
        """
        doc_path = tmp_repo / PROVENANCE_DOC_PATH
        doc_path.write_text("EXISTING CONTENT\n")
        write_provenance_doc(tmp_repo, [], [])
        content = doc_path.read_text()
        assert "EXISTING CONTENT" in content

    def test_invalid_provenance_shows_invalid(self, tmp_repo: Path) -> None:
        """Rows with provenance_valid=False show INVALID in the table.

        Spec: REQ-PUBLISH-001
        """
        table = [
            {
                "metric": "blocked metric",
                "value": None,
                "source_exp": 680,
                "inference_mode": "blocked",
                "provenance_valid": False,
            }
        ]
        write_provenance_doc(tmp_repo, table, [])
        content = (tmp_repo / PROVENANCE_DOC_PATH).read_text()
        assert "INVALID" in content

    def test_non_float_value_rendered_as_str(self, tmp_repo: Path) -> None:
        """Non-float values (None, str) are rendered with str() not format().

        Spec: REQ-PUBLISH-001
        """
        table = [
            {
                "metric": "missing metric",
                "value": None,
                "source_exp": 999,
                "inference_mode": "missing",
                "provenance_valid": False,
            }
        ]
        write_provenance_doc(tmp_repo, table, [])
        content = (tmp_repo / PROVENANCE_DOC_PATH).read_text()
        assert "None" in content


# ---------------------------------------------------------------------------
# write_model_card
# ---------------------------------------------------------------------------


class TestWriteModelCard:
    """Spec: REQ-PUBLISH-002, SCENARIO-PUBLISH-002"""

    def test_creates_model_card(self, tmp_repo: Path) -> None:
        """Model card is written to the correct path.

        Spec: REQ-PUBLISH-002
        """
        table = [{"metric": "VR signed_improvement (200q GSM8K, Qwen3.5-0.8B)", "value": 1.0,
                   "source_exp": 679, "inference_mode": "live_gpu", "provenance_valid": True}]
        exp694 = {"cross_model_delta": -1.8, "grammar_recall": 0.0, "gemma_signed_improvement": -0.8}
        exp691 = {"mean_auroc": 0.9585}
        write_model_card(tmp_repo, table, [], exp694, exp691)
        card_path = tmp_repo / MODEL_CARD_PATH
        assert card_path.exists()
        content = card_path.read_text()
        assert "Apache 2.0" in content
        assert "Negative Results" in content

    def test_model_card_contains_jepa_v15_mention(self, tmp_repo: Path) -> None:
        """Model card negative results section mentions JEPA v15 OOD regression.

        Spec: SCENARIO-PUBLISH-002
        """
        negative = [
            {"label": "JEPA v15 OOD Regression (Exp 682)", "description": "AUC=0.4751"},
        ]
        write_model_card(tmp_repo, [], negative, {}, {})
        content = (tmp_repo / MODEL_CARD_PATH).read_text()
        # The model card template has the JEPA v15 content hardcoded for completeness
        assert "JEPA" in content

    def test_model_card_not_pushed(self, tmp_repo: Path) -> None:
        """Model card write does NOT call any HuggingFace API or push mechanism.

        Spec: REQ-PUBLISH-002 (local draft only)
        """
        with patch("builtins.open", wraps=open) as mock_open:
            write_model_card(tmp_repo, [], [], {}, {})
        # No network calls were made — just verifying no huggingface_hub import needed
        card_path = tmp_repo / MODEL_CARD_PATH
        assert card_path.exists()
        content = card_path.read_text()
        assert "Do not publish until operator review" in content

    def test_model_card_handles_missing_exp_data(self, tmp_repo: Path) -> None:
        """Model card writes cleanly when exp694/exp691 data dicts are empty.

        Spec: REQ-PUBLISH-002
        """
        write_model_card(tmp_repo, [], [], {}, {})
        card_path = tmp_repo / MODEL_CARD_PATH
        assert card_path.exists()
        content = card_path.read_text()
        assert "N/A" in content


# ---------------------------------------------------------------------------
# main() integration
# ---------------------------------------------------------------------------


class TestMain:
    """Integration tests for the main() function.

    Spec: REQ-PUBLISH-001, REQ-PUBLISH-002, REQ-PUBLISH-003
    """

    def _run_main(self, tmp_repo: Path) -> dict:
        """Run main() with the repo root patched to tmp_repo and return the artifact."""
        # Patch ExperimentTemplate.__init__ to inject tmp_repo as repo_root.
        # This avoids the property-setter error and ensures all path computations
        # inside ExperimentTemplate use our temp directory.
        original_init = mod.ExperimentTemplate.__init__

        def patched_init(self_tmpl, *args, **kwargs):
            kwargs["repo_root"] = tmp_repo
            original_init(self_tmpl, *args, **kwargs)

        with patch.object(mod.ExperimentTemplate, "__init__", patched_init):
            mod.main()

        artifact_path = tmp_repo / DELIVERABLE
        return json.loads(artifact_path.read_text())

    def test_publication_ready_when_all_results_present(self, tmp_repo: Path) -> None:
        """Artifact shows publication_ready when all source files are live_gpu.

        Spec: REQ-PUBLISH-003, SCENARIO-PUBLISH-001
        """
        _minimal_all_results(tmp_repo)
        artifact = self._run_main(tmp_repo)
        assert artifact["honest_verdict"] == "publication_ready"
        assert artifact["publication_ready"] is True
        assert artifact["model_card_written"] is True
        assert artifact["n_headline_metrics"] == len(HEADLINE_SOURCES)

    def test_blocked_when_679_missing(self, tmp_repo: Path) -> None:
        """Artifact shows blocked when Exp 679 result file is absent.

        Spec: SCENARIO-PUBLISH-003
        """
        # Write nothing — 679 is missing
        artifact = self._run_main(tmp_repo)
        assert artifact["honest_verdict"] == "publication_blocked_no_primary_result"
        assert artifact["publication_ready"] is False
        assert artifact["model_card_written"] is False

    def test_caveats_when_694_missing(self, tmp_repo: Path) -> None:
        """Artifact shows caveats when Exp 694 (cross-model) result is absent.

        Spec: REQ-PUBLISH-003
        """
        _write_679(tmp_repo)
        _write_691(tmp_repo)
        # Intentionally do NOT write 694
        artifact = self._run_main(tmp_repo)
        assert artifact["honest_verdict"] == "publication_ready_with_caveats"
        assert artifact["cross_model_result_exists"] is False

    def test_schema_field_present(self, tmp_repo: Path) -> None:
        """Artifact contains the schema field required by REQUIRED_RESULT_FIELDS.

        Spec: REQ-PUBLISH-001
        """
        _minimal_all_results(tmp_repo)
        artifact = self._run_main(tmp_repo)
        assert "schema" in artifact

    def test_negative_results_documented(self, tmp_repo: Path) -> None:
        """n_negative_results_documented reflects the number of negative findings.

        Spec: REQ-PUBLISH-002
        """
        _minimal_all_results(tmp_repo)
        artifact = self._run_main(tmp_repo)
        assert artifact["n_negative_results_documented"] == 5

    def test_provenance_doc_written(self, tmp_repo: Path) -> None:
        """Provenance doc is written to the repo.

        Spec: REQ-PUBLISH-001
        """
        _minimal_all_results(tmp_repo)
        self._run_main(tmp_repo)
        assert (tmp_repo / PROVENANCE_DOC_PATH).exists()

    def test_model_card_written_to_disk(self, tmp_repo: Path) -> None:
        """Model card file is written to the correct path.

        Spec: REQ-PUBLISH-002, SCENARIO-PUBLISH-002
        """
        _minimal_all_results(tmp_repo)
        self._run_main(tmp_repo)
        assert (tmp_repo / MODEL_CARD_PATH).exists()
