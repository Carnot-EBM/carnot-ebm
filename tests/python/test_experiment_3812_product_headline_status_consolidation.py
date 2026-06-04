"""Tests for Exp 3812 product-headline status consolidation.

Spec refs: REQ-PUBLISH-3812, SCENARIO-PUBLISH-3812.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts import experiment_3812_product_headline_status_consolidation as exp3812


ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = Path("openspec/capabilities/publication/spec.md")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _gate_data() -> dict[str, object]:
    return {
        "paper_ready": True,
        "gates": {
            "G1": {"pass": True},
            "G2": {"pass": True},
            "G3": {"pass": True},
            "G4": {"pass": True},
        },
        "unmet_gates": [],
    }


def _reports() -> dict[int, dict[str, object]]:
    return {
        3798: {
            "loaded": True,
            "max_severity": 2,
            "flags": [
                {
                    "kind": "TAUTOLOGY",
                    "severity": "critical",
                    "detail": "baseline_pass1 and repair_pass1 match",
                }
            ],
        },
        3799: {"loaded": True, "max_severity": -1, "flags": []},
        2090: {
            "loaded": True,
            "max_severity": 2,
            "flags": [
                {
                    "kind": "DURATION_TOO_SHORT",
                    "severity": "critical",
                    "detail": "duration_s=0.009 with compute-bound evidence",
                }
            ],
        },
    }


def _seed_repo(root: Path) -> dict[str, str]:
    _write_json(
        root / "results" / "experiment_3798_g4_product_headline_restoration.json",
        {
            "honest_verdict": (
                "complete: g4_product_headline_restoration_baseline_0.13_repair_"
                "0.13_delta_0.0pp_g4_provenance_complete_headline_stays_demoted"
            ),
            "baseline_pass1": 0.13333333333333333,
            "repair_pass1": 0.13333333333333333,
            "repair_delta_pp": 0.0,
            "n": 30,
            "random_seed": 42,
            "reproducibility_checksum": "a854c6f82908fec3",
            "inference_substrate": "live_llm_inference",
            "flagged_adversarial": True,
            "duration_s": 167.532,
        },
    )
    _write_json(
        root / "results" / "experiment_3799_product_headline_provenance_reconfirmation.json",
        {
            "honest_verdict": (
                "complete: product_headline_provenance_reconfirmed_rerun_g4_true_"
                "exp2090_g4_true_headline_not_yet_eligible_operator_curated_doc_unedited"
            ),
            "product_headline_restorable": "not_yet_headline_eligible",
            "rerun_code_repair_g4_pass": True,
            "exp2090_g4_pass": True,
            "random_seed": {"exp3798": 42, "exp2090": 42},
            "reproducibility_checksum": {
                "exp3798": "a854c6f82908fec3",
                "exp2090": "bfb0acdb53773a49",
            },
            "inference_substrate": "aggregation_from_upstream_artifacts",
            "duration_s": 0.000368,
        },
    )
    _write_json(
        root / "results" / "experiment_2090_crane_humaneval.json",
        {
            "honest_verdict": "CRANE evaluated vs rigid grammar on 50 HumanEval problems.",
            "rigid_pass_rate": 0.7,
            "crane_pass_rate": 0.85,
            "pass_rate_delta": 0.15000000000000002,
            "random_seed": 42,
            "reproducibility_checksum": "bfb0acdb53773a49",
            "duration_s": 0.009,
        },
    )
    paths = {
        "technical": "docs/technical-report.md",
        "north": "ops/north-star.md",
        "status": "ops/status.md",
        "changelog": "ops/changelog.md",
        "trace": "_bmad/traceability.md",
        "conductor": "scripts/research_conductor.py",
    }
    before: dict[str, str] = {}
    for label, rel_path in paths.items():
        path = root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        text = f"{label} before\n"
        path.write_text(text, encoding="utf-8")
        before[rel_path] = text
    return before


def test_req_publish_3812_spec_anchor_exists() -> None:
    """REQ-PUBLISH-3812: OpenSpec declares the consolidation contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PUBLISH-3812" in spec
    assert "SCENARIO-PUBLISH-3812" in spec
    assert exp3812.OUTPUT_REL_PATH.as_posix() in spec


def test_scenario_publish_3812_records_status_and_doc_proposal(tmp_path: Path) -> None:
    """SCENARIO-PUBLISH-3812: stale product positives stay demoted."""

    before = _seed_repo(tmp_path)

    out_path = exp3812.run(
        tmp_path,
        adversarial_reports=_reports(),
        publication_gate_data=_gate_data(),
        started_s=10.0,
        now_s=10.25,
    )
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    rows = {row["number"]: row for row in artifact["product_headline_status_table"]}

    exp3812.validate_artifact(artifact)
    assert artifact["honest_verdict"] == exp3812.TERMINAL_VERDICT
    assert set(exp3812.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(exp3812.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["code_repair_supports_headline"] is False
    assert artifact["crane_supports_headline"] is False
    assert artifact["product_headline_recommendation"] == "stays_demoted"
    assert artifact["sole_defensible_headline"] == (
        "FoVer methods headline 0.9131 (G1-G4 pass via publication_gate.py)"
    )
    assert artifact["doc_proposal_emitted_not_curated_edit"] is True
    assert artifact["operator_curated_doc_unedited"] is True
    assert artifact["random_seed"] == 3812
    assert artifact["duration_s"] == 0.25
    assert artifact["reproducibility_checksum"] == exp3812.payload_checksum(artifact)

    code_repair = rows["exp3798_code_repair_rerun_delta_0.0pp"]
    assert code_repair["source_artifact"].endswith(
        "results/experiment_3798_g4_product_headline_restoration.json"
    )
    assert Path(code_repair["source_artifact"]).is_absolute()
    assert code_repair["n"] == 30
    assert code_repair["seed_present"] is True
    assert code_repair["checksum_present"] is True
    assert code_repair["live_adversarial_recheck"] == "CRITICAL"
    assert code_repair["g4_pass"] is False
    assert "delta=0.0pp" in code_repair["why"]

    crane = rows["exp2090_crane_plus15pp"]
    assert crane["source_artifact"].endswith("results/experiment_2090_crane_humaneval.json")
    assert crane["n"] == 50
    assert crane["seed_present"] is True
    assert crane["checksum_present"] is True
    assert crane["substrate"] is None
    assert crane["live_adversarial_recheck"] == "CRITICAL"
    assert crane["g4_pass"] is False
    assert "stale" in crane["why"]

    statuses = {entry["experiment_id"]: entry for entry in artifact["artifact_provenance_status"]}
    assert statuses[3799]["live_adversarial_recheck"] == "clean"
    assert statuses[3799]["stale_exp2090_g4_stamp"] is True
    assert {entry["experiment_id"] for entry in artifact["cited_upstream_artifacts"]} == {
        3798,
        3799,
        2090,
    }

    proposal_path = tmp_path / exp3812.DOC_PROPOSAL_REL_PATH
    proposal = proposal_path.read_text(encoding="utf-8")
    assert "Product Headline Status Doc Proposal" in proposal
    assert "FoVer methods headline 0.9131" in proposal
    assert "operator-curated" in proposal

    for rel_path, original in before.items():
        assert (tmp_path / rel_path).read_text(encoding="utf-8") == original

    encoded = json.dumps(artifact, sort_keys=True)
    assert "GGUF" not in encoded
    assert "CUDA" not in encoded
    assert "model_specs" not in artifact
    assert "target_model" not in artifact


def test_actual_artifacts_live_recheck_demotes_crane() -> None:
    """REQ-PUBLISH-3812: actual Exp 2090 is classified from live re-check."""

    artifact = exp3812.build_artifact(
        ROOT,
        started_s=20.0,
        now_s=20.5,
    )
    rows = {row["number"]: row for row in artifact["product_headline_status_table"]}

    assert rows["exp3798_code_repair_rerun_delta_0.0pp"]["live_adversarial_recheck"] == (
        "CRITICAL"
    )
    assert rows["exp2090_crane_plus15pp"]["live_adversarial_recheck"] == "CRITICAL"
    assert artifact["code_repair_supports_headline"] is False
    assert artifact["crane_supports_headline"] is False


def test_fallback_helpers_cover_substrate_and_precondition_failures(monkeypatch) -> None:
    """REQ-PUBLISH-3812: helper fallbacks stay explicit and honest."""

    assert exp3812._source_substrate({"inference_mode": "archival_replay"}) == (
        "archival_replay"
    )
    assert exp3812._source_substrate({"metadata": {"inference_mode": "metadata_replay"}}) == (
        "metadata_replay"
    )

    original_import = exp3812.importlib.import_module

    def missing_summary(name: str):
        if name == "scripts.summarize_artifact":
            raise ImportError("missing summary")
        return original_import(name)

    monkeypatch.setattr(exp3812.importlib, "import_module", missing_summary)
    assert exp3812._interpreter_preconditions()["summarize_artifact_importable"] is False

    def missing_adversarial(name: str):
        if name == "scripts.adversarial_verify":
            raise ImportError("missing adversarial")
        return original_import(name)

    monkeypatch.setattr(exp3812.importlib, "import_module", missing_adversarial)
    assert exp3812._interpreter_preconditions()["adversarial_verify_importable"] is False


def test_missing_upstreams_are_recorded_without_fabrication(tmp_path: Path) -> None:
    """REQ-PUBLISH-3812: missing artifacts produce honest non-support rows."""

    artifact = exp3812.build_artifact(
        tmp_path,
        adversarial_reports={},
        publication_gate_data={"paper_ready": False, "gates": {}, "unmet_gates": ["G1"]},
        started_s=30.0,
        now_s=30.1,
    )
    rows = {row["number"]: row for row in artifact["product_headline_status_table"]}

    assert rows["exp3798_code_repair_rerun_delta_0.0pp"]["source_artifact"].endswith(
        "results/experiment_3798_g4_product_headline_restoration.json"
    )
    assert rows["exp3798_code_repair_rerun_delta_0.0pp"]["n"] is None
    assert rows["exp3798_code_repair_rerun_delta_0.0pp"]["g4_pass"] is False
    assert "artifact_missing" in rows["exp3798_code_repair_rerun_delta_0.0pp"]["why"]
    assert rows["exp2090_crane_plus15pp"]["g4_pass"] is False
    assert artifact["publication_gate_state"]["paper_ready"] is False
    assert artifact["doc_proposal_emitted_not_curated_edit"] is True
