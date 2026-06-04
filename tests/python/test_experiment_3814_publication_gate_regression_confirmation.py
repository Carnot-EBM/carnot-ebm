"""Tests for Exp 3814 publication-gate regression confirmation.

Spec refs: REQ-PUBLISH-3814, SCENARIO-PUBLISH-3814.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

from scripts import experiment_3814_publication_gate_regression_confirmation as exp3814


SPEC_PATH = Path("openspec/capabilities/publication/spec.md")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _gate_data(*, paper_ready: bool = True) -> dict[str, object]:
    gates = {
        "G1": {"pass": True, "detail": "headline measured", "source": "experiment_2850.json"},
        "G2": {"pass": True, "detail": "external CI reproducer"},
        "G3": {"pass": True, "detail": "narrowing clean", "hits": []},
        "G4": {"pass": True, "detail": "seed/checksum present", "source": "experiment_2850.json"},
    }
    if not paper_ready:
        gates["G3"]["pass"] = False
    return {
        "paper_ready": paper_ready,
        "gates": gates,
        "unmet_gates": [] if paper_ready else ["G3"],
        "note": "Stable 4-condition gate",
    }


def _seed_repo(root: Path, *, include_headline: bool = True) -> dict[Path, str]:
    gate_script = root / "scripts" / "publication_gate.py"
    gate_script.parent.mkdir(parents=True, exist_ok=True)
    gate_payload = json.dumps(_gate_data(), sort_keys=True)
    gate_script.write_text(
        "import json\n"
        f"GATE = json.loads({gate_payload!r})\n"
        "def evaluate():\n"
        "    return GATE\n"
        "if __name__ == '__main__':\n"
        "    print(json.dumps(evaluate()))\n",
        encoding="utf-8",
    )
    if include_headline:
        _write_json(
            root / "results" / "experiment_2850_fover_dual_condition_integrity_v4.json",
            {
                "condition_a_production_auroc_mean": 0.9131335999999999,
                "n_seeds": 5,
                "random_seed": 42,
                "random_seeds_used": [42, 137, 271, 314, 1729],
                "reproducibility_checksum": "b3d8a0ea0ed6180e0120eb909513436d2e0a43055",
                "adversarial_verify_passed": True,
                "live_model_invoked": False,
            },
        )
    _write_json(
        root / "ops" / "publication_gate_state.json",
        {"g2_independent_reproducer": True, "g2_evidence": "CI run"},
    )
    files = {
        root / "ops" / "north-star.md": "fixed gate definitions before\n",
        root / "ops" / "reproduction-runbook-fover-headline.md": "G2 runbook before\n",
        root / "docs" / "technical-report.md": "technical before\n",
        root / "docs" / "arxiv-paper" / "main.tex": "paper before\n",
        root / "scripts" / "research_conductor.py": "conductor before\n",
    }
    for path, text in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return files


def test_req_publish_3814_spec_anchor_exists() -> None:
    """REQ-PUBLISH-3814: OpenSpec declares the regression-confirmation contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PUBLISH-3814" in spec
    assert "SCENARIO-PUBLISH-3814" in spec
    assert exp3814.OUTPUT_REL_PATH.as_posix() in spec
    assert exp3814.TERMINAL_VERDICT in spec


def test_scenario_publish_3814_records_gate_booleans_and_provenance(tmp_path: Path) -> None:
    """SCENARIO-PUBLISH-3814: G1-G4 pass and the frozen FoVer AUROC is unchanged."""

    untouched = _seed_repo(tmp_path)

    out_path = exp3814.run(
        tmp_path,
        started_s=10.0,
        now_s=11.25,
        executable=sys.executable,
    )
    artifact = json.loads(out_path.read_text(encoding="utf-8"))

    exp3814.validate_artifact(artifact)
    assert artifact["honest_verdict"] == exp3814.TERMINAL_VERDICT
    assert set(exp3814.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(exp3814.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["inference_substrate"] == exp3814.INFERENCE_SUBSTRATE
    assert artifact["g1_pass"] is True
    assert artifact["g2_pass"] is True
    assert artifact["g3_pass"] is True
    assert artifact["g4_pass"] is True
    assert artifact["paper_ready"] is True
    assert artifact["frozen_fover_auroc"] == 0.9131
    assert artifact["frozen_fover_auroc_unchanged"] is True
    assert artifact["any_gate_regressed"] is False
    assert artifact["gate_definitions_unchanged"] is True
    assert artifact["publication_gate_json"] == _gate_data()
    assert artifact["random_seed"] == 3814
    assert artifact["duration_s"] == 1.25
    assert artifact["reproducibility_checksum"] == exp3814.payload_checksum(artifact)

    cited_paths = {Path(entry["path"]).name for entry in artifact["cited_upstream_artifacts"]}
    assert "publication_gate.py" in cited_paths
    assert "experiment_2850_fover_dual_condition_integrity_v4.json" in cited_paths
    assert "publication_gate_state.json" in cited_paths
    assert "north-star.md" in cited_paths

    encoded = json.dumps(artifact, sort_keys=True)
    assert "GGUF" not in encoded
    assert "CUDA" not in encoded
    assert "live_llm_inference" not in encoded

    for path, original in untouched.items():
        assert path.read_text(encoding="utf-8") == original


def test_req_publish_3814_missing_headline_source_blocks_without_fabrication(
    tmp_path: Path,
) -> None:
    """REQ-PUBLISH-3814: missing FoVer source writes blocked_ without a fake pass."""

    _seed_repo(tmp_path, include_headline=False)

    artifact = exp3814.build_artifact(
        tmp_path,
        publication_gate_data=_gate_data(),
        started_s=20.0,
        now_s=20.5,
        executable=sys.executable,
    )

    exp3814.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_frozen_fover_headline_source_missing"
    assert artifact["g1_pass"] is True
    assert artifact["g2_pass"] is True
    assert artifact["g3_pass"] is True
    assert artifact["g4_pass"] is True
    assert artifact["paper_ready"] is True
    assert artifact["frozen_fover_auroc"] is None
    assert artifact["frozen_fover_auroc_unchanged"] is False
    assert artifact["any_gate_regressed"] is False
    assert artifact["gate_definitions_unchanged"] is True
    assert artifact["reproducibility_checksum"] == exp3814.payload_checksum(artifact)


def test_req_publish_3814_blocked_and_regression_branches_are_explicit(
    tmp_path: Path,
) -> None:
    """REQ-PUBLISH-3814: blocked and regressed outcomes are closed-set honest records."""

    _seed_repo(tmp_path)
    gate_script = tmp_path / "scripts" / "publication_gate.py"

    gate_script.write_text("import sys\nsys.exit(3)\n", encoding="utf-8")
    failed = exp3814._run_publication_gate_json(tmp_path, sys.executable)
    assert failed["ok"] is False
    assert failed["returncode"] == 3

    gate_script.write_text("print('not json')\n", encoding="utf-8")
    invalid = exp3814._run_publication_gate_json(tmp_path, sys.executable)
    assert invalid["ok"] is False
    assert invalid["data"] == {}

    assert exp3814._gate_bool({}, "G1") is False
    assert exp3814._verdict(
        preconditions_ok=False,
        gate_ok=True,
        frozen_fover_auroc_unchanged=True,
        any_gate_regressed=False,
        gate_definitions_unchanged=True,
    ) == exp3814.BLOCKED_INTERPRETER_VERDICT
    assert exp3814._verdict(
        preconditions_ok=True,
        gate_ok=False,
        frozen_fover_auroc_unchanged=True,
        any_gate_regressed=False,
        gate_definitions_unchanged=True,
    ) == exp3814.BLOCKED_GATE_VERDICT
    assert exp3814._verdict(
        preconditions_ok=True,
        gate_ok=True,
        frozen_fover_auroc_unchanged=True,
        any_gate_regressed=True,
        gate_definitions_unchanged=True,
    ) == "complete: publication_gate_regression_detected_operator_review_required"
    assert exp3814._verdict(
        preconditions_ok=True,
        gate_ok=True,
        frozen_fover_auroc_unchanged=True,
        any_gate_regressed=False,
        gate_definitions_unchanged=False,
    ) == "complete: publication_gate_definition_changed_operator_review_required"
