"""Tests for Exp 3815 EDLM operator seed staging package.

Spec refs: REQ-REPORT-3815, SCENARIO-REPORT-3815,
SCENARIO-REPORT-3815-MISSING-PREFLIGHT.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

from carnot.reporting import edlm_operator_seed_staging_3815 as exp3815


ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = ROOT / "openspec/capabilities/research-reporting/spec.md"
SCRIPT_PATH = ROOT / "scripts/experiment_3815_edlm_operator_seed_staging_package.py"
OPERATOR_SEED_COMMAND = (
    "git clone https://github.com/MinkaiXu/Energy-Diffusion-LLM.git && "
    "cd Energy-Diffusion-LLM && git checkout main && echo 'Seed ready'"
)


def _load_script() -> ModuleType:
    for path in (ROOT, ROOT / "python"):
        while str(path) in sys.path:
            sys.path.remove(str(path))
    spec = importlib.util.spec_from_file_location("experiment_3815", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _seed_common_docs(root: Path) -> dict[Path, str]:
    docs = {
        root / "README.md": "# README\noperator curated\n",
        root / "docs/index.html": "<html>landing</html>\n",
        root / "docs/roadmap.md": "# Roadmap\noperator curated\n",
        root / "ops/north-star.md": "# North Star\n",
        root / "ops/status.md": "status before\n",
        root / "ops/changelog.md": "changelog before\n",
        root / "_bmad/traceability.md": "trace before\n",
        root / "scripts/research_conductor.py": "# conductor before\n",
    }
    for path, text in docs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return docs


def _seed_upstream(root: Path, *, include_preflight: bool = True) -> None:
    (root / ".venv/bin").mkdir(parents=True, exist_ok=True)
    (root / "docs/research-notes").mkdir(parents=True, exist_ok=True)
    (root / "docs/research-notes/phase3-alternative-thesis-menu.md").write_text(
        "# Phase-3 Alternative-Thesis Menu\n\n"
        "Thesis B names discrete diffusion as an operator-gated route. "
        "It is different from P0.1 selection and Thesis-A generation, but "
        "still needs an operator seed and a falsifiable kill-gate.\n",
        encoding="utf-8",
    )
    _write_json(
        root / "results/experiment_3781_edlm_next_thesis_feasibility_scoping.json",
        {
            "honest_verdict": (
                "complete: edlm_feasibility_scoped_residual_corrector_not_blocked_"
                "by_either_negative_minimal_kill_gate_designed_operator_decision_"
                "surface_loop_does_not_commit"
            ),
            "minimal_kill_gate_design": "matched-COMPUTE PPL gate on tiny corpus",
            "operator_decision_framing": "seed vs don't seed; no loop commitment",
            "random_seed": 3781,
            "reproducibility_checksum": "1" * 64,
        },
    )
    if include_preflight:
        _write_json(
            root / "results/experiment_3793_edlm_no_train_preflight_readiness.json",
            {
                "honest_verdict": (
                    "complete: edlm_no_train_preflight_go_reference_impl_fetchable_"
                    "true_minimal_kill_gate_sound_operator_seed_command_emitted_"
                    "loop_does_not_commit"
                ),
                "inference_substrate": "aggregation_from_upstream_artifacts",
                "readiness_verdict": "go",
                "minimal_kill_gate_sound": True,
                "operator_seed_command": OPERATOR_SEED_COMMAND,
                "loop_does_not_commit": True,
                "random_seed": 3793,
                "reproducibility_checksum": "2" * 64,
            },
        )


def _clean_verify_report(_path: Path) -> dict[str, object]:
    return {"flags": [], "flag_count": 0, "max_severity": -1}


def test_req_report_3815_spec_anchor_exists() -> None:
    """REQ-REPORT-3815: OpenSpec declares the EDLM staging package contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-3815" in spec
    assert "SCENARIO-REPORT-3815" in spec
    assert "SCENARIO-REPORT-3815-MISSING-PREFLIGHT" in spec
    assert exp3815.OUTPUT_REL_PATH.as_posix() in spec
    assert exp3815.STAGING_NOTE_REL_PATH.as_posix() in spec


def test_scenario_report_3815_packages_seed_note_without_seeding(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3815: complete package is operator-ready but not seeded."""

    before_docs = _seed_common_docs(tmp_path)
    _seed_upstream(tmp_path, include_preflight=True)

    output_path = exp3815.run(
        tmp_path,
        executable=str(tmp_path / ".venv/bin/python"),
        started_s=10.0,
        now_s=10.5,
        verify_runner=_clean_verify_report,
    )

    artifact = json.loads(output_path.read_text(encoding="utf-8"))
    note = (tmp_path / exp3815.STAGING_NOTE_REL_PATH).read_text(encoding="utf-8")

    assert artifact["honest_verdict"] == exp3815.TERMINAL_VERDICT
    assert artifact["inference_substrate"] == exp3815.INFERENCE_SUBSTRATE
    assert artifact["staging_note_written"] is True
    assert artifact["operator_seed_command"] == OPERATOR_SEED_COMMAND
    assert artifact["kill_gate_design_documented"] is True
    assert artifact["loop_does_not_seed"] is True
    assert artifact["edlm_remains_operator_gated"] is True
    assert artifact["operator_curated_doc_unedited"] is True
    assert artifact["random_seed"] == 3815
    assert artifact["duration_s"] == 0.5
    assert len(artifact["reproducibility_checksum"]) == 64
    assert {item["experiment_id"] for item in artifact["cited_upstream_artifacts"]} == {3781, 3793}
    assert all(Path(item["absolute_path"]).is_absolute() for item in artifact["cited_upstream_artifacts"])

    encoded_artifact = json.dumps(artifact, sort_keys=True)
    assert "model_specs" not in artifact
    assert "target_model" not in artifact
    assert "GGUF" not in encoded_artifact
    assert "CUDA" not in encoded_artifact
    assert "live-model" not in encoded_artifact

    assert OPERATOR_SEED_COMMAND in note
    for required_phrase in (
        "OPERATOR-GATED",
        "loop does NOT seed",
        "clones nothing, trains nothing, and runs no model",
        ".venv/bin/python",
        "internal 3090",
        "hard cuda-block",
        "matched-COMPUTE",
        "equal total inference FLOPs",
        "Diverges/NaN",
        "STOP",
        ".350 roadmap",
        "vendor+audit",
        "tiny-EDLM fit smoke",
        "matched-compute harness",
        "kill-gate verdict",
        "P0.1",
        "Thesis-A",
    ):
        assert required_phrase in note

    assert not (tmp_path / "Energy-Diffusion-LLM").exists()
    for path, text in before_docs.items():
        assert path.read_text(encoding="utf-8") == text


def test_scenario_report_3815_missing_preflight_blocks_honestly(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3815-MISSING-PREFLIGHT: absent Exp 3793 blocks honestly."""

    _seed_common_docs(tmp_path)
    _seed_upstream(tmp_path, include_preflight=False)

    output_path = exp3815.run(
        tmp_path,
        executable=str(tmp_path / ".venv/bin/python"),
        started_s=20.0,
        now_s=20.25,
        verify_runner=_clean_verify_report,
    )

    artifact = json.loads(output_path.read_text(encoding="utf-8"))

    assert artifact["honest_verdict"].startswith("blocked_edlm_preflight_missing")
    assert artifact["staging_note_written"] is False
    assert artifact["operator_seed_command"] is None
    assert artifact["kill_gate_design_documented"] is False
    assert artifact["loop_does_not_seed"] is True
    assert artifact["edlm_remains_operator_gated"] is True
    assert not (tmp_path / exp3815.STAGING_NOTE_REL_PATH).exists()


def test_req_report_3815_wrong_interpreter_blocks_before_packaging(tmp_path: Path) -> None:
    """REQ-REPORT-3815: packaging must be pinned to `.venv/bin/python`."""

    _seed_common_docs(tmp_path)
    _seed_upstream(tmp_path, include_preflight=True)

    output_path = exp3815.run(
        tmp_path,
        executable="/usr/bin/python3",
        started_s=30.0,
        now_s=30.125,
        verify_runner=_clean_verify_report,
    )
    artifact = json.loads(output_path.read_text(encoding="utf-8"))

    assert artifact["honest_verdict"].startswith("blocked_edlm_interpreter_not_venv_python")
    assert artifact["staging_note_written"] is False
    assert artifact["operator_seed_command"] is None
    assert artifact["loop_does_not_seed"] is True
    assert not (tmp_path / exp3815.STAGING_NOTE_REL_PATH).exists()


def test_req_report_3815_script_main_delegates_to_runner(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    """REQ-REPORT-3815: requested script entrypoint delegates to the runner."""

    script = _load_script()
    output_path = tmp_path / "artifact.json"
    calls: dict[str, object] = {}

    def fake_run(root: Path, *, executable: str | None, output_path: Path | None):
        calls["root"] = root
        calls["executable"] = executable
        calls["output_path"] = output_path
        output_path = output_path or tmp_path / "unused.json"
        output_path.write_text(
            json.dumps({"honest_verdict": exp3815.TERMINAL_VERDICT}) + "\n",
            encoding="utf-8",
        )
        return output_path

    monkeypatch.setattr(script.exp3815, "run", fake_run)

    assert script.main(["--output", str(output_path), "--executable", "python"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["honest_verdict"] == exp3815.TERMINAL_VERDICT
    assert calls == {
        "root": ROOT,
        "executable": "python",
        "output_path": output_path,
    }
