"""Tests for the Exp 3905 cost-instrumented verification harness.

Spec refs: REQ-VERIFY-3905, SCENARIO-VERIFY-3905,
SCENARIO-VERIFY-3905-BLOCKED.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from carnot.verify import cost_instrumented_verification as civ
from scripts.experiments import experiment_3905_cost_instrumented_verify_harness as exp3905


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "verification" / "spec.md"
RESULT_PATH = REPO_ROOT / "results" / "experiment_3905_cost_instrumented_verify_harness.json"


class ScriptedLlama:
    """Tiny llama.cpp stand-in for deterministic token-accounting tests."""

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.prompts: list[str] = []

    def tokenize(self, payload: bytes, add_bos: bool = True, **_kwargs: object) -> list[int]:
        tokens = payload.decode("utf-8", errors="ignore").replace("\n", " ").split()
        return [1, *range(2, len(tokens) + 2)] if add_bos else list(range(len(tokens)))

    def __call__(self, prompt: str, **kwargs: object) -> dict[str, object]:
        self.prompts.append(prompt)
        incorrect_markers = ("= 65", "= 124", "area 40", "then all wugs are blickets")
        is_incorrect = any(marker in prompt for marker in incorrect_markers)
        response = {
            "verdict": "incorrect" if is_incorrect else "correct",
            "error_confidence": 0.91 if is_incorrect else 0.09,
        }
        assert kwargs["temperature"] == 0.0
        return {"choices": [{"text": json.dumps(response)}]}


def _fixture_artifact_path() -> Path | None:
    configured = os.environ.get("CARNOT_EXP3905_LIVE_ARTIFACT")
    if configured and Path(configured).is_file():
        return Path(configured)
    return RESULT_PATH if RESULT_PATH.is_file() else None


def test_req_verify_3905_spec_anchor_exists() -> None:
    """REQ-VERIFY-3905: the cost harness is OpenSpec anchored."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-VERIFY-3905" in spec
    assert "SCENARIO-VERIFY-3905" in spec
    assert "python/carnot/verify/cost_instrumented_verification.py" in spec
    assert "results/experiment_3905_cost_instrumented_verify_harness.json" in spec


def test_req_verify_3905_measurement_uses_monotonic_clock_and_auroc() -> None:
    """REQ-VERIFY-3905: measure_verification_cost emits timed bare metrics."""

    ticks = iter([10.0, 10.25])
    items = (
        {"step": "2 + 2 = 4.", "gold_error": 0},
        {"step": "2 + 2 = 5.", "gold_error": 1},
    )

    def verifier_fn(rows: tuple[dict[str, object], ...]) -> dict[str, object]:
        assert rows == items
        return {"scores": [0.1, 0.9], "est_tokens": 7, "est_flops": 13}

    measured = civ.measure_verification_cost(verifier_fn, items, "scripted", clock=lambda: next(ticks))

    assert measured == {
        "auroc": 1.0,
        "total_wall_s": 0.25,
        "per_item_wall_ms": 125.0,
        "est_tokens": 7,
        "est_flops": 13,
        "n_items": 2,
    }


def test_req_verify_3905_fixture_wrappers_with_scripted_llama() -> None:
    """REQ-VERIFY-3905: both fixture wrappers expose score and cost evidence."""

    fixture = civ.build_cost_fixture()
    energy = civ.measure_verification_cost(civ.run_energy_verifier, fixture, "energy")
    llm = civ.measure_verification_cost(
        lambda rows: civ.run_llm_judge_verifier(
            rows,
            model_path="/tmp/scripted.gguf",
            llama_factory=ScriptedLlama,
            model_params=11,
            random_seed=3905,
        ),
        fixture,
        "llm_judge",
    )

    assert len(fixture) == 10
    assert energy["n_items"] == llm["n_items"] == 10
    assert energy["total_wall_s"] > 0
    assert llm["total_wall_s"] > 0
    assert energy["per_item_wall_ms"] > 0
    assert llm["per_item_wall_ms"] > 0
    assert energy["est_flops"] > 0
    assert llm["est_tokens"] > 0
    assert llm["est_flops"] == 2 * 11 * llm["est_tokens"]


def test_req_verify_3905_artifact_builder_uses_bare_fields(tmp_path: Path) -> None:
    """REQ-VERIFY-3905: Exp 3905 artifacts use bare scalar gate fields."""

    energy_cost = {
        "auroc": 0.5,
        "total_wall_s": 0.1,
        "per_item_wall_ms": 1.0,
        "est_tokens": 20,
        "est_flops": 200,
        "n_items": 10,
    }
    llm_cost = {
        "auroc": 0.7,
        "total_wall_s": 65.0,
        "per_item_wall_ms": 6500.0,
        "est_tokens": 1200,
        "est_flops": 72_000,
        "n_items": 10,
    }
    artifact = exp3905.build_artifact(
        config=exp3905.ExperimentConfig(
            repo_root=tmp_path,
            started_monotonic_s=100.0,
            clock=lambda: 166.0,
            run_unit_test=False,
        ),
        preconditions_checked=[exp3905.PreconditionCheck("cuda_available", True, "ok")],
        model_specs={"hf_id": "fixture", "model_path": "fixture.gguf"},
        energy_cost=energy_cost,
        llm_cost=llm_cost,
        unit_test_passed=True,
    )

    exp3905.validate_artifact(artifact)
    assert artifact["harness_module_path"] == "python/carnot/verify/cost_instrumented_verification.py"
    assert artifact["fixture_cost_ratio"] == 6500.0
    assert artifact["fixture_energy_per_item_ms"] == 1.0
    assert artifact["fixture_llm_per_item_ms"] == 6500.0
    assert artifact["unit_test_passed"] is True
    assert artifact["honest_verdict"].startswith("complete: cost_harness_READY_ratio")


def test_scenario_verify_3905_blocked_artifact_has_no_cost_claims() -> None:
    """SCENARIO-VERIFY-3905-BLOCKED: blocked runs do not fabricate costs."""

    artifact = exp3905.build_blocked_artifact(
        reason="blocked_model_not_cached",
        preconditions_checked=[exp3905.PreconditionCheck("sota_gguf_cached", False, "missing")],
        duration_s=0.5,
        model_specs={"hf_id": "fixture", "model_path": None},
    )

    exp3905.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_model_not_cached"
    assert artifact["fixture_cost_ratio"] is None
    assert artifact["fixture_energy_per_item_ms"] is None
    assert artifact["fixture_llm_per_item_ms"] is None
    assert artifact["unit_test_passed"] is False


def test_scenario_verify_3905_live_fixture_cost_ratio() -> None:
    """SCENARIO-VERIFY-3905: live fixture proves LLM judge cost exceeds energy cost."""

    artifact_path = _fixture_artifact_path()
    if artifact_path is not None:
        artifact: dict[str, Any] = json.loads(artifact_path.read_text(encoding="utf-8"))
    else:
        output_path = REPO_ROOT / ".tmp-pytest" / "experiment_3905_live_fixture.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        child_env = os.environ.copy()
        for key in list(child_env):
            if key.startswith(("PYTEST_", "COV_CORE")):
                child_env.pop(key, None)
        child_env["CARNOT_EXP3905_LIVE_ARTIFACT"] = str(output_path)
        proc = subprocess.run(
            [
                str(REPO_ROOT / ".venv" / "bin" / "python"),
                "scripts/experiments/experiment_3905_cost_instrumented_verify_harness.py",
                "--output-path",
                str(output_path),
                "--no-unit-test",
            ],
            cwd=REPO_ROOT,
            env=child_env,
            capture_output=True,
            text=True,
            timeout=1200,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr or proc.stdout
        artifact = json.loads(output_path.read_text(encoding="utf-8"))

    exp3905.validate_artifact(artifact)
    assert artifact["energy_cost"]["n_items"] == 10
    assert artifact["llm_cost"]["n_items"] == 10
    assert artifact["fixture_energy_per_item_ms"] > 0
    assert artifact["fixture_llm_per_item_ms"] > 0
    assert artifact["fixture_llm_per_item_ms"] != artifact["fixture_energy_per_item_ms"]
    assert artifact["fixture_cost_ratio"] > 1
