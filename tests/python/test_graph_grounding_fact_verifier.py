"""Tests for the Exp 3920 graph-grounding facts verifier last retry.

Spec refs: REQ-VERIFY-3920, SCENARIO-VERIFY-3920,
SCENARIO-VERIFY-3920-BLOCKED.
"""

from __future__ import annotations

from collections.abc import Sequence
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import pytest

from carnot.verify import graph_grounding_fact_verifier_defabricated as mod
from scripts.experiments import experiment_3920_facts_graph_grounding_last_retry as exp3920


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "verification" / "spec.md"


def _payload(
    response: dict[str, Any],
    evidence: dict[str, Any],
) -> str:
    return json.dumps({"response": response, "evidence": evidence}, sort_keys=True)


class ScriptedGenerator:
    def __init__(self, outputs: Sequence[str]) -> None:
        self.outputs = list(outputs)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        assert "Return ONLY compact JSON" in prompt
        assert "Response:" in prompt
        assert "Evidence:" in prompt
        self.calls.append({"prompt": prompt, "kwargs": kwargs})
        text = self.outputs.pop(0)
        return {"choices": [{"text": text}], "usage": {"completion_tokens": 7}}


def test_req_verify_3920_spec_anchor_exists() -> None:
    """REQ-VERIFY-3920: last-retry graph grounding is OpenSpec anchored."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-VERIFY-3920" in spec
    assert "SCENARIO-VERIFY-3920" in spec
    assert "graph_ground_score(item, generator" in spec
    assert "results/experiment_3920_facts_graph_grounding_last_retry.json" in spec
    assert "tests/python/test_graph_grounding_fact_verifier.py" in spec


def test_req_verify_3920_graph_ground_score_invokes_robust_generator() -> None:
    """REQ-VERIFY-3920: graph_ground_score uses the supplied generator."""

    generator = ScriptedGenerator(
        [
            _payload(
                {
                    "entities": ["Marie Curie", "radium"],
                    "relations": [
                        {
                            "subject": "Marie Curie",
                            "relation": "discovered",
                            "object": "radium",
                        }
                    ],
                },
                {
                    "entities": ["Marie Curie", "polonium", "Pierre Curie", "radium"],
                    "relations": [
                        {
                            "subject": "Marie Curie",
                            "relation": "discovered",
                            "object": "polonium",
                        },
                        {"subject": "Pierre Curie", "relation": "studied", "object": "radium"},
                    ],
                },
            )
        ]
    )

    score = mod.graph_ground_score(
        {
            "claim": "Marie Curie discovered radium.",
            "source": "Marie Curie discovered polonium. Pierre Curie studied radium.",
        },
        generator,
        max_tokens=48,
    )

    assert len(generator.calls) == 1
    assert score["model_invoked"] is True
    assert score["completion_tokens"] > 0
    assert score["eg"] == pytest.approx(1.0)
    assert score["rp"] == pytest.approx(0.0)
    assert 0.0 <= score["cfi"] < 1.0
    assert score["hallucination_score"] > 0.0
    assert score["unsupported_relations"][0]["relation"] == "discovered"


def test_scenario_verify_3920_nonseparable_fixture_rejects_auroc_one() -> None:
    """SCENARIO-VERIFY-3920: scripted fixture is intentionally non-separable."""

    fixture = mod.build_nonseparable_graph_grounding_fixture()
    outputs = [
        _payload(item["scripted_response_graph"], item["scripted_evidence_graph"])
        for item in fixture
    ]
    generator = ScriptedGenerator(outputs)

    result = mod.score_nonseparable_graph_grounding_fixture(
        fixture,
        generator=generator,
        max_tokens=48,
    )

    assert len(generator.calls) == len(fixture)
    assert result["model_invoked"] is True
    assert result["model_call_count"] == len(fixture)
    assert result["fixture_n_items"] == 12
    assert result["fixture_n_hallucinated"] == 6
    assert 0.6 <= result["fixture_auroc"] <= 0.95
    assert result["fixture_auroc"] != pytest.approx(1.0)
    assert result["stub_rejected"] is True
    assert all(item["completion_tokens"] > 0 for item in result["per_item_scores"])
    assert result["fixture_token_count"] == 7 * len(fixture)


def test_req_verify_3920_artifact_schema_uses_bare_ready_fields(tmp_path: Path) -> None:
    """REQ-VERIFY-3920: READY artifact is gated by bare scalar evidence."""

    artifact = exp3920.build_artifact(
        fixture_result={
            "fixture_auroc": 0.916667,
            "model_invoked": True,
            "model_call_count": 12,
            "fixture_token_count": 84,
            "per_item_scores": [{"id": "fixture", "completion_tokens": 7}],
        },
        corpus_result={
            "n_items": 60,
            "model_invoked": True,
            "corpus_run_token_count": 420,
            "per_item_scores_path": "results/experiment_3920_scores.jsonl",
            "per_item_scores_sha256": "a" * 64,
        },
        config=exp3920.ExperimentConfig(
            repo_root=tmp_path,
            started_at=10.0,
            clock=lambda: 75.0,
        ),
        preconditions_checked=[exp3920.PreconditionCheck("cuda_available", True, "ok")],
        model_specs={"model_used": "fixture", "gguf_path": "/tmp/scripted.gguf"},
        unit_test_passed=True,
    )

    exp3920.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith(
        "complete: facts_graph_verifier_READY_fixture_auroc0.9167_model_invoked_tokens420"
    )
    assert artifact["verifier_module_path"] == (
        "python/carnot/verify/graph_grounding_fact_verifier_defabricated.py"
    )
    assert artifact["gguf_harness_model_used"] == "fixture"
    assert artifact["fixture_auroc"] == pytest.approx(0.916667)
    assert artifact["model_invoked"] is True
    assert artifact["corpus_run_token_count"] == 420
    assert artifact["unit_test_path"] == "tests/python/test_graph_grounding_fact_verifier.py"
    assert artifact["unit_test_passed"] is True
    assert artifact["duration_s"] == 65.0


def test_req_verify_3920_default_context_matches_robust_harness_floor() -> None:
    """REQ-VERIFY-3920: corpus scoring uses enough context for clipped facts rows."""

    assert exp3920.ExperimentConfig().n_ctx >= 1024


def test_scenario_verify_3920_not_ready_for_separable_fixture(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3920-BLOCKED: AUROC 1.0 retires the facts route."""

    artifact = exp3920.build_artifact(
        fixture_result={
            "fixture_auroc": 1.0,
            "model_invoked": True,
            "model_call_count": 12,
            "fixture_token_count": 84,
            "per_item_scores": [],
        },
        corpus_result={
            "n_items": 60,
            "model_invoked": True,
            "corpus_run_token_count": 420,
            "per_item_scores_path": "results/experiment_3920_scores.jsonl",
            "per_item_scores_sha256": "b" * 64,
        },
        config=exp3920.ExperimentConfig(
            repo_root=tmp_path,
            started_at=10.0,
            clock=lambda: 75.0,
        ),
        preconditions_checked=[exp3920.PreconditionCheck("cuda_available", True, "ok")],
        model_specs={"model_used": "fixture", "gguf_path": "/tmp/scripted.gguf"},
        unit_test_passed=True,
    )

    exp3920.validate_artifact(artifact)
    assert artifact["fixture_auroc"] == 1.0
    assert artifact["honest_verdict"].startswith(
        "complete: facts_graph_verifier_NOT_READY_fixture_auroc1.0000"
    )


def test_scenario_verify_3920_blocked_artifact_is_terminal(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3920-BLOCKED: missing resources do not fabricate metrics."""

    artifact = exp3920.build_blocked_artifact(
        reason="blocked_upstream_gguf_harness_not_ready",
        preconditions_checked=[
            exp3920.PreconditionCheck("exp3915_robust_harness_ready", False, "missing")
        ],
        duration_s=0.25,
        model_specs={},
    )
    output = tmp_path / exp3920.OUTPUT_REL_PATH
    exp3920.write_artifact(output, artifact)
    persisted = json.loads(output.read_text(encoding="utf-8"))

    exp3920.validate_artifact(persisted)
    assert persisted["honest_verdict"] == "blocked_upstream_gguf_harness_not_ready"
    assert persisted["fixture_auroc"] is None
    assert persisted["model_invoked"] is False
    assert persisted["corpus_run_token_count"] == 0
    assert persisted["unit_test_passed"] is False
    assert persisted["inference_substrate"] == "none_blocked_preflight"


def test_scenario_verify_3920_live_fixture_or_artifact_positive_control() -> None:
    """SCENARIO-VERIFY-3920: live evidence has tokens and non-separable AUROC."""

    artifact_env = os.environ.get("CARNOT_3920_ARTIFACT_UNDER_TEST")
    if artifact_env:
        artifact = json.loads(Path(artifact_env).read_text(encoding="utf-8"))
        exp3920.validate_artifact(artifact)
        assert artifact["model_invoked"] is True
        assert artifact["corpus_run_token_count"] > 0
        assert 0.6 <= artifact["fixture_auroc"] <= 0.95
        assert artifact["fixture_auroc"] != pytest.approx(1.0)
        return

    code = """
import json
from carnot.verify import graph_grounding_fact_verifier_defabricated as mod

generator, meta = mod.load_robust_graph_grounding_generator(n_ctx=512)
result = mod.score_nonseparable_graph_grounding_fixture(
    mod.build_nonseparable_graph_grounding_fixture(),
    generator=generator,
    max_tokens=64,
)
print(json.dumps({"meta": meta, "result": result}, sort_keys=True))
"""
    child_env = os.environ.copy()
    for key in list(child_env):
        if key.startswith(("PYTEST_", "COV_CORE")):
            child_env.pop(key, None)
    child_env["CUDA_VISIBLE_DEVICES"] = ""
    proc = subprocess.run(
        [str(REPO_ROOT / ".venv" / "bin" / "python"), "-c", code],
        capture_output=True,
        check=False,
        cwd=REPO_ROOT,
        env=child_env,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    payload = json.loads(proc.stdout)
    meta = payload["meta"]
    result = payload["result"]

    assert int(meta["smoke_tokens"]) > 0
    assert result["model_invoked"] is True
    assert result["model_call_count"] >= result["fixture_n_items"]
    assert result["fixture_token_count"] > 0
    assert all(item["completion_tokens"] > 0 for item in result["per_item_scores"])
    assert 0.6 <= result["fixture_auroc"] <= 0.95
    assert result["fixture_auroc"] != pytest.approx(1.0)
