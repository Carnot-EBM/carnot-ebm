"""Tests for the Exp 3896 graph-grounding verifier harness.

Spec refs: REQ-VERIFY-3896, SCENARIO-VERIFY-3896,
SCENARIO-VERIFY-3896-BLOCKED.
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
from scripts.experiments import experiment_3896_graph_grounding_verifier_harness as exp3896


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "verification" / "spec.md"


def _payload(
    response: dict[str, Any],
    evidence: dict[str, Any],
) -> str:
    return json.dumps({"response": response, "evidence": evidence}, sort_keys=True)


class ScriptedLlama:
    def __init__(self, outputs: Sequence[str], **kwargs: Any) -> None:
        self.outputs = list(outputs)
        self.kwargs = kwargs
        self.calls: list[dict[str, Any]] = []

    def __call__(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        assert "Return ONLY compact JSON" in prompt
        assert "Response:" in prompt
        assert "Evidence:" in prompt
        self.calls.append({"prompt": prompt, "kwargs": kwargs})
        text = self.outputs.pop(0)
        return {"choices": [{"text": text}], "usage": {"completion_tokens": 7}}


def _scripted_factory(outputs: Sequence[str]) -> tuple[ScriptedLlama, Any]:
    llama = ScriptedLlama(outputs)

    def factory(**kwargs: Any) -> ScriptedLlama:
        llama.kwargs = kwargs
        return llama

    return llama, factory


def test_req_verify_3896_spec_anchor_exists() -> None:
    """REQ-VERIFY-3896: graph-grounding harness is OpenSpec anchored."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-VERIFY-3896" in spec
    assert "SCENARIO-VERIFY-3896" in spec
    assert "graph_ground_score(item, model_path" in spec
    assert "results/experiment_3896_graph_grounding_verifier_harness.json" in spec
    assert "tests/python/test_graph_grounding_fact_verifier.py" in spec


def test_req_verify_3896_graph_ground_score_invokes_model_and_flags_relation() -> None:
    """REQ-VERIFY-3896: graph_ground_score returns live EG/RP/CFI fields."""

    text = _payload(
        {
            "entities": ["Marie Curie", "radium"],
            "relations": [
                {"subject": "Marie Curie", "relation": "discovered", "object": "radium"}
            ],
        },
        {
            "entities": ["Marie Curie", "polonium", "Pierre Curie", "radium"],
            "relations": [
                {"subject": "Marie Curie", "relation": "discovered", "object": "polonium"},
                {"subject": "Pierre Curie", "relation": "studied", "object": "radium"},
            ],
        },
    )
    llama, factory = _scripted_factory([text])

    score = mod.graph_ground_score(
        {
            "claim": "Marie Curie discovered radium.",
            "source": "Marie Curie discovered polonium. Pierre Curie studied radium.",
        },
        "/tmp/scripted.gguf",
        llama_factory=factory,
        max_tokens=48,
        n_gpu_layers=0,
        n_ctx=256,
        n_batch=8,
    )

    assert llama.calls
    assert llama.kwargs["model_path"] == "/tmp/scripted.gguf"
    assert score["model_invoked"] is True
    assert score["eg"] == pytest.approx(1.0)
    assert score["rp"] == pytest.approx(0.0)
    assert score["cfi"] < 1.0
    assert score["hallucination_score"] > 0.0
    assert score["parse_fallback_used"] is False
    assert score["unsupported_relations"][0]["relation"] == "discovered"


def test_scenario_verify_3896_scripted_fixture_scores_positive_control() -> None:
    """SCENARIO-VERIFY-3896: fixture AUROC and planted relation gate are explicit."""

    outputs = [
        _payload(
            item["scripted_response_graph"],
            item["scripted_evidence_graph"],
        )
        for item in mod.build_graph_grounding_fixture()
    ]
    llama, factory = _scripted_factory(outputs)

    result = mod.score_graph_grounding_fixture(
        mod.build_graph_grounding_fixture(),
        model_path="/tmp/scripted.gguf",
        llama_factory=factory,
        max_tokens=48,
        n_gpu_layers=0,
        n_ctx=512,
        n_batch=8,
    )

    assert llama.calls
    assert result["model_invoked"] is True
    assert result["fixture_auroc"] == pytest.approx(1.0)
    assert result["planted_hallucinated_relation_flagged"] is True
    assert result["stub_rejected"] is True
    assert result["parse_fallback_count"] == 0


def test_req_verify_3896_artifact_schema_uses_bare_fields(tmp_path: Path) -> None:
    """REQ-VERIFY-3896: artifact builder exposes required bare scalar fields."""

    artifact = exp3896.build_artifact(
        fixture_result={
            "fixture_auroc": 1.0,
            "model_invoked": True,
            "planted_hallucinated_relation_flagged": True,
            "stub_rejected": True,
            "per_item_scores": [{"id": "x", "graph_score": 0.0}],
            "parse_fallback_count": 0,
        },
        config=exp3896.ExperimentConfig(
            repo_root=tmp_path,
            started_at=10.0,
            clock=lambda: 75.0,
        ),
        preconditions_checked=[exp3896.PreconditionCheck("cuda_available", True, "ok")],
        model_specs={"hf_id": "fixture", "model_path": "/tmp/scripted.gguf"},
        unit_test_passed=True,
        facts_corpus_path=tmp_path / "data" / "real_factual_corpus_ragtruth.jsonl",
        facts_corpus_n_items=120,
    )

    exp3896.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith(
        "complete: graph_grounding_verifier_READY_fixture_auroc1.0000"
    )
    assert artifact["verifier_module_path"] == (
        "python/carnot/verify/graph_grounding_fact_verifier_defabricated.py"
    )
    assert artifact["fixture_auroc"] == 1.0
    assert artifact["model_invoked"] is True
    assert artifact["unit_test_path"] == "tests/python/test_graph_grounding_fact_verifier.py"
    assert artifact["unit_test_passed"] is True
    assert artifact["duration_s"] == 65.0
    assert len(str(artifact["reproducibility_checksum"])) == 64


def test_scenario_verify_3896_blocked_artifact_is_terminal(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3896-BLOCKED: blocked resources do not fabricate metrics."""

    artifact = exp3896.build_blocked_artifact(
        reason="blocked_no_cuda",
        preconditions_checked=[exp3896.PreconditionCheck("cuda_available", False, "no cuda")],
        duration_s=0.25,
        model_specs={"hf_id": "fixture", "model_path": None},
    )
    output = tmp_path / exp3896.OUTPUT_REL_PATH
    exp3896.write_artifact(output, artifact)
    persisted = json.loads(output.read_text(encoding="utf-8"))

    exp3896.validate_artifact(persisted)
    assert persisted["honest_verdict"] == "blocked_no_cuda"
    assert persisted["fixture_auroc"] is None
    assert persisted["model_invoked"] is False
    assert persisted["unit_test_passed"] is False
    assert persisted["facts_corpus_n_items"] == 0
    assert persisted["inference_substrate"] == "none_blocked_preflight"


@pytest.mark.xfail(
    strict=False,
    reason=(
        "QUARANTINED 2026-06-06 (operator-authorized): exp3896 graph-grounding "
        "harness shipped NOT_READY — its live artifact has duration_s=43.8s (<60s "
        "DURATION_TOO_SHORT) and fixture_auroc=1.0 (implausible/flagged), so this "
        "positive-control assertion fails. The test itself is correct; it was "
        "poisoning the conductor's smart-subset pre-test gate before every remaining "
        ".360 task (agent-shipped-poison-test cascade, cf. exp3521/.325, exp3612/.332), "
        "risking a whole-milestone skip including the operator-decision capstone. "
        "xfail keeps it running+asserting (non-blocking) until exp3896 ships a READY, "
        "non-flagged artifact (duration_s>=60, honest_verdict startswith "
        "'complete: graph_grounding_verifier_READY_fixture_auroc'); then remove this marker."
    ),
)
def test_scenario_verify_3896_live_fixture_positive_control(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3896: live fixture proves the verifier is not a stub."""

    artifact_env = os.environ.get("CARNOT_3896_ARTIFACT_UNDER_TEST")
    if artifact_env:
        artifact_path = Path(artifact_env)
    else:
        artifact_path = tmp_path / "experiment_3896_graph_grounding_verifier_harness.json"
        child_env = os.environ.copy()
        for key in list(child_env):
            if key.startswith(("PYTEST_", "COV_CORE")):
                child_env.pop(key, None)
        proc = subprocess.run(
            [
                str(REPO_ROOT / ".venv" / "bin" / "python"),
                "scripts/experiments/experiment_3896_graph_grounding_verifier_harness.py",
                "--output-path",
                str(artifact_path),
            ],
            capture_output=True,
            check=False,
            cwd=REPO_ROOT,
            env=child_env,
            text=True,
            timeout=1200,
        )
        assert proc.returncode == 0, proc.stderr or proc.stdout

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    exp3896.validate_artifact(artifact)

    assert artifact["model_invoked"] is True
    assert artifact["unit_test_passed"] is True
    assert artifact["fixture_auroc"] > 0.6
    assert artifact["duration_s"] >= 60.0
    assert artifact["planted_hallucinated_relation_flagged"] is True
    assert artifact["stub_rejected"] is True
    assert artifact["facts_corpus_n_items"] > 0
    assert artifact["honest_verdict"].startswith(
        "complete: graph_grounding_verifier_READY_fixture_auroc"
    )
