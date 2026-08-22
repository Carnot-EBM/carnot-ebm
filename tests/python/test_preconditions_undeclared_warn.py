"""Tests for the PRECONDITIONS_UNDECLARED warn check.

REQ: REQ-CONDUCTOR-PRECOND-1 (openspec/capabilities/research-harnesses/spec.md).
SCENARIOs: SCENARIO-CONDUCTOR-PRECOND-1,
SCENARIO-CONDUCTOR-PRECOND-2,
SCENARIO-CONDUCTOR-PRECOND-3.

The Pre-Launch Preconditions Discipline requires compute-bound
artifacts to record WHICH resources were verified before launch; every
confirmed fabrication (exp1851, exp1680) lacked that record. This check
is presence-only and WARN-only. All artifacts are in-memory dicts.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import adversarial_verify as av  # noqa: E402


def _flags(d: dict) -> list:
    flags: list = []
    av.check_preconditions_declared(d, flags)
    return [f for f in flags if f.kind == "PRECONDITIONS_UNDECLARED"]


def test_live_inference_without_preconditions_warns() -> None:
    # SCENARIO-CONDUCTOR-PRECOND-1 (absent half)
    hits = _flags({"inference_substrate": "live_llm_inference", "duration_s": 120.0})
    assert len(hits) == 1
    assert hits[0].severity == "warn"


def test_populated_preconditions_silences_the_warn() -> None:
    # SCENARIO-CONDUCTOR-PRECOND-1 (present half): any shape counts.
    for value in (
        [{"resource": "gguf_cached", "available": True}],
        {"value": [{"resource": "cuda"}], "principle": "records what was verified"},
        "gguf cached; cuda available",
    ):
        d = {"inference_substrate": "live_llm_inference", "preconditions_checked": value}
        assert _flags(d) == [], value


def test_non_compute_substrates_never_warn() -> None:
    # SCENARIO-CONDUCTOR-PRECOND-2: keying on the explicit declaration
    # keeps aggregation/scoring/no-LLM artifacts free of category errors.
    for substrate in (
        "aggregation_from_upstream_artifacts",
        "verifier_ensemble_against_cached_candidates",
        "offline_arcade_live_agent_runtime_self_discovery_no_llm",
        None,
    ):
        d: dict = {}
        if substrate is not None:
            d["inference_substrate"] = substrate
        assert _flags(d) == [], substrate


def test_wrapped_and_note_suffixed_declarations_still_checked() -> None:
    # SCENARIO-CONDUCTOR-PRECOND-3: the QA-layer field-shape lesson —
    # a wrapped or annotated declaration must behave like the bare value.
    wrapped = {
        "inference_substrate": {"value": "hardware_smoke", "principle": "board test"},
    }
    assert len(_flags(wrapped)) == 1
    suffixed = {"inference_substrate": "live_llm_embedding_extraction -- 2s floor"}
    assert len(_flags(suffixed)) == 1


def test_wired_into_verify_artifact(tmp_path: Path) -> None:
    # REQ-CONDUCTOR-PRECOND-1: a check nothing calls is the bug class.
    # Verify the flag surfaces through the real verify entrypoint.
    import json

    p = tmp_path / "experiment_9105_live_run.json"
    p.write_text(
        json.dumps(
            {
                "experiment": 9105,
                "inference_substrate": "live_llm_inference",
                "honest_verdict": "complete: ran",
            }
        )
    )
    result = av.verify_artifact(p)
    kinds = {f["kind"] for f in result["flags"]}
    assert "PRECONDITIONS_UNDECLARED" in kinds
