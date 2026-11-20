"""The ARC no-LLM duration floor was UNREACHABLE for a clean artifact (2026-07-30).

Spec: REQ-ARC-WMTE-6050 (the ARC no-LLM duration floor SHALL be reachable without a
vestigial model marker).

WHY THIS TEST EXISTS -- the shape of the bug, in plain terms.

`scripts/adversarial_verify.py` keeps a per-substrate duration floor. For the substrate
`offline_arcade_live_agent_runtime_self_discovery_no_llm` -- the live ARC agent taking real
actions against the offline arcade with NO model loaded -- the floor is 0.01s, on the reasoning
that even a single real environment action takes non-zero time, so anything faster means the
duration was never measured.

`check_duration_vs_claim` has a DEDICATED branch to enforce that floor (`floor["reason"] ==
"arc_live_agent_no_llm"`). But that branch sat BEHIND an early return:

    if not _has_compute_bound_marker(d) and not _is_live_llm_inference(d):
        return

which means the branch only ever ran for an artifact that ALSO carried a GGUF/CUDA-ish string.
That is exactly backwards. The whole reason the branch exists is that GGUF strings in THIS
substrate are VESTIGIAL -- they name the generator that WOULD have fired if the LLM tier were
used (`invoked: false`). So the floor fired only for artifacts that happened to mention a model
they never ran, and NOT for the clean case: a no-LLM artifact with no vestigial marker could
declare any duration at all -- 0.002s, or 0.0 -- and be reported clean.

HOW IT WAS FOUND. Not by a unit test (the existing tests covered the branch by handing it an
artifact WITH a marker, so they passed while the guard above made the branch dead for real clean
inputs -- the "tests test what the author thought to test" failure mode named in CLAUDE.md's
QA-Layer Authenticity Discipline). It was found by probing the linter with a hand-built clean
artifact while verifying an artifact from this session that declared this substrate at
duration_s=0.002 -- BELOW ITS OWN FLOOR -- and was reported clean by the full scan.

CORPUS IMPACT OF THE FIX: zero. All 99 artifacts on disk declaring this substrate have
duration_s >= 0.01, so closing the hole newly flags no historical artifact. The fix removes a
blind spot going forward rather than reinterpreting the past.

Cross-references: CLAUDE.md "Inference-Substrate Declaration Discipline" (the substrate table and
its floors), CLAUDE.md "QA-Layer Authenticity Discipline" (this is the bug class it names:
a check that cannot fire on the inputs it was written for).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]


def _load_av() -> Any:
    """Load `scripts/adversarial_verify.py` by path (it is a script, not a package module)."""
    path = REPO / "scripts" / "adversarial_verify.py"
    spec = importlib.util.spec_from_file_location("adversarial_verify_floor_probe", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


AV = _load_av()

# A CLEAN artifact of this substrate: it declares the no-LLM live-agent substrate and carries NO
# GGUF / CUDA / model_specs marker, because by that substrate's own definition there is no model
# to name. This is the input the floor branch was written for and could not see.
_CLEAN_NO_LLM_BELOW_FLOOR = {
    "experiment": "floor_probe",
    "schema": "carnot.floor_probe",
    "run_date": "20260730",
    "inference_substrate": "offline_arcade_live_agent_runtime_self_discovery_no_llm",
    "duration_s": 0.002,
    "random_seed": 1,
    "reproducibility_checksum": "deadbeef",
    "honest_verdict": "complete_probe",
    "levels_completed": 0,
}


def _duration_flags(d: dict) -> list:
    """Collect DURATION_TOO_SHORT flags. `Flag` is a plain object with attributes, not a dict."""
    flags: list = []
    AV.check_duration_vs_claim(d, flags)
    return [f for f in flags if getattr(f, "kind", None) == "DURATION_TOO_SHORT"]


def test_the_substrate_is_recognised_and_its_floor_is_one_hundredth_of_a_second() -> None:
    """Guard the premise: if either of these drifts, the rest of the test proves nothing."""
    assert AV._is_arc_live_agent_no_llm(_CLEAN_NO_LLM_BELOW_FLOOR) is True
    floor = AV.duration_floor_for_artifact(_CLEAN_NO_LLM_BELOW_FLOOR)
    assert floor is not None
    assert floor["reason"] == "arc_live_agent_no_llm"
    assert float(floor["min_duration_s"]) == 0.01


def test_req_arc_wmte_6681_outcome_transport_uses_reviewed_no_llm_classification() -> None:
    """REQ-ARC-WMTE-6681 must not rely on the generic no-LLM name fallback."""

    payload = {
        "inference_substrate": AV.ARC_CANONICAL_OUTCOME_TRANSPORT_NO_LLM_SUBSTRATE,
        "duration_s": 0.02,
    }
    classification = AV._classify_inference_substrate(payload)
    warning_flags: list = []
    AV._emit_no_llm_by_name_warning(payload, warning_flags)

    assert classification == {
        "kind": AV.SUBSTRATE_KIND_NO_LLM,
        "declared_value": AV.ARC_CANONICAL_OUTCOME_TRANSPORT_NO_LLM_SUBSTRATE,
        "matched_value": AV.ARC_CANONICAL_OUTCOME_TRANSPORT_NO_LLM_SUBSTRATE,
        "source": "top_level_inference_substrate",
    }
    assert AV._is_arc_live_agent_no_llm(payload) is True
    assert AV.duration_floor_for_artifact(payload)["reason"] == "arc_live_agent_no_llm"
    assert warning_flags == []


def test_req_arc_wmte_6682_supervisor_ab_uses_reviewed_no_llm_classification() -> None:
    """REQ-ARC-WMTE-6682 must not rely on the generic no-LLM name fallback."""

    payload = {
        "inference_substrate": AV.ARC_SUPERVISOR_AB_NO_LLM_SUBSTRATE,
        "duration_s": 0.02,
    }
    classification = AV._classify_inference_substrate(payload)
    warning_flags: list = []
    AV._emit_no_llm_by_name_warning(payload, warning_flags)

    assert classification == {
        "kind": AV.SUBSTRATE_KIND_NO_LLM,
        "declared_value": AV.ARC_SUPERVISOR_AB_NO_LLM_SUBSTRATE,
        "matched_value": AV.ARC_SUPERVISOR_AB_NO_LLM_SUBSTRATE,
        "source": "top_level_inference_substrate",
    }
    assert AV._is_arc_live_agent_no_llm(payload) is True
    assert AV.duration_floor_for_artifact(payload)["reason"] == "arc_live_agent_no_llm"
    assert warning_flags == []


def test_clean_no_llm_artifact_below_floor_is_flagged_without_any_vestigial_marker() -> None:
    """THE REGRESSION. This is the exact input that was silently passing before 2026-07-30.

    No `model_specs`, no GGUF string, no CUDA string -- so `_has_compute_bound_marker` is False
    and the old early return fired before the floor branch could run.
    """
    assert AV._has_compute_bound_marker(_CLEAN_NO_LLM_BELOW_FLOOR) is False, (
        "premise: this artifact must carry NO compute marker, or it would have reached the "
        "floor check even under the old code and would not reproduce the bug"
    )
    assert AV._is_live_llm_inference(_CLEAN_NO_LLM_BELOW_FLOOR) is False

    flags = _duration_flags(_CLEAN_NO_LLM_BELOW_FLOOR)
    assert len(flags) == 1, f"expected the no-LLM floor to fire, got {flags}"
    assert flags[0].severity == "critical"
    assert "no-LLM substrate" in flags[0].detail


def test_a_zero_duration_is_also_caught() -> None:
    """0.0 is the degenerate case the floor exists for: a duration that was never measured."""
    d = dict(_CLEAN_NO_LLM_BELOW_FLOOR, duration_s=0.0)
    assert len(_duration_flags(d)) == 1


def test_a_plausible_duration_for_this_substrate_is_not_flagged() -> None:
    """The fix must not make the substrate un-declarable. A real no-LLM run clears the floor.

    Without this the change could 'pass' by flagging everything, which would be worse than the
    hole it closes: 99 historical artifacts declare this substrate legitimately.
    """
    for dur in (0.01, 0.5, 42.0, 3600.0):
        d = dict(_CLEAN_NO_LLM_BELOW_FLOOR, duration_s=dur)
        assert _duration_flags(d) == [], f"duration_s={dur} should clear the 0.01s floor"


def test_the_live_llm_floor_is_untouched_by_this_change() -> None:
    """Scope guard: a live-LLM claim still gets the 60s floor, not the 0.01s one.

    The fix widened WHICH artifacts reach the floor dispatch; it must not have altered which
    floor any of them gets.
    """
    d = {
        "experiment": "live_probe",
        "run_date": "20260730",
        "inference_substrate": "live_llm_inference",
        "model_specs": [{"name": "gemma-4-31B-it", "hf_id": "unsloth/gemma-4-31B-it-GGUF"}],
        "duration_s": 3.0,
        "random_seed": 1,
        "reproducibility_checksum": "deadbeef",
        "honest_verdict": "complete_probe",
    }
    flags = _duration_flags(d)
    assert len(flags) == 1
    assert "0.01" not in flags[0].detail, (
        "a live-LLM artifact must be judged against the live-model floor, not the no-LLM one"
    )


def test_no_artifact_on_disk_is_newly_flagged_by_closing_this_hole() -> None:
    """The corpus-impact claim in this module's docstring, asserted rather than asserted-in-prose.

    Every committed artifact declaring this substrate must already satisfy the floor. If this
    ever fails, a real historical artifact has a duration below the floor it declares, and that
    needs a corrigendum decision by the operator -- NOT a quiet loosening of the floor.
    """
    import json

    checked = 0
    offenders = []
    for path in (REPO / "results").rglob("*.json"):
        try:
            d = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(d, dict) or not AV._is_arc_live_agent_no_llm(d):
            continue
        checked += 1
        dur = d.get("duration_s")
        if isinstance(dur, (int, float)) and not isinstance(dur, bool) and float(dur) < 0.01:
            offenders.append((str(path.relative_to(REPO)), dur))
    assert checked > 0, "expected to find artifacts declaring this substrate"
    assert offenders == [], f"sub-floor artifacts declaring the no-LLM substrate: {offenders}"
