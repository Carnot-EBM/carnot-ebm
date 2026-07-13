"""Tests for Exp 5594 /think vs /no_think induction quality A/B.

Spec refs: REQ-ARC-WMTE-5594, SCENARIO-ARC-WMTE-5594-INCOMPATIBLE-BLOCKS-CLEANLY,
SCENARIO-ARC-WMTE-5594-TAG-VARIANT-RECOGNIZED.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from carnot import experiment_5594_think_mode_induction_quality_ab as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def test_req_arc_wmte_5594_spec_declares_ab_contract() -> None:
    """REQ-ARC-WMTE-5594: OpenSpec declares the think-mode A/B contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-ARC-WMTE-5594") :]

    for marker in (
        "REQ-ARC-WMTE-5594",
        "SCENARIO-ARC-WMTE-5594-INCOMPATIBLE-BLOCKS-CLEANLY",
        "SCENARIO-ARC-WMTE-5594-TAG-VARIANT-RECOGNIZED",
        "check_think_mode_compatibility",
        "_L2_CODEONLY_DIRECTIVE",
        "<thinking>",
    ):
        assert marker in section


def test_scenario_arc_wmte_5594_blocked_precondition_never_runs(monkeypatch) -> None:
    """A missing resource fails closed without attempting any induction call."""

    monkeypatch.setattr(
        mod,
        "preconditions",
        lambda root=mod.REPO_ROOT: {
            "offline_arcade_importable": False,
            "offline_arcade_makes_env": False,
            "e3_policy_import": True,
            "gguf_cached": True,
            "llama_server_binary_present": True,
            "port_8920_prewarmed": True,
            "ok": False,
        },
    )

    def _fail_if_called(*_args, **_kwargs):
        raise AssertionError("_run_one_arm must not run when a precondition is missing")

    monkeypatch.setattr(mod, "_run_one_arm", _fail_if_called)
    monkeypatch.setattr(mod, "check_think_mode_compatibility", _fail_if_called)

    artifact = mod.build_artifact()

    assert artifact["honest_verdict"].startswith("complete: blocked_")
    assert artifact["per_game_results"] == []
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact


def test_scenario_arc_wmte_5594_incompatible_blocks_cleanly(monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-5594-INCOMPATIBLE-BLOCKS-CLEANLY: when the compatibility probe
    finds no reasoning-tag prefix and no material length delta, the experiment stops
    with the task's own instructed blocked verdict and never attempts a roster game."""

    monkeypatch.setattr(
        mod,
        "preconditions",
        lambda root=mod.REPO_ROOT: {
            "offline_arcade_importable": True,
            "offline_arcade_makes_env": True,
            "e3_policy_import": True,
            "gguf_cached": True,
            "llama_server_binary_present": True,
            "port_8920_prewarmed": True,
            "ok": True,
        },
    )
    monkeypatch.setattr(
        mod,
        "check_think_mode_compatibility",
        lambda: (
            False,
            "no reasoning-tag prefix and no material length delta (10 vs 9 chars) -- "
            "MTP may be silently ignoring the think-mode toggle",
        ),
    )

    def _fail_if_called(*_args, **_kwargs):
        raise AssertionError("_run_one_arm must not run when think mode is incompatible")

    monkeypatch.setattr(mod, "_run_one_arm", _fail_if_called)

    artifact = mod.build_artifact()

    assert artifact["honest_verdict"] == "complete: blocked_think_mode_incompatible_with_mtp"
    assert artifact["think_mode_compatible_with_mtp"] is False
    assert artifact["per_game_results"] == []
    assert artifact["no_think_induction_success_count"] == 0
    assert artifact["think_induction_success_count"] == 0


def test_scenario_arc_wmte_5594_tag_variant_recognized(monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-5594-TAG-VARIANT-RECOGNIZED: a `/think`-arm response that opens
    with `<thinking>` (not the literal `<think>`) is recognized as a reasoning-tag match,
    not a false `no observable difference` negative -- the exact bug found and fixed in
    this session (the substring "<think>" is not present inside "<thinking>", since the
    closing ">" does not align at the same position)."""

    think_body = json.dumps(
        {"content": "<thinking>\nThe user wants me to write a world-model engine..."}
    ).encode()
    no_think_body = json.dumps({"content": "def engine(grid, action, data):\n    pass"}).encode()

    call_order: list[str] = []

    class _FakeResponse:
        def __init__(self, body: bytes) -> None:
            self._body = body

        def __enter__(self) -> "_FakeResponse":
            return self

        def __exit__(self, *_exc) -> bool:
            return False

        def read(self) -> bytes:
            return self._body

    def _fake_urlopen(req, timeout=60):  # noqa: ARG001 - signature matches real urlopen call
        # source order: no_think request is opened before the think request
        if not call_order:
            call_order.append("no_think")
            return _FakeResponse(no_think_body)
        call_order.append("think")
        return _FakeResponse(think_body)

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)

    compatible, detail = mod.check_think_mode_compatibility()

    assert compatible is True
    assert "reasoning tag" in detail
    assert call_order == ["no_think", "think"]


def test_scenario_arc_wmte_5594_no_tag_and_no_length_delta_reports_incompatible(
    monkeypatch,
) -> None:
    """The inverse of the tag-variant scenario: genuinely indistinguishable output (no
    reasoning-tag prefix, length within the 1.15x fallback tolerance) is honestly
    reported as incompatible, not papered over."""

    think_body = json.dumps({"content": "def engine(grid, action, data):\n    pass  # a"}).encode()
    no_think_body = json.dumps({"content": "def engine(grid, action, data):\n    pass"}).encode()

    class _FakeResponse:
        def __init__(self, body: bytes) -> None:
            self._body = body

        def __enter__(self) -> "_FakeResponse":
            return self

        def __exit__(self, *_exc) -> bool:
            return False

        def read(self) -> bytes:
            return self._body

    responses = iter([_FakeResponse(no_think_body), _FakeResponse(think_body)])
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda req, timeout=60: next(responses),  # noqa: ARG005
    )

    compatible, detail = mod.check_think_mode_compatibility()

    assert compatible is False
    assert "no reasoning-tag prefix" in detail


def test_req_arc_wmte_5594_repository_artifact_is_a_real_measured_result() -> None:
    """REQ-ARC-WMTE-5594: the checked-in real run measured induction quality with the
    live default Qwen3.5-9B-MTP proposer, both arms inducing successfully on both
    roster games, and an honest per-game accuracy comparison -- not a fabricated or
    blocked stub."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    assert result["honest_verdict"].startswith("complete: think_mode_ab_")
    assert result["think_mode_compatible_with_mtp"] is True
    assert result["inference_substrate"] == "live_llm_inference"
    assert result["solve_provenance"] == "development_proxy"
    assert result["no_think_induction_success_count"] == 2
    assert result["think_induction_success_count"] == 2
    assert len(result["per_game_results"]) == 4
    assert all(row["induction_ok"] for row in result["per_game_results"])
    assert result["duration_s"] > 60.0
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in result
