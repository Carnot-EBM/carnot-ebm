"""Tests for the Exp 3769 Phase-1 package/CLI/MCP E2E smoke.

Spec: REQ-SPOE-3769, SCENARIO-SPOE-3769.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from carnot.reporting import phase1_e2e_smoke as exp3769


ROOT = Path(__file__).resolve().parents[2]


def _surface(
    name: str,
    *,
    passed: bool = True,
    data: dict[str, object] | None = None,
) -> exp3769.SurfaceResult:
    return exp3769.SurfaceResult(
        name=name,
        passed=passed,
        detail=f"{name} detail",
        data={"shape": "ok"} if data is None else data,
    )


def test_complete_artifact_records_required_bare_surface_flags() -> None:
    """REQ-SPOE-3769 stores bare booleans for every E2E surface."""

    artifact = exp3769.assemble_artifact(
        start_time=0.0,
        end_time=1.25,
        preconditions={
            "interpreter": {"passed": True},
            "package_import": {"passed": True},
            "mcp_server_module": {"passed": True},
            "mcp_runtime": {"passed": True},
            "cli_entrypoint": {"passed": True},
        },
        package_result=_surface("package_import", data={"version": "0.1.0b1"}),
        pipeline_result=_surface("pipeline"),
        mcp_result=_surface("mcp_protocol"),
        cli_result=_surface("cli"),
        build_result={"attempted": True, "passed": True},
        model_specs=[{"name": "Qwen3-0.6B", "hf_id": "Qwen/Qwen3-0.6B"}],
        random_seed=exp3769.RANDOM_SEED,
    )

    assert set(exp3769.REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert artifact["honest_verdict"] == exp3769.COMPLETE_VERDICT
    assert artifact["package_importable"] is True
    assert artifact["pipeline_e2e_passed"] is True
    assert artifact["mcp_protocol_exchange_passed"] is True
    assert artifact["cli_passed"] is True
    assert artifact["surfaces_passed"] == [
        "package_import",
        "pipeline",
        "mcp_protocol",
        "cli",
    ]
    assert artifact["is_wiring_smoke_not_accuracy_claim"] is True
    assert isinstance(artifact["reproducibility_checksum"], str)
    assert len(artifact["reproducibility_checksum"]) == 16


def test_blocked_artifact_does_not_fabricate_surface_passes() -> None:
    """SCENARIO-SPOE-3769 emits blocked_<resource> before E2E on missing runtime."""

    artifact = exp3769.blocked_artifact(
        "blocked_mcp_runtime",
        preconditions={
            "interpreter": {"passed": True},
            "package_import": {"passed": True},
            "mcp_server_module": {"passed": True},
            "mcp_runtime": {"passed": False, "detail": "mcp missing"},
            "cli_entrypoint": {"passed": True},
        },
        start_time=0.0,
        end_time=0.5,
        model_specs=[],
    )

    assert artifact["honest_verdict"] == "blocked_mcp_runtime"
    assert artifact["package_importable"] is True
    assert artifact["pipeline_e2e_passed"] is False
    assert artifact["mcp_protocol_exchange_passed"] is False
    assert artifact["cli_passed"] is False
    assert artifact["surfaces_passed"] == ["package_import"]


def test_cli_score_candidates_invokes_real_packaged_module() -> None:
    """REQ-SPOE-3769 invokes the packaged CLI instead of an in-process fake."""

    result = exp3769.run_cli_score_candidates(
        ROOT,
        sys.executable,
        exp3769.tiny_candidates(),
    )

    assert result.passed is True, result.detail
    assert result.data["returncode"] == 0
    assert result.data["scores"][0]["calibrated_error_score"] is not None
    assert result.data["scores"][0]["operating_point"] is not None


def test_mcp_score_candidates_uses_real_stdio_protocol() -> None:
    """REQ-SPOE-3769 requires a real MCP stdio protocol exchange."""

    result = exp3769.run_mcp_protocol_exchange(
        ROOT,
        sys.executable,
        exp3769.tiny_candidates(),
    )

    assert result.passed is True, result.detail
    assert result.data["protocol"] == "mcp_stdio_json_rpc"
    assert result.data["tool_name"] == "score_candidates"
    assert result.data["scores"][0]["calibrated_error_score"] is not None
    assert result.data["scores"][0]["operating_point"] is not None


def test_run_writes_artifact_with_injected_surface_runners(tmp_path: Path) -> None:
    """SCENARIO-SPOE-3769 writes the Exp 3769 artifact after all surfaces pass."""

    output_path = tmp_path / "results/experiment_3769_package_cli_mcp_e2e_smoke.json"

    artifact = exp3769.run(
        ROOT,
        output_path=output_path,
        executable=sys.executable,
        build_runner=lambda _root, _exe: {"attempted": False, "passed": None},
        pipeline_runner=lambda _root, _exe: _surface("pipeline"),
        mcp_runner=lambda _root, _exe, _candidates: _surface("mcp_protocol"),
        cli_runner=lambda _root, _exe, _candidates: _surface("cli"),
        model_resolver=lambda _root: [
            {"name": "Qwen3-0.6B", "hf_id": "Qwen/Qwen3-0.6B", "role": "smoke"}
        ],
    )

    assert artifact["honest_verdict"] == exp3769.COMPLETE_VERDICT
    assert output_path.exists()
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
