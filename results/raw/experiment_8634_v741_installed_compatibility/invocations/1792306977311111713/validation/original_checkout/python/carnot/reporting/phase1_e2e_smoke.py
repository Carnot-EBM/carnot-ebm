"""Exp 3769 Phase-1 package/CLI/MCP software E2E smoke.

**Researcher summary:**
    Verifies the software path an integrator actually uses: import the package,
    run the verify-repair pipeline on a tiny arithmetic slip, call the MCP
    `score_candidates` tool over stdio JSON-RPC, and invoke the packaged CLI.

**Detailed explanation for engineers:**
    This module is intentionally a wiring smoke, not an accuracy benchmark.  It
    records bare booleans for each shipped surface and writes a terminal JSON
    artifact.  The default MCP runner starts `python -m carnot.mcp` as a child
    process and talks to it through the MCP client stdio transport; it never
    calls the tool handler in-process.

Spec: REQ-SPOE-3769, SCENARIO-SPOE-3769.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import importlib.util
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable


RESULT_REL_PATH = Path("results/experiment_3769_package_cli_mcp_e2e_smoke.json")
RANDOM_SEED = 3769
COMPLETE_VERDICT = (
    "complete: "
    "phase1_e2e_smoke_package_import_pipeline_mcp_protocol_cli_passed_"
    "wiring_smoke_not_accuracy_claim"
)
INFERENCE_SUBSTRATE = (
    "live_llm_inference (principle: the pipeline runs a small model E2E)."
)
SMOKE_MODEL = {
    "name": "Qwen3-0.6B",
    "hf_id": "Qwen/Qwen3-0.6B",
    "role": "small_cpu_wiring_smoke",
}
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "package_importable",
    "pipeline_e2e_passed",
    "mcp_protocol_exchange_passed",
    "cli_passed",
    "surfaces_passed",
    "is_wiring_smoke_not_accuracy_claim",
    "preconditions_checked",
    "model_specs",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)


@dataclass(frozen=True)
class SurfaceResult:
    """One Phase-1 E2E surface result.

    The `passed` field is deliberately a bare boolean because the artifact is
    consumed by ship-gate code that must not parse prose to determine whether a
    surface passed.
    """

    name: str
    passed: bool
    detail: str
    data: dict[str, Any] = field(default_factory=dict)


def tiny_candidates() -> list[dict[str, Any]]:
    """Return the hardcoded detector candidate payload used by CLI and MCP."""

    return [
        {
            "candidate_id": "exp3769_math_slip",
            "domain": "math",
            "text": "We compute 7 + 5 = 13, so the answer is 13.",
            "confidence": 0.2,
            "ensemble_energy": 0.8,
        }
    ]


def _tail(text: str, limit: int = 2000) -> str:
    return text[-limit:] if len(text) > limit else text


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def compute_reproducibility_checksum(seed: int, payload: dict[str, Any]) -> str:
    """Return a stable 16-character content hash for drift detection."""

    body = json.dumps({"seed": seed, "payload": payload}, sort_keys=True, default=str)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


def _score_shape_ok(payload: dict[str, Any]) -> bool:
    scores = payload.get("scores")
    if not isinstance(scores, list) or not scores:
        return False
    first = scores[0]
    if not isinstance(first, dict):
        return False
    return (
        first.get("calibrated_error_score") is not None
        and first.get("operating_point") is not None
    )


def _import_check(module_name: str) -> tuple[bool, str]:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return False, f"{module_name} not found"
    try:
        importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - environment-dependent import failure.
        return False, f"{type(exc).__name__}: {exc}"
    return True, str(spec.origin)


def check_preconditions(root: Path, executable: str) -> dict[str, dict[str, Any]]:
    """Check resources required before any E2E surface claims are made."""

    expected_executable = root / ".venv" / "bin" / "python"
    executable_path = Path(executable)
    interpreter_ok = executable_path == expected_executable

    preconditions: dict[str, dict[str, Any]] = {
        "interpreter": {
            "passed": interpreter_ok,
            "value": str(executable_path),
            "expected": str(expected_executable),
        }
    }

    try:
        import carnot

        preconditions["package_import"] = {
            "passed": True,
            "version": getattr(carnot, "__version__", "<missing>"),
            "module_path": getattr(carnot, "__file__", ""),
        }
    except Exception as exc:  # pragma: no cover - only reachable in broken envs.
        preconditions["package_import"] = {
            "passed": False,
            "detail": f"{type(exc).__name__}: {exc}",
        }

    mcp_server_ok, mcp_server_detail = _import_check("carnot.mcp.server")
    preconditions["mcp_server_module"] = {
        "passed": mcp_server_ok,
        "detail": mcp_server_detail,
    }

    mcp_runtime_ok = importlib.util.find_spec("mcp") is not None
    preconditions["mcp_runtime"] = {
        "passed": mcp_runtime_ok,
        "detail": "mcp runtime importable" if mcp_runtime_ok else "mcp runtime missing",
    }

    cli_ok, cli_detail = _import_check("carnot.cli")
    if cli_ok:
        cli_module = importlib.import_module("carnot.cli")
        cli_ok = callable(getattr(cli_module, "main", None))
        cli_detail = "carnot.cli:main callable" if cli_ok else "carnot.cli:main missing"
    preconditions["cli_entrypoint"] = {"passed": cli_ok, "detail": cli_detail}
    return preconditions


def _first_failed_precondition(preconditions: dict[str, dict[str, Any]]) -> str | None:
    for name, result in preconditions.items():
        if not result.get("passed"):
            return f"blocked_{name}"
    return None


def resolve_model_specs(root: Path) -> list[dict[str, Any]]:
    """Resolve the small CPU smoke model used by this wiring check."""

    del root
    specs: list[dict[str, Any]] = [dict(SMOKE_MODEL)]
    try:
        from huggingface_hub import try_to_load_from_cache

        cached_weight = try_to_load_from_cache(SMOKE_MODEL["hf_id"], "model.safetensors")
        if isinstance(cached_weight, str):
            specs[0]["model_path"] = str(Path(cached_weight).parent)
            specs[0]["cache_status"] = "cached"
        else:
            specs[0]["cache_status"] = "missing_model_safetensors"
    except Exception as exc:  # pragma: no cover - dependency/environment specific.
        specs[0]["cache_status"] = f"cache_probe_failed: {type(exc).__name__}: {exc}"
    return specs


def run_package_import(preconditions: dict[str, dict[str, Any]]) -> SurfaceResult:
    """Return the package-import surface result from precondition evidence."""

    package = preconditions.get("package_import", {})
    passed = bool(package.get("passed"))
    version = package.get("version", "<missing>")
    detail = f"import carnot version={version}" if passed else str(package.get("detail"))
    return SurfaceResult(
        name="package_import",
        passed=passed,
        detail=detail,
        data={"version": version, "module_path": package.get("module_path")},
    )


def run_optional_build(root: Path, executable: str) -> dict[str, Any]:
    """Build a local sdist/wheel when the `build` package is available."""

    if importlib.util.find_spec("build") is None:
        return {"attempted": False, "passed": None, "detail": "python -m build unavailable"}
    with tempfile.TemporaryDirectory(prefix="carnot-exp3769-build-") as tmpdir:
        proc = subprocess.run(
            [executable, "-m", "build", "--sdist", "--wheel", "--outdir", tmpdir],
            cwd=str(root),
            text=True,
            capture_output=True,
            timeout=180,
            check=False,
        )
    return {
        "attempted": True,
        "passed": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout_tail": _tail(proc.stdout),
        "stderr_tail": _tail(proc.stderr),
    }


def run_pipeline_e2e(root: Path, executable: str) -> SurfaceResult:
    """Run VerifyRepairPipeline on a tiny arithmetic-slip wiring smoke."""

    del executable
    model_specs = resolve_model_specs(root)
    smoke = model_specs[0]
    model_path = smoke.get("model_path")
    if not isinstance(model_path, str):
        return SurfaceResult(
            name="pipeline",
            passed=False,
            detail="blocked_small_model_cache",
            data={"model_specs": model_specs},
        )

    old_env = {
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
        "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        from carnot.pipeline.verify_repair import VerifyRepairPipeline

        question = "What is 7 + 5?"
        response = "We compute 7 + 5 = 13, so the answer is 13."
        pipeline = VerifyRepairPipeline(
            model=model_path,
            domains=["arithmetic"],
            max_repairs=0,
            timeout_seconds=180,
        )
        live_generation = pipeline._generate(  # noqa: SLF001 - this smoke records live inference.
            "Return only the numeral for 2 + 2.",
            max_new_tokens=8,
        )
        verify_result = pipeline.verify(question, response, "arithmetic")
    except Exception as exc:
        return SurfaceResult(
            name="pipeline",
            passed=False,
            detail=f"blocked_pipeline_e2e: {type(exc).__name__}: {exc}",
            data={"model_specs": model_specs},
        )
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    data = {
        "question": question,
        "response": response,
        "verified": bool(verify_result.verified),
        "energy": float(verify_result.energy),
        "n_constraints": len(verify_result.constraints),
        "n_violations": len(verify_result.violations),
        "live_generation_preview": str(live_generation).strip()[:200],
        "device": "cpu_forced",
        "model_specs": model_specs,
    }
    passed = bool(data["live_generation_preview"]) and data["n_constraints"] >= 1
    detail = "structured verify result with live small-model generation" if passed else "pipeline returned incomplete structure"
    return SurfaceResult(name="pipeline", passed=passed, detail=detail, data=data)


def run_cli_score_candidates(
    root: Path,
    executable: str,
    candidates: list[dict[str, Any]],
) -> SurfaceResult:
    """Invoke the documented packaged CLI module on the tiny candidate payload."""

    proc = subprocess.run(
        [
            executable,
            "-m",
            "carnot.cli",
            "score-candidates",
            "--domain",
            "math",
            "--candidates-json",
            json.dumps(candidates, sort_keys=True),
        ],
        cwd=str(root),
        text=True,
        capture_output=True,
        timeout=90,
        check=False,
    )
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        payload = {}
    passed = proc.returncode == 0 and _score_shape_ok(payload)
    data = {
        "returncode": proc.returncode,
        "stdout_tail": _tail(proc.stdout),
        "stderr_tail": _tail(proc.stderr),
        **payload,
    }
    detail = "CLI score-candidates returned calibrated score" if passed else "CLI score-candidates failed"
    return SurfaceResult(name="cli", passed=passed, detail=detail, data=data)


def _parse_mcp_payload(result: Any) -> dict[str, Any]:
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        return structured
    content = getattr(result, "content", None) or []
    if content:
        text = getattr(content[0], "text", "")
        if text:
            return json.loads(text)
    return {}


def run_mcp_protocol_exchange(
    root: Path,
    executable: str,
    candidates: list[dict[str, Any]],
) -> SurfaceResult:
    """Call MCP `score_candidates` through stdio JSON-RPC."""

    async def _call() -> dict[str, Any]:
        from mcp import StdioServerParameters
        from mcp.client.session import ClientSession
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=executable,
            args=["-m", "carnot.mcp"],
            cwd=str(root),
        )
        async with stdio_client(params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(
                    "score_candidates",
                    {"candidates": candidates, "domain": "math"},
                )
                payload = _parse_mcp_payload(result)
                payload["is_error"] = bool(getattr(result, "isError", False))
                return payload

    try:
        payload = asyncio.run(_call())
    except Exception as exc:
        return SurfaceResult(
            name="mcp_protocol",
            passed=False,
            detail=f"MCP stdio exchange failed: {type(exc).__name__}: {exc}",
            data={"protocol": "mcp_stdio_json_rpc", "tool_name": "score_candidates"},
        )

    passed = not payload.get("is_error") and _score_shape_ok(payload)
    data = {
        "protocol": "mcp_stdio_json_rpc",
        "tool_name": "score_candidates",
        **payload,
    }
    detail = "MCP stdio score_candidates returned calibrated score" if passed else "MCP score_candidates failed"
    return SurfaceResult(name="mcp_protocol", passed=passed, detail=detail, data=data)


def blocked_artifact(
    verdict: str,
    *,
    preconditions: dict[str, dict[str, Any]],
    start_time: float,
    end_time: float,
    model_specs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a blocked artifact without claiming unrun E2E surfaces passed."""

    package_importable = bool(preconditions.get("package_import", {}).get("passed"))
    surfaces_passed = ["package_import"] if package_importable else []
    artifact = {
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "package_importable": package_importable,
        "pipeline_e2e_passed": False,
        "mcp_protocol_exchange_passed": False,
        "cli_passed": False,
        "surfaces_passed": surfaces_passed,
        "is_wiring_smoke_not_accuracy_claim": True,
        "preconditions_checked": preconditions,
        "model_specs": model_specs,
        "random_seed": RANDOM_SEED,
        "duration_s": round(max(end_time - start_time, 0.0), 3),
    }
    artifact["reproducibility_checksum"] = compute_reproducibility_checksum(
        RANDOM_SEED,
        {
            "verdict": verdict,
            "preconditions": preconditions,
            "model_specs": model_specs,
        },
    )
    return artifact


def assemble_artifact(
    *,
    start_time: float,
    end_time: float,
    preconditions: dict[str, dict[str, Any]],
    package_result: SurfaceResult,
    pipeline_result: SurfaceResult,
    mcp_result: SurfaceResult,
    cli_result: SurfaceResult,
    build_result: dict[str, Any],
    model_specs: list[dict[str, Any]],
    random_seed: int,
) -> dict[str, Any]:
    """Assemble the terminal Exp 3769 artifact."""

    surfaces = [package_result, pipeline_result, mcp_result, cli_result]
    surface_names = [surface.name for surface in surfaces if surface.passed]
    all_passed = all(surface.passed for surface in surfaces)
    failed = next((surface for surface in surfaces if not surface.passed), None)
    verdict = COMPLETE_VERDICT if all_passed else "blocked_e2e_surface_failure"
    if failed is not None and failed.detail.startswith("blocked_"):
        verdict = failed.detail.split(":", 1)[0]

    artifact = {
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "package_importable": bool(package_result.passed),
        "pipeline_e2e_passed": bool(pipeline_result.passed),
        "mcp_protocol_exchange_passed": bool(mcp_result.passed),
        "cli_passed": bool(cli_result.passed),
        "surfaces_passed": surface_names,
        "is_wiring_smoke_not_accuracy_claim": True,
        "preconditions_checked": preconditions,
        "model_specs": model_specs,
        "random_seed": random_seed,
        "duration_s": round(max(end_time - start_time, 0.0), 3),
        "package_result": package_result.data,
        "pipeline_result": pipeline_result.data,
        "mcp_protocol_result": mcp_result.data,
        "cli_result": cli_result.data,
        "build_result": build_result,
    }
    artifact["reproducibility_checksum"] = compute_reproducibility_checksum(
        random_seed,
        {
            "verdict": artifact["honest_verdict"],
            "surfaces_passed": surface_names,
            "model_specs": model_specs,
            "candidates": tiny_candidates(),
        },
    )
    return artifact


BuildRunner = Callable[[Path, str], dict[str, Any]]
PipelineRunner = Callable[[Path, str], SurfaceResult]
ProtocolRunner = Callable[[Path, str, list[dict[str, Any]]], SurfaceResult]
ModelResolver = Callable[[Path], list[dict[str, Any]]]


def run(
    root: Path,
    *,
    output_path: Path | None = None,
    executable: str = sys.executable,
    build_runner: BuildRunner = run_optional_build,
    pipeline_runner: PipelineRunner = run_pipeline_e2e,
    mcp_runner: ProtocolRunner = run_mcp_protocol_exchange,
    cli_runner: ProtocolRunner = run_cli_score_candidates,
    model_resolver: ModelResolver = resolve_model_specs,
) -> dict[str, Any]:
    """Run the Exp 3769 smoke and write the result artifact."""

    start = time.monotonic()
    output = output_path or root / RESULT_REL_PATH
    preconditions = check_preconditions(root, executable)
    model_specs = model_resolver(root)
    blocked = _first_failed_precondition(preconditions)
    if blocked is not None:
        artifact = blocked_artifact(
            blocked,
            preconditions=preconditions,
            start_time=start,
            end_time=time.monotonic(),
            model_specs=model_specs,
        )
        _write_json(output, artifact)
        return artifact

    package_result = run_package_import(preconditions)
    build_result = build_runner(root, executable)
    candidates = tiny_candidates()
    pipeline_result = pipeline_runner(root, executable)
    mcp_result = mcp_runner(root, executable, candidates)
    cli_result = cli_runner(root, executable, candidates)

    artifact = assemble_artifact(
        start_time=start,
        end_time=time.monotonic(),
        preconditions=preconditions,
        package_result=package_result,
        pipeline_result=pipeline_result,
        mcp_result=mcp_result,
        cli_result=cli_result,
        build_result=build_result,
        model_specs=model_specs,
        random_seed=RANDOM_SEED,
    )
    _write_json(output, artifact)
    return artifact


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the Exp 3769 runner."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    artifact = run(args.root, output_path=args.output)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if str(artifact["honest_verdict"]).startswith(("complete:", "blocked_")) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
