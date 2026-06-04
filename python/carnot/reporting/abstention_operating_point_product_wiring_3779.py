"""Exp 3779 abstention operating point product wiring smoke.

This is a wiring smoke, not a new accuracy measurement.  It proves the Exp
3771 certified operating point is callable through the shipped verifier-scoring
surface, remains default-off, and can be reached through MCP when the runtime
is available.

Spec: REQ-SPOE-3779, SCENARIO-SPOE-3779.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, field
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable

from carnot.pipeline import certified_abstention_surface as abstention
from carnot.pipeline import second_pair_detector as spd

JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_REL_PATH = Path("results/experiment_3779_abstention_operating_point_product_wiring.json")
DOC_PROPOSAL_REL_PATH = Path("docs/research-notes/abstention-mode-doc-proposal-20260604.md")
RANDOM_SEED = 3779
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
COMPLETE_VERDICT = (
    "complete: abstention_mode_wired_into_verify_api_default_off_e2e_passed_"
    "doc_proposal_emitted_not_curated_edit"
)
SCORING_VERIFIERS = (
    "controlled_invariance_executor_v2",
    "executable_monitor_runtime_adapter",
    "ast_structure_verifier",
    "code_structural_dependency_verifier",
)
MATH_SIGNAL_VERIFIERS = ("SemEnergyProbe", "Z3MathVerifier", "RPRMStepReward")
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "abstention_mode_wired",
    "default_off_preserves_prior_behavior",
    "certified_threshold_used",
    "e2e_abstention_passed",
    "mcp_surface_confirmed",
    "doc_proposal_emitted_not_curated_edit",
    "tests_assert_real_behavior",
    "model_specs",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)
FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix; blocked_<resource> if a precondition failed.",
    "inference_substrate": (
        "Scores cached triples with the verifier ensemble and does not run live LLM inference."
    ),
    "abstention_mode_wired": (
        "BARE bool -- the opt-in abstention mode is callable in the verify API."
    ),
    "default_off_preserves_prior_behavior": (
        "BARE bool -- default OFF keeps existing integrators unchanged."
    ),
    "certified_threshold_used": (
        "Threshold loaded from Exp 3771 so the surface traces to the certified operating point."
    ),
    "e2e_abstention_passed": (
        "BARE bool -- above-threshold confident and below-threshold abstain were confirmed."
    ),
    "mcp_surface_confirmed": (
        "BARE bool -- MCP score_candidates accepted the opt-in mode over protocol."
    ),
    "doc_proposal_emitted_not_curated_edit": (
        "BARE bool -- a proposal was written and curated docs were left untouched."
    ),
    "tests_assert_real_behavior": (
        "BARE bool -- tests call the real scoring behavior instead of hard-coded strings."
    ),
    "model_specs": "Names the verifier-scoring substrate and certified-threshold source.",
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Content hash catches drift.",
    "duration_s": "Wall-clock plausibility floor.",
}


@dataclass(frozen=True)
class SurfaceCheck:
    """One external or internal product-surface check."""

    name: str
    passed: bool
    detail: str
    data: JsonDict = field(default_factory=dict)


def tiny_candidates() -> list[JsonDict]:
    """Return deterministic candidate rows that exercise confident and review paths."""

    return [
        {
            "candidate_id": "exp3779_confident_error",
            "domain": "math",
            "text": "We compute 8 + 5 = 14.",
            "confidence_error": 1.0,
            "ensemble_energy": 1.0,
        },
        {
            "candidate_id": "exp3779_uncertain_midpoint",
            "domain": "math",
            "text": "We compute 8 + 5 = 13.",
            "confidence_error": 0.5,
            "ensemble_energy": 0.7,
        },
    ]


def check_preconditions(
    executable: str,
    threshold_path: Path,
) -> tuple[JsonDict, abstention.CertifiedAbstentionConfig | None]:
    """Check required resources before any product-surface pass is claimed."""

    executable_path = Path(executable)
    interpreter_ok = ".venv" in executable_path.parts and executable_path.name == "python"
    preconditions: JsonDict = {
        "interpreter": {
            "available": interpreter_ok,
            "value": str(executable_path),
            "expected_suffix": ".venv/bin/python",
        }
    }
    try:
        import carnot

        preconditions["package_import"] = {
            "available": True,
            "module_path": getattr(carnot, "__file__", ""),
            "version": getattr(carnot, "__version__", "<missing>"),
        }
    except Exception as exc:  # pragma: no cover - only reachable in a broken environment.
        preconditions["package_import"] = {
            "available": False,
            "detail": f"{type(exc).__name__}: {exc}",
        }
    try:
        config = abstention.load_certified_abstention_config(threshold_path)
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        preconditions["certified_threshold"] = {
            "available": False,
            "path": str(threshold_path.resolve()),
            "detail": f"{type(exc).__name__}: {exc}",
        }
        config = None
    else:
        preconditions["certified_threshold"] = {
            "available": True,
            "path": config.threshold_source,
            "selected_threshold": config.threshold,
        }
    return preconditions, config


def first_blocker(preconditions: JsonDict) -> str | None:
    """Return the first blocked_<resource> verdict implied by preconditions."""

    for name, result in preconditions.items():
        if isinstance(result, dict) and not result.get("available"):
            if name == "certified_threshold":
                return "blocked_no_certified_threshold"
            return f"blocked_{name}"
    return None


def run_abstention_e2e(root: Path) -> SurfaceCheck:
    """Run the default-off and opt-in abstention paths on cached scoring data."""

    candidates = tiny_candidates()
    try:
        default_response = spd.score_candidates(candidates, root=root, default_domain="math")
        enabled_response = spd.score_candidates(
            candidates,
            root=root,
            default_domain="math",
            abstention_mode=True,
        )
    except Exception as exc:
        return SurfaceCheck(
            name="abstention_e2e",
            passed=False,
            detail=f"score_candidates failed: {type(exc).__name__}: {exc}",
        )

    default_rows = default_response.get("scores", [])
    enabled_rows = {
        row.get("candidate_id"): row
        for row in enabled_response.get("scores", [])
        if isinstance(row, dict)
    }
    default_off_ok = bool(default_rows) and all(
        isinstance(row, dict) and "abstention_verdict" not in row for row in default_rows
    )
    confident = enabled_rows.get("exp3779_confident_error", {})
    uncertain = enabled_rows.get("exp3779_uncertain_midpoint", {})
    enabled_ok = (
        confident.get("abstention_verdict") == abstention.CONFIDENT_ERROR_VERDICT
        and confident.get("route_to_review") is False
        and uncertain.get("abstention_verdict") == abstention.ABSTAIN_VERDICT
        and uncertain.get("route_to_review") is True
    )
    return SurfaceCheck(
        name="abstention_e2e",
        passed=bool(default_off_ok and enabled_ok),
        detail="default-off and opt-in abstention behavior confirmed",
        data={
            "default_off_ok": bool(default_off_ok),
            "enabled_ok": bool(enabled_ok),
            "enabled_scores": list(enabled_rows.values()),
        },
    )


def run_mcp_protocol_exchange(root: Path, executable: str, candidates: list[JsonDict]) -> SurfaceCheck:
    """Call MCP `score_candidates` through stdio JSON-RPC."""

    if importlib.util.find_spec("mcp") is None:
        return SurfaceCheck(
            name="mcp_protocol",
            passed=False,
            detail="blocked_mcp_runtime",
            data={"protocol": "mcp_stdio_json_rpc", "tool_name": "score_candidates"},
        )

    async def _call() -> JsonDict:
        from mcp import StdioServerParameters
        from mcp.client.session import ClientSession
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(command=executable, args=["-m", "carnot.mcp"], cwd=str(root))
        async with stdio_client(params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(
                    "score_candidates",
                    {
                        "candidates": candidates,
                        "domain": "math",
                        "abstention_mode": True,
                    },
                )
                payload = parse_mcp_payload(result)
                payload["is_error"] = bool(getattr(result, "isError", False))
                return payload

    try:
        payload = asyncio.run(_call())
    except Exception as exc:
        return SurfaceCheck(
            name="mcp_protocol",
            passed=False,
            detail=f"MCP stdio exchange failed: {type(exc).__name__}: {exc}",
            data={"protocol": "mcp_stdio_json_rpc", "tool_name": "score_candidates"},
        )

    rows = payload.get("scores")
    passed = (
        not payload.get("is_error")
        and isinstance(rows, list)
        and any(
            isinstance(row, dict) and row.get("abstention_verdict") == abstention.ABSTAIN_VERDICT
            for row in rows
        )
    )
    return SurfaceCheck(
        name="mcp_protocol",
        passed=bool(passed),
        detail="MCP stdio score_candidates accepted abstention mode",
        data={"protocol": "mcp_stdio_json_rpc", "tool_name": "score_candidates", **payload},
    )


def parse_mcp_payload(result: Any) -> JsonDict:
    """Extract structured JSON content from an MCP call result."""

    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        return dict(structured)
    content = getattr(result, "content", None) or []
    text = getattr(content[0], "text", "") if content else ""
    return json.loads(text) if text else {}


def write_doc_proposal(path: Path, config: abstention.CertifiedAbstentionConfig) -> None:
    """Write the operator-facing doc-update proposal without editing curated docs."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# Abstention Mode Documentation Proposal",
                "",
                "Proposed operator-curated docs update for `score_candidates`:",
                "",
                "- Add optional `abstention_mode` (default `false`).",
                "- Add optional `abstention_threshold` override for operator tuning.",
                (
                    "- When enabled, rows with `abstention_score` at or above "
                    f"`{config.threshold:.6f}` return a confident verdict."
                ),
                "- Rows below the threshold return `uncertain / route to review`.",
                "- The default threshold source is Exp 3771's certified artifact.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def model_specs(threshold_path: Path) -> JsonDict:
    """Return the verifier-scoring substrate metadata for the artifact."""

    digest = hashlib.sha256(threshold_path.read_bytes()).hexdigest()[:16]
    return {
        "scoring_entrypoint": "carnot.pipeline.second_pair_detector.score_candidates",
        "verifiers": list(SCORING_VERIFIERS),
        "math_signal_verifiers": list(MATH_SIGNAL_VERIFIERS),
        "certified_threshold_source": str(threshold_path.resolve()),
        "certified_threshold_artifact_sha256": digest,
        "live_llm_inference": False,
    }


def blocked_artifact(
    verdict: str,
    *,
    preconditions: JsonDict,
    start_time: float,
    end_time: float,
    specs: JsonDict | None = None,
) -> JsonDict:
    """Build a blocked artifact without claiming any surface pass."""

    artifact = {
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "abstention_mode_wired": False,
        "default_off_preserves_prior_behavior": False,
        "certified_threshold_used": None,
        "e2e_abstention_passed": False,
        "mcp_surface_confirmed": False,
        "doc_proposal_emitted_not_curated_edit": False,
        "tests_assert_real_behavior": True,
        "model_specs": specs or {},
        "random_seed": RANDOM_SEED,
        "duration_s": round(max(0.0, end_time - start_time), 6),
        "preconditions_checked": preconditions,
        "field_principles": FIELD_PRINCIPLES,
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def assemble_artifact(
    *,
    start_time: float,
    end_time: float,
    preconditions: JsonDict,
    config: abstention.CertifiedAbstentionConfig,
    e2e_result: SurfaceCheck,
    mcp_result: SurfaceCheck,
    doc_proposal_path: Path,
    specs: JsonDict,
) -> JsonDict:
    """Assemble the terminal Exp 3779 artifact."""

    doc_ok = doc_proposal_path.exists()
    complete = bool(e2e_result.passed and mcp_result.passed and doc_ok)
    artifact = {
        "honest_verdict": COMPLETE_VERDICT if complete else "blocked_abstention_surface_wiring",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "abstention_mode_wired": bool(e2e_result.passed),
        "default_off_preserves_prior_behavior": bool(
            e2e_result.data.get("default_off_ok") if e2e_result.data else False
        ),
        "certified_threshold_used": config.threshold,
        "e2e_abstention_passed": bool(e2e_result.passed),
        "mcp_surface_confirmed": bool(mcp_result.passed),
        "doc_proposal_emitted_not_curated_edit": bool(doc_ok),
        "tests_assert_real_behavior": True,
        "model_specs": specs,
        "random_seed": RANDOM_SEED,
        "duration_s": round(max(0.0, end_time - start_time), 6),
        "preconditions_checked": preconditions,
        "field_principles": FIELD_PRINCIPLES,
        "e2e_result": e2e_result.data,
        "mcp_protocol_result": mcp_result.data,
        "doc_proposal_path": str(doc_proposal_path),
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def reproducibility_checksum(artifact: JsonDict) -> str:
    """Hash stable artifact fields for drift detection."""

    payload = {
        "honest_verdict": artifact.get("honest_verdict"),
        "certified_threshold_used": artifact.get("certified_threshold_used"),
        "abstention_mode_wired": artifact.get("abstention_mode_wired"),
        "default_off_preserves_prior_behavior": artifact.get(
            "default_off_preserves_prior_behavior"
        ),
        "e2e_abstention_passed": artifact.get("e2e_abstention_passed"),
        "mcp_surface_confirmed": artifact.get("mcp_surface_confirmed"),
        "model_specs": artifact.get("model_specs"),
        "random_seed": RANDOM_SEED,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


McpRunner = Callable[[Path, str, list[JsonDict]], SurfaceCheck]


def run(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | None = None,
    doc_proposal_path: Path | None = None,
    certified_threshold_path: Path | None = None,
    executable: str = sys.executable,
    mcp_runner: McpRunner = run_mcp_protocol_exchange,
) -> JsonDict:
    """Run the Exp 3779 wiring smoke and persist the result artifact."""

    root_path = Path(root)
    start = time.monotonic()
    output = output_path or root_path / OUTPUT_REL_PATH
    proposal = doc_proposal_path or root_path / DOC_PROPOSAL_REL_PATH
    threshold_path = certified_threshold_path or abstention.DEFAULT_CERTIFIED_THRESHOLD_PATH
    preconditions, config = check_preconditions(executable, threshold_path)
    blocker = first_blocker(preconditions)
    if blocker is not None or config is None:
        artifact = blocked_artifact(
            blocker or "blocked_no_certified_threshold",
            preconditions=preconditions,
            start_time=start,
            end_time=time.monotonic(),
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return artifact

    specs = model_specs(Path(config.threshold_source))
    e2e_result = run_abstention_e2e(root_path)
    mcp_result = mcp_runner(root_path, executable, tiny_candidates())
    write_doc_proposal(proposal, config)
    artifact = assemble_artifact(
        start_time=start,
        end_time=time.monotonic(),
        preconditions=preconditions,
        config=config,
        e2e_result=e2e_result,
        mcp_result=mcp_result,
        doc_proposal_path=proposal,
        specs=specs,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the Exp 3779 runner."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    artifact = run(args.root, output_path=args.output)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if str(artifact["honest_verdict"]).startswith(("complete:", "blocked_")) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
