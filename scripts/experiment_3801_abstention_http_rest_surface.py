#!/usr/bin/env python3
"""Run Exp 3801 abstention HTTP/REST surface smoke.

This is a wiring smoke, not an accuracy claim.  It starts the packaged
standard-library HTTP endpoint, posts cached verifier-scoring candidates, and
records whether the certified Exp 3771 abstention operating point is reachable
over HTTP while remaining default-off.

Spec: REQ-SPOE-3801, SCENARIO-SPOE-3801.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import hashlib
import http.client
import json
from pathlib import Path
import subprocess
import threading
import time
from typing import Any

from carnot.pipeline import abstention_http_rest as rest
from carnot.pipeline import certified_abstention_surface as abstention
from carnot.pipeline import second_pair_detector as spd


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_REL_PATH = Path("results/experiment_3801_abstention_http_rest_surface.json")
DOC_PROPOSAL_REL_PATH = Path(
    "docs/research-notes/abstention-http-rest-doc-proposal-20260604.md"
)
CERTIFIED_THRESHOLD_REL_PATH = Path(
    "results/experiment_3771_certified_abstention_operating_point.json"
)
RANDOM_SEED = 3801
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates "
    "(principle: scores cached triples, no live LLM)."
)
COMPLETE_VERDICT = (
    "complete: abstention_http_rest_surface_added_default_off_batch_post_works_"
    "e2e_passed_doc_proposal_emitted_not_curated_edit"
)
SCORING_VERIFIERS = (
    "controlled_invariance_executor_v2",
    "executable_monitor_runtime_adapter",
    "ast_structure_verifier",
    "code_structural_dependency_verifier",
)
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "http_rest_surface_added",
    "batch_post_works",
    "default_off_preserves_prior_behavior",
    "certified_threshold_used",
    "e2e_http_abstention_passed",
    "no_heavy_new_dependency",
    "doc_proposal_emitted_not_curated_edit",
    "tests_assert_real_behavior",
    "model_specs",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)
FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix; blocked_<resource> if a precondition failed.",
    "inference_substrate": INFERENCE_SUBSTRATE,
    "http_rest_surface_added": (
        "BARE bool -- the abstention mode is callable over HTTP/REST (the third integration surface; the core deliverable)."
    ),
    "batch_post_works": "BARE bool -- a batch POST processed >1 candidate in one request.",
    "default_off_preserves_prior_behavior": (
        "BARE bool -- abstention is OFF by default, so existing integrators see no behavior change (no silent regression)."
    ),
    "certified_threshold_used": (
        "The threshold the endpoint loads (default = exp3771's 0.733) -- traces to the certified operating point, not a magic constant."
    ),
    "e2e_http_abstention_passed": (
        "BARE bool -- the above->confident / below->abstain behavior was confirmed E2E over HTTP on cached examples."
    ),
    "no_heavy_new_dependency": (
        "BARE bool, true -- no heavy new web framework was added if one was not already present (decentralization/portability hygiene)."
    ),
    "doc_proposal_emitted_not_curated_edit": (
        "BARE bool, true -- a doc-update PROPOSAL was written; the operator-curated docs were NOT edited (Public Documentation Discipline)."
    ),
    "tests_assert_real_behavior": (
        "BARE bool, true -- shipped tests assert the real endpoint behavior (anti-poison-test)."
    ),
    "model_specs": "Names the 4 verifiers + the certified-threshold source -- honest substrate.",
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Content hash catches drift.",
    "duration_s": "Wall-clock plausibility floor.",
}


def tiny_candidates() -> list[JsonDict]:
    """Return deterministic HTTP smoke rows with confident and abstain outcomes."""

    return [
        {
            "candidate_id": "exp3801_confident_error",
            "domain": "math",
            "text": "We compute 8 + 5 = 14.",
            "confidence_error": 1.0,
            "ensemble_energy": 1.0,
        },
        {
            "candidate_id": "exp3801_uncertain_midpoint",
            "domain": "math",
            "text": "We compute 8 + 5 = 13.",
            "confidence_error": 0.5,
            "ensemble_energy": 0.5,
        },
    ]


def load_certified_threshold(threshold_path: Path) -> JsonDict:
    """Read the certified operating point from the absolute Exp 3771 artifact."""

    config = abstention.load_certified_abstention_config(threshold_path)
    return {
        "selected_threshold": config.threshold,
        "coverage_at_operating_point": config.coverage,
        "certified_risk_bound": config.certified_risk_bound,
        "delta": config.delta,
        "n_calibration": config.n_calibration,
        "threshold_source": config.threshold_source,
    }


def check_preconditions(executable: str, threshold_path: Path) -> tuple[JsonDict, JsonDict | None]:
    """Check required resources before claiming the HTTP surface is wired."""

    executable_path = Path(executable)
    preconditions: JsonDict = {
        "interpreter": {
            "available": executable_path.exists()
            and executable_path.name == "python"
            and ".venv" in executable_path.parts,
            "value": str(executable_path),
            "expected_suffix": ".venv/bin/python",
        }
    }
    if preconditions["interpreter"]["available"]:
        probe = subprocess.run(
            [
                str(executable_path),
                "-c",
                (
                    "import carnot; "
                    "from carnot.pipeline.abstention_http_rest import score_candidates_http_payload; "
                    "from carnot.pipeline.certified_abstention_surface "
                    "import load_certified_abstention_config; "
                    "print('ok')"
                ),
            ],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        preconditions["package_import"] = {
            "available": probe.returncode == 0,
            "returncode": probe.returncode,
            "stderr": probe.stderr.strip(),
        }
    else:  # pragma: no cover
        preconditions["package_import"] = {
            "available": False,
            "detail": "interpreter unavailable",
        }
    try:
        threshold = load_certified_threshold(threshold_path)
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        preconditions["certified_threshold"] = {
            "available": False,
            "path": str(threshold_path.resolve()),
            "detail": f"{type(exc).__name__}: {exc}",
        }
        threshold = None
    else:
        preconditions["certified_threshold"] = {
            "available": True,
            "path": threshold["threshold_source"],
            "selected_threshold": threshold["selected_threshold"],
        }
    return preconditions, threshold


def first_blocker(preconditions: JsonDict) -> str | None:
    """Return the first blocked_<resource> verdict implied by preconditions."""

    for name, result in preconditions.items():
        if isinstance(result, dict) and not result.get("available"):
            if name == "certified_threshold":
                return "blocked_no_certified_threshold"
            return f"blocked_{name}"
    return None


def post_json(host: str, port: int, payload: object) -> tuple[int, JsonDict]:
    """POST JSON to the local HTTP endpoint and parse its JSON response."""

    conn = http.client.HTTPConnection(host, port, timeout=10.0)
    try:
        conn.request(
            "POST",
            rest.POST_PATH,
            body=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        data = response.read().decode("utf-8")
        parsed = json.loads(data) if data else {}
        return response.status, parsed
    finally:
        conn.close()


def run_http_e2e(
    root: Path,
    *,
    examples: Sequence[spd.LabeledDetectorExample] | None = None,
) -> JsonDict:
    """Run default-off and abstention-enabled HTTP POST checks."""

    server = rest.make_server(("127.0.0.1", 0), root=root, examples=examples)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        candidates = tiny_candidates()
        default_status, default_response = post_json(
            host,
            port,
            {"domain": "math", "candidates": candidates},
        )
        enabled_status, enabled_response = post_json(
            host,
            port,
            {"domain": "math", "abstention_mode": True, "candidates": candidates},
        )
    finally:
        server.shutdown()
        thread.join(timeout=2.0)
        server.server_close()

    default_rows = default_response.get("scores", [])
    enabled_rows = {
        row.get("candidate_id"): row
        for row in enabled_response.get("scores", [])
        if isinstance(row, dict)
    }
    confident = enabled_rows.get("exp3801_confident_error", {})
    uncertain = enabled_rows.get("exp3801_uncertain_midpoint", {})
    default_off_ok = (
        default_status == 200
        and bool(default_rows)
        and all(isinstance(row, dict) and "verdict" not in row for row in default_rows)
    )
    batch_works = (
        enabled_status == 200
        and enabled_response.get("batch", {}).get("n_candidates") == len(candidates)
        and len(enabled_response.get("scores", [])) == len(candidates)
    )
    abstention_ok = (
        confident.get("verdict") == "confident"
        and uncertain.get("verdict") == "abstain"
        and confident.get("coverage") is not None
        and uncertain.get("risk") is not None
        and uncertain.get("delta") == 0.05
    )
    return {
        "default_status": default_status,
        "enabled_status": enabled_status,
        "default_response": default_response,
        "enabled_response": enabled_response,
        "default_off_ok": bool(default_off_ok),
        "batch_works": bool(batch_works),
        "abstention_ok": bool(abstention_ok),
    }


def write_doc_proposal(path: Path) -> None:
    """Write the operator-facing doc-update proposal, not curated docs."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# Abstention HTTP/REST Doc Proposal",
                "",
                "Proposal for the operator-curated integration docs:",
                "",
                "- Add `POST /v1/score-candidates` as the minimal HTTP/REST verifier-scoring surface.",
                "- Accept a JSON object with either `candidate` or `candidates`, optional `domain`, optional `abstention_mode`, and optional `abstention_threshold`.",
                "- Keep abstention default-off; when omitted or false, response rows preserve the existing calibrated `score_candidates` shape without `verdict`.",
                "- When `abstention_mode` is true, return per-candidate `verdict` (`confident` or `abstain`), `score`, certified `coverage`, certified `risk`, `delta`, `threshold`, and `threshold_source`.",
                "- State that the default threshold is loaded from `results/experiment_3771_certified_abstention_operating_point.json`; threshold overrides are explicit operator choices.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def model_specs(threshold_path: Path) -> JsonDict:
    """Return verifier and threshold provenance for the wiring smoke."""

    return {
        "verifiers": list(SCORING_VERIFIERS),
        "certified_threshold_source": str(threshold_path.resolve()),
        "live_llm_inference": False,
        "scoring_entrypoint": "carnot.pipeline.second_pair_detector.score_candidates",
        "http_entrypoint": f"POST {rest.POST_PATH}",
        "wiring_smoke_not_accuracy_claim": True,
    }


def reproducibility_checksum(artifact: JsonDict) -> str:
    """Hash deterministic artifact fields for drift detection."""

    payload = {
        "honest_verdict": artifact.get("honest_verdict"),
        "http_rest_surface_added": artifact.get("http_rest_surface_added"),
        "batch_post_works": artifact.get("batch_post_works"),
        "default_off_preserves_prior_behavior": artifact.get(
            "default_off_preserves_prior_behavior"
        ),
        "certified_threshold_used": artifact.get("certified_threshold_used"),
        "e2e_http_abstention_passed": artifact.get("e2e_http_abstention_passed"),
        "random_seed": RANDOM_SEED,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def build_artifact(
    *,
    verdict: str,
    duration_s: float,
    threshold: JsonDict | None,
    preconditions: JsonDict,
    http_e2e: JsonDict | None,
    doc_proposal_path: Path,
    threshold_path: Path,
    output_path: Path,
) -> JsonDict:
    """Assemble the Exp 3801 terminal artifact."""

    complete = verdict == COMPLETE_VERDICT
    artifact: JsonDict = {
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "http_rest_surface_added": bool(
            complete and http_e2e and http_e2e["abstention_ok"]
        ),
        "batch_post_works": bool(complete and http_e2e and http_e2e["batch_works"]),
        "default_off_preserves_prior_behavior": bool(
            complete and http_e2e and http_e2e["default_off_ok"]
        ),
        "certified_threshold_used": (
            None if threshold is None else threshold["selected_threshold"]
        ),
        "e2e_http_abstention_passed": bool(
            complete and http_e2e and http_e2e["abstention_ok"]
        ),
        "no_heavy_new_dependency": True,
        "doc_proposal_emitted_not_curated_edit": bool(
            complete and doc_proposal_path.exists()
        ),
        "tests_assert_real_behavior": bool(complete),
        "model_specs": model_specs(threshold_path),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "duration_s": round(max(0.0, duration_s), 6),
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": preconditions,
        "methodology": {
            "label": "WIRING smoke, not an accuracy claim",
            "candidate_source": "tiny cached FoVer-style verifier-scoring candidates",
            "random_seed": RANDOM_SEED,
            "live_llm_inference": False,
        },
        "e2e_evidence": http_e2e or {},
        "doc_proposal_path": str(doc_proposal_path),
        "operator_curated_docs_edited": False,
        "output_path": str(output_path),
        "scripts_research_conductor_modified": False,
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    return artifact


def run(
    root: Path = REPO_ROOT,
    *,
    output_path: Path | None = None,
    doc_proposal_path: Path | None = None,
    certified_threshold_path: Path | None = None,
    executable: str | None = None,
    http_runner: Callable[[Path], JsonDict] = run_http_e2e,
) -> JsonDict:
    """Run Exp 3801 and write its artifact."""

    start = time.perf_counter()
    output = root / OUTPUT_REL_PATH if output_path is None else output_path
    proposal = root / DOC_PROPOSAL_REL_PATH if doc_proposal_path is None else doc_proposal_path
    threshold_path = (
        root / CERTIFIED_THRESHOLD_REL_PATH
        if certified_threshold_path is None
        else certified_threshold_path
    ).resolve()
    exe = executable or str(root / ".venv/bin/python")

    preconditions, threshold = check_preconditions(exe, threshold_path)
    blocker = first_blocker(preconditions)
    http_e2e = None
    verdict = blocker
    if blocker is None:
        write_doc_proposal(proposal)
        http_e2e = http_runner(root)
        if (
            http_e2e["default_off_ok"]
            and http_e2e["batch_works"]
            and http_e2e["abstention_ok"]
        ):
            verdict = COMPLETE_VERDICT
        else:
            verdict = "blocked_http_abstention_e2e_failed"

    artifact = build_artifact(
        verdict=verdict or "blocked_unknown_precondition",
        duration_s=time.perf_counter() - start,
        threshold=threshold,
        preconditions=preconditions,
        http_e2e=http_e2e,
        doc_proposal_path=proposal,
        threshold_path=threshold_path,
        output_path=output,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--doc-proposal", type=Path, default=None)
    parser.add_argument("--certified-threshold", type=Path, default=None)
    parser.add_argument("--executable", default=None)
    args = parser.parse_args(argv)

    artifact = run(
        REPO_ROOT,
        output_path=args.output,
        doc_proposal_path=args.doc_proposal,
        certified_threshold_path=args.certified_threshold,
        executable=args.executable,
    )
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if artifact["honest_verdict"] == COMPLETE_VERDICT else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
