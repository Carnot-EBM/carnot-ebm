#!/usr/bin/env python3
"""Run Exp 3789 abstention CLI batch surface smoke.

This is a wiring smoke, not an accuracy claim.  It calls the packaged CLI over
cached verifier-scoring candidates and records whether the certified Exp 3771
abstention operating point is reachable through the batch CLI surface.

Spec: REQ-SPOE-3789, SCENARIO-SPOE-3789.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_REL_PATH = Path("results/experiment_3789_abstention_cli_batch_surface.json")
DOC_PROPOSAL_REL_PATH = Path(
    "docs/research-notes/abstention-cli-doc-proposal-20260604.md"
)
CERTIFIED_THRESHOLD_REL_PATH = Path(
    "results/experiment_3771_certified_abstention_operating_point.json"
)
RANDOM_SEED = 3789
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates "
    "(principle: scores cached triples, no live LLM)."
)
COMPLETE_VERDICT = (
    "complete: abstention_cli_batch_surface_added_default_off_e2e_passed_"
    "doc_proposal_emitted_not_curated_edit"
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
    "cli_abstention_surface_added",
    "batch_path_works",
    "default_off_preserves_prior_behavior",
    "certified_threshold_used",
    "e2e_cli_abstention_passed",
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
        "verifier_ensemble_against_cached_candidates "
        "(principle: scores cached triples, no live LLM)."
    ),
    "cli_abstention_surface_added": (
        "BARE bool -- the abstention mode is callable from the CLI."
    ),
    "batch_path_works": (
        "BARE bool -- the CLI processed >1 candidate in one invocation."
    ),
    "default_off_preserves_prior_behavior": (
        "BARE bool -- the new flag is OFF by default."
    ),
    "certified_threshold_used": (
        "Threshold loaded by the CLI default from Exp 3771's certified operating point."
    ),
    "e2e_cli_abstention_passed": (
        "BARE bool -- above->confident / below->abstain confirmed through the CLI."
    ),
    "doc_proposal_emitted_not_curated_edit": (
        "BARE bool -- a proposal was written and curated docs were not edited."
    ),
    "tests_assert_real_behavior": (
        "BARE bool -- tests assert the real CLI/batch behavior."
    ),
    "model_specs": "Names the verifiers and certified-threshold source.",
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Content hash catches drift.",
    "duration_s": "Wall-clock plausibility floor.",
}


def tiny_candidates() -> list[JsonDict]:
    """Return deterministic CLI smoke rows with confident and abstain outcomes."""

    return [
        {
            "candidate_id": "exp3789_confident_error",
            "domain": "math",
            "text": "We compute 8 + 5 = 14.",
            "confidence_error": 1.0,
            "ensemble_energy": 1.0,
        },
        {
            "candidate_id": "exp3789_uncertain_midpoint",
            "domain": "math",
            "text": "We compute 8 + 5 = 13.",
            "confidence_error": 0.5,
            "ensemble_energy": 0.7,
        },
    ]


def load_certified_threshold(threshold_path: Path) -> JsonDict:
    """Read the certified operating point from the absolute Exp 3771 artifact."""

    payload = json.loads(threshold_path.read_text(encoding="utf-8"))
    return {
        "selected_threshold": float(payload["selected_threshold"]),
        "coverage_at_operating_point": float(payload["coverage_at_operating_point"]),
        "certified_risk_bound": float(payload["certified_risk_bound"]),
        "n_calibration": int(payload["n_calibration"]),
        "threshold_source": str(threshold_path.resolve()),
    }


def check_preconditions(executable: str, threshold_path: Path) -> tuple[JsonDict, JsonDict | None]:
    """Check required resources before claiming any CLI surface pass."""

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
                    "from carnot.cli import cmd_verify_batch; "
                    "from carnot.pipeline.certified_abstention_surface "
                    "import load_certified_abstention_config; "
                    "print('ok')"
                ),
            ],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        preconditions["package_import"] = {
            "available": probe.returncode == 0,
            "returncode": probe.returncode,
            "stderr": probe.stderr.strip(),
        }
    else:
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


def run_cli_json(executable: str, args: list[str]) -> JsonDict:
    """Run a packaged CLI command and parse its JSON stdout."""

    completed = subprocess.run(
        [executable, "-m", "carnot.cli", *args],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"CLI failed rc={completed.returncode}: {completed.stderr.strip()}"
        )
    return json.loads(completed.stdout)


def run_cli_e2e(executable: str) -> JsonDict:
    """Run default-off and abstention-enabled CLI batch checks."""

    candidates = tiny_candidates()
    with tempfile.TemporaryDirectory(prefix="carnot-exp3789-") as tmp:
        tmp_path = Path(tmp)
        json_array = tmp_path / "candidates.json"
        jsonl = tmp_path / "candidates.jsonl"
        json_array.write_text(json.dumps(candidates), encoding="utf-8")
        jsonl.write_text(
            "\n".join(json.dumps(candidate) for candidate in candidates) + "\n",
            encoding="utf-8",
        )

        default_batch = run_cli_json(
            executable,
            [
                "verify-batch",
                "--candidates-file",
                str(json_array),
                "--domain",
                "math",
            ],
        )
        prior_score_candidates = run_cli_json(
            executable,
            [
                "score-candidates",
                "--candidates-file",
                str(json_array),
                "--domain",
                "math",
            ],
        )
        enabled_batch = run_cli_json(
            executable,
            [
                "verify-batch",
                "--candidates-file",
                str(jsonl),
                "--domain",
                "math",
                "--abstention-mode",
            ],
        )

    default_rows = default_batch.get("scores", [])
    prior_rows = prior_score_candidates.get("scores", [])
    enabled_rows = {
        row.get("candidate_id"): row
        for row in enabled_batch.get("scores", [])
        if isinstance(row, dict)
    }
    confident = enabled_rows.get("exp3789_confident_error", {})
    uncertain = enabled_rows.get("exp3789_uncertain_midpoint", {})
    default_off_ok = bool(default_rows) and bool(prior_rows) and all(
        isinstance(row, dict) and "abstention_verdict" not in row
        for row in [*default_rows, *prior_rows]
    )
    batch_works = (
        enabled_batch.get("batch", {}).get("n_candidates") == len(candidates)
        and len(enabled_batch.get("scores", [])) == len(candidates)
    )
    abstention_ok = (
        enabled_batch.get("cli_surface") == "verify-batch"
        and enabled_batch.get("abstention_mode", {}).get("enabled") is True
        and confident.get("abstention_verdict") == "confident_error"
        and confident.get("route_to_review") is False
        and uncertain.get("abstention_verdict") == "uncertain / route to review"
        and uncertain.get("route_to_review") is True
        and uncertain.get("certified_abstention", {}).get("delta") == 0.05
        and isinstance(
            uncertain.get("certified_abstention", {}).get("threshold_source"),
            str,
        )
    )
    return {
        "default_batch": default_batch,
        "prior_score_candidates": prior_score_candidates,
        "enabled_batch": enabled_batch,
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
                "# Abstention CLI Batch Doc Proposal",
                "",
                "Proposal for the operator-curated CLI docs:",
                "",
                "- Add `carnot verify-batch --candidates-file <path> --domain math` as the batch verifier-scoring CLI surface.",
                "- State that `<path>` may be a JSON array, JSONL candidate objects, or one raw candidate text per non-empty line.",
                "- Document that abstention remains default-off; add `--abstention-mode` to emit `abstention_verdict`, `route_to_review`, and `certified_abstention` metadata.",
                "- Note that the default threshold is loaded from `results/experiment_3771_certified_abstention_operating_point.json`; `--abstention-threshold` is an explicit operator override.",
                "- Include a two-row example showing one `confident_error` row and one `uncertain / route to review` row.",
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
        "cli_entrypoint": "python -m carnot.cli verify-batch",
        "wiring_smoke_not_accuracy_claim": True,
    }


def reproducibility_checksum(artifact: JsonDict) -> str:
    """Hash deterministic artifact fields for drift detection."""

    payload = {
        "honest_verdict": artifact.get("honest_verdict"),
        "cli_abstention_surface_added": artifact.get("cli_abstention_surface_added"),
        "batch_path_works": artifact.get("batch_path_works"),
        "default_off_preserves_prior_behavior": artifact.get(
            "default_off_preserves_prior_behavior"
        ),
        "certified_threshold_used": artifact.get("certified_threshold_used"),
        "e2e_cli_abstention_passed": artifact.get("e2e_cli_abstention_passed"),
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
    cli_e2e: JsonDict | None,
    doc_proposal_path: Path,
    threshold_path: Path,
    output_path: Path,
) -> JsonDict:
    """Assemble the Exp 3789 terminal artifact."""

    complete = verdict == COMPLETE_VERDICT
    artifact: JsonDict = {
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "cli_abstention_surface_added": bool(
            complete and cli_e2e and cli_e2e["abstention_ok"]
        ),
        "batch_path_works": bool(complete and cli_e2e and cli_e2e["batch_works"]),
        "default_off_preserves_prior_behavior": bool(
            complete and cli_e2e and cli_e2e["default_off_ok"]
        ),
        "certified_threshold_used": (
            None if threshold is None else threshold["selected_threshold"]
        ),
        "e2e_cli_abstention_passed": bool(
            complete and cli_e2e and cli_e2e["abstention_ok"]
        ),
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
        "e2e_evidence": cli_e2e or {},
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
) -> JsonDict:
    """Run Exp 3789 and write its artifact."""

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
    cli_e2e = None
    verdict = blocker
    if blocker is None:
        write_doc_proposal(proposal)
        cli_e2e = run_cli_e2e(exe)
        if (
            cli_e2e["default_off_ok"]
            and cli_e2e["batch_works"]
            and cli_e2e["abstention_ok"]
        ):
            verdict = COMPLETE_VERDICT
        else:
            verdict = "blocked_cli_abstention_e2e_failed"

    artifact = build_artifact(
        verdict=verdict or "blocked_unknown_precondition",
        duration_s=time.perf_counter() - start,
        threshold=threshold,
        preconditions=preconditions,
        cli_e2e=cli_e2e,
        doc_proposal_path=proposal,
        threshold_path=threshold_path,
        output_path=output,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main(argv: list[str] | None = None) -> int:
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


if __name__ == "__main__":
    raise SystemExit(main())
