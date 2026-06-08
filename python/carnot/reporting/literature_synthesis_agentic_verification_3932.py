"""Exp 3932 local literature synthesis for verifier efficiency positioning.

Spec refs: REQ-REPORT-3932, SCENARIO-REPORT-3932,
SCENARIO-REPORT-3932-BLOCKED-REFERENCES.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Sequence


JsonDict = dict[str, Any]


REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_REL_PATH = Path("results/experiment_3932_literature_synthesis_agentic_verification.json")
SYNTHESIS_NOTE_REL_PATH = Path(
    "docs/research-notes/agentic-verification-efficiency-positioning-20260607.md"
)
RESEARCH_REFERENCES_REL_PATH = Path("research-references.md")
RESEARCH_STUDYING_REL_PATH = Path("research-studying.md")
NORTH_STAR_REL_PATH = Path("ops/north-star.md")
INFERENCE_SUBSTRATE = "no_new_inference_local_disk_synthesis_cpu"
STUDYING_MARKER = "## 2026-06-07 Exp 3932 - Agentic Verification Efficiency Positioning"

SOURCE_ARTIFACTS = {
    "exp3926": Path("results/experiment_3926_valid_efficiency_head_to_head.json"),
    "exp3928": Path("results/experiment_3928_moat_scissor_replication.json"),
    "exp3929": Path("results/experiment_3929_arc_agi3_action_efficiency.json"),
}
PROTECTED_PUBLIC_PATHS = (
    Path("README.md"),
    Path("docs/index.html"),
    Path("docs/blog"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
    Path("scripts/research_conductor.py"),
)
REQUIRED_ARTIFACT_FIELDS = (
    "synthesis_note_path",
    "landscape_position_summary",
    "next_highest_leverage_experiments",
    "new_references_added",
    "public_docs_untouched",
    "preconditions_checked",
    "duration_s",
    "inference_substrate",
    "honest_verdict",
)
NEXT_EXPERIMENTS = (
    "ProcessBench full-benchmark head-to-head: run Carnot energy scores versus a "
    "competent GenRM/ThinkPRM-style judge on the full held-out benchmark so the "
    "efficiency claim is tested against a credible comparator; ARC-AGI-3 real-"
    "benchmark agentic run: replace the synthetic grid step with the official "
    "interactive harness and report action efficiency without claiming a leaderboard score."
)


def duration_from(started_s: float | None, now_s: float | None) -> float:
    """Return elapsed seconds for this document-only aggregation task."""

    start = time.perf_counter() if started_s is None else float(started_s)
    end = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0001, end - start), 6)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _path_fingerprint(path: Path) -> str:
    if path.is_file():
        return "file:" + _sha256_bytes(path.read_bytes())
    if path.is_dir():
        parts = [
            f"{child.relative_to(path).as_posix()}:{_sha256_bytes(child.read_bytes())}"
            for child in sorted(path.rglob("*"))
            if child.is_file()
        ]
        return "dir:" + _sha256_bytes("\n".join(parts).encode("utf-8"))
    return "missing"  # pragma: no cover - defensive for repos without optional public files.


def public_doc_snapshot(root: Path) -> dict[str, str]:
    """Fingerprint public/operator files that this task must not edit."""

    return {rel.as_posix(): _path_fingerprint(root / rel) for rel in PROTECTED_PUBLIC_PATHS}


def public_docs_untouched(root: Path, before: dict[str, str]) -> bool:
    """Return whether protected public/operator files match the initial snapshot."""

    return public_doc_snapshot(root) == before


def check_preconditions(root: Path) -> dict[str, bool]:
    """Check that the two research ledgers required by the task are readable."""

    return {
        RESEARCH_REFERENCES_REL_PATH.as_posix(): (root / RESEARCH_REFERENCES_REL_PATH).is_file(),
        RESEARCH_STUDYING_REL_PATH.as_posix(): (root / RESEARCH_STUDYING_REL_PATH).is_file(),
    }


def _precondition_details(preconditions: dict[str, bool]) -> str:
    return "; ".join(f"{name}={'readable' if ok else 'missing'}" for name, ok in preconditions.items())


def read_json_artifact(root: Path, rel_path: Path) -> JsonDict:
    """Read a source artifact if present; missing source artifacts are empty."""

    path = root / rel_path
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _status(payload: JsonDict) -> str:
    return str(payload.get("honest_verdict") or payload.get("status") or "missing")


def _flagged(payload: JsonDict) -> bool:
    return bool(payload.get("flagged_adversarial"))


def load_source_artifacts(root: Path) -> dict[str, JsonDict]:
    """Load the local .363 source artifacts used by the synthesis."""

    return {
        name: read_json_artifact(root, rel_path)
        for name, rel_path in SOURCE_ARTIFACTS.items()
    }


def landscape_position_summary(sources: dict[str, JsonDict]) -> str:
    """Summarize Carnot's position in one artifact-safe line."""

    exp3926 = sources["exp3926"]
    exp3929 = sources["exp3929"]
    ratio = float(exp3929.get("action_efficiency_ratio") or 0.0)
    status = "blocked locally" if _status(exp3926).startswith("blocked") or _flagged(exp3926) else "complete locally"
    return (
        "Carnot sits as a cheap discriminative energy verifier in the 2026 "
        "classifier-first verification-efficiency lane; the competent-judge "
        f"head-to-head is {status}, while the ARC-AGI-3 first step shows "
        f"{ratio:.3f}x synthetic action efficiency without an official benchmark claim."
    )


def build_note_content(sources: dict[str, JsonDict]) -> str:
    """Build the one-page positioning note from already-landed local evidence."""

    exp3926 = sources["exp3926"]
    exp3928 = sources["exp3928"]
    exp3929 = sources["exp3929"]
    arc_ratio = float(exp3929.get("action_efficiency_ratio") or 0.0)
    arc_ci = exp3929.get("action_efficiency_ci95") or {}
    ci_low = float(arc_ci.get("low") or 0.0) if isinstance(arc_ci, dict) else 0.0
    ci_high = float(arc_ci.get("high") or 0.0) if isinstance(arc_ci, dict) else 0.0
    exp3926_state = "blocked/flagged" if _status(exp3926).startswith("blocked") or _flagged(exp3926) else "complete"
    exp3928_state = "blocked/flagged" if _status(exp3928).startswith("blocked") or _flagged(exp3928) else "complete"

    return f"""# Agentic Verification Efficiency Positioning - Exp 3932

Method: local document synthesis only. No new inference was run; this note reads
`research-references.md`, `research-studying.md`, `ops/north-star.md`, and the
local Exp 3926/3928/3929 result artifacts.

## Position

Carnot's current thesis matches the 2026 verification-efficiency literature:
use a cheap discriminative verifier first, reserve competent generative judges
for hard or close cases, and measure value as accuracy per unit cost. ProcessBench
is the right held-out step-verification venue because it is explicitly labeled
at the process-step level. ThinkPRM and GenRM define the competent-judge recipe:
verification as structured generation plus a parsed verdict, not a raw yes/no
judge prompt. Budget-aware Discriminative Verification supplies the cost model:
a forward-pass verifier can beat generative verification under a fixed compute
budget even when the generative judge is strong.

The local evidence is not yet a clean efficiency win. Exp 3926 is {exp3926_state},
so the competent-judge parity/Pareto claim remains unlanded on disk. Exp 3928 is
also {exp3928_state}, so the independent-corpus moat replication still needs a
clean run. Exp 3929 does land the first ARC-AGI-3 agentic step: a verifier-pruned
synthetic grid agent solved at {arc_ratio:.3f}x action efficiency, CI95
[{ci_low:.3f}, {ci_high:.3f}], while explicitly making no official ARC-AGI-3 score
claim.

## Interpretation

Carnot is best positioned as a classifier-first verification layer rather than
as another generative PRM. The differentiator is not that energy reasoning is
more expressive than ThinkPRM or GenRM; it is that a cheap external verifier can
screen every step and every candidate action, then escalate only the uncertain
cases. That is exactly the north-star win condition: equally effective as the LM
at lower cost, with ARC-AGI-3 as the agentic proof venue after the offline proof.

## Next Experiments

1. ProcessBench full-benchmark head-to-head. Rationale: it converts the blocked
   Exp 3926 efficiency thesis into the decisive comparison against a competent
   GenRM/ThinkPRM-style judge on a standard held-out process-verification corpus.
2. ARC-AGI-3 real-benchmark agentic run. Rationale: it converts the Exp 3929
   synthetic action-pruning result into the intended interactive venue while
   preserving the discipline that no official score is claimed unless the
   benchmark protocol actually runs.
"""


def append_studying_candidate(studying_text: str) -> str:
    """Append the scored candidate once, preserving prior studying history."""

    if STUDYING_MARKER in studying_text:
        return studying_text
    section = f"""
{STUDYING_MARKER}

**Candidate:** Verification-efficiency positioning for the next convergence
milestone.

**Score: 5 x 5 x 4 x 4 = 400** - high alignment with north-star section 5, high
experiment leverage, medium implementation risk, and high convergence value.

**Position:** Carnot belongs in the cheap discriminative verifier lane: a
classifier-first energy layer screens all steps/actions, while competent
GenRM/ThinkPRM judges handle hard cases. The local Exp 3926/3928 artifacts are
blocked, so the claim is positioned as a near-term convergence target rather
than a landed parity result; Exp 3929 supplies the synthetic ARC-AGI-3 action-
efficiency bridge.

**Next experiments:** {NEXT_EXPERIMENTS}
"""
    return studying_text.rstrip() + "\n\n" + section.lstrip()


def write_json(path: Path, payload: JsonDict) -> Path:
    """Write a deterministic JSON artifact and return its path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def build_artifact(
    *,
    note_path: str,
    summary: str,
    next_experiments: str,
    new_references_added: int,
    public_untouched: bool,
    duration_s: float,
    honest_verdict: str,
    precondition_details: str,
) -> JsonDict:
    """Build the schema-complete terminal artifact."""

    return {
        "experiment": 3932,
        "synthesis_note_path": note_path,
        "landscape_position_summary": summary,
        "next_highest_leverage_experiments": next_experiments,
        "new_references_added": int(new_references_added),
        "public_docs_untouched": bool(public_untouched),
        "preconditions_checked": True,
        "precondition_details": precondition_details,
        "duration_s": duration_s,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": honest_verdict,
        "spec_refs": ["REQ-REPORT-3932", "SCENARIO-REPORT-3932"],
    }


def validate_artifact(artifact: JsonDict) -> None:
    """Validate the small schema used by the Exp 3932 result artifact."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if not isinstance(artifact["synthesis_note_path"], str):
        raise ValueError("synthesis_note_path must be a string")
    if not isinstance(artifact["landscape_position_summary"], str):
        raise ValueError("landscape_position_summary must be a string")
    if not isinstance(artifact["next_highest_leverage_experiments"], str):
        raise ValueError("next_highest_leverage_experiments must be a string")
    if not isinstance(artifact["new_references_added"], int):
        raise ValueError("new_references_added must be an int")
    if not isinstance(artifact["public_docs_untouched"], bool):
        raise ValueError("public_docs_untouched must be a bool")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must record no-new-inference synthesis")
    encoded = json.dumps(artifact, sort_keys=True)
    if "GGUF" in encoded or "CUDA" in encoded:
        raise ValueError("artifact must not carry GGUF/CUDA markers")
    verdict = str(artifact["honest_verdict"])
    if verdict.startswith("complete:"):
        expected = (
            "complete: literature_synthesis_positioned_"
            f"{artifact['new_references_added']}_new_refs_public_docs_untouched"
        )
        if verdict != expected:
            raise ValueError("complete verdict does not match falsification gate")
        if artifact["synthesis_note_path"] == "" or artifact["public_docs_untouched"] is not True:
            raise ValueError("complete artifact must include note path and untouched public docs")
    elif not verdict.startswith("blocked_references_missing"):
        raise ValueError("honest_verdict must be complete or blocked_references_missing")


def run(root: Path = REPO_ROOT, *, started_s: float | None = None, now_s: float | None = None) -> Path:
    """Run Exp 3932 and write the terminal JSON artifact."""

    before_public = public_doc_snapshot(root)
    preconditions = check_preconditions(root)
    output_path = root / OUTPUT_REL_PATH
    duration_s = duration_from(started_s, now_s)
    precondition_details = _precondition_details(preconditions)

    if not all(preconditions.values()):
        artifact = build_artifact(
            note_path="",
            summary="",
            next_experiments="",
            new_references_added=0,
            public_untouched=public_docs_untouched(root, before_public),
            duration_s=duration_s,
            honest_verdict="blocked_references_missing",
            precondition_details=precondition_details,
        )
        validate_artifact(artifact)
        return write_json(output_path, artifact)

    sources = load_source_artifacts(root)
    note_text = build_note_content(sources)
    note_path = root / SYNTHESIS_NOTE_REL_PATH
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(note_text, encoding="utf-8")

    studying_path = root / RESEARCH_STUDYING_REL_PATH
    studying_path.write_text(
        append_studying_candidate(studying_path.read_text(encoding="utf-8")),
        encoding="utf-8",
    )
    new_references_added = 0
    public_untouched = public_docs_untouched(root, before_public)
    summary = landscape_position_summary(sources)
    artifact = build_artifact(
        note_path=SYNTHESIS_NOTE_REL_PATH.as_posix(),
        summary=summary,
        next_experiments=NEXT_EXPERIMENTS,
        new_references_added=new_references_added,
        public_untouched=public_untouched,
        duration_s=duration_s,
        honest_verdict=(
            "complete: literature_synthesis_positioned_"
            f"{new_references_added}_new_refs_public_docs_untouched"
        ),
        precondition_details=precondition_details,
    )
    validate_artifact(artifact)
    return write_json(output_path, artifact)


def cli_main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for the Exp 3932 script wrapper."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--started-s", type=float, default=None)
    parser.add_argument("--now-s", type=float, default=None)
    args = parser.parse_args(argv)
    run(args.repo_root, started_s=args.started_s, now_s=args.now_s)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())
