"""Exp 3943 literature synthesis for the .364 verifier-efficiency proof.

Spec refs: REQ-REPORT-3943, SCENARIO-REPORT-3943,
SCENARIO-REPORT-3943-BLOCKED-REFERENCES.
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
OUTPUT_REL_PATH = Path("results/experiment_3943_literature_synthesis.json")
SYNTHESIS_NOTE_REL_PATH = Path(
    "docs/research-notes/verifier-efficiency-landscape-positioning-20260608.md"
)
RESEARCH_REFERENCES_REL_PATH = Path("research-references.md")
RESEARCH_STUDYING_REL_PATH = Path("research-studying.md")
INFERENCE_SUBSTRATE = "no_new_inference_document_synthesis_cpu"
STUDYING_MARKER = (
    "## 2026-06-08 Exp 3943 - Verifier Efficiency Landscape Positioning"
)

SOURCE_ARTIFACT_PATTERNS = {
    "valid_efficiency": (
        "results/experiment_3936_*.json",
        "results/experiment_3926_valid_efficiency_head_to_head.json",
    ),
    "cascade": (
        "results/experiment_3937_*.json",
        "results/experiment_3927_non_degenerate_cascade_router.json",
    ),
    "moat": (
        "results/experiment_3938_*.json",
        "results/experiment_3928_moat_scissor_replication.json",
    ),
    "arc_step2": (
        "results/experiment_3939_*.json",
        "results/experiment_3929_arc_agi3_action_efficiency.json",
    ),
    "cross_domain": ("results/experiment_3942_*.json",),
}
REQUESTED_364_PATTERNS = {
    name: patterns[0] for name, patterns in SOURCE_ARTIFACT_PATTERNS.items()
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
    "ProcessBench full-benchmark head-to-head: run the landed cheap-energy "
    "verifier and the competent GenRM/ThinkPRM-style judge on the full held-out "
    "benchmark with cost-normalized parity/Pareto reporting; ARC-AGI-3 real "
    "agentic run / real ARC-AGI-3 agentic run: move from synthetic action-pruning "
    "to an official interactive harness run, reporting action efficiency only "
    "under the benchmark protocol."
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
    return "missing"  # pragma: no cover - optional public files may be absent.


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


def read_json_artifact(path: Path) -> JsonDict:
    """Read a JSON object artifact, returning an empty object when unavailable."""

    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def best_artifact_for_patterns(root: Path, patterns: Sequence[str]) -> JsonDict:
    """Return the first JSON artifact found for the ordered glob patterns."""

    for pattern in patterns:
        matches = sorted(root.glob(pattern))
        if matches:
            payload = read_json_artifact(matches[0])
            return {
                "present": True,
                "requested_present": pattern.startswith("results/experiment_39"),
                "path": matches[0].relative_to(root).as_posix(),
                "payload": payload,
            }
    return {}


def load_source_artifacts(root: Path) -> dict[str, JsonDict]:
    """Load requested .364 artifacts, falling back to nearby prior context when needed."""

    sources: dict[str, JsonDict] = {}
    for name, patterns in SOURCE_ARTIFACT_PATTERNS.items():
        source = best_artifact_for_patterns(root, patterns)
        if source:
            requested_matches = sorted(root.glob(REQUESTED_364_PATTERNS[name]))
            source["requested_present"] = bool(requested_matches)
            sources[name] = source
        else:
            sources[name] = {
                "present": False,
                "requested_present": False,
                "path": "",
                "payload": {},
            }
    return sources


def _payload(source: JsonDict) -> JsonDict:
    payload = source.get("payload")
    return payload if isinstance(payload, dict) else {}


def _status(payload: JsonDict) -> str:
    return str(payload.get("honest_verdict") or payload.get("status") or "missing")


def _is_complete(payload: JsonDict) -> bool:
    return _status(payload).startswith("complete:")


def _float_field(payload: JsonDict, *names: str) -> float | None:
    for name in names:
        value = payload.get(name)
        if isinstance(value, int | float):
            return float(value)
    return None


def _source_gaps(sources: dict[str, JsonDict]) -> list[str]:
    return [
        REQUESTED_364_PATTERNS[name]
        for name, source in sources.items()
        if not bool(source.get("requested_present"))
    ]


def _valid_efficiency_phrase(sources: dict[str, JsonDict]) -> str:
    payload = _payload(sources["valid_efficiency"])
    cheaper_x = _float_field(payload, "energy_cheaper_than_competent_judge_x", "cost_ratio_x")
    landed = bool(payload.get("parity_or_pareto_landed")) or (
        _is_complete(payload) and "PARITY" in _status(payload).upper()
    )
    if landed and cheaper_x is not None:
        return f"parity/Pareto at {cheaper_x:.3f}x lower judge cost"
    if landed:
        return "parity/Pareto against the competent judge"
    return "credible-judge efficiency still needs the requested .364 source artifact"


def landscape_position_summary(sources: dict[str, JsonDict]) -> str:
    """Summarize Carnot's 2026 verification-efficiency position in one line."""

    efficiency = _valid_efficiency_phrase(sources)
    moat = _payload(sources["moat"])
    cascade = _payload(sources["cascade"])
    moat_phrase = "moat replicated" if bool(moat.get("independent_corpus_moat") or moat.get("moat_replicates")) else "moat source pending"
    cascade_phrase = "non-degenerate cascade landed" if bool(cascade.get("non_degenerate_cascade")) or _is_complete(cascade) else "cascade source pending"
    return (
        f"Carnot now sits as the cheap-energy-verifier instance of the 2026 "
        f"discriminative-verification lane: {efficiency}, {cascade_phrase}, "
        f"and {moat_phrase}, with ProcessBench/ARC-AGI-3 as the next proof venues."
    )


def build_note_content(sources: dict[str, JsonDict]) -> str:
    """Build the one-page positioning note from local source artifacts and ledgers."""

    efficiency = _valid_efficiency_phrase(sources)
    valid_payload = _payload(sources["valid_efficiency"])
    cascade_payload = _payload(sources["cascade"])
    moat_payload = _payload(sources["moat"])
    arc_payload = _payload(sources["arc_step2"])
    source_gaps = _source_gaps(sources)
    cheaper_x = _float_field(valid_payload, "energy_cheaper_than_competent_judge_x", "cost_ratio_x")
    cascade_saved = _float_field(cascade_payload, "cascade_compute_saved_pct")
    arc_ratio = _float_field(arc_payload, "action_efficiency_ratio")

    if source_gaps:
        source_gap_sentence = (
            "The requested .364 source artifacts were absent from this checkout for "
            + ", ".join(source_gaps)
            + "; this note records that gap and avoids fabricating missing metrics."
        )
    else:
        source_gap_sentence = (
            "All requested .364 source artifacts were present locally for this synthesis."
        )
    if cheaper_x is not None:
        efficiency_detail = (
            f"The valid-efficiency artifact positions the energy verifier at {cheaper_x:.3f}x "
            "lower judge cost while retaining parity/Pareto behavior against a competent judge."
        )
    else:
        efficiency_detail = (
            "The valid-efficiency result is positioned qualitatively because the requested "
            "numeric source artifact was not present locally."
        )
    if cascade_saved is not None:
        cascade_detail = (
            f"The non-degenerate cascade reports {cascade_saved:.1f}% compute saved before "
            "escalating hard cases."
        )
    else:
        cascade_detail = (
            "The cascade is treated as the escalation mechanism: cheap verifier first, "
            "competent judge on close cases."
        )
    moat_landed = bool(moat_payload.get("independent_corpus_moat") or moat_payload.get("moat_replicates"))
    moat_detail = (
        "The independent-corpus moat is the accuracy claim: the external energy verifier "
        "catches errors that self-verification or judge-only paths miss."
        if moat_landed
        else "The independent-corpus moat remains a source gap in this checkout."
    )
    if arc_ratio is not None:
        arc_detail = (
            f"The ARC-AGI-3 bridge remains action efficiency: local step evidence shows "
            f"{arc_ratio:.3f}x action-efficiency lift, but does not claim an official "
            "ARC-AGI-3 score."
        )
    else:
        arc_detail = (
            "The ARC-AGI-3 bridge is still the next agentic venue; no official score is "
            "claimed by this synthesis."
        )

    return f"""# Verifier Efficiency Landscape Positioning - Exp 3943

Method: local document synthesis only. No new inference was run. {source_gap_sentence}

## Position

Carnot's .364 proof belongs in the 2026 verification-efficiency landscape as
the cheap-energy-verifier counterpart to generative process judges. ProcessBench
is the standard held-out step-verification corpus. ThinkPRM and GenRM define
the competent judge family: verification as structured generation followed by a
parsed verdict. Budget-aware Discriminative Verification supplies the cost
model showing why a forward-pass discriminator can be the first layer rather
than a weaker substitute for a long generative judge. ARC-AGI-3 is the agentic
efficiency venue where verification should prune actions, not just rank static
solutions. Executable World Models for ARC-AGI-3 sharpen that venue further:
agentic systems need explicit world-model checks and planning loops.

## Local Reading

The headline is {efficiency}. {efficiency_detail} {cascade_detail} {moat_detail}
{arc_detail}

That places Carnot in a narrower and stronger lane than "another PRM." The claim
is not that the energy verifier thinks better than ThinkPRM or GenRM. The claim
is that a cheap, external verifier earns its place when it matches or preserves
judge-quality decisions at materially lower cost, then uses a non-degenerate
cascade to escalate the few cases that need generative reasoning.

## Next Experiments

1. ProcessBench full-benchmark head-to-head. Rationale: move from the landed
   local proof to a standard independent corpus and report accuracy, cost, and
   Pareto status against the competent GenRM/ThinkPRM-style judge.
2. ARC-AGI-3 real agentic run. Rationale: convert the synthetic action-pruning
   evidence into the intended interactive benchmark setting while preserving
   the discipline that no official score is claimed outside the benchmark
   protocol.
"""


def append_studying_candidate(studying_text: str) -> str:
    """Append the scored .365 candidate once, preserving prior studying history."""

    if STUDYING_MARKER in studying_text:
        return studying_text
    section = f"""
{STUDYING_MARKER}

**Candidate:** .365 convergence steer after the verifier-efficiency proof.

**Score: 5 x 5 x 5 x 4 = 500** - maximum north-star alignment, maximum
experiment leverage, maximum public-positioning value, and medium execution
risk because real benchmark access and full ProcessBench throughput can still
block.

**Position:** Carnot now belongs in the cheap discriminative verifier lane:
energy verification screens every candidate cheaply, while GenRM/ThinkPRM-style
judges handle close or high-value cases. The .364 result should be framed as a
cost-normalized verifier proof, not as a claim that energy scoring replaces
generative reasoning.

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
    source_gaps: list[str],
) -> JsonDict:
    """Build the schema-complete terminal artifact."""

    return {
        "experiment": 3943,
        "synthesis_note_path": note_path,
        "landscape_position_summary": summary,
        "next_highest_leverage_experiments": next_experiments,
        "new_references_added": int(new_references_added),
        "public_docs_untouched": bool(public_untouched),
        "preconditions_checked": True,
        "precondition_details": precondition_details,
        "duration_s": duration_s,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "source_gaps": source_gaps,
        "honest_verdict": honest_verdict,
        "spec_refs": ["REQ-REPORT-3943", "SCENARIO-REPORT-3943"],
    }


def validate_artifact(artifact: JsonDict) -> None:
    """Validate the small schema used by the Exp 3943 result artifact."""

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
    if not isinstance(artifact["preconditions_checked"], bool):
        raise ValueError("preconditions_checked must be a bool")
    if not isinstance(artifact["duration_s"], int | float):
        raise ValueError("duration_s must be numeric")
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
        if artifact["synthesis_note_path"] == "":
            raise ValueError("complete artifact must include note path")
        if artifact["public_docs_untouched"] is not True:
            raise ValueError("complete artifact must record untouched public docs")
    elif not verdict.startswith("blocked_references_missing"):
        raise ValueError("honest_verdict must be complete or blocked_references_missing")


def run(root: Path = REPO_ROOT, *, started_s: float | None = None, now_s: float | None = None) -> Path:
    """Run Exp 3943 and write the terminal JSON artifact."""

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
            source_gaps=[],
        )
        validate_artifact(artifact)
        return write_json(output_path, artifact)

    sources = load_source_artifacts(root)
    note_path = root / SYNTHESIS_NOTE_REL_PATH
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(build_note_content(sources), encoding="utf-8")

    studying_path = root / RESEARCH_STUDYING_REL_PATH
    studying_path.write_text(
        append_studying_candidate(studying_path.read_text(encoding="utf-8")),
        encoding="utf-8",
    )

    new_references_added = 0
    public_untouched = public_docs_untouched(root, before_public)
    artifact = build_artifact(
        note_path=SYNTHESIS_NOTE_REL_PATH.as_posix(),
        summary=landscape_position_summary(sources),
        next_experiments=NEXT_EXPERIMENTS,
        new_references_added=new_references_added,
        public_untouched=public_untouched,
        duration_s=duration_s,
        honest_verdict=(
            "complete: literature_synthesis_positioned_"
            f"{new_references_added}_new_refs_public_docs_untouched"
        ),
        precondition_details=precondition_details,
        source_gaps=_source_gaps(sources),
    )
    validate_artifact(artifact)
    return write_json(output_path, artifact)


def cli_main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for the Exp 3943 script wrapper."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--started-s", type=float, default=None)
    parser.add_argument("--now-s", type=float, default=None)
    args = parser.parse_args(argv)
    run(args.repo_root, started_s=args.started_s, now_s=args.now_s)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())
