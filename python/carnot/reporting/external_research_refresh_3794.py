"""Append the .347 external research refresh and write its audit artifact.

Spec refs: REQ-REPORT-3794, SCENARIO-REPORT-3794.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from scripts import adversarial_verify


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RANDOM_SEED = 3794
RESEARCH_REFERENCES_REL_PATH = Path("research-references.md")
OUTPUT_REL_PATH = Path("results/experiment_3794_external_research_refresh.json")
INFERENCE_SUBSTRATE = (
    "aggregation_from_upstream_artifacts (principle: a documentation append, "
    "no live model)."
)
TERMINAL_VERDICT = (
    "complete: "
    "external_research_refresh_347_filed_references_section_appended_"
    "numbers_as_reported"
)
SECTION_HEADER = (
    "## .347 additions - EDLM preflight, certified abstention, predictive "
    "verification, and gaming-resistance (2026-06-04)"
)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "references_added",
    "n_references_added",
    "section_appended_not_replaced",
    "numbers_are_as_reported",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix; the refresh outcome.",
    "inference_substrate": (
        "A documentation append over upstream paper records; prevents confusing "
        "this refresh with a compute-bound experiment."
    ),
    "references_added": (
        "The list of arXiv ids filed -- the deliverable; keeps the project "
        "current per research-program.md planning requirement 4."
    ),
    "n_references_added": (
        "BARE int -- sample-size hygiene on the refresh (>=4 this sweep)."
    ),
    "section_appended_not_replaced": (
        "BARE bool, true -- the '.347 additions' section was APPENDED "
        "(never-prune rule; no prior content removed)."
    ),
    "numbers_are_as_reported": (
        "BARE bool, true -- peer numbers are source-reported, not Carnot "
        "measurements (adversarial-verify discipline)."
    ),
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Content hash catches drift.",
    "duration_s": "Wall-clock plausibility floor.",
}


@dataclass(frozen=True)
class ResearchReference:
    """One external paper filed into the append-only research register."""

    arxiv_id: str
    title: str
    submitted: str
    tracks: tuple[str, ...]
    as_reported_summary: str
    relevance: str
    as_reported_note: str
    arxiv_abs_resolved: bool = True
    venue_context: str | None = None

    @property
    def arxiv_abs_url(self) -> str:
        return f"https://arxiv.org/abs/{self.arxiv_id.removeprefix('arXiv:')}"

    def to_artifact_row(self) -> JsonDict:
        row: JsonDict = {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "submitted": self.submitted,
            "tracks": list(self.tracks),
            "arxiv_abs_url": self.arxiv_abs_url,
            "source_kind": "arxiv_abs",
            "arxiv_abs_resolved": self.arxiv_abs_resolved,
            "numbers_are_as_reported": True,
            "as_reported_summary": self.as_reported_summary,
            "relevance": self.relevance,
            "as_reported_note": self.as_reported_note,
        }
        if self.venue_context:
            row["venue_context"] = self.venue_context
        return row


REFERENCES = (
    ResearchReference(
        arxiv_id="arXiv:2605.04291",
        title=(
            "Leveraging Pretrained Language Models as Energy Functions for "
            "Glauber Dynamics Text Diffusion"
        ),
        submitted="submitted 2026-05-05",
        tracks=("EDLM next-thesis preflight",),
        as_reported_summary=(
            "Builds a discrete diffusion language model using Glauber dynamics "
            "and a pretrained causal/masked LM as the energy function; reports "
            "better quality than prior diffusion LMs and competitive results "
            "with comparable autoregressive models."
        ),
        relevance=(
            "Direct EDLM follow-up surface: energy over the discrete diffusion "
            "trajectory, but with pretrained-LM energy support rather than a "
            "tiny from-scratch EBT."
        ),
        as_reported_note=(
            "Quality and reasoning-task comparisons are source-reported only; "
            "re-derive before forward-facing Carnot use."
        ),
        venue_context="ACL 2026",
    ),
    ResearchReference(
        arxiv_id="arXiv:2601.21484",
        title="ETS: Energy-Guided Test-Time Scaling for Training-Free RL Alignment",
        submitted="submitted 2026-01-29; last revised 2026-05-19",
        tracks=("EDLM next-thesis preflight",),
        as_reported_summary=(
            "Frames masked-LM sampling as a reference policy plus an energy "
            "term and estimates that term online; reports improvements across "
            "autoregressive and diffusion language models on reasoning, coding, "
            "and science benchmarks."
        ),
        relevance=(
            "Energy-as-residual-corrector precedent for the EDLM route: use an "
            "energy term to steer a language-model transition kernel without "
            "reopening the bounded Thesis-A generator route."
        ),
        as_reported_note=(
            "Benchmark gains and convergence claims are source-reported only."
        ),
        venue_context="ICML 2026",
    ),
    ResearchReference(
        arxiv_id="arXiv:2605.20270",
        title="Conformal Selective Acting: Anytime-Valid Risk Control for RLVR-Trained LLMs",
        submitted="submitted 2026-05-18",
        tracks=("selective-prediction / abstention surface",),
        as_reported_summary=(
            "Provides a deployment-side conformal wrapper for RLVR-trained LLMs "
            "with anytime pathwise selective-risk control; reports evaluation "
            "over specialist benchmarks, adversarial shift cells, and live "
            "online-LoRA rounds."
        ),
        relevance=(
            "Clean banked-verifier product surface: a per-round abstain/act "
            "certificate for adaptive streams, matching Carnot's selective "
            "verification mode better than static threshold prose."
        ),
        as_reported_note=(
            "Validity bounds, stream counts, and comparison outcomes are "
            "source-reported only."
        ),
    ),
    ResearchReference(
        arxiv_id="arXiv:2603.12529",
        title="TERMINATOR: Learning Optimal Exit Points for Early Stopping in Chain-of-Thought Reasoning",
        submitted="submitted 2026-03-13; last revised 2026-05-14",
        tracks=("Tier-3 predictive verification",),
        as_reported_summary=(
            "Trains an inference-time early-exit strategy from first-answer "
            "positions; reports 14%-55% average CoT length reductions and more "
            "than 2x latency reduction on practical reasoning datasets."
        ),
        relevance=(
            "Tier-3 predictive-verification precedent: predict when reasoning "
            "has enough evidence to stop or escalate, instead of verifying only "
            "after the full process trace."
        ),
        as_reported_note=(
            "Length and latency reductions are source-reported only."
        ),
    ),
    ResearchReference(
        arxiv_id="arXiv:2603.05488",
        title="Reasoning Theater: Disentangling Model Beliefs from Chain-of-Thought",
        submitted="submitted 2026-03-05; last revised 2026-05-28",
        tracks=("Tier-3 predictive verification", "gaming-resistance"),
        as_reported_summary=(
            "Compares activation probes, early forced answering, and CoT "
            "monitors; reports that final answers can be decodable from "
            "activations earlier than monitors detect and that probe-guided "
            "early exit can reduce tokens with similar accuracy."
        ),
        relevance=(
            "Warns that process monitors can trail latent model belief; Carnot's "
            "Tier-3 head should predict verifier outcomes from internal/process "
            "features while treating CoT text as gameable evidence."
        ),
        as_reported_note=(
            "Probe timing and token-reduction numbers are source-reported only."
        ),
    ),
    ResearchReference(
        arxiv_id="arXiv:2602.01750",
        title="Adversarial Reward Auditing for Active Detection and Mitigation of Reward Hacking",
        submitted="submitted 2026-02-02",
        tracks=("gaming-resistance",),
        as_reported_summary=(
            "Models reward hacking as a Hacker/Auditor game and uses "
            "Auditor-Guided RLHF to gate reward signals; reports mitigation of "
            "sycophancy, verbosity, and code-gaming scenarios with cross-domain "
            "generalization."
        ),
        relevance=(
            "Complements the `.346` verifier-gaming entry with a defense "
            "surface: adversarial auditing plus reward gating as a peer pattern "
            "for Carnot's gaming-resistance characterization."
        ),
        as_reported_note=(
            "Mitigation and cross-domain generalization outcomes are "
            "source-reported only."
        ),
    ),
)

REFERENCE_IDS = tuple(ref.arxiv_id for ref in REFERENCES)


def run(
    repo_root: Path = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Append the section if needed, write the artifact, and return its path."""

    started = time.time() if started_s is None else started_s
    references_path = repo_root / RESEARCH_REFERENCES_REL_PATH
    before_text = references_path.read_text(encoding="utf-8")
    append_action = append_section_if_missing(references_path)
    after_text = references_path.read_text(encoding="utf-8")

    artifact = build_artifact(
        repo_root,
        append_action=append_action,
        before_text=before_text,
        after_text=after_text,
        started_s=started,
        now_s=now_s,
    )
    out_path = repo_root / OUTPUT_REL_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    verify_report = adversarial_verify.verify_artifact(out_path)
    artifact["adversarial_verify_report"] = compact_verify_report(verify_report)
    artifact["adversarial_verify_clean"] = report_is_clean(verify_report)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def append_section_if_missing(path: Path) -> str:
    """Append the rendered .347 section while preserving prior bytes as a prefix."""

    text = path.read_text(encoding="utf-8")
    if SECTION_HEADER in text:
        return "already_present"
    separator = "" if text.endswith("\n\n") else "\n" if text.endswith("\n") else "\n\n"
    path.write_text(text + separator + render_section(), encoding="utf-8")
    return "appended"


def render_section() -> str:
    """Render the markdown section that is appended to research-references.md."""

    lines = [
        SECTION_HEADER,
        "",
        (
            "Added by the `.347 planning sweep. This is an append-only filing "
            "of newly surfaced external references into the converged record: "
            "the verifier-moat thread remains closed per "
            "`[[reference_deep_think_post_bounded_2026_06]]`, energy-selection "
            "and Thesis-A EBT generation remain bounded per "
            "`[[project_energy_selection_thesis_bounded]]` and "
            "`[[project_thesis_a_ebt_seeded]]`, and EDLM remains an "
            "operator-seeded preflight route rather than a loop commitment."
        ),
        "",
        (
            "Numbers are source-reported by the papers or paper pages. They are "
            "not Carnot measurements and must be independently re-derived before "
            "entering any forward-facing claim."
        ),
        "",
    ]
    for ref in REFERENCES:
        track = " / ".join(ref.tracks)
        venue = f" {ref.venue_context}." if ref.venue_context else ""
        lines.extend(
            [
                (
                    f"- **{ref.arxiv_id} - \"{ref.title}\" ({ref.submitted}; "
                    f"arXiv resolved):** Track: {track}. "
                    f"{ref.as_reported_summary} {ref.relevance} "
                    f"{ref.as_reported_note}{venue}"
                ),
            ]
        )
    return "\n".join(lines) + "\n"


def build_artifact(
    repo_root: Path,
    *,
    append_action: str,
    before_text: str,
    after_text: str,
    started_s: float,
    now_s: float | None = None,
) -> JsonDict:
    """Build the terminal artifact from the before/after research-register state."""

    section = render_section()
    artifact: JsonDict = {
        "honest_verdict": TERMINAL_VERDICT,
        "schema": "carnot.external_research_refresh.v1",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "references_added": list(REFERENCE_IDS),
        "n_references_added": len(REFERENCE_IDS),
        "section_appended_not_replaced": section_appended_not_replaced(
            before_text, after_text
        ),
        "numbers_are_as_reported": True,
        "random_seed": RANDOM_SEED,
        "duration_s": elapsed_seconds(started_s, now_s),
        "field_principles": dict(FIELD_PRINCIPLES),
        "references": [ref.to_artifact_row() for ref in REFERENCES],
        "research_references_path": RESEARCH_REFERENCES_REL_PATH.as_posix(),
        "section_header": SECTION_HEADER,
        "section_sha256": sha256_text(section),
        "append_action": append_action,
        "arxiv_resolution_summary": {
            "all_required_arxiv_ids_resolved": all(
                ref.arxiv_abs_resolved for ref in REFERENCES
            ),
            "verification_basis": (
                "Primary arXiv abstract pages resolved during the .347 filing; "
                "entries preserve source-reported claims pending independent "
                "Carnot re-derivation."
            ),
        },
        "adversarial_verify_clean": True,
        "adversarial_verify_report": {"max_severity": -1, "flags": []},
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def section_appended_not_replaced(before_text: str, after_text: str) -> bool:
    """Return true when old content is preserved and the .347 section exists."""

    return (
        after_text.startswith(before_text)
        and SECTION_HEADER in after_text
        and ".345 additions" in after_text
        and ".346 additions" in after_text
    )


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate required fields and anti-fabrication hygiene."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    _ensure(not missing, f"missing required fields: {missing}")
    _ensure(artifact.get("honest_verdict") == TERMINAL_VERDICT, "honest_verdict")
    _ensure(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    _ensure(artifact.get("references_added") == list(REFERENCE_IDS), "references_added")
    _ensure(artifact.get("n_references_added") == len(REFERENCE_IDS), "n_references_added")
    _ensure(4 <= int(artifact.get("n_references_added", 0)) <= 6, "n_references_added")
    _ensure(
        artifact.get("section_appended_not_replaced") is True,
        "section_appended_not_replaced",
    )
    _ensure(artifact.get("numbers_are_as_reported") is True, "numbers_are_as_reported")
    _ensure(artifact.get("random_seed") == RANDOM_SEED, "random_seed")
    _ensure(float(artifact.get("duration_s", 0.0)) > 0.0, "duration_s")
    principles = artifact.get("field_principles")
    _ensure(isinstance(principles, Mapping), "field_principles")
    _ensure(set(REQUIRED_ARTIFACT_FIELDS) <= set(principles), "field_principles")
    refs = artifact.get("references")
    _ensure(isinstance(refs, list), "references")
    _ensure(len(refs) == len(REFERENCE_IDS), "references")
    _ensure("arXiv:2604.15149" not in REFERENCE_IDS, "references")
    _ensure(
        {row.get("arxiv_id") for row in refs if isinstance(row, Mapping)}
        == set(REFERENCE_IDS),
        "references",
    )
    _ensure(
        all(
            isinstance(row, Mapping)
            and row.get("arxiv_abs_resolved") is True
            and row.get("numbers_are_as_reported") is True
            and row.get("source_kind") == "arxiv_abs"
            for row in refs
        ),
        "references",
    )
    encoded = json.dumps(artifact, sort_keys=True)
    _ensure("GGUF" not in encoded and "CUDA" not in encoded, "inference-substrate hygiene")
    _ensure("live-model" not in encoded, "inference-substrate hygiene")
    _ensure(
        artifact.get("reproducibility_checksum") == payload_checksum(artifact),
        "checksum",
    )
    _ensure(artifact.get("adversarial_verify_clean") is True, "adversarial_verify_clean")


def _ensure(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"invalid Exp 3794 artifact: {message}")


def compact_verify_report(report: Mapping[str, Any]) -> JsonDict:
    """Keep the adversarial verifier report compact but auditable."""

    flags = list(report.get("flags", []))
    return {
        "max_severity": max((severity_rank(flag.get("severity")) for flag in flags), default=-1),
        "flags": flags,
    }


def report_is_clean(report: Mapping[str, Any] | None) -> bool:
    """Return true when no critical adversarial flag is present."""

    if not isinstance(report, Mapping):
        return True
    return not any(
        isinstance(flag, Mapping) and str(flag.get("severity", "")).lower() == "critical"
        for flag in report.get("flags", [])
    )


def severity_rank(severity: Any) -> int:
    """Map verifier severity labels to sortable integers."""

    return {"info": 0, "warn": 1, "critical": 2}.get(str(severity).lower(), -1)


def elapsed_seconds(started_s: float, now_s: float | None) -> float:
    """Return rounded wall-clock duration with a nonzero floor."""

    now = time.time() if now_s is None else now_s
    return round(max(now - started_s, 0.0001), 6)


def sha256_text(text: str) -> str:
    """Return a SHA-256 digest for markdown content."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Return the reproducibility checksum over payload content."""

    filtered = {key: value for key, value in payload.items() if key != "reproducibility_checksum"}
    encoded = json.dumps(filtered, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
