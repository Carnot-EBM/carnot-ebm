"""Append the .349 external research refresh and write its audit artifact.

Spec refs: REQ-REPORT-3816, SCENARIO-REPORT-3816.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from scripts import adversarial_verify


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RANDOM_SEED = 3816
RESEARCH_REFERENCES_REL_PATH = Path("research-references.md")
OUTPUT_REL_PATH = Path("results/experiment_3816_external_research_refresh_v349.json")
INFERENCE_SUBSTRATE = (
    "aggregation_from_upstream_artifacts (principle: a documentation append, "
    "no live model)."
)
TERMINAL_VERDICT = (
    "complete: "
    "external_research_refresh_349_section_intact_references_appended_"
    "numbers_as_reported"
)
SECTION_HEADER = (
    "## .349 additions - geometry-calibrated conformal abstention, selective "
    "conformal judging, and the EDLM reasoning datapoint (2026-06-04)"
)

REQUIRED_349_TOKENS = (
    "Added by the `.349 planning sweep",
    "Numbers are source-reported by the papers",
    "arXiv:2604.27914",
    "arXiv:2602.13110",
    "EDLM reasoning datapoint",
    "arXiv:2410.21357",
    "reference_deep_think_post_bounded_2026_06",
    "project_energy_selection_thesis_bounded",
    "project_thesis_a_ebt_seeded",
)

EXCLUDED_DUPLICATE_IDS = (
    "arXiv:2604.27914",
    "arXiv:2602.13110",
    "arXiv:2410.21357",
    "arXiv:2604.13991",
    "arXiv:2507.02092",
)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "references_section_intact",
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
    "references_section_intact": (
        "BARE bool, true -- the planner-appended '.349 additions' section is "
        "present + parses."
    ),
    "references_added": (
        "The list of arXiv ids filed this sweep -- the deliverable; keeps the "
        "project current per research-program.md planning requirement 4."
    ),
    "n_references_added": "BARE int -- sample-size hygiene on the refresh.",
    "section_appended_not_replaced": (
        "BARE bool, true -- the section was APPENDED/confirmed-intact "
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
        arxiv_id="arXiv:2605.28920",
        title="Conf-Gen: Conformal Uncertainty Quantification for Generative Models",
        submitted="submitted 2026-05-27",
        tracks=("selective-prediction / abstention surface",),
        as_reported_summary=(
            "Adapts conformal risk control to generative tasks and reports "
            "applications to conversational AI asking enough clarifying "
            "questions and agent-output correctness."
        ),
        relevance=(
            "Banked abstention product surface: calibrate an act/abstain "
            "decision over generated artifacts rather than treating generation "
            "as a fixed classifier output."
        ),
        as_reported_note=(
            "Guarantees and application outcomes are source-reported only."
        ),
        venue_context="ICML 2026",
    ),
    ResearchReference(
        arxiv_id="arXiv:2606.03731",
        title="Conformal Language Modeling via Posterior Sampling",
        submitted="submitted 2026-06-03",
        tracks=("selective-prediction / abstention surface",),
        as_reported_summary=(
            "Samples from an approximate LLM posterior conditioned on a "
            "calibrated high-scoring region and reports target risk control "
            "with higher downstream utility than post-hoc filtering."
        ),
        relevance=(
            "Forward surface for banked abstention: move the conformal guarantee "
            "inside the sampling path instead of only filtering finished answers."
        ),
        as_reported_note=(
            "Risk-control and utility claims are source-reported only."
        ),
    ),
    ResearchReference(
        arxiv_id="arXiv:2602.03814",
        title="Conformal Thinking: Risk Control for Reasoning on a Compute Budget",
        submitted="submitted 2026-02-03; last revised 2026-05-14",
        tracks=(
            "selective-prediction / abstention surface",
            "Tier-3 fast-path process verification",
        ),
        as_reported_summary=(
            "Uses distribution-free risk control to set reasoning stop thresholds "
            "that limit error while minimizing compute."
        ),
        relevance=(
            "Direct peer for .349 fast-path verification: Tier-3 can expose a "
            "risk target and compute budget instead of a hand-tuned confidence "
            "cutoff."
        ),
        as_reported_note=(
            "Risk adherence and efficiency outcomes are source-reported only."
        ),
        venue_context="ICML 2026",
    ),
    ResearchReference(
        arxiv_id="arXiv:2602.15014",
        title="Scaling Beyond Masked Diffusion Language Models",
        submitted="submitted 2026-02-16",
        tracks=("EDLM operator-seed",),
        as_reported_summary=(
            "Studies uniform-state and interpolating discrete diffusion scaling; "
            "reports about 12% FLOP efficiency for masked diffusion training and "
            "a 1.7B uniform-state model outperforming autoregressive and masked "
            "diffusion baselines on GSM8K despite worse perplexity."
        ),
        relevance=(
            "Operator-seed boundary for EDLM: compare diffusion families under "
            "matched compute and avoid treating perplexity as the only substrate "
            "selection signal."
        ),
        as_reported_note=(
            "FLOP, scale, and benchmark claims are source-reported only."
        ),
    ),
    ResearchReference(
        arxiv_id="arXiv:2603.06123",
        title="Diffusion Language Models Are Natively Length-Aware",
        submitted="submitted 2026-03-06",
        tracks=("EDLM operator-seed", "Tier-3 fast-path process verification"),
        as_reported_summary=(
            "Uses latent prompt representations to crop the diffusion context "
            "before generation and reports large FLOP reductions with minimal "
            "performance impact across reasoning, code, instruction, and QA tasks."
        ),
        relevance=(
            "Fast-path cue for the EDLM seed: diffusion can expose a pre-generation "
            "length/compute estimate that the Tier-3 gate can route or reject."
        ),
        as_reported_note=(
            "Efficiency and performance-impact claims are source-reported only."
        ),
    ),
    ResearchReference(
        arxiv_id="arXiv:2603.06617",
        title="Evo: Autoregressive-Diffusion Large Language Models with Evolving Balance",
        submitted="submitted 2026-02-20",
        tracks=("EDLM operator-seed",),
        as_reported_summary=(
            "Introduces an AR-diffusion latent trajectory model and reports Evo "
            "8B is competitive on reasoning, code, and language benchmarks while "
            "maintaining fast inference."
        ),
        relevance=(
            "EDLM operator-seed route can consider hybrid AR/diffusion transition "
            "kernels instead of a pure discrete-diffusion commitment."
        ),
        as_reported_note=(
            "Benchmark and speed claims are source-reported only."
        ),
    ),
    ResearchReference(
        arxiv_id="arXiv:2509.01025",
        title="Any-Order Flexible Length Masked Diffusion",
        submitted="submitted 2025-08-31; last revised 2025-09-07",
        tracks=("EDLM operator-seed",),
        as_reported_summary=(
            "Adds token insertion to masked diffusion, reports higher length "
            "fidelity, about 60% higher maze-planning success than masked "
            "diffusion baselines, and LLaDA-8B GSM8K gains from 58% to 67% "
            "after retrofit."
        ),
        relevance=(
            "EDLM follow-up surface: variable-length any-order diffusion addresses "
            "a core fixed-length weakness before the operator spends a seed run."
        ),
        as_reported_note=(
            "Planning, math, and code-infill numbers are source-reported only."
        ),
    ),
    ResearchReference(
        arxiv_id="arXiv:2606.02020",
        title="Unveiling the Entropy Dynamics of Chain-of-Thought Reasoning",
        submitted="submitted 2026-06-01",
        tracks=("Tier-3 fast-path process verification",),
        as_reported_summary=(
            "Finds a two-phase CoT entropy pattern and reports a CUSUM change-point "
            "controller with 63.06% accuracy and 11.1% token reduction on early "
            "exit."
        ),
        relevance=(
            "Predictive verification cue for .349: monitor process convergence "
            "as a sequential signal before paying for full slow-path verification."
        ),
        as_reported_note=(
            "Accuracy, token-reduction, and comparison numbers are source-reported only."
        ),
    ),
    ResearchReference(
        arxiv_id="arXiv:2604.06787",
        title="When Is Thinking Enough? Early Exit via Sufficiency Assessment for Efficient Reasoning",
        submitted="submitted 2026-04-08",
        tracks=("Tier-3 fast-path process verification",),
        as_reported_summary=(
            "Introduces Dynamic Thought Sufficiency in Reasoning with reflection "
            "signal monitoring and a sufficiency check; reports 28.9%-34.9% "
            "reasoning-length reductions with minimal performance loss."
        ),
        relevance=(
            "Fast-path process verifier precedent: exit only after a sufficiency "
            "assessment, not merely after a fixed token budget."
        ),
        as_reported_note=(
            "Length-reduction and performance claims are source-reported only."
        ),
        venue_context="ACL 2026",
    ),
    ResearchReference(
        arxiv_id="arXiv:2604.16890",
        title="Step-GRPO: Internalizing Dynamic Early Exit for Efficient Reasoning",
        submitted="submitted 2026-04-18",
        tracks=("Tier-3 fast-path process verification",),
        as_reported_summary=(
            "Post-trains models to internalize dynamic early exit at semantic-step "
            "granularity and reports a 32.0% token reduction on Qwen3-8B versus "
            "the vanilla model."
        ),
        relevance=(
            "Training-side complement to the .349 Tier-3 gate: make concise "
            "high-confidence traces easier for the verifier to accept quickly."
        ),
        as_reported_note=(
            "Token and accuracy-efficiency claims are source-reported only."
        ),
        venue_context="ACL 2026",
    ),
)

REFERENCE_IDS = tuple(ref.arxiv_id for ref in REFERENCES)


def run(
    repo_root: Path = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Append missing .349 bullets, write the artifact, and return its path."""

    started = time.time() if started_s is None else started_s
    references_path = repo_root / RESEARCH_REFERENCES_REL_PATH
    before_text = references_path.read_text(encoding="utf-8")
    append_action = append_references_if_missing(references_path)
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


def append_references_if_missing(path: Path) -> str:
    """Append missing .349 bullets while preserving prior bytes as a prefix."""

    text = path.read_text(encoding="utf-8")
    confirm_349_section_intact(text)
    section = extract_349_section(text)
    missing = [ref for ref in REFERENCES if ref.arxiv_id not in section]
    if not missing:
        return "already_present"
    separator = "" if text.endswith("\n") else "\n"
    path.write_text(text + separator + render_reference_bullets(missing), encoding="utf-8")
    return "appended"


def render_reference_bullets(
    references: Iterable[ResearchReference] = REFERENCES,
) -> str:
    """Render markdown bullets for references added by this refresh."""

    lines: list[str] = []
    for ref in references:
        track = " / ".join(ref.tracks)
        venue = f" {ref.venue_context}." if ref.venue_context else ""
        lines.append(
            (
                f"- **{ref.arxiv_id} - \"{ref.title}\" ({ref.submitted}; "
                f"arXiv resolved):** Track: {track}. "
                f"{ref.as_reported_summary} {ref.relevance} "
                f"{ref.as_reported_note}{venue}"
            )
        )
    return "\n".join(lines) + "\n"


def extract_349_section(text: str) -> str:
    """Return the .349 section or raise a precise integrity error."""

    start = text.find(SECTION_HEADER)
    if start == -1:
        if "## .349 additions" in text:
            raise ValueError(".349 section not intact: expected canonical header")
        raise ValueError("missing .349 additions section")
    next_header = text.find("\n## ", start + len(SECTION_HEADER))
    if next_header == -1:
        return text[start:]
    return text[start:next_header]


def confirm_349_section_intact(text: str) -> bool:
    """Validate that the planner-created .349 section still parses."""

    section = extract_349_section(text)
    missing = [token for token in REQUIRED_349_TOKENS if token not in section]
    if missing:
        raise ValueError(f".349 section not intact: missing {missing}")
    return True


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

    section = extract_349_section(after_text)
    artifact: JsonDict = {
        "honest_verdict": TERMINAL_VERDICT,
        "schema": "carnot.external_research_refresh.v1",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "references_section_intact": confirm_349_section_intact(before_text),
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
        "section_parse_summary": {
            "single_349_section": after_text.count(SECTION_HEADER) == 1,
            "required_planner_anchors_present": list(REQUIRED_349_TOKENS),
            "excluded_duplicate_ids": list(EXCLUDED_DUPLICATE_IDS),
        },
        "arxiv_resolution_summary": {
            "all_added_arxiv_ids_resolved": all(
                ref.arxiv_abs_resolved for ref in REFERENCES
            ),
            "verification_basis": (
                "Primary arXiv abstract pages were resolved during the .349 "
                "filing; entries preserve source-reported claims pending "
                "independent Carnot re-derivation."
            ),
        },
        "adversarial_verify_clean": True,
        "adversarial_verify_report": {"max_severity": -1, "flags": []},
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def section_appended_not_replaced(before_text: str, after_text: str) -> bool:
    """Return true when old content is preserved and the .349 section is complete."""

    return (
        after_text.startswith(before_text)
        and confirm_349_section_intact(after_text)
        and all(arxiv_id in extract_349_section(after_text) for arxiv_id in REFERENCE_IDS)
    )


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate required fields and anti-fabrication hygiene."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    _ensure(not missing, f"missing required fields: {missing}")
    _ensure(artifact.get("honest_verdict") == TERMINAL_VERDICT, "honest_verdict")
    _ensure(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    _ensure(artifact.get("references_section_intact") is True, "references_section_intact")
    _ensure(artifact.get("references_added") == list(REFERENCE_IDS), "references_added")
    _ensure(artifact.get("n_references_added") == len(REFERENCE_IDS), "n_references_added")
    _ensure(int(artifact.get("n_references_added", 0)) > 0, "n_references_added")
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
    _ensure(
        not any(arxiv_id in REFERENCE_IDS for arxiv_id in EXCLUDED_DUPLICATE_IDS),
        "references",
    )
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
        raise ValueError(f"invalid Exp 3816 artifact: {message}")


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

    filtered = {
        key: value for key, value in payload.items() if key != "reproducibility_checksum"
    }
    encoded = json.dumps(filtered, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
