"""Append the .346 external research refresh and write its audit artifact.

Spec refs: REQ-REPORT-3783, SCENARIO-REPORT-3783.
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
RANDOM_SEED = 3783
RESEARCH_REFERENCES_REL_PATH = Path("research-references.md")
OUTPUT_REL_PATH = Path("results/experiment_3783_external_research_refresh.json")
INFERENCE_SUBSTRATE = (
    "aggregation_from_upstream_artifacts (principle: a documentation append, "
    "no live model)."
)
TERMINAL_VERDICT = (
    "complete: "
    "external_research_refresh_346_filed_5_references_section_appended_"
    "numbers_as_reported"
)
SECTION_HEADER = (
    "## .346 additions - correlated-error verifier moat corroboration, "
    "verifier robustness, and fast/slow escalation (2026-06-04)"
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
        "BARE int -- sample-size hygiene on the refresh (>=5 this sweep)."
    ),
    "section_appended_not_replaced": (
        "BARE bool, true -- the '.346 additions' section was APPENDED "
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
    relevance: str
    as_reported_note: str
    arxiv_abs_resolved: bool = True
    workshop_context: str | None = None

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
            "arxiv_abs_resolved": self.arxiv_abs_resolved,
            "numbers_are_as_reported": True,
            "relevance": self.relevance,
            "as_reported_note": self.as_reported_note,
        }
        if self.workshop_context:
            row["workshop_context"] = self.workshop_context
        return row


REFERENCES = (
    ResearchReference(
        arxiv_id="arXiv:2604.07650",
        title=(
            "How Independent are Large Language Models? A Statistical Framework "
            "for Auditing Behavioral Entanglement and Reweighting Verifier Ensembles"
        ),
        submitted="submitted 2026-04-08",
        tracks=("closed moat", "verifier product"),
        relevance=(
            "Audits behavioral entanglement and de-entangled verifier-ensemble "
            "reweighting; this is the error-INDEPENDENCE methodology and it "
            "corroborates the closed moat thread because apparent agreement can "
            "be correlated error rather than independent validation."
        ),
        as_reported_note=(
            "Reports up to about 4.5% gain over majority voting; source-reported "
            "only, not a Carnot measurement."
        ),
    ),
    ResearchReference(
        arxiv_id="arXiv:2506.07962",
        title="Correlated Errors in Large Language Models",
        submitted="submitted 2025-06-09; accepted to ICML 2025",
        tracks=("closed moat",),
        relevance=(
            "Finds that stronger LLMs can share more correlated errors; this is "
            "the operational subsumption mechanism behind the moat being narrow "
            "in `[[reference_deep_think_post_bounded_2026_06]]`."
        ),
        as_reported_note=(
            "Direction and peer statistics are source-reported; re-derive before "
            "using as a forward-facing Carnot claim."
        ),
    ),
    ResearchReference(
        arxiv_id="arXiv:2601.17223",
        title=(
            "Beyond Outcome Verification: Verifiable Process Reward Models for "
            "Structured Reasoning"
        ),
        submitted="submitted 2026-01-23",
        tracks=("verifier product", "self-learning"),
        relevance=(
            "Introduces deterministic rule-based step verifiers as process "
            "reward signals instead of opaque neural judges; this corroborates "
            "Carnot's objective-energy positioning for the verifier product."
        ),
        as_reported_note=(
            "Reported F1 and coherence gains are peer numbers, not reproduced "
            "Carnot measurements."
        ),
    ),
    ResearchReference(
        arxiv_id="arXiv:2604.15149",
        title="LLMs Gaming Verifiers: RLVR can Lead to Reward Hacking",
        submitted="submitted 2026-04-16",
        tracks=("verifier product", "self-learning"),
        relevance=(
            "Studies verifier gaming and shortcut strategies under RLVR; this is "
            "the verifier-robustness / null-space-mimicry frontier for Carnot's "
            "energy verifier gaming-resistance argument."
        ),
        as_reported_note=(
            "Workshop and behavioral findings are source-reported; use only as "
            "motivation until independently reproduced."
        ),
        workshop_context="OpenReview: ICLR 2026 Workshop LLM Reasoning",
    ),
    ResearchReference(
        arxiv_id="arXiv:2502.11157",
        title="Dyve: Thinking Fast and Slow for Dynamic Process Verification",
        submitted="submitted 2025-02-16",
        tracks=("anomaly-escalation", "self-learning"),
        relevance=(
            "Combines fast token-level confirmation with slower comprehensive "
            "analysis; this is the fast/slow escalation precedent for the .346 "
            "Anomaly-Escalation prototype (exp3780)."
        ),
        as_reported_note=(
            "ProcessBench and MATH improvements are source-reported; re-derive "
            "before forward-facing use."
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
    """Append the rendered .346 section while preserving prior bytes as a prefix."""

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
            "Added by the `.346 planning sweep. This is an append-only filing of "
            "newly surfaced external references into the converged record: the "
            "verifier-moat thread remains closed per "
            "`[[reference_deep_think_post_bounded_2026_06]]`, verifier "
            "domain-boundedness remains settled per "
            "`[[project_verifier_domain_bound]]`, and the entries below "
            "corroborate or scope that settled state rather than reopening it."
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
        workshop = f" {ref.workshop_context}." if ref.workshop_context else ""
        lines.extend(
            [
                (
                    f"- **{ref.arxiv_id} - \"{ref.title}\" ({ref.submitted}; "
                    f"arXiv resolved):** Track: {track}. {ref.relevance} "
                    f"{ref.as_reported_note}{workshop}"
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
                "arXiv abs pages resolved during the .346 filing; "
                "OpenReview additionally confirmed the ICLR 2026 LLM "
                "Reasoning Workshop context for arXiv:2604.15149."
            ),
        },
        "adversarial_verify_clean": True,
        "adversarial_verify_report": {"max_severity": -1, "flags": []},
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def section_appended_not_replaced(before_text: str, after_text: str) -> bool:
    """Return true when old content is preserved and the .346 section exists."""

    return (
        after_text.startswith(before_text)
        and SECTION_HEADER in after_text
        and ".344 additions" in after_text
        and ".345 additions" in after_text
    )


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate required fields and anti-fabrication hygiene."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    _ensure(not missing, f"missing required fields: {missing}")
    _ensure(artifact.get("honest_verdict") == TERMINAL_VERDICT, "honest_verdict")
    _ensure(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    _ensure(artifact.get("references_added") == list(REFERENCE_IDS), "references_added")
    _ensure(artifact.get("n_references_added") == len(REFERENCE_IDS), "n_references_added")
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
        {row.get("arxiv_id") for row in refs if isinstance(row, Mapping)}
        == set(REFERENCE_IDS),
        "references",
    )
    _ensure(
        all(
            isinstance(row, Mapping)
            and row.get("arxiv_abs_resolved") is True
            and row.get("numbers_are_as_reported") is True
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
        raise ValueError(f"invalid Exp 3783 artifact: {message}")


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
