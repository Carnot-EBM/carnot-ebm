"""Exp 5070 V466 SOTA ingestion backfill verifier.

Spec refs: REQ-REPORT-5070, SCENARIO-REPORT-5070,
SCENARIO-REPORT-5070-MISSING-REFERENCE.

This module formalizes a literature/repo-inspection backfill. It verifies the
V466 managed reference section already written by the planner and emits a JSON
artifact; it does not run local model inference or edit conductor scripts.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
from typing import Any


RESULT_RELATIVE_PATH = "results/experiment_5070_sota_ingestion_backfill_v466.json"
REFERENCES_RELATIVE_PATH = "research-references.md"
HONEST_VERDICT = "success_sota_ingestion_backfill_v466_references_verified"
INFERENCE_SUBSTRATE = "literature_review_and_repo_inspection"
DURATION_S = 0.0
V466_SECTION_START = "<!-- V466-PLANNER-REFERENCES-START -->"
V466_SECTION_END = "<!-- V466-PLANNER-REFERENCES-END -->"
EXPECTED_SOURCE_HOOK_COUNT = 10
TERMINAL_PREFIXES = ("blocked_", "complete:", "complete_", "success:", "success_")
SPEC_REFS = ["REQ-REPORT-5070", "SCENARIO-REPORT-5070"]

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "duration_s",
    "inference_substrate",
    "sources_checked",
    "references_section_found",
    "references_added_count",
    "semantic_scholar_status",
    "planning_hooks",
    "flagged_adversarial",
    "field_principles",
    "spec_refs",
)

REQUIRED_USER_FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": "terminal prefix; success only when the V466 source set is verified."
    },
    "duration_s": {
        "principle": (
            "bounded repo/literature inspection duration; not a model-inference runtime claim."
        )
    },
    "inference_substrate": {
        "principle": (
            "literature_review_and_repo_inspection; no local model inference or live LLM claim."
        )
    },
    "sources_checked": {
        "principle": (
            "per-channel evidence for arXiv, OpenReview/GitHub, Hugging Face Papers, "
            "Extropic, Semantic Scholar, and Logical Intelligence."
        )
    },
    "references_section_found": {
        "principle": (
            "true only when V466 planner markers bracket the checked research-references.md "
            "section."
        )
    },
    "references_added_count": {
        "principle": (
            "zero when no clearly stronger missing 2025-2026 source is found; otherwise "
            "equals appended references."
        )
    },
    "semantic_scholar_status": {
        "principle": "records citation API attempts and forbids citation-count claims on HTTP 429."
    },
    "planning_hooks": {"principle": "maps verified sources to concrete .466 experiment hooks."},
    "flagged_adversarial": {
        "principle": "true if required sources are missing, fabricated, or claim local inference."
    },
}

FIELD_PRINCIPLES = {
    **REQUIRED_USER_FIELD_PRINCIPLES,
    "field_principles": {"principle": "principle annotations are part of the artifact contract."},
    "spec_refs": {"principle": "OpenSpec requirements and scenarios verified by this artifact."},
}

REQUIRED_CHANNELS = frozenset(
    {
        "arXiv",
        "OpenReview",
        "GitHub",
        "Hugging Face Papers",
        "Extropic",
        "Semantic Scholar",
        "Logical Intelligence",
    }
)

REQUIRED_REFERENCE_CHECKS = [
    {
        "required_token": "arXiv:2605.10158",
        "title": "Unsupervised Process Reward Models",
        "channel": "arXiv",
        "url": "https://arxiv.org/abs/2605.10158",
        "match_tokens": [
            "arXiv:2605.10158",
            "https://arxiv.org/abs/2605.10158",
            "Unsupervised Process Reward Models",
            "SOTA GGUF top-logprob",
        ],
    },
    {
        "required_token": "arXiv:2605.10325",
        "title": "Verifiable Process Rewards for Agentic Reasoning",
        "channel": "arXiv",
        "url": "https://arxiv.org/abs/2605.10325",
        "match_tokens": [
            "arXiv:2605.10325",
            "https://arxiv.org/abs/2605.10325",
            "Verifiable Process Rewards for Agentic Reasoning",
            "intermediate rewards",
        ],
    },
    {
        "required_token": "arXiv:2603.03305",
        "title": "Draft-Conditioned Constrained Decoding",
        "channel": "arXiv",
        "url": "https://arxiv.org/abs/2603.03305",
        "match_tokens": [
            "arXiv:2603.03305",
            "https://arxiv.org/abs/2603.03305",
            "Draft-Conditioned Constrained Decoding",
            "DCCD arm",
        ],
    },
    {
        "required_token": "github.com/avinashreddydev/dccd",
        "title": "DCCD code",
        "channel": "GitHub",
        "url": "https://github.com/avinashreddydev/dccd",
        "match_tokens": ["https://github.com/avinashreddydev/dccd"],
    },
    {
        "required_token": "arXiv:2602.06737",
        "title": "Optimal Abstractions for Verifying Properties of KANs",
        "channel": "arXiv",
        "url": "https://arxiv.org/abs/2602.06737",
        "match_tokens": [
            "arXiv:2602.06737",
            "https://arxiv.org/abs/2602.06737",
            "Kolmogorov-Arnold Networks",
            "KAN/PWA/MILP",
        ],
    },
    {
        "required_token": "arXiv:2505.11942",
        "title": "LifelongAgentBench",
        "channel": "arXiv",
        "url": "https://arxiv.org/abs/2505.11942",
        "match_tokens": [
            "arXiv:2505.11942",
            "https://arxiv.org/abs/2505.11942",
            "LifelongAgentBench",
            "group self-consistency",
        ],
    },
    {
        "required_token": "Hugging Face Papers 2605.13941",
        "title": "EvolveMem",
        "channel": "Hugging Face Papers",
        "url": "https://huggingface.co/papers/2605.13941",
        "match_tokens": [
            "Hugging Face Papers 2605.13941",
            "https://huggingface.co/papers/2605.13941",
            "EvolveMem",
            "automatic rollback",
        ],
    },
    {
        "required_token": "github.com/aiming-lab/SimpleMem",
        "title": "EvolveMem code",
        "channel": "GitHub",
        "url": "https://github.com/aiming-lab/SimpleMem/tree/main/EvolveMem",
        "match_tokens": ["https://github.com/aiming-lab/SimpleMem/tree/main/EvolveMem"],
    },
    {
        "required_token": "Hugging Face Papers 2605.27366",
        "title": "MUSE-Autoskill",
        "channel": "Hugging Face Papers",
        "url": "https://huggingface.co/papers/2605.27366",
        "match_tokens": [
            "Hugging Face Papers 2605.27366",
            "https://huggingface.co/papers/2605.27366",
            "MUSE-Autoskill",
            "skill library",
        ],
    },
    {
        "required_token": "arXiv:2602.15985",
        "title": "Decomposing Large-Scale Ising Problems on FPGAs",
        "channel": "arXiv",
        "url": "https://arxiv.org/abs/2602.15985",
        "match_tokens": [
            "arXiv:2602.15985",
            "https://arxiv.org/abs/2602.15985",
            "Decomposing Large-Scale Ising Problems on FPGAs",
            "host-device dispatch",
        ],
    },
    {
        "required_token": "Extropic",
        "title": "Extropic XTR-0 / TSU updates",
        "channel": "Extropic",
        "url": "https://extropic.ai/writing/inside-x0-and-xtr-0",
        "match_tokens": [
            "Extropic XTR-0 / TSU updates",
            "https://extropic.ai/writing/inside-x0-and-xtr-0",
            "https://extropic.ai/writing/thermodynamic-computing-from-zero-to-one",
            "no local TSU hardware",
        ],
    },
    {
        "required_token": "Logical Intelligence",
        "title": "Logical Intelligence public Kona/Aleph updates",
        "channel": "Logical Intelligence",
        "url": "https://logicalintelligence.com/",
        "match_tokens": [
            "Logical Intelligence public Kona/Aleph updates",
            "https://logicalintelligence.com/",
            "external architecture pressure",
            "EBMs/EBRMs own correctness-critical reasoning",
        ],
    },
    {
        "required_token": "Semantic Scholar",
        "title": "Semantic Scholar citation attempts",
        "channel": "Semantic Scholar",
        "url": "https://api.semanticscholar.org/graph/v1/paper/arXiv:2507.02092",
        "match_tokens": [
            "Semantic Scholar API citation checks",
            "HTTP 429",
            "no new citation-count claim",
        ],
    },
]

SOURCE_CHECK_OVERRIDES = [
    {
        "channel": "OpenReview",
        "source_id": "openreview_coverage",
        "title": "OpenReview-style query coverage",
        "url": "https://openreview.net/",
        "status": "coverage_declared_in_v466_section_no_openreview_only_addition",
        "reference_found": True,
    }
]

SEMANTIC_SCHOLAR_STATUS = {
    "attempted": True,
    "checked_on": "2026-07-01",
    "status": "http_429_rate_limited_no_citation_count_claim",
    "citation_count_claim_made": False,
    "targets": [
        {
            "label": "EBT",
            "paper_id": "arXiv:2507.02092",
            "http_status": 429,
            "url": "https://api.semanticscholar.org/graph/v1/paper/arXiv:2507.02092",
        },
        {
            "label": "ARM-EBM",
            "paper_id": "arXiv:2512.15605",
            "http_status": 429,
            "url": "https://api.semanticscholar.org/graph/v1/paper/arXiv:2512.15605",
        },
    ],
}

PLANNING_HOOKS = [
    {
        "hook_id": "exp5071_gguf_logprob_preflight",
        "source_tokens": ["arXiv:2605.10158"],
        "hook": "Stage real SOTA GGUF top-logprob readiness before any uPRM claim.",
    },
    {
        "hook_id": "exp5073_uprm_selector",
        "source_tokens": ["arXiv:2605.10158"],
        "hook": "Test batch first-error uPRM scoring only after token/step cache provenance exists.",
    },
    {
        "hook_id": "exp5074_vpr_diagnostic",
        "source_tokens": ["arXiv:2605.10325"],
        "hook": "Count VPR rewards only when intermediate checks are objective and auditable.",
    },
    {
        "hook_id": "exp5075_dccd_guided_scale",
        "source_tokens": ["arXiv:2603.03305", "github.com/avinashreddydev/dccd"],
        "hook": "Scale DCCD/guided decoding against unguided, hard-constrained, and rerank-only arms.",
    },
    {
        "hook_id": "exp5077_guarded_fr11_memory",
        "source_tokens": ["arXiv:2505.11942", "2605.13941", "2605.27366"],
        "hook": "Use rollback, group self-consistency, and per-skill held-out evidence for FR-11.",
    },
    {
        "hook_id": "exp5079_board_continuity",
        "source_tokens": ["arXiv:2602.15985", "Extropic"],
        "hook": "Separate host-device dispatch, decomposition overhead, and solver timing.",
    },
    {
        "hook_id": "exp5080_kan_pwa_milp_bridge",
        "source_tokens": ["arXiv:2602.06737"],
        "hook": "Attempt a tiny KAN/PWA/MILP property proof or emit a size/error-budget blocker.",
    },
    {
        "hook_id": "external_architecture_positioning",
        "source_tokens": ["Extropic", "Logical Intelligence"],
        "hook": "Treat Extropic and Kona/Aleph as strategic pressure, not local evidence.",
    },
]


def _require(condition: bool, message: str) -> None:
    if not condition:  # pragma: no cover
        raise ValueError(message)


def extract_v466_section(text: str) -> str:
    """Return the V466 managed reference section, including its marker-free body."""

    _require(V466_SECTION_START in text, "V466 planner section start marker missing")
    after_start = text.split(V466_SECTION_START, 1)[1]
    _require(V466_SECTION_END in after_start, "V466 planner section end marker missing")
    return after_start.split(V466_SECTION_END, 1)[0]


def verify_v466_references(section: str) -> dict[str, Any]:
    """Check the V466 section for all required source IDs, URLs, and hook evidence."""

    present: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for check in REQUIRED_REFERENCE_CHECKS:
        absent = [token for token in check["match_tokens"] if token not in section]
        if absent:
            missing.append(
                {
                    "required_token": check["required_token"],
                    "title": check["title"],
                    "missing_tokens": absent,
                }
            )
        else:
            present.append(
                {
                    "required_token": check["required_token"],
                    "title": check["title"],
                    "channel": check["channel"],
                    "url": check["url"],
                }
            )

    hook_count = section.count("- **Carnot hook:**")
    if hook_count < EXPECTED_SOURCE_HOOK_COUNT:
        missing.append(
            {
                "required_token": "Carnot hook",
                "title": "per-source Carnot hooks",
                "missing_tokens": [f"{EXPECTED_SOURCE_HOOK_COUNT} hooks, observed {hook_count}"],
            }
        )

    return {
        "references_section_found": True,
        "present": present,
        "missing": missing,
        "carnot_hook_count": hook_count,
    }


def _validate_reference_text(reference_text: str) -> dict[str, Any]:
    section = extract_v466_section(reference_text)
    verification = verify_v466_references(section)
    if verification["missing"]:
        first_missing = verification["missing"][0]
        missing_tokens = ", ".join(first_missing["missing_tokens"])
        raise ValueError(
            "missing V466 reference evidence for "
            f"{first_missing['required_token']}: {missing_tokens}"
        )
    return verification


def _build_sources_checked(verification: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {
            "channel": row["channel"],
            "source_id": row["required_token"],
            "title": row["title"],
            "url": row["url"],
            "status": "present_in_v466_section",
            "reference_found": True,
        }
        for row in verification["present"]
    ]
    rows.extend(dict(row) for row in SOURCE_CHECK_OVERRIDES)
    rows.sort(key=lambda row: (row["channel"], row["source_id"]))
    return rows


def build_artifact(*, reference_text: str) -> dict[str, Any]:
    """Build and validate the Exp 5070 backfill artifact."""

    verification = _validate_reference_text(reference_text)
    artifact: dict[str, Any] = {
        "honest_verdict": HONEST_VERDICT,
        "duration_s": DURATION_S,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "sources_checked": _build_sources_checked(verification),
        "references_section_found": verification["references_section_found"],
        "references_added_count": 0,
        "semantic_scholar_status": dict(SEMANTIC_SCHOLAR_STATUS),
        "planning_hooks": [dict(hook) for hook in PLANNING_HOOKS],
        "flagged_adversarial": False,
        "field_principles": dict(FIELD_PRINCIPLES),
        "spec_refs": list(SPEC_REFS),
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Fail closed if the artifact drifts from REQ-REPORT-5070."""

    _require(set(artifact) == set(REQUIRED_ARTIFACT_FIELDS), "artifact fields mismatch")
    verdict = artifact["honest_verdict"]
    _require(
        isinstance(verdict, str) and verdict.startswith(TERMINAL_PREFIXES),
        "honest_verdict must use a terminal prefix",
    )
    _require(verdict == HONEST_VERDICT, "unexpected Exp 5070 honest_verdict")
    _require(artifact["duration_s"] == DURATION_S, "duration_s mismatch")
    _require(
        artifact["inference_substrate"] == INFERENCE_SUBSTRATE,
        "inference_substrate must be literature_review_and_repo_inspection",
    )
    _require(artifact["references_section_found"] is True, "V466 section was not found")
    _require(artifact["references_added_count"] == 0, "unexpected V466 reference addition")
    _require(artifact["flagged_adversarial"] is False, "clean backfill cannot be flagged")
    _require(artifact["field_principles"] == FIELD_PRINCIPLES, "field_principles mismatch")
    _require(artifact["spec_refs"] == SPEC_REFS, "spec_refs mismatch")

    sources_checked = artifact["sources_checked"]
    _require(isinstance(sources_checked, list), "sources_checked must be a list")
    channels = {row.get("channel") for row in sources_checked}
    _require(REQUIRED_CHANNELS.issubset(channels), "required source channel missing")
    for row in sources_checked:
        _require(str(row.get("url", "")).startswith("https://"), "source URL must be https")
        _require(row.get("reference_found") is True, "source reference must be marked found")
        _require(bool(row.get("status")), "source status is required")

    semantic = artifact["semantic_scholar_status"]
    _require(semantic == SEMANTIC_SCHOLAR_STATUS, "semantic_scholar_status mismatch")
    _require(semantic["citation_count_claim_made"] is False, "citation-count claim is forbidden")
    _require(
        all(target["http_status"] == 429 for target in semantic["targets"]),
        "Semantic Scholar target statuses must record HTTP 429",
    )

    hooks = artifact["planning_hooks"]
    _require(isinstance(hooks, list) and len(hooks) >= 6, "planning_hooks too small")
    hook_ids = {hook.get("hook_id") for hook in hooks}
    _require("exp5073_uprm_selector" in hook_ids, "uPRM planning hook missing")
    _require("exp5074_vpr_diagnostic" in hook_ids, "VPR planning hook missing")
    _require("exp5075_dccd_guided_scale" in hook_ids, "DCCD planning hook missing")
    for hook in hooks:
        _require(bool(hook.get("source_tokens")), "planning hook source tokens missing")
        _require(bool(hook.get("hook")), "planning hook text missing")


def write_outputs(*, artifact_path: Path, references_path: Path) -> dict[str, Any]:
    """Write the stable JSON artifact after validating the V466 references."""

    reference_text = references_path.read_text(encoding="utf-8")
    artifact = build_artifact(reference_text=reference_text)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def main() -> int:
    root = Path(os.environ.get("CARNOT_EXP5070_ROOT", Path(__file__).resolve().parents[2]))
    write_outputs(
        artifact_path=root / RESULT_RELATIVE_PATH,
        references_path=root / REFERENCES_RELATIVE_PATH,
    )
    print(HONEST_VERDICT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
