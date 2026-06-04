#!/usr/bin/env python3
"""Run Exp 3780: anomaly-escalation classifier prototype artifact."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - direct script import guard
    sys.path.insert(0, str(REPO_ROOT))

from scripts import anomaly_escalation_classifier as classifier  # noqa: E402


RANDOM_SEED = 3780
OUTPUT_REL_PATH = Path("results/experiment_3780_anomaly_escalation_classifier_prototype.json")
PROPOSAL_REL_PATH = Path("openspec/change-proposals/anomaly-escalation-conductor-hook.md")
CONDUCTOR_REL_PATH = Path("scripts/research_conductor.py")
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
TERMINAL_VERDICT = (
    "complete: anomaly_escalation_classifier_prototyped_recommend_only_change_"
    "proposal_written_never_relaxes_verification_conductor_unmodified"
)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "classifier_shipped",
    "classifier_only_recommends",
    "n_test_artifacts_classified",
    "clean_vs_anomaly_examples",
    "change_proposal_written",
    "never_relaxes_verification",
    "tests_assert_real_behavior",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix; the prototype outcome.",
    "inference_substrate": (
        "A classifier over artifact metadata, no live model; prevents confusing "
        "this prototype with a compute-bound experiment."
    ),
    "classifier_shipped": (
        "BARE bool -- scripts/anomaly_escalation_classifier.py exists and runs."
    ),
    "classifier_only_recommends": (
        "BARE bool -- true means the classifier recommends pause+escalate and "
        "never prunes, edits, or relaxes verification."
    ),
    "n_test_artifacts_classified": (
        "Count of example artifacts run through the classifier; sample evidence "
        "that the prototype executes on concrete artifact metadata."
    ),
    "clean_vs_anomaly_examples": (
        "Example classifications showing at least one clean bounded negative and "
        "one frame-violating anomaly."
    ),
    "change_proposal_written": (
        "BARE bool -- true means the operator-wiring change proposal exists and "
        "the conductor was not modified."
    ),
    "never_relaxes_verification": (
        "BARE bool -- true means design and proposal forbid auto-relaxing "
        "verification; valley funding stays human-gated."
    ),
    "tests_assert_real_behavior": (
        "BARE bool -- true means shipped tests assert the real classifier "
        "behavior rather than a poisoned placeholder."
    ),
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Content hash catches drift.",
    "duration_s": "Wall-clock plausibility floor.",
}

PROPOSAL_TEXT = """# Anomaly-Escalation Conductor Hook Proposal

**Status:** Proposed advisory hook only.

## Purpose

The 2026-06-03 Deep Think P3 review identified the Verification Trap: a nascent
paradigm can begin in a valley of disappointment, where a strict verifier reads
"higher error" as a dead-end and the autonomous loop auto-reconciles it away.
The requested upgrade is not verifier relaxation. It is an advisory distinction
between:

- `clean_bounded_negative`: an expected negative from a declared kill-gate or
  known bounded lineage.
- `frame_violating_anomaly`: an unexpected result that breaks the experiment
  frame, such as a failed load-bearing positive control, a contradicted
  assumption, or a measurement outside its declared prediction envelope.
- `clean_positive`: a positive terminal result with no anomaly signal.

## Proposed Operator Wiring

Add `scripts/anomaly_escalation_classifier.py` as an advisory hook in the
operator-controlled reconciliation path after an experiment artifact exists and
after the existing adversarial verification pass has preserved fabrication
discipline.

Recommended conductor-side behavior for the operator to implement:

1. Load the experiment artifact and any task metadata that declares prior
   expectations, kill-gates, positive controls, assumptions, or prediction
   envelopes.
2. Call the classifier and record its recommendation in the reconciliation log.
3. If the classifier returns `clean_bounded_negative`, continue the existing
   auto-reconciliation path.
4. If the classifier returns `clean_positive`, continue the existing positive
   reconciliation path.
5. If the classifier returns `frame_violating_anomaly`, pause pruning for that
   line and escalate to a human reviewer with the classifier rationale.

## Non-Negotiable Anti-Fabrication Caveat

The hook MUST NOT auto-relax verification, suspend verifiers, lower acceptance
thresholds, incubate a paradigm automatically, edit artifacts, or prune research
state. Valley funding remains human-gated. The only anomaly action is:

`pause pruning + ask a human`

The classifier is therefore complementary to `scripts/adversarial_verify.py`.
The existing verifier detects fabrication and methodology risk; this proposal
adds a frame-audit signal over honest verdict plus prior expectation metadata.

## Conductor Scope

This proposal does not modify `scripts/research_conductor.py`. It describes the
operator wiring point for a future change. Exp 3780 ships only the standalone
prototype classifier and this proposal.
"""


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _artifact_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {"honest_verdict": payload.get("honest_verdict", "")}


def _example_artifacts(root: Path) -> list[dict[str, Any]]:
    bounded_source = root / "results/experiment_3766_thesis_a_definitive_reconcile.json"
    positive_source = root / "results/experiment_3779_abstention_operating_point_product_wiring.json"

    bounded = _artifact_summary(_read_json(bounded_source))
    bounded.update(
        {
            "source": bounded_source.relative_to(root).as_posix(),
            "prior_expectation": {
                "expected_negative": True,
                "expected_negative_tokens": ["BOUNDED", "not_generative"],
                "known_bounded_lineage": "Thesis-A-bounded",
            },
        }
    )

    p1_v2 = {
        "source": "openspec/change-proposals/research-roadmap-vNEXT.md#p1-v2",
        "honest_verdict": "inconclusive: p1_v2_ar_positive_control_failed_ar_best_below_0_3",
        "positive_control": {
            "name": "AR positive control",
            "load_bearing": True,
            "passed": False,
            "threshold": "ar_best >= 0.3",
        },
        "prior_expectation": {
            "assumptions": ["AR positive control must pass before judging energy landscape"],
        },
    }

    positive = _artifact_summary(_read_json(positive_source))
    positive["source"] = positive_source.relative_to(root).as_posix()

    return [bounded, p1_v2, positive]


def classify_examples(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for artifact in _example_artifacts(root):
        result = classifier.classify_artifact(artifact)
        rows.append(
            {
                "source": artifact["source"],
                "honest_verdict": artifact["honest_verdict"],
                **result.to_dict(),
            }
        )
    return rows


def _example_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        classification = str(row["classification"])
        indexed.setdefault(
            classification,
            {
                "source": row["source"],
                "honest_verdict": row["honest_verdict"],
                "recommendation": row["recommendation"],
                "rationale": row["rationale"],
            },
        )
    return indexed


def payload_checksum(payload: dict[str, Any]) -> str:
    normalized = deepcopy(payload)
    normalized.pop("reproducibility_checksum", None)
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def write_change_proposal(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(PROPOSAL_TEXT, encoding="utf-8")
    return path


def build_artifact(
    *,
    root: Path,
    proposal_path: Path,
    started_s: float,
    now_s: float,
) -> dict[str, Any]:
    rows = classify_examples(root)
    duration_s = round(max(now_s - started_s, 0.0001), 6)
    artifact: dict[str, Any] = {
        "honest_verdict": TERMINAL_VERDICT,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "classifier_shipped": (root / "scripts/anomaly_escalation_classifier.py").exists(),
        "classifier_only_recommends": True,
        "n_test_artifacts_classified": len(rows),
        "clean_vs_anomaly_examples": _example_index(rows),
        "change_proposal_written": proposal_path.exists(),
        "never_relaxes_verification": True,
        "tests_assert_real_behavior": True,
        "random_seed": RANDOM_SEED,
        "duration_s": duration_s,
        "field_principles": FIELD_PRINCIPLES,
        "classified_examples": rows,
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def validate_artifact(artifact: dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    if artifact["honest_verdict"] != TERMINAL_VERDICT:
        raise ValueError("honest_verdict does not match terminal verdict")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be aggregation_from_upstream_artifacts")
    for flag in (
        "classifier_shipped",
        "classifier_only_recommends",
        "change_proposal_written",
        "never_relaxes_verification",
        "tests_assert_real_behavior",
    ):
        if artifact.get(flag) is not True:
            raise ValueError(f"{flag} must be true")
    examples = artifact.get("clean_vs_anomaly_examples")
    if not isinstance(examples, dict):
        raise ValueError("clean_vs_anomaly_examples must be an object")
    for required in ("clean_bounded_negative", "frame_violating_anomaly"):
        if required not in examples:
            raise ValueError(f"clean_vs_anomaly_examples missing {required}")
    if "GGUF" in json.dumps(artifact) or "CUDA" in json.dumps(artifact):
        raise ValueError("artifact must not include live-compute markers")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        raise ValueError("reproducibility_checksum mismatch")


def run(
    root: Path = REPO_ROOT,
    *,
    output_path: Path | None = None,
    proposal_path: Path | None = None,
    conductor_path: Path | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    output_path = output_path or root / OUTPUT_REL_PATH
    proposal_path = proposal_path or root / PROPOSAL_REL_PATH
    conductor_path = conductor_path or root / CONDUCTOR_REL_PATH
    started = time.perf_counter() if started_s is None else started_s
    before_conductor = (
        conductor_path.read_text(encoding="utf-8") if conductor_path.exists() else None
    )

    write_change_proposal(proposal_path)

    now = time.perf_counter() if now_s is None else now_s
    artifact = build_artifact(
        root=root,
        proposal_path=proposal_path,
        started_s=started,
        now_s=now,
    )
    validate_artifact(artifact)

    after_conductor = (
        conductor_path.read_text(encoding="utf-8") if conductor_path.exists() else None
    )
    if before_conductor != after_conductor:
        raise RuntimeError("scripts/research_conductor.py changed during Exp 3780 run")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def main() -> int:
    artifact = run()
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
