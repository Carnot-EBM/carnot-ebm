#!/usr/bin/env python3
"""Exp 3768: extend and wire the Paper-v6 G3 narrowing lint."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import paper_v6_narrowing_lint as narrowing_lint


RESULT_PATH = REPO_ROOT / "results" / "experiment_3768_g3_narrowing_lint.json"
PRECOMMIT_PATH = REPO_ROOT / ".pre-commit-config.yaml"
TEST_PATH = REPO_ROOT / "tests" / "python" / "test_paper_v6_narrowing_lint.py"
LINT_PATH = REPO_ROOT / "scripts" / "paper_v6_narrowing_lint.py"
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "publication" / "spec.md"
RANDOM_SEED = 3768
INFERENCE_SUBSTRATE = (
    "aggregation_from_upstream_artifacts (principle: a lint over tracked docs, no live model)."
)
ENERGY_PATTERN_NAME = "ENERGY_AS_GENERATOR_GENERATIVE_SUCCESS"
ENERGY_RETRACTION_CONTROLS = (
    "energy-as-generator works at scale",
    "energy-as-generator scales",
    "EBT generates tokens",
    "EBT generates text",
    "energy-as-generator viable as a generator",
)
FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix; the G3-mechanization outcome.",
    "inference_substrate": INFERENCE_SUBSTRATE,
    "lint_extended_and_wired": (
        "True iff the EXISTING scripts/paper_v6_narrowing_lint.py was extended "
        "(12th retraction + paper_v6_* scan) AND the pre-commit stanza was added "
        "-- the G3 mechanization deliverable (extend+wire, not regenerate)."
    ),
    "n_retracted_phrasings_covered": (
        "Count of forbidden phrasings encoded (>=12 incl. the new "
        "energy-as-generator one) -- coverage of the discipline."
    ),
    "twelfth_retraction_added": (
        "True iff the 'energy-as-generator works/scales' phrasing is in the lint "
        "-- keeps the narrowing current with the Thesis-A bound."
    ),
    "violations_found": (
        "BARE int. Violations in the current tree (0 = docs clean; >0 = "
        "operator-action signal, never an auto-edit)."
    ),
    "precommit_hook_wired": (
        "True iff the .pre-commit-config.yaml stanza was added (additively) -- "
        "enforcement, not honor-discipline."
    ),
    "test_asserts_real_behavior": (
        "True iff the shipped test asserts the lint fires on retracted prose and "
        "passes on clean prose (anti-poison-test)."
    ),
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Content hash catches drift.",
    "duration_s": "Wall-clock plausibility floor.",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scan_text(text: str) -> list[narrowing_lint.LintHit]:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        doc = root / "paper.md"
        doc.write_text(text, encoding="utf-8")
        return narrowing_lint.scan_paths([doc], root=root)


def _twelfth_retraction_added() -> bool:
    pattern_names = {spec.name for spec in narrowing_lint.FORBIDDEN_PATTERNS}
    if ENERGY_PATTERN_NAME not in pattern_names:
        return False
    for control in ENERGY_RETRACTION_CONTROLS:
        hits = _scan_text(f"The draft claims {control}.")
        if not any(hit.pattern_name == ENERGY_PATTERN_NAME for hit in hits):
            return False
    clean_hits = _scan_text(
        "The EBT is discriminative-not-generative at tested scale."
    )
    return clean_hits == []


def _precommit_hook_wired() -> bool:
    text = PRECOMMIT_PATH.read_text(encoding="utf-8")
    required = (
        "id: paper-v6-narrowing-lint",
        "entry: python3 scripts/paper_v6_narrowing_lint.py",
        "docs/arxiv-paper/main\\.tex",
        "docs/technical-report\\.md",
        "results/paper_v6_.*\\.json",
        "pass_filenames: false",
    )
    return all(fragment in text for fragment in required)


def _test_asserts_real_behavior() -> bool:
    text = TEST_PATH.read_text(encoding="utf-8")
    return all(
        fragment in text
        for fragment in (
            "SCENARIO-PUBLISH-3768",
            "clean_doc_passes",
            "energy_as_generator_retraction_fails",
            "energy-as-generator works at scale",
            "assert (not hits) is should_pass",
        )
    )


def _paper_v6_json_scan_extended() -> bool:
    return bool(
        narrowing_lint.RESULTS_ARTIFACT_RE.match("results/paper_v6_fixture.json")
    )


def _artifact_allowlisted() -> bool:
    return narrowing_lint.should_skip(RESULT_PATH, root=REPO_ROOT)


def _checksum(payload: dict[str, Any]) -> str:
    stable = {
        "artifact": {
            k: payload[k]
            for k in sorted(payload)
            if k not in {"duration_s", "reproducibility_checksum"}
        },
        "lint_sha256": _sha256(LINT_PATH),
        "test_sha256": _sha256(TEST_PATH),
        "precommit_sha256": _sha256(PRECOMMIT_PATH),
        "spec_sha256": _sha256(SPEC_PATH),
    }
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_artifact() -> dict[str, Any]:
    started = time.perf_counter()
    target_paths = narrowing_lint.discover_targets(REPO_ROOT)
    hits = narrowing_lint.scan_paths(target_paths, root=REPO_ROOT)
    twelfth_added = _twelfth_retraction_added()
    precommit_wired = _precommit_hook_wired()
    test_real = _test_asserts_real_behavior()
    paper_v6_scan = _paper_v6_json_scan_extended()
    artifact_allowlisted = _artifact_allowlisted()
    n_phrasings = len(narrowing_lint.FORBIDDEN_PATTERNS)
    lint_extended_and_wired = bool(
        twelfth_added
        and precommit_wired
        and paper_v6_scan
        and artifact_allowlisted
        and n_phrasings >= 12
    )
    violations_found = len(hits)
    honest_verdict = (
        "complete: "
        f"g3_narrowing_lint_shipped_{n_phrasings}_phrasings_"
        f"12th_added_violations_{violations_found}_precommit_wired"
        if lint_extended_and_wired and test_real
        else "complete: g3_narrowing_lint_incomplete_acceptance_gate_unmet"
    )

    artifact: dict[str, Any] = {
        "schema": "carnot.g3_narrowing_lint.v1",
        "experiment": 3768,
        "experiment_id": 3768,
        "honest_verdict": honest_verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "lint_extended_and_wired": lint_extended_and_wired,
        "n_retracted_phrasings_covered": n_phrasings,
        "twelfth_retraction_added": twelfth_added,
        "violations_found": violations_found,
        "precommit_hook_wired": precommit_wired,
        "test_asserts_real_behavior": test_real,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "duration_s": max(time.perf_counter() - started, 0.0001),
        "target_files_existing": len(target_paths),
        "target_paths": [
            narrowing_lint._rel_string(path, REPO_ROOT) for path in target_paths
        ],
        "violating_files": sorted(
            {narrowing_lint._rel_string(hit.path, REPO_ROOT) for hit in hits}
        ),
        "paper_v6_json_scan_extended": paper_v6_scan,
        "experiment_artifact_allowlisted": artifact_allowlisted,
        "energy_retraction_pattern_name": ENERGY_PATTERN_NAME,
        "field_principles": FIELD_PRINCIPLES,
    }
    artifact["reproducibility_checksum"] = _checksum(artifact)
    return artifact


def write_artifact(artifact: dict[str, Any]) -> None:
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    artifact = build_artifact()
    write_artifact(artifact)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    print(artifact["honest_verdict"])
    return 0 if artifact["lint_extended_and_wired"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
