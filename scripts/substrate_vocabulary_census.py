#!/usr/bin/env python3
"""Count the `inference_substrate` vocabulary over results/experiment_*.json and report it.

REQ-SUBSTRATE-CENSUS-1 (openspec/capabilities/research-harnesses/spec.md).

WHY THIS EXISTS. CLAUDE.md's Inference-Substrate Declaration Discipline names six legal
values. Measured 2026-09-05: the corpus holds 1036 distinct strings, and the fabrication
gate recognises none of the 585 distinct values it calls "unknown". A commit-time hook
cannot see this: result artifacts are written once by conductor commits that skip hooks.
The 2026-08-29 known-issues entry asked for "a periodic full-corpus sweep, not a
commit-time hook, reporting the count rather than refusing anything". This is that sweep.

WHAT IT REPORTS. For every artifact: the declared value, its shape (string, principle-
wrapped, dict without a value, missing), what the gate's own classifier makes of it, and
the duration floor the gate would actually apply. Aggregates name each population.

WHAT IT DOES NOT DO. It never writes under results/. It never refuses. It never adds a
name to any allowlist. Exit code is 0 on a complete sweep and 2 if the results directory
cannot be read, so a caller can tell "counted zero" from "could not count".
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import adversarial_verify as av  # noqa: E402

PROJECT_ROOT = SCRIPTS_DIR.parent
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"

# The six values CLAUDE.md's table calls legal. Listed for MEASUREMENT only: this
# script reports how many artifacts use them. It does not enforce them.
LEGAL_VALUES_PER_CLAUDE_MD: tuple[str, ...] = (
    "live_llm_inference",
    "verifier_ensemble_against_cached_candidates",
    "aggregation_from_upstream_artifacts",
    "hardware_smoke",
    "offline_arcade_live_agent_runtime_self_discovery_no_llm",
    "live_llm_embedding_extraction",
)

# The gate picks a duration floor by a "reason" string. Each reason belongs to one of
# a few classes, grouped by how much model compute the floor assumes. This map is the
# measurement behind the class enum proposed in the 2026-09-05 research note.
EFFECTIVE_CLASS_OF_REASON: dict[str, str] = {
    "aggregation": "aggregation",
    "verifier_scoring": "no_model_load",
    "deterministic_verifier": "no_model_load",
    "no_llm_declared": "no_model_load",
    "no_llm_declared_by_name": "no_model_load",
    "arc_live_agent_no_llm": "no_model_load",
    "arc_live_agent_filter_runtime_no_llm": "no_model_load",
    "web_bibliographic_search_only": "no_model_load",
    "artifact_qa_lint_tests": "no_model_load",
    "deterministic_smt_hint_validation": "no_model_load",
    "log_analysis_local_timing": "no_model_load",
    "cheap_learned_value_scoring": "no_model_load",
    "llm_embedding_extraction": "model_load_no_generation",
    "local_sota_gguf_small_n": "model_bounded_generation",
    "native_gguf_backend_bisect": "model_bounded_generation",
    "live_model": "model_full_generation",
}
UNFLOORED = "unfloored"
# A `blocked_*` verdict makes the gate return no floor on purpose: the run stopped
# before any compute. That is a class of its own, not an ignored declaration.
BLOCKED_NO_RUN = "blocked_no_run"
SHAPE_MISSING = "missing"
SHAPE_STRING = "string"
SHAPE_WRAPPED = "principle_wrapped"
SHAPE_DICT_NO_VALUE = "dict_without_value"
SHAPE_OTHER = "other"

EXIT_OK = 0
EXIT_UNREADABLE_DIR = 2


def unwrap_substrate(value: Any) -> tuple[str, str]:
    """Return (shape, text) for a raw `inference_substrate` field.

    The gate stringifies a dict that has no `value` key, so `{"kind": ...}` becomes
    the text "{'kind': ...}" and matches nothing. This helper names that shape instead.
    """
    if value is None:
        return SHAPE_MISSING, ""
    if isinstance(value, str):
        text = value.strip()
        return (SHAPE_STRING, text) if text else (SHAPE_MISSING, "")
    if isinstance(value, dict):
        if "value" in value:
            inner = value.get("value")
            text = inner.strip() if isinstance(inner, str) else ""
            return (SHAPE_WRAPPED, text) if text else (SHAPE_MISSING, "")
        return SHAPE_DICT_NO_VALUE, ""
    return SHAPE_OTHER, ""


def effective_class(floor: dict[str, Any] | None, *, blocked: bool = False) -> str:
    """Map the gate's floor decision to the class it implies."""
    if not floor:
        return BLOCKED_NO_RUN if blocked else UNFLOORED
    reason = str(floor.get("reason", ""))
    return EFFECTIVE_CLASS_OF_REASON.get(reason, f"other:{reason}")


def census_artifact(d: dict[str, Any]) -> dict[str, Any]:
    """One row of the census for one artifact dict. Pure; reads nothing from disk."""
    shape, text = unwrap_substrate(d.get("inference_substrate"))
    # `gate_text` is what the gate compares. A dict with no `value` key is stringified
    # there, so the gate treats it as a declared value that matches nothing. The census
    # counts it the same way. A first version dropped these 169 artifacts from every
    # aggregate but `shapes`; the adversarial review of 2026-09-05 caught that.
    gate_text = av._inference_substrate_text(d)
    lead = av._substrate_leading_token(gate_text) if gate_text else ""
    blocked = av._is_precondition_check_only_blocked(d)
    classification = av._classify_inference_substrate(d)
    floor = av.duration_floor_for_artifact(d)
    return {
        "shape": shape,
        "raw": text,
        "gate_raw": gate_text,
        "lead": lead,
        "legal_exact": text in LEGAL_VALUES_PER_CLAUDE_MD,
        "legal_leading": bool(lead) and lead in LEGAL_VALUES_PER_CLAUDE_MD,
        "classifier_kind": classification["kind"],
        "classifier_source": classification["source"],
        "floor_reason": str(floor.get("reason")) if floor else "NO_FLOOR",
        "floor_s": floor.get("min_duration_s") if floor else None,
        "precondition_blocked": blocked,
        "effective_class": effective_class(floor, blocked=blocked),
        "duration_s": d.get("duration_s"),
    }


def iter_artifacts(results_dir: Path) -> list[tuple[Path, dict[str, Any] | None]]:
    """Every experiment_*.json with its parsed dict, or None when unreadable."""
    out: list[tuple[Path, dict[str, Any] | None]] = []
    for path in sorted(results_dir.glob("experiment_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            out.append((path, None))
            continue
        out.append((path, data if isinstance(data, dict) else None))
    return out


def census(results_dir: Path) -> dict[str, Any]:
    """Sweep one results directory and return named populations."""
    files = iter_artifacts(results_dir)
    rows: list[dict[str, Any]] = []
    unreadable = 0
    for path, data in files:
        if data is None:
            unreadable += 1
            continue
        row = census_artifact(data)
        row["file"] = path.name
        rows.append(row)

    # Two populations, named. P-declared is the gate's view: every artifact whose
    # stringified field is non-empty, dict-shaped included. P-string is the subset a
    # human wrote as a string. Keys without a suffix are over P-string; `gate_view`
    # and `dict_shaped_gate_view` are over the gate's view.
    gate_declared = [r for r in rows if r["gate_raw"]]
    declared = [r for r in gate_declared if r["raw"]]
    dict_shaped = [r for r in gate_declared if not r["raw"]]
    values = collections.Counter(r["raw"] for r in declared)
    leads = collections.Counter(r["lead"] for r in declared)
    by_source = collections.Counter(r["classifier_source"] for r in declared)
    by_floor = collections.Counter(r["floor_reason"] for r in declared)
    by_class = collections.Counter(r["effective_class"] for r in declared)
    distinct_per_class: dict[str, set[str]] = collections.defaultdict(set)
    for r in declared:
        distinct_per_class[r["effective_class"]].add(r["lead"])
    legal_per_value = {v: leads.get(v, 0) for v in LEGAL_VALUES_PER_CLAUDE_MD}
    unfloored_no_duration = sum(
        1 for r in declared if r["effective_class"] == UNFLOORED and not r["duration_s"]
    )
    unknown_values = sorted(
        {r["lead"] for r in declared if r["classifier_kind"] == av.SUBSTRATE_KIND_UNKNOWN}
    )

    def _view(pop: list[dict[str, Any]]) -> dict[str, Any]:
        distinct: dict[str, set[str]] = collections.defaultdict(set)
        for r in pop:
            distinct[r["effective_class"]].add(r["gate_raw"])
        return {
            "artifacts": len(pop),
            "classifier_source": dict(collections.Counter(r["classifier_source"] for r in pop)),
            "floor_reason": dict(collections.Counter(r["floor_reason"] for r in pop)),
            "effective_class": dict(collections.Counter(r["effective_class"] for r in pop)),
            "distinct_per_class": {k: len(v) for k, v in distinct.items()},
            "unknown_distinct_raw": len(
                {r["gate_raw"] for r in pop if r["classifier_kind"] == av.SUBSTRATE_KIND_UNKNOWN}
            ),
            "unfloored_without_duration": sum(
                1 for r in pop if r["effective_class"] == UNFLOORED and not r["duration_s"]
            ),
        }

    return {
        "results_dir": str(results_dir),
        "files": len(files),
        "unreadable": unreadable,
        "shapes": dict(collections.Counter(r["shape"] for r in rows)),
        "declared_gate_view": len(gate_declared),
        "declared_string": len(declared),
        "distinct_raw": len(values),
        "distinct_leading": len(leads),
        "singleton_raw": sum(1 for n in values.values() if n == 1),
        "legal_exact_artifacts": sum(1 for r in declared if r["legal_exact"]),
        "legal_leading_artifacts": sum(1 for r in declared if r["legal_leading"]),
        "legal_per_value": legal_per_value,
        "classifier_source": dict(by_source),
        "floor_reason": dict(by_floor),
        "effective_class": dict(by_class),
        "distinct_per_class": {k: len(v) for k, v in distinct_per_class.items()},
        "precondition_blocked": sum(1 for r in declared if r["precondition_blocked"]),
        "unfloored_without_duration": unfloored_no_duration,
        "unknown_distinct": len(unknown_values),
        "unknown_values": unknown_values,
        "top_leading": leads.most_common(40),
        "gate_view": _view(gate_declared),
        "dict_shaped_gate_view": _view(dict_shaped),
    }


def render(report: dict[str, Any], top: int = 25) -> str:
    """Plain-text report. One fact per line so a reader can grep it."""
    lines = [
        f"substrate census over {report['results_dir']}",
        f"files={report['files']} unreadable={report['unreadable']}",
        f"shapes={report['shapes']}",
        f"declared_gate_view={report['declared_gate_view']} (P-declared: what the gate compares)",
        "-- keys below without a view prefix are over P-string (string-valued declarations) --",
        (
            f"declared_string={report['declared_string']} distinct_raw={report['distinct_raw']} "
            f"distinct_leading={report['distinct_leading']} singleton_raw={report['singleton_raw']}"
        ),
        (
            f"CLAUDE.md legal values: exact={report['legal_exact_artifacts']} "
            f"leading={report['legal_leading_artifacts']} per_value={report['legal_per_value']}"
        ),
        f"classifier_source={report['classifier_source']}",
        f"floor_reason={report['floor_reason']}",
        f"effective_class={report['effective_class']}",
        f"distinct_per_class={report['distinct_per_class']}",
        f"precondition_blocked={report['precondition_blocked']}",
        f"unfloored_without_duration={report['unfloored_without_duration']}",
        f"unknown_distinct={report['unknown_distinct']}",
        f"gate_view (P-declared)={report['gate_view']}",
        f"dict_shaped_gate_view (dicts the gate stringifies)={report['dict_shaped_gate_view']}",
        f"top {top} leading tokens:",
    ]
    for token, n in report["top_leading"][:top]:
        lines.append(f"  {n:5d}  {token}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args(argv)
    if not args.results_dir.is_dir():
        print(f"substrate-census: cannot read {args.results_dir}", file=sys.stderr)
        return EXIT_UNREADABLE_DIR
    report = census(args.results_dir)
    if args.json:
        print(json.dumps(report, indent=1, default=list))
    else:
        print(render(report, top=args.top))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
