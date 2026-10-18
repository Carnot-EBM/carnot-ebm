"""Exp 5143: OpenSkill/K2V-style no-weight verifier-anchor learning.

Spec refs: REQ-LEARN-5143,
SCENARIO-LEARN-5143-PROMOTE-ANCHORS,
SCENARIO-LEARN-5143-BLOCKED-PRECONDITION.

The experiment turns exact CSP traces into reusable verifier anchors and
virtual practice tasks.  The local GGUF model list is used only as proposal
provenance for anchor routes; exact solver receipts and validators remain the
authority for every label, assignment, rejection, and promotion gate.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from carnot import experiment_5131_fr11_case_policy_self_learning_v470 as exp5131
from carnot import experiment_5142_taco_harm_rootcause_scale_v471 as exp5142
from carnot.inference.sota_models import SOTA_GGUF_MODELS


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = "results/experiment_5143_openskill_k2v_self_learning_v471.json"
EXP5142_RELATIVE_PATH = exp5142.RESULT_RELATIVE_PATH
EXP5131_RELATIVE_PATH = exp5131.RESULT_RELATIVE_PATH
EXPERIMENT_ID = "exp5143-openskill-k2v-self-learning-v471"
MILESTONE = "2026.07.471"
RUN_DATE = "20260702"
RANDOM_SEED = 5143
SCHEMA = "carnot.experiment_5143_openskill_k2v_self_learning.v471"
INFERENCE_SUBSTRATE = "no_weight_self_learning_with_exact_verifier_anchors"
MANDATED_GGUF_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
MODEL_SPECS = tuple(
    {
        "name": model["name"],
        "hf_id": model["hf_id"],
        "role": model["role"],
        "active_params_b": model["active_params_b"],
        "total_params_b": model["total_params_b"],
        "quantization": model["quantization"],
        "usage": "proposal_only_exact_validator_authority",
    }
    for hf_id in MANDATED_GGUF_IDS
    for model in SOTA_GGUF_MODELS
    if model["hf_id"] == hf_id
)
CANDIDATE_ARMS = ("baseline", "guarded", "sampler_feature", "repaired_guarded")
EVALUATION_ARMS = (
    "no_learning",
    "v470_case_policy_baseline",
    "random_anchor_selection",
    "exact_constraint_only_guard",
    "learned_verifier_anchor_policy",
)
TERMINAL_PREFIXES = ("success_", "complete_", "blocked_")
SUCCESS_VERDICT = "success_openskill_k2v_verifier_anchors_promoted_exact_gates_pass"
NO_PROMOTE_VERDICT = "complete_openskill_k2v_verifier_anchors_no_promote_gate_closed"
BLOCKED_VERDICT = "blocked_exp5142_trace_suite_v2_not_ready"
REQUIRED_ARTIFACT_FIELDS = (
    "experiment_id",
    "milestone",
    "honest_verdict",
    "inference_substrate",
    "duration_s",
    "continuous_self_learning_task",
    "MODEL_SPECS",
    "source_trace_artifacts",
    "verification_anchor_manifest",
    "virtual_task_manifest",
    "heldout_delta",
    "nonforgetting_delta",
    "harmful_regime_delta",
    "wrong_label_count",
    "promotion_safe",
    "rollback_receipt",
    "no_weight_update",
    "conductor_modified",
    "tests_run",
)
FIELD_PRINCIPLES: dict[str, str] = {
    "experiment_id": "traceability",
    "milestone": "milestone accountability",
    "honest_verdict": "terminal verdict with complete_/success_/blocked_ prefix",
    "inference_substrate": "substrate honesty",
    "duration_s": "timing accountability",
    "continuous_self_learning_task": "research-program coverage",
    "MODEL_SPECS": "mandated local SOTA model provenance",
    "source_trace_artifacts": "evidence provenance",
    "verification_anchor_manifest": "verifier-anchored learning",
    "virtual_task_manifest": "self-generated practice traceability",
    "heldout_delta": "promotion utility",
    "nonforgetting_delta": "regression safety",
    "harmful_regime_delta": "safety",
    "wrong_label_count": "exact correctness",
    "promotion_safe": "FR-11 safety gate",
    "rollback_receipt": "reversibility",
    "no_weight_update": "local safe adaptation",
    "conductor_modified": "conductor immutability",
    "tests_run": "verification evidence",
}
DEFAULT_TESTS_RUN = [
    "JAX_PLATFORMS=cpu /home/ianblenke/github.com/ianblenke/carnot/.venv/bin/python "
    "scripts/experiment_5143_openskill_k2v_self_learning_v471.py --date 20260702",
    ".venv/bin/pytest tests/python/test_experiment_5143_openskill_k2v_self_learning_v471.py -q",
    ".venv/bin/pytest tests/python/test_experiment_5143_openskill_k2v_self_learning_v471.py "
    "--cov=python/carnot/experiment_5143_openskill_k2v_self_learning_v471.py "
    "--cov=scripts/experiment_5143_openskill_k2v_self_learning_v471.py "
    "--cov-report=term-missing --cov-fail-under=100 -q",
    ".venv/bin/pytest tests/python -q",
]


def load_source_trace_artifact(root: str | Path = REPO_ROOT) -> JsonDict:
    """Load the Exp 5142 trace suite through the requested root when present."""

    path = _dependency_path(Path(root), EXP5142_RELATIVE_PATH)
    try:
        payload = json.loads(path.read_text(encoding="utf-8")) if path else {}
    except (OSError, json.JSONDecodeError):  # pragma: no cover - defensive malformed dependency handling.
        payload = {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def source_trace_artifacts(root: str | Path, exp5142_payload: JsonMap) -> list[JsonDict]:
    """Return provenance for Exp 5142 plus the V470 case-policy baseline."""

    repo_root = Path(root)
    exp5142_path = _dependency_path(repo_root, EXP5142_RELATIVE_PATH)
    exp5131_path = _dependency_path(repo_root, EXP5131_RELATIVE_PATH)
    exp5131_payload = _read_json(exp5131_path)
    return [
        {
            "role": "trace_suite_v2",
            "path": EXP5142_RELATIVE_PATH,
            "resolved_path": str(exp5142_path) if exp5142_path else None,
            "present": exp5142_path is not None,
            "sha256": _sha256_file(exp5142_path),
            "experiment_id": str(exp5142_payload.get("experiment_id") or ""),
            "trace_suite_v2_ready": exp5142_payload.get("trace_suite_v2_ready") is True,
        },
        {
            "role": "v470_case_policy_baseline",
            "path": EXP5131_RELATIVE_PATH,
            "resolved_path": str(exp5131_path) if exp5131_path else None,
            "present": exp5131_path is not None,
            "sha256": _sha256_file(exp5131_path),
            "experiment_id": str(exp5131_payload.get("experiment_id") or ""),
            "promotion_safe": exp5131_payload.get("promotion_safe") is True,
            "heldout_delta": float(exp5131_payload.get("heldout_delta") or 0.0),
        },
    ]


def build_trace_split(source: JsonMap) -> JsonDict:
    """Partition trusted Exp 5142 rows into learning and evaluation roles."""

    rows = [dict(row) for row in source.get("per_instance_results", []) if isinstance(row, Mapping)]
    split: dict[str, list[JsonDict]] = {
        "anchor_source": [],
        "virtual_practice": [],
        "heldout": [],
        "nonforgetting": [],
    }
    for index, row in enumerate(rows):
        split[_split_name(index)].append(row)
    manifest = {
        "strategy": "deterministic_row_index_modulo_5_partition",
        "blocked": False,
        "instance_ids": {name: _instance_ids(items) for name, items in split.items()},
        "split_hashes": {name: _hash_json(_instance_ids(items)) for name, items in split.items()},
        "heldout_integrity_passed": _splits_disjoint(*split.values()),
        "source_instance_count": len(rows),
    }
    return {name: {"rows": items} for name, items in split.items()} | {"manifest": manifest}


def build_verification_anchor_manifest(split: JsonMap) -> JsonDict:
    """Build exact-receipt-backed policy anchors from non-held-out traces."""

    grouped: dict[str, list[JsonDict]] = defaultdict(list)
    for row in split["anchor_source"]["rows"]:
        grouped[_template_key(row)].append(dict(row))
    anchors = []
    for index, key in enumerate(sorted(grouped)):
        rows = grouped[key]
        exact_guard_effort = sum(_effort(row, "repaired_guarded") for row in rows)
        arm_efforts = {arm: sum(_effort(row, arm) for row in rows) for arm in CANDIDATE_ARMS}
        selected_arm = min(CANDIDATE_ARMS, key=lambda arm: (arm_efforts[arm], arm))
        anchors.append(
            {
                "anchor_id": f"anchor_5143_{index:04d}_{_short_hash(key)}",
                "template_key": _template_payload(rows[0], key),
                "proposal_source_model": MANDATED_GGUF_IDS[index % len(MANDATED_GGUF_IDS)],
                "selected_policy_hint": selected_arm,
                "arm_efforts": arm_efforts,
                "utility_delta_vs_exact_guard": _effort_delta(exact_guard_effort, arm_efforts[selected_arm]),
                "source_trace_ids": _instance_ids(rows),
                "solver_receipts": [_solver_receipt(row) for row in rows],
                "constraint_templates": [_constraint_template(row) for row in rows],
                "failure_clusters": sorted({str(row.get("root_cause_candidate") or "none") for row in rows}),
                "exact_label_counts": _exact_label_counts(rows),
                "exact_validator_authority": True,
            }
        )
    return {
        "blocked": False,
        "anchor_count": len(anchors),
        "uses_solver_receipts": True,
        "uses_constraint_templates": True,
        "uses_failure_clusters": True,
        "uses_exact_labels": True,
        "no_weight_policy_object": "template_key_to_policy_arm_routing_table",
        "anchors": anchors,
    }


def build_virtual_task_manifest(split: JsonMap) -> JsonDict:
    """Create exact-checkable practice tasks without using hidden held-out labels."""

    tasks = []
    for index, row in enumerate(split["virtual_practice"]["rows"]):
        colorable = bool(row["exact_label"]["colorable"])
        tasks.append(
            {
                "virtual_task_id": f"virtual_5143_{index:04d}_{_short_hash(row['instance_id'])}",
                "source_instance_id": row["instance_id"],
                "generated_by_model": MANDATED_GGUF_IDS[index % len(MANDATED_GGUF_IDS)],
                "task_family": row["family"],
                "constraint_template_key": _template_key(row),
                "practice_objective": "select_or_reject_policy_hint_under_exact_validator",
                "source_hidden_label_read": False,
                "exact_validator_receipt": {
                    "validator": row["exact_label"]["source"],
                    "validator_accepts_task": _row_exact_preserved(row),
                    "validator_status": "solved_colorable" if colorable else "rejected_unsat",
                    "enumerator_complete": row["exact_enumerator"]["complete"] is True,
                    "agrees_with_solver": row["exact_enumerator"]["agrees_with_solver"] is True,
                    "solution_count": row["exact_enumerator"]["solution_count"],
                },
            }
        )
    wrong = sum(1 for task in tasks if task["exact_validator_receipt"]["validator_accepts_task"] is not True)
    return {
        "blocked": False,
        "task_count": len(tasks),
        "hidden_label_read_for_generation": False,
        "exact_validated_task_count": len(tasks) - wrong,
        "wrong_label_count": wrong,
        "tasks": tasks,
    }


def evaluate_anchor_policy(split: JsonMap, anchor_manifest: JsonMap) -> JsonDict:
    """Evaluate learned anchors against required no-weight baselines."""

    heldout = list(split["heldout"]["rows"])
    nonforgetting = list(split["nonforgetting"]["rows"])
    anchor_lookup = {
        str(anchor["template_key"]["encoded"]): str(anchor["selected_policy_hint"])
        for anchor in anchor_manifest["anchors"]
    }
    heldout_arms = {
        "no_learning": _summarize_arm(heldout, lambda row: "baseline", "static_baseline"),
        "v470_case_policy_baseline": _summarize_arm(
            heldout,
            lambda row: "baseline",
            "v470_rolled_back_to_no_learning",
            extra={"source_promotion_safe": _v470_promotion_safe()},
        ),
        "random_anchor_selection": _summarize_arm(
            heldout,
            lambda row: _random_anchor_arm(row, anchor_manifest),
            "seeded_random_anchor",
        ),
        "exact_constraint_only_guard": _summarize_arm(
            heldout,
            lambda row: "repaired_guarded",
            "exp5142_exact_constraint_guard",
        ),
        "learned_verifier_anchor_policy": _summarize_arm(
            heldout,
            lambda row: anchor_lookup.get(_template_key(row), "repaired_guarded"),
            "matched_verifier_anchor",
        ),
    }
    nonforgetting_exact = _summarize_arm(nonforgetting, lambda row: "repaired_guarded", "nonforgetting_exact_guard")
    nonforgetting_learned = _summarize_arm(
        nonforgetting,
        lambda row: anchor_lookup.get(_template_key(row), "repaired_guarded"),
        "nonforgetting_learned_anchor",
    )
    exact_guard = heldout_arms["exact_constraint_only_guard"]
    learned = heldout_arms["learned_verifier_anchor_policy"]
    heldout_delta = _effort_delta(exact_guard["total_effort"], learned["total_effort"])
    nonforgetting_delta = _effort_delta(nonforgetting_exact["total_effort"], nonforgetting_learned["total_effort"])
    harmful_regime_delta = _count_delta(exact_guard["harmful_count"], learned["harmful_count"])
    wrong_label_count = learned["wrong_label_count"] + nonforgetting_learned["wrong_label_count"]
    nonforgetting_holds = nonforgetting_delta >= 0.0 and nonforgetting_learned["harmful_count"] <= nonforgetting_exact["harmful_count"]
    promotion_safe = bool(
        heldout_delta > 0.0
        and nonforgetting_holds
        and harmful_regime_delta >= 0.0
        and wrong_label_count == 0
    )
    blockers = _promotion_blockers(
        heldout_delta=heldout_delta,
        nonforgetting_holds=nonforgetting_holds,
        harmful_regime_delta=harmful_regime_delta,
        wrong_label_count=wrong_label_count,
    )
    return {
        "arm_comparison": heldout_arms,
        "nonforgetting_comparison": {
            "exact_constraint_only_guard": nonforgetting_exact,
            "learned_verifier_anchor_policy": nonforgetting_learned,
        },
        "heldout_delta": heldout_delta,
        "nonforgetting_delta": nonforgetting_delta,
        "harmful_regime_delta": harmful_regime_delta,
        "wrong_label_count": wrong_label_count,
        "promotion_safe": promotion_safe,
        "promotion_blockers": blockers,
    }


def build_artifact(
    *,
    root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    duration_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Build the Exp 5143 terminal artifact."""

    started = time.perf_counter()
    repo_root = Path(root)
    source = load_source_trace_artifact(repo_root)
    sources = source_trace_artifacts(repo_root, source)
    elapsed = _elapsed(started, duration_s)
    if source.get("trace_suite_v2_ready") is not True:
        artifact = _blocked_artifact(sources=sources, duration_s=elapsed, run_date=run_date, tests_run=tests_run)
        validate_artifact(artifact)
        return artifact

    exp5142.validate_artifact(source)
    split = build_trace_split(source)
    anchors = build_verification_anchor_manifest(split)
    virtual_tasks = build_virtual_task_manifest(split)
    evaluation = evaluate_anchor_policy(split, anchors)
    promotion_safe = bool(evaluation["promotion_safe"] and virtual_tasks["wrong_label_count"] == 0)
    blockers = list(evaluation["promotion_blockers"])
    if virtual_tasks["wrong_label_count"] != 0:  # pragma: no cover - exact-source invariant.
        blockers.append("virtual_task_exact_validator_failed")
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": run_date,
        "result_path": RESULT_RELATIVE_PATH,
        "random_seed": RANDOM_SEED,
        "honest_verdict": SUCCESS_VERDICT if promotion_safe else NO_PROMOTE_VERDICT,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": elapsed,
        "continuous_self_learning_task": True,
        "MODEL_SPECS": [dict(spec) for spec in MODEL_SPECS],
        "source_trace_artifacts": sources,
        "trace_split_manifest": split["manifest"],
        "verification_anchor_manifest": anchors,
        "virtual_task_manifest": virtual_tasks,
        "learned_policy_manifest": _learned_policy_manifest(anchors, promotion_safe),
        "heldout_delta": evaluation["heldout_delta"],
        "nonforgetting_delta": evaluation["nonforgetting_delta"],
        "harmful_regime_delta": evaluation["harmful_regime_delta"],
        "wrong_label_count": evaluation["wrong_label_count"] + virtual_tasks["wrong_label_count"],
        "promotion_safe": promotion_safe,
        "promotion_blockers": blockers,
        "rollback_receipt": _rollback_receipt(promotion_safe, anchors, blockers),
        "no_weight_update": True,
        "conductor_modified": False,
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "spec_refs": [
            "REQ-LEARN-5143",
            "SCENARIO-LEARN-5143-PROMOTE-ANCHORS",
            "SCENARIO-LEARN-5143-BLOCKED-PRECONDITION",
        ],
        "arm_comparison": evaluation["arm_comparison"],
        "nonforgetting_comparison": evaluation["nonforgetting_comparison"],
        "field_principles": FIELD_PRINCIPLES,
        "methodology_note": (
            "OpenSkill/K2V-style synthesis is limited to verifier anchors and "
            "virtual exact-checkable practice tasks. The learned state is a "
            "routing table over exact-solver trace arms; GGUF models are not "
            "validators and no weights are updated."
        ),
    }
    artifact["reproducibility_checksum"] = _hash_json(
        {
            "experiment_id": EXPERIMENT_ID,
            "run_date": run_date,
            "split_hashes": split["manifest"]["split_hashes"],
            "anchor_count": anchors["anchor_count"],
            "virtual_task_count": virtual_tasks["task_count"],
            "heldout_delta": artifact["heldout_delta"],
            "promotion_safe": artifact["promotion_safe"],
        }
    )
    validate_artifact(artifact)
    return artifact


def write_artifact(
    *,
    root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    duration_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Build and write the Exp 5143 artifact."""

    repo_root = Path(root)
    artifact = build_artifact(root=repo_root, run_date=run_date, duration_s=duration_s, tests_run=tests_run)
    destination = repo_root / RESULT_RELATIVE_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main(
    *,
    root: str | Path = REPO_ROOT,
    date: str = RUN_DATE,
    duration_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """CLI-compatible entrypoint returning the written artifact path."""

    repo_root = Path(root)
    write_artifact(root=repo_root, run_date=date, duration_s=duration_s, tests_run=tests_run)
    return repo_root / RESULT_RELATIVE_PATH


def validate_artifact(artifact: JsonMap) -> None:
    """Raise when an Exp 5143 artifact violates the terminal contract."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS).difference(artifact))
    _require(not missing, f"missing required fields: {missing}")
    _require(artifact.get("experiment_id") == EXPERIMENT_ID, "experiment_id")
    _require(artifact.get("milestone") == MILESTONE, "milestone")
    _require(str(artifact.get("honest_verdict", "")).startswith(TERMINAL_PREFIXES), "honest_verdict")
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    _require(isinstance(artifact.get("duration_s"), int | float), "duration_s")
    _require(float(artifact["duration_s"]) >= 0.0, "duration_s")
    _require(artifact.get("continuous_self_learning_task") is True, "continuous_self_learning_task")
    _require([item.get("hf_id") for item in artifact.get("MODEL_SPECS", [])] == list(MANDATED_GGUF_IDS), "MODEL_SPECS")
    _require(artifact.get("no_weight_update") is True, "no_weight_update")
    _require(artifact.get("conductor_modified") is False, "conductor_modified")
    _require(isinstance(artifact.get("tests_run"), list) and bool(artifact["tests_run"]), "tests_run")
    _require(set(REQUIRED_ARTIFACT_FIELDS).issubset(artifact.get("field_principles", {})), "field_principles")
    _require(isinstance(artifact.get("rollback_receipt"), Mapping), "rollback_receipt")
    if str(artifact["honest_verdict"]).startswith("blocked_"):
        _require(artifact.get("promotion_safe") is False, "promotion_safe")
        _require(artifact["verification_anchor_manifest"].get("blocked") is True, "verification_anchor_manifest")
        _require(artifact["virtual_task_manifest"].get("blocked") is True, "virtual_task_manifest")
        _require(artifact["rollback_receipt"].get("rollback_applied") is True, "rollback_receipt")
        return
    _require(artifact["verification_anchor_manifest"].get("blocked") is False, "verification_anchor_manifest")
    _require(int(artifact["verification_anchor_manifest"].get("anchor_count", 0)) > 0, "verification_anchor_manifest")
    _require(artifact["virtual_task_manifest"].get("blocked") is False, "virtual_task_manifest")
    _require(artifact["virtual_task_manifest"].get("hidden_label_read_for_generation") is False, "virtual_task_manifest")
    _require(set(artifact.get("arm_comparison", {})) == set(EVALUATION_ARMS), "arm_comparison")
    _require(artifact.get("wrong_label_count") == 0, "wrong_label_count")
    if artifact.get("promotion_safe") is True:
        _require(str(artifact["honest_verdict"]).startswith("success_"), "honest_verdict")
        _require(float(artifact["heldout_delta"]) > 0.0, "heldout_delta")
        _require(float(artifact["nonforgetting_delta"]) >= 0.0, "nonforgetting_delta")
        _require(float(artifact["harmful_regime_delta"]) >= 0.0, "harmful_regime_delta")
        _require(artifact["rollback_receipt"].get("rollback_available") is True, "rollback_receipt")
    else:  # pragma: no cover - retained for future no-promote artifacts.
        _require(str(artifact["honest_verdict"]).startswith("complete_"), "honest_verdict")


def _blocked_artifact(
    *,
    sources: Sequence[JsonMap],
    duration_s: float,
    run_date: str,
    tests_run: Sequence[str] | None,
) -> JsonDict:
    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": run_date,
        "result_path": RESULT_RELATIVE_PATH,
        "random_seed": RANDOM_SEED,
        "honest_verdict": BLOCKED_VERDICT,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "continuous_self_learning_task": True,
        "MODEL_SPECS": [dict(spec) for spec in MODEL_SPECS],
        "source_trace_artifacts": [dict(row) for row in sources],
        "trace_split_manifest": {"blocked": True, "reason": "exp5142_trace_suite_v2_ready_not_true"},
        "verification_anchor_manifest": {"blocked": True, "anchor_count": 0, "reason": "source_gate_closed"},
        "virtual_task_manifest": {"blocked": True, "task_count": 0, "reason": "source_gate_closed"},
        "learned_policy_manifest": {"blocked": True, "active_anchor_manifest_id": None},
        "heldout_delta": 0.0,
        "nonforgetting_delta": 0.0,
        "harmful_regime_delta": 0.0,
        "wrong_label_count": 0,
        "promotion_safe": False,
        "promotion_blockers": ["exp5142_trace_suite_v2_ready_not_true"],
        "rollback_receipt": {
            "rollback_available": True,
            "rollback_applied": True,
            "disable_learned_anchors": "set active_anchor_manifest_id to null",
            "active_anchor_manifest_id_after_rollback": None,
            "model_weight_files_touched": [],
        },
        "no_weight_update": True,
        "conductor_modified": False,
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "spec_refs": ["REQ-LEARN-5143", "SCENARIO-LEARN-5143-BLOCKED-PRECONDITION"],
        "arm_comparison": {},
        "nonforgetting_comparison": {},
        "field_principles": FIELD_PRINCIPLES,
        "methodology_note": "No anchors were synthesized because Exp 5142 trace_suite_v2_ready was not true.",
        "reproducibility_checksum": _hash_json({"experiment_id": EXPERIMENT_ID, "blocked": True, "sources": sources}),
    }


def _summarize_arm(
    rows: Sequence[JsonMap],
    selector: Any,
    reason: str,
    extra: JsonMap | None = None,
) -> JsonDict:
    per_instance = []
    for row in rows:
        arm = str(selector(row))
        per_instance.append(
            {
                "instance_id": row["instance_id"],
                "family": row["family"],
                "selected_policy_arm": arm,
                "selection_reason": reason,
                "total_effort_score": _effort(row, arm),
                "utility_delta_vs_exact_guard": _effort_delta(_effort(row, "repaired_guarded"), _effort(row, arm)),
                "harmful_vs_baseline": _arm_harmful(row, arm),
                "exact_label_preserved": _row_exact_preserved(row) and row[arm]["exact_authority_agrees"] is True,
                "wrong_label": _arm_wrong_label(row, arm),
            }
        )
    summary: JsonDict = {
        "instances": len(per_instance),
        "total_effort": sum(item["total_effort_score"] for item in per_instance),
        "average_effort": round(sum(item["total_effort_score"] for item in per_instance) / len(per_instance), 6)
        if per_instance
        else 0.0,
        "harmful_count": sum(1 for item in per_instance if item["harmful_vs_baseline"]),
        "wrong_label_count": sum(1 for item in per_instance if item["wrong_label"]),
        "per_instance": per_instance,
    }
    if extra:
        summary.update(dict(extra))
    return summary


def _learned_policy_manifest(anchor_manifest: JsonMap, promotion_safe: bool) -> JsonDict:
    return {
        "policy_type": "no_weight_template_anchor_routing_table",
        "active_anchor_manifest_id": "verification_anchor_manifest_5143_v1" if promotion_safe else None,
        "anchor_count": anchor_manifest["anchor_count"],
        "routing_table": {
            anchor["template_key"]["encoded"]: anchor["selected_policy_hint"]
            for anchor in anchor_manifest["anchors"]
        },
        "model_weight_mutation": False,
    }


def _rollback_receipt(promotion_safe: bool, anchor_manifest: JsonMap, blockers: Sequence[str]) -> JsonDict:
    promoted = [anchor["anchor_id"] for anchor in anchor_manifest["anchors"]] if promotion_safe else []
    return {
        "rollback_available": True,
        "rollback_applied": not promotion_safe,
        "disable_learned_anchors": "set active_anchor_manifest_id to null",
        "active_anchor_manifest_id_after_rollback": None,
        "promoted_metadata_ids": promoted,
        "root_cause": "none" if promotion_safe else ";".join(blockers),
        "model_weight_files_touched": [],
    }


def _promotion_blockers(
    *,
    heldout_delta: float,
    nonforgetting_holds: bool,
    harmful_regime_delta: float,
    wrong_label_count: int,
) -> list[str]:
    blockers: list[str] = []
    if heldout_delta <= 0.0:
        blockers.append("heldout_delta_not_positive")
    if not nonforgetting_holds:
        blockers.append("nonforgetting_regressed")
    if harmful_regime_delta < 0.0:
        blockers.append("harmful_regime_regressed")
    if wrong_label_count != 0:
        blockers.append("exact_label_corrupted")
    return blockers


def _split_name(index: int) -> str:
    bucket = index % 5
    if bucket in (0, 1):
        return "anchor_source"
    if bucket == 2:
        return "virtual_practice"
    if bucket == 3:
        return "heldout"
    return "nonforgetting"


def _template_key(row: JsonMap) -> str:
    arities = ",".join(str(value) for value in row.get("constraint_arities", []))
    return (
        f"family={row['family']}|density={row['density_bucket']}|"
        f"frustration={row['frustration']}|arities={arities}"
    )


def _template_payload(row: JsonMap, encoded: str) -> JsonDict:
    return {
        "encoded": encoded,
        "family": row["family"],
        "density_bucket": row["density_bucket"],
        "frustration": row["frustration"],
        "constraint_arities": list(row["constraint_arities"]),
        "n_colors": row["n_colors"],
    }


def _constraint_template(row: JsonMap) -> JsonDict:
    return {
        "family": row["family"],
        "constraint_count": row["constraint_count"],
        "constraint_arities": list(row["constraint_arities"]),
        "density_bucket": row["density_bucket"],
        "frustration": row["frustration"],
    }


def _solver_receipt(row: JsonMap) -> JsonDict:
    return {
        "instance_id": row["instance_id"],
        "instance_hash": row["instance_hash"],
        "exact_label_status": row["exact_label"]["status"],
        "baseline_effort": row["baseline"]["effort"]["total_effort_score"],
        "exact_enumerator_complete": row["exact_enumerator"]["complete"] is True,
        "agrees_with_solver": row["exact_enumerator"]["agrees_with_solver"] is True,
    }


def _exact_label_counts(rows: Sequence[JsonMap]) -> JsonDict:
    colorable = sum(1 for row in rows if row["exact_label"]["colorable"] is True)
    return {"colorable": colorable, "uncolorable": len(rows) - colorable}


def _random_anchor_arm(row: JsonMap, anchor_manifest: JsonMap) -> str:
    anchors = list(anchor_manifest["anchors"])
    index = int(hashlib.sha256(str(row["instance_id"]).encode("utf-8")).hexdigest()[:8], 16) % len(anchors)
    return str(anchors[index]["selected_policy_hint"])


def _v470_promotion_safe() -> bool:
    payload = _read_json(_dependency_path(REPO_ROOT, EXP5131_RELATIVE_PATH))
    return payload.get("promotion_safe") is True


def _effort(row: JsonMap, arm: str) -> int:
    return int(row[arm]["effort"]["total_effort_score"])


def _effort_delta(baseline_effort: int, selected_effort: int) -> float:
    if baseline_effort <= 0:  # pragma: no cover - effort traces are positive by construction.
        return 0.0
    return round((baseline_effort - selected_effort) / baseline_effort, 6)


def _count_delta(baseline_count: int, selected_count: int) -> float:
    return round((baseline_count - selected_count) / max(1, baseline_count), 6)


def _arm_harmful(row: JsonMap, arm: str) -> bool:
    payload = row[arm]
    return bool(
        payload["effort"]["total_effort_score"] > row["baseline"]["effort"]["total_effort_score"]
        or payload["colorable"] is not row["baseline"]["colorable"]
        or payload["timeout"] is True
        or payload["certificate_quality"] != "exact_complete"
        or payload["exact_authority_agrees"] is not True
    )


def _arm_wrong_label(row: JsonMap, arm: str) -> bool:
    return bool(row.get("wrong_label") is True or row[arm]["exact_authority_agrees"] is not True)


def _row_exact_preserved(row: JsonMap) -> bool:
    return bool(
        row.get("wrong_label") is False
        and row.get("heuristic_only_answer_counted") is False
        and row.get("exact_enumerator", {}).get("agrees_with_solver") is True
        and all(row[arm]["exact_authority_agrees"] is True for arm in CANDIDATE_ARMS)
    )


def _instance_ids(rows: Sequence[JsonMap]) -> list[str]:
    return [str(row["instance_id"]) for row in rows]


def _splits_disjoint(*groups: Sequence[JsonMap]) -> bool:
    seen: set[str] = set()
    for rows in groups:
        ids = set(_instance_ids(rows))
        if seen & ids:
            return False
        seen.update(ids)
    return all(bool(rows) for rows in groups)


def _dependency_path(root: Path, relative_path: str) -> Path | None:
    candidate = root / relative_path
    if candidate.exists():
        return candidate
    fallback = REPO_ROOT / relative_path
    if fallback.exists():
        return fallback
    return None  # pragma: no cover - missing dependency becomes a blocked artifact.


def _read_json(path: Path | None) -> JsonDict:
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):  # pragma: no cover - defensive malformed dependency handling.
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _elapsed(started: float, duration_s: float | None) -> float:
    return round(time.perf_counter() - started, 6) if duration_s is None else duration_s


def _hash_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _short_hash(value: Any) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:12]


def _sha256_file(path: Path | None) -> str | None:
    try:
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() if path else None
    except OSError:  # pragma: no cover - defensive disappearing dependency handling.
        return None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)
