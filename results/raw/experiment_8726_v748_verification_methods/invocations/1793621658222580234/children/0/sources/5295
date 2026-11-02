"""Reconcile the operator's definitive Thesis-A direct runs.

Spec refs: REQ-REPORT-3766, SCENARIO-REPORT-3766.

This module is deliberately an aggregation step. The upstream artifacts already
contain the live training evidence; Exp 3766 only imports those checked-in JSON
records, updates the research-note menu, and writes a provenance-rich
corrigendum so incomplete in-loop kill-gate attempts are not mistaken for real
experimental evidence.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - direct import guard
    sys.path.insert(0, str(REPO_ROOT))

from scripts import adversarial_verify  # noqa: E402


OUTPUT_REL_PATH = Path("results/experiment_3766_thesis_a_definitive_reconcile.json")
PART_A_REL_PATH = Path("results/thesis_a_direct_definitive_run.json")
PART_B_REL_PATH = Path("results/thesis_a_part_b_scaled_seed1.json")
THESIS_MENU_REL_PATH = Path("docs/research-notes/phase3-alternative-thesis-menu.md")
EXCLUSION_MANIFEST_REL_PATH = Path("ops/exclusion_manifest.yaml")

RANDOM_SEED = 3766
INFERENCE_SUBSTRATE = (
    "aggregation_from_upstream_artifacts (principle: a record reconciliation over "
    "upstream JSON, no live model)."
)
TERMINAL_VERDICT = (
    "complete: "
    "thesis_a_definitive_reconciled_part_a_PASS_discriminative_part_b_BOUNDED_"
    "not_generative_in_loop_chain_superseded_menu_updated_not_retired"
)
PART_A_OUTCOME = (
    "PASS_discriminative: tiny 38M byte-EBT trained stably for 800 GSM8K steps "
    "and learned a held-out energy landscape"
)
PART_B_OUTCOME = (
    "BOUNDED_at_scale_not_generative: learned emb->token decoder plus 3-digit "
    "addition scale-up leaves EBT at 0.000 vs matched AR 0.840"
)
FIELD_CONSISTENCY_CITATION = (
    "EBT-Policy (arXiv:2510.27545): EBT-generation wins in low-dimensional "
    "continuous control, not discrete text at matched compute."
)
IN_LOOP_SUPERSESSION_REASON = (
    "Exp 3745-3750 in-loop EBT kill-gate chain is SUPERSEDED/OBE by the "
    "operator's direct runs because kill_zombies reaped unowned GPU processes; "
    "no incomplete in-loop result is treated as real evidence."
)
THESIS_MENU_NOTE = (
    "- **Exp 3766 definitive reconciliation (2026-06-03):** Thesis A is "
    "BOUNDED: part-(a) PASS/discriminative from "
    "`results/thesis_a_direct_definitive_run.json` (800 stable GSM8K steps; "
    "held-out margin 0.723 vs untrained 0.084, about 8.6x), but part-(b) "
    "BOUNDED/not-generative from `results/thesis_a_part_b_scaled_seed1.json` "
    "(learned emb->token decoder, 3-digit addition, 16k steps; EBT 0.000 under "
    "argmin and descent+decoder vs AR 0.820 greedy / 0.840 matched "
    "self-consistency). Field boundary: EBT-Policy (arXiv:2510.27545) shows "
    "EBT-generation wins in low-dimensional continuous control, not discrete "
    "text at matched compute."
)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "thesis_a_part_a_outcome",
    "thesis_a_part_b_outcome",
    "ebt_discriminative_not_generative",
    "in_loop_chain_superseded",
    "field_consistency_citation",
    "thesis_menu_updated",
    "not_added_to_exclusion_manifest",
    "cited_upstream_artifacts",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix; states the reconciliation outcome.",
    "inference_substrate": (
        "A record reconciliation over upstream JSON, no live model; keeps this "
        "artifact out of live-compute methodology claims."
    ),
    "thesis_a_part_a_outcome": (
        "The definitive part-(a) result (PASS / discriminative) -- the EBT "
        "trains stably and learns a held-out energy landscape."
    ),
    "thesis_a_part_b_outcome": (
        "The definitive part-(b) result (BOUNDED at scale / not-generative) -- "
        "the actual thesis signal; the milestone-defining bound."
    ),
    "ebt_discriminative_not_generative": (
        "BARE bool. The one-line scientific finding: the energy landscape "
        "discriminates but does not generate tokens at this scale."
    ),
    "in_loop_chain_superseded": (
        "Records that exp3745-3750 are OBE'd by the operator's direct runs -- "
        "the record-honesty deliverable."
    ),
    "field_consistency_citation": (
        "EBT-Policy (arXiv:2510.27545) frames the negative as field-consistent, "
        "not a Carnot-only artifact."
    ),
    "thesis_menu_updated": (
        "True iff the Phase-3 thesis menu reflects Thesis A bounded -- keeps "
        "the operator's next-thesis surface current."
    ),
    "not_added_to_exclusion_manifest": (
        "Energy-as-generator is recorded as a BOUND, not retired as a doomed "
        "rerun id; future human-seeded variants are not pre-blocked."
    ),
    "cited_upstream_artifacts": (
        "Provenance for the imported definitive numbers (anti-fabrication audit trail)."
    ),
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Content hash catches drift.",
    "duration_s": "Wall-clock plausibility floor.",
}

PART_A_FIELDS = (
    "honest_verdict",
    "ebt_param_count",
    "ar_param_count",
    "cumulative_steps_trained",
    "nan_or_divergence_events",
    "ebt_trained_stably",
    "ebt_learned_heldout",
    "ar_learned_heldout",
    "ebt_heldout_margin_final",
    "ebt_heldout_margin_untrained_baseline",
    "ar_heldout_ce_init",
    "ar_heldout_ce_final",
    "grad_norm_max",
    "grad_norm_mean",
    "peak_vram_mb",
    "stabilizers_applied",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)
PART_B_FIELDS = (
    "honest_verdict",
    "task",
    "training_diverged",
    "headroom_ok",
    "ar1_greedy_acc",
    "arV_selfconsistency_acc",
    "arK_selfconsistency_acc",
    "ebt_argmin_acc",
    "ebt_descent_decoder_acc",
    "best_ebt_acc",
    "matched_ar_acc",
    "delta_best_ebt_minus_matched_ar",
    "matched_compute.argmin_ratio",
    "matched_compute.descent_ratio",
    "matched_compute.K",
    "n_eval",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    adversarial_report: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Build the Exp 3766 reconciliation artifact from checked-in evidence."""

    root_path = Path(root)
    part_a = read_json_object(root_path / PART_A_REL_PATH)
    part_b = read_json_object(root_path / PART_B_REL_PATH)
    ensure_part_a_direct_run(part_a)
    ensure_part_b_direct_run(part_b)

    thesis_menu_updated = ensure_thesis_menu_bounded(root_path)
    not_added_to_exclusion_manifest = ensure_energy_generator_not_retired(root_path)
    report = compact_verify_report(adversarial_report or {"flags": []})
    payload: JsonDict = {
        "schema": "carnot.archive_activation.thesis_a_definitive_reconcile_3766.v1",
        "experiment_id": "exp3766",
        "task_id": "exp3766-thesis-a-definitive-reconcile",
        "honest_verdict": TERMINAL_VERDICT,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "thesis_a_part_a_outcome": PART_A_OUTCOME,
        "thesis_a_part_b_outcome": PART_B_OUTCOME,
        "ebt_discriminative_not_generative": True,
        "in_loop_chain_superseded": True,
        "field_consistency_citation": FIELD_CONSISTENCY_CITATION,
        "thesis_menu_updated": thesis_menu_updated,
        "not_added_to_exclusion_manifest": not_added_to_exclusion_manifest,
        "definitive_direct_runs": {
            "part_a_discriminative": direct_part_a_summary(part_a),
            "part_b_scaled_seed1": direct_part_b_summary(part_b),
        },
        "clean_corrigendum": {
            "superseded_chain_experiment_ids": [
                "exp3745",
                "exp3746",
                "exp3747",
                "exp3748",
                "exp3749",
                "exp3750",
            ],
            "supersession_status": "SUPERSEDED_OBE_by_operator_direct_runs",
            "supersession_reason": IN_LOOP_SUPERSESSION_REASON,
            "fabricated_in_loop_result_preserved": "none",
        },
        "cited_upstream_artifacts": [
            citation(
                "thesis_a_direct_definitive_run",
                root_path / PART_A_REL_PATH,
                part_a,
                PART_A_FIELDS,
            ),
            citation(
                "thesis_a_part_b_scaled_seed1",
                root_path / PART_B_REL_PATH,
                part_b,
                PART_B_FIELDS,
            ),
        ],
        "adversarial_verify_clean": report_is_clean(report),
        "adversarial_verify_report": report,
        "field_principles": dict(FIELD_PRINCIPLES),
        "random_seed": RANDOM_SEED,
        "duration_s": duration_from(started_s, now_s),
        "reproducibility_checksum": "",
    }
    payload["reproducibility_checksum"] = payload_checksum(payload)
    validate_artifact(payload)
    return payload


def run(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Write the reconciliation artifact after updating the Thesis-A menu."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    output_path = root_path / OUTPUT_REL_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_artifact(root_path, started_s=start, now_s=now_s)
    write_json(output_path, payload)

    verify_report = adversarial_verify.verify_artifact(output_path)
    payload["adversarial_verify_report"] = compact_verify_report(verify_report)
    payload["adversarial_verify_clean"] = report_is_clean(verify_report)
    payload["reproducibility_checksum"] = payload_checksum(payload)
    validate_artifact(payload)
    write_json(output_path, payload)
    return output_path


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the schema and the record-honesty constraints."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    ensure(not missing, f"missing required artifact fields: {missing}")
    principles = artifact.get("field_principles")
    ensure(isinstance(principles, Mapping), "field_principles must be a mapping")
    missing_principles = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in principles]
    ensure(not missing_principles, f"missing field principles: {missing_principles}")
    ensure("model_specs" not in artifact, "model_specs must not be present")
    ensure("target_model" not in artifact, "target_model must not be present")
    ensure(no_forbidden_markers(artifact), "GGUF/CUDA markers must not be present")
    ensure(artifact.get("honest_verdict") == TERMINAL_VERDICT, "terminal verdict mismatch")
    ensure(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference substrate mismatch")
    ensure(artifact.get("thesis_a_part_a_outcome") == PART_A_OUTCOME, "part-a outcome mismatch")
    ensure(artifact.get("thesis_a_part_b_outcome") == PART_B_OUTCOME, "part-b outcome mismatch")
    ensure(
        artifact.get("ebt_discriminative_not_generative") is True,
        "scientific finding must be discriminative-not-generative",
    )
    ensure(artifact.get("in_loop_chain_superseded") is True, "in-loop chain must be superseded")
    ensure(
        artifact.get("field_consistency_citation") == FIELD_CONSISTENCY_CITATION,
        "field consistency citation mismatch",
    )
    ensure(artifact.get("thesis_menu_updated") is True, "thesis menu must be updated")
    ensure(
        artifact.get("not_added_to_exclusion_manifest") is True,
        "exclusion manifest must not retire energy-as-generator",
    )
    direct_runs = artifact.get("definitive_direct_runs")
    ensure(isinstance(direct_runs, Mapping), "definitive_direct_runs must be present")
    part_b = direct_runs.get("part_b_scaled_seed1") if isinstance(direct_runs, Mapping) else None
    ensure(isinstance(part_b, Mapping), "part-b direct-run summary must be present")
    ensure(isinstance(part_b.get("matched_ar_acc_label"), str), "matched AR label must be a string")
    ensure_citations(artifact.get("cited_upstream_artifacts"))
    ensure(
        artifact.get("adversarial_verify_clean") is True,
        "adversarial verification must have no critical flag",
    )
    ensure(artifact.get("random_seed") == RANDOM_SEED, "random_seed must equal 3766")
    duration_s = artifact.get("duration_s")
    ensure(
        isinstance(duration_s, (int, float)) and not isinstance(duration_s, bool) and float(duration_s) >= 0.0001,
        "duration_s must be numeric with the aggregation floor",
    )
    checksum = artifact.get("reproducibility_checksum")
    ensure(is_sha256(checksum), "reproducibility_checksum must be a sha256 hex string")
    ensure(checksum == payload_checksum(artifact), "reproducibility_checksum does not match artifact content")


def ensure_part_a_direct_run(artifact: Mapping[str, Any]) -> None:
    """Fail closed unless part-(a) shows stable discriminative EBT learning."""

    ensure(artifact.get("ebt_trained_stably") is True, "part-a direct run must train stably")
    ensure(artifact.get("ebt_learned_heldout") is True, "part-a direct run must learn held-out energy")
    ensure(artifact.get("nan_or_divergence_events") is False, "part-a direct run must not diverge")
    ensure(artifact.get("cumulative_steps_trained") == 800, "part-a direct run must record 800 steps")
    ensure(
        number(artifact.get("ebt_heldout_margin_final")) >= 0.72
        and number(artifact.get("ebt_heldout_margin_untrained_baseline")) <= 0.085,
        "part-a direct run must preserve the learned margin evidence",
    )


def ensure_part_b_direct_run(artifact: Mapping[str, Any]) -> None:
    """Fail closed unless part-(b) preserves the scaled not-generative result."""

    ensure(artifact.get("training_diverged") is False, "part-b direct run must not diverge")
    ensure(artifact.get("headroom_ok") is True, "AR headroom must be present")
    ensure(artifact.get("ebt_argmin_acc") == 0.0, "part-b direct run must keep argmin EBT at 0.000")
    ensure(
        artifact.get("ebt_descent_decoder_acc") == 0.0,
        "part-b direct run must keep descent+decoder EBT at 0.000",
    )
    ensure(artifact.get("best_ebt_acc") == 0.0, "part-b direct run must keep best EBT at 0.000")
    ensure(number(artifact.get("ar1_greedy_acc")) >= 0.82, "part-b direct run must preserve AR greedy headroom")
    ensure(number(artifact.get("matched_ar_acc")) >= 0.84, "part-b direct run must preserve matched AR headroom")
    matched_compute = artifact.get("matched_compute")
    ensure(isinstance(matched_compute, Mapping), "part-b direct run must include matched-compute details")
    ensure(
        matched_compute.get("argmin_ratio") == 1.0 and matched_compute.get("descent_ratio") == 1.0,
        "part-b direct run must preserve matched-compute ratios",
    )


def direct_part_a_summary(artifact: Mapping[str, Any]) -> JsonDict:
    """Return the load-bearing part-(a) values without live-compute markers."""

    return {
        "source_artifact": str(PART_A_REL_PATH),
        "honest_verdict": artifact.get("honest_verdict"),
        "cumulative_steps_trained": artifact.get("cumulative_steps_trained"),
        "ebt_param_count": artifact.get("ebt_param_count"),
        "ar_param_count": artifact.get("ar_param_count"),
        "ebt_trained_stably": artifact.get("ebt_trained_stably"),
        "nan_or_divergence_events": artifact.get("nan_or_divergence_events"),
        "ebt_learned_heldout": artifact.get("ebt_learned_heldout"),
        "ar_learned_heldout": artifact.get("ar_learned_heldout"),
        "ebt_heldout_margin_final": artifact.get("ebt_heldout_margin_final"),
        "ebt_heldout_margin_untrained_baseline": artifact.get("ebt_heldout_margin_untrained_baseline"),
        "margin_summary": "pos/neg margin 0.723 vs untrained 0.084 (~8.6x)",
        "ar_heldout_ce_init": artifact.get("ar_heldout_ce_init"),
        "ar_heldout_ce_final": artifact.get("ar_heldout_ce_final"),
        "grad_norm_max": artifact.get("grad_norm_max"),
        "grad_norm_mean": artifact.get("grad_norm_mean"),
        "peak_vram_mb": artifact.get("peak_vram_mb"),
        "stabilizers_applied": artifact.get("stabilizers_applied"),
        "random_seed": artifact.get("random_seed"),
        "upstream_reproducibility_checksum": artifact.get("reproducibility_checksum"),
        "duration_s": artifact.get("duration_s"),
    }


def direct_part_b_summary(artifact: Mapping[str, Any]) -> JsonDict:
    """Return the load-bearing part-(b) values without duplicate numeric aliases."""

    matched_compute = artifact.get("matched_compute") if isinstance(artifact.get("matched_compute"), Mapping) else {}
    return {
        "source_artifact": str(PART_B_REL_PATH),
        "honest_verdict": artifact.get("honest_verdict"),
        "task": artifact.get("task"),
        "training_diverged": artifact.get("training_diverged"),
        "headroom_ok": artifact.get("headroom_ok"),
        "n_eval": artifact.get("n_eval"),
        "ar1_greedy_acc": artifact.get("ar1_greedy_acc"),
        "arK_selfconsistency_acc": artifact.get("arK_selfconsistency_acc"),
        "arV_selfconsistency_acc": artifact.get("arV_selfconsistency_acc"),
        "matched_ar_acc_label": str(artifact.get("matched_ar_acc")),
        "ebt_argmin_acc": artifact.get("ebt_argmin_acc"),
        "ebt_descent_decoder_acc": artifact.get("ebt_descent_decoder_acc"),
        "best_ebt_acc": artifact.get("best_ebt_acc"),
        "delta_best_ebt_minus_matched_ar": artifact.get("delta_best_ebt_minus_matched_ar"),
        "matched_compute_ratios": {
            "argmin_ratio": matched_compute.get("argmin_ratio"),
            "descent_ratio": matched_compute.get("descent_ratio"),
        },
        "matched_compute_counts": {
            "ebt_argmin_evals": matched_compute.get("ebt_argmin_evals"),
            "arV_forward": matched_compute.get("arV_forward"),
            "ebt_descent_evals": matched_compute.get("ebt_descent_evals"),
            "arK_forward": matched_compute.get("arK_forward"),
            "K": matched_compute.get("K"),
        },
        "random_seed": artifact.get("random_seed"),
        "upstream_reproducibility_checksum": artifact.get("reproducibility_checksum"),
        "duration_s": artifact.get("duration_s"),
    }


def ensure_thesis_menu_bounded(root: Path) -> bool:
    """Make the research-note menu explicitly show Thesis A as bounded."""

    path = root / THESIS_MENU_REL_PATH
    text = path.read_text(encoding="utf-8")
    updated = update_thesis_menu_text(text)
    if updated != text:
        path.write_text(updated, encoding="utf-8")
    return thesis_menu_is_updated(updated)


def update_thesis_menu_text(text: str) -> str:
    """Return menu text with one bounded heading and one Exp 3766 note."""

    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("## Thesis A") and "Energy as the GENERATOR" in line:
            if "BOUNDED" not in line:
                suffix = line[line.index("Energy as the GENERATOR") :]
                lines[index] = f"## Thesis A - BOUNDED: {suffix}"
            break
    updated = "\n".join(lines) + ("\n" if text.endswith("\n") else "")
    if "Exp 3766 definitive reconciliation" in updated:
        return updated
    marker = "\n## Thesis B"
    if marker in updated:
        return updated.replace(marker, f"\n{THESIS_MENU_NOTE}\n{marker}", 1)
    return f"{updated.rstrip()}\n\n{THESIS_MENU_NOTE}\n"


def thesis_menu_is_updated(text: str) -> bool:
    """Return true when the menu contains the definitive bounded Thesis-A record."""

    required_tokens = (
        "Thesis A",
        "BOUNDED",
        "Exp 3766 definitive reconciliation",
        "results/thesis_a_direct_definitive_run.json",
        "results/thesis_a_part_b_scaled_seed1.json",
        "arXiv:2510.27545",
    )
    return all(token in text for token in required_tokens)


def ensure_energy_generator_not_retired(root: Path) -> bool:
    """Return true unless the exclusion manifest wrongly retires the thesis."""

    manifest = (root / EXCLUSION_MANIFEST_REL_PATH).read_text(encoding="utf-8").lower()
    forbidden = ("energy-as-generator", "energy_as_generator", "thesis-a-generator")
    ensure(
        not any(token in manifest for token in forbidden),
        "exclusion manifest must not retire energy-as-generator",
    )
    return True


def citation(
    experiment_id: str,
    path: Path,
    artifact: Mapping[str, Any],
    fields: tuple[str, ...],
) -> JsonDict:
    """Build a citation row for imported fields and source-file hash."""

    imported = [field for field in fields if get_nested(artifact, field) is not None]
    return {
        "experiment_id": experiment_id,
        "fields_imported": imported,
        "sha256": sha256_path(path),
    }


def ensure_citations(citations: Any) -> None:
    """Validate the two direct-run citations."""

    ensure(isinstance(citations, list), "cited_upstream_artifacts must be a list")
    ids = {item.get("experiment_id") for item in citations if isinstance(item, Mapping)}
    ensure(
        ids == {"thesis_a_direct_definitive_run", "thesis_a_part_b_scaled_seed1"},
        "cited_upstream_artifacts must cite the two direct-run files",
    )
    for item in citations:
        ensure(isinstance(item, Mapping), "each citation must be an object")
        ensure(item.get("fields_imported"), "each citation must include fields_imported")
        ensure(is_sha256(item.get("sha256")), "each citation must include a sha256 hex string")


def compact_verify_report(report: Mapping[str, Any]) -> JsonDict:
    """Keep deterministic adversarial-verifier fields in the artifact."""

    flags = [dict(flag) for flag in report.get("flags", []) if isinstance(flag, Mapping)]
    severities = [severity_rank(flag.get("severity")) for flag in flags]
    return {
        "flag_count": len(flags),
        "max_severity": max(severities) if severities else -1,
        "flags": flags,
    }


def report_is_clean(report: Mapping[str, Any]) -> bool:
    """Return true when the adversarial report has no critical flag."""

    flags = report.get("flags", [])
    return not any(
        isinstance(flag, Mapping) and str(flag.get("severity", "")).lower() == "critical"
        for flag in flags
    )


def severity_rank(severity: Any) -> int:
    """Map verifier severities to a stable integer order."""

    return {"info": 0, "warn": 1, "critical": 2}.get(str(severity).lower(), -1)


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON object from disk and reject non-object payloads."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    ensure(isinstance(payload, dict), f"expected JSON object in {path}")
    return payload


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write deterministic JSON with a trailing newline."""

    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def get_nested(artifact: Mapping[str, Any], field: str) -> Any:
    """Return dotted-path values for citation field accounting."""

    current: Any = artifact
    for part in field.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def duration_from(started_s: float | None, now_s: float | None) -> float:
    """Compute a duration with the aggregation plausibility floor."""

    if started_s is None:
        return 0.0001
    end = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0001, end - float(started_s)), 6)


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Return a checksum over the payload, excluding the checksum field."""

    filtered = {key: value for key, value in payload.items() if key != "reproducibility_checksum"}
    encoded = json.dumps(filtered, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_path(path: Path) -> str:
    """Return the SHA-256 hash for a source artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_sha256(value: Any) -> bool:
    """Return true when the value is a lowercase SHA-256 hex digest."""

    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def no_forbidden_markers(value: Mapping[str, Any]) -> bool:
    """Return true when the aggregation artifact copied no live-compute markers."""

    encoded = json.dumps(value, sort_keys=True)
    forbidden = ("GGUF", "CUDA", "torch.cuda", ".cuda(", "live-model", "live_llm_inference")
    return all(marker not in encoded for marker in forbidden)


def number(value: Any) -> float:
    """Coerce numeric evidence to float and fail closed for non-numbers."""

    ensure(isinstance(value, (int, float)) and not isinstance(value, bool), "numeric evidence is required")
    return float(value)


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for the Exp 3766 reconciliation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    output_path = run(args.root)
    payload = read_json_object(output_path)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
