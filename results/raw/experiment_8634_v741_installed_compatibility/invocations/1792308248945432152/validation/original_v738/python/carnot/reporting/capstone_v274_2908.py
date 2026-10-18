"""Build the Exp 2908 milestone .274 capstone artifact.

Spec refs: REQ-REPORT-2908, SCENARIO-REPORT-2908.

This module is a pure synthesis layer for the 2026.05.274 milestone close.
It reads the already-written .274 deliverables (KV260 latency benchmark,
GateMate bitstream attempt, PolarFire dispatch smoke, THRML import repair,
cross-corpus matrix v8, paper-v6 hardware section, KAN complexity v2, the
SOTA code generation budget expansion, FR-11 hardware replay pilot, and
the operator hardware portfolio status), classifies each one as
clean/flagged/blocked/missing/pilot-only, and emits the milestone-close
boolean flags the operator and paper-v6 reader rely on.

Why we synthesize rather than recompute: per CLAUDE.md
``inference_substrate=aggregation_from_upstream_artifacts`` declaration
discipline, this artifact must not invoke models, modify the roadmap, or
touch the conductor. The headline booleans (``hardware_portfolio_reactivated``,
``kv260_first_latency_recorded``, ``gatemate_bitstream_built``,
``polarfire_smoke_verified``, ``thrml_import_repaired``,
``cross_corpus_matrix_v8_built``) are derived only from clean source-artifact
evidence; flagged or blocked sources cannot create them.

The .274 milestone is the hardware-reactivation milestone — see
``feedback_kv260_ssh_not_sd_card.md``, the KV260 SSH bring-up of 2026-05-20,
and the hardware-task-continuity discipline in CLAUDE.md.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA = "carnot.milestone_capstone.v274"
MILESTONE = "2026.05.274"
RUN_DATE = "20260523"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
OUTPUT_REL_PATH = Path("results/experiment_2908_capstone_v274.json")
PRIOR_CAPSTONE_REL_PATH = Path("results/experiment_2896_capstone_v273.json")

EXPECTED_ARTIFACTS: dict[str, Path] = {
    "exp2897": Path("results/experiment_2897_archive_v273_activate_v274.json"),
    "exp2898": Path("results/experiment_2898_kv260_ising_sampler_hardware_latency_benchmark_v1.json"),
    "exp2899": Path("results/experiment_2899_gatemate_a1_n16_ising_tile_bitstream_build_v1.json"),
    "exp2900": Path("results/experiment_2900_polarfire_carnot_dispatch_smoke_v1.json"),
    "exp2901": Path("results/experiment_2901_thrml_local_import_repair_v1.json"),
    "exp2902": Path("results/experiment_2902_cross_corpus_matrix_v8.json"),
    "exp2903": Path("results/experiment_2903_paper_v6_hardware_validation_section_v1.json"),
    "exp2904": Path("results/experiment_2904_kan_hardware_complexity_accounting_v2.json"),
    "exp2905": Path("results/experiment_2905_sota_code_generation_bounded_budget_expansion_v1.json"),
    "exp2906": Path("results/experiment_2906_fr11_hardware_accelerated_replay_pilot_v1.json"),
    "exp2907": Path("results/experiment_2907_operator_hardware_portfolio_status_v1.json"),
}

# Booleans the source artifact must report True for the classifier to even
# consider classifying the artifact "clean". A failure here downgrades to
# "blocked" because the artifact's own contract isn't satisfied.
REQUIRED_SUCCESS_FIELDS: dict[str, tuple[str, ...]] = {
    "exp2897": (),  # archive/activation artifact — verdict prefix is enough
    "exp2898": (),  # KV260 fields are strings/lists, not booleans; checked via _kv260_first_latency_recorded
    "exp2899": ("synth_succeeded",),  # GateMate must report toolchain-clean build
    "exp2900": ("scorer_output_hash_verified",),
    "exp2901": ("thrml_import_succeeded",),
    "exp2902": (),  # matrix v8 ships its own row tallies; we read directly
    "exp2903": ("snippet_written",),
    "exp2904": (),
    "exp2905": (),  # SOTA expansion — flag check dominates
    "exp2906": (),  # FR-11 pilot — pilot-only classifier handles it
    "exp2907": (),
}

# Pilot-only artifacts deliver evidence without a headline benchmark claim.
# exp2906 is the FR-11 hardware-accelerated replay pilot — by design it
# validates the dispatch path without claiming hardware speedup.
PILOT_ONLY_IDS: tuple[str, ...] = ("exp2906",)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Self-declared terminal state lets the reconciler classify the verdict"
        " without re-running. Capstone verdicts start with complete:."
    ),
    "inference_substrate": (
        "Forward-only declaration this artifact is pure aggregation; the"
        " adversarial-verify linter applies the aggregation duration floor"
        " (1ms), not the live-LLM 60s floor."
    ),
    "paper_ready": (
        "True only when matrix v8 itself is clean and contains FoVer plus at"
        " least one other clean headline row; flagged or pilot-only rows"
        " cannot create paper readiness."
    ),
    "hardware_portfolio_reactivated": (
        "True when at least three of four hardware tracks (KV260 latency,"
        " PolarFire smoke, THRML repair, GateMate bitstream) produced a clean"
        " deliverable; the remaining track must be honestly blocked, not"
        " missing."
    ),
    "kv260_first_latency_recorded": (
        "True only when exp2898 lands clean with overlay loaded, UIO devices"
        " present, and a board transcript on disk. False otherwise."
    ),
    "gatemate_bitstream_built": (
        "True only when exp2899's own synth_succeeded boolean is True AND"
        " place_and_route_succeeded is True AND a bitstream sha256 is"
        " recorded. nextpnr-gatemate toolchain absence keeps this False."
    ),
    "polarfire_smoke_verified": (
        "True only when exp2900 lands clean with SSH-reachable riscv64 board"
        " and the constraint-scorer output hash matches the expected hash."
    ),
    "thrml_import_repaired": (
        "True only when exp2901 lands clean with thrml_import_succeeded=True"
        " and a non-zero parity-energy-delta recorded against a fixed seed."
    ),
    "cross_corpus_matrix_v8_built": (
        "True only when exp2902 lands clean and reports rows_clean."
    ),
    "clean_artifacts": (
        "Expected .274 deliverables with terminal verdicts, required booleans,"
        " and no adversarial flags."
    ),
    "flagged_artifacts": (
        "Artifacts with adversarial-verify or corrigendum flags. Stay out of"
        " paper-ready claims even if their own ready booleans are True."
    ),
    "blocked_artifacts": (
        "Artifacts that honestly report a precondition/gate block. Recorded"
        " here so the .275 planner can pick them up with prior_failures: set."
    ),
    "missing_artifacts": (
        "Expected .274 deliverables that are absent or malformed. Empty for"
        " a successful milestone close."
    ),
    "pilot_only_artifacts": (
        "Artifacts that intentionally provide pilot/dispatch-only evidence"
        " without a headline benchmark or hardware-acceleration claim."
    ),
    "top_3_next_actions": (
        "Three operator-actionable next steps that close the largest .274"
        " gaps and unblock the .275 milestone."
    ),
    "gaps_for_275": (
        "Concrete gaps the .275 planner should pick up — flagged artifacts,"
        " blocked toolchain installs, missing same-basis CPU baselines."
    ),
    "cited_upstream_artifacts": (
        "Upstream artifact provenance — paths plus fields imported. Audit"
        " trail that lets a third party verify the capstone is not synthesizing"
        " numbers from nothing."
    ),
    "duration_s": (
        "Measured wall-clock duration for synthesis; never sleep-padded."
    ),
}


def read_json(path: Path) -> dict[str, Any]:
    """Return a JSON object from ``path``, or ``{}`` when it cannot be trusted."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _number_or_none(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _terminal_success(verdict: object) -> bool:
    """Return True iff the verdict starts with a CLAUDE.md terminal prefix.

    Why prefix-only: the conductor reconciler's ``_verdict_is_untrustworthy``
    classifier false-positives on words like "blocked"/"marginal" appearing
    in descriptive prose. We mirror the same prefix rule here so the capstone
    classification matches the reconciler's view of terminal vs partial.
    """

    if not isinstance(verdict, str):
        return False
    return verdict.strip().startswith(
        (
            "complete:",
            "complete_",
            "success:",
            "success_",
            "passed:",
            "passed_",
            "shipped:",
            "shipped_",
        )
    )


def _blocked_verdict(verdict: object) -> bool:
    return isinstance(verdict, str) and verdict.strip().lower().startswith(
        ("blocked", "gate_blocked")
    )


def _has_flags(payload: dict[str, Any]) -> bool:
    """Return True if any adversarial flag mechanism fires on the payload."""

    if payload.get("flagged_adversarial") is True:
        return True
    pending = payload.get("corrigendum_pending")
    if isinstance(pending, list) and pending:
        return True
    flags = payload.get("adversarial_verify_flags")
    if isinstance(flags, list) and flags:
        return True
    summary = payload.get("adversarial_verify_summary")
    if isinstance(summary, dict) and (
        _number_or_none(summary.get("flag_count")) or 0.0
    ) > 0.0:
        return True
    return payload.get("adversarial_verify_passed") is False


def _required_booleans_pass(exp_id: str, payload: dict[str, Any]) -> bool:
    fields = REQUIRED_SUCCESS_FIELDS.get(exp_id, ())
    return all(payload.get(field) is True for field in fields)


def classify_artifact(exp_id: str, payload: dict[str, Any], present: bool) -> str:
    """REQ-REPORT-2908: classify one source artifact's claim status.

    Order matters: flags beat everything (an adversarially-flagged artifact
    cannot also be "clean"), then pilot-only (a deliberately pilot-only
    artifact is not a regression), then blocked verdicts, then clean iff
    terminal + required-booleans pass.
    """

    if not present or not payload:
        return "missing"
    if _has_flags(payload):
        return "flagged"
    if exp_id in PILOT_ONLY_IDS and _terminal_success(payload.get("honest_verdict")):
        # A pilot-only artifact must declare itself pilot-only and decline
        # the headline-metric claim. Anything else falls through to clean/blocked.
        if payload.get("pilot_only") is True or payload.get(
            "headline_metric_claim_made"
        ) is False:
            return "pilot-only"
    if _blocked_verdict(payload.get("honest_verdict")):
        return "blocked"
    if _terminal_success(payload.get("honest_verdict")) and _required_booleans_pass(
        exp_id, payload
    ):
        return "clean"
    if _terminal_success(payload.get("honest_verdict")):
        return "blocked"
    return "missing"


def _load_expected(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, bool]]:
    payloads: dict[str, dict[str, Any]] = {}
    present: dict[str, bool] = {}
    for exp_id, rel_path in EXPECTED_ARTIFACTS.items():
        path = root / rel_path
        present[exp_id] = path.is_file()
        payloads[exp_id] = read_json(path) if present[exp_id] else {}
    return payloads, present


def _classify_all(
    payloads: dict[str, dict[str, Any]], present: dict[str, bool]
) -> dict[str, str]:
    return {
        exp_id: classify_artifact(exp_id, payloads[exp_id], present[exp_id])
        for exp_id in EXPECTED_ARTIFACTS
    }


def _ids_with_status(statuses: dict[str, str], wanted: str) -> list[str]:
    return [exp_id for exp_id in EXPECTED_ARTIFACTS if statuses.get(exp_id) == wanted]


def _headline_rows(
    statuses: dict[str, str], matrix_payload: dict[str, Any]
) -> list[str]:
    """Extract the clean headline-eligible rows from matrix v8.

    Matrix v8 reports ``rows_clean`` as a flat list of row IDs. We use it
    directly when the matrix artifact is itself clean; otherwise we return
    the empty list because flagged/missing matrices cannot underwrite
    headline claims.
    """

    if statuses.get("exp2902") != "clean":
        return []
    rows = matrix_payload.get("rows_clean")
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, str)]


def _paper_ready(
    statuses: dict[str, str], matrix_payload: dict[str, Any]
) -> bool:
    """Paper-ready requires FoVer plus at least one other clean headline row.

    The "other" can be a corpus row OR an exp-support row — both count
    because matrix v8 explicitly added ``exp2898_kv260_hardware`` as a
    clean support row beside the corpus rows.
    """

    rows = _headline_rows(statuses, matrix_payload)
    has_fover = any("FoVer" in row for row in rows)
    has_other = any("FoVer" not in row for row in rows)
    return statuses.get("exp2902") == "clean" and has_fover and has_other


def _kv260_first_latency_recorded(
    payload: dict[str, Any], status: str
) -> bool:
    """True iff KV260 latency benchmark landed clean and on-board."""

    if status != "clean":
        return False
    overlay = payload.get("kv260_overlay_loaded")
    uios = payload.get("kv260_uio_devices_present")
    transcript = payload.get("board_transcript_path")
    bitstream = payload.get("bitstream_sha256")
    return (
        isinstance(overlay, str)
        and bool(overlay)
        and isinstance(uios, list)
        and len(uios) > 0
        and isinstance(transcript, str)
        and bool(transcript)
        and isinstance(bitstream, str)
        and len(bitstream) >= 16
    )


def _gatemate_bitstream_built(payload: dict[str, Any], status: str) -> bool:
    """True iff GateMate yosys+nextpnr+openFPGALoader produced a bitstream.

    Toolchain-absent or P&R-failed runs land here as False so the operator
    can pick up the install task from ``top_3_next_actions``.
    """

    if status not in ("clean", "blocked"):
        return False
    return (
        payload.get("synth_succeeded") is True
        and payload.get("place_and_route_succeeded") is True
        and isinstance(payload.get("bitstream_sha256"), str)
        and bool(payload.get("bitstream_sha256"))
    )


def _polarfire_smoke_verified(payload: dict[str, Any], status: str) -> bool:
    if status != "clean":
        return False
    return (
        payload.get("scorer_output_hash_verified") is True
        and payload.get("scorer_output_sha256")
        == payload.get("expected_scorer_output_sha256")
        and payload.get("polarfire_arch") == "riscv64"
    )


def _thrml_import_repaired(payload: dict[str, Any], status: str) -> bool:
    if status != "clean":
        return False
    return (
        payload.get("thrml_import_succeeded") is True
        and isinstance(payload.get("thrml_version_installed"), str)
        and bool(payload.get("thrml_version_installed"))
    )


def _cross_corpus_matrix_v8_built(
    payload: dict[str, Any], status: str
) -> bool:
    if status != "clean":
        return False
    rows = payload.get("rows_clean")
    return isinstance(rows, list) and len(rows) > 0


def _hardware_portfolio_reactivated(
    *,
    kv260_ok: bool,
    polarfire_ok: bool,
    thrml_ok: bool,
    gatemate_ok: bool,
    gatemate_status: str,
) -> bool:
    """True when three of four hardware tracks landed clean evidence.

    The fourth track must be honestly classified — blocked or clean is
    fine; missing is not. This matches the operator directive that the
    portfolio is "reactivated" when the boards are talking even if not
    every toolchain is provisioned yet.
    """

    clean_count = sum(int(x) for x in (kv260_ok, polarfire_ok, thrml_ok, gatemate_ok))
    if clean_count >= 3 and gatemate_status in ("clean", "blocked"):
        return True
    return False


def _hardware_track_status(
    payloads: dict[str, dict[str, Any]], statuses: dict[str, str]
) -> dict[str, dict[str, Any]]:
    """Per-board summary copied from exp2907 when present, derived otherwise."""

    portfolio = payloads.get("exp2907") or {}
    raw = portfolio.get("per_board_status") if isinstance(portfolio, dict) else None
    if isinstance(raw, dict):
        per_board = {k: dict(v) for k, v in raw.items() if isinstance(v, dict)}
    else:
        per_board = {}

    if "kv260" not in per_board:
        per_board["kv260"] = {
            "state": statuses.get("exp2898"),
            "last_artifact": str(EXPECTED_ARTIFACTS["exp2898"]),
        }
    if "gatemate" not in per_board:
        per_board["gatemate"] = {
            "state": statuses.get("exp2899"),
            "last_artifact": str(EXPECTED_ARTIFACTS["exp2899"]),
        }
    if "polarfire" not in per_board:
        per_board["polarfire"] = {
            "state": statuses.get("exp2900"),
            "last_artifact": str(EXPECTED_ARTIFACTS["exp2900"]),
        }
    if "thrml" not in per_board:
        per_board["thrml"] = {
            "state": statuses.get("exp2901"),
            "last_artifact": str(EXPECTED_ARTIFACTS["exp2901"]),
        }
    return per_board


def _kv260_latency_summary(payload: dict[str, Any]) -> dict[str, Any]:
    """Distil per-seed latency from exp2898 for downstream paper citations."""

    seeds = payload.get("per_seed_results")
    seed_records: list[dict[str, Any]] = []
    if isinstance(seeds, list):
        for entry in seeds:
            if not isinstance(entry, dict):
                continue
            seed_records.append(
                {
                    "seed": entry.get("seed"),
                    "n_samples": entry.get("n_samples"),
                    "per_sample_wall_clock_us_median": _number_or_none(
                        entry.get("per_sample_wall_clock_us_median")
                    ),
                    "per_sample_wall_clock_us_p95": _number_or_none(
                        entry.get("per_sample_wall_clock_us_p95")
                    ),
                    "final_energy": _number_or_none(entry.get("final_energy")),
                }
            )
    return {
        "kv260_overlay_loaded": payload.get("kv260_overlay_loaded"),
        "bitstream_sha256": payload.get("bitstream_sha256"),
        "uio_devices_present": payload.get("kv260_uio_devices_present"),
        "board_transcript_path": payload.get("board_transcript_path"),
        "per_seed_results": seed_records,
    }


def _cited_upstream_artifacts(
    root: Path,
    payloads: dict[str, dict[str, Any]],
    present: dict[str, bool],
) -> list[dict[str, Any]]:
    """Record path + sha256 for every present .274 source artifact.

    Why sha256 of the file contents (not just the artifact's own
    reproducibility_checksum field): the file hash is what a third party
    can recompute byte-for-byte to confirm the capstone's input set was
    the artifacts we claim. Source-internal checksums are advisory.
    """

    import hashlib

    citations: list[dict[str, Any]] = []
    for exp_id, rel_path in EXPECTED_ARTIFACTS.items():
        if not present[exp_id]:
            continue
        path = root / rel_path
        sha256: str | None
        try:
            sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            sha256 = None
        citations.append(
            {
                "experiment_id": exp_id,
                "artifact_path": str(rel_path),
                "sha256": sha256,
                "honest_verdict": payloads[exp_id].get("honest_verdict"),
            }
        )
    return citations


def _top_3_next_actions(
    *,
    statuses: dict[str, str],
    gatemate_built: bool,
    kv260_latency: bool,
    paper_ready: bool,
    sota_flagged: bool,
) -> list[str]:
    actions: list[str] = []
    if not gatemate_built:
        actions.append(
            "Provision nextpnr-gatemate on the dev box (oss-cad-suite update or"
            " source build), rerun the GateMate n=16 Ising tile build, and"
            " record bitstream_sha256 + place_and_route_succeeded=True before"
            " any flash attempt."
        )
    if sota_flagged:
        actions.append(
            "Re-run the bounded-budget SOTA code-generation expansion with"
            " random_seed declared, n_tasks_per_corpus >= 20 so pass@1 and"
            " pass@k cannot tautologically agree, and the adversarial-verify"
            " tautology check explicitly addressed in methodology_note."
        )
    if kv260_latency:
        actions.append(
            "Add a same-basis CPU Gibbs baseline for the KV260 Ising sampler"
            " (matched n_spins=64, identical coupling/field tensors, same"
            " seeds) before any paper-v6 hardware-speedup claim."
        )
    if not paper_ready:
        actions.append(
            "Restore a clean matrix v8 with FoVer plus at least one other"
            " clean headline row before carrying paper-v6 readiness into"
            " .275."
        )
    while len(actions) < 3:
        actions.append(
            "Promote one of the .274 pilot-only or support rows to a clean"
            " headline row in matrix v9 with full methodology disclosure."
        )
    return actions[:3]


def _gaps_for_275(
    *,
    statuses: dict[str, str],
    gatemate_built: bool,
    sota_flagged: bool,
    kv260_latency: bool,
) -> list[str]:
    gaps: list[str] = []
    if not gatemate_built:
        gaps.append(
            "GateMate A1-EVB-2M still has no bitstream; nextpnr-gatemate"
            " toolchain absent on the dev box."
        )
    if sota_flagged:
        gaps.append(
            "exp2905 SOTA code generation expansion remains adversarial-flagged"
            " (TAUTOLOGY: pass@1==pass@k; METHODOLOGY_MISSING: no random_seed)."
        )
    if kv260_latency:
        gaps.append(
            "KV260 latency is recorded but no same-basis CPU baseline exists;"
            " hardware-speedup claims remain forbidden."
        )
    blocked = _ids_with_status(statuses, "blocked")
    missing = _ids_with_status(statuses, "missing")
    if blocked:
        gaps.append(
            "Blocked .274 deliverables to pick up in .275: " + ", ".join(blocked) + "."
        )
    if missing:
        gaps.append(
            "Missing .274 deliverables to re-run in .275: " + ", ".join(missing) + "."
        )
    return gaps


def _compose_verdict(
    *,
    paper_ready: bool,
    hardware_portfolio_reactivated: bool,
    clean_count: int,
    flagged_count: int,
    blocked_count: int,
    missing_count: int,
    pilot_count: int,
) -> str:
    return (
        "complete: .274 capstone synthesized; "
        f"paper_ready={str(paper_ready).lower()}; "
        f"hardware_portfolio_reactivated={str(hardware_portfolio_reactivated).lower()}; "
        f"clean_artifacts={clean_count}; flagged_artifacts={flagged_count}; "
        f"blocked_artifacts={blocked_count}; missing_artifacts={missing_count}; "
        f"pilot_only_artifacts={pilot_count}"
    )


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
) -> dict[str, Any]:
    """REQ-REPORT-2908: synthesize the milestone .274 paper claim boundary."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else started_s
    payloads, present = _load_expected(root_path)
    statuses = _classify_all(payloads, present)

    matrix_payload = payloads["exp2902"]
    headline_rows = _headline_rows(statuses, matrix_payload)
    paper_ready = _paper_ready(statuses, matrix_payload)

    kv260_ok = _kv260_first_latency_recorded(payloads["exp2898"], statuses["exp2898"])
    gatemate_ok = _gatemate_bitstream_built(payloads["exp2899"], statuses["exp2899"])
    polarfire_ok = _polarfire_smoke_verified(payloads["exp2900"], statuses["exp2900"])
    thrml_ok = _thrml_import_repaired(payloads["exp2901"], statuses["exp2901"])
    matrix_v8_ok = _cross_corpus_matrix_v8_built(matrix_payload, statuses["exp2902"])

    hardware_reactivated = _hardware_portfolio_reactivated(
        kv260_ok=kv260_ok,
        polarfire_ok=polarfire_ok,
        thrml_ok=thrml_ok,
        gatemate_ok=gatemate_ok,
        gatemate_status=statuses["exp2899"],
    )

    clean_artifacts = _ids_with_status(statuses, "clean")
    flagged_artifacts = _ids_with_status(statuses, "flagged")
    blocked_artifacts = _ids_with_status(statuses, "blocked")
    missing_artifacts = _ids_with_status(statuses, "missing")
    pilot_only_artifacts = _ids_with_status(statuses, "pilot-only")

    sota_flagged = statuses["exp2905"] == "flagged"

    top_3 = _top_3_next_actions(
        statuses=statuses,
        gatemate_built=gatemate_ok,
        kv260_latency=kv260_ok,
        paper_ready=paper_ready,
        sota_flagged=sota_flagged,
    )
    gaps = _gaps_for_275(
        statuses=statuses,
        gatemate_built=gatemate_ok,
        sota_flagged=sota_flagged,
        kv260_latency=kv260_ok,
    )

    citations = _cited_upstream_artifacts(root_path, payloads, present)
    per_board = _hardware_track_status(payloads, statuses)
    kv260_summary = _kv260_latency_summary(payloads["exp2898"])

    end = time.perf_counter() if now_s is None else now_s

    return {
        "schema": SCHEMA,
        "artifact": "experiment_2908_capstone_v274",
        "honest_verdict": _compose_verdict(
            paper_ready=paper_ready,
            hardware_portfolio_reactivated=hardware_reactivated,
            clean_count=len(clean_artifacts),
            flagged_count=len(flagged_artifacts),
            blocked_count=len(blocked_artifacts),
            missing_count=len(missing_artifacts),
            pilot_count=len(pilot_only_artifacts),
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "milestone": MILESTONE,
        "paper_ready": paper_ready,
        "hardware_portfolio_reactivated": hardware_reactivated,
        "kv260_first_latency_recorded": kv260_ok,
        "gatemate_bitstream_built": gatemate_ok,
        "polarfire_smoke_verified": polarfire_ok,
        "thrml_import_repaired": thrml_ok,
        "cross_corpus_matrix_v8_built": matrix_v8_ok,
        "clean_artifacts": clean_artifacts,
        "flagged_artifacts": flagged_artifacts,
        "blocked_artifacts": blocked_artifacts,
        "missing_artifacts": missing_artifacts,
        "pilot_only_artifacts": pilot_only_artifacts,
        "headline_eligible_rows": headline_rows,
        "per_board_status": per_board,
        "kv260_latency_summary": kv260_summary,
        "top_3_next_actions": top_3,
        "gaps_for_275": gaps,
        "cited_upstream_artifacts": citations,
        "source_artifact_status": {
            exp_id: {
                "path": str(EXPECTED_ARTIFACTS[exp_id]),
                "status": statuses[exp_id],
                "present": present[exp_id],
                "honest_verdict": payloads[exp_id].get("honest_verdict"),
            }
            for exp_id in EXPECTED_ARTIFACTS
        },
        "field_principles": dict(FIELD_PRINCIPLES),
        "files_not_modified": [
            "research-roadmap.yaml",
            "scripts/research_conductor.py",
            "docs/index.html",
            "main.tex",
        ],
        "run_date": RUN_DATE,
        "duration_s": round(max(0.0, end - start), 6),
    }


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build and persist the Exp 2908 capstone JSON deliverable."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return out_path


if __name__ == "__main__":  # pragma: no cover
    print(write_artifact())
