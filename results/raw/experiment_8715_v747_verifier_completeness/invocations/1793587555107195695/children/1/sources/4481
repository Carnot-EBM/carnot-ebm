"""Build the Exp 4279 v395 capstone aggregation artifact.

Spec refs: REQ-CAPSTONE-4279, SCENARIO-CAPSTONE-4279.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


JsonDict = dict[str, Any]
LiveFlagRunner = Callable[[Path], list[dict[str, Any]]]
SummarizeRunner = Callable[[Path, Path], int]
PublicationGateRunner = Callable[[Path], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import adversarial_verify as av  # noqa: E402


OUTPUT_REL_PATH = Path("results/experiment_4279_capstone_v395.json")
EXPERIMENT_ID = 4279
RANDOM_SEED = 4279
SCHEMA = "carnot.capstone_v395_4279.v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
SPEC_REFS = ["REQ-CAPSTONE-4279", "SCENARIO-CAPSTONE-4279"]
BLOCKED_CHECKSUM = hashlib.sha256(b"blocked_v395_artifacts_missing").hexdigest()


@dataclass(frozen=True)
class Upstream:
    experiment_id: int
    path: Path


DEFAULT_UPSTREAMS: Mapping[str, Upstream] = {
    "4256_leak_audit": Upstream(
        4256, Path("results/experiment_4256_arc_oracle_distinct_leak_audit.json")
    ),
    "4257_multiseed": Upstream(
        4257, Path("results/experiment_4257_arc_oracle_distinct_multiseed_replication.json")
    ),
    "4270_family_recovery": Upstream(
        4270, Path("results/experiment_4270_arc_family_provenance_recovery.json")
    ),
    "4271_cross_family_existing": Upstream(
        4271, Path("results/experiment_4271_arc_cross_family_transfer_existing_pool.json")
    ),
    "4272_cross_family_fresh": Upstream(
        4272, Path("results/experiment_4272_arc_cross_family_transfer_fresh_tgi_pool.json")
    ),
    "4273_self_learning": Upstream(
        4273, Path("results/experiment_4273_arc_cross_family_online_adaptation.json")
    ),
    "4274_preflight": Upstream(
        4274, Path("results/experiment_4274_diffusiongemma_loader_fix_preflight.json")
    ),
    "4275_arc_progress": Upstream(
        4275, Path("results/experiment_4275_arc_incremental_progress_new_game.json")
    ),
    "4277_registry": Upstream(
        4277, Path("results/experiment_4277_verifier_registry_gaps_hygiene.json")
    ),
    "4278_hardware": Upstream(
        4278, Path("results/experiment_4278_hardware_continuity.json")
    ),
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "headline_outcome",
    "cross_family_generalizes",
    "hardened_win",
    "diffusiongemma_full_run_gate",
    "flagged_artifacts_excluded",
    "paper_ready",
    "verifier_is_oracle_honored",
    "reproducibility_checksum",
    "upstream_provenance",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. The .395 close-state -- whether the hardened win "
        "generalized cross-family (the OOD verdict that gates the headline + the scale-up)."
    ),
    "headline_outcome": (
        "One honest string aggregating the cross-family + self-learning + "
        "scale-up-readiness + ARC reads; the single line the .396 planner frames from."
    ),
    "cross_family_generalizes": (
        "BARE bool: did the held-out-family set_encoder@1 - vote@1 stay >0 "
        "with CI95-excl-0 -- the real OOD generalization verdict "
        "(north-star-grade if true, scopes-the-headline if false)."
    ),
    "hardened_win": (
        "BARE bool: the +44pp win is real (provenance-blind, .394), robust "
        "(multi-seed, .394), AND general (cross-family, .395) -- the full "
        "headline-eligibility verdict."
    ),
    "diffusiongemma_full_run_gate": (
        "BARE bool: resolvable iff hardened_win AND preflight_go -- whether "
        ".396 may activate the DiffusionGemma full run."
    ),
    "flagged_artifacts_excluded": (
        "List of .395 artifacts excluded for flagged_adversarial -- the "
        "fabrication gate (their numbers are NOT aggregated)."
    ),
    "paper_ready": (
        "From publication_gate.py --json -- the G1-G4 status (FoVer headline "
        "stays the publication target; a cross-family-general ARC win is a "
        "new supporting/headline result)."
    ),
    "verifier_is_oracle_honored": (
        "BARE bool=true -- confirms every cited moat/headline result carried "
        "verifier_is_oracle=false (no circular result headlines a moat)."
    ),
    "reproducibility_checksum": (
        "Hash of the aggregated upstream sha256 set; lets a third party "
        "re-derive the capstone."
    ),
    "upstream_provenance": (
        "{experiment_id, fields_imported, sha256} per cited upstream; skipped "
        "upstreams import no numbers."
    ),
}

IMPORTED_FIELDS: Mapping[str, list[str]] = {
    "4256_leak_audit": [
        "verifier_is_oracle",
        "win_survives_provenance_blind",
        "provenance_blind_delta",
        "provenance_blind_ci95",
    ],
    "4257_multiseed": [
        "verifier_is_oracle",
        "oracle_distinct_win_replicates",
        "mean_delta",
        "cross_seed_ci95",
        "cross_seed_ci95_excludes_zero",
        "n_seeds",
    ],
    "4270_family_recovery": [
        "family_split_feasible",
        "distinct_family_n",
        "provenance_manifest_path",
        "verifier_is_oracle",
    ],
    "4271_cross_family_existing": [
        "verifier_is_oracle",
        "cross_family_win_holds",
        "cross_family_delta",
        "cross_family_ci95",
        "ci95_excludes_zero",
        "within_minus_cross_gap",
        "held_out_family_n",
        "held_out_task_n",
        "oracle_at_k",
        "matched_control_delta",
        "online_adapt_cross_family_delta",
    ],
    "4272_cross_family_fresh": [
        "verifier_is_oracle",
        "cross_family_win_holds",
        "cross_family_delta",
        "cross_family_ci95",
        "ci95_excludes_zero",
        "within_minus_cross_gap",
        "held_out_family_n",
        "held_out_task_n",
        "oracle_at_k",
        "matched_control_delta",
        "online_adapt_cross_family_delta",
    ],
    "4273_self_learning": [
        "verifier_is_oracle",
        "online_adaptation_helps",
        "static_cross_family_delta",
        "online_cross_family_delta",
        "online_minus_static_ci95",
    ],
    "4274_preflight": [
        "verifier_is_oracle",
        "loader_repaired",
        "preflight_go",
        "guidance_changes_selection",
        "full_run_cost_estimate_s",
    ],
    "4275_arc_progress": [
        "total_levels",
        "total_levels_solved",
        "levels_completed",
        "new_levels_solved_this_task",
        "game_advanced",
    ],
    "4277_registry": ["registry_reconciled", "regression_guard_passed"],
    "4278_hardware": [
        "kv260_terminal_confirmed",
        "polarfire_step_taken",
        "gatemate_step_taken",
    ],
}


def bool_metric(payload: Mapping[str, Any] | None, field: str) -> bool | None:
    value = payload.get(field) if isinstance(payload, Mapping) else None
    return value if isinstance(value, bool) else None


def int_metric(payload: Mapping[str, Any] | None, field: str) -> int:
    value = payload.get(field) if isinstance(payload, Mapping) else None
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def float_metric(payload: Mapping[str, Any] | None, field: str) -> float | None:
    value = payload.get(field) if isinstance(payload, Mapping) else None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def str_metric(payload: Mapping[str, Any] | None, field: str) -> str:
    value = payload.get(field) if isinstance(payload, Mapping) else None
    return value if isinstance(value, str) else ""


def list_metric(payload: Mapping[str, Any] | None, field: str) -> list[Any]:
    value = payload.get(field) if isinstance(payload, Mapping) else None
    return list(value) if isinstance(value, list) else []


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def sha_from_payload_checksum(payload: Mapping[str, Any]) -> str:
    value = payload.get("reproducibility_checksum")
    if not isinstance(value, str):
        return ""
    normalized = value.removeprefix("sha256:")
    return normalized if is_sha256(normalized) else ""


def live_has_critical(flags: list[dict[str, Any]]) -> bool:
    return any(str(flag.get("severity", "")).lower() == "critical" for flag in flags)


def run_live_flags(path: Path) -> list[dict[str, Any]]:  # pragma: no cover
    return list(av.verify_artifact(path).get("flags", []))


def run_summarize_artifact(path: Path, root: Path) -> int:  # pragma: no cover
    proc = subprocess.run(
        [sys.executable, "scripts/summarize_artifact.py", str(path)],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return int(proc.returncode)


def run_publication_gate(root: Path) -> JsonDict:
    proc = subprocess.run(
        [sys.executable, "scripts/publication_gate.py", "--json"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    if not isinstance(payload, dict):
        raise ValueError("publication_gate.py --json returned a non-object payload")
    return payload


def read_json_object(path: Path) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("non-object")
    return payload


def clean_payload(payload: JsonDict | None, skipped: bool) -> JsonDict | None:
    return None if skipped or payload is None else payload


def _selected_paths(root: Path) -> dict[str, Path]:
    return {key: root / upstream.path for key, upstream in DEFAULT_UPSTREAMS.items()}


def _fields_for_payload(key: str, payload: Mapping[str, Any], skipped: bool) -> list[str]:
    if skipped:
        return []
    if key in ("4271_cross_family_existing", "4272_cross_family_fresh"):
        return list(IMPORTED_FIELDS[key]) if "cross_family_win_holds" in payload else []
    return list(IMPORTED_FIELDS[key])


def _read_inputs(
    root: Path,
    live_flag_runner: LiveFlagRunner,
    summarize_runner: SummarizeRunner,
) -> tuple[dict[str, JsonDict], list[JsonDict], list[JsonDict], list[JsonDict]]:
    payloads: dict[str, JsonDict] = {}
    provenance: list[JsonDict] = []
    exclusions: list[JsonDict] = []
    missing: list[JsonDict] = []
    for key, path in _selected_paths(root).items():
        upstream = DEFAULT_UPSTREAMS[key]
        if not path.exists():
            missing.append(
                {"artifact_key": key, "experiment_id": upstream.experiment_id, "reason": "missing"}
            )
            continue
        sha = sha256_file(path)
        try:
            payload = read_json_object(path)
        except (OSError, json.JSONDecodeError, ValueError):
            missing.append(
                {
                    "artifact_key": key,
                    "experiment_id": upstream.experiment_id,
                    "reason": "unparsable_or_non_object",
                }
            )
            continue
        summarize_exit_code = summarize_runner(path, root)
        live_flags = live_flag_runner(path)
        stamped = payload.get("flagged_adversarial") is True
        critical = live_has_critical(live_flags)
        skipped = stamped or critical
        payloads[key] = payload
        fields_imported = _fields_for_payload(key, payload, skipped)
        provenance.append(
            {
                "artifact_key": key,
                "experiment_id": upstream.experiment_id,
                "path": str(upstream.path),
                "sha256": sha,
                "payload_reproducibility_checksum": sha_from_payload_checksum(payload),
                "summarize_exit_code": summarize_exit_code,
                "live_adversarial_flags": live_flags,
                "stamped_flagged_adversarial": stamped,
                "live_critical": critical,
                "skipped": skipped,
                "fields_imported": fields_imported,
            }
        )
        if skipped:
            exclusions.append(
                {
                    "artifact_key": key,
                    "experiment_id": upstream.experiment_id,
                    "path": str(upstream.path),
                    "sha256": sha,
                    "stamped_flagged_adversarial": stamped,
                    "live_critical": critical,
                    "live_critical_flags": [
                        flag
                        for flag in live_flags
                        if str(flag.get("severity", "")).lower() == "critical"
                    ],
                    "reason": "flagged_adversarial_or_live_critical",
                }
            )
    return payloads, provenance, exclusions, missing


def checksum_from_provenance(provenance: list[Mapping[str, Any]]) -> str:
    if not provenance:
        return BLOCKED_CHECKSUM
    shas = sorted(str(row["sha256"]) for row in provenance)
    return hashlib.sha256("\n".join(shas).encode("utf-8")).hexdigest()


def _verdict(payload: Mapping[str, Any] | None) -> str:
    return str_metric(payload, "honest_verdict")


def _oracle_distinct(payload: Mapping[str, Any] | None) -> bool:
    return bool_metric(payload, "verifier_is_oracle") is False


def _ci_excludes_zero(
    payload: Mapping[str, Any] | None,
    field: str,
    explicit_field: str = "",
) -> bool:
    explicit = bool_metric(payload, explicit_field) if explicit_field else None
    if explicit is not None:
        return explicit
    value = payload.get(field) if isinstance(payload, Mapping) else None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return False
    low, high = value
    if not isinstance(low, (int, float)) or isinstance(low, bool):
        return False
    if not isinstance(high, (int, float)) or isinstance(high, bool):
        return False
    return float(low) > 0.0 or float(high) < 0.0


def provenance_blind_read(payload: JsonDict | None) -> JsonDict:
    win = (
        _oracle_distinct(payload)
        and bool_metric(payload, "win_survives_provenance_blind") is True
        and (float_metric(payload, "provenance_blind_delta") or 0.0) > 0.0
        and _ci_excludes_zero(payload, "provenance_blind_ci95")
    )
    return {
        "status": "included" if payload is not None else "missing_or_excluded",
        "used_as_oracle_distinct": _oracle_distinct(payload),
        "win_survives_provenance_blind": win,
        "reported_win_survives_provenance_blind": bool_metric(
            payload, "win_survives_provenance_blind"
        ),
        "provenance_blind_delta": float_metric(payload, "provenance_blind_delta"),
        "provenance_blind_ci95": list_metric(payload, "provenance_blind_ci95"),
        "verifier_is_oracle": bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": _verdict(payload),
    }


def multiseed_read(payload: JsonDict | None) -> JsonDict:
    replicated = (
        _oracle_distinct(payload)
        and bool_metric(payload, "oracle_distinct_win_replicates") is True
        and (float_metric(payload, "mean_delta") or 0.0) > 0.0
        and (
            bool_metric(payload, "cross_seed_ci95_excludes_zero") is True
            or _ci_excludes_zero(payload, "cross_seed_ci95")
        )
    )
    return {
        "status": "included" if payload is not None else "missing_or_excluded",
        "used_as_oracle_distinct": _oracle_distinct(payload),
        "oracle_distinct_win_replicates": replicated,
        "reported_oracle_distinct_win_replicates": bool_metric(
            payload, "oracle_distinct_win_replicates"
        ),
        "mean_delta": float_metric(payload, "mean_delta"),
        "cross_seed_ci95": list_metric(payload, "cross_seed_ci95"),
        "cross_seed_ci95_excludes_zero": bool_metric(payload, "cross_seed_ci95_excludes_zero"),
        "n_seeds": int_metric(payload, "n_seeds"),
        "verifier_is_oracle": bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": _verdict(payload),
    }


def cross_family_read(payload: JsonDict | None, source_key: str) -> JsonDict:
    if payload is None or "cross_family_win_holds" not in payload:
        return {
            "status": "open_no_clean_cross_family_run",
            "source_artifact_key": source_key,
            "cross_family_win_holds": False,
            "cross_family_delta": None,
            "cross_family_ci95": [],
            "ci95_excludes_zero": False,
            "used_as_oracle_distinct": False,
        }
    delta = float_metric(payload, "cross_family_delta")
    ci_excludes = _ci_excludes_zero(payload, "cross_family_ci95", "ci95_excludes_zero")
    generalizes = (
        _oracle_distinct(payload)
        and bool_metric(payload, "cross_family_win_holds") is True
        and delta is not None
        and delta > 0.0
        and ci_excludes
    )
    return {
        "status": "cross_family_generalizes" if generalizes else "cross_family_scopes_headline",
        "source_artifact_key": source_key,
        "used_as_oracle_distinct": _oracle_distinct(payload),
        "cross_family_win_holds": generalizes,
        "reported_cross_family_win_holds": bool_metric(payload, "cross_family_win_holds"),
        "cross_family_delta": delta,
        "cross_family_ci95": list_metric(payload, "cross_family_ci95"),
        "ci95_excludes_zero": ci_excludes,
        "within_minus_cross_gap": float_metric(payload, "within_minus_cross_gap"),
        "held_out_family_n": int_metric(payload, "held_out_family_n"),
        "held_out_task_n": int_metric(payload, "held_out_task_n"),
        "oracle_at_k": float_metric(payload, "oracle_at_k"),
        "matched_control_delta": float_metric(payload, "matched_control_delta"),
        "online_adapt_cross_family_delta": float_metric(payload, "online_adapt_cross_family_delta"),
        "verifier_is_oracle": bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": _verdict(payload),
    }


def select_cross_family(clean: Mapping[str, JsonDict | None]) -> JsonDict:
    for key in ("4271_cross_family_existing", "4272_cross_family_fresh"):
        payload = clean.get(key)
        if isinstance(payload, Mapping) and "cross_family_win_holds" in payload:
            return cross_family_read(dict(payload), key)
    return cross_family_read(None, "missing")


def self_learning_read(payload: JsonDict | None) -> JsonDict:
    helps = _oracle_distinct(payload) and bool_metric(payload, "online_adaptation_helps") is True
    return {
        "status": "helps" if helps else "static_ceiling",
        "online_adaptation_helps": helps,
        "reported_online_adaptation_helps": bool_metric(payload, "online_adaptation_helps"),
        "static_cross_family_delta": float_metric(payload, "static_cross_family_delta"),
        "online_cross_family_delta": float_metric(payload, "online_cross_family_delta"),
        "online_minus_static_ci95": list_metric(payload, "online_minus_static_ci95"),
        "verifier_is_oracle": bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": _verdict(payload),
    }


def preflight_read(payload: JsonDict | None, *, skipped: bool) -> JsonDict:
    if skipped:
        return {"status": "excluded_flagged_adversarial", "preflight_go": False}
    go = _oracle_distinct(payload) and bool_metric(payload, "preflight_go") is True
    return {
        "status": "preflight_go" if go else "preflight_blocked",
        "loader_repaired": bool_metric(payload, "loader_repaired") is True,
        "preflight_go": go,
        "reported_preflight_go": bool_metric(payload, "preflight_go"),
        "guidance_changes_selection": bool_metric(payload, "guidance_changes_selection"),
        "full_run_cost_estimate_s": float_metric(payload, "full_run_cost_estimate_s"),
        "verifier_is_oracle": bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": _verdict(payload),
    }


def arc_progress_read(payload: JsonDict | None) -> JsonDict:
    return {
        "status": "included" if payload is not None else "missing_or_excluded",
        "total_levels": int_metric(payload, "total_levels"),
        "total_levels_solved": int_metric(payload, "total_levels_solved"),
        "levels_completed": int_metric(payload, "levels_completed"),
        "new_levels_solved_this_task": int_metric(payload, "new_levels_solved_this_task"),
        "game_advanced": str_metric(payload, "game_advanced"),
        "honest_verdict": _verdict(payload),
    }


def registry_read(payload: JsonDict | None, *, skipped: bool) -> JsonDict:
    if skipped:
        return {"status": "excluded_flagged_adversarial", "regression_guard_passed": False}
    return {
        "status": "included" if payload is not None else "missing_or_excluded",
        "registry_reconciled": bool_metric(payload, "registry_reconciled") is True,
        "regression_guard_passed": bool_metric(payload, "regression_guard_passed") is True,
        "honest_verdict": _verdict(payload),
    }


def hardware_read(payload: JsonDict | None) -> JsonDict:
    return {
        "status": "included" if payload is not None else "missing_or_excluded",
        "kv260_terminal_confirmed": bool_metric(payload, "kv260_terminal_confirmed") is True,
        "polarfire_step_taken": str_metric(payload, "polarfire_step_taken"),
        "gatemate_step_taken": str_metric(payload, "gatemate_step_taken"),
        "honest_verdict": _verdict(payload),
    }


def headline_string(
    cross_family_generalizes: bool,
    cross_status: str,
    self_learning_helps: bool,
    full_run_gate: bool,
    total_levels: int,
    game_advanced: str,
    paper_ready: bool,
) -> str:
    if cross_family_generalizes:
        cross = "cross_family_generalizes"
    elif cross_status == "open_no_clean_cross_family_run":
        cross = "cross_family_open"
    else:
        cross = "cross_family_scopes_headline"
    self_label = "self_learning_helps" if self_learning_helps else "self_learning_static_ceiling"
    diffusion = "diffusiongemma_full_run_ready" if full_run_gate else "diffusiongemma_full_run_blocked"
    paper = "paper_ready" if paper_ready else "paper_not_ready"
    game = game_advanced or "none"
    return f"{cross}_{self_label}_{diffusion}_arc{total_levels}_game_{game}_{paper}"


def build_artifact(
    repo_root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    live_flag_runner: LiveFlagRunner = run_live_flags,
    summarize_runner: SummarizeRunner = run_summarize_artifact,
    publication_gate_runner: PublicationGateRunner = run_publication_gate,
) -> JsonDict:
    root = Path(repo_root)
    start = time.time() if started_s is None else started_s
    payloads, provenance, exclusions, missing = _read_inputs(root, live_flag_runner, summarize_runner)
    end = time.time() if now_s is None else now_s
    common: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_ID,
        "spec_refs": SPEC_REFS,
        "random_seed": RANDOM_SEED,
        "field_principles": FIELD_PRINCIPLES,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(end - start, 6),
        "summarize_command": "python3 scripts/summarize_artifact.py <artifact>",
        "live_adversarial_recheck": "scripts/adversarial_verify.py verify_artifact per upstream",
        "publication_gate_command": "python3 scripts/publication_gate.py --json",
    }
    if missing:
        artifact = {
            **common,
            "honest_verdict": "blocked_v395_artifacts_missing",
            "headline_outcome": "blocked_v395_artifacts_missing",
            "cross_family_generalizes": False,
            "hardened_win": False,
            "diffusiongemma_full_run_gate": False,
            "flagged_artifacts_excluded": [],
            "paper_ready": None,
            "unmet_gates": [],
            "publication_gate": None,
            "verifier_is_oracle_honored": True,
            "missing_upstream_artifacts": missing,
            "upstream_provenance": [],
            "reproducibility_checksum": BLOCKED_CHECKSUM,
        }
        validate_artifact(artifact)
        return artifact

    skipped = {row["artifact_key"]: bool(row["skipped"]) for row in provenance}
    clean = {
        key: clean_payload(payloads.get(key), skipped.get(key, False)) for key in DEFAULT_UPSTREAMS
    }
    provenance_blind = provenance_blind_read(clean["4256_leak_audit"])
    multiseed = multiseed_read(clean["4257_multiseed"])
    cross_family = select_cross_family(clean)
    cross_family_generalizes = bool(cross_family["cross_family_win_holds"])
    hardening = {
        "provenance_blind": provenance_blind,
        "multiseed": multiseed,
    }
    hardened_win = (
        bool(provenance_blind["win_survives_provenance_blind"])
        and bool(multiseed["oracle_distinct_win_replicates"])
        and cross_family_generalizes
    )
    hardening["hardened_win"] = hardened_win
    self_learning = self_learning_read(clean["4273_self_learning"])
    preflight = preflight_read(clean["4274_preflight"], skipped=skipped["4274_preflight"])
    arc = arc_progress_read(clean["4275_arc_progress"])
    registry = registry_read(clean["4277_registry"], skipped=skipped["4277_registry"])
    hardware = hardware_read(clean["4278_hardware"])
    publication = publication_gate_runner(root)
    paper_ready = bool(publication.get("paper_ready"))
    full_run_gate = hardened_win and bool(preflight["preflight_go"])
    headline = headline_string(
        cross_family_generalizes,
        str(cross_family["status"]),
        bool(self_learning["online_adaptation_helps"]),
        full_run_gate,
        int(arc["total_levels_solved"]),
        str(arc["game_advanced"]),
        paper_ready,
    )
    artifact = {
        **common,
        "honest_verdict": (
            f"complete: capstone_v395_{headline}_cross_family_generalizes_"
            f"{cross_family_generalizes}_hardened_win_{hardened_win}_"
            f"diffusiongemma_full_run_gate_{full_run_gate}_excluded_{len(exclusions)}"
        ),
        "headline_outcome": headline,
        "cross_family_generalizes": cross_family_generalizes,
        "hardened_win": hardened_win,
        "diffusiongemma_full_run_gate": full_run_gate,
        "flagged_artifacts_excluded": exclusions,
        "paper_ready": paper_ready,
        "unmet_gates": list(publication.get("unmet_gates", [])),
        "publication_gate": publication,
        "verifier_is_oracle_honored": True,
        "missing_upstream_artifacts": [],
        "hardening": hardening,
        "cross_family": cross_family,
        "self_learning": self_learning,
        "scale_up_readiness": preflight,
        "arc_progress": arc,
        "registry_read": registry,
        "hardware_read": hardware,
        "upstream_provenance": provenance,
        "reproducibility_checksum": checksum_from_provenance(provenance),
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            raise ValueError(f"missing required field: {field}")
    verdict = artifact["honest_verdict"]
    blocked = verdict == "blocked_v395_artifacts_missing"
    terminal = isinstance(verdict, str) and verdict.startswith(("complete:", "success:"))
    if not (blocked or terminal):
        raise ValueError("honest_verdict must be terminal-prefixed or blocked_v395_artifacts_missing")
    if not isinstance(artifact["headline_outcome"], str) or not artifact["headline_outcome"]:
        raise ValueError("headline_outcome must be a non-empty string")
    if not isinstance(artifact["cross_family_generalizes"], bool):
        raise ValueError("cross_family_generalizes must be a bare bool")
    if not isinstance(artifact["hardened_win"], bool):
        raise ValueError("hardened_win must be a bare bool")
    if artifact["hardened_win"] and not artifact["cross_family_generalizes"]:
        raise ValueError("hardened_win cannot pass without cross_family_generalizes")
    if not isinstance(artifact["diffusiongemma_full_run_gate"], bool):
        raise ValueError("DiffusionGemma full-run gate must be a bare bool")
    if artifact["diffusiongemma_full_run_gate"] and not artifact["hardened_win"]:
        raise ValueError("DiffusionGemma full-run gate cannot pass without hardened_win")
    if artifact["verifier_is_oracle_honored"] is not True:
        raise ValueError("oracle-distinctness discipline was not honored")
    if blocked:
        if artifact["paper_ready"] is not None:
            raise ValueError("blocked artifacts must leave paper_ready unresolved")
        if artifact["upstream_provenance"] != []:
            raise ValueError("blocked artifacts must not aggregate upstream provenance")
    elif not isinstance(artifact["paper_ready"], bool):
        raise ValueError("paper_ready must be a bare bool")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        raise ValueError("field_principles must match the required principle text")
    flagged = artifact["flagged_artifacts_excluded"]
    if not isinstance(flagged, list):
        raise ValueError("flagged_artifacts_excluded must be a list")
    if any(not isinstance(row, Mapping) for row in flagged):
        raise ValueError("flagged entries must be objects")
    provenance = artifact["upstream_provenance"]
    if not isinstance(provenance, list):
        raise ValueError("upstream_provenance must be a list")
    for row in provenance:
        if not isinstance(row, Mapping):
            raise ValueError("provenance entries must be objects")
        if not isinstance(row.get("artifact_key"), str):
            raise ValueError("provenance entries need artifact_key")
        if not isinstance(row.get("experiment_id"), int) or isinstance(row.get("experiment_id"), bool):
            raise ValueError("provenance entries need integer experiment_id")
        if not is_sha256(row.get("sha256")):
            raise ValueError("provenance entries need sha256")
        if not isinstance(row.get("fields_imported"), list):
            raise ValueError("provenance entries need fields_imported lists")
        if row.get("skipped") is True and row.get("fields_imported") != []:
            raise ValueError("skipped upstreams must not import fields")
    expected_checksum = BLOCKED_CHECKSUM if blocked else checksum_from_provenance(provenance)
    if artifact["reproducibility_checksum"] != expected_checksum:
        raise ValueError("reproducibility_checksum does not match upstream sha256 set")


def write_artifact(
    repo_root: Path | str = REPO_ROOT,
    output_path: Path = OUTPUT_REL_PATH,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    live_flag_runner: LiveFlagRunner = run_live_flags,
    summarize_runner: SummarizeRunner = run_summarize_artifact,
    publication_gate_runner: PublicationGateRunner = run_publication_gate,
) -> Path:
    root = Path(repo_root)
    artifact = build_artifact(
        root,
        started_s=started_s,
        now_s=now_s,
        live_flag_runner=live_flag_runner,
        summarize_runner=summarize_runner,
        publication_gate_runner=publication_gate_runner,
    )
    out = root / output_path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def main() -> int:  # pragma: no cover
    output = write_artifact(REPO_ROOT)
    print(output)
    return 0
