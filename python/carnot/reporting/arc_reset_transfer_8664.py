"""REQ-REPORT-8664: bind corrected current recording to the fixed transfer panel."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import arc_reset_transfer_8650 as prior
from carnot.reporting.arc_observer_8624 import checked, checked_bytes
from carnot.reporting.current_work_receipt import canonical_hash, sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar

Json = dict[str, Any]
ROOT = prior.ROOT
NAME = "experiment_8664_v743_arc_reset_transfer"
TASK = "exp8664-arc-reset-transfer"
TASK_PIN = "sha256:16407709630e5726353270fa58c085e19ed69c28190b9fecb59d65c2e160bd55"
CLI = f"scripts/experiments/{NAME}.py"
MODEL_SPECS: list[Json] = []
TESTS = [
    "tests/python/test_arc_reset_transfer_8664.py",
    "tests/python/test_arc_reset_runner_8664.py",
]
OWNED = [f"python/carnot/reporting/arc_reset_{n}_8664.py" for n in ("transfer", "runner")] + [CLI]
PINS = {
    "results/experiment_8653_v743_contract_methods.json": "sha256:c1176a28073046f66bd1eb8ba412ec025b4695e3bc14a212d829a5c5818e04f9",
    "results/experiment_8663_v743_arc_reset_provenance.json": "sha256:a409fd8fde401ff3d84251ce8043d84245fb124b25b43b85c43d695ae9d25e66",
}
GATE_FIELDS = ("arc_inputs_ready_score", "reset_provenance_ready_score")
OBSERVER_PIN = prior.original.UPSTREAM_PIN
GAMES = prior.GAMES
REQUIRED_CHECKS = prior.REQUIRED_CHECKS
original = prior.original
units = prior.units
BASE_REDUCE, BASE_BUILD, BASE_REPLAY = prior.reduce, prior.build, prior.replay


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush boundaries so the parent can supervise slow official SDK children."""
    print(f"[exp8664] phase={phase} completed={completed} pending={pending}", flush=True)


def gate_summary(root: Path) -> list[Json]:
    """Keep absent operands separate from observed zeros before hash admission."""
    rows = []
    for (name, pin), field in zip(PINS.items(), GATE_FIELDS, strict=True):
        path = root / name
        value = json.loads(path.read_bytes()) if path.is_file() else {}
        observed = value.get(field)
        digest = sha256_file(path) if path.is_file() else None
        state = (
            "missing_file"
            if not path.is_file()
            else "missing_field"
            if field not in value
            else "observed_zero"
            if observed == 0
            else "observed_value"
        )
        rows.append(
            dict(
                upstream="exp8653-contract-methods"
                if "8653" in name
                else "exp8663-arc-reset-provenance",
                check="required_upstream",
                path=str(path),
                hash=digest,
                expected_hash=pin,
                field=field,
                operator="==",
                expected=1,
                observed=observed,
                observed_state=state,
                passed=observed == 1 and digest == pin,
            )
        )
    return rows


def authenticate(root: Path) -> Json:
    """Original observer authority stays separate from the corrected current source."""
    values = [checked(root / name, pin) for name, pin in PINS.items()]
    for value, name, field in zip(values, PINS, GATE_FIELDS, strict=True):
        if value[field] != 1 or not value["required_checks_passed"] or value["flagged_adversarial"]:
            raise ValueError("upstream_readiness:" + field)
        side = json.loads(Path(value["terminal_validation_sidecar_path"]).read_bytes())
        if not read_bound_sidecar(root / name, Path(side["publication"]["sidecar_path"]))["report"][
            "passed"
        ]:
            raise ValueError("upstream_publication")
    observer = checked(root / original.UPSTREAM, OBSERVER_PIN)
    closure = checked(
        original.CLOSURE, "sha256:cb51a5341c669d08d119ed8e0d8f6ca8e5ce1df3573db8e8b9c43ffb4bd1f04b"
    )
    for ref in closure["files"]:
        checked_bytes(Path(ref["sealed_path"]), ref["sha256"])
        relative = ref["sealed_path"].split("/original_source_closure/", 1)[1]
        if relative.startswith("python/") and (
            "arc_observer" in relative or "arc_outcome_observer" in relative
        ):
            checked_bytes(ROOT / relative, ref["sha256"])
    corrected = checked(
        Path(values[1]["work_reference"]["path"]), values[1]["work_reference"]["sha256"]
    )
    for field in ("source_artifact_hashes", "raw_file_hashes"):
        for name, digest in corrected[field].items():
            checked_bytes(Path(name), digest)
    for name, digest in corrected["code_config_hashes"].items():
        checked_bytes(Path(name), digest)
        checked_bytes(ROOT / name.split("/source_closure/", 1)[1], digest)
    return dict(
        arc_inputs_ready_score=1,
        reset_provenance_ready_score=1,
        historical_model_provenance=observer["historical_model_provenance"],
        original_observer_reference=dict(path=str(root / original.UPSTREAM), sha256=OBSERVER_PIN),
        current_recorder_reference=values[1]["work_reference"],
    )


def reduce(episodes: list[Json]) -> Json:
    """Require current action indices and callbacks as well as the prior cause checks."""
    result = BASE_REDUCE(episodes)
    callbacks, errors, invalid = [], [], []
    official = all(ep["sdk_identity"]["official"] for ep in episodes)
    for ep in episodes:
        context = dict(
            game=ep["unit"]["game"], seed=ep["unit"]["seed"], episode_id=ep["unit"]["episode_id"]
        )
        errors.extend(dict(context, **r) for r in ep["recorder_error_metadata"])
        if ep["unit"]["reset_provenance"] != "on":
            continue
        sequence = ep["reset_provenance"]["callback_sequence_rows"]
        callbacks.extend(
            dict(
                r,
                **context,
                callback_id=f"{context['episode_id']}:{r['action_index']}:{r['method']}:{r['owner']}:{r['invocation']}",
            )
            for r in sequence
        )
        observed = {r["action_index"]: r for r in ep["reset_observations"]}
        for row in ep["reset_provenance"]["reset_request_rows"]:
            index = row["action_index"]
            callback = [
                c
                for c in sequence
                if c["action_index"] == index
                and c["method"] == "returned"
                and c["owner"] == "sdk_return"
                and c["invocation"] == 1
            ]
            if not (
                row["reset_request_id"].split(":")[0] == str(index)
                and index in observed
                and row["post_observation_sha256"] == observed[index]["observation_sha256"]
                and canonical_hash(observed[index]["public_observation"])
                == observed[index]["observation_sha256"]
                and len(callback) == 1
            ):
                invalid.append(
                    dict(
                        context,
                        action_index=index,
                        reset_request_id=row["reset_request_id"],
                        origin="unknown",
                        missing_reason="current_index_observation_or_callback_join",
                    )
                )
    result["missing_cause_rows"].extend(invalid)
    result["callback_sequence_rows"] = callbacks
    result["recorder_error_rows"] = errors
    result["reset_transfer_rows_ready_score"] *= int(not errors and not invalid and official)
    result["diagnostic_transfer_score"] *= result["reset_transfer_rows_ready_score"]
    for cell in result["cause_support_rows"]:
        if any(r["game"] == cell["game"] and r["seed"] == cell["seed"] for r in invalid):
            cell["support_count"] = 0
    for row in result["per_game_results"]:
        bad = sum(r["game"] == row["game"] for r in invalid)
        row["missing_cause_count"] += bad
        if bad:
            row["missing_reason"] = "current_index_observation_or_callback_join"
            row["authentic_origin_count"] = max(0, row["authentic_origin_count"] - bad)
        row["reset_support_count"] = sum(
            c["support_count"] for c in result["cause_support_rows"] if c["game"] == row["game"]
        )
    for join in result["redirect_reset_joins"]:
        join["callback_ids"] = [
            c["callback_id"]
            for c in callbacks
            if c["episode_id"] == join["episode_id"]
            and c["method"] == "returned"
            and join["reset_request_id"] is not None
            and str(c["action_index"]) == join["reset_request_id"].split(":")[0]
        ]
    return result


def build(path: Path, receipts: list[Json], output: Path) -> Json:
    """Reuse fixed outcome definitions while binding this invocation and its checks."""
    with patch.multiple(
        prior, NAME=NAME, TASK=TASK, PINS=PINS, reduce=reduce, REQUIRED_CHECKS=REQUIRED_CHECKS
    ):
        value = BASE_BUILD(path, receipts, output)
    work = json.loads(path.read_bytes())
    if not work["failures"] and len(work["references"]) != 16:
        value.update(
            verdict_class="disqualified",
            honest_verdict="complete_disqualified_arc_reset_transfer",
            flagged_adversarial=True,
            scope_closed=False,
            failed_count=16 - len(work["references"]),
        )
    value.update(
        experiment_id=8664,
        milestone="2026.10.743",
        random_seed=8664,
        independent_count=len({r["game"] for r in value["rows"] if r["status"] == "completed"}),
        gate_check_summary=work.get("upstream_gate_checks", []) + work["failures"],
        preserved_history=dict(
            original_observer_verdict="complete_null_arc_observer_transfer",
            complete_horizons=0,
            censored_redirects=16,
            prior_transfer_verdict="complete_disqualified_arc_reset_transfer",
            prior_missing_causes=236,
            quarantines_unchanged=True,
        ),
        adapter_binding=work.get("adapter_binding"),
        diagnostic_transfer_status="established"
        if value["diagnostic_transfer_score"]
        else "unestablished",
        methodology_note="Corrected recorder on a fixed public adapter-withheld panel. Eight game/seed pairs are dependent within four games. Original observer authority is separate. Zero full horizons grants no policy benefit, arm promotion, solve or significance claim.",
    )
    value["acceptance_gates"].update(
        {
            k: dict(field=k, operator="==", expected=v, observed=value[k], passed=value[k] == v)
            for k, v in value["branch_gate_values"].items()
        }
    )
    value["field_principles"] = {
        k: "Invocation-bound immutable evidence; missing differs from zero; diagnostic readiness gives no policy benefit."
        for k in value
    }
    value["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in value.items() if k != "reproducibility_checksum"}
    )
    return value


def replay(value: Json) -> bool:
    """Keep the independent seal outside an envelope that an attacker can rehash."""
    with patch.multiple(prior, NAME=NAME, build=build):
        return bool(BASE_REPLAY(value))
