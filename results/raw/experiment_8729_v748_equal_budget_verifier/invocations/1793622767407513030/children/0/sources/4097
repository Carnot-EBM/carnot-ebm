"""REQ-REPORT-8588: reuse the qualified live wrapper without changing its policy.

These adapters bind the current invocation and add explicit association fields.
Original episode collection, predicates and primitive replay remain authoritative.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
from typing import Any, Iterator
from unittest.mock import patch

from carnot.reporting import arc_capability_panel_8537 as panel
from carnot.reporting import arc_capability_panel_runner_8537 as runner
from carnot.reporting import arc_redirect_runner_8509 as validation
from carnot.reporting.current_work_receipt import sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar

Json = dict[str, Any]
ROOT = panel.ROOT
NAME, TASK = "experiment_8588_v737_arc_capability_panel", "exp8588-arc-capability-panel"
TASK_PIN = "sha256:d6173cfa57ceaeb6fe9f9a3039d574128e27584bae8ed4a4fbe9dea6efcc1ad2"
PUBLICATION = "results/experiment_8587_v737_arc_publication_qualification.json"
PUBLICATION_PIN = "sha256:94f8314c802dfc49eb559c579c1f767b43d8f5718ef4811d0fe05abed9023816"
CLI = f"scripts/experiments/{NAME}.py"
TESTS = ["tests/python/test_arc_capability_panel_8588.py"]
OWNED = ["python/carnot/reporting/arc_capability_panel_8588.py", CLI]
HISTORICAL = [*validation.HISTORICAL, "tests/python/test_primary_publication_7928.py"]
MODEL_SPECS: list[Json] = []
AUTHENTICATE, MEASURE, BUILD, REPLAY = panel.authenticate, panel.measure, panel.build, panel.replay
checked, episode, canonical_hash = panel.checked, panel.episode, panel.canonical_hash


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush phase counts while inherited children retain their original identities."""
    print(f"[exp8588] phase={phase} completed={completed} pending={pending}", flush=True)


def authenticate(root: Path, raw: Path) -> Json:
    """Independent pins prevent current readiness from rewriting old qualification."""
    publication = checked(root / PUBLICATION, PUBLICATION_PIN)
    if not (
        publication["arc_panel_execution_ready_score"] == 1
        and publication["required_checks_passed"]
        and publication["flagged_adversarial"] is False
    ):
        raise ValueError("publication_readiness")
    side = json.loads(Path(publication["terminal_validation_sidecar_path"]).read_bytes())
    bound = read_bound_sidecar(root / PUBLICATION, Path(side["publication"]["sidecar_path"]))
    if not side["passed"] or not bound["report"]["passed"]:
        raise ValueError("publication_binding")
    for code, digest in publication["code_config_hashes"].items():
        panel.checked_bytes(code, digest)
    protocols = [
        json.loads((root / f"openspec/change-proposals/v{v}-methods-protocol.json").read_bytes())[
            "arc_capability_mask"
        ]
        for v in [735, 737]
    ]
    if protocols[0] != protocols[1]:
        raise ValueError("registered_protocol_drift")
    source: Json = AUTHENTICATE(root, raw)
    source["publication"] = publication
    source["publication_reference"] = panel.freeze(
        root / PUBLICATION, raw / "custody", PUBLICATION_PIN
    )
    return source


def measure(root: Path, raw: Path, private: Path) -> Json:
    """Inherited preconditions gate CPU resources before any live episode starts."""
    work: Json = MEASURE(root, raw, private)
    path = root / PUBLICATION
    work["source_artifact_hashes"][str(path)] = sha256_file(path) if path.is_file() else None
    for failure in work["failures"]:
        if failure["check"] == "qualified_upstream":
            failure.update(
                upstream="exp8587-arc-publication-qualification",
                path=str(path),
                hash=work["source_artifact_hashes"][str(path)],
                field="authenticated_publication_and_original_wrapper",
                expected=dict(publication=PUBLICATION_PIN, wrapper=panel.UPSTREAM_PIN),
                observed=failure.get("authentication_error"),
            )
    return work


def association(event: Json) -> str:
    """Observed follow-up labels describe association without a causal counterfactual."""
    if event.get("resolved_by_levelup"):
        return "progress_associated"
    if event.get("observed_level_delta", 0) < 0:
        return "regression_associated"
    return "unresolved"


def build(work_path: Path, receipts: list[Json], output: Path) -> Json:
    """Expose readiness and strict efficiency separately; repeated seeds are not new games."""
    value: Json = BUILD(work_path, receipts, output)
    work = json.loads(work_path.read_bytes())
    value.update(
        experiment_id=8588,
        task_id=TASK,
        milestone="2026.10.737",
        run_date="20261016",
        random_seed=7378588,
    )
    value["honest_verdict"] = f"complete_{value['verdict_class']}_v737_arc_capability_panel"
    pairs = value["paired_results"]
    value.update(
        arc_panel_rows_ready_score=value["arc_capability_panel_ready_score"],
        scoped_arc_efficiency_score=value["capability_efficiency_score"],
        refinement_eligible_score=int(value["arm_refinement_authorized"]),
        unavailable_work_delta=[
            dict(pair_key=p["pair_key"], delta=p["deltas"]["actual_disabled_attempts"])
            for p in pairs
        ],
        paired_level_deltas=[
            dict(pair_key=p["pair_key"], delta=p["deltas"]["peak_level"]) for p in pairs
        ],
        paired_wall_deltas=[
            dict(pair_key=p["pair_key"], delta=p["deltas"]["runtime_s"]) for p in pairs
        ],
        intervention_outcome_rows=[
            dict(r, association=association(r), causal_effect_unmeasured=True)
            for r in value["redirect_effect_rows"]
        ],
        censoring_rows=[
            dict(
                episode_id=r["episode_id"],
                censoring=r["censoring"],
                missing_reason=r["missing_reason"],
            )
            for r in value["rows"]
        ],
        per_action_rows_reference=[
            dict(
                directory=r["directory"],
                steps_sha256=r["steps_sha256"],
                terminal_sha256=r["terminal_sha256"],
            )
            for r in work["references"]
        ],
        hidden_game_performance="unmeasured",
        default_flag_off_unchanged=True,
        preserved_historical_outcomes=[
            "Exp8529 source null",
            "Exp8553 label rejection",
            "Exp8559 cold-cost null",
            "all prior quarantines",
        ],
    )
    value["cited_upstream_artifacts"].append(
        dict(path=PUBLICATION, sha256=PUBLICATION_PIN, original_date="20261016")
    )
    value["methodology_note"] += (
        " SEVRA motivates separating progress-associated, regression-associated and unresolved interventions. These are associations; causal counterfactual effects remain unmeasured."
    )
    value["field_principles"].update(
        {
            k: "Retain immutable live operands and all intended slots; unavailable work and associations do not establish causal or generalized benefit."
            for k in value
            if k not in value["field_principles"]
        }
    )
    return value


def replay(value: Json) -> bool:
    """The shipped reducer recomputes the entire current object from sealed primitives."""
    return bool(REPLAY(value))


def validate_checks(raw: Path, private: Path) -> list[Json]:
    """Run only owned statements and original private E2E nodes, without historical controllers."""
    with patch.multiple(validation, e=sys.modules[__name__], TESTS=TESTS, HISTORICAL=HISTORICAL):
        return list(validation.validate_checks(raw, private))


@contextmanager
def adapters() -> Iterator[None]:
    """Limit identity changes to this invocation so historical imports keep their authority."""
    with patch.multiple(
        panel,
        NAME=NAME,
        TASK=TASK,
        TASK_PIN=TASK_PIN,
        CLI=CLI,
        TESTS=TESTS,
        OWNED=OWNED,
        progress=progress,
        authenticate=authenticate,
        build=build,
    ):
        yield


def main(argv: list[str] | None = None) -> int:
    """Accept the current date while reusing the original deadline and publication machinery."""
    progress("start_no_model_load_MODEL_SPECS_empty_LLM_calls_zero")
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--date", choices=["20261016"], default="20261016")
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--owned-failure", action="store_true")
    args, rest = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    checks = validate_checks
    if args.private_control:

        def checks(raw: Path, private: Path) -> list[Json]:
            return [dict(name="private_control_only", passed=not args.owned_failure)]

    with adapters(), patch.multiple(runner, e=sys.modules[__name__], validate_checks=checks):
        return int(runner.main(["--date", "20261015", *rest]))
