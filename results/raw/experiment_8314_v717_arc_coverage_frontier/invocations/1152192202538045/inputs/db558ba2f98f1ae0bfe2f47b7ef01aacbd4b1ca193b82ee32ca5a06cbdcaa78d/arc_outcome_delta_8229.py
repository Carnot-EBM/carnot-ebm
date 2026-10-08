"""REQ-REPORT-8229: count environment outcomes after the qualified frontier.

An unchanged source identity needs no repeat inventory reduction. Authentication
still reads exact bytes, because a remembered path cannot prove present evidence.
"""

import json
from pathlib import Path
from typing import Any

from carnot.reporting import arc_authoritative_frontier_8215 as authority
from carnot.reporting.arc_supervisor_v688_receipts import event_order
from carnot.reporting.current_work_receipt import canonical_hash, sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar

ROOT = authority.ROOT
FRONTIER = ROOT / "results/experiment_8215_v709_arc_authoritative_frontier.json"
PINNED_FRONTIER = "sha256:6c402a39da67bed4881005f3f7e2c4f0f1b57d6025832b8afde1c25200554301"
TASK_ID = "exp8229-arc-outcome-delta"
Json = dict[str, Any]


def signature(locator: Json) -> Json:
    """Compare producer identities rather than modification times or old failures."""
    return {
        k: locator.get(k)
        for k in (
            "schema",
            "task_id",
            "ledger_definition",
            "ledger",
            "sources",
            "producer_code_hashes",
            "prior",
            "inventory",
            "terminal",
        )
    }


def summarize(rows: list[Json], prior: Json) -> Json:
    """Keep observational game support separate from any transferable benefit."""
    reduced = authority.reduction(rows, prior)
    reduced.pop("arm_selection_proposal")
    eligible = [r for r in rows if r["status"] in {"completed", "censored"}]
    cells = []
    support = {}
    for game in sorted({r["game"] for r in eligible}):
        arms = {r["arm"] for r in eligible if r["game"] == game}
        support[game] = arms
        for arm in sorted(arms):
            selected = [r for r in eligible if r["game"] == game and r["arm"] == arm]
            helped = sum(r["helped"] for r in selected)
            episodes = {r["receipt_id"]: r for r in selected}.values()
            stagnations = [r.get("stagnations_unredirected") for r in episodes]
            cells.append(
                dict(
                    game=game,
                    arm=arm,
                    fired=len(selected),
                    helped=helped,
                    resolved_by_levelup=dict(
                        numerator=helped, denominator=len(selected), status="measured"
                    ),
                    actions_to_levelup=dict(
                        values=[r["actions_to_levelup"] for r in selected],
                        numerator=sum(r["actions_to_levelup"] or 0 for r in selected),
                        denominator=helped,
                        missing_count=len(selected) - helped,
                        status="measured" if helped else "censored",
                    ),
                    stagnations_unredirected=dict(
                        values=stagnations,
                        numerator=sum(v for v in stagnations if v is not None),
                        denominator=sum(v is not None for v in stagnations),
                        missing_count=sum(v is None for v in stagnations),
                    ),
                    game_context_support="observed_redirect_context_only",
                )
            )
    overlap = set.intersection(*support.values()) if support else set()
    loo = []
    if len(support) >= 3 and len(overlap) >= 2:
        for game in sorted(support):
            training = [r for r in eligible if r["game"] != game and r["arm"] in overlap]
            rates = {
                a: sum(r["helped"] for r in training if r["arm"] == a)
                / sum(r["arm"] == a for r in training)
                for a in sorted(overlap)
            }
            loo.append(
                dict(
                    held_game=game,
                    training_games=sorted(set(support) - {game}),
                    shared_arms=sorted(overlap),
                    helped_per_firing=rates,
                    descriptive_arm=max(rates, key=lambda a: rates[a]),
                    claim="descriptive_only_unknown_propensity",
                )
            )
    return dict(
        reduced,
        per_game_arm_rows=cells,
        descriptive_leave_one_game_out=loo,
        selection_recommendations=[],
        selection_propensity=dict(status="missing", value=None),
        confounding_limits=[
            "prior_selection_propensity_unrecorded",
            "game_and_context_support",
            "nonrandom_curated_arm_selection",
            "confidence_and_rationales_are_descriptive",
        ],
        arm_overlap_games=len(support),
        shared_arms=sorted(overlap),
    )


def inspect(locator_path: Path, frontier_path: Path = FRONTIER) -> Json:
    """Authenticate named authority before distinguishing absence from zero progress."""
    checks: list[Json] = []
    hashes: dict[str, str] = {}
    documents: dict[str, Json] = {}

    def check(path: Path, field: str, expected: Any, observed: Any) -> None:
        checks.append(
            dict(
                authority.operand(
                    path, field, expected, observed, sha256_file(path) if path.is_file() else None
                ),
                passed=expected == observed,
            )
        )

    def load(path: Path, expected: str | None = None) -> Json:
        check(path, "is_file", True, path.is_file())
        if not path.is_file():
            return {}
        hashes[str(path)] = sha256_file(path)
        if expected is not None:
            check(path, "sha256", expected, hashes[str(path)])
        try:
            if path.stat().st_size > 33554432:
                raise ValueError("source_exceeds_32MiB")
            value = json.loads(path.read_text())
            check(path, "json_object", True, isinstance(value, dict))
            documents[str(path)] = value if isinstance(value, dict) else {}
            return value if isinstance(value, dict) else {}
        except (ValueError, OSError) as error:
            check(path, "bounded_json", True, str(error))
            return {}

    registry_hash = sha256_file(authority.REGISTRY) if authority.REGISTRY.is_file() else None
    check(authority.REGISTRY, "registry_sha256", authority.PINNED_REGISTRY, registry_hash)
    if registry_hash:
        hashes[str(authority.REGISTRY)] = registry_hash
    previous = load(frontier_path, PINNED_FRONTIER if frontier_path == FRONTIER else None)
    for key, expected in dict(
        experiment_id=8215,
        schema="arc-authoritative-frontier-v709",
        required_checks_passed=True,
        arc_delta_ready_score=1,
        verdict_class="null",
    ).items():
        check(frontier_path, key, expected, previous.get(key))
    if previous:
        digest = sha256_file(frontier_path)
        sidecar = (
            frontier_path.parent
            / "raw"
            / frontier_path.stem
            / "validators"
            / (digest[7:] + ".json")
        )
        try:
            terminal = read_bound_sidecar(frontier_path, sidecar)
            hashes[str(sidecar)] = sha256_file(sidecar)
            check(sidecar, "report.passed", True, terminal.get("report", {}).get("passed"))
        except (ValueError, OSError) as error:
            check(sidecar, "bound_terminal", True, str(error))
    locator = load(locator_path)
    check(locator_path, "schema", authority.SCHEMA, locator.get("schema"))
    check(locator_path, "task_id", authority.TASK_ID, locator.get("task_id"))
    refs = [locator[k] for k in ("ledger", "prior", "inventory", "terminal") if k in locator]
    refs += locator.get("sources", [])
    for ref in refs:
        load(Path(ref["path"]), ref.get("sha256"))
    for label, expected in locator.get("producer_code_hashes", {}).items():
        path = Path(label)
        actual = sha256_file(path) if path.is_file() else None
        check(path, "producer_code_sha256", expected, actual)
        if actual:
            hashes[label] = actual
    prior_signature = previous.get("authority_signature")
    if prior_signature is None and previous.get("locator_path"):
        historical = Path(previous["locator_path"])
        prior_signature = signature(
            load(historical, previous.get("source_artifact_hashes", {}).get(str(historical)))
        )
    unchanged = signature(locator) == prior_signature
    failures = [r for r in checks if not r["passed"]]
    prior = previous.get("current_frontier", {})
    rows = []
    if not failures and not unchanged:
        delta = authority.read_delta(locator_path, current_date="20261007")
        failures += delta["failures"]
        checks += delta["authority_locator_rows"]
        hashes.update(delta["source_artifact_hashes"])
        for row in delta["rows"]:
            row = dict(row)
            if row["status"] in {"completed", "censored"}:
                episodes = reader_episodes = authority.extract_rows(documents[row["source_path"]])
                original = next(
                    (e for e in episodes if authority.receipt_id_for_row(e) == row["receipt_id"]),
                    {},
                )
                chronology = event_order(
                    row, dict(event_timestamp=previous.get("finished_at")), "20261007"
                )
                if original.get("solve_provenance") != "live_agent_self_discovery":
                    row.update(
                        status="excluded",
                        reason="non_live_provenance",
                        solve_provenance=original.get("solve_provenance"),
                    )
                elif row["event_id"] in prior.get("event_ids", []) or row[
                    "receipt_id"
                ] in prior.get("receipt_ids", []):
                    row.update(status="excluded", reason="duplicate_frontier")
                elif chronology != "after_cutoff":
                    row.update(status="excluded", reason="before_exp8215_frontier")
            rows.append(row)
    if failures:
        rows = [
            dict(
                r,
                status="failed",
                reason="external_authority",
                condition="authority_operand",
                metric="operand_authenticated",
                numerator=0,
                denominator=1,
            )
            for r in failures
        ]
    return dict(
        summarize(rows, prior),
        failures=failures,
        authority_locator_rows=checks,
        source_artifact_hashes=hashes,
        unchanged_authority=unchanged,
        frontier_sha256=hashes.get(str(frontier_path)),
        historical_excluded_count=previous.get("excluded_count"),
        qualified_authority_signature=signature(locator),
    )
