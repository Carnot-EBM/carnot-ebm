"""REQ-REPORT-8243: count environment outcomes after the qualified frontier.

An unchanged source identity needs no repeat inventory reduction. Authentication
still reads exact bytes, because a remembered path cannot prove present evidence.
"""

import json
from pathlib import Path
from typing import Any

from carnot.reporting import arc_authoritative_frontier_8215 as authority
from carnot.reporting import arc_outcome_delta_8229 as qualified
from carnot.reporting.arc_supervisor_v688_receipts import event_order
from carnot.reporting.current_work_receipt import canonical_hash, sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar

ROOT = authority.ROOT
FRONTIER = ROOT / "results/experiment_8229_v711_arc_outcome_delta.json"
PINNED_FRONTIER = "sha256:b9358da4b4929b6b363dd82d4c4f0ded9be035de5fa394647e3da95056ca2caf"
TASK_ID = "exp8243-arc-supervisor-frontier"
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
    """Require real game groups before describing a candidate; live priorities stay fixed."""
    value = qualified.summarize(rows, prior)
    support = [
        dict(
            arm=a,
            **r,
            game_count=sum(
                a
                in {
                    x["arm"]
                    for x in rows
                    if x.get("game") == g and x["status"] in {"completed", "censored"}
                }
                for g in value["per_game_results"]
            ),
        )
        for a, r in value["per_arm_results"].items()
    ]
    supported = (
        value["arm_overlap_games"] >= 3
        and len(value["shared_arms"]) >= 2
        and all(value["per_arm_results"][a]["fired"] >= 10 for a in value["shared_arms"])
    )
    return dict(
        value,
        arm_support_rows=support,
        proposed_generalization_change=dict(
            status="candidate_only",
            compared_arms=value["shared_arms"],
            leave_one_game_out=value["descriptive_leave_one_game_out"],
            causal_superiority=False,
            live_priority_modified=False,
            specification="Compare curated arms prospectively with matched game/context and recorded propensity.",
        )
        if supported
        else None,
        reopen_condition="Changed authenticated producer source hashes after Exp8229; later finished_at; new redirect IDs; matching environment level/action receipts. Candidate refinement additionally requires >=3 shared games and >=10 grounded redirects per compared arm.",
    )


def grounded(episode: Json, redirect: Json) -> bool:
    """Join the supervisor counter to native gateway level/action evidence, never confidence."""
    level, action = redirect.get("level"), redirect.get("action_index")
    resolved, delta = redirect.get("resolved_by_levelup"), redirect.get("actions_to_levelup")
    transitions = episode.get("gateway_card_actions_by_level")
    valid = (
        type(level) is int
        and type(action) is int
        and action >= 0
        and type(resolved) is bool
        and isinstance(transitions, list)
        and type(episode.get("actions")) is int
        and episode["actions"] >= action
        and type(episode.get("levels")) is int
        and type(episode.get("seed")) is int
        and redirect.get("arm") in authority.ARM_ORDER
    )
    if not valid:
        return False
    if resolved:
        return (
            type(delta) is int
            and delta > 0
            and [level + 1, action + delta] in transitions
            and episode["levels"] > level
            and episode["actions"] >= action + delta
        )
    return (
        delta is None
        and episode["levels"] == level
        and not any(
            isinstance(t, list) and len(t) == 2 and t[0] > level and t[1] > action
            for t in transitions
        )
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
    registry = (
        authority.yaml.safe_load(authority.REGISTRY.read_text())
        if registry_hash == authority.PINNED_REGISTRY
        else {}
    )
    registry_games = registry.get("games", {})
    games = {r["game"]: r for r in registry_games}
    previous = load(frontier_path, PINNED_FRONTIER if frontier_path == FRONTIER else None)
    for key, expected in dict(
        experiment_id=8229,
        schema="arc-outcome-delta-v711",
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
    check(
        locator_path,
        "ledger_definition",
        list(authority.DEFAULT_LEDGER_PARTS),
        locator.get("ledger_definition"),
    )
    check(
        locator_path,
        "producer_paths",
        sorted(authority.producer_hashes()),
        sorted(locator.get("producer_code_hashes", {})),
    )
    refs = [locator[k] for k in ("ledger", "prior", "inventory", "terminal") if k in locator]
    refs += locator.get("sources", [])
    for ref in refs:
        load(Path(ref["path"]), ref.get("sha256"))
    ledger = documents.get(locator.get("ledger", {}).get("path", ""), {})
    check(locator_path, "ledger.schema", authority.LEDGER_SCHEMA, ledger.get("schema"))
    check(locator_path, "ledger.entries_object", True, isinstance(ledger.get("entries"), dict))
    for label, expected in locator.get("producer_code_hashes", {}).items():
        path = Path(label)
        actual = sha256_file(path) if path.is_file() else None
        check(path, "producer_code_sha256", expected, actual)
        if actual:
            hashes[label] = actual
    prior_signature = previous.get("qualified_authority_signature")
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
        old_hashes = {r["path"]: r["sha256"] for r in (prior_signature or {}).get("sources", [])}
        seen_events = set(prior.get("event_ids", []))
        seen_receipts = set(prior.get("receipt_ids", []))
        sources = locator.get("sources", [])
        check(locator_path, "source_count_at_most_64", True, len(sources) <= 64)
        for index, ref in enumerate(sources[:64]):
            print(
                f"[exp8243] phase=new_source completed={index} pending={len(sources) - index}",
                flush=True,
            )
            if old_hashes.get(ref["path"]) == ref["sha256"]:
                continue
            doc = documents.get(ref["path"], {})
            native = doc.get("experiment") == "arc_leaderboard_eval" and doc.get("policy") == "e3"
            harness = ref.get("kind") == "harness" and isinstance(doc.get("rows"), list)
            check(Path(ref["path"]), "live_producer_schema", True, native or harness)
            for episode in authority.extract_rows(doc):
                if authority.classify_receipt(episode) != "applied":
                    continue
                identity = authority.receipt_id_for_row(episode)
                receipt = episode["trajectory_supervisor"]
                expected = ledger.get("entries", {}).get(identity)
                if expected:
                    observed = authority._evidence_from_row(episode, ref["path"])
                    check(
                        Path(ref["path"]),
                        "receipt_identity:" + identity,
                        True,
                        expected.get("receipt_id") == identity
                        and all(observed.get(k) == v for k, v in expected.items() if k in observed),
                    )
                for redirect in receipt["redirects"]:
                    event_id = canonical_hash(
                        dict(game=episode.get("game"), seed=episode.get("seed"), redirect=redirect)
                    )
                    chronology = event_order(
                        dict(event_timestamp=episode.get("finished_at")),
                        dict(event_timestamp=previous.get("finished_at")),
                        "20261007",
                    )
                    reason = (
                        "duplicate_frontier"
                        if identity in seen_receipts or event_id in seen_events
                        else "before_exp8229_frontier"
                        if chronology != "after_cutoff"
                        else "non_live_provenance"
                        if episode.get("solve_provenance") != "live_agent_self_discovery"
                        else "environment_outcome_missing_or_mismatch"
                        if episode.get("game") not in games or not grounded(episode, redirect)
                        else None
                    )
                    rows.append(
                        dict(
                            redirect,
                            game=episode.get("game"),
                            seed=episode.get("seed"),
                            receipt_id=identity,
                            event_id=event_id,
                            status="excluded"
                            if reason
                            else "completed"
                            if redirect["resolved_by_levelup"]
                            else "censored",
                            reason=reason,
                            source_path=ref["path"],
                            source_sha256=ref["sha256"],
                            entrypoint="E3AgentPolicy/make_carnot_agent",
                            chronology=chronology,
                            event_timestamp=episode.get("finished_at"),
                            fired=True,
                            helped=redirect.get("resolved_by_levelup") is True,
                            metric="resolved_by_levelup",
                            numerator=int(redirect.get("resolved_by_levelup") is True),
                            denominator=1,
                            condition="observational_curated_redirect",
                            solve_provenance=episode.get("solve_provenance"),
                            environment_actions_by_level=episode.get(
                                "gateway_card_actions_by_level"
                            ),
                            stagnations_unredirected=receipt.get("stagnations_unredirected"),
                            unredirected_windows=receipt.get("unredirected_windows", []),
                        )
                    )
                    seen_events.add(event_id)
                seen_receipts.add(identity)
        failures = [r for r in checks if not r["passed"]]
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
        frontier_hashes={r["path"]: r["sha256"] for r in locator.get("sources", [])},
        registry_precheck={g: r.get("levels_reproduced", 0) for g, r in games.items()},
    )
