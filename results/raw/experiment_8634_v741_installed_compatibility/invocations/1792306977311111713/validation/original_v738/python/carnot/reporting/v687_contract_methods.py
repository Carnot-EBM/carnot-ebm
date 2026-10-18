"""Bind current authority through shared reducers (REQ-REPORT-7915-V687).

Administrative agreement cannot measure scientific benefit. Version arguments
keep historical receipts stable while checking the thirteen current tasks.
"""

from copy import deepcopy
import gzip
import json
from pathlib import Path
import time
from typing import Any

import yaml

from carnot.reporting import v686_contract_methods as shared
from carnot.reporting.current_work_receipt import sha256_file
from carnot.reporting.roadmap_contract import parse_design

MILESTONE = "2026.09.687"
SOURCE_SHA256 = shared.SOURCE_SHA256
source_custody = shared.source_custody
primitive_rows = shared.primitive_rows


def assess(design: Path, staged: Path, active: Path, snapshots: Path) -> dict[str, Any]:
    """Read every authority separately, including a consumed staging observation."""
    return shared.assess(
        design, staged, active, snapshots, milestone=MILESTONE, first_id=7915, count=13
    )


def method_freeze(root: Path) -> dict[str, Any]:
    """Keep prospective protocols separate from the current custody computation."""
    text = gzip.decompress((root / "tests/fixtures/v687/design.md.gz").read_bytes()).decode()
    _, tasks = parse_design(text, milestone=MILESTONE)
    freeze = shared.method_freeze(root, tasks=tasks)
    freeze["fragility"] = {
        "groups": {
            "unigram": list(range(64)),
            "bigram": list(range(64, 128)),
            "overlap_uncovered": [128, 131],
            "negation_numeric": [129, 130],
        },
        "imputation": "label_free_fit_coordinate_means",
        "mask_seed": 68721,
        "mask_rates": [0.25, 0.50],
        "masks_per_rate": 32,
        "families": 64,
        "head_seeds": [67801, 67802, 67803],
        "greedy_steps": 4,
        "review_fraction": 0.20,
        "brittle_threshold": 0.25,
        "bootstrap_unit": "source_cluster",
        "bootstrap_draws": 10000,
        "multiplicity": "Holm_three_primary_comparisons",
        "capture_gain_min": 0.10,
        "minimum_eligible": 32,
        "minimum_brittle": 10,
        "score_label_channels": "separate_seeded_draw_provenance",
    }
    freeze["delayed_aci"].update(
        upstream_delay=20, total_delays=[21, 24, 36], phase_queues="D", issued_state_required=True
    )
    freeze["literature_adoption_decisions"].insert(
        0,
        {
            "source": "https://arxiv.org/html/2609.00366v1",
            "decision": "adapt",
            "task": "exp7921",
            "method": "grouped ablation with separate seeded stress labels",
            "limit": "protocol relative fragility; no natural correctness claim",
            "measured_benefit": None,
        },
    )
    for decision in freeze["literature_adoption_decisions"]:
        decision["reason"] = decision.get("limit", decision["method"])
        decision["task"] = {
            "exp7910": "exp7923",
            "exp7912": "exp7925",
            "exp7909/7910": "exp7922/7923",
            "exp7912/7913": "exp7925/7926",
        }.get(decision.get("task"), decision.get("task"))
    freeze["authority_status"] = "prospective_methods_not_outcomes"
    freeze["primary_rechecks"] = {
        "date": "20260930",
        "request_count": 2,
        "bounded_requests": True,
        "sources": ["https://arxiv.org/html/2609.00366v1", "https://arxiv.org/html/2609.07251v1"],
        "status": "method_pages_accessed",
        "metadata_discrepancy": "fragility page dates itself 29 Jun 2026",
    }
    return freeze


def mutations(design: Path, active: Path, source: Path, private: Path) -> list[dict[str, Any]]:
    """Exercise drift in private authorities; every expected rejection is retained."""
    baseline = yaml.safe_load(active.read_text())
    rows = []
    started = time.monotonic()
    names = (
        "consumed_staging",
        "later_staging",
        "order",
        "prompt",
        "prior_failures",
        "gate",
        "count",
        "date",
        "digest",
        "MODEL_SPECS",
        "inference_substrate_class",
        "source_hash_drift",
    )
    for index, name in enumerate(names, 1):
        directory = private / name
        directory.mkdir(parents=True, exist_ok=True)
        value = deepcopy(baseline)
        staged, actual, plan = (
            directory / label for label in ("stage.yaml", "active.yaml", "design.md")
        )
        text = design.read_text()
        expected = name in {"consumed_staging", "later_staging"}
        if name == "later_staging":
            staged.write_text("milestone: 2026.10.688\ntasks: []\n")
        elif name == "order":
            value["tasks"].reverse()
        elif name == "count":
            value["tasks"].pop()
        elif name == "date":
            value["tasks"][0]["prompt"] += " historical_fixture_date=20260930"
        elif name == "digest":
            text = text.replace(shared.lifecycle.tasks_digest(baseline["tasks"]), "0" * 64, 1)
        elif not expected and name != "source_hash_drift":
            value["tasks"][0]["gated_on" if name == "gate" else name] = "changed"
        plan.write_text(text)
        actual.write_text(yaml.safe_dump(value, sort_keys=False))
        observed = (
            source_custody(source, "sha256:drift")["ready"]
            if name == "source_hash_drift"
            else assess(plan, staged, actual, directory / "snapshots")["activated"]
        )
        rows.append(
            {
                "unit_id": name,
                "arm": "private_oracle_mutation",
                "status": "completed",
                "expected_activation": expected,
                "observed_activation": observed,
                "passed": observed is expected,
                "claim_scope": "circular_positive",
            }
        )
        print(
            f"[exp7915] phase=mutations elapsed_s={time.monotonic() - started:.3f} completed_units={index}/12",
            flush=True,
        )
    return rows


def candidate(root: Path, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """Use the common schema while binding producer identity and fixture dates."""
    value = shared.candidate(
        root, *args, **kwargs, experiment_id=7915, milestone=MILESTONE, count=13
    )
    old = root / "results/experiment_7903_v686_contract_methods.json"
    prior = json.loads(old.read_text())
    value["historical_required_failures"].append(
        {
            "experiment_id": 7903,
            "path": str(old),
            "sha256": sha256_file(old),
            "honest_verdict": prior["honest_verdict"],
            "resolved": False,
            "required_failures": prior["gate_check_summary"],
        }
    )
    same = prior["honest_verdict"] == value["honest_verdict"]
    value["retire_if_same_verdict"].update(
        identical_prior_failure=same,
        action="retire_unchanged_scope"
        if same
        else "preserve_prior_failures; versioned contract changed",
    )
    value["started_monotonic_timestamp_ns"] = value.pop("started_monotonic_ns")
    value["ended_monotonic_timestamp_ns"] = value.pop("ended_monotonic_ns")
    value.update(
        historical_fixture_date="20260929",
        execution_date="20260930",
        lifecycle_task_count=13,
        random_seed=6877915,
        coverage_statement_counts={},
        activation_before_observation=False,
    )
    value["resolved_imports"]["carnot.reporting.v687_contract_methods"] = str(
        Path(__file__).resolve()
    )
    manifest = Path(value["validation_command_manifest_path"])
    value["repository_health"]["prior_full_suite_observation"] = json.loads(
        manifest.read_text()
    ).get("repository_health_prior_observation")
    observation = value["repository_health"]["prior_full_suite_observation"]
    if observation:
        prior_path = Path(observation["prior_candidate_path"])
        old_attempt = json.loads(prior_path.read_text())
        value["historical_required_failures"].append(
            {
                "experiment_id": old_attempt["experiment_id"],
                "path": str(prior_path),
                "sha256": sha256_file(prior_path),
                "honest_verdict": old_attempt["honest_verdict"],
                "resolved": False,
                "required_failures": old_attempt["gate_check_summary"],
            }
        )
    value["observed_child_commands"] = [
        row["command_argv"] for row in value["validation_receipts"] if "command_argv" in row
    ]
    for report in manifest.parent.glob("coverage.json-*"):
        value["coverage_statement_counts"] = {
            name: data["summary"] for name, data in json.loads(report.read_text())["files"].items()
        }
    value["field_principles"].update(
        {
            key: "Keep producer identity, fixture time and actual measured scope explicit."
            for key in value
            if key not in value["field_principles"]
        }
    )
    value["field_principles"]["sample_size_budget"] = (
        "Count thirteen contract tasks; scientific independent units remain zero."
    )
    return value


def cold_replay(path: Path, raw: Path) -> bool:
    """Delegate primitive reduction with the current producer and task count."""
    if not shared.cold_replay(path, raw, experiment_id=7915, count=13):
        return False
    value = json.loads(path.read_text())
    return (
        value["milestone"] == MILESTONE
        and value["run_date"] == "20260930"
        and value["historical_fixture_date"] == "20260929"
        and value["execution_date"] == "20260930"
        and value["lifecycle_task_count"] == 13
        and value["duration_s"]
        == (value["ended_monotonic_timestamp_ns"] - value["started_monotonic_timestamp_ns"]) / 1e9
    )
