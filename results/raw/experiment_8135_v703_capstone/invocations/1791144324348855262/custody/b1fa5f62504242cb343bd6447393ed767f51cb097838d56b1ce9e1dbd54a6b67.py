"""REQ-REPORT-8069: primitive equations keep measurement distinct from benefit."""

import json
from pathlib import Path
import tempfile
from typing import Any

from carnot import experiment_8059_v698_fit_source_scoring as source
from carnot import experiment_8066_v698_content_addressed_feature_service as cache
from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting import hardware_feature_8068 as hardware
from carnot.verify import fresh_learning_audit_8065 as learning

Json = dict[str, Any]


def independent(data: Json, number: int, copies: Json | None = None) -> Json:
    """Reopen tokens and journals; producer summary fields supply no observations."""
    copies = copies or {}

    def location(path: Path) -> Path:
        return Path(copies.get(str(path), str(path)))

    result: Json = dict(measurement_available=False)
    if data.get("verifier_is_oracle") or not data.get("rows"):
        return result
    if number == 8059:
        raw = Path(data["terminal_validation_sidecar_path"]).parent
        raw = location(raw / "runtime.json").parent
        measured = json.loads((raw / "runtime.json").read_text())
        calls = [json.loads(p.read_text()) for p in sorted((raw / "forwards").glob("pass-*.json"))]
        reduced = source.reduce(measured["panel"]["rows"], calls)
        for key, observed in reduced.items():
            learning.math.equal("source." + key, observed, data[key])
        result.update(measurement_available=True, **reduced)
    if number == 8065:
        raw = location(Path(data["raw_directory"]) / "plan.json").parent
        plan = json.loads((raw / "plan.json").read_text())["data"]
        targets = {
            r["family_id"]: r["eligible_y"]
            for r in json.loads(location(Path(plan["target_reference"]["path"])).read_text())[
                "rows"
            ]
        }
        retained_targets = {
            r["family_id"]: r["eligible_y"]
            for r in json.loads(location(Path(plan["retention_target"]["path"])).read_text())[
                "rows"
            ]
        }
        replay = learning.reconstruct(raw / "trajectory", targets)
        with tempfile.TemporaryDirectory(prefix="capstone-retention-") as temp:
            retained = learning.retention(
                plan, replay["final_head_seals"], Path(temp), lambda: retained_targets
            )
            seal = json.loads((Path(temp) / "retention_prediction_seal.json").read_text())
        learning.math.equal(
            "retention_prediction_seal",
            seal,
            json.loads((raw / "retention_prediction_seal.json").read_text()),
        )
        comparisons = learning.comparisons(replay["later_source_rows"], retained)
        for fields in (replay, dict(retention_rows=retained), comparisons):
            for key, observed in fields.items():
                learning.math.equal("learning." + key, observed, data[key])
        result.update(measurement_available=True, **replay, retention_rows=retained, **comparisons)
    if number == 8066:
        raw = json.loads(location(Path(data["raw_directory"]) / "observations.json").read_text())
        learning.math.equal("cache.rows", raw["rows"], data["rows"])
        ratios = cache.reduce_rows(raw["rows"])
        learning.math.equal("cache.ratios", ratios, data["complete_workload_ratios"])
        result.update(
            measurement_available=True,
            rows=raw["rows"],
            ratios=ratios,
            censored_count=sum(r["status"] == "censored" for r in raw["rows"]),
            complete_service_measured=False,
            unpriced=data["acquisition_cost_status"],
        )
    if number == 8068:
        inputs = json.loads(location(Path(data["replay_input_reference"]["path"])).read_text())
        reduced = hardware.reduce(inputs)
        for key, observed in reduced.items():
            learning.math.equal("hardware." + key, observed, data[key])
        result.update(**reduced)
    if number == 8063:
        pools = [
            r
            for r in data["candidate_pool_rows"]
            if r["arm"] == "feedback_constrained" and r["opportunity"].startswith("gradient/")
        ]
        missed = sum(r["numerator"] for r in pools)
        eligible = sum(r["denominator"] for r in pools)
        learning.math.equal("opportunity.numerator", missed, data["missed_opportunity_numerator"])
        learning.math.equal(
            "opportunity.denominator", eligible, data["missed_opportunity_denominator"]
        )
        result.update(
            measurement_available=True,
            candidate_pool_count=len(pools),
            missed_numerator=missed,
            missed_denominator=eligible,
            iid_certificate=False,
        )
    result["primitive_reduction_sha256"] = canonical_hash(result)
    return result


def holm(h3: Json | None, *, valid: bool) -> list[Json]:
    """All registered tests remain in the family, including missing source tests."""
    rows: list[Json] = [
        dict(
            hypothesis=f"H{i + 1}",
            margin=m,
            raw_p=1.0,
            family_p=1.0,
            qualified=False,
            positive_claim=False,
            missing_reason="absent frozen source heads and evaluation tokens",
        )
        for i, m in enumerate((0.01, 0.02, 0.02))
    ]
    if h3:
        qualified = bool(valid and h3["support_passed"] and h3["safety_passed"])
        rows[2].update(
            h3,
            margin=0.02,
            raw_p=h3["tests"][0]["raw_p"],
            qualified=qualified,
            family_p=h3["tests"][0]["raw_p"] if qualified else 1.0,
            missing_reason=None,
        )
    previous = 0.0
    for rank, index in enumerate(sorted(range(3), key=lambda i: (rows[i]["family_p"], i))):
        row = rows[index]
        previous = max(previous, min(1.0, row["family_p"] * (3 - rank)))
        row.update(
            holm_adjusted_p=previous,
            positive_claim=bool(
                row["qualified"] and row.get("qualified_benefit") and previous < 0.05
            ),
            multiplicity="Holm .05 across exactly H1/H2/H3",
            uncertainty_scope="one historically exposed development timeline; seeds do not add n",
        )
    return rows
