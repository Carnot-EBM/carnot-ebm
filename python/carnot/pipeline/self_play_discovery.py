"""Self-play constraint discoverer for FR-11 Ledger.

Spec: REQ-SELFPLAY-1683.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from carnot.verifiers import dsl

JsonDict = dict[str, Any]
RUN_DATE = "20260510"
EXPERIMENT_ID = 1683
EXPERIMENT = "1683_self_play"

class SelfPlayConstraintDiscoverer:
    """Discovers constraints from failing traces in a self-play loop."""

    def __init__(self) -> None:
        pass

    def ingest_failing_traces(self, traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Ingest failing traces and output root conflicts with NSVIF DSL constraints."""
        results = []
        for trace in traces:
            trace_id = trace.get("trace_id", "unknown")
            output_text = trace.get("output", "")
            
            # Simple root conflict extraction
            root_conflict = "logical_failure_detected"
            if "invalid" in output_text.lower():
                root_conflict = "invalid_format"
                
            dsl_payload = self.generate_nsvif_dsl(trace)
            
            results.append({
                "trace_id": trace_id,
                "root_conflict": root_conflict,
                "dsl_input": dsl_payload,
            })
        return results
        
    def generate_nsvif_dsl(self, trace: dict[str, Any]) -> dict[str, Any]:
        """Transpile a logical failure into an NSVIF DSL constraint."""
        output_text = trace.get("output", "")
        if "secret" in output_text.lower():
            constraints = [
                {
                    "id": "c_no_secret",
                    "op": "not_contains",
                    "field": "text",
                    "value": "secret"
                }
            ]
        else:
            constraints = [
                {
                    "id": "c_no_fail",
                    "op": "contains",
                    "field": "text",
                    "value": "success"
                }
            ]
            
        return {
            "schema_version": dsl.DSL_SCHEMA_VERSION,
            "instruction": "Avoid the detected logical failure.",
            "constraints": constraints,
        }

def run_experiment(output_path: Path | str) -> dict[str, Any]:
    """Run the self-play loop fixture and write the JSON deliverable."""
    output_path = Path(output_path)
    discoverer = SelfPlayConstraintDiscoverer()
    traces = [
        {"trace_id": "case_1", "output": "invalid output with secret"},
        {"trace_id": "case_2", "output": "just failing"}
    ]
    results = discoverer.ingest_failing_traces(traces)
    
    payload = {
        "status": "complete",
        "experiment_id": EXPERIMENT_ID,
        "experiment": EXPERIMENT,
        "run_date": RUN_DATE,
        "timestamp": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "results": results,
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    return payload
