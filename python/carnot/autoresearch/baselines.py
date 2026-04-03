"""Baseline registry for autoresearch evaluation.

Maintains versioned performance baselines that hypotheses are evaluated against.

Spec: REQ-AUTO-002
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class BenchmarkMetrics:
    """Metrics from a single benchmark run.

    Spec: REQ-AUTO-001
    """

    benchmark_name: str
    final_energy: float
    convergence_steps: int
    wall_clock_seconds: float
    peak_memory_mb: float = 0.0


@dataclass
class BaselineRecord:
    """Complete baseline record for autoresearch evaluation.

    Spec: REQ-AUTO-002
    """

    version: str = "0.1.0"
    commit: str = ""
    timestamp: str = ""
    benchmarks: dict[str, BenchmarkMetrics] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        """Save baseline to JSON file.

        Spec: REQ-AUTO-002
        """
        data = {
            "version": self.version,
            "commit": self.commit,
            "timestamp": self.timestamp,
            "benchmarks": {
                name: {
                    "benchmark_name": m.benchmark_name,
                    "final_energy": m.final_energy,
                    "convergence_steps": m.convergence_steps,
                    "wall_clock_seconds": m.wall_clock_seconds,
                    "peak_memory_mb": m.peak_memory_mb,
                }
                for name, m in self.benchmarks.items()
            },
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> BaselineRecord:
        """Load baseline from JSON file.

        Spec: REQ-AUTO-002
        """
        data = json.loads(path.read_text())
        record = cls(
            version=data["version"],
            commit=data.get("commit", ""),
            timestamp=data.get("timestamp", ""),
        )
        for name, m in data["benchmarks"].items():
            record.benchmarks[name] = BenchmarkMetrics(**m)
        return record
