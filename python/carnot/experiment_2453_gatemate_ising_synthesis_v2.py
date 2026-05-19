import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List, Union

@dataclass
class GateMateIsingSynthesisV2Artifact:
    honest_verdict: str
    synthesis_completed: bool
    pnr_completed: bool
    gatemate_bitstream_flashed: bool
    lut_utilization: str
    thermal_note: str
    yosys_version: str
    duration_s: float
    preconditions_checked: List[str]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GateMateIsingSynthesisV2Artifact":
        return cls(**data)

    @classmethod
    def load(cls, path: str) -> "GateMateIsingSynthesisV2Artifact":
        content = Path(path).read_text()
        return cls.from_dict(json.loads(content))
