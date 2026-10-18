import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class GateMateFlashArtifact:
    schema: str
    board: str
    idcode_verified: str
    toolchain: Dict[str, str]
    yosys_invocation: str
    synthesis_completed: bool
    pnr_completed: bool
    lut_utilization: float
    ff_utilization: float
    max_clock_mhz: float
    flash_succeeded: bool
    bitstream_size_bytes: int
    random_seed: int
    reproducibility_checksum: str
    n_samples: int
    n_samples_justification: str
    actual_agent_backend: str
    thermal_note: str
    acceptance_gate_passed: bool
    honest_verdict: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GateMateFlashArtifact":
        return cls(**data)

    @classmethod
    def load(cls, path: str) -> "GateMateFlashArtifact":
        content = Path(path).read_text()
        return cls.from_dict(json.loads(content))
