"""REQ-REPORT-9208-V784: inherited replay reads each protocol's original owner."""

import json
from pathlib import Path
from typing import Any

from carnot.reporting.current_work_receipt import sha256_file

OWNERS = {
    "entropy_reference_protocol": {
        "original_path": "/home/ianblenke/github.com/ianblenke/carnot/openspec/change-proposals/v782-contract-inputs.json",
        "path": "/home/ianblenke/github.com/ianblenke/carnot/results/raw/experiment_9193_v782_capstone/invocations/1794920809899628968/retained/b03464b51db80706efc6c11f730180fe39d425501a52e2b3bf769e2a52c99421/v782-contract-inputs.json",
        "sha256": "sha256:b03464b51db80706efc6c11f730180fe39d425501a52e2b3bf769e2a52c99421",
    },
    "label_ambiguity_protocol": {
        "path": "/home/ianblenke/github.com/ianblenke/carnot/results/raw/experiment_9194_v783_contract/invocations/1794933643715263146/requalified_final/pre_ces_binding_config.json",
        "sha256": "sha256:e5459a911a5b0754b31176874a82ea475ae9fedf782cc32651b3ebbde41a662f",
    },
}


def load(protocol: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Original configuration hashes cannot be supplied by a candidate being verified."""
    reference = OWNERS[protocol]
    path = Path(reference["path"])
    if not path.is_file() or sha256_file(path) != reference["sha256"]:
        raise ValueError("protocol_owner_hash")
    return json.loads(path.read_bytes())[protocol], dict(reference)


def validate(work: dict[str, Any], protocol: str) -> dict[str, Any]:
    """Rehashing a fabricated sidecar cannot make it equal its original authority."""
    expected, owner = load(protocol)
    reference = work["protocol_sidecars"].get(protocol + "_reference", {})
    path = Path(reference.get("path", ""))
    if (
        not path.is_file()
        or sha256_file(path) != reference.get("sha256")
        or path.read_bytes() != (json.dumps(expected, indent=2, sort_keys=True) + "\n").encode()
    ):
        raise ValueError("protocol_drift:" + protocol)
    return dict(
        protocol=protocol,
        owning_configuration=owner,
        original_hash=owner["sha256"],
        sidecar=reference,
        cold_replay_outcome="validated_in_current_process",
    )
