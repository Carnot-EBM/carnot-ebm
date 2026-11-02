"""Build the Exp 1361 CPU-only p-dit certificate-state mapping artifact.

The p-dit literature gives Carnot a cleaner way to talk about certificate
states that are inherently categorical.  This module does not emulate hardware:
it compares the bookkeeping cost of binary one-hot p-bits with one q=4
p-dit/p-int variable and uses a tiny exact energy-table check to prove that the
valid one-hot states preserve the same proxy energies.

Spec refs: REQ-HW-048, SCENARIO-HW-048.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PRIOR_PBIT_PACKET_PATH = (
    PROJECT_ROOT / "results" / "experiment_1348_pbit_update_dynamics_dual_bram_packet_v2.json"
)
PROMPT_PRIOR_PBIT_PACKET_PATH = (
    PROJECT_ROOT / "results" / "experiment_1348_pbit_dual_bram_certificate_packet_update.json"
)
DELIVERABLE_PATH = (
    PROJECT_ROOT / "results" / "experiment_1361_pdit_certificate_state_hardware_mapping.json"
)

EXPERIMENT_ID = 1361
SCHEMA = "pdit_certificate_state_hardware_mapping_v1"
DEFAULT_RUN_DATE = "20260505"
CPU_ONLY_HONEST_VERDICT = "cpu_only_pdit_certificate_state_mapping_ready_hardware_not_run"
HARDWARE_HONEST_VERDICT = "hardware_claim_allowed_after_local_execution"
CERTIFICATE_STATES = ("SAT", "UNSAT", "UNKNOWN", "REPAIR")

REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "certificate_states_mapped",
    "binary_spin_count",
    "pdit_variable_count",
    "state_expansion_ratio",
    "energy_equivalence_error",
    "pbit_packet_delta",
    "hardware_claim_allowed",
    "kv260_claim_allowed",
    "next_hardware_requirements",
    "honest_verdict",
}

REFERENCE_BASIS = [
    'arXiv 2506.00269 "Extended-variable probabilistic computing with p-dits"',
    "results/experiment_1348_pbit_update_dynamics_dual_bram_packet_v2.json",
]


def certificate_state_alphabet() -> list[str]:
    """Return the categorical certificate-state alphabet used by the study.

    These states are a single semantic choice, not four independent Boolean
    facts.  Keeping them as a q=4 alphabet is the point of the p-dit mapping.
    """
    return list(CERTIFICATE_STATES)


def build_binary_one_hot_mapping(
    alphabet: Sequence[str] | None = None,
) -> dict[str, list[int]]:
    """Map each certificate state to a binary one-hot p-bit spin vector.

    The p-bit fallback needs one spin per categorical value.  A valid state has
    exactly one +1 spin and all other spins at -1; every other binary
    configuration is bookkeeping overhead that hardware or an energy penalty
    would need to reject.
    """
    states = list(alphabet or certificate_state_alphabet())
    mapping: dict[str, list[int]] = {}
    for active_index, state in enumerate(states):
        spins = [-1 for _state in states]
        spins[active_index] = 1
        mapping[state] = spins
    return mapping


def build_pdit_mapping(alphabet: Sequence[str] | None = None) -> dict[str, dict[str, int]]:
    """Map each certificate state to one q-valued p-dit and p-int code.

    The p-dit code is categorical and the p-int value is the matching integer
    label.  The integer label is only a transport representation here; it is not
    a calibrated physical energy or an ordering claim about certificate quality.
    """
    states = list(alphabet or certificate_state_alphabet())
    q = len(states)
    return {
        state: {"pdit_code": index, "pint_value": index, "alphabet_size": q}
        for index, state in enumerate(states)
    }


def build_certificate_energy_table(alphabet: Sequence[str] | None = None) -> dict[str, float]:
    """Build a tiny deterministic proxy energy table for mapping checks.

    The values are deliberately simple code energies.  They let the study check
    that binary one-hot and q=4 p-dit encodings preserve the same table lookup
    on all valid certificate states without pretending to model analog device
    physics.
    """
    states = list(alphabet or certificate_state_alphabet())
    return {state: float(index) for index, state in enumerate(states)}


def _state_by_binary_vector(
    binary_mapping: Mapping[str, Sequence[int]],
) -> dict[tuple[int, ...], str]:
    return {tuple(int(value) for value in spins): state for state, spins in binary_mapping.items()}


def binary_one_hot_energy(
    spins: Sequence[int],
    binary_mapping: Mapping[str, Sequence[int]],
    energy_table: Mapping[str, float],
    invalid_penalty: float = 10.0,
) -> float:
    """Return the proxy energy for a binary one-hot p-bit state.

    Valid one-hot states read directly from the certificate energy table.
    Invalid binary configurations receive a penalty because one-hot expansion
    creates states that do not correspond to any Carnot certificate label.
    """
    spin_tuple = tuple(int(value) for value in spins)
    inverse = _state_by_binary_vector(binary_mapping)
    if spin_tuple in inverse:
        return float(energy_table[inverse[spin_tuple]])

    active_count = sum(1 for value in spin_tuple if value == 1)
    non_spin_count = sum(1 for value in spin_tuple if value not in {-1, 1})
    length_error = abs(len(spin_tuple) - len(binary_mapping))
    return invalid_penalty + abs(active_count - 1) + non_spin_count + length_error


def pdit_energy(
    pdit_code: int,
    pdit_mapping: Mapping[str, Mapping[str, int]],
    energy_table: Mapping[str, float],
) -> float:
    """Return the proxy energy for a q-valued p-dit/p-int code."""
    code = int(pdit_code)
    for state, row in pdit_mapping.items():
        if int(row["pdit_code"]) == code:
            return float(energy_table[state])
    raise ValueError(f"unknown pdit_code: {pdit_code}")


def compute_energy_equivalence_error(
    binary_mapping: Mapping[str, Sequence[int]],
    pdit_mapping: Mapping[str, Mapping[str, int]],
    energy_table: Mapping[str, float],
) -> float:
    """Compute max valid-state energy mismatch between binary and p-dit maps."""
    max_error = 0.0
    for state, spins in binary_mapping.items():
        binary_energy = binary_one_hot_energy(spins, binary_mapping, energy_table)
        pdit_code = int(pdit_mapping[state]["pdit_code"])
        pdit_table_energy = pdit_energy(pdit_code, pdit_mapping, energy_table)
        max_error = max(max_error, abs(binary_energy - pdit_table_energy))
    return float(max_error)


def build_state_space_proxy(alphabet: Sequence[str] | None = None) -> dict[str, float | int]:
    """Summarize the state-space overhead created by binary one-hot expansion."""
    states = list(alphabet or certificate_state_alphabet())
    binary_spin_count = len(states)
    valid_state_count = len(states)
    raw_binary_configurations = 2**binary_spin_count
    return {
        "binary_raw_configurations": raw_binary_configurations,
        "valid_certificate_states": valid_state_count,
        "invalid_binary_configurations": raw_binary_configurations - valid_state_count,
        "raw_binary_to_valid_state_ratio": raw_binary_configurations / valid_state_count,
    }


def load_prior_pbit_packet(path: str | Path = PRIOR_PBIT_PACKET_PATH) -> dict[str, Any]:
    """Load the Exp 1348 CPU-only p-bit update-dynamics packet."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_pbit_packet_delta(
    prior_pbit_packet: Mapping[str, Any],
    binary_spin_count: int,
    pdit_variable_count: int,
) -> dict[str, Any]:
    """Record what changed relative to Exp 1348's p-bit handoff packet."""
    metadata = prior_pbit_packet.get("metadata", {})
    prior_reuse_rows = prior_pbit_packet.get("reuse_factor_grid", [])
    return {
        "prior_experiment_id": int(metadata.get("experiment_id", 1348)),
        "prior_schema": str(metadata.get("schema", "pbit_update_dynamics_dual_bram_packet_v2")),
        "prior_artifact_used": "results/experiment_1348_pbit_update_dynamics_dual_bram_packet_v2.json",
        "prompt_artifact_path": (
            "results/experiment_1348_pbit_dual_bram_certificate_packet_update.json"
        ),
        "prompt_artifact_path_found": PROMPT_PRIOR_PBIT_PACKET_PATH.exists(),
        "prior_hardware_claim_allowed": bool(prior_pbit_packet.get("hardware_claim_allowed")),
        "prior_kv260_claim_allowed": bool(prior_pbit_packet.get("kv260_claim_allowed")),
        "prior_honest_verdict": str(prior_pbit_packet.get("honest_verdict", "unknown")),
        "prior_reuse_factor_rows": len(prior_reuse_rows),
        "new_mapping": "q4_pdit_or_pint_certificate_state",
        "variable_delta": (
            f"{binary_spin_count} binary p-bit spins in one-hot form collapse to "
            f"{pdit_variable_count} q=4 p-dit/p-int variable"
        ),
        "energy_proxy_delta": (
            "adds exact valid-state energy-table equivalence check; no RTL timing, "
            "DAC, analog, or board energy measurement"
        ),
        "hardware_scope_delta": "still_cpu_only_no_vivado_fpga_kv260_tsu_analog_run",
    }


def build_next_hardware_requirements() -> list[dict[str, str]]:
    """List concrete gates required before any real hardware claim."""
    return [
        {
            "requirement": "define q=4 p-dit/p-int update semantics for certificate states",
            "path": "hardware/kv260/pdit_certificate_state_pkg.sv",
            "claim_gate": "required_before_hardware_claim",
        },
        {
            "requirement": "extend the sampler packet format for q-state energy tables and categorical draws",
            "path": "hardware/kv260/ising_sampler_v8_pdit_certificate.v",
            "claim_gate": "required_before_hardware_claim",
        },
        {
            "requirement": "write a CPU golden-model to RTL equivalence test over all four certificate states",
            "path": "tests/python/test_pdit_certificate_state_rtl_contract.py",
            "claim_gate": "required_before_hardware_claim",
        },
        {
            "requirement": "synthesize the q=4 design with Vivado and record LUT, FF, BRAM, timing, and seed metadata",
            "path": "hardware/kv260/synth_pdit_certificate_state.tcl",
            "claim_gate": "required_before_hardware_claim",
        },
        {
            "requirement": "run the generated bitfile on a real KV260 with AXI/PYNQ readback logs",
            "path": "python/carnot/hardware/pdit_certificate_state_driver.py",
            "claim_gate": "required_before_kv260_claim",
        },
    ]


def hardware_claim_allowed(
    synthesis_performed: bool = False,
    board_executed: bool = False,
    external_hardware_executed: bool = False,
) -> bool:
    """Allow hardware claims only after actual local or external hardware work."""
    return bool(synthesis_performed or board_executed or external_hardware_executed)


def kv260_claim_allowed(synthesis_performed: bool = False, board_executed: bool = False) -> bool:
    """Allow KV260 claims only after synthesis or board execution for that target."""
    return bool(synthesis_performed or board_executed)


def build_artifact(
    project_root: str | Path = PROJECT_ROOT,
    run_date: str = DEFAULT_RUN_DATE,
    prior_pbit_packet: Mapping[str, Any] | None = None,
    synthesis_performed: bool = False,
    board_executed: bool = False,
    external_hardware_executed: bool = False,
) -> dict[str, Any]:
    """Build the complete Exp 1361 CPU-only mapping study artifact."""
    prior = dict(prior_pbit_packet or load_prior_pbit_packet())
    alphabet = certificate_state_alphabet()
    binary_mapping = build_binary_one_hot_mapping(alphabet)
    pdit_mapping = build_pdit_mapping(alphabet)
    energy_table = build_certificate_energy_table(alphabet)
    binary_spin_count = len(alphabet)
    pdit_variable_count = 1
    hardware_claim = hardware_claim_allowed(
        synthesis_performed=synthesis_performed,
        board_executed=board_executed,
        external_hardware_executed=external_hardware_executed,
    )
    kv260_claim = kv260_claim_allowed(
        synthesis_performed=synthesis_performed,
        board_executed=board_executed,
    )
    artifact = {
        "metadata": {
            "experiment_id": EXPERIMENT_ID,
            "schema": SCHEMA,
            "run_date": run_date,
            "project_root": str(project_root),
            "prior_pbit_packet": (
                "results/experiment_1348_pbit_update_dynamics_dual_bram_packet_v2.json"
            ),
            "synthesis_performed": bool(synthesis_performed),
            "board_executed": bool(board_executed),
            "external_hardware_executed": bool(external_hardware_executed),
            "hardware_executed": hardware_claim,
            "local_hardware_synthesis_or_board_run": bool(synthesis_performed or board_executed),
        },
        "status": "complete",
        "reference_basis": REFERENCE_BASIS,
        "certificate_states_mapped": alphabet,
        "certificate_state_descriptions": {
            "SAT": "certificate proves the constraint set is satisfiable",
            "UNSAT": "certificate proves the constraint set is unsatisfiable",
            "UNKNOWN": "certificate does not prove SAT or UNSAT",
            "REPAIR": "certificate records that a repair action is needed before trust",
        },
        "binary_one_hot_mapping": binary_mapping,
        "pdit_pint_mapping": pdit_mapping,
        "binary_spin_count": binary_spin_count,
        "pdit_variable_count": pdit_variable_count,
        "state_expansion_ratio": binary_spin_count / pdit_variable_count,
        "state_space_proxy": build_state_space_proxy(alphabet),
        "energy_table": energy_table,
        "energy_equivalence_error": compute_energy_equivalence_error(
            binary_mapping,
            pdit_mapping,
            energy_table,
        ),
        "energy_equivalence_proxy": {
            "method": "max_abs_valid_one_hot_energy_minus_pdit_energy",
            "invalid_binary_state_penalty": 10.0,
            "hardware_energy_measurement": False,
        },
        "pbit_packet_delta": build_pbit_packet_delta(
            prior,
            binary_spin_count=binary_spin_count,
            pdit_variable_count=pdit_variable_count,
        ),
        "hardware_claim_allowed": hardware_claim,
        "kv260_claim_allowed": kv260_claim,
        "next_hardware_requirements": build_next_hardware_requirements(),
        "honest_verdict": HARDWARE_HONEST_VERDICT if hardware_claim else CPU_ONLY_HONEST_VERDICT,
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate public schema fields and hardware-claim honesty gates."""
    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"missing required artifact fields: {sorted(missing)}")
    if artifact["status"] != "complete":
        raise ValueError("status must be complete")
    expected_states = certificate_state_alphabet()
    if list(artifact["certificate_states_mapped"]) != expected_states:
        raise ValueError(f"certificate_states_mapped must be {expected_states}")
    if int(artifact["binary_spin_count"]) != len(expected_states):
        raise ValueError("binary_spin_count must equal the certificate alphabet size")
    if int(artifact["pdit_variable_count"]) != 1:
        raise ValueError("pdit_variable_count must be one q=4 variable")
    expected_ratio = artifact["binary_spin_count"] / artifact["pdit_variable_count"]
    if float(artifact["state_expansion_ratio"]) != float(expected_ratio):
        raise ValueError("state_expansion_ratio must equal binary_spin_count / pdit_variable_count")
    if float(artifact["energy_equivalence_error"]) != 0.0:
        raise ValueError("energy_equivalence_error must be zero for valid states")
    delta = artifact["pbit_packet_delta"]
    if int(delta.get("prior_experiment_id", -1)) != 1348:
        raise ValueError("pbit_packet_delta must reference Exp 1348")
    if not artifact["next_hardware_requirements"]:
        raise ValueError("next_hardware_requirements must not be empty")

    metadata = artifact.get("metadata", {})
    any_hardware_executed = bool(
        metadata.get("hardware_executed")
        or metadata.get("synthesis_performed")
        or metadata.get("board_executed")
        or metadata.get("external_hardware_executed")
    )
    any_kv260_work = bool(metadata.get("synthesis_performed") or metadata.get("board_executed"))
    if artifact["hardware_claim_allowed"] and not any_hardware_executed:
        raise ValueError("hardware_claim_allowed requires actual hardware execution metadata")
    if artifact["kv260_claim_allowed"] and not any_kv260_work:
        raise ValueError("kv260_claim_allowed requires synthesis_performed or board_executed")


def write_artifact(
    path: str | Path = DELIVERABLE_PATH,
    artifact: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the validated Exp 1361 artifact and return the payload."""
    payload = dict(artifact or build_artifact())
    validate_artifact(payload)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    print(json.dumps(write_artifact(), indent=2, sort_keys=True))
