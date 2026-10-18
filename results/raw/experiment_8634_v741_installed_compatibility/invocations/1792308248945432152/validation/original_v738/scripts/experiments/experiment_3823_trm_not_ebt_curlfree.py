"""Build the Exp 3823 TRM curl-free falsification artifact.

Spec refs: REQ-3823, SCENARIO-3823, SCENARIO-3823-POSITIVE-CONTROL.

The experiment asks whether a trained TRM update field can be written as
negative gradient descent on one scalar energy. A conservative field has a
symmetric Jacobian, so any persistent antisymmetric Jacobian component is
direct evidence that the update dynamics are outside ordinary EBT descent.
"""

from __future__ import annotations

from carnot.serialization_safety import safe_torch_load

import hashlib
import importlib
import json
import math
import time
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ARTIFACT_PATH = REPO_ROOT / "results/experiment_3821_latent_symbol_bridge_unblocked.json"
OUTPUT_PATH = REPO_ROOT / "results/experiment_3823_trm_not_ebt_curlfree.json"
SCHEMA = "carnot.trm_curlfree_falsification.v1"
RANDOM_SEED = 3823
BLOCKED_VERDICT = "blocked_trm_checkpoint_not_available"

POSITIVE_CONTROL_MAX_RESIDUAL = 1e-4
LOW_TRM_RESIDUAL = 5e-2
CURL_PRESENT_MIN = 5e-2
FIT_FAIL_MIN = 1.5e-1

REQUIRED_PRINCIPLES = {
    "jacobian_antisymmetry_fraction": (
        "Fraction of the update Jacobian that is antisymmetric; >0 implies a "
        "non-conservative (curl-bearing) field EBT cannot express."
    ),
    "scalar_potential_fit_residual": (
        "Relative residual of the best -grad(E) fit to TRM's update field; "
        "large residual = TRM != energy descent."
    ),
    "positive_control_fit_residual": (
        "Same fit on a KNOWN conservative field must give ~0 residual -- proves "
        "the test detects conservativity and a large TRM residual is real, not "
        "a fitter failure."
    ),
    "n_states_sampled": (
        "M>=50 instances so the curl/residual estimates are not single-trajectory artifacts."
    ),
    "preconditions_checked": (
        "Standard methodology field; records torch/numpy and checkpoint-source gates before TRM inference."
    ),
    "inference_substrate": (
        "Standard methodology field; names the actual substrate used for deterministic numerical testing."
    ),
    "random_seed": (
        "Standard methodology field; deterministic numerical test, real compute over M states."
    ),
    "reproducibility_checksum": (
        "Standard methodology field; stable checksum catches silent source or configuration drift."
    ),
    "duration_s": (
        "Standard methodology field; measured wall-clock duration for the precondition and diagnostic run."
    ),
}


class LinearUpdateCheckpoint:
    """Minimal torch-loadable update substrate used by tests and future fixtures."""

    def __init__(self, matrix: Any, bias: Any | None = None) -> None:
        torch = importlib.import_module("torch")
        self.matrix = torch.as_tensor(matrix, dtype=torch.float64)
        if self.matrix.ndim != 2 or self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("update_matrix must be square")
        self.bias = torch.zeros(self.matrix.shape[0], dtype=torch.float64)
        if bias is not None:
            self.bias = torch.as_tensor(bias, dtype=torch.float64)
        if self.bias.shape != (self.matrix.shape[0],):
            raise ValueError("update_bias must match update_matrix dimension")
        self.latent_dim = int(self.matrix.shape[0])
        self.substrate_label = "linear_update_checkpoint"

    def forward_delta(self, h: Any) -> Any:
        """Return the discrete-time update delta F(h)."""
        return h @ self.matrix.T + self.bias

    def __call__(self, h: Any) -> Any:
        return h + self.forward_delta(h)


def principled(field_name: str, value: Any) -> dict[str, Any]:
    """Wrap a required metric with the method principle the artifact must expose."""
    return {"value": value, "principle": REQUIRED_PRINCIPLES[field_name]}


def field_value(wrapped: Any) -> Any:
    """Read the value from a principle-bearing artifact field."""
    if not isinstance(wrapped, dict) or "value" not in wrapped or "principle" not in wrapped:
        raise TypeError("artifact field is not principle-bearing")
    return wrapped["value"]


def _import_available(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
    except Exception:
        return False
    return True


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("source artifact must be a JSON object")
    return payload


def resolve_checkpoint_path(source: Any, *, base_dir: Path | None = None) -> Path | None:
    """Resolve a checkpoint source only when it is a local existing file path."""
    if not isinstance(source, str) or not source.strip():
        return None
    source = source.strip()
    if "://" in source:
        return None

    candidates = [Path(source).expanduser()]
    if base_dir is not None:
        candidates.append((base_dir / source).expanduser())
    candidates.append((REPO_ROOT / source).expanduser())

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _torch_load(path: Path) -> Any:
    torch = importlib.import_module("torch")
    try:
        return safe_torch_load(path, map_location="cpu", allow_unsafe_pickle=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def checkpoint_loadable(path: Path | None) -> tuple[bool, str | None]:
    """Return whether a checkpoint path can be torch-loaded."""
    if path is None:
        return False, "checkpoint source is not a local file path"
    try:
        _torch_load(path)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, None


def run_preconditions_check(
    source_artifact_path: Path = SOURCE_ARTIFACT_PATH,
) -> dict[str, Any]:
    """Check torch/numpy availability and the Exp 3821 checkpoint source."""
    preconditions: dict[str, Any] = {
        "torch_available": _import_available("torch"),
        "numpy_available": _import_available("numpy"),
        "source_artifact_available": False,
        "trm_checkpoint_source": None,
        "trm_checkpoint_path": None,
        "trm_checkpoint_loadable": False,
        "checkpoint_load_error": None,
        "block_reason": BLOCKED_VERDICT,
    }

    try:
        source_artifact = _read_json(source_artifact_path)
    except Exception as exc:
        preconditions["checkpoint_load_error"] = f"source_artifact_{type(exc).__name__}: {exc}"
        return preconditions

    preconditions["source_artifact_available"] = True
    checkpoint_source = source_artifact.get("trm_checkpoint_source")
    preconditions["trm_checkpoint_source"] = checkpoint_source
    checkpoint_path = resolve_checkpoint_path(
        checkpoint_source, base_dir=source_artifact_path.parent
    )
    preconditions["trm_checkpoint_path"] = (
        str(checkpoint_path) if checkpoint_path is not None else None
    )

    if not preconditions["torch_available"] or not preconditions["numpy_available"]:
        preconditions["checkpoint_load_error"] = "torch_or_numpy_unavailable"
        return preconditions

    loadable, error = checkpoint_loadable(checkpoint_path)
    preconditions["trm_checkpoint_loadable"] = loadable
    preconditions["checkpoint_load_error"] = error
    preconditions["block_reason"] = None if loadable else BLOCKED_VERDICT
    return preconditions


def load_update_model(checkpoint_path: Path) -> Any:
    """Load a local update model from a supported checkpoint shape."""
    torch = importlib.import_module("torch")
    payload = _torch_load(checkpoint_path)

    if isinstance(payload, dict) and "update_matrix" in payload:
        return LinearUpdateCheckpoint(payload["update_matrix"], payload.get("update_bias"))

    module_type = getattr(torch.nn, "Module")
    if isinstance(payload, module_type):
        payload.eval()
        payload.latent_dim = int(getattr(payload, "latent_dim", getattr(payload, "hidden_size", 0)))
        payload.substrate_label = "torch_module_checkpoint"
        if payload.latent_dim <= 0:
            raise ValueError("torch module checkpoint must expose latent_dim or hidden_size")
        return payload

    if isinstance(payload, dict) and isinstance(payload.get("model"), module_type):
        model = payload["model"]
        model.eval()
        model.latent_dim = int(getattr(model, "latent_dim", getattr(model, "hidden_size", 0)))
        model.substrate_label = "torch_module_checkpoint"
        if model.latent_dim <= 0:
            raise ValueError("nested torch model must expose latent_dim or hidden_size")
        return model

    raise ValueError("unsupported TRM checkpoint payload")


def make_delta_fn(model: Any) -> Callable[[Any], Any]:
    """Build a function returning F(h)=h_next-h for a loaded update model."""
    if hasattr(model, "forward_delta"):
        return model.forward_delta

    def delta_fn(h: Any) -> Any:
        output = model(h)
        if isinstance(output, tuple):
            output = output[0]
        if isinstance(output, dict):
            output = output.get("h_next", output.get("next_state"))
        if output is None or getattr(output, "shape", None) != h.shape:
            raise ValueError(
                "model output must be a next latent state with the same shape as input"
            )
        return output - h

    return delta_fn


def sample_update_pairs(
    model: Any,
    *,
    n_instances: int,
    steps: int,
    latent_dim: int,
    random_seed: int,
) -> tuple[Any, Any]:
    """Run the update map over seeded states and collect h_t, delta_h_t pairs."""
    torch = importlib.import_module("torch")
    if n_instances < 50:
        n_instances = 50
    if steps < 1:
        raise ValueError("steps must be >= 1")
    generator = torch.Generator().manual_seed(random_seed)
    h = torch.randn(n_instances, latent_dim, dtype=torch.float64, generator=generator)
    states = []
    deltas = []
    delta_fn = make_delta_fn(model)
    with torch.no_grad():
        for _ in range(steps):
            delta = delta_fn(h)
            states.append(h.detach().clone())
            deltas.append(delta.detach().clone())
            h = h + delta
    return torch.cat(states, dim=0), torch.cat(deltas, dim=0)


def quadratic_conservative_delta(h: Any, energy_matrix: Any) -> Any:
    """Return -grad E for E(h)=0.5*h^T*A*h with symmetric A."""
    return -(h @ energy_matrix.T)


def sample_quadratic_conservative_field(
    *,
    n_instances: int,
    latent_dim: int,
    steps: int,
    random_seed: int,
) -> tuple[Any, Any]:
    """Generate h, delta pairs from a known conservative quadratic energy."""
    torch = importlib.import_module("torch")
    generator = torch.Generator().manual_seed(random_seed)
    h = torch.randn(n_instances, latent_dim, dtype=torch.float64, generator=generator)
    scale = torch.linspace(0.05, 0.35, latent_dim, dtype=torch.float64)
    energy_matrix = torch.diag(scale)
    states = []
    deltas = []
    for _ in range(steps):
        delta = quadratic_conservative_delta(h, energy_matrix)
        states.append(h.detach().clone())
        deltas.append(delta.detach().clone())
        h = h + delta
    return torch.cat(states, dim=0), torch.cat(deltas, dim=0)


def _call_delta_single(delta_fn: Callable[[Any], Any], h: Any) -> Any:
    try:
        output = delta_fn(h)
        if getattr(output, "shape", None) == h.shape:
            return output
    except Exception:
        pass
    output = delta_fn(h.unsqueeze(0)).squeeze(0)
    if output.shape != h.shape:
        raise ValueError("delta_fn must return a vector matching h")
    return output


def jacobian_antisymmetry_fraction(
    delta_fn: Callable[[Any], Any],
    states: Any,
    *,
    max_points: int = 16,
) -> float:
    """Estimate ||0.5*(J-J^T)||_F / ||J||_F over sampled states."""
    torch = importlib.import_module("torch")
    if states.numel() == 0:
        return 0.0
    points = states[:max_points].detach().clone().to(dtype=torch.float64)
    fractions = []
    for point in points:
        h = point.detach().clone().requires_grad_(True)

        def single_fn(z: Any) -> Any:
            return _call_delta_single(delta_fn, z)

        jacobian = torch.autograd.functional.jacobian(single_fn, h)
        antisymmetric = 0.5 * (jacobian - jacobian.T)
        denom = torch.linalg.norm(jacobian).item()
        if denom <= 1e-12:
            fractions.append(0.0)
        else:
            fractions.append(torch.linalg.norm(antisymmetric).item() / denom)
    return float(sum(fractions) / len(fractions))


def scalar_potential_fit_residual(states: Any, deltas: Any) -> float:
    """Fit the best quadratic scalar potential and return relative residual.

    The fitted family is E(h)=0.5*h^T*S*h+b^T*h+c with symmetric S. Its
    negative gradient is the closest affine conservative field in least squares.
    """
    torch = importlib.import_module("torch")
    states = states.detach().clone().to(dtype=torch.float64)
    deltas = deltas.detach().clone().to(dtype=torch.float64)
    if states.shape != deltas.shape or states.ndim != 2:
        raise ValueError("states and deltas must be matching rank-2 tensors")

    n_rows, latent_dim = states.shape
    parameter_index: dict[tuple[int, int], int] = {}
    cursor = 0
    for row in range(latent_dim):
        for col in range(row, latent_dim):
            parameter_index[(row, col)] = cursor
            cursor += 1
    bias_start = cursor
    n_parameters = cursor + latent_dim

    design = torch.zeros(n_rows * latent_dim, n_parameters, dtype=torch.float64)
    target = (-deltas).reshape(-1)
    for sample_idx in range(n_rows):
        h = states[sample_idx]
        for output_dim in range(latent_dim):
            design_row = sample_idx * latent_dim + output_dim
            for input_dim in range(latent_dim):
                key = (
                    (output_dim, input_dim) if output_dim <= input_dim else (input_dim, output_dim)
                )
                design[design_row, parameter_index[key]] += h[input_dim]
            design[design_row, bias_start + output_dim] = 1.0

    solution = torch.linalg.lstsq(design, target).solution
    predicted_grad = (design @ solution).reshape(n_rows, latent_dim)
    residual_norm = torch.linalg.norm(deltas + predicted_grad).item()
    field_norm = torch.linalg.norm(deltas).item()
    if field_norm <= 1e-12:
        return 0.0 if residual_norm <= 1e-12 else math.inf
    return float(residual_norm / field_norm)


def run_positive_control(*, random_seed: int, latent_dim: int) -> float:
    """Run the scalar-potential fit on a known conservative field."""
    states, deltas = sample_quadratic_conservative_field(
        n_instances=64,
        latent_dim=max(2, min(latent_dim, 8)),
        steps=3,
        random_seed=random_seed + 17,
    )
    return scalar_potential_fit_residual(states, deltas)


def classify_verdict(
    jacobian_antisymmetry_fraction_value: float,
    scalar_potential_fit_residual_value: float,
    *,
    positive_control_residual: float | None,
) -> str:
    """Apply the terminal gate for Exp 3823."""
    if (
        positive_control_residual is None
        or positive_control_residual > POSITIVE_CONTROL_MAX_RESIDUAL
    ):
        return "complete: INCONCLUSIVE_curlfree_positive_control_failed"
    if scalar_potential_fit_residual_value <= LOW_TRM_RESIDUAL:
        return (
            "complete: trm_is_secretly_energy_descent_surprising_residual"
            f"{scalar_potential_fit_residual_value:.6f}"
        )
    if (
        jacobian_antisymmetry_fraction_value >= CURL_PRESENT_MIN
        or scalar_potential_fit_residual_value >= FIT_FAIL_MIN
    ):
        return "complete: trm_not_ebt_curlfree_falsified_asymmetric_field_bounded_ebt_does_not_cover_trm"
    return (
        "complete: trm_is_secretly_energy_descent_surprising_residual"
        f"{scalar_potential_fit_residual_value:.6f}"
    )


def _checksum(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def blocked_artifact(
    *,
    preconditions: dict[str, Any],
    duration_s: float,
    random_seed: int,
) -> dict[str, Any]:
    """Build the fail-closed artifact when the TRM checkpoint is unavailable."""
    checksum = _checksum(
        {
            "schema": SCHEMA,
            "source": preconditions.get("trm_checkpoint_source"),
            "path": preconditions.get("trm_checkpoint_path"),
            "random_seed": random_seed,
            "blocked": True,
        }
    )
    return {
        "schema": SCHEMA,
        "honest_verdict": BLOCKED_VERDICT,
        "trm_checkpoint_source": preconditions.get("trm_checkpoint_source"),
        "jacobian_antisymmetry_fraction": principled("jacobian_antisymmetry_fraction", None),
        "scalar_potential_fit_residual": principled("scalar_potential_fit_residual", None),
        "positive_control_fit_residual": principled("positive_control_fit_residual", None),
        "n_states_sampled": principled("n_states_sampled", 0),
        "preconditions_checked": principled("preconditions_checked", preconditions),
        "inference_substrate": principled("inference_substrate", "none (blocked)"),
        "random_seed": principled("random_seed", random_seed),
        "reproducibility_checksum": principled("reproducibility_checksum", checksum),
        "duration_s": principled("duration_s", float(duration_s)),
        "field_principles": REQUIRED_PRINCIPLES,
    }


def build_artifact(
    *,
    source_artifact_path: Path = SOURCE_ARTIFACT_PATH,
    n_instances: int = 64,
    steps: int = 4,
    latent_dim: int | None = None,
    random_seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Build the terminal Exp 3823 artifact."""
    started_at = time.time()
    preconditions = run_preconditions_check(source_artifact_path)
    if preconditions.get("block_reason") == BLOCKED_VERDICT:
        return blocked_artifact(
            preconditions=preconditions,
            duration_s=time.time() - started_at,
            random_seed=random_seed,
        )

    checkpoint_path = Path(str(preconditions["trm_checkpoint_path"]))
    try:
        model = load_update_model(checkpoint_path)
    except Exception as exc:
        preconditions["trm_checkpoint_loadable"] = False
        preconditions["checkpoint_load_error"] = f"{type(exc).__name__}: {exc}"
        preconditions["block_reason"] = BLOCKED_VERDICT
        return blocked_artifact(
            preconditions=preconditions,
            duration_s=time.time() - started_at,
            random_seed=random_seed,
        )

    resolved_latent_dim = int(latent_dim or getattr(model, "latent_dim"))
    states, deltas = sample_update_pairs(
        model,
        n_instances=n_instances,
        steps=steps,
        latent_dim=resolved_latent_dim,
        random_seed=random_seed,
    )
    delta_fn = make_delta_fn(model)
    asymmetry = jacobian_antisymmetry_fraction(delta_fn, states)
    potential_residual = scalar_potential_fit_residual(states, deltas)
    positive_control_residual = run_positive_control(
        random_seed=random_seed,
        latent_dim=resolved_latent_dim,
    )
    verdict = classify_verdict(
        asymmetry,
        potential_residual,
        positive_control_residual=positive_control_residual,
    )
    substrate_label = getattr(model, "substrate_label", "torch_checkpoint")
    duration_s = time.time() - started_at
    checksum = _checksum(
        {
            "schema": SCHEMA,
            "checkpoint_path": str(checkpoint_path),
            "n_instances": n_instances,
            "steps": steps,
            "latent_dim": resolved_latent_dim,
            "random_seed": random_seed,
            "asymmetry": round(asymmetry, 12),
            "potential_residual": round(potential_residual, 12),
            "positive_control_residual": round(positive_control_residual, 12),
        }
    )

    return {
        "schema": SCHEMA,
        "honest_verdict": verdict,
        "trm_checkpoint_source": preconditions.get("trm_checkpoint_source"),
        "jacobian_antisymmetry_fraction": principled("jacobian_antisymmetry_fraction", asymmetry),
        "scalar_potential_fit_residual": principled(
            "scalar_potential_fit_residual", potential_residual
        ),
        "positive_control_fit_residual": principled(
            "positive_control_fit_residual", positive_control_residual
        ),
        "n_states_sampled": principled("n_states_sampled", int(states.shape[0])),
        "preconditions_checked": principled("preconditions_checked", preconditions),
        "inference_substrate": principled(
            "inference_substrate", f"{substrate_label}:{checkpoint_path}"
        ),
        "random_seed": principled("random_seed", random_seed),
        "reproducibility_checksum": principled("reproducibility_checksum", checksum),
        "duration_s": principled("duration_s", float(duration_s)),
        "n_instances": n_instances,
        "n_refinement_steps": steps,
        "latent_dim": resolved_latent_dim,
        "field_principles": REQUIRED_PRINCIPLES,
    }


def write_artifact(artifact: dict[str, Any], output_path: Path = OUTPUT_PATH) -> None:
    """Persist the artifact as stable JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    """Entrypoint for the requested experiment command."""
    artifact = build_artifact()
    write_artifact(artifact, OUTPUT_PATH)


if __name__ == "__main__":  # pragma: no cover
    main()
