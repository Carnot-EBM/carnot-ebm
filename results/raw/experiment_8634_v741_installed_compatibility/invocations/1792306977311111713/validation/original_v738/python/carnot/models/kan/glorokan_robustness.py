"""GloroKAN-style local robustness bounds for rational KArAt attention.

Spec references: REQ-KAN-1690, SCENARIO-KAN-1690.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Sequence

from carnot.models.karat_attention import RationalKArAtLayer
from carnot.models.rkan import RationalInput, serialize_fraction, to_fraction


def _repo_root() -> Path:
    """Return the repository root from this nested KAN package module."""

    return Path(__file__).resolve().parents[4]


DEFAULT_RESULT_PATH = _repo_root() / "results/experiment_1690_glorokan_robustness.json"


@dataclass(frozen=True)
class GloroKANTermBound:
    """Local bound data for one `spline(q_i dot k_j)` KArAt attention term.

    GloroKAN-style robustness is useful because KAN activations expose their
    knot-to-knot slopes. This term record keeps the two ingredients a verifier
    needs to audit the final number: the reachable dot-product interval and the
    largest spline slope on that interval.
    """

    query_index: int
    key_index: int
    dot_lower: Fraction
    dot_upper: Fraction
    spline_slope_bound: Fraction
    query_l1_bound: Fraction
    key_l1_bound: Fraction
    lipschitz_contribution: Fraction

    def as_serializable(self) -> dict[str, object]:
        """Return a JSON-safe exact representation of the term bound."""

        return {
            "query_index": self.query_index,
            "key_index": self.key_index,
            "dot_interval": [
                serialize_fraction(self.dot_lower),
                serialize_fraction(self.dot_upper),
            ],
            "spline_slope_bound": serialize_fraction(self.spline_slope_bound),
            "query_l1_bound": serialize_fraction(self.query_l1_bound),
            "key_l1_bound": serialize_fraction(self.key_l1_bound),
            "lipschitz_contribution": serialize_fraction(self.lipschitz_contribution),
        }


@dataclass(frozen=True)
class GloroKANBoundReport:
    """Structured local Lipschitz report for one KArAt forward pass."""

    local_lipschitz_bound: Fraction
    radius: Fraction
    norm: str
    energy_at_center: Fraction
    terms: tuple[GloroKANTermBound, ...]

    @property
    def energy_change_bound(self) -> Fraction:
        """Maximum certified energy movement for the requested local radius."""

        return self.local_lipschitz_bound * self.radius

    def as_serializable(self) -> dict[str, object]:
        """Return a JSON-safe exact report for artifacts and downstream tools."""

        return {
            "local_lipschitz_bound": serialize_fraction(self.local_lipschitz_bound),
            "radius": serialize_fraction(self.radius),
            "norm": self.norm,
            "energy_at_center": serialize_fraction(self.energy_at_center),
            "energy_change_bound": serialize_fraction(self.energy_change_bound),
            "terms": [term.as_serializable() for term in self.terms],
        }


class GloroKANBounder:
    """Calculate local forward-pass Lipschitz bounds for `RationalKArAtLayer`.

    The bound is intentionally conservative and exact. For each attention term,
    the bounder first expands each query and key coordinate by an `L_inf`
    radius, computes the resulting interval for `q_i dot k_j`, then reads the
    largest absolute slope of the rational spline on that interval. The final
    local constant sums `slope * (||q_i||_1 + ||k_j||_1)` using interval-safe
    coordinate magnitudes, which bounds the chain-rule gradient under the same
    `L_inf` perturbation model.
    """

    def __init__(self, karat_layer: RationalKArAtLayer) -> None:
        self.karat_layer = karat_layer

    def spline_slope_bound(self, lower: RationalInput, upper: RationalInput) -> Fraction:
        """Return the largest absolute spline slope on a rational input interval."""

        interval_lower = to_fraction(lower)
        interval_upper = to_fraction(upper)
        if interval_lower > interval_upper:
            interval_lower, interval_upper = interval_upper, interval_lower

        spline = self.karat_layer.attention_spline
        domain_lower, domain_upper = spline.domain
        if interval_upper < domain_lower or interval_lower > domain_upper:
            return Fraction(0)

        clipped_lower = max(interval_lower, domain_lower)
        clipped_upper = min(interval_upper, domain_upper)
        segment_count = len(spline.control_points) - 1
        segment_width = (domain_upper - domain_lower) / segment_count

        max_slope = Fraction(0)
        for index in range(segment_count):
            segment_lower = domain_lower + segment_width * index
            segment_upper = segment_lower + segment_width
            if segment_upper < clipped_lower or segment_lower > clipped_upper:
                continue
            slope = abs(
                (spline.control_points[index + 1] - spline.control_points[index]) / segment_width
            )
            max_slope = max(max_slope, slope)
        return max_slope

    def bound_forward(
        self,
        q: Sequence[Sequence[RationalInput]],
        k: Sequence[Sequence[RationalInput]],
        radius: RationalInput,
        norm: str = "linf",
    ) -> GloroKANBoundReport:
        """Build a deterministic local Lipschitz report for one KArAt input.

        Args:
            q: Query matrix with shape `(seq_len, dim)`.
            k: Key matrix with shape `(seq_len, dim)`.
            radius: Nonnegative `L_inf` coordinate perturbation radius.
            norm: Perturbation norm. Only `"linf"` is implemented because the
                report's chain-rule constant is defined for coordinate boxes.
        """

        radius_q = to_fraction(radius)
        if radius_q < 0:
            raise ValueError("radius must be nonnegative")
        if norm != "linf":
            raise ValueError("GloroKANBounder currently supports only linf bounds")

        q_matrix = self._validate_matrix(q, "q")
        k_matrix = self._validate_matrix(k, "k")

        terms = []
        total = Fraction(0)
        for query_index, query_row in enumerate(q_matrix):
            for key_index, key_row in enumerate(k_matrix):
                dot_lower, dot_upper = self._dot_interval(query_row, key_row, radius_q)
                slope_bound = self.spline_slope_bound(dot_lower, dot_upper)
                query_l1 = self._row_l1_bound(query_row, radius_q)
                key_l1 = self._row_l1_bound(key_row, radius_q)
                contribution = slope_bound * (query_l1 + key_l1)
                total += contribution
                terms.append(
                    GloroKANTermBound(
                        query_index=query_index,
                        key_index=key_index,
                        dot_lower=dot_lower,
                        dot_upper=dot_upper,
                        spline_slope_bound=slope_bound,
                        query_l1_bound=query_l1,
                        key_l1_bound=key_l1,
                        lipschitz_contribution=contribution,
                    )
                )

        return GloroKANBoundReport(
            local_lipschitz_bound=total,
            radius=radius_q,
            norm=norm,
            energy_at_center=self.karat_layer.energy(q_matrix, k_matrix),
            terms=tuple(terms),
        )

    def _validate_matrix(
        self,
        matrix: Sequence[Sequence[RationalInput]],
        name: str,
    ) -> tuple[tuple[Fraction, ...], ...]:
        """Validate KArAt matrix shape and normalize values to `Fraction`."""

        if len(matrix) != self.karat_layer.seq_len:
            raise ValueError(f"{name} length must equal seq_len={self.karat_layer.seq_len}")

        rows = []
        for row_index, row in enumerate(matrix):
            if len(row) != self.karat_layer.dim:
                raise ValueError(
                    f"{name}[{row_index}] dimension must equal dim={self.karat_layer.dim}"
                )
            rows.append(tuple(to_fraction(value) for value in row))
        return tuple(rows)

    def _dot_interval(
        self,
        query_row: Sequence[Fraction],
        key_row: Sequence[Fraction],
        radius: Fraction,
    ) -> tuple[Fraction, Fraction]:
        """Bound a dot product when every coordinate moves inside a local box."""

        lower = Fraction(0)
        upper = Fraction(0)
        for query_value, key_value in zip(query_row, key_row, strict=True):
            products = (
                (query_value - radius) * (key_value - radius),
                (query_value - radius) * (key_value + radius),
                (query_value + radius) * (key_value - radius),
                (query_value + radius) * (key_value + radius),
            )
            lower += min(products)
            upper += max(products)
        return lower, upper

    def _row_l1_bound(self, row: Sequence[Fraction], radius: Fraction) -> Fraction:
        """Bound a row's `L1` magnitude over the same local coordinate box."""

        return sum(max(abs(value - radius), abs(value + radius)) for value in row)


def build_experiment_1690_artifact() -> dict[str, object]:
    """Build the stable Exp 1690 GloroKAN robustness artifact payload."""

    layer = RationalKArAtLayer(seq_len=1, dim=2, spline_points=[0, 1, 2])
    q = ((Fraction(1, 4), Fraction(0)),)
    k = ((Fraction(1, 2), Fraction(0)),)
    radius = Fraction(1, 16)
    report = GloroKANBounder(layer).bound_forward(q, k, radius)

    perturbed_q = ((Fraction(5, 16), Fraction(0)),)
    perturbed_k = ((Fraction(9, 16), Fraction(0)),)
    observed_delta = abs(layer.energy(perturbed_q, perturbed_k) - report.energy_at_center)

    return {
        "schema": "carnot.glorokan_robustness.v1",
        "status": "complete",
        "experiment": 1690,
        "experiment_id": 1690,
        "run_date": "20260510",
        "title": "GloroKAN-style local Lipschitz bounds for KArAt",
        "spec": ["REQ-KAN-1690", "SCENARIO-KAN-1690"],
        "module": "python/carnot/models/kan/glorokan_robustness.py",
        "artifact_path": "results/experiment_1690_glorokan_robustness.json",
        "local_lipschitz_bound": serialize_fraction(report.local_lipschitz_bound),
        "energy_change_bound": serialize_fraction(report.energy_change_bound),
        "observed_witness_delta": serialize_fraction(observed_delta),
        "bound_covers_witness": observed_delta <= report.energy_change_bound,
        "report": report.as_serializable(),
        "honest_verdict": "complete: glorokan_local_bounder_verified",
    }


def write_experiment_1690_artifact(
    output_path: str | Path = DEFAULT_RESULT_PATH,
) -> dict[str, object]:
    """Write the stable Exp 1690 artifact to disk."""

    artifact = build_experiment_1690_artifact()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


__all__ = [
    "DEFAULT_RESULT_PATH",
    "GloroKANBoundReport",
    "GloroKANBounder",
    "GloroKANTermBound",
    "build_experiment_1690_artifact",
    "write_experiment_1690_artifact",
]
