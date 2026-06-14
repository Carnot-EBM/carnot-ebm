"""Entrypoint for Exp 4178 GAP-3 Stage-1 latent energy rescore."""

from __future__ import annotations

from carnot.research.gap3_stage1_model_native_arc_energy_4178 import (
    DEFAULT_OUTPUT,
    write_experiment_artifact,
)


def main() -> int:
    """Run the CPU-only scorer and write the terminal JSON artifact."""

    output_path = write_experiment_artifact(output_path=DEFAULT_OUTPUT)
    print(output_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
