"""Shared fixtures for Carnot integration tests.

Integration tests exercise the full pipeline as a user experiences it,
including real constraint extraction, JAX-based verification, and CLI
subprocess calls. No mocking of core components.
"""

import jax

# Force CPU for reproducibility (REQ-VERIFY-001).
jax.config.update("jax_platform_name", "cpu")


# The same foreign-checkout refusal the python suite carries (REQ-INFRA-6810). `testpaths`
# includes this directory and `test_install.py` imports carnot, so `pytest tests/integration`
# from an unpinned worktree was exactly "tests and package from different checkouts" and was
# not refused -- the guard was wired into one of the two conftests that needed it.
from pathlib import Path as _Path  # noqa: E402

import carnot as _carnot  # noqa: E402

from carnot.testing.worktree_import_guard import check as _check_worktree_import  # noqa: E402

_check_worktree_import(
    _Path(__file__).resolve().parents[2], _Path(_carnot.__file__).resolve().parent
)
