"""Shared fixtures for Carnot integration tests.

Integration tests exercise the full pipeline as a user experiences it,
including real constraint extraction, JAX-based verification, and CLI
subprocess calls. No mocking of core components.
"""

import jax
import pytest

# Force CPU for reproducibility (REQ-VERIFY-001).
jax.config.update("jax_platform_name", "cpu")
