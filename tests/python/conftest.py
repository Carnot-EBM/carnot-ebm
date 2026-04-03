"""Shared test fixtures for Carnot Python tests."""

import jax
import pytest

# Disable JAX GPU for testing (CPU only)
jax.config.update("jax_platform_name", "cpu")
