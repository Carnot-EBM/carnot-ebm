"""Shared test fixtures for Carnot Python tests."""

import sys
from pathlib import Path

import jax
import pytest

# Add repo root to sys.path so tests can import from scripts/
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

# Disable JAX GPU for testing (CPU only)
jax.config.update("jax_platform_name", "cpu")
