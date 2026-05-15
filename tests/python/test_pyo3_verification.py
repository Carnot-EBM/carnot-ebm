import pytest

def test_pyo3_bindings_load():
    try:
        import carnot._rust
    except ImportError as e:
        pytest.fail(f"Failed to load pyo3 rust bindings: {e}")
