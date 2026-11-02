"""REQ-VERIFY-8633 / SCENARIO-VERIFY-8633-COMPATIBILITY: preserve import contracts."""

import ast
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


def child(code, tmp_path):
    """Fresh processes prevent earlier training imports from hiding eager loading."""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_lean_import_and_no_path_mutation(tmp_path):
    """REQ-VERIFY-8633: module imports leave the application search path alone."""
    child(
        """
import sys
before = list(sys.path)
from carnot.inference import resident_protocol
assert sys.path == before
assert not any(n.startswith(('carnot.training', 'carnot.reporting', 'jax')) for n in sys.modules)
assert resident_protocol.MAX_FRAME == 65536
assert resident_protocol.MAX_REQUESTS == 1025
""",
        tmp_path,
    )


def test_original_export_contract(tmp_path):
    """REQ-VERIFY-8633: all original from-import names resolve to original objects."""
    child(
        """
import importlib
import carnot
assert carnot.JAX_AVAILABLE is True
assert carnot.Path is __import__('pathlib').Path
expected = {
 'core': ['AutoGradMixin','EnergyFunction','ModelConfig','ModelState'],
 'models': ['BoltzmannConfig','BoltzmannModel','EBTConfig','EBTransformer',
            'GibbsConfig','GibbsModel','IsingConfig','IsingModel'],
 'samplers': ['HMCSampler','LangevinSampler'],
 'training': ['ReplayBuffer','dsm_loss','dsm_loss_stochastic','nce_loss',
 'nce_loss_stochastic','nce_loss_with_replay','optimization_training_loss',
 'snl_loss','snl_loss_stochastic']}
names = ['__version__','RUST_AVAILABLE','JAX_AVAILABLE']
for module, symbols in expected.items():
 for symbol in symbols:
  scope = {}
  exec('from carnot import ' + symbol, scope)
  assert scope[symbol] is getattr(importlib.import_module('carnot.' + module), symbol)
 names.extend(symbols)
assert carnot.__all__ == names
from carnot._version import __version__
from carnot._rust_compat import RUST_AVAILABLE
assert carnot.__version__ == __version__
assert carnot.RUST_AVAILABLE is RUST_AVAILABLE
assert carnot.JAX_AVAILABLE is True
assert carnot.jax.__name__ == 'jax'
assert carnot.jnp.__name__ == 'jax.numpy'
try: carnot.unknown_export
except AttributeError: pass
else: raise AssertionError('unknown attribute accepted')
""",
        tmp_path,
    )


def test_optional_jax_absence(tmp_path):
    """REQ-VERIFY-8633: missing JAX retains the original small export list."""
    child(
        """
import sys, importlib.abc
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname == 'jax' or fullname.startswith('jax.'):
   raise ModuleNotFoundError('test absent JAX', name='jax')
sys.meta_path.insert(0, Block())
import carnot
assert carnot.__all__ == ['__version__','RUST_AVAILABLE','JAX_AVAILABLE']
assert carnot.JAX_AVAILABLE is False and carnot.RUST_AVAILABLE is False
assert carnot.jax is None
try: carnot.jnp
except AttributeError: pass
else: raise AssertionError('unavailable numpy export')
try: carnot.IsingModel
except (AttributeError, ModuleNotFoundError): pass
else: raise AssertionError('unavailable model exported')
""",
        tmp_path,
    )


def test_module_cli_help_and_rejection(tmp_path):
    """REQ-VERIFY-8633: package CLI exists and rejects ambiguous configuration."""
    for args, expected in [(["--help"], 0), ([], 2), (["x", "hash", "bad"], 2)]:
        result = subprocess.run(
            [sys.executable, "-m", "carnot.inference.resident_protocol", *args],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == expected, result.stderr
    from carnot.inference import resident_protocol as p

    assert p.main([str(tmp_path / "missing"), "hash", "python"]) == 1
    with pytest.raises(SystemExit):
        p.main(["x", "hash", "python", "--idle-timeout", "nan"])


def test_legacy_contract_snapshot():
    """REQ-VERIFY-8633: public inference exports retain their exact initial order."""
    from carnot import inference

    assert inference.__all__[:4] == [
        "DualGPUExecutionResult",
        "DualGPURunner",
        "estimate_model_size_billions",
        "requires_device_map_auto",
    ]
    for name in inference.__all__[:4]:
        scope = {}
        exec("from carnot.inference import " + name, scope)
        assert scope[name] is getattr(inference, name)
    assert inference.model_server is inference.model_server
    with pytest.raises(AttributeError):
        _ = inference.unknown_export
    tree = ast.parse((ROOT / "scripts/experiments/resident_worker_8608.py").read_text())
    assert any(isinstance(n, ast.FunctionDef) and n.name == "main" for n in tree.body)
