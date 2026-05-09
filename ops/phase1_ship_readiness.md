# Phase 1 Ship Readiness

Phase 1 is treated here as a software-operational ship gate only. Paper, arXiv, GateMate, PolarFire SoC, and other hardware validation are out of scope for this ledger.

Honest verdict: `blocked_9_items_remaining`
Blocking items: `9`

## Checklist

| Gate | Status |
|------|--------|
| PyPI package | FAIL |
| HuggingFace mirror | FAIL |
| Second mirror | FAIL |
| MCP/CLI docs | FAIL |
| Independent reproducer path | PASS |
| Safe local smoke | PASS |

## Exact Blockers

1. `pypi_package_ready`: public docs still use the unavailable package name `pip install carnot`: README.md:18: Rust + Python/JAX, Apache 2.0, `pip install carnot`.; README.md:192: GPU: `pip install carnot[cuda]` for CUDA 12. On AMD/ROCm, use; README.md:193: `JAX_PLATFORMS=cpu`. Rust bindings (optional): `pip install carnot[rust]`; docs/getting-started.md:13: > JAX runs on CPU by default. For GPU acceleration, install with `pip install carnot[cuda]`.; docs/usage-guide.md:27: - **Install:** `pip install carnot` (~50MB with JAX CPU); docs/usage-guide.md:39: - **Optional:** TensorRT-LLM for 2-4x inference speedup (`pip install carnot[cuda]`); examples/README.md:22: pip install carnot

   Commands:
   - `python3 -m venv /tmp/carnot-phase1-package-smoke`
   - `/tmp/carnot-phase1-package-smoke/bin/python -m pip install --upgrade pip`
   - `/tmp/carnot-phase1-package-smoke/bin/python -m pip install --no-deps .`
   - `/tmp/carnot-phase1-package-smoke/bin/python -c "import carnot; print(carnot.__version__)"`
   - `/tmp/carnot-phase1-package-smoke/bin/carnot --help`
   - `python -m build && twine check dist/*`

2. `pypi_package_ready`: HuggingFace model loading paths import huggingface_hub, but pyproject.toml does not declare huggingface-hub in dependencies or optional extras.

   Commands:
   - `python3 -m venv /tmp/carnot-phase1-package-smoke`
   - `/tmp/carnot-phase1-package-smoke/bin/python -m pip install --upgrade pip`
   - `/tmp/carnot-phase1-package-smoke/bin/python -m pip install --no-deps .`
   - `/tmp/carnot-phase1-package-smoke/bin/python -c "import carnot; print(carnot.__version__)"`
   - `/tmp/carnot-phase1-package-smoke/bin/carnot --help`
   - `python -m build && twine check dist/*`

3. `hf_mirror_ready`: docs/huggingface-plan.md lists data/token_activations_tqa_qwen35.safetensors, but that dataset artifact is not present locally.

   Commands:
   - `find exports -maxdepth 2 -type f | sort`
   - `python - <<'PY'
from pathlib import Path
for p in sorted(Path('exports').glob('per-token-ebm-*')):
    print(p.name, all((p / f).exists() for f in ['README.md','config.json','model.safetensors','training_metadata.json']))
PY`
   - `huggingface-cli repo-files Carnot-EBM/<repo-id>`

4. `hf_mirror_ready`: docs/huggingface-plan.md still contains unresolved 'Action needed' export work.

   Commands:
   - `find exports -maxdepth 2 -type f | sort`
   - `python - <<'PY'
from pathlib import Path
for p in sorted(Path('exports').glob('per-token-ebm-*')):
    print(p.name, all((p / f).exists() for f in ['README.md','config.json','model.safetensors','training_metadata.json']))
PY`
   - `huggingface-cli repo-files Carnot-EBM/<repo-id>`

5. `hf_mirror_ready`: README.md still says only two Phase 1 research artifacts are published, which conflicts with later per-token/model-card references.

   Commands:
   - `find exports -maxdepth 2 -type f | sort`
   - `python - <<'PY'
from pathlib import Path
for p in sorted(Path('exports').glob('per-token-ebm-*')):
    print(p.name, all((p / f).exists() for f in ['README.md','config.json','model.safetensors','training_metadata.json']))
PY`
   - `huggingface-cli repo-files Carnot-EBM/<repo-id>`

6. `second_mirror_ready`: results/ipfs_mirrors.json lacks content-addressed CIDs for required keys: per_token_ebm_exports, pypi_sdist

   Commands:
   - `ipfs add -r exports/per-token-ebm-*`
   - `ipfs add dist/carnot_ebm-*.tar.gz`
   - `python -m json.tool results/ipfs_mirrors.json`

7. `mcp_cli_docs_ready`: .mcp.json.example/docs still point at tools/verify-mcp/server.py instead of the packaged `python -m carnot.mcp` entry point.

   Commands:
   - `carnot verify examples/math_funcs.py --func gcd --test '(12,8):4' --test '(7,13):1'`
   - `carnot verify-code examples/math_funcs.py --func gcd --pbt`
   - `python -m carnot.mcp`

8. `mcp_cli_docs_ready`: README.md reports 7 MCP tools, but python/carnot/mcp/server.py exposes 9.

   Commands:
   - `carnot verify examples/math_funcs.py --func gcd --test '(12,8):4' --test '(7,13):1'`
   - `carnot verify-code examples/math_funcs.py --func gcd --pbt`
   - `python -m carnot.mcp`

9. `mcp_cli_docs_ready`: docs/integrator-guide.md is missing.

   Commands:
   - `carnot verify examples/math_funcs.py --func gcd --test '(12,8):4' --test '(7,13):1'`
   - `carnot verify-code examples/math_funcs.py --func gcd --pbt`
   - `python -m carnot.mcp`

## Independent Reproducer Plan

Safe local smoke, no publishing:

```bash
python3 -m venv /tmp/carnot-phase1-repro
/tmp/carnot-phase1-repro/bin/python -m pip install --upgrade pip
/tmp/carnot-phase1-repro/bin/python -m pip install --no-deps .
/tmp/carnot-phase1-repro/bin/python -c "import carnot; print(carnot.__version__)"
/tmp/carnot-phase1-repro/bin/carnot --help
```

Independent path before declaring Phase 1 shipped:

```bash
python3 -m venv /tmp/carnot-phase1-independent
/tmp/carnot-phase1-independent/bin/python -m pip install carnot-ebm
/tmp/carnot-phase1-independent/bin/python -c "import carnot; print(carnot.__version__)"
/tmp/carnot-phase1-independent/bin/carnot verify examples/math_funcs.py --func gcd --test '(12,8):4' --test '(7,13):1'
```

CI path before declaring Phase 1 shipped:

```bash
python3 -m build
twine check dist/*
python3 -m venv /tmp/carnot-wheel-smoke
/tmp/carnot-wheel-smoke/bin/python -m pip install dist/*.whl
/tmp/carnot-wheel-smoke/bin/carnot --help
```

## What Remains Before Phase 1 Ship

Resolve every blocker above, then collect one independent reproducer log.
