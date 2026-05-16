# Phase 1 Ship-Track Status

**Last Updated:** 2026-05-16

| Prong | Status | Evidence / URL | Date | Next Action |
|-------|--------|----------------|------|-------------|
| PyPI Publish Workflow | Pending | [Workflow Run](https://github.com/Carnot-EBM/carnot-ebm/actions/runs/25951913694) | 2026-05-16T05:34:00Z | Operator manual approval at GH Environment 'pypi' |
| HuggingFace Mirror | Shipped | [ThinkPRM-v3](https://huggingface.co/Carnot-EBM/ThinkPRM-v3) | 2026-05-16T05:45:39Z | None |
| MCP Integrator Docs | Shipped | `docs/integrator-guide.md` | 2026-05-16T12:00:00Z | None |
| Independent Reproducer | Pending | `ops/phase-1-reproducers.md` | 2026-05-16T07:35:54Z | Pending remote trigger |

**Ship Percentage:** 50% (2/4 prongs shipped)

### Operator Action Items
* Approve PyPI workflow at `https://github.com/Carnot-EBM/carnot-ebm/actions/runs/25951913694`
* Trigger or await remote trigger for Phase 1 Reproducer workflow

### Cross-References
* **Fast-Slow Codification:** Verified complete via `results/experiment_1929_fast_slow_codification.json`.

## Update 2026-05-16T12:02:27.847409+00:00
# Phase 1 Ship-Track Dashboard

- PyPI: Pending operator approval.
- Huggingface: Shipped.
- Fast-Slow Codification: Shipped.
- MCP/CLI Docs: Shipped.
- Independent Reproducer: Shipped.

Overall ship percentage: 80%.

## Update 2026-05-16T15:00:00Z - Phase 1 Consolidated Audit

| Prong | Status | Evidence | Date | Next Action |
|-------|--------|----------|------|-------------|
| PyPI Publish Workflow | PENDING | experiment_2011_pypi_final_recheck.json | 2026-05-16 | Operator manual approval required |
| HuggingFace Mirror | SHIPPED | experiment_1931_huggingface_mirror.json | 2026-05-16 | None |
| MCP Integrator Docs | SHIPPED | experiment_1981_mcp_cli_integrator_docs.json | 2026-05-16 | None |
| Independent Reproducer | SHIPPED | experiment_1982_independent_reproducer.json | 2026-05-16 | None |
| Fast-Slow Codification | SHIPPED | experiment_1929_fast_slow_codification.json | 2026-05-16 | None |

**Ship Percentage:** 80% (4/5 prongs shipped)

### Bash-Failure Window Observations
SKIP-cascade due to unhealed pre-test failures starting after a 600s stall failure, alongside DOOMED_RERUN_BLOCK and GATE_BLOCKs on downstream hardware tasks since 2026-05-16T13:13 UTC.
