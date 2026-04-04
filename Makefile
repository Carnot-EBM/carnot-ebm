.PHONY: up down logs status build test clean

# ─── Infrastructure ───────────────────────────────────────────────

## Start all long-running services (Claude API bridge + WebGPU gateway)
up:
	docker compose up -d
	@echo ""
	@echo "Services started:"
	@echo "  Claude API Bridge:  http://localhost:$${BRIDGE_PORT:-8080}/v1/models"
	@echo "  WebGPU Gateway:     http://localhost:$${GATEWAY_PORT:-3000}/"
	@echo ""
	@echo "Check status: make status"

## Stop all services
down:
	docker compose down

## Tail service logs
logs:
	docker compose logs -f

## Check service health
status:
	@echo "=== Claude API Bridge ==="
	@curl -sf http://localhost:$${BRIDGE_PORT:-8080}/health 2>/dev/null && echo " OK" || echo " DOWN"
	@echo "=== WebGPU Gateway ==="
	@curl -sf http://localhost:$${GATEWAY_PORT:-3000}/health 2>/dev/null && echo " OK" || echo " DOWN"
	@echo "=== Workers ==="
	@curl -sf http://localhost:$${GATEWAY_PORT:-3000}/workers 2>/dev/null || echo " N/A"

## Rebuild service images
rebuild:
	docker compose build --no-cache

# ─── Build ────────────────────────────────────────────────────────

## Build Rust workspace (excluding Python bindings)
build:
	cargo build --workspace --exclude carnot-python

## Build Rust in release mode
build-release:
	cargo build --workspace --exclude carnot-python --release

## Build Python bindings
build-python:
	PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo build -p carnot-python

# ─── Test ─────────────────────────────────────────────────────────

## Run all tests (Rust + Python)
test: test-rust test-python

## Run Rust tests
test-rust:
	cargo test --workspace --exclude carnot-python

## Run Python tests with 100% coverage
test-python:
	.venv/bin/pytest tests/python --cov=python/carnot --cov-report=term-missing --cov-fail-under=100

## Run spec coverage check
test-specs:
	.venv/bin/python scripts/check_spec_coverage.py

# ─── Lint ─────────────────────────────────────────────────────────

## Run all linters
lint: lint-rust lint-python

## Rust linting
lint-rust:
	cargo fmt --all -- --check
	cargo clippy --workspace --exclude carnot-python -- -D warnings

## Python linting
lint-python:
	.venv/bin/ruff check python/ tests/
	.venv/bin/ruff format --check python/ tests/
	.venv/bin/mypy python/carnot

# ─── GPU ──────────────────────────────────────────────────────────

## Run GPU vs CPU benchmark
bench-gpu:
	cargo run -p carnot-gpu --example bench_gpu_vs_cpu --release

## Start WebGPU gateway (native, no Docker)
gateway:
	cargo run -p carnot-webgpu-gateway --bin gateway --release

# ─── Autoresearch ─────────────────────────────────────────────────

## Run LLM-powered autoresearch (requires Claude API bridge)
autoresearch:
	.venv/bin/python scripts/run_autoresearch_llm.py --max-iterations 50

## Run LLM benchmark (SAT + coloring)
benchmark:
	.venv/bin/python scripts/run_llm_benchmark.py

## Run code verification autoresearch
autoresearch-code:
	.venv/bin/python scripts/run_code_verification_autoresearch.py

# ─── Research Conductor ───────────────────────────────────

## Run one research step (pick task, implement, test, commit)
research-step:
	.venv/bin/python scripts/research_conductor.py

## Run research conductor in continuous loop (every 30 min)
research-loop:
	.venv/bin/python scripts/research_conductor.py --loop --interval 30 --push

## Dry run — show what conductor would do
research-dry:
	.venv/bin/python scripts/research_conductor.py --dry-run

# ─── Clean ────────────────────────────────────────────────────────

## Clean build artifacts
clean:
	cargo clean
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true

## Show help
help:
	@echo "Carnot — Energy Based Model Framework"
	@echo ""
	@echo "Infrastructure:"
	@echo "  make up          Start Claude API bridge + WebGPU gateway"
	@echo "  make down        Stop all services"
	@echo "  make logs        Tail service logs"
	@echo "  make status      Check service health"
	@echo ""
	@echo "Build:"
	@echo "  make build       Build Rust workspace"
	@echo "  make test        Run all tests (Rust + Python)"
	@echo "  make lint        Run all linters"
	@echo ""
	@echo "GPU:"
	@echo "  make bench-gpu   GPU vs CPU benchmark"
	@echo "  make gateway     Start WebGPU gateway (native)"
	@echo ""
	@echo "Autoresearch:"
	@echo "  make autoresearch     50-iteration LLM autoresearch"
	@echo "  make benchmark        LLM-EBM SAT/coloring benchmark"
	@echo "  make autoresearch-code   Code verification self-improvement"
	@echo ""
	@echo "Research Conductor:"
	@echo "  make research-step    Run one research step"
	@echo "  make research-loop    Continuous loop (tmux/screen)"
	@echo "  make research-dry     Show what would be done"
