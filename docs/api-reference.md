# API Reference

All public classes and methods in `carnot.pipeline`.

```python
from carnot.pipeline import (
    VerifyRepairPipeline,
    VerificationResult,
    RepairResult,
    ConstraintResult,
    ConstraintExtractor,
    AutoExtractor,
    ArithmeticExtractor,
    CodeExtractor,
    LogicExtractor,
    NLExtractor,
    CarnotError,
    ExtractionError,
    VerificationError,
    RepairError,
    ModelLoadError,
    PipelineTimeoutError,
)
```

---

## VerifyRepairPipeline

The main entry point. Extracts constraints from text, verifies them, and optionally repairs violations via an LLM.

### Constructor

```python
VerifyRepairPipeline(
    model: str | None = None,
    domains: list[str] | None = None,
    max_repairs: int = 3,
    extractor: ConstraintExtractor | None = None,
    timeout_seconds: float = 30.0,
)
```

| Parameter | Description |
|-----------|-------------|
| `model` | HuggingFace model name or path. `None` for verify-only mode. |
| `domains` | Restrict constraint extraction to these domains (e.g., `["arithmetic", "code"]`). `None` means auto-detect all. |
| `max_repairs` | Maximum LLM repair iterations. Default 3. |
| `extractor` | Custom `ConstraintExtractor` instance. Default: `AutoExtractor` (covers arithmetic, code, logic, NL). |
| `timeout_seconds` | Wall-clock timeout per verify/repair call. Default 30s. Set to 0 to disable. |

**Raises:** `ModelLoadError` if `model` is specified but cannot be loaded.

#### Examples

```python
# Verify-only (no model needed)
pipeline = VerifyRepairPipeline()

# Verify-only, restricted to arithmetic
pipeline = VerifyRepairPipeline(domains=["arithmetic"])

# Verify-and-repair with a local model
pipeline = VerifyRepairPipeline(model="Qwen/Qwen2.5-0.5B-Instruct")

# Custom extractor
from carnot.pipeline import AutoExtractor
auto = AutoExtractor()
auto.add_extractor(MyCustomExtractor())
pipeline = VerifyRepairPipeline(extractor=auto)
```

### Properties

#### `has_model -> bool`

`True` if an LLM model is loaded and available for generation/repair.

### Methods

#### `verify(question, response, domain=None) -> VerificationResult`

Extract constraints from `response` and check them.

| Parameter | Description |
|-----------|-------------|
| `question` | The original question (for context/logging). |
| `response` | The response text to verify. |
| `domain` | Optional domain hint (e.g., `"arithmetic"`, `"code"`). |

**Returns:** `VerificationResult`

**Raises:** `PipelineTimeoutError` if the call exceeds `timeout_seconds`.

```python
result = pipeline.verify("What is 5 + 3?", "5 + 3 = 8")
print(result.verified)     # True
print(result.energy)       # 0.0
print(result.violations)   # []
print(result.constraints)  # [ConstraintResult: "5 + 3 = 8"]
print(result.certificate)  # {"total_energy": 0.0, "per_constraint": [...], ...}
```

#### `verify_and_repair(question, response=None, domain=None) -> RepairResult`

Verify a response and iteratively repair violations via the LLM.

| Parameter | Description |
|-----------|-------------|
| `question` | The question to answer. |
| `response` | Initial response. If `None` and model is loaded, generates one. |
| `domain` | Optional domain hint. |

**Returns:** `RepairResult`

**Raises:**
- `ValueError` if `response` is `None` and no model is loaded.
- `PipelineTimeoutError` if the call exceeds `timeout_seconds`.

```python
# With a model: full verify-and-repair loop
pipeline = VerifyRepairPipeline(model="Qwen/Qwen2.5-0.5B-Instruct")
result = pipeline.verify_and_repair("What is 99 + 1?")
print(result.final_response)  # The (hopefully corrected) answer
print(result.repaired)        # True if the LLM fixed its mistakes
print(result.iterations)      # Number of repair attempts

# Without a model: verification only, no repair possible
pipeline = VerifyRepairPipeline()
result = pipeline.verify_and_repair("What is 99 + 1?", response="99 + 1 = 99")
print(result.verified)  # False
print(result.repaired)  # False (no model to repair with)
```

#### `extract_constraints(text, domain=None) -> list[ConstraintResult]`

Extract constraints from text without running verification.

```python
constraints = pipeline.extract_constraints("15 + 27 = 42 and 10 - 3 = 7")
for c in constraints:
    print(f"[{c.constraint_type}] {c.description}")
    # [arithmetic] 15 + 27 = 42
    # [arithmetic] 10 - 3 = 7
```

---

## VerificationResult

Returned by `verify()`. Contains the outcome of constraint checking.

| Field | Type | Description |
|-------|------|-------------|
| `verified` | `bool` | `True` if all constraints pass. |
| `constraints` | `list[ConstraintResult]` | All extracted constraints. |
| `energy` | `float` | Total weighted energy. 0.0 means all energy-backed constraints pass. |
| `violations` | `list[ConstraintResult]` | Subset of constraints that failed. |
| `certificate` | `dict` | Energy decomposition: `total_energy`, `per_constraint`, `n_constraints`, `n_violations`. |

---

## RepairResult

Returned by `verify_and_repair()`. Contains the full repair trajectory.

| Field | Type | Description |
|-------|------|-------------|
| `initial_response` | `str` | The first response (from LLM or user). |
| `final_response` | `str` | The response after all repair iterations. |
| `verified` | `bool` | `True` if `final_response` passes all checks. |
| `repaired` | `bool` | `True` if final differs from initial AND passes verification. |
| `iterations` | `int` | Number of repair iterations (0 if initially correct). |
| `history` | `list[VerificationResult]` | One `VerificationResult` per iteration. |

---

## ConstraintResult

A single extracted constraint.

| Field | Type | Description |
|-------|------|-------------|
| `constraint_type` | `str` | Category: `"arithmetic"`, `"type_check"`, `"return_type"`, `"bound"`, `"initialization"`, `"implication"`, `"factual"`, etc. |
| `description` | `str` | Human-readable summary. |
| `energy_term` | `ConstraintTerm \| None` | Optional energy term for EBM verification. |
| `metadata` | `dict` | Domain-specific details. Always includes `"satisfied"` (bool) for checkable constraints. |

### Metadata by constraint type

| Type | Key metadata fields |
|------|-------------------|
| `arithmetic` | `a`, `b`, `operator`, `claimed_result`, `correct_result`, `satisfied` |
| `type_check` | `function`, `variable`, `expected_type` |
| `return_type` | `function`, `expected_type` |
| `return_value_type` | `function`, `expected_type`, `actual_type`, `satisfied` |
| `bound` | `function`, `variable`, `lower`, `upper_expr`, `satisfied` |
| `initialization` | `function`, `variable`, `satisfied` (always `False`) |
| `implication` | `antecedent`, `consequent`, `raw` |
| `exclusion` | `positive`, `negative`, `raw` |
| `factual` | `subject`, `predicate`, `raw` |
| `factual_relation` | `subject`, `relation`, `object`, `raw` |
| `quantity` | `quantity`, `subject`, `raw` |

---

## Extractors

### ConstraintExtractor (Protocol)

The interface for all extractors. Implement this to create your own.

```python
from carnot.pipeline import ConstraintExtractor, ConstraintResult

class MyExtractor:
    @property
    def supported_domains(self) -> list[str]:
        return ["my_domain"]

    def extract(self, text: str, domain: str | None = None) -> list[ConstraintResult]:
        if domain is not None and domain not in self.supported_domains:
            return []
        # ... extract constraints from text ...
        return [
            ConstraintResult(
                constraint_type="my_check",
                description="Some check passed",
                metadata={"satisfied": True},
            )
        ]
```

### AutoExtractor

Combines multiple extractors and runs them all (or filters by domain).

```python
auto = AutoExtractor()
print(auto.supported_domains)  # ["arithmetic", "code", "logic", "nl"]

# Add a custom extractor
auto.add_extractor(MyExtractor())
print(auto.supported_domains)  # ["arithmetic", "code", "logic", "nl", "my_domain"]

# Extract from text (all domains)
constraints = auto.extract("15 + 27 = 42")

# Extract from text (specific domain)
constraints = auto.extract("15 + 27 = 42", domain="arithmetic")
```

### Built-in Extractors

| Class | Domains | What it checks |
|-------|---------|---------------|
| `ArithmeticExtractor` | `arithmetic` | `X + Y = Z` and `X - Y = Z` patterns |
| `CodeExtractor` | `code` | Type annotations, return types, undefined variables, loop bounds |
| `LogicExtractor` | `logic` | Implications, exclusions, disjunctions, negations, universals |
| `NLExtractor` | `nl` | Factual claims (`X is Y`), relations (`X is the Y of Z`), quantities |

---

## Errors

All exceptions inherit from `CarnotError`. Each carries a `details` dict with structured context.

| Exception | When it's raised |
|-----------|-----------------|
| `CarnotError` | Base class. Catch this for "anything from Carnot". |
| `ExtractionError` | Constraint extraction fails on the input. |
| `VerificationError` | Energy computation fails (e.g., JAX error). |
| `RepairError` | LLM repair loop fails unrecoverably. |
| `ModelLoadError` | HuggingFace model cannot be loaded. |
| `PipelineTimeoutError` | Operation exceeds `timeout_seconds`. |

```python
from carnot.pipeline import CarnotError, PipelineTimeoutError

try:
    result = pipeline.verify(question, response)
except PipelineTimeoutError:
    print("Verification timed out")
except CarnotError as e:
    print(f"Carnot error: {e}")
    print(f"Details: {e.details}")
```

---

## CLI Reference

### `carnot verify`

```
carnot verify FILE --func NAME --test INPUT:EXPECTED [--test ...] [--type TYPE]
                    [--properties] [--prop-samples N] [--prop-seed N]
```

| Flag | Description |
|------|-------------|
| `FILE` | Path to Python source file. |
| `--func` | Function name to verify. |
| `--test` | Test case in `input:expected` format. Repeatable. |
| `--type` | Expected return type (default: `int`). |
| `--properties` | Also run property-based tests with random inputs. |
| `--prop-samples` | Samples per property (default: 100). |
| `--prop-seed` | Random seed (default: 42). |

### `carnot score`

```
carnot score [--model MODEL_ID] [--activations-file PATH] [--list-models]
```

| Flag | Description |
|------|-------------|
| `--model` | EBM model ID (default: `per-token-ebm-qwen35-08b-nothink`). |
| `--activations-file` | Path to safetensors file with `activations` key. |
| `--list-models` | List available pre-trained EBM models. |
