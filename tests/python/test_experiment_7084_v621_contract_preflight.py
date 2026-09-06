"""Tests for REQ-REPORT-7084, the independent V621 contract preflight."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Callable

import pytest
import yaml

from carnot import experiment_7084_v621_contract_preflight as mod


ROOT = Path(__file__).resolve().parents[2]

YAML_TASK_ROWS = (
    (
        "exp7084-v621-contract-preflight",
        "V621 Markdown and YAML task-contract preflight",
        "results/experiment_7084_v621_contract_preflight.json",
    ),
    (
        "exp7085-three-family-chat-transport-canary",
        "Three-family GGUF chat-template transport canary",
        "results/experiment_7085_v621_chat_transport_canary.json",
    ),
    (
        "exp7086-chat-correct-three-family-entrance-bank",
        "Chat-correct three-family SOTA entrance bank",
        "results/experiment_7086_v621_three_family_entrance_bank.json",
    ),
    (
        "exp7087-cold-entrance-bank-sufficiency-audit",
        "Cold entrance-bank sufficiency and headroom audit",
        "results/experiment_7087_v621_entrance_bank_sufficiency_audit.json",
    ),
    (
        "exp7088-entrance-energy-strong-controls",
        "Entrance energy versus strong matched controls",
        "results/experiment_7088_v621_entrance_energy_controls.json",
    ),
    (
        "exp7089-cold-entrance-energy-abstention-audit",
        "Cold entrance-energy abstention and shift audit",
        "results/experiment_7089_v621_entrance_energy_abstention_audit.json",
    ),
    (
        "exp7090-entrance-ising-degree16-sampling-receipt",
        "Entrance QUBO, Ising, and degree-16 sampling receipt",
        "results/experiment_7090_v621_entrance_ising_degree16_receipt.json",
    ),
    (
        "exp7091-sealed-causal-entrance-stream",
        "Sealed causal entrance outcome stream",
        "results/experiment_7091_v621_causal_entrance_stream.json",
    ),
    (
        "exp7092-verifier-signed-constraint-memory-learning",
        "Verifier-signed prospective constraint-memory learning",
        "results/experiment_7092_v621_constraint_memory_learning.json",
    ),
    (
        "exp7093-constraint-memory-cold-audit",
        "Cold drift, guess-poison, and rollback audit",
        "results/experiment_7093_v621_constraint_memory_cold_audit.json",
    ),
    (
        "exp7094-live-arc-causal-single-credit-replay",
        "Live ARC causal single-credit supervisor replay",
        "results/experiment_7094_v621_arc_causal_single_credit.json",
    ),
    (
        "exp7095-v621-capstone",
        "V621 independent evidence matrix and branch disposition",
        "results/experiment_7095_v621_capstone.json",
    ),
)

MARKDOWN_CONTRACT = """# Independent V621 design fixture

**Milestone:** `2026.09.621`

## Exact Task Contract

| Order | Task ID | Title | Deliverable | Structured gates |
|---:|---|---|---|---|
| 1 | `exp7084-v621-contract-preflight` | V621 Markdown and YAML task-contract preflight | `results/experiment_7084_v621_contract_preflight.json` | None |
| 2 | `exp7085-three-family-chat-transport-canary` | Three-family GGUF chat-template transport canary | `results/experiment_7085_v621_chat_transport_canary.json` | None |
| 3 | `exp7086-chat-correct-three-family-entrance-bank` | Chat-correct three-family SOTA entrance bank | `results/experiment_7086_v621_three_family_entrance_bank.json` | `exp7085-three-family-chat-transport-canary.chat_transport_ready_score == 1` |
| 4 | `exp7087-cold-entrance-bank-sufficiency-audit` | Cold entrance-bank sufficiency and headroom audit | `results/experiment_7087_v621_entrance_bank_sufficiency_audit.json` | `exp7086-chat-correct-three-family-entrance-bank.entrance_proposal_bank_complete_score == 1` |
| 5 | `exp7088-entrance-energy-strong-controls` | Entrance energy versus strong matched controls | `results/experiment_7088_v621_entrance_energy_controls.json` | `exp7087-cold-entrance-bank-sufficiency-audit.entrance_support_audit_ready_score == 1 AND entrance_selector_headroom_ready_score == 1` |
| 6 | `exp7089-cold-entrance-energy-abstention-audit` | Cold entrance-energy abstention and shift audit | `results/experiment_7089_v621_entrance_energy_abstention_audit.json` | `exp7088-entrance-energy-strong-controls.entrance_energy_comparison_complete_score == 1` |
| 7 | `exp7090-entrance-ising-degree16-sampling-receipt` | Entrance QUBO, Ising, and degree-16 sampling receipt | `results/experiment_7090_v621_entrance_ising_degree16_receipt.json` | `exp7088-entrance-energy-strong-controls.entrance_energy_comparison_complete_score == 1` |
| 8 | `exp7091-sealed-causal-entrance-stream` | Sealed causal entrance outcome stream | `results/experiment_7091_v621_causal_entrance_stream.json` | None |
| 9 | `exp7092-verifier-signed-constraint-memory-learning` | Verifier-signed prospective constraint-memory learning | `results/experiment_7092_v621_constraint_memory_learning.json` | `exp7091-sealed-causal-entrance-stream.causal_entrance_stream_ready_score == 1` |
| 10 | `exp7093-constraint-memory-cold-audit` | Cold drift, guess-poison, and rollback audit | `results/experiment_7093_v621_constraint_memory_cold_audit.json` | `exp7092-verifier-signed-constraint-memory-learning.constraint_memory_comparison_complete_score == 1` |
| 11 | `exp7094-live-arc-causal-single-credit-replay` | Live ARC causal single-credit supervisor replay | `results/experiment_7094_v621_arc_causal_single_credit.json` | None |
| 12 | `exp7095-v621-capstone` | V621 independent evidence matrix and branch disposition | `results/experiment_7095_v621_capstone.json` | None |

### Exp7084 - V621 Markdown and YAML task-contract preflight
**Gate:** None.
**Prior failures:**
- `exp7050-v618-active-contract-preflight` — mismatch.
- `exp7076-v620-contract-preflight` — mismatch.

### Exp7085 - Three-family GGUF chat-template transport canary
**Prior failures:**
- `exp6200-three-family-raw-code-transport-canary` — incomplete.
- `exp7080-recovered-three-family-entrance-bank` — incomplete.

### Exp7086 - Chat-correct three-family SOTA entrance bank
**Gate:**
- `exp7085-three-family-chat-transport-canary.chat_transport_ready_score == 1`
**Prior failures:**
- `exp7065-three-family-entrance-proposal-bank` — blocked.
- `exp7080-recovered-three-family-entrance-bank` — incomplete.

### Exp7087 - Cold entrance-bank sufficiency and headroom audit
**Gate:**
- `exp7086-chat-correct-three-family-entrance-bank.entrance_proposal_bank_complete_score == 1`
**Prior failures:**
- `exp7066-entrance-bank-independent-audit` — blocked.
- `exp7081-entrance-bank-set-sufficiency-audit` — blocked.

### Exp7088 - Entrance energy versus strong matched controls
**Gates:**
- `exp7087-cold-entrance-bank-sufficiency-audit.entrance_support_audit_ready_score == 1`
- `exp7087-cold-entrance-bank-sufficiency-audit.entrance_selector_headroom_ready_score == 1`
**Prior failures:**
- `exp1006-energy-selection-ssd` — blocked.
- `exp7067-hopfield-entrance-energy-selection` — blocked.
- `exp7082-entrance-energy-likelihood-controls` — blocked.

### Exp7089 - Cold entrance-energy abstention and shift audit
**Gate:**
- `exp7088-entrance-energy-strong-controls.entrance_energy_comparison_complete_score == 1`
**Prior failures:**
- `exp533-cold-decoding-energy-guidance` — null.

### Exp7090 - Entrance QUBO, Ising, and degree-16 sampling receipt
**Gate:**
- `exp7088-entrance-energy-strong-controls.entrance_energy_comparison_complete_score == 1`
**Prior failures:**
- `exp7073-entrance-energy-ising-parity` — blocked.
- `exp7074-degree16-placement-sampler-audit` — blocked.
- `exp7083-entrance-ising-degree16-parity` — blocked.

### Exp7091 - Sealed causal entrance outcome stream
**Prior failures:**
- `exp7070-bcit-prospective-self-learning` — insufficient.

### Exp7092 - Verifier-signed prospective constraint-memory learning
**Gate:**
- `exp7091-sealed-causal-entrance-stream.causal_entrance_stream_ready_score == 1`
**Prior failures:**
- `exp6978-transactional-constraint-self-learning` — null.
- `exp7070-bcit-prospective-self-learning` — blocked.

### Exp7093 - Cold drift, guess-poison, and rollback audit
**Gate:**
- `exp7092-verifier-signed-constraint-memory-learning.constraint_memory_comparison_complete_score == 1`
**Prior failures:**
- `exp6979-self-learning-cold-audit` — null.
- `exp7071-bcit-drift-rollback-audit` — blocked.

### Exp7094 - Live ARC causal single-credit supervisor replay
**Prior failures:**
- `exp6524-arc-supervisor-redirect-generalization` — absent.
- `exp6921-arc-dynamic-supervisor-banked-credit` — insufficient.

### Exp7095 - V621 independent evidence matrix and branch disposition
**Prior failures:** None.
"""

GATES = {
    7086: [(7085, "chat_transport_ready_score")],
    7087: [(7086, "entrance_proposal_bank_complete_score")],
    7088: [
        (7087, "entrance_support_audit_ready_score"),
        (7087, "entrance_selector_headroom_ready_score"),
    ],
    7089: [(7088, "entrance_energy_comparison_complete_score")],
    7090: [(7088, "entrance_energy_comparison_complete_score")],
    7092: [(7091, "causal_entrance_stream_ready_score")],
    7093: [(7092, "constraint_memory_comparison_complete_score")],
}
PRODUCED_FIELDS = {
    7085: ("chat_transport_ready_score",),
    7086: ("entrance_proposal_bank_complete_score",),
    7087: ("entrance_support_audit_ready_score", "entrance_selector_headroom_ready_score"),
    7088: ("entrance_energy_comparison_complete_score",),
    7091: ("causal_entrance_stream_ready_score",),
    7092: ("constraint_memory_comparison_complete_score",),
}
PRIOR_IDS = {
    7084: ("exp7050-v618-active-contract-preflight", "exp7076-v620-contract-preflight"),
    7085: (
        "exp6200-three-family-raw-code-transport-canary",
        "exp7080-recovered-three-family-entrance-bank",
    ),
    7086: (
        "exp7065-three-family-entrance-proposal-bank",
        "exp7080-recovered-three-family-entrance-bank",
    ),
    7087: (
        "exp7066-entrance-bank-independent-audit",
        "exp7081-entrance-bank-set-sufficiency-audit",
    ),
    7088: (
        "exp1006-energy-selection-ssd",
        "exp7067-hopfield-entrance-energy-selection",
        "exp7082-entrance-energy-likelihood-controls",
    ),
    7089: ("exp533-cold-decoding-energy-guidance",),
    7090: (
        "exp7073-entrance-energy-ising-parity",
        "exp7074-degree16-placement-sampler-audit",
        "exp7083-entrance-ising-degree16-parity",
    ),
    7091: ("exp7070-bcit-prospective-self-learning",),
    7092: (
        "exp6978-transactional-constraint-self-learning",
        "exp7070-bcit-prospective-self-learning",
    ),
    7093: ("exp6979-self-learning-cold-audit", "exp7071-bcit-drift-rollback-audit"),
    7094: (
        "exp6524-arc-supervisor-redirect-generalization",
        "exp6921-arc-dynamic-supervisor-banked-credit",
    ),
}
SUBSTRATE_CLASSES = {
    7084: "aggregation",
    7085: "model_bounded_generation",
    7086: "model_full_generation",
    7087: "no_model_load",
    7088: "no_model_load",
    7089: "no_model_load",
    7090: "no_model_load",
    7091: "no_model_load",
    7092: "no_model_load",
    7093: "no_model_load",
    7094: "no_model_load",
    7095: "aggregation",
}


def _prompt(number: int) -> str:
    """Build a YAML prompt without taking contract values from Markdown."""

    substrate_class = SUBSTRATE_CLASSES[number]
    fields = [
        "field_principles",
        "preconditions_checked",
        f"inference_substrate (fixture substrate {number})",
        f"inference_substrate_class ({substrate_class}, or blocked_no_run on a blocked precondition)",
        "duration_s",
        "source_artifact_hashes",
        "rows",
        *PRODUCED_FIELDS.get(number, ()),
        "random_seed",
        "reproducibility_checksum",
        "gate_check_summary with failed check, expected value, and observed value for any blocked verdict",
        "verifier_is_oracle (false)",
        "verdict_class (positive | circular_positive | null | blocked | disqualified | partial)",
        "honest_verdict with a terminal prefix consistent with verdict_class",
    ]
    lines = ["Compare every per-unit row under matched controls."]
    if number in {7085, 7086}:
        fields.insert(-1, "MODEL_SPECS")
        lines.append("MODEL_SPECS resolves through cached_sota_pair().")
        lines.extend(f"Use {model}." for model in mod.EXPECTED_LLM_MODELS[number])
        lines.append("No legacy-small headline fallback is allowed.")
    lines.extend(
        [
            "REQUIRED ARTIFACT FIELDS: " + "; ".join(fields) + ".",
            "Run command: fixture",
            mod.PROMPT_TAIL,
        ]
    )
    return "\n".join(lines)


def _roadmap() -> dict[str, Any]:
    """Build a YAML fixture from a YAML-only literal task list."""

    tasks: list[dict[str, Any]] = []
    for task_id, title, deliverable in YAML_TASK_ROWS:
        number = int(task_id[3:7])
        local_llm = number in {7085, 7086}
        task: dict[str, Any] = {
            "id": task_id,
            "title": title,
            "track": "infrastructure",
            "priority": "critical",
            "agent_type": "claude" if number <= 7086 else "codex",
            "model": "opus" if number <= 7086 else "gpt-5.6-sol",
            "requires_gpu": local_llm,
            "max_turns": 100 if number <= 7086 else 50,
            "estimated_wall_time_min": 30,
            "per_unit_rows": True,
            "milestone": mod.MILESTONE,
            "deliverable": deliverable,
            "prompt": _prompt(number),
        }
        if number in GATES:
            task["gated_on"] = [
                {
                    "upstream": next(
                        row[0] for row in YAML_TASK_ROWS if row[0].startswith(f"exp{upstream}-")
                    ),
                    "artifact_field": field,
                    "op": "==",
                    "value": 1,
                }
                for upstream, field in GATES[number]
            ]
        if number in PRIOR_IDS:
            task["prior_failures"] = [
                {
                    "experiment_id": prior,
                    "verdict": "blocked_prior",
                    "addressed_by": "The V621 method changes the failed mechanism.",
                    "retire_if_same_verdict": True,
                }
                for prior in PRIOR_IDS[number]
            ]
        tasks.append(task)
    return {
        "milestone": mod.MILESTONE,
        "milestone_title": "fixture",
        "milestone_doc": str(mod.DESIGN_PATH),
        "tasks": tasks,
    }


def _write_preconditions(root: Path, roadmap: dict[str, Any] | None = None) -> None:
    """Create only the runtime inputs that Exp7084 is allowed to read."""

    values = {
        mod.DESIGN_PATH: MARKDOWN_CONTRACT,
        mod.ACTIVE_ROADMAP_PATH: yaml.safe_dump(roadmap or _roadmap()),
        mod.EXCLUSION_PATH: "retired: []\nretired_extras: []\n",
    }
    for relative, content in values.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def test_req_report_7084_spec_defines_exact_contract() -> None:
    """REQ-REPORT-7084 names all rows, evidence fields, and scenarios."""

    spec = (ROOT / mod.SPEC_PATH).read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-REPORT-7084") :]
    assert mod.EXPECTED_TASK_COUNT == len(YAML_TASK_ROWS) == 12
    assert mod.EXPECTED_TASK_IDS == tuple(row[0] for row in YAML_TASK_ROWS)
    for scenario in ("PARITY", "GATES", "DISCIPLINE", "PREFLIGHT", "ARTIFACT"):
        assert f"SCENARIO-REPORT-7084-{scenario}" in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in section


def test_scenario_report_7084_parity_accepts_twelve_independent_rows() -> None:
    """SCENARIO-REPORT-7084-PARITY accepts the exact independent fixtures."""

    result = mod.evaluate_contract(MARKDOWN_CONTRACT, _roadmap(), set())
    assert result["passed"] is True
    assert len(result["task_contract_rows"]) == 12
    assert len(result["gate_producer_rows"]) == 8
    assert result["markdown_task_rows"] is not result["yaml_task_rows"]
    assert all(row["passed"] for row in result["task_contract_rows"])


@pytest.mark.parametrize(
    "case", ("missing", "extra", "reordered", "title", "deliverable", "prior", "tail", "capstone")
)
def test_req_report_7084_yaml_contract_mutations_fail(case: str) -> None:
    """REQ-REPORT-7084 rejects each required active-YAML mutation."""

    roadmap = _roadmap()
    if case == "missing":
        roadmap["tasks"].pop(5)
    elif case == "extra":
        roadmap["tasks"].append(deepcopy(roadmap["tasks"][-1]))
    elif case == "reordered":
        roadmap["tasks"][5], roadmap["tasks"][6] = roadmap["tasks"][6], roadmap["tasks"][5]
    elif case == "title":
        roadmap["tasks"][0]["title"] += " changed"
    elif case == "deliverable":
        roadmap["tasks"][0]["deliverable"] = "results/wrong.json"
    elif case == "prior":
        roadmap["tasks"][0]["prior_failures"][0].pop("addressed_by")
    elif case == "tail":
        roadmap["tasks"][0]["prompt"] = roadmap["tasks"][0]["prompt"].replace(mod.PROMPT_TAIL, "")
    elif case == "capstone":
        roadmap["tasks"][-1]["gated_on"] = deepcopy(roadmap["tasks"][2]["gated_on"])
    assert mod.evaluate_contract(MARKDOWN_CONTRACT, roadmap, set())["passed"] is False, case


def test_scenario_report_7084_markdown_rows_and_gates_are_strict() -> None:
    """SCENARIO-REPORT-7084-PARITY rejects pipe corruption and malformed gates."""

    with pytest.raises(ValueError, match="task row"):
        mod.parse_markdown_contract(MARKDOWN_CONTRACT.replace("| 1 |", "| 1 | extra |", 1))
    with pytest.raises(ValueError, match="structured gate"):
        mod.parse_markdown_contract(MARKDOWN_CONTRACT.replace(" == 1`", " is ready`", 1))
    roadmap = _roadmap()
    roadmap["tasks"][2]["gated_on"][0]["op"] = None
    assert mod.evaluate_contract(MARKDOWN_CONTRACT, roadmap, set())["passed"] is False


def test_scenario_report_7084_discipline_rejects_each_defect() -> None:
    """SCENARIO-REPORT-7084-DISCIPLINE fails closed across task policy checks."""

    mutations: tuple[Callable[[dict[str, Any]], None], ...] = (
        lambda value: value["tasks"][0].__setitem__("per_unit_rows", False),
        lambda value: value["tasks"][1].__setitem__("model", "wrong"),
        lambda value: value["tasks"][1].__setitem__("requires_gpu", False),
        lambda value: value["tasks"][1].__setitem__(
            "prompt", value["tasks"][1]["prompt"].replace("MODEL_SPECS", "models")
        ),
        lambda value: value["tasks"][1].__setitem__(
            "prompt", value["tasks"][1]["prompt"].replace("cached_sota_pair()", "direct lookup")
        ),
        lambda value: value["tasks"][1].__setitem__(
            "prompt",
            value["tasks"][1]["prompt"].replace("unsloth/Qwen3.6-35B-A3B-GGUF", "legacy/model"),
        ),
        lambda value: value["tasks"][1].__setitem__(
            "prompt",
            value["tasks"][1]["prompt"].replace(
                "No legacy-small headline fallback is allowed.", "Fallback is unspecified."
            ),
        ),
        lambda value: value["tasks"][0].__setitem__(
            "prompt",
            value["tasks"][0]["prompt"].replace(
                "aggregation, or blocked_no_run", "unknown, or blocked_no_run"
            ),
        ),
        lambda value: value["tasks"][0].__setitem__(
            "prompt", value["tasks"][0]["prompt"].replace(" | partial", " | deferred")
        ),
        lambda value: value["tasks"][0].__setitem__(
            "prompt", value["tasks"][0]["prompt"].replace("observed value", "observation")
        ),
        lambda value: value["tasks"][3]["gated_on"][0].__setitem__(
            "artifact_field", "nested.value"
        ),
        lambda value: value["tasks"][3]["gated_on"][0].__setitem__(
            "upstream", value["tasks"][-1]["id"]
        ),
    )
    for mutate in mutations:
        roadmap = _roadmap()
        mutate(roadmap)
        assert mod.evaluate_contract(MARKDOWN_CONTRACT, roadmap, set())["passed"] is False
    assert mod.evaluate_contract(MARKDOWN_CONTRACT, _roadmap(), {"exp7088"})["passed"] is False


def test_scenario_report_7084_prior_failures_require_all_four_fields() -> None:
    """SCENARIO-REPORT-7084-DISCIPLINE rejects incomplete prior records."""

    for field, value in (
        ("experiment_id", ""),
        ("verdict", ""),
        ("addressed_by", ""),
        ("retire_if_same_verdict", False),
    ):
        roadmap = _roadmap()
        roadmap["tasks"][0]["prior_failures"][0][field] = value
        assert mod.evaluate_contract(MARKDOWN_CONTRACT, roadmap, set())["passed"] is False
    roadmap = _roadmap()
    roadmap["tasks"][0]["prior_failures"] = []
    assert mod.evaluate_contract(MARKDOWN_CONTRACT, roadmap, set())["passed"] is False


def test_scenario_report_7084_preflight_separates_terminal_classes(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7084-PREFLIGHT separates positive, mismatch, and no-run."""

    _write_preconditions(tmp_path)
    positive = mod.build_artifact(tmp_path, "20260906", output_path=tmp_path / "positive.json")
    assert positive["verdict_class"] == "positive"
    assert positive["inference_substrate_class"] == "aggregation"
    assert mod.validate_artifact(positive) == []

    changed = _roadmap()
    changed["tasks"].pop()
    _write_preconditions(tmp_path, changed)
    disqualified = mod.build_artifact(tmp_path, "20260906", output_path=tmp_path / "bad.json")
    assert disqualified["verdict_class"] == "disqualified"
    assert disqualified["v621_task_contract_conforms_score"] == 0
    assert mod.validate_artifact(disqualified) == []

    (tmp_path / mod.DESIGN_PATH).unlink()
    blocked = mod.build_artifact(tmp_path, "20260906", output_path=tmp_path / "blocked.json")
    assert blocked["verdict_class"] == "blocked"
    assert blocked["inference_substrate_class"] == "blocked_no_run"
    assert blocked["gate_check_summary"]["failed_check"] == "v621_markdown_readable"
    assert "FileNotFoundError" in blocked["gate_check_summary"]["observed_value"]
    assert mod.validate_artifact(blocked) == []


def test_scenario_report_7084_artifact_validator_rejects_forgery(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7084-ARTIFACT recomputes evidence and checksum."""

    _write_preconditions(tmp_path)
    good = mod.build_artifact(tmp_path, "20260906", output_path=tmp_path / "good.json")
    cases = (
        ("inference_substrate_class", "blocked_no_run", "inference_substrate_class_invalid"),
        ("v621_task_contract_conforms_score", 0, "v621_task_contract_conforms_score_invalid"),
        ("verdict_class", "partial", "verdict_class_not_derived"),
        ("honest_verdict", "complete_wrong", "honest_verdict_not_derived"),
        ("gate_check_summary", {}, "gate_check_summary_invalid"),
    )
    for field, value, error in cases:
        changed = deepcopy(good)
        changed[field] = value
        changed["reproducibility_checksum"] = mod.reproducibility_checksum(changed)
        assert error in mod.validate_artifact(changed)
    changed = deepcopy(good)
    changed["task_contract_rows"][0]["passed"] = False
    changed["reproducibility_checksum"] = mod.reproducibility_checksum(changed)
    assert "v621_task_contract_conforms_score_invalid" in mod.validate_artifact(changed)
    changed = deepcopy(good)
    changed["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum_invalid" in mod.validate_artifact(changed)


def test_req_report_7084_cli_and_parse_error_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7084 exposes deterministic generation and validation paths."""

    with pytest.raises(Exception, match="YYYYMMDD"):
        mod._date_argument("2026-09-06")
    with pytest.raises(SystemExit):
        mod.main([])
    assert mod.main(["--validate-artifact", str(tmp_path / "missing.json")]) == 1
    _write_preconditions(tmp_path)
    output = Path("results/result.json")
    assert mod.main(["--date", "20260906", "--root", str(tmp_path), "--output", str(output)]) == 0
    artifact_path = tmp_path / output
    assert json.loads(artifact_path.read_text(encoding="utf-8"))["verdict_class"] == "positive"
    assert mod.main(["--validate-artifact", str(artifact_path)]) == 0
    (tmp_path / mod.DESIGN_PATH).write_text("readable but malformed\n", encoding="utf-8")
    bad = mod.build_artifact(tmp_path, "20260906", output_path=tmp_path / "malformed.json")
    assert bad["verdict_class"] == "disqualified"
    assert bad["gate_check_summary"]["failed_check"] == "contract_parse"
    assert mod.validate_artifact(bad) == []
    monkeypatch.setattr(
        mod.tempfile,
        "NamedTemporaryFile",
        lambda **_kwargs: (_ for _ in ()).throw(OSError("read only")),
    )
    assert mod._artifact_path_writable(tmp_path / "blocked.json") == (False, "OSError: read only")


def test_req_report_7084_preconditions_and_failure_summaries_are_exact(tmp_path: Path) -> None:
    """REQ-REPORT-7084 preserves each missing input and mismatch source."""

    rows, roadmap, exclusion = mod._preconditions(tmp_path, tmp_path / "result.json")
    assert roadmap is exclusion is None
    assert {row["check"] for row in rows if not row["available"]} >= {
        "v621_markdown_readable",
        "v621_active_yaml_readable",
        "exclusion_manifest_readable",
    }
    assert (
        mod._contract_failure_summary(
            {"markdown_task_count": 11, "yaml_task_rows": [], "task_contract_rows": []}
        )["failed_check"]
        == "markdown_task_count"
    )
    assert (
        mod._contract_failure_summary(
            {"markdown_task_count": 12, "yaml_task_rows": [], "task_contract_rows": []}
        )["failed_check"]
        == "yaml_task_count"
    )
    assert (
        mod._contract_failure_summary(
            {
                "markdown_task_count": 12,
                "yaml_task_rows": [{}] * 12,
                "task_contract_rows": [{"order": 4, "passed": False}],
            }
        )["failed_check"]
        == "task_contract_order_4"
    )


def test_req_report_7084_score_rejects_each_evidence_shape(tmp_path: Path) -> None:
    """REQ-REPORT-7084 recomputes conformance from every evidence table."""

    _write_preconditions(tmp_path)
    good = mod.build_artifact(tmp_path, "20260906", output_path=tmp_path / "good.json")
    mutations: tuple[Callable[[dict[str, Any]], None], ...] = (
        lambda value: value["markdown_task_rows"][0].__setitem__("id", "exp9999-wrong"),
        lambda value: value["yaml_task_rows"][0].__setitem__("id", "exp9999-wrong"),
        lambda value: value["task_contract_rows"][0].__setitem__("order", 99),
        lambda value: value.__setitem__("title_parity_rows", []),
        lambda value: value["title_parity_rows"][0].__setitem__("passed", False),
        lambda value: value["gate_producer_rows"].pop(),
        lambda value: value.__setitem__("prior_failure_rows", value["prior_failure_rows"][:11]),
    )
    for mutate in mutations:
        changed = deepcopy(good)
        mutate(changed)
        assert mod._contract_score_from_artifact(changed) == 0


def test_req_report_7084_validator_covers_each_headline(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7084-ARTIFACT rejects every forged headline field."""

    _write_preconditions(tmp_path)
    good = mod.build_artifact(tmp_path, "20260906", output_path=tmp_path / "good.json")
    cases = (
        ("field_principles", {}, "field_principles_invalid"),
        ("inference_substrate", "wrong", "inference_substrate_invalid"),
        ("verifier_is_oracle", True, "verifier_is_oracle_invalid"),
        ("expected_task_count", 11, "expected_task_count_invalid"),
        ("expected_id_order", [], "expected_id_order_invalid"),
        ("active_roadmap_path", "research-roadmap-next.yaml", "active_roadmap_path_invalid"),
        ("staging_file_required_at_execution", True, "staging_file_required_invalid"),
        ("random_seed", 0, "random_seed_invalid"),
        ("duration_s", -1, "duration_s_invalid"),
        ("observed_task_count", 0, "observed_task_count_invalid"),
        ("observed_id_order", [], "observed_id_order_invalid"),
        ("rows", [], "rows_not_task_contract_rows"),
    )
    for field, value, error in cases:
        changed = deepcopy(good)
        changed[field] = value
        changed["reproducibility_checksum"] = mod.reproducibility_checksum(changed)
        assert error in mod.validate_artifact(changed), field
    missing = deepcopy(good)
    missing.pop("rows")
    assert mod.validate_artifact(missing) == ["missing_required_field:rows"]


def test_req_report_7084_validator_derives_each_mismatch_summary(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7084-ARTIFACT derives count and row diagnostics."""

    _write_preconditions(tmp_path)
    good = mod.build_artifact(tmp_path, "20260906", output_path=tmp_path / "good.json")

    markdown_short = deepcopy(good)
    markdown_short["markdown_task_rows"].pop()
    markdown_short["v621_task_contract_conforms_score"] = 0
    markdown_short["verdict_class"] = "disqualified"
    markdown_short["honest_verdict"] = mod.DISQUALIFIED_VERDICT
    markdown_short["gate_check_summary"] = {
        "failed_check": "markdown_task_count",
        "expected_value": 12,
        "observed_value": 11,
        "passed": False,
    }
    markdown_short["reproducibility_checksum"] = mod.reproducibility_checksum(markdown_short)
    assert mod.validate_artifact(markdown_short) == []

    row_failure = deepcopy(good)
    row_failure["task_contract_rows"][0]["passed"] = False
    row_failure["rows"][0]["passed"] = False
    row_failure["title_parity_rows"][0]["passed"] = False
    row_failure["v621_task_contract_conforms_score"] = 0
    row_failure["verdict_class"] = "disqualified"
    row_failure["honest_verdict"] = mod.DISQUALIFIED_VERDICT
    row_failure["gate_check_summary"] = {
        "failed_check": "task_contract_order_1",
        "expected_value": True,
        "observed_value": False,
        "passed": False,
    }
    row_failure["reproducibility_checksum"] = mod.reproducibility_checksum(row_failure)
    assert mod.validate_artifact(row_failure) == []


def test_req_report_7084_cli_rejects_invalid_artifacts_and_generated_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7084 CLI returns failure for invalid stored or generated data."""

    bad = tmp_path / "bad.json"
    bad.write_text("{}\n", encoding="utf-8")
    assert mod.main(["--validate-artifact", str(bad)]) == 1
    listed = tmp_path / "list.json"
    listed.write_text("[]\n", encoding="utf-8")
    assert mod.main(["--validate-artifact", str(listed)]) == 1
    _write_preconditions(tmp_path)
    monkeypatch.setattr(mod, "validate_artifact", lambda _artifact: ["forced failure"])
    assert (
        mod.main(["--date", "20260906", "--root", str(tmp_path), "--output", "invalid.json"]) == 1
    )
