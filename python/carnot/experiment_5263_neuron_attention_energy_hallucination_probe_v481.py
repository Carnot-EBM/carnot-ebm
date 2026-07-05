"""Exp 5263: bounded neuron/attention/logit hallucination-energy probe.

Spec refs: REQ-VERIFY-5263, SCENARIO-VERIFY-5263.

This experiment is a receipt gate before any larger hallucination verifier
work. The local GGUF runtime may expose generated text, token logprobs, final
logits, attention tensors, or hidden states. Only the latter four are useful
for this task. If the runtime is text-only, the honest result is a blocked
capability artifact, not a replacement text scorer.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import inspect
import json
import math
from pathlib import Path
import time
from typing import Any


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5263_neuron_attention_energy_hallucination_probe_v481.json")
EXP5259_RELATIVE_PATH = Path("results/experiment_5259_sota_gguf_gpu_offload_preflight_v481.json")
SCHEMA = "carnot.experiment_5263.neuron_attention_energy_hallucination_probe.v481"
SPEC_REFS = ("REQ-VERIFY-5263", "SCENARIO-VERIFY-5263")
LIVE_SUBSTRATE = "live_llm_inference_local_gguf_sota"
PREFLIGHT_SUBSTRATE = "llama_cpp_runtime_preflight_no_quality_claim"
RANDOM_SEED = 5263
TERMINAL_PREFIXES = ("complete:", "blocked_")
SIGNAL_KEYS = ("hidden_states", "attention_tensors", "logits", "token_logprobs")

GGUF_PROBE_CONFIG: JsonDict = {
    "n_gpu_layers": -1,
    "n_ctx": 512,
    "max_tokens": 4,
    "temperature": 0.0,
    "seed": RANDOM_SEED,
    "logprobs": 5,
    "logits_all": True,
}

FIELD_PRINCIPLES: dict[str, str] = {
    "honest_verdict": (
        "Terminal Exp 5263 verdict; starts with complete: or blocked_ and states "
        "whether the available internal/logit/attention signal was useful, null, "
        "harmful, or unavailable."
    ),
    "inference_substrate": (
        "Declares live local SOTA GGUF inference only when internal/logit receipts "
        "exist; otherwise remains a llama.cpp runtime preflight with no quality claim."
    ),
    "preconditions_checked": (
        "Records Exp 5259 readiness, llama.cpp signal surface, model/runtime receipts, "
        "fixture label adequacy, and retired Phase D exclusion before scoring."
    ),
    "MODEL_SPECS": (
        "Records mandated SOTA GGUF model IDs, roles, quantization/file receipts, "
        "runtime status, and selected pilot model."
    ),
    "internal_signal_available": (
        "Bare boolean; true only when hidden states, attention tensors, logits, or "
        "token logprobs are exposed with receipts beyond generated text."
    ),
    "internal_signal_available_principle": (
        "Explains which signal classes were available and whether the runtime was text-only."
    ),
    "hidden_energy_probe_signal_delta": (
        "Unsupported-minus-supported mean of the pre-registered available energy score; "
        "zero only for blocked or no-signal artifacts."
    ),
    "false_accepts_at_threshold": (
        "Count of unsupported fixtures accepted as safe at the pre-registered zero "
        "false-accept budget threshold."
    ),
    "external_text_scorer_used": (
        "Must be false; Phase D external generated-text/logprob scorers are retired "
        "and not reopened."
    ),
    "fixture_checksums": (
        "Records fixture prompt checksums, label checksum, and fixture-set checksum for "
        "the bounded supported/unsupported panel."
    ),
    "commands_run": "Commands run to create and validate the artifact, with outcomes.",
}

MANDATED_MODEL_SPECS: tuple[JsonDict, ...] = (
    {
        "slot": "flagship_moe",
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "role": "flagship_moe",
        "quantization": "Q4_K_M",
    },
    {
        "slot": "flagship_dense",
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "role": "flagship_dense",
        "quantization": "Q4_K_M",
    },
    {
        "slot": "middle_moe",
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "role": "optional_middle_moe",
        "quantization": "Q4_K_M",
    },
)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "preconditions_checked",
    "MODEL_SPECS",
    "internal_signal_available",
    "internal_signal_available_principle",
    "hidden_energy_probe_signal_delta",
    "hidden_energy_probe_signal_delta_principle",
    "false_accepts_at_threshold",
    "external_text_scorer_used",
    "fixture_checksums",
    "commands_run",
)
WRAPPED_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "preconditions_checked",
    "MODEL_SPECS",
    "false_accepts_at_threshold",
    "external_text_scorer_used",
    "fixture_checksums",
)


@dataclass(frozen=True)
class HallucinationFixture:
    """One local evidence-relative claim with a model-independent label."""

    fixture_id: str
    evidence: str
    claim: str
    unsupported_label: bool
    label_source: str = "curated_local_evidence_label"
    category: str = "bounded_factual_claim"


GenerationRunner = Callable[[HallucinationFixture, JsonDict, int], JsonDict]


def default_fixtures() -> list[HallucinationFixture]:
    """Return a tiny balanced panel with supported and unsupported claims."""

    return [
        HallucinationFixture(
            fixture_id="hae-001-supported-runtime",
            evidence="Larkspur audit memo: the Aster-9 battery ran 47 minutes under the amber-load test.",
            claim="The Aster-9 battery ran 47 minutes under the amber-load test.",
            unsupported_label=False,
        ),
        HallucinationFixture(
            fixture_id="hae-002-unsupported-runtime",
            evidence="Larkspur audit memo: the Aster-9 battery ran 47 minutes under the amber-load test.",
            claim="The Aster-9 battery ran 74 minutes under the amber-load test.",
            unsupported_label=True,
        ),
        HallucinationFixture(
            fixture_id="hae-003-supported-enrollment",
            evidence="Mira Vale clinic snippet: Trial Nacre enrolled 18 participants and used a saline control.",
            claim="Trial Nacre enrolled 18 participants.",
            unsupported_label=False,
        ),
        HallucinationFixture(
            fixture_id="hae-004-unsupported-manufacturer",
            evidence="Solenne lab note: reagent Pavo turned violet at pH 5.2.",
            claim="Reagent Pavo was manufactured by HelioChem.",
            unsupported_label=True,
        ),
        HallucinationFixture(
            fixture_id="hae-005-supported-route",
            evidence="Noma transit note: Route 6 skipped Pear Gate during the lantern parade.",
            claim="Route 6 skipped Pear Gate during the lantern parade.",
            unsupported_label=False,
        ),
        HallucinationFixture(
            fixture_id="hae-006-unsupported-year",
            evidence="Tavian school board note: the Cedar lab opened in 2024 with six benches.",
            claim="The Cedar lab opened in 2023.",
            unsupported_label=True,
        ),
    ]


def render_prompt(fixture: HallucinationFixture) -> str:
    """Render the local verdict prompt without exposing the fixture label."""

    return (
        "Use only the evidence below. Reply with exactly SUPPORTED or UNSUPPORTED.\n\n"
        f"Evidence:\n{fixture.evidence}\n\n"
        f"Claim:\n{fixture.claim}\n\n"
        "Verdict:"
    )


def fixture_checksums(fixtures: Sequence[HallucinationFixture]) -> JsonDict:
    """Return stable checksums for prompts and labels used by the pilot."""

    prompt_checksums = {fixture.fixture_id: sha16(render_prompt(fixture)) for fixture in fixtures}
    labels = {fixture.fixture_id: bool(fixture.unsupported_label) for fixture in fixtures}
    payload = [
        {
            "fixture_id": fixture.fixture_id,
            "evidence_sha16": sha16(fixture.evidence),
            "claim_sha16": sha16(fixture.claim),
            "unsupported_label": fixture.unsupported_label,
            "prompt_sha16": prompt_checksums[fixture.fixture_id],
        }
        for fixture in fixtures
    ]
    return {
        "fixture_count": len(fixtures),
        "prompt_checksums": prompt_checksums,
        "label_checksum": sha16(_stable_json(labels)),
        "fixture_set_checksum": sha16(_stable_json(payload)),
        "label_source": "curated_local_evidence_label",
        "supported_count": sum(1 for fixture in fixtures if not fixture.unsupported_label),
        "unsupported_count": sum(1 for fixture in fixtures if fixture.unsupported_label),
    }


def compute_energy_features(generation: Mapping[str, Any]) -> JsonDict:
    """Compute the pre-registered energy features from available receipts."""

    token_logprobs = _numeric_values(generation.get("token_logprobs"))
    top_rows = _top_logprob_rows(generation.get("top_logprobs"))
    final_logits = _final_logits(generation)
    full_logit_summary = _full_logit_summary(final_logits) if final_logits else _summary_from_generation(generation)

    top_distribution = _softmax_log_values(list(top_rows[-1].values())) if top_rows else []
    selected_final_prob = _selected_final_probability(token_logprobs)
    if top_distribution:
        final_top1_probability = max(top_distribution)
    else:
        final_top1_probability = selected_final_prob

    sequence_spilled = _mean([1.0 - math.exp(min(0.0, value)) for value in token_logprobs])
    sequence_marginal = -_mean(token_logprobs) if token_logprobs else None
    final_spilled = 1.0 - final_top1_probability if final_top1_probability is not None else None
    final_marginal = -math.log(max(final_spilled, 1e-12)) if final_spilled is not None else None
    full_logit_top1 = _optional_float(full_logit_summary.get("top1_probability"))
    full_logit_spilled = 1.0 - full_logit_top1 if full_logit_top1 is not None else None
    primary = sequence_marginal if sequence_marginal is not None else full_logit_spilled
    if primary is None:
        primary = final_spilled

    signal_available = bool(token_logprobs or top_rows or final_logits or full_logit_summary)
    return {
        "signal_available": signal_available,
        "token_count": _token_count(generation, token_logprobs),
        "token_logprob_count": len(token_logprobs),
        "top_logprobs_count": len(top_rows),
        "final_logits_count": len(final_logits),
        "final_token_top1_probability": final_top1_probability,
        "final_token_spilled_energy": final_spilled,
        "final_token_marginal_energy": final_marginal,
        "sequence_spilled_energy": sequence_spilled,
        "sequence_marginal_energy": sequence_marginal,
        "full_logit_top1_probability": full_logit_top1,
        "full_logit_spilled_energy": full_logit_spilled,
        "full_logit_entropy_topk": full_logit_summary.get("entropy_topk"),
        "primary_energy": primary,
        "primary_energy_name": (
            "sequence_marginal_energy"
            if sequence_marginal is not None
            else "full_logit_spilled_energy"
            if full_logit_spilled is not None
            else "final_token_spilled_energy"
        ),
    }


def summarize_energy(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Summarize the fixed energy score without fitting a detector."""

    pairs: list[tuple[int, float]] = []
    for row in rows:
        energy = _optional_float(_nested_value(row.get("energy_features", {}), "primary_energy"))
        if energy is None:
            continue
        pairs.append((1 if bool(row.get("unsupported_label")) else 0, energy))

    supported = [score for label, score in pairs if label == 0]
    unsupported = [score for label, score in pairs if label == 1]
    threshold = _zero_false_accept_threshold(unsupported)
    false_accepts = sum(1 for label, score in pairs if label == 1 and threshold is not None and score <= threshold)
    flagged = [(label, score) for label, score in pairs if threshold is not None and score > threshold]
    true_flags = sum(1 for label, _score in flagged if label == 1)
    precision = true_flags / len(flagged) if flagged else None
    signal_delta = (_mean(unsupported) - _mean(supported)) if supported and unsupported else 0.0
    return {
        "n_scored": len(pairs),
        "label_counts": {"supported": len(supported), "unsupported": len(unsupported)},
        "supported_mean_energy": _mean(supported),
        "unsupported_mean_energy": _mean(unsupported),
        "signal_delta": float(signal_delta),
        "auroc": _auroc([label for label, _score in pairs], [score for _label, score in pairs]),
        "threshold_policy": "zero_false_accept_budget_on_unsupported_labels",
        "threshold": threshold,
        "false_accepts_at_threshold": int(false_accepts),
        "precision_at_threshold": precision,
        "detector_trained": False,
    }


def deterministic_baselines(fixtures: Sequence[HallucinationFixture]) -> JsonDict:
    """Compute claim/evidence-only baselines for context, not as model scorers."""

    total = len(fixtures)
    unsupported = sum(1 for fixture in fixtures if fixture.unsupported_label)
    lexical_predictions = {
        fixture.fixture_id: not _claim_supported_by_evidence(fixture.claim, fixture.evidence)
        for fixture in fixtures
    }
    lexical_correct = sum(
        1 for fixture in fixtures if lexical_predictions[fixture.fixture_id] == fixture.unsupported_label
    )
    lexical_false_accepts = sum(
        1 for fixture in fixtures if fixture.unsupported_label and not lexical_predictions[fixture.fixture_id]
    )
    return {
        "always_supported": {
            "accuracy": (total - unsupported) / total if total else 0.0,
            "false_accepts": unsupported,
        },
        "lexical_claim_terms": {
            "accuracy": lexical_correct / total if total else 0.0,
            "false_accepts": lexical_false_accepts,
            "uses_model_output": False,
        },
    }


def run_pilot(
    *,
    result_path: Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    preflight_artifact: Mapping[str, Any] | None = None,
    signal_surface: Mapping[str, Any] | None = None,
    generation_runner: GenerationRunner | None = None,
    fixtures: Sequence[HallucinationFixture] | None = None,
    commands_run: Sequence[Mapping[str, Any]] = (),
    root: Path = REPO_ROOT,
    write: bool = True,
) -> JsonDict:
    """Gate runtime signals, run the bounded pilot when possible, and write JSON."""

    started = time.perf_counter()
    root = Path(root)
    active_preflight = dict(preflight_artifact or load_preflight_artifact(root))
    active_surface = dict(signal_surface or inspect_llama_cpp_signal_surface())
    active_fixtures = list(fixtures or default_fixtures())
    model_specs = _model_specs_from_preflight(active_preflight)
    selected_model = _select_pilot_model(model_specs)

    if not active_preflight.get("sota_runtime_ready") or selected_model is None:
        artifact = _blocked_artifact(
            root=root,
            preflight_artifact=active_preflight,
            signal_surface=active_surface,
            model_specs=model_specs,
            fixtures=active_fixtures,
            commands_run=commands_run,
            blocker="blocked_sota_runtime_unavailable",
            live_signal_receipts_found=False,
            duration_s=time.perf_counter() - started,
        )
        validate_artifact(artifact)
        _write_json_if_requested(result_path, artifact, write)
        return artifact

    if not _surface_has_internal_signal(active_surface):
        artifact = _blocked_artifact(
            root=root,
            preflight_artifact=active_preflight,
            signal_surface=active_surface,
            model_specs=model_specs,
            fixtures=active_fixtures,
            commands_run=commands_run,
            blocker="blocked_internal_signal_unavailable",
            live_signal_receipts_found=False,
            duration_s=time.perf_counter() - started,
        )
        validate_artifact(artifact)
        _write_json_if_requested(result_path, artifact, write)
        return artifact

    runner = generation_runner or live_llama_cpp_generation_runner(active_preflight, active_surface)
    pilot_rows = [
        _evaluate_fixture(fixture, selected_model, runner, seed=RANDOM_SEED + index)
        for index, fixture in enumerate(active_fixtures)
    ]
    live_signal_receipts_found = any(row["energy_features"]["signal_available"] for row in pilot_rows)
    label_counts = fixture_checksums(active_fixtures)
    labels_adequate = bool(label_counts["supported_count"] and label_counts["unsupported_count"])
    if not live_signal_receipts_found or not labels_adequate:
        artifact = _blocked_artifact(
            root=root,
            preflight_artifact=active_preflight,
            signal_surface=active_surface,
            model_specs=model_specs,
            fixtures=active_fixtures,
            commands_run=commands_run,
            blocker="blocked_internal_signal_unavailable",
            live_signal_receipts_found=live_signal_receipts_found,
            duration_s=time.perf_counter() - started,
        )
        validate_artifact(artifact)
        _write_json_if_requested(result_path, artifact, write)
        return artifact

    summary = summarize_energy(pilot_rows)
    artifact = _complete_artifact(
        root=root,
        preflight_artifact=active_preflight,
        signal_surface=active_surface,
        model_specs=model_specs,
        fixtures=active_fixtures,
        pilot_rows=pilot_rows,
        separation_summary=summary,
        commands_run=commands_run,
        duration_s=time.perf_counter() - started,
    )
    validate_artifact(artifact)
    _write_json_if_requested(result_path, artifact, write)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when an Exp 5263 artifact violates the required schema."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact, f"missing required field {field}"
    for field in WRAPPED_FIELDS:
        value = artifact.get(field)
        assert isinstance(value, Mapping), f"{field} must be principle-wrapped"
        assert "value" in value and "principle" in value, f"{field} must be principle-wrapped"
        assert value["principle"] == FIELD_PRINCIPLES[field], f"{field} principle mismatch"

    verdict = artifact["honest_verdict"]["value"]
    assert isinstance(verdict, str) and verdict.startswith(TERMINAL_PREFIXES), (
        "honest_verdict.value must start with complete: or blocked_"
    )
    assert any(word in verdict for word in ("signal", "null", "harmful", "unavailable")), (
        "honest_verdict.value must state signal, null, harmful, or unavailable"
    )
    substrate = artifact["inference_substrate"]["value"]
    assert substrate in (LIVE_SUBSTRATE, PREFLIGHT_SUBSTRATE), "inference_substrate.value invalid"
    assert isinstance(artifact["internal_signal_available"], bool), (
        "internal_signal_available must be a bare bool"
    )
    delta = artifact["hidden_energy_probe_signal_delta"]
    assert isinstance(delta, int | float) and not isinstance(delta, bool), (
        "hidden_energy_probe_signal_delta must be a bare float"
    )
    assert artifact["hidden_energy_probe_signal_delta_principle"] == FIELD_PRINCIPLES[
        "hidden_energy_probe_signal_delta"
    ]
    assert isinstance(artifact["false_accepts_at_threshold"]["value"], int), (
        "false_accepts_at_threshold.value must be int"
    )
    assert artifact["external_text_scorer_used"]["value"] is False, (
        "external_text_scorer_used.value must remain false"
    )
    assert isinstance(artifact["fixture_checksums"]["value"], Mapping), (
        "fixture_checksums.value must be object"
    )
    assert isinstance(artifact["MODEL_SPECS"]["value"], Mapping), "MODEL_SPECS.value must be object"
    assert isinstance(artifact["commands_run"], list), "commands_run must be a list"


def inspect_llama_cpp_signal_surface() -> JsonDict:  # pragma: no cover - unit tests inject this.
    """Inspect the installed llama.cpp Python API for non-text signal access."""

    surface: JsonDict = {
        "hidden_states": False,
        "attention_tensors": False,
        "logits": False,
        "token_logprobs": False,
        "generated_text": False,
        "api_receipts": {},
    }
    try:
        from llama_cpp import Llama  # noqa: PLC0415
    except Exception as exc:
        surface["api_receipts"]["import_error"] = f"{type(exc).__name__}: {exc}"
        return surface

    init_sig = inspect.signature(Llama.__init__)
    call_sig = inspect.signature(Llama.__call__)
    init_params = set(init_sig.parameters)
    call_params = set(call_sig.parameters)
    attrs = set(dir(Llama))
    surface.update(
        {
            "hidden_states": "output_hidden_states" in init_params or any("hidden" in attr.lower() for attr in attrs),
            "attention_tensors": "output_attentions" in call_params or "attention_scores" in attrs,
            "logits": "logits_all" in init_params and hasattr(Llama, "eval_logits"),
            "token_logprobs": "logprobs" in call_params,
            "generated_text": callable(getattr(Llama, "__call__", None)),
            "api_receipts": {
                "Llama.__init__": str(init_sig),
                "Llama.__call__": str(call_sig),
                "eval_logits_property": hasattr(Llama, "eval_logits"),
                "logits_all_parameter": "logits_all" in init_params,
                "logprobs_parameter": "logprobs" in call_params,
            },
        }
    )
    return surface


def live_llama_cpp_generation_runner(
    preflight_artifact: Mapping[str, Any],
    signal_surface: Mapping[str, Any],
) -> GenerationRunner:  # pragma: no cover - exercised by artifact generation, not unit tests.
    """Build one live local GGUF runner that returns logprob/logit receipts."""

    model_specs = _model_specs_from_preflight(preflight_artifact)
    selected = _select_pilot_model(model_specs)
    if selected is None:
        raise RuntimeError("no ready mandated GGUF model receipt available")
    model_path = selected["file_receipts"].get("path")
    if not model_path:
        raise RuntimeError("selected GGUF model path unavailable")

    from llama_cpp import Llama  # noqa: PLC0415

    llm = Llama(
        model_path=str(model_path),
        n_gpu_layers=int(GGUF_PROBE_CONFIG["n_gpu_layers"]),
        n_ctx=int(GGUF_PROBE_CONFIG["n_ctx"]),
        seed=int(GGUF_PROBE_CONFIG["seed"]),
        logits_all=bool(signal_surface.get("logits")),
        verbose=False,
    )

    def run(fixture: HallucinationFixture, model_spec: JsonDict, seed: int) -> JsonDict:
        del model_spec
        response = llm(
            render_prompt(fixture),
            max_tokens=int(GGUF_PROBE_CONFIG["max_tokens"]),
            temperature=float(GGUF_PROBE_CONFIG["temperature"]),
            logprobs=int(GGUF_PROBE_CONFIG["logprobs"]) if signal_surface.get("token_logprobs") else None,
            echo=False,
            seed=int(seed),
        )
        choice = _first_choice(response)
        logprobs = choice.get("logprobs") if isinstance(choice, Mapping) else {}
        token_logprobs = _numeric_values(_nested_value(logprobs, "token_logprobs"))
        top_logprobs = _top_logprob_rows(_nested_value(logprobs, "top_logprobs"))
        logits_summary: JsonDict = {}
        if signal_surface.get("logits"):
            logits = getattr(llm, "eval_logits", None)
            if logits:
                final_logits = logits[-1]
                logits_summary = _full_logit_summary(final_logits)
                logits_summary["steps"] = len(logits)
        return {
            "raw_response": str(choice.get("text", "")) if isinstance(choice, Mapping) else str(response),
            "token_logprobs": token_logprobs,
            "top_logprobs": top_logprobs,
            "tokens": list(logprobs.get("tokens", [])) if isinstance(logprobs, Mapping) else [],
            "token_count": len(token_logprobs),
            "final_logits_summary": logits_summary,
            "logit_receipt": logits_summary,
            "seed": int(seed),
        }

    return run


def load_preflight_artifact(root: Path = REPO_ROOT) -> JsonDict:  # pragma: no cover
    """Load Exp 5259 readiness receipts from the checked-in result artifact."""

    path = root / EXP5259_RELATIVE_PATH
    if not path.exists():
        return {"sota_runtime_ready": False, "blocker": f"missing {path}"}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return dict(loaded) if isinstance(loaded, Mapping) else {"sota_runtime_ready": False}


def sha16(text: str) -> str:
    """Return a short stable checksum for local prompts and labels."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _complete_artifact(
    *,
    root: Path,
    preflight_artifact: Mapping[str, Any],
    signal_surface: Mapping[str, Any],
    model_specs: Mapping[str, JsonDict],
    fixtures: Sequence[HallucinationFixture],
    pilot_rows: Sequence[Mapping[str, Any]],
    separation_summary: Mapping[str, Any],
    commands_run: Sequence[Mapping[str, Any]],
    duration_s: float,
) -> JsonDict:
    delta = float(separation_summary["signal_delta"])
    return {
        "schema": SCHEMA,
        "spec_refs": list(SPEC_REFS),
        "duration_s": round(duration_s, 6),
        "honest_verdict": _wrap("honest_verdict", _honest_verdict(delta)),
        "inference_substrate": _wrap("inference_substrate", LIVE_SUBSTRATE),
        "preconditions_checked": _preconditions(
            root,
            preflight_artifact,
            signal_surface,
            fixtures,
            live_signal_receipts_found=True,
            blocker="",
        ),
        "MODEL_SPECS": _wrap("MODEL_SPECS", model_specs),
        "internal_signal_available": True,
        "internal_signal_available_principle": _internal_signal_principle(signal_surface, live_receipts=True),
        "hidden_energy_probe_signal_delta": delta,
        "hidden_energy_probe_signal_delta_principle": FIELD_PRINCIPLES["hidden_energy_probe_signal_delta"],
        "false_accepts_at_threshold": _wrap(
            "false_accepts_at_threshold",
            int(separation_summary["false_accepts_at_threshold"]),
        ),
        "external_text_scorer_used": _wrap("external_text_scorer_used", False),
        "fixture_checksums": _wrap("fixture_checksums", fixture_checksums(fixtures)),
        "commands_run": [dict(row) for row in commands_run],
        "pilot_rows": [dict(row) for row in pilot_rows],
        "separation_summary": dict(separation_summary),
        "deterministic_baselines": deterministic_baselines(fixtures),
        "feature_preregistration": {
            "detector_trained": False,
            "primary_energy_order": [
                "sequence_marginal_energy",
                "full_logit_spilled_energy",
                "final_token_spilled_energy",
            ],
            "higher_energy_means_more_unsupported_risk": True,
        },
        "reproducibility_checksum": sha16(
            _stable_json(
                {
                    "schema": SCHEMA,
                    "fixtures": fixture_checksums(fixtures),
                    "summary": separation_summary,
                    "model_specs": model_specs,
                }
            )
        ),
    }


def _blocked_artifact(
    *,
    root: Path,
    preflight_artifact: Mapping[str, Any],
    signal_surface: Mapping[str, Any],
    model_specs: Mapping[str, JsonDict],
    fixtures: Sequence[HallucinationFixture],
    commands_run: Sequence[Mapping[str, Any]],
    blocker: str,
    live_signal_receipts_found: bool,
    duration_s: float,
) -> JsonDict:
    verdict_detail = (
        "unavailable exp5259_sota_runtime_ready_not_true"
        if blocker == "blocked_sota_runtime_unavailable"
        else "unavailable local runtime exposes only generated text or no live signal receipts"
    )
    return {
        "schema": SCHEMA,
        "spec_refs": list(SPEC_REFS),
        "duration_s": round(duration_s, 6),
        "honest_verdict": _wrap("honest_verdict", f"{blocker}: {verdict_detail}"),
        "inference_substrate": _wrap("inference_substrate", PREFLIGHT_SUBSTRATE),
        "preconditions_checked": _preconditions(
            root,
            preflight_artifact,
            signal_surface,
            fixtures,
            live_signal_receipts_found=live_signal_receipts_found,
            blocker=blocker,
        ),
        "MODEL_SPECS": _wrap("MODEL_SPECS", model_specs),
        "internal_signal_available": False,
        "internal_signal_available_principle": _internal_signal_principle(signal_surface, live_receipts=False),
        "hidden_energy_probe_signal_delta": 0.0,
        "hidden_energy_probe_signal_delta_principle": FIELD_PRINCIPLES["hidden_energy_probe_signal_delta"],
        "false_accepts_at_threshold": _wrap("false_accepts_at_threshold", 0),
        "external_text_scorer_used": _wrap("external_text_scorer_used", False),
        "fixture_checksums": _wrap("fixture_checksums", fixture_checksums(fixtures)),
        "commands_run": [dict(row) for row in commands_run],
        "pilot_rows": [],
        "separation_summary": {
            "n_scored": 0,
            "signal_delta": 0.0,
            "auroc": None,
            "false_accepts_at_threshold": 0,
            "blocked_reason": blocker,
        },
        "deterministic_baselines": deterministic_baselines(fixtures),
        "feature_preregistration": {
            "detector_trained": False,
            "primary_energy_order": [
                "sequence_marginal_energy",
                "full_logit_spilled_energy",
                "final_token_spilled_energy",
            ],
            "higher_energy_means_more_unsupported_risk": True,
        },
        "blocker": blocker,
    }


def _evaluate_fixture(
    fixture: HallucinationFixture,
    model_spec: JsonDict,
    generation_runner: GenerationRunner,
    *,
    seed: int,
) -> JsonDict:
    generation = generation_runner(fixture, model_spec, seed)
    features = compute_energy_features(generation)
    raw_response = str(generation.get("raw_response") or generation.get("response_text") or "")
    return {
        "fixture_id": fixture.fixture_id,
        "unsupported_label": bool(fixture.unsupported_label),
        "label_source": fixture.label_source,
        "claim_checksum": sha16(fixture.claim),
        "evidence_checksum": sha16(fixture.evidence),
        "prompt_checksum": sha16(render_prompt(fixture)),
        "raw_response_checksum": sha16(raw_response),
        "raw_response_excerpt": raw_response[:160],
        "energy_features": features,
        "signal_receipts": {
            "token_logprob_count": features["token_logprob_count"],
            "top_logprobs_count": features["top_logprobs_count"],
            "final_logits_count": features["final_logits_count"],
            "logit_receipt": dict(generation.get("logit_receipt") or generation.get("final_logits_summary") or {}),
        },
        "seed": int(seed),
    }


def _preconditions(
    root: Path,
    preflight_artifact: Mapping[str, Any],
    signal_surface: Mapping[str, Any],
    fixtures: Sequence[HallucinationFixture],
    *,
    live_signal_receipts_found: bool,
    blocker: str,
) -> JsonDict:
    checksums = fixture_checksums(fixtures)
    value = {
        "exp5259_artifact_path": str(root / EXP5259_RELATIVE_PATH),
        "exp5259_sota_runtime_ready": bool(preflight_artifact.get("sota_runtime_ready")),
        "exp5259_sota_runtime_ready_principle": preflight_artifact.get("sota_runtime_ready_principle"),
        "signal_surface": dict(signal_surface),
        "signal_surface_has_internal_signal": _surface_has_internal_signal(signal_surface),
        "live_signal_receipts_found": bool(live_signal_receipts_found),
        "text_only_runtime": bool(signal_surface.get("generated_text")) and not _surface_has_internal_signal(signal_surface),
        "fixture_label_counts": {
            "supported": checksums["supported_count"],
            "unsupported": checksums["unsupported_count"],
        },
        "model_runtime_receipts": _model_runtime_receipt_summary(preflight_artifact),
        "phase_d_external_text_scorer_retired": True,
        "external_text_scorer_used": False,
        "blocker": blocker,
    }
    return _wrap("preconditions_checked", value)


def _model_specs_from_preflight(preflight_artifact: Mapping[str, Any]) -> JsonDict:
    receipts = _nested_value(preflight_artifact, "model_receipts")
    if not isinstance(receipts, Mapping):
        receipts = {}
    ready_slot = _first_ready_slot(receipts)
    specs: JsonDict = {}
    for mandated in MANDATED_MODEL_SPECS:
        slot = str(mandated["slot"])
        receipt = receipts.get(slot, {})
        if not isinstance(receipt, Mapping):
            receipt = {}
        specs[slot] = {
            "slot": slot,
            "hf_id": str(mandated["hf_id"]),
            "role": str(mandated["role"]),
            "quantization": receipt.get("preferred_quant") or mandated["quantization"],
            "runtime_status": receipt.get("status", "missing_receipt"),
            "runtime_ready": bool(receipt.get("runtime_ready")),
            "selected_for_pilot": slot == ready_slot,
            "file_receipts": {
                "path": receipt.get("path"),
                "size_bytes": receipt.get("size_bytes"),
                "checksum_sha256": receipt.get("checksum_sha256"),
                "checksum_head_1m_sha256": receipt.get("checksum_head_1m_sha256"),
            },
        }
    return specs


def _model_runtime_receipt_summary(preflight_artifact: Mapping[str, Any]) -> JsonDict:
    return {
        slot: {
            "hf_id": spec["hf_id"],
            "status": spec["runtime_status"],
            "runtime_ready": spec["runtime_ready"],
            "path": spec["file_receipts"]["path"],
        }
        for slot, spec in _model_specs_from_preflight(preflight_artifact).items()
    }


def _first_ready_slot(receipts: Mapping[str, Any]) -> str | None:
    for mandated in MANDATED_MODEL_SPECS:
        slot = str(mandated["slot"])
        receipt = receipts.get(slot, {})
        if isinstance(receipt, Mapping) and receipt.get("runtime_ready") and receipt.get("path"):
            return slot
    return None


def _select_pilot_model(model_specs: Mapping[str, JsonDict]) -> JsonDict | None:
    for mandated in MANDATED_MODEL_SPECS:
        slot = str(mandated["slot"])
        spec = model_specs.get(slot)
        if spec and spec.get("runtime_ready") and spec.get("file_receipts", {}).get("path"):
            return dict(spec)
    return None


def _surface_has_internal_signal(surface: Mapping[str, Any]) -> bool:
    return any(bool(surface.get(key)) for key in SIGNAL_KEYS)


def _internal_signal_principle(surface: Mapping[str, Any], *, live_receipts: bool) -> str:
    available = [key for key in SIGNAL_KEYS if surface.get(key)]
    if live_receipts and available:
        return (
            "internal_signal_available=true because the runtime exposed "
            f"{available} with live receipts; hidden_states={bool(surface.get('hidden_states'))}, "
            f"attention_tensors={bool(surface.get('attention_tensors'))}."
        )
    if available:
        return (
            "internal_signal_available=false because the API surface advertised "
            f"{available} but the pilot did not receive live non-text receipts."
        )
    return "internal_signal_available=false because the runtime was generated-text-only."


def _honest_verdict(delta: float) -> str:
    if delta > 0.05:
        return f"complete: signal logit-energy unsupported-minus-supported delta={delta:.6f}"
    if delta < -0.05:
        return f"complete: harmful logit-energy unsupported-minus-supported delta={delta:.6f}"
    return f"complete: null logit-energy unsupported-minus-supported delta={delta:.6f}"


def _claim_supported_by_evidence(claim: str, evidence: str) -> bool:
    evidence_terms = set(_content_terms(evidence))
    return all(term in evidence_terms for term in _content_terms(claim))


def _content_terms(text: str) -> list[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "as",
        "by",
        "in",
        "of",
        "on",
        "the",
        "to",
        "under",
        "was",
        "with",
    }
    cleaned = "".join(char.lower() if char.isalnum() else " " for char in text)
    return [term for term in cleaned.split() if term and term not in stopwords]


def _zero_false_accept_threshold(unsupported_scores: Sequence[float]) -> float | None:
    if not unsupported_scores:
        return None
    return min(unsupported_scores) - 1e-12


def _auroc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    positives = [score for label, score in zip(labels, scores, strict=False) if label == 1]
    negatives = [score for label, score in zip(labels, scores, strict=False) if label == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    total = 0
    for positive in positives:
        for negative in negatives:
            total += 1
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / total if total else None


def _first_choice(response: Any) -> JsonDict:
    if isinstance(response, Mapping):
        choices = response.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
            return dict(choices[0])
    return {"text": str(response), "logprobs": {}}


def _full_logit_summary(values: Any, *, top_k: int = 8) -> JsonDict:
    numeric = _numeric_values(values)
    if not numeric:
        return {}
    maximum = max(numeric)
    weights = [math.exp(value - maximum) for value in numeric]
    total = sum(weights)
    if total <= 0.0:  # pragma: no cover - finite exponentials have positive sum.
        return {}
    top = sorted(enumerate(numeric), key=lambda item: item[1], reverse=True)[:top_k]
    top_probs = [math.exp(value - maximum) / total for _index, value in top]
    top_mass = sum(top_probs)
    normalized_top = [prob / top_mass for prob in top_probs] if top_mass > 0 else []
    entropy_topk = -sum(prob * math.log(max(prob, 1e-12)) for prob in normalized_top)
    return {
        "vocab_size": len(numeric),
        "top1_probability": max(weights) / total,
        "top_mass_probability": top_mass,
        "entropy_topk": entropy_topk,
        "top_logits": [
            {"token_index": int(index), "logit": float(value), "probability": float(prob)}
            for (index, value), prob in zip(top, top_probs, strict=False)
        ],
    }


def _summary_from_generation(generation: Mapping[str, Any]) -> JsonDict:
    summary = generation.get("final_logits_summary") or generation.get("logit_receipt")
    return dict(summary) if isinstance(summary, Mapping) else {}


def _final_logits(generation: Mapping[str, Any]) -> list[float]:
    for key in ("final_logits", "logits"):
        values = generation.get(key)
        if isinstance(values, list):
            if values and isinstance(values[-1], list):
                return _numeric_values(values[-1])
            return _numeric_values(values)
    return []


def _top_logprob_rows(value: Any) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    if not isinstance(value, list):
        return rows
    for item in value:
        if isinstance(item, Mapping):
            converted = {
                str(key): numeric
                for key, raw in item.items()
                if (numeric := _optional_float(raw)) is not None
            }
            if converted:
                rows.append(converted)
    return rows


def _numeric_values(value: Any) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    values: list[float] = []
    for item in value:
        number = _optional_float(item)
        if number is not None:
            values.append(number)
    return values


def _optional_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _softmax_log_values(values: Sequence[float]) -> list[float]:
    numeric = [float(value) for value in values if math.isfinite(float(value))]
    if not numeric:
        return []
    maximum = max(numeric)
    weights = [math.exp(value - maximum) for value in numeric]
    total = sum(weights)
    return [weight / total for weight in weights] if total else []


def _selected_final_probability(token_logprobs: Sequence[float]) -> float | None:
    if not token_logprobs:
        return None
    return max(0.0, min(1.0, math.exp(min(0.0, float(token_logprobs[-1])))))


def _token_count(generation: Mapping[str, Any], token_logprobs: Sequence[float]) -> int:
    explicit = generation.get("token_count") or generation.get("tokens_generated")
    if isinstance(explicit, int) and explicit >= 0:
        return explicit
    tokens = generation.get("tokens")
    if isinstance(tokens, list):
        return len(tokens)
    return len(token_logprobs)


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _nested_value(payload: Any, field: str) -> Any:
    if isinstance(payload, Mapping):
        value = payload.get(field)
        if isinstance(value, Mapping) and "value" in value:
            return value["value"]
        return value
    return None


def _wrap(field: str, value: Any) -> JsonDict:
    return {"value": value, "principle": FIELD_PRINCIPLES[field]}


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def _write_json_if_requested(path: Path, artifact: Mapping[str, Any], write: bool) -> None:
    if not write:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--preflight", default=str(REPO_ROOT / EXP5259_RELATIVE_PATH))
    args = parser.parse_args(argv)
    preflight = json.loads(Path(args.preflight).read_text(encoding="utf-8"))
    artifact = run_pilot(
        result_path=Path(args.output),
        preflight_artifact=preflight,
        commands_run=[
            {
                "command": (
                    ".venv/bin/python -m carnot.experiment_5263_neuron_attention_energy_hallucination_probe_v481 "
                    "--output results/experiment_5263_neuron_attention_energy_hallucination_probe_v481.json"
                ),
                "outcome": "completed module invocation",
            }
        ],
    )
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
