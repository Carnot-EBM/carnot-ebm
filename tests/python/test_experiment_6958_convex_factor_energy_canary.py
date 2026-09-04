"""Tests for the bounded input-convex factor-energy canary.

Spec refs: REQ-ENERGY-6958 and SCENARIO-ENERGY-6958-*.
"""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from carnot import experiment_6958_convex_factor_energy_canary as exp


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "results/experiment_6955_reformulation_fixture.json"
CORPUS = ROOT / "results/checkpoints/experiment_6955_reformulation_fixture_corpus.json"


@pytest.fixture(scope="module")
def fixture_pairs() -> list[dict[str, object]]:
    """REQ-ENERGY-6958 uses only the serialized Exp6955 corpus as candidate input."""

    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))["pairs"]
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return exp.attach_fixture_witnesses(corpus, fixture)


@pytest.fixture(scope="module")
def candidates(fixture_pairs: list[dict[str, object]]) -> list[exp.Candidate]:
    """SCENARIO-ENERGY-6958-FACTORS freezes one valid/corrupt pair per mapping."""

    return exp.freeze_candidates(fixture_pairs, exp.RANDOM_SEED)


@pytest.fixture(scope="module")
def artifact(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """REQ-ENERGY-6958 builds one reduced-budget artifact for schema tests."""

    work = tmp_path_factory.mktemp("exp6958")
    return exp.build_artifact(
        date="20260904",
        repo_root=ROOT,
        fixture_path=FIXTURE,
        corpus_path=CORPUS,
        checkpoint_dir=work / "checkpoints",
        replay_manifest_path=work / "replay.json",
        seeds=(exp.RANDOM_SEED, exp.RANDOM_SEED + 1),
        epochs=12,
        convexity_samples=4,
        bootstrap_samples=200,
    )


def test_req_energy_6958_spec_precedes_and_owns_contract() -> None:
    """REQ-ENERGY-6958 declares required fields and scenarios before implementation."""

    text = (ROOT / exp.SPEC_PATH).read_text(encoding="utf-8")
    section = text[text.index("### REQ-ENERGY-6958") :]

    assert all(f"`{field}`" in section for field in exp.REQUIRED_ARTIFACT_FIELDS)
    assert all(
        f"SCENARIO-ENERGY-6958-{name}" in section
        for name in (
            "PRECONDITIONS",
            "FACTORS",
            "CONVEXITY",
            "BUDGET",
            "ORDERING",
            "OPTIMIZATION",
            "TRANSFER",
            "REPLAY",
        )
    )


def test_scenario_energy_6958_preconditions_pass_and_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-ENERGY-6958-PRECONDITIONS reports exact blocked observations."""

    ready = exp.check_preconditions(ROOT, FIXTURE, CORPUS, tmp_path / "ready")
    assert ready["passed"] is True
    assert all(row["passed"] for row in ready["checks"])

    broken_fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    broken_fixture["reformulation_fixture_ready_score"] = 0
    broken_path = tmp_path / "broken.json"
    broken_path.write_text(json.dumps(broken_fixture), encoding="utf-8")
    blocked = exp.build_artifact(
        date="20260904",
        repo_root=ROOT,
        fixture_path=broken_path,
        corpus_path=CORPUS,
        checkpoint_dir=tmp_path / "blocked-checkpoints",
        replay_manifest_path=tmp_path / "blocked-replay.json",
        seeds=(exp.RANDOM_SEED,),
        epochs=1,
        convexity_samples=1,
        bootstrap_samples=10,
    )
    failed = {row["check"]: row for row in blocked["gate_check_summary"]["checks"]}
    assert failed["reformulation_fixture_ready_score"] == {
        "check": "reformulation_fixture_ready_score",
        "expected": 1,
        "observed": 0,
        "passed": False,
    }
    assert blocked["convex_factor_run_complete_score"] == 0
    assert blocked["verdict_class"] == "blocked"
    assert blocked["honest_verdict"] == "blocked_convex_factor_energy_canary"
    assert set(blocked["field_principles"]) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    exp.validate_artifact(blocked)


def test_scenario_energy_6958_factors_cover_corruptions_without_label_leakage(
    fixture_pairs: list[dict[str, object]], candidates: list[exp.Candidate]
) -> None:
    """SCENARIO-ENERGY-6958-FACTORS covers all channels and excludes oracle fields."""

    equivalents = [row for row in fixture_pairs if row["expected_label"] == "equivalent"]
    assert len(candidates) == 2 * len(equivalents)
    assert {row.corruption_kind for row in candidates if row.is_corrupt} == set(
        exp.CORRUPTION_KINDS
    )
    by_pair: dict[str, list[exp.Candidate]] = {}
    for candidate in candidates:
        by_pair.setdefault(candidate.pair_id, []).append(candidate)
        payload = exp.feature_payload(candidate)
        assert not (set(exp.FORBIDDEN_FEATURE_FIELDS) & set(payload))
        assert len(candidate.factors) == candidate.factor_count
        assert all(len(factor) == exp.FEATURE_DIM for factor in candidate.factors)
    assert all(sorted(row.is_corrupt for row in pair) == [0, 1] for pair in by_pair.values())
    for pair in by_pair.values():
        valid = next(row for row in pair if not row.is_corrupt)
        corrupt = next(row for row in pair if row.is_corrupt)
        assert sum(sum(factor) for factor in valid.factors) == 0.0
        assert sum(sum(factor) for factor in corrupt.factors) > 0.0


def test_scenario_energy_6958_factor_sum_ignores_permutation_and_padding() -> None:
    """SCENARIO-ENERGY-6958-FACTORS makes masks and factor order semantically inert."""

    model = exp.make_model(exp.ARM_CONVEX, seed=17)
    factors = torch.tensor([[[0.1] * exp.FEATURE_DIM, [0.4] * exp.FEATURE_DIM]])
    mask = torch.tensor([[1.0, 1.0]])
    base = model(factors, mask)
    assert torch.equal(base, model(factors.flip(1), mask.flip(1)))

    padded = torch.cat((factors, torch.full((1, 2, exp.FEATURE_DIM), 99.0)), dim=1)
    padded_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    assert torch.equal(base, model(padded, padded_mask))


def test_scenario_energy_6958_split_templates_cannot_leak(
    candidates: list[exp.Candidate],
) -> None:
    """SCENARIO-ENERGY-6958-FACTORS rejects a generator template crossing splits."""

    rows = exp.audit_split_isolation(candidates)
    assert rows
    assert all(not row["crosses_split"] for row in rows)

    leaked = list(candidates)
    clone = deepcopy(leaked[0])
    clone.split = "held_out" if clone.split != "held_out" else "train"
    leaked.append(clone)
    with pytest.raises(ValueError, match="generator_template_crosses_split"):
        exp.audit_split_isolation(leaked)


def test_scenario_energy_6958_projection_jensen_finite_difference_and_gradient() -> None:
    """SCENARIO-ENERGY-6958-CONVEXITY projects weights and verifies convex geometry."""

    model = exp.make_model(exp.ARM_CONVEX, seed=23)
    with torch.no_grad():
        for parameter in model.constrained_parameters():
            parameter.fill_(-0.5)
    assert exp.minimum_convex_weight(model) < 0
    exp.project_nonnegative_(model)
    assert exp.minimum_convex_weight(model) == 0

    checks = exp.convexity_checks(model, seed=23, samples=8)
    assert all(row["passed"] for rows in checks.values() for row in rows)


def test_scenario_energy_6958_training_is_matched_and_replays_deterministically(
    candidates: list[exp.Candidate],
) -> None:
    """SCENARIO-ENERGY-6958-BUDGET matches nonlinear capacity and seeded fitting."""

    train = [row for row in candidates if row.split == "train"]
    first = exp.fit_arm(exp.ARM_CONVEX, train, seed=101, epochs=5)
    second = exp.fit_arm(exp.ARM_CONVEX, train, seed=101, epochs=5)
    assert first.metadata == second.metadata
    assert first.training_rows == second.training_rows
    assert all(
        torch.equal(first.model.state_dict()[key], second.model.state_dict()[key])
        for key in first.model.state_dict()
    )
    mlp = exp.make_model(exp.ARM_MLP, seed=101)
    assert exp.model_parameter_count(first.model) == exp.model_parameter_count(mlp)

    shuffled_a = exp.training_labels(train, seed=101, shuffled=True)
    shuffled_b = exp.training_labels(train, seed=101, shuffled=True)
    assert torch.equal(shuffled_a, shuffled_b)
    assert not torch.equal(shuffled_a, exp.training_labels(train, seed=101, shuffled=False))


def test_scenario_energy_6958_ordering_ties_and_ci_are_conservative() -> None:
    """SCENARIO-ENERGY-6958-ORDERING preserves half-credit ties and strict CI gates."""

    assert exp.ordering_credit(0.0, 1.0) == 1.0
    assert exp.ordering_credit(1.0, 0.0) == 0.0
    assert exp.ordering_credit(1.0, 1.0) == 0.5
    low, high = exp.bootstrap_ci([0.0, 0.0, 0.0], seed=9, samples=100)
    assert (low, high) == (0.0, 0.0)
    assert exp.positive_gate(True, True, [(0.0, 1.0), (0.1, 1.0)]) is False
    assert exp.positive_gate(True, True, [(0.1, 1.0), (0.2, 1.0)]) is True


def test_scenario_energy_6958_projected_optimizer_is_bounded_and_repeatable() -> None:
    """SCENARIO-ENERGY-6958-OPTIMIZATION reaches the box minimum from every start."""

    model = exp.make_model(exp.ARM_CONVEX, seed=31)
    exp.project_nonnegative_(model)
    starts = [torch.zeros((3, exp.FEATURE_DIM)), torch.ones((3, exp.FEATURE_DIM))]
    first = [exp.projected_optimize(model, start, steps=80) for start in starts]
    second = [exp.projected_optimize(model, start, steps=80) for start in starts]
    assert first == second
    assert all(row["terminal"] for row in first)
    assert all(
        0.0 <= row["minimum_coordinate"] <= row["maximum_coordinate"] <= 1.0 for row in first
    )
    assert first[0]["final_energy"] == pytest.approx(first[1]["final_energy"], abs=1e-8)


def test_req_energy_6958_complete_artifact_has_rows_transfer_and_fresh_replay(
    artifact: dict[str, object],
) -> None:
    """REQ-ENERGY-6958 requires complete evidence, size transfer, and checkpoint replay."""

    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(artifact["field_principles"]) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False
    assert artifact["convex_factor_run_complete_score"] == 1
    assert artifact["convex_factor_positive_score"] in {0, 1}
    assert artifact["verdict_class"] in {"positive", "null"}
    assert artifact["honest_verdict"].startswith("complete_")
    assert len(artifact["arm_rows"]) == len(exp.ARMS)
    assert len(artifact["seed_rows"]) == len(exp.ARMS) * 2
    assert {row["size_band"] for row in artifact["size_transfer_rows"]} == {
        "smaller_or_equal_two_variables",
        "larger_than_two_variables",
    }
    assert all(row["terminal"] for row in artifact["optimizer_rows"])
    assert all(row["replay_matches"] for row in artifact["fresh_process_replay_rows"])
    assert all(Path(path).exists() for path in artifact["checkpoint_paths"])
    assert artifact["gate_check_summary"]["passed"] is True
    exp.validate_artifact(artifact)
    assert exp.payload_checksum(artifact) == artifact["reproducibility_checksum"]


def test_scenario_energy_6958_replay_rejects_tampered_checkpoint(
    artifact: dict[str, object], tmp_path: Path
) -> None:
    """SCENARIO-ENERGY-6958-REPLAY binds replay to checkpoint content hashes."""

    source = Path(artifact["checkpoint_paths"][0])
    tampered = tmp_path / "tampered.pt"
    tampered.write_bytes(source.read_bytes() + b"tamper")
    manifest = {
        "schema_version": exp.REPLAY_SCHEMA_VERSION,
        "models": [
            {
                "arm": exp.ARM_CONVEX,
                "seed": exp.RANDOM_SEED,
                "checkpoint_path": str(tampered),
                "checkpoint_sha256": "sha256:not-the-file",
                "candidates": [],
            }
        ],
    }
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint_hash_mismatch"):
        exp.replay_manifest(path)


def test_req_energy_6958_defensive_input_and_checkpoint_failures(
    fixture_pairs: list[dict[str, object]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ENERGY-6958 fails closed on malformed inputs and checkpoint bindings."""

    assert exp._fraction(Fraction(1, 2)) == Fraction(1, 2)
    assert exp._fraction_text(Fraction(1, 2)) == "1/2"
    assert exp.sha256_path(tmp_path / "missing") is None
    assert exp._same_fraction("bad", "1") is False
    with pytest.raises(ValueError, match="unknown_corruption"):
        exp.corrupt_mapping(fixture_pairs[0], "not_registered")
    with pytest.raises(ValueError, match="empty_candidate_batch"):
        exp.batch_tensors([])
    with pytest.raises(ValueError, match="unknown_arm"):
        exp.make_model("not_an_arm", 1)
    with pytest.raises(ValueError, match="arm_is_not_learned"):
        exp.fit_arm(exp.ARM_HAND, [], seed=1, epochs=1)
    assert exp.make_model(exp.ARM_MLP, 1).constrained_parameters() == ()
    assert exp.minimum_convex_weight(exp.make_model(exp.ARM_LINEAR, 1)) is None
    assert exp.bootstrap_ci([], seed=1, samples=10) == (None, None)

    malformed = deepcopy(fixture_pairs[0]["mapping"])
    malformed["objective"].pop("scale")
    factors = exp.encode_candidate(fixture_pairs[0], malformed)
    assert factors[-1][-1] == 1.0
    malformed_affine = deepcopy(fixture_pairs[0]["mapping"])
    malformed_affine["variables"][0]["scale"] = "bad"
    factors = exp.encode_candidate(fixture_pairs[0], malformed_affine)
    assert factors[0][2] == 1.0

    incomplete_fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    incomplete_fixture["objective_order_rows"].pop()
    with pytest.raises(ValueError, match="missing_fixture_witness_row"):
        exp.attach_fixture_witnesses(fixture_pairs, incomplete_fixture)

    missing_preconditions = exp.check_preconditions(
        ROOT, tmp_path / "missing-fixture", tmp_path / "missing-corpus", tmp_path / "writable"
    )
    assert missing_preconditions["passed"] is False

    original_tensor = exp.torch.tensor
    monkeypatch.setattr(
        exp.torch,
        "tensor",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("cpu unavailable")),
    )
    cpu_failed = exp.check_preconditions(ROOT, FIXTURE, CORPUS, tmp_path / "cpu-failed")
    assert (
        next(row for row in cpu_failed["checks"] if row["check"] == "pytorch_cpu_autograd")[
            "passed"
        ]
        is False
    )
    monkeypatch.setattr(exp.torch, "tensor", original_tensor)

    monkeypatch.setattr(
        exp.tempfile,
        "NamedTemporaryFile",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("read only")),
    )
    write_failed = exp.check_preconditions(ROOT, FIXTURE, CORPUS, tmp_path / "write-failed")
    assert (
        next(row for row in write_failed["checks"] if row["check"] == "writable_checkpoints")[
            "passed"
        ]
        is False
    )


def test_scenario_energy_6958_checkpoint_and_child_replay_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-ENERGY-6958-REPLAY rejects schemas, bindings, writes, and child failures."""

    model = exp.make_model(exp.ARM_CONVEX, 7)
    checkpoint = tmp_path / "model.pt"
    exp._save_checkpoint(checkpoint, exp.ARM_CONVEX, 7, model)
    with pytest.raises(ValueError, match="checkpoint_binding_mismatch"):
        exp._load_checkpoint(checkpoint, exp.ARM_CONVEX, 8)

    bad_schema_checkpoint = tmp_path / "bad-schema.pt"
    torch.save(
        {
            "schema_version": "wrong",
            "arm": exp.ARM_CONVEX,
            "seed": 7,
            "state_dict": model.state_dict(),
        },
        bad_schema_checkpoint,
    )
    with pytest.raises(ValueError, match="checkpoint_schema_mismatch"):
        exp._load_checkpoint(bad_schema_checkpoint, exp.ARM_CONVEX, 7)

    bad_manifest = tmp_path / "bad-manifest.json"
    bad_manifest.write_text('{"schema_version":"wrong"}', encoding="utf-8")
    with pytest.raises(ValueError, match="replay_manifest_schema"):
        exp.replay_manifest(bad_manifest)

    original_save = exp.torch.save
    monkeypatch.setattr(
        exp.torch,
        "save",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("write interrupted")),
    )
    interrupted = tmp_path / "interrupted.pt"
    with pytest.raises(RuntimeError, match="write interrupted"):
        exp._save_checkpoint(interrupted, exp.ARM_CONVEX, 7, model)
    assert not list(tmp_path.glob(".interrupted.pt-*"))
    monkeypatch.setattr(exp.torch, "save", original_save)

    manifest = tmp_path / "child.json"
    manifest.write_text(
        json.dumps({"schema_version": exp.REPLAY_SCHEMA_VERSION, "models": []}), encoding="utf-8"
    )
    monkeypatch.setattr(
        exp.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stderr="child failed"),
    )
    with pytest.raises(RuntimeError, match="fresh_process_replay_failed"):
        exp._fresh_process_replay(ROOT, manifest)

    def write_nonlist(*args: object, **kwargs: object) -> SimpleNamespace:
        manifest.with_suffix(".child.json").write_text("{}", encoding="utf-8")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(exp.subprocess, "run", write_nonlist)
    with pytest.raises(RuntimeError, match="fresh_process_replay_not_rows"):
        exp._fresh_process_replay(ROOT, manifest)


def test_req_energy_6958_validation_guards_and_run_surface(
    artifact: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ENERGY-6958 validates every gate binding and atomically exposes ``run``."""

    variants: list[tuple[dict[str, object], str]] = []
    missing = deepcopy(artifact)
    missing.pop("rows")
    variants.append((missing, "missing_artifact_fields"))
    principles = deepcopy(artifact)
    principles["field_principles"] = {}
    variants.append((principles, "field_principles_mismatch"))
    substrate = deepcopy(artifact)
    substrate["inference_substrate"] = "wrong"
    variants.append((substrate, "inference_substrate_mismatch"))
    oracle = deepcopy(artifact)
    oracle["verifier_is_oracle"] = True
    variants.append((oracle, "verifier_is_oracle_mismatch"))
    positive_without_completion = deepcopy(artifact)
    positive_without_completion["convex_factor_positive_score"] = 1
    positive_without_completion["convex_factor_run_complete_score"] = 0
    variants.append((positive_without_completion, "positive_without_completion"))
    wrong_class = deepcopy(artifact)
    wrong_class["verdict_class"] = "partial"
    variants.append((wrong_class, "verdict_class_score_mismatch"))
    checksum = deepcopy(artifact)
    checksum["reproducibility_checksum"] = "wrong"
    variants.append((checksum, "reproducibility_checksum_mismatch"))
    for value, reason in variants:
        if reason not in {"missing_artifact_fields", "reproducibility_checksum_mismatch"}:
            value["reproducibility_checksum"] = exp.payload_checksum(value)
        with pytest.raises(ValueError, match=reason):
            exp.validate_artifact(value)

    payload = deepcopy(artifact)
    output = tmp_path / "artifact.json"
    monkeypatch.setattr(exp, "build_artifact", lambda **kwargs: payload)
    observed = exp.run("20260904", repo_root=ROOT, output_path=output)
    assert observed == payload
    assert json.loads(output.read_text(encoding="utf-8")) == payload
