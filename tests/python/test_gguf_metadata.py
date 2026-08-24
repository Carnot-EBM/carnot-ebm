"""Focused tests for bounded GGUF content and Hugging Face provenance."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
import struct

import pytest

import carnot.inference.gguf_metadata as gguf_metadata
from carnot.inference.gguf_metadata import (
    GgufMetadataError,
    bind_hf_cache_provenance,
    build_gguf_admission_record,
    read_gguf_metadata,
)


def _string(value: str | bytes) -> bytes:
    raw = value.encode("utf-8") if isinstance(value, str) else value
    return struct.pack("<Q", len(raw)) + raw


def _value(value_type: int, value: object) -> bytes:
    if value_type == 4:
        return struct.pack("<I", int(value))
    if value_type == 7:
        return struct.pack("<B", int(bool(value)))
    if value_type == 8:
        assert isinstance(value, (str, bytes))
        return _string(value)
    if value_type == 9:
        element_type, elements = value  # type: ignore[misc]
        encoded = struct.pack("<IQ", element_type, len(elements))
        return encoded + b"".join(_value(element_type, element) for element in elements)
    if value_type == 10:
        return struct.pack("<Q", int(value))
    raise AssertionError(f"fixture type not supported: {value_type}")


def make_gguf(
    *,
    architecture: str | bytes = "gemma4",
    file_type: int = 15,
    tokenizer_model: str | bytes = "gemma4",
    tensor_count: int = 1,
    version: int = 3,
    split_no: int | None = None,
    split_count: int | None = None,
    extra_metadata: list[tuple[str | bytes, int, object]] | None = None,
    tensor_dimensions: tuple[int, ...] = (1,),
    tensor_offset: int = 0,
) -> bytes:
    metadata: list[tuple[str | bytes, int, object]] = [
        ("general.architecture", 8, architecture),
        ("general.name", 8, "fixture-model"),
        ("general.file_type", 4, file_type),
        ("tokenizer.ggml.model", 8, tokenizer_model),
        ("tokenizer.ggml.tokens", 9, (8, ["one", "two", "three"])),
        ("tokenizer.ggml.bos_token_id", 4, 1),
        ("tokenizer.ggml.eos_token_id", 4, 2),
    ]
    if split_no is not None:
        metadata.append(("split.no", 4, split_no))
    if split_count is not None:
        metadata.append(("split.count", 4, split_count))
        metadata.append(("split.tensors.count", 10, tensor_count * split_count))
    metadata.extend(extra_metadata or [])
    payload = bytearray(b"GGUF")
    payload.extend(struct.pack("<IQQ", version, tensor_count, len(metadata)))
    for key, value_type, value in metadata:
        payload.extend(_string(key))
        payload.extend(struct.pack("<I", value_type))
        payload.extend(_value(value_type, value))
    for index in range(tensor_count):
        payload.extend(_string(f"tensor-{index}"))
        payload.extend(struct.pack("<I", len(tensor_dimensions)))
        payload.extend(b"".join(struct.pack("<Q", value) for value in tensor_dimensions))
        payload.extend(struct.pack("<IQ", 0, tensor_offset + index * 4))
    padding = (-len(payload)) % 32
    payload.extend(b"\x00" * padding)
    payload.extend(b"\x00" * max(4, tensor_count * 4))
    return bytes(payload)


def cache_blob(
    tmp_path: Path,
    content: bytes,
    *,
    repository_id: str = "unsloth/fixture-GGUF",
    revision: str = "revision-a",
    filename: str = "fixture-Q4_K_M.gguf",
) -> tuple[Path, Path, str]:
    cache_root = tmp_path / "hub"
    repo_dir = cache_root / f"models--{repository_id.replace('/', '--')}"
    digest = hashlib.sha256(content).hexdigest()
    blob = repo_dir / "blobs" / digest
    blob.parent.mkdir(parents=True)
    blob.write_bytes(content)
    snapshot = repo_dir / "snapshots" / revision
    snapshot.mkdir(parents=True)
    link = snapshot / filename
    link.symlink_to(Path("../../blobs") / digest)
    (repo_dir / "refs").mkdir()
    (repo_dir / "refs" / "main").write_text(revision)
    return blob, cache_root, f"sha256:{digest}"


# REQ-REPORT-6572-CONTENT / SCENARIO-REPORT-6572-HASH-BLOB:
# identity and quantization come from bytes even when the path is a hash.
def test_reads_content_identity_from_hash_only_blob(tmp_path: Path) -> None:
    blob, _, _ = cache_blob(tmp_path, make_gguf())

    row = read_gguf_metadata(blob)

    assert blob.suffix == ""
    assert row["magic"] == "GGUF"
    assert row["version"] == 3
    assert row["architecture"] == "gemma4"
    assert row["general_file_type"] == 15
    assert row["quantization"] == "Q4_K_M"
    assert row["tensor_count"] == 1
    assert row["tokenizer_metadata"]["model"] == "gemma4"
    assert row["tokenizer_metadata"]["token_count"] == 3
    assert row["is_language_model"] is True


# REQ-REPORT-6572-CONTENT: a misleading local name cannot override file type.
def test_quantization_is_not_guessed_from_filename(tmp_path: Path) -> None:
    path = tmp_path / "renamed-BF16.gguf"
    path.write_bytes(make_gguf(file_type=15))

    assert read_gguf_metadata(path)["quantization"] == "Q4_K_M"


# REQ-REPORT-6572-CONTENT: bounded scalar arrays are skipped by encoded size.
def test_scalar_metadata_array_is_bounded(tmp_path: Path) -> None:
    path = tmp_path / "scalar-array"
    path.write_bytes(make_gguf(extra_metadata=[("fixture.ids", 9, (4, [1, 2]))]))

    row = read_gguf_metadata(path)

    source = row["field_provenance"]["metadata_keys"]["fixture.ids"]
    assert source["value_end_offset"] > source["value_offset"]


# REQ-REPORT-6572-BOUNDED / SCENARIO-REPORT-6572-BOUNDED:
# the parser reaches tensor data without consuming it.
def test_bounded_receipt_records_no_tensor_payload_reads(tmp_path: Path) -> None:
    path = tmp_path / "bounded"
    path.write_bytes(make_gguf() + b"tensor-payload-never-read" * 100)

    row = read_gguf_metadata(path, max_header_bytes=4096)

    receipt = row["bounded_read_receipt"]
    assert receipt["physical_bytes_read"] <= 4096
    assert receipt["maximum_header_bytes"] == 4096
    assert receipt["tensor_payload_bytes_read"] == 0
    assert receipt["data_offset"] < receipt["file_size"]
    assert receipt["required_minimum_file_size"] <= receipt["file_size"]


@pytest.mark.parametrize(
    ("name", "payload", "reason"),
    [
        ("non_gguf", b"not a model", "invalid_magic"),
        ("truncated", make_gguf()[:17], "truncated_header"),
        ("unsupported_version", make_gguf(version=99), "unsupported_version"),
        ("tokenizer_only", make_gguf(tensor_count=0), "tokenizer_only"),
        ("prefix_collision", b"GGUFjunk-prefix", "unsupported_version"),
        (
            "tensor_overflow",
            b"GGUF" + struct.pack("<IQQ", 3, 1_000_001, 0),
            "tensor_count_limit",
        ),
        (
            "huge_key",
            b"GGUF" + struct.pack("<IQQQ", 3, 1, 1, 1 << 40),
            "string_length_limit",
        ),
        ("malformed_utf8", make_gguf(architecture=b"\xff"), "malformed_utf8"),
    ],
)
# REQ-REPORT-6572-ATTACKS / SCENARIO-REPORT-6572-REJECT:
# malformed inputs fail with stable reasons.
def test_malformed_fixtures_fail_closed(
    tmp_path: Path,
    name: str,
    payload: bytes,
    reason: str,
) -> None:
    path = tmp_path / name
    path.write_bytes(payload)

    with pytest.raises(GgufMetadataError, match=reason):
        read_gguf_metadata(path)


# REQ-REPORT-6572-PROVENANCE / SCENARIO-REPORT-6572-PROVENANCE:
# repository and revision derive from the cache symlink, not model metadata.
def test_binds_blob_to_repository_revision_link_and_hash(tmp_path: Path) -> None:
    blob, cache_root, trusted = cache_blob(tmp_path, make_gguf())

    row = bind_hf_cache_provenance(
        blob,
        repository_id="unsloth/fixture-GGUF",
        cache_root=cache_root,
        trusted_sha256=trusted,
    )

    assert row["valid"] is True
    assert row["repository_id"] == "unsloth/fixture-GGUF"
    assert row["revision"] == "revision-a"
    assert row["snapshot_filename"] == "fixture-Q4_K_M.gguf"
    assert row["symlink_target_matches_blob"] is True
    assert row["trusted_hash_matches_blob_key"] is True
    assert row["full_blob_rehash_performed"] is False


@pytest.mark.parametrize(
    ("repository_id", "trusted_suffix", "reason"),
    [
        ("unsloth/wrong-GGUF", None, "repository_mapping_missing"),
        ("unsloth/fixture-GGUF", "0" * 64, "trusted_hash_mismatch"),
    ],
)
# REQ-REPORT-6572-PROVENANCE: valid bytes cannot bypass repository binding.
def test_wrong_repository_or_hash_fails_closed(
    tmp_path: Path,
    repository_id: str,
    trusted_suffix: str | None,
    reason: str,
) -> None:
    blob, cache_root, trusted = cache_blob(tmp_path, make_gguf())
    supplied_hash = trusted if trusted_suffix is None else f"sha256:{trusted_suffix}"

    with pytest.raises(GgufMetadataError, match=reason):
        bind_hf_cache_provenance(
            blob,
            repository_id=repository_id,
            cache_root=cache_root,
            trusted_sha256=supplied_hash,
        )


# REQ-REPORT-6572-PROVENANCE: an external symlink alias has no cache authority.
def test_external_symlink_alias_fails_provenance(tmp_path: Path) -> None:
    blob, cache_root, trusted = cache_blob(tmp_path, make_gguf())
    alias = tmp_path / "renamed.gguf"
    alias.symlink_to(blob)

    with pytest.raises(GgufMetadataError, match="path_outside_repository_cache"):
        bind_hf_cache_provenance(
            alias,
            repository_id="unsloth/fixture-GGUF",
            cache_root=cache_root,
            trusted_sha256=trusted,
        )


# REQ-REPORT-6572-SHARDS / SCENARIO-REPORT-6572-SHARDS:
# a declared split must have the complete ordered snapshot set.
def test_complete_shard_set_is_bound_in_order(tmp_path: Path) -> None:
    first = make_gguf(split_no=0, split_count=2)
    second = make_gguf(split_no=1, split_count=2)
    blob, cache_root, trusted = cache_blob(
        tmp_path,
        first,
        filename="fixture-Q4_K_M-00001-of-00002.gguf",
    )
    repo_dir = cache_root / "models--unsloth--fixture-GGUF"
    second_digest = hashlib.sha256(second).hexdigest()
    second_blob = repo_dir / "blobs" / second_digest
    second_blob.write_bytes(second)
    second_link = repo_dir / "snapshots" / "revision-a" / "fixture-Q4_K_M-00002-of-00002.gguf"
    second_link.symlink_to(Path("../../blobs") / second_digest)

    row = build_gguf_admission_record(
        blob,
        repository_id="unsloth/fixture-GGUF",
        cache_root=cache_root,
        trusted_sha256=trusted,
        expected_architectures={"gemma4"},
    )

    assert row["admitted"] is True
    assert [item["shard_number"] for item in row["provenance"]["ordered_shards"]] == [1, 2]


@pytest.mark.parametrize(
    ("split_no", "split_count", "reason"),
    [
        (2, 2, "invalid_shard_index"),
        (0, 2, "partial_shard_set"),
    ],
)
# REQ-REPORT-6572-SHARDS: invalid metadata and partial sets fail closed.
def test_inconsistent_or_partial_shards_fail_closed(
    tmp_path: Path,
    split_no: int,
    split_count: int,
    reason: str,
) -> None:
    content = make_gguf(split_no=split_no, split_count=split_count)
    blob, cache_root, trusted = cache_blob(
        tmp_path,
        content,
        filename="fixture-Q4_K_M-00001-of-00002.gguf",
    )

    row = build_gguf_admission_record(
        blob,
        repository_id="unsloth/fixture-GGUF",
        cache_root=cache_root,
        trusted_sha256=trusted,
        expected_architectures={"gemma4"},
    )

    assert row["admitted"] is False
    assert reason in row["rejection_reasons"]


# REQ-REPORT-6572-MODEL: architecture mismatch cannot be laundered by repo ID.
def test_architecture_mismatch_rejects_valid_gguf(tmp_path: Path) -> None:
    blob, cache_root, trusted = cache_blob(tmp_path, make_gguf(architecture="clip"))

    row = build_gguf_admission_record(
        blob,
        repository_id="unsloth/fixture-GGUF",
        cache_root=cache_root,
        trusted_sha256=trusted,
        expected_architectures={"gemma4"},
    )

    assert row["admitted"] is False
    assert "architecture_mismatch" in row["rejection_reasons"]
    assert "not_language_model" in row["rejection_reasons"]


# REQ-REPORT-6572-MODEL: parser failures become reusable closed records.
def test_admission_record_contains_parser_rejection(tmp_path: Path) -> None:
    path = tmp_path / "bad"
    path.write_bytes(b"no")

    row = build_gguf_admission_record(
        path,
        repository_id="unsloth/fixture-GGUF",
        cache_root=tmp_path / "hub",
        trusted_sha256="sha256:" + "0" * 64,
        expected_architectures={"gemma4"},
    )

    assert row["admitted"] is False
    assert row["content_metadata"] is None
    assert "truncated_header" in row["rejection_reasons"]


# REQ-REPORT-6572-BOUNDED: low-level bounds fail before an oversized read.
def test_low_level_reader_rejects_negative_limit_and_short_physical_read() -> None:
    reader = gguf_metadata._BoundedReader(  # noqa: SLF001
        io.BytesIO(b"x"),
        file_size=2,
        max_header_bytes=2,
        max_string_bytes=2,
        max_array_elements=2,
    )
    with pytest.raises(GgufMetadataError, match="negative_length"):
        reader._check_span(-1)  # noqa: SLF001
    with pytest.raises(GgufMetadataError, match="truncated_header"):
        reader.read(2)


@pytest.mark.parametrize(
    ("name", "payload", "kwargs", "reason"),
    [
        ("missing", None, {}, "file_missing"),
        ("header_limit", make_gguf(), {"max_header_bytes": 10}, "header_read_limit"),
        (
            "metadata_limit",
            make_gguf(),
            {"max_metadata_pairs": 1},
            "metadata_count_limit",
        ),
        (
            "duplicate_key",
            make_gguf(extra_metadata=[("general.architecture", 8, "gemma4")]),
            {},
            "duplicate_metadata_key",
        ),
        (
            "bad_dimensions",
            make_gguf(tensor_dimensions=()),
            {},
            "tensor_dimension_limit",
        ),
        (
            "shape_overflow",
            make_gguf(tensor_dimensions=(2**63, 2)),
            {},
            "tensor_shape_overflow",
        ),
        (
            "bad_alignment",
            make_gguf(extra_metadata=[("general.alignment", 4, 3)]),
            {},
            "invalid_alignment",
        ),
        (
            "truncated_tensor_region",
            make_gguf(tensor_offset=1_000_000),
            {},
            "truncated_tensor_region",
        ),
        ("missing_architecture", make_gguf(architecture=""), {}, "architecture_missing"),
        ("bad_file_type", make_gguf(file_type=999), {}, "file_type_unsupported"),
    ],
)
# REQ-REPORT-6572-ATTACKS: all declared count and shape limits fail closed.
def test_additional_header_attacks_fail_closed(
    tmp_path: Path,
    name: str,
    payload: bytes | None,
    kwargs: dict,
    reason: str,
) -> None:
    path = tmp_path / name
    if payload is not None:
        path.write_bytes(payload)
    with pytest.raises(GgufMetadataError, match=reason):
        read_gguf_metadata(path, **kwargs)


@pytest.mark.parametrize(
    ("encoded_value", "reason"),
    [
        (struct.pack("<IQ", 9, 0), "unsupported_array_type"),
        (struct.pack("<IQ", 8, 2_000_001), "array_length_limit"),
        (b"", "unsupported_metadata_type"),
    ],
)
# REQ-REPORT-6572-ATTACKS: unknown metadata encodings cannot skip validation.
def test_unknown_metadata_types_fail_closed(
    tmp_path: Path,
    encoded_value: bytes,
    reason: str,
) -> None:
    value_type = 13 if reason == "unsupported_metadata_type" else 9
    payload = (
        b"GGUF"
        + struct.pack("<IQQ", 3, 1, 1)
        + _string("attack")
        + struct.pack("<I", value_type)
        + encoded_value
    )
    path = tmp_path / reason
    path.write_bytes(payload)
    with pytest.raises(GgufMetadataError, match=reason):
        read_gguf_metadata(path)


# REQ-REPORT-6572-PROVENANCE: absent and malformed cache support files stay closed.
def test_cache_provenance_handles_missing_snapshot_refs_and_broken_links(tmp_path: Path) -> None:
    content = make_gguf()
    blob, cache_root, trusted = cache_blob(tmp_path, content)
    repo_dir = cache_root / "models--unsloth--fixture-GGUF"
    (repo_dir / "snapshots" / "revision-a" / "broken.gguf").symlink_to("missing")
    (repo_dir / "refs" / "invalid").write_bytes(b"\xff")
    row = bind_hf_cache_provenance(
        blob,
        repository_id="unsloth/fixture-GGUF",
        cache_root=cache_root,
        trusted_sha256=trusted,
    )
    assert row["valid"] is True
    assert gguf_metadata._refs_for_revision(tmp_path / "absent", "none") == []  # noqa: SLF001

    no_snapshot_repo = cache_root / "models--unsloth--no-snapshot-GGUF"
    digest = hashlib.sha256(content).hexdigest()
    no_snapshot_blob = no_snapshot_repo / "blobs" / digest
    no_snapshot_blob.parent.mkdir(parents=True)
    no_snapshot_blob.write_bytes(content)
    with pytest.raises(GgufMetadataError, match="repository_mapping_missing"):
        bind_hf_cache_provenance(
            no_snapshot_blob,
            repository_id="unsloth/no-snapshot-GGUF",
            cache_root=cache_root,
            trusted_sha256=f"sha256:{digest}",
        )


# REQ-REPORT-6572-PROVENANCE: broken or non-content-addressed blob targets fail.
def test_cache_target_failures_are_explicit(tmp_path: Path) -> None:
    cache_root = tmp_path / "hub"
    repo_dir = cache_root / "models--unsloth--fixture-GGUF"
    blobs = repo_dir / "blobs"
    blobs.mkdir(parents=True)
    broken = blobs / ("a" * 64)
    broken.symlink_to("missing")
    with pytest.raises(GgufMetadataError, match="cache_path_missing"):
        bind_hf_cache_provenance(
            broken,
            repository_id="unsloth/fixture-GGUF",
            cache_root=cache_root,
            trusted_sha256="sha256:" + "a" * 64,
        )
    invalid = blobs / "not-a-hash"
    invalid.write_bytes(make_gguf())
    with pytest.raises(GgufMetadataError, match="blob_target_invalid"):
        bind_hf_cache_provenance(
            invalid,
            repository_id="unsloth/fixture-GGUF",
            cache_root=cache_root,
            trusted_sha256="sha256:not-a-hash",
        )


@pytest.mark.parametrize(
    ("selected_no", "selected_count", "filename", "reason"),
    [
        (None, None, "fixture-00001-of-00002.gguf", "shard_filename_metadata_mismatch"),
        (0, 2, "fixture.gguf", "shard_filename_metadata_mismatch"),
        (1, 2, "fixture-00001-of-00002.gguf", "shard_order_mismatch"),
    ],
)
# REQ-REPORT-6572-SHARDS: filename and content shard claims must agree.
def test_shard_filename_mismatches_fail_closed(
    tmp_path: Path,
    selected_no: int | None,
    selected_count: int | None,
    filename: str,
    reason: str,
) -> None:
    content = make_gguf(split_no=selected_no, split_count=selected_count)
    blob, cache_root, trusted = cache_blob(tmp_path, content, filename=filename)
    row = build_gguf_admission_record(
        blob,
        repository_id="unsloth/fixture-GGUF",
        cache_root=cache_root,
        trusted_sha256=trusted,
        expected_architectures={"gemma4"},
    )
    assert reason in row["rejection_reasons"]


# REQ-REPORT-6572-SHARDS: every shard repeats compatible content metadata.
def test_inconsistent_second_shard_content_fails_closed(tmp_path: Path) -> None:
    first = make_gguf(split_no=0, split_count=2)
    second = make_gguf(architecture="qwen35moe", split_no=1, split_count=2)
    blob, cache_root, trusted = cache_blob(
        tmp_path,
        first,
        filename="fixture-00001-of-00002.gguf",
    )
    repo_dir = cache_root / "models--unsloth--fixture-GGUF"
    digest = hashlib.sha256(second).hexdigest()
    second_blob = repo_dir / "blobs" / digest
    second_blob.write_bytes(second)
    (repo_dir / "snapshots" / "revision-a" / "ignore.txt").write_text("ignored")
    (repo_dir / "snapshots" / "revision-a" / "other-00001-of-00003.gguf").symlink_to(
        Path("../../blobs") / digest
    )
    (repo_dir / "snapshots" / "revision-a" / "fixture-00001-of-00003.gguf").symlink_to(
        Path("../../blobs") / digest
    )
    (repo_dir / "snapshots" / "revision-a" / "fixture-00002-of-00002.gguf").symlink_to(
        Path("../../blobs") / digest
    )
    row = build_gguf_admission_record(
        blob,
        repository_id="unsloth/fixture-GGUF",
        cache_root=cache_root,
        trusted_sha256=trusted,
        expected_architectures={"gemma4"},
    )
    assert "inconsistent_shard_metadata" in row["rejection_reasons"]


@pytest.mark.parametrize(
    ("second_kwargs", "second_file_type"),
    [
        ({"split_no": 1, "split_count": 3}, 15),
        ({"split_no": 0, "split_count": 2}, 15),
        ({"split_no": 1, "split_count": 2}, 32),
    ],
)
# REQ-REPORT-6572-SHARDS: count, number, and file type repeat on every shard.
def test_other_inconsistent_shard_fields_fail_closed(
    tmp_path: Path,
    second_kwargs: dict,
    second_file_type: int,
) -> None:
    first = make_gguf(split_no=0, split_count=2)
    second = make_gguf(file_type=second_file_type, **second_kwargs)
    blob, cache_root, trusted = cache_blob(
        tmp_path,
        first,
        filename="fixture-00001-of-00002.gguf",
    )
    repo_dir = cache_root / "models--unsloth--fixture-GGUF"
    digest = hashlib.sha256(second).hexdigest()
    (repo_dir / "blobs" / digest).write_bytes(second)
    (repo_dir / "snapshots" / "revision-a" / "fixture-00002-of-00002.gguf").symlink_to(
        Path("../../blobs") / digest
    )
    row = build_gguf_admission_record(
        blob,
        repository_id="unsloth/fixture-GGUF",
        cache_root=cache_root,
        trusted_sha256=trusted,
        expected_architectures={"gemma4"},
    )
    assert "inconsistent_shard_metadata" in row["rejection_reasons"]


# REQ-REPORT-6572-SHARDS: a shard symlink cannot leave the repository blob set.
def test_shard_target_outside_repository_fails_closed(tmp_path: Path) -> None:
    first = make_gguf(split_no=0, split_count=2)
    second = make_gguf(split_no=1, split_count=2)
    blob, cache_root, trusted = cache_blob(
        tmp_path,
        first,
        filename="fixture-00001-of-00002.gguf",
    )
    outside = tmp_path / hashlib.sha256(second).hexdigest()
    outside.write_bytes(second)
    repo_dir = cache_root / "models--unsloth--fixture-GGUF"
    second_link = repo_dir / "snapshots" / "revision-a" / "fixture-00002-of-00002.gguf"
    second_link.symlink_to(outside)
    row = build_gguf_admission_record(
        blob,
        repository_id="unsloth/fixture-GGUF",
        cache_root=cache_root,
        trusted_sha256=trusted,
        expected_architectures={"gemma4"},
    )
    assert "shard_target_outside_repository" in row["rejection_reasons"]
