"""Read bounded GGUF metadata and bind it to Hugging Face cache provenance.

The local cache can store a large model under a hash-only blob name. A path
suffix therefore cannot prove that the file is a GGUF model. This module reads
only the GGUF header and tensor descriptors. It then binds the blob to an exact
repository snapshot link and an existing trusted hash. Tensor payload bytes are
never read.

Spec: REQ-REPORT-6572 and SCENARIO-REPORT-6572-HASH-BLOB through
SCENARIO-REPORT-6572-SHARDS.
"""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
import re
import struct
from typing import Any, BinaryIO


JsonDict = dict[str, Any]

GGUF_MAGIC = b"GGUF"
SUPPORTED_VERSIONS = frozenset({2, 3})
DEFAULT_MAX_HEADER_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_METADATA_PAIRS = 1_000_000
DEFAULT_MAX_STRING_BYTES = 16 * 1024 * 1024
DEFAULT_MAX_ARRAY_ELEMENTS = 2_000_000
DEFAULT_MAX_TENSORS = 1_000_000
MAX_TENSOR_DIMENSIONS = 4

_SCALAR_FORMATS = {
    0: "<B",
    1: "<b",
    2: "<H",
    3: "<h",
    4: "<I",
    5: "<i",
    6: "<f",
    7: "<B",
    10: "<Q",
    11: "<q",
    12: "<d",
}
_TYPE_NAMES = {
    0: "uint8",
    1: "int8",
    2: "uint16",
    3: "int16",
    4: "uint32",
    5: "int32",
    6: "float32",
    7: "bool",
    8: "string",
    9: "array",
    10: "uint64",
    11: "int64",
    12: "float64",
}
_FILE_TYPE_NAMES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    7: "Q8_0",
    8: "Q5_0",
    9: "Q5_1",
    10: "Q2_K",
    11: "Q3_K_S",
    12: "Q3_K_M",
    13: "Q3_K_L",
    14: "Q4_K_S",
    15: "Q4_K_M",
    16: "Q5_K_S",
    17: "Q5_K_M",
    18: "Q6_K",
    19: "IQ2_XXS",
    20: "IQ2_XS",
    21: "Q2_K_S",
    22: "IQ3_XS",
    23: "IQ3_XXS",
    24: "IQ1_S",
    25: "IQ4_NL",
    26: "IQ3_S",
    27: "IQ3_M",
    28: "IQ2_S",
    29: "IQ2_M",
    30: "IQ4_XS",
    31: "IQ1_M",
    32: "BF16",
    36: "TQ1_0",
    37: "TQ2_0",
    38: "MXFP4_MOE",
    39: "NVFP4",
    40: "Q1_0",
}
_PROJECTOR_ARCHITECTURES = frozenset({"clip", "llava", "minicpmv", "mtp", "projector"})
_SHARD_NAME = re.compile(r"^(?P<prefix>.+)-(?P<number>\d{5})-of-(?P<count>\d{5})\.gguf$")
_HASH_NAME = re.compile(r"^[0-9a-f]{64}$")


class GgufMetadataError(ValueError):
    """A stable fail-closed GGUF parsing or provenance error."""

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        self.detail = detail
        message = reason if not detail else f"{reason}: {detail}"
        super().__init__(message)


class _BoundedReader:
    """Track physical reads and reject header offsets beyond a fixed budget."""

    def __init__(
        self,
        handle: BinaryIO,
        *,
        file_size: int,
        max_header_bytes: int,
        max_string_bytes: int,
        max_array_elements: int,
    ) -> None:
        self.handle = handle
        self.file_size = file_size
        self.max_header_bytes = max_header_bytes
        self.max_string_bytes = max_string_bytes
        self.max_array_elements = max_array_elements
        self.physical_bytes_read = 0
        self.maximum_offset_inspected = 0

    @property
    def offset(self) -> int:
        return self.handle.tell()

    def _check_span(self, size: int) -> None:
        if size < 0:
            raise GgufMetadataError("negative_length")
        end = self.offset + size
        if end > self.max_header_bytes:
            raise GgufMetadataError("header_read_limit", f"offset={end}")
        if end > self.file_size:
            raise GgufMetadataError("truncated_header", f"offset={end}")
        self.maximum_offset_inspected = max(self.maximum_offset_inspected, end)

    def read(self, size: int) -> bytes:
        self._check_span(size)
        data = self.handle.read(size)
        self.physical_bytes_read += len(data)
        if len(data) != size:
            raise GgufMetadataError("truncated_header", f"wanted={size}, got={len(data)}")
        return data

    def skip(self, size: int) -> None:
        self._check_span(size)
        self.handle.seek(size, os.SEEK_CUR)

    def unpack(self, fmt: str) -> Any:
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self.read(size))[0]

    def string(self, *, decode: bool = True) -> tuple[str | None, int, int]:
        length_offset = self.offset
        length = int(self.unpack("<Q"))
        if length > self.max_string_bytes:
            raise GgufMetadataError("string_length_limit", f"length={length}")
        value_offset = self.offset
        if not decode:
            self.skip(length)
            return None, value_offset, self.offset
        raw = self.read(length)
        try:
            value = raw.decode("utf-8", "strict")
        except UnicodeDecodeError as exc:
            raise GgufMetadataError("malformed_utf8", f"offset={value_offset}") from exc
        return value, length_offset, self.offset


def _array_summary(reader: _BoundedReader, key: str) -> JsonDict:
    element_type = int(reader.unpack("<I"))
    count = int(reader.unpack("<Q"))
    if element_type not in _TYPE_NAMES or element_type == 9:
        raise GgufMetadataError("unsupported_array_type", f"key={key}, type={element_type}")
    if count > reader.max_array_elements:
        raise GgufMetadataError("array_length_limit", f"key={key}, count={count}")
    data_offset = reader.offset
    if element_type == 8:
        for _ in range(count):
            reader.string(decode=False)
    else:
        fmt = _SCALAR_FORMATS[element_type]
        reader.skip(struct.calcsize(fmt) * count)
    return {
        "element_type": _TYPE_NAMES[element_type],
        "element_count": count,
        "data_offset": data_offset,
        "data_end_offset": reader.offset,
    }


def _metadata_value(reader: _BoundedReader, value_type: int, key: str) -> Any:
    if value_type == 8:
        value, _, _ = reader.string(decode=True)
        return value
    if value_type == 9:
        return _array_summary(reader, key)
    fmt = _SCALAR_FORMATS.get(value_type)
    if fmt is None:
        raise GgufMetadataError("unsupported_metadata_type", f"key={key}, type={value_type}")
    value = reader.unpack(fmt)
    return bool(value) if value_type == 7 else value


def _align(offset: int, alignment: int) -> int:
    return offset + (-offset % alignment)


def read_gguf_metadata(
    path: str | Path,
    *,
    max_header_bytes: int = DEFAULT_MAX_HEADER_BYTES,
    max_metadata_pairs: int = DEFAULT_MAX_METADATA_PAIRS,
    max_string_bytes: int = DEFAULT_MAX_STRING_BYTES,
    max_array_elements: int = DEFAULT_MAX_ARRAY_ELEMENTS,
    max_tensors: int = DEFAULT_MAX_TENSORS,
) -> JsonDict:
    """Return content-derived model metadata without reading tensor payloads."""

    candidate = Path(path).expanduser()
    if not candidate.is_file():
        raise GgufMetadataError("file_missing", str(candidate))
    file_size = candidate.stat().st_size
    if file_size < 4:
        raise GgufMetadataError("truncated_header", f"file_size={file_size}")
    with candidate.open("rb") as handle:
        reader = _BoundedReader(
            handle,
            file_size=file_size,
            max_header_bytes=max_header_bytes,
            max_string_bytes=max_string_bytes,
            max_array_elements=max_array_elements,
        )
        magic = reader.read(4)
        if magic != GGUF_MAGIC:
            raise GgufMetadataError("invalid_magic", magic.hex())
        version = int(reader.unpack("<I"))
        if version not in SUPPORTED_VERSIONS:
            raise GgufMetadataError("unsupported_version", str(version))
        tensor_count = int(reader.unpack("<Q"))
        metadata_count = int(reader.unpack("<Q"))
        if tensor_count > max_tensors:
            raise GgufMetadataError("tensor_count_limit", str(tensor_count))
        if metadata_count > max_metadata_pairs:
            raise GgufMetadataError("metadata_count_limit", str(metadata_count))

        metadata: JsonDict = {}
        sources: JsonDict = {}
        for _ in range(metadata_count):
            key, key_offset, key_end = reader.string(decode=True)
            assert key is not None
            type_offset = reader.offset
            value_type = int(reader.unpack("<I"))
            value_offset = reader.offset
            value = _metadata_value(reader, value_type, key)
            if key in metadata:
                raise GgufMetadataError("duplicate_metadata_key", key)
            metadata[key] = value
            sources[key] = {
                "key_offset": key_offset,
                "key_end_offset": key_end,
                "type_offset": type_offset,
                "value_offset": value_offset,
                "value_end_offset": reader.offset,
                "value_type": _TYPE_NAMES.get(value_type, f"unknown:{value_type}"),
            }

        tensor_info_start = reader.offset
        maximum_tensor_offset = 0
        for index in range(tensor_count):
            reader.string(decode=False)
            dimensions = int(reader.unpack("<I"))
            if dimensions < 1 or dimensions > MAX_TENSOR_DIMENSIONS:
                raise GgufMetadataError(
                    "tensor_dimension_limit", f"tensor={index}, dimensions={dimensions}"
                )
            element_count = 1
            for _ in range(dimensions):
                dimension = int(reader.unpack("<Q"))
                if dimension < 1 or element_count > (2**63 - 1) // dimension:
                    raise GgufMetadataError("tensor_shape_overflow", f"tensor={index}")
                element_count *= dimension
            reader.unpack("<I")
            tensor_offset = int(reader.unpack("<Q"))
            maximum_tensor_offset = max(maximum_tensor_offset, tensor_offset)

        tensor_info_end = reader.offset
        alignment = int(metadata.get("general.alignment", 32))
        if alignment < 1 or alignment > 4096 or alignment & (alignment - 1):
            raise GgufMetadataError("invalid_alignment", str(alignment))
        data_offset = _align(tensor_info_end, alignment) if tensor_count else tensor_info_end
        required_minimum_file_size = (
            data_offset + maximum_tensor_offset + (1 if tensor_count else 0)
        )
        if required_minimum_file_size > file_size:
            raise GgufMetadataError(
                "truncated_tensor_region",
                f"required={required_minimum_file_size}, file_size={file_size}",
            )

    architecture = metadata.get("general.architecture")
    file_type = metadata.get("general.file_type")
    if not isinstance(architecture, str) or not architecture:
        raise GgufMetadataError("architecture_missing")
    if not isinstance(file_type, int) or file_type not in _FILE_TYPE_NAMES:
        raise GgufMetadataError("file_type_unsupported", str(file_type))
    tokens = metadata.get("tokenizer.ggml.tokens")
    token_count = int(tokens.get("element_count", 0)) if isinstance(tokens, Mapping) else 0
    tokenizer_model = metadata.get("tokenizer.ggml.model")
    if tensor_count == 0:
        raise GgufMetadataError("tokenizer_only")
    is_language_model = (
        architecture.lower() not in _PROJECTOR_ARCHITECTURES
        and isinstance(tokenizer_model, str)
        and bool(tokenizer_model)
        and token_count > 0
    )
    split_no = metadata.get("split.no")
    split_count = metadata.get("split.count")
    split_tensors_count = metadata.get("split.tensors.count")
    return {
        "path": str(candidate.absolute()),
        "magic": magic.decode("ascii"),
        "version": version,
        "architecture": architecture,
        "model_name": metadata.get("general.name"),
        "general_file_type": file_type,
        "quantization": _FILE_TYPE_NAMES[file_type],
        "tensor_count": tensor_count,
        "metadata_count": metadata_count,
        "is_language_model": is_language_model,
        "tokenizer_metadata": {
            "model": tokenizer_model,
            "pre": metadata.get("tokenizer.ggml.pre"),
            "token_count": token_count,
            "bos_token_id": metadata.get("tokenizer.ggml.bos_token_id"),
            "eos_token_id": metadata.get("tokenizer.ggml.eos_token_id"),
            "padding_token_id": metadata.get("tokenizer.ggml.padding_token_id"),
            "chat_template_present": bool(metadata.get("tokenizer.chat_template")),
        },
        "shard_metadata": {
            "split_no": split_no,
            "split_count": split_count,
            "split_tensors_count": split_tensors_count,
        },
        "bounded_read_receipt": {
            "file_size": file_size,
            "maximum_header_bytes": max_header_bytes,
            "physical_bytes_read": reader.physical_bytes_read,
            "maximum_offset_inspected": reader.maximum_offset_inspected,
            "tensor_info_start": tensor_info_start,
            "tensor_info_end": tensor_info_end,
            "data_offset": data_offset,
            "required_minimum_file_size": required_minimum_file_size,
            "tensor_payload_bytes_read": 0,
        },
        "field_provenance": {
            "magic": {"byte_offset": 0, "byte_end_offset": 4},
            "version": {"byte_offset": 4, "byte_end_offset": 8},
            "tensor_count": {"byte_offset": 8, "byte_end_offset": 16},
            "metadata_count": {"byte_offset": 16, "byte_end_offset": 24},
            "metadata_keys": sources,
            "quantization": {
                "source_key": "general.file_type",
                "source": sources.get("general.file_type"),
                "mapping": "llama.cpp llama_ftype enum",
            },
            "repository_id": "not derived from GGUF bytes; cache provenance required",
        },
    }


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _snapshot_links(repo_dir: Path, blob: Path) -> list[Path]:
    snapshots = repo_dir / "snapshots"
    if not snapshots.is_dir():
        return []
    matches = []
    for candidate in snapshots.rglob("*"):
        if candidate.is_symlink():
            try:
                if candidate.resolve(strict=True) == blob:
                    matches.append(candidate)
            except OSError:
                continue
    return sorted(matches)


def _refs_for_revision(repo_dir: Path, revision: str) -> list[str]:
    refs_dir = repo_dir / "refs"
    if not refs_dir.is_dir():
        return []
    refs = []
    for ref in refs_dir.rglob("*"):
        if ref.is_file() and not ref.is_symlink():
            try:
                if ref.read_text(encoding="utf-8").strip() == revision:
                    refs.append(ref.relative_to(refs_dir).as_posix())
            except (OSError, UnicodeDecodeError):
                continue
    return sorted(refs)


def _ordered_shards(
    selected_link: Path,
    selected_metadata: Mapping[str, Any],
    repo_dir: Path,
) -> list[JsonDict]:
    shard = selected_metadata.get("shard_metadata", {})
    shard = shard if isinstance(shard, Mapping) else {}
    split_count = int(shard.get("split_count") or 1)
    split_no_value = shard.get("split_no")
    split_no = int(split_no_value or 0)
    match = _SHARD_NAME.match(selected_link.name)
    if split_count <= 1:
        if match and int(match.group("count")) > 1:
            raise GgufMetadataError("shard_filename_metadata_mismatch")
        return [
            {
                "shard_number": 1,
                "shard_count": 1,
                "snapshot_path": str(selected_link),
                "blob_key": selected_link.resolve(strict=True).name,
            }
        ]
    if split_no < 0 or split_no >= split_count:
        raise GgufMetadataError("invalid_shard_index", f"split_no={split_no}")
    if not match or int(match.group("count")) != split_count:
        raise GgufMetadataError("shard_filename_metadata_mismatch")
    if int(match.group("number")) != split_no + 1:
        raise GgufMetadataError("shard_order_mismatch")

    candidates: dict[int, Path] = {}
    for path in selected_link.parent.iterdir():
        path_match = _SHARD_NAME.match(path.name)
        if not path_match or path_match.group("prefix") != match.group("prefix"):
            continue
        if int(path_match.group("count")) != split_count:
            continue
        number = int(path_match.group("number"))
        if number in candidates:  # pragma: no cover - one directory cannot contain duplicate names.
            raise GgufMetadataError("duplicate_shard", str(number))
        candidates[number] = path
    if set(candidates) != set(range(1, split_count + 1)):
        raise GgufMetadataError("partial_shard_set")

    rows = []
    for number in range(1, split_count + 1):
        link = candidates[number]
        blob = link.resolve(strict=True)
        if blob.parent != repo_dir / "blobs":
            raise GgufMetadataError("shard_target_outside_repository")
        metadata = read_gguf_metadata(link)
        other_shard = metadata["shard_metadata"]
        if int(other_shard.get("split_count") or 0) != split_count:
            raise GgufMetadataError("inconsistent_shard_metadata")
        if int(other_shard.get("split_no") or 0) != number - 1:
            raise GgufMetadataError("inconsistent_shard_metadata")
        if metadata["architecture"] != selected_metadata["architecture"]:
            raise GgufMetadataError("inconsistent_shard_metadata")
        if metadata["general_file_type"] != selected_metadata["general_file_type"]:
            raise GgufMetadataError("inconsistent_shard_metadata")
        rows.append(
            {
                "shard_number": number,
                "shard_count": split_count,
                "snapshot_path": str(link),
                "blob_key": blob.name,
                "split_no": other_shard.get("split_no"),
                "split_tensors_count": other_shard.get("split_tensors_count"),
            }
        )
    return rows


def bind_hf_cache_provenance(
    path: str | Path,
    *,
    repository_id: str,
    cache_root: str | Path | None = None,
    trusted_sha256: str,
    content_metadata: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Bind a local blob to repository cache structure and a trusted hash."""

    root = (
        Path(cache_root).expanduser().absolute()
        if cache_root is not None
        else (Path.home() / ".cache" / "huggingface" / "hub").absolute()
    )
    repo_dir = (root / f"models--{repository_id.replace('/', '--')}").absolute()
    lexical_path = Path(path).expanduser().absolute()
    if not _inside(lexical_path, repo_dir):
        if _inside(lexical_path, root):
            raise GgufMetadataError("repository_mapping_missing", repository_id)
        raise GgufMetadataError("path_outside_repository_cache", str(lexical_path))
    try:
        blob = lexical_path.resolve(strict=True)
    except OSError as exc:
        raise GgufMetadataError("cache_path_missing", str(lexical_path)) from exc
    blobs_dir = repo_dir / "blobs"
    if blob.parent != blobs_dir or not _HASH_NAME.fullmatch(blob.name):
        raise GgufMetadataError("blob_target_invalid", str(blob))
    expected_hash = f"sha256:{blob.name}"
    if trusted_sha256 != expected_hash:
        raise GgufMetadataError(
            "trusted_hash_mismatch", f"expected={expected_hash}, observed={trusted_sha256}"
        )
    links = _snapshot_links(repo_dir, blob)
    gguf_links = [link for link in links if link.name.lower().endswith(".gguf")]
    if not gguf_links:
        raise GgufMetadataError("repository_mapping_missing", repository_id)
    selected_link = gguf_links[0]
    relative = selected_link.relative_to(repo_dir / "snapshots")
    revision = relative.parts[0]
    metadata = content_metadata or read_gguf_metadata(blob)
    ordered_shards = _ordered_shards(selected_link, metadata, repo_dir)
    return {
        "valid": True,
        "repository_id": repository_id,
        "cache_repository_path": str(repo_dir),
        "revision": revision,
        "refs_pointing_to_revision": _refs_for_revision(repo_dir, revision),
        "snapshot_path": str(selected_link),
        "snapshot_filename": selected_link.name,
        "snapshot_symlink_target": os.readlink(selected_link),
        "resolved_blob_path": str(blob),
        "blob_key": blob.name,
        "trusted_sha256": trusted_sha256,
        "trusted_hash_matches_blob_key": True,
        "symlink_target_matches_blob": selected_link.resolve(strict=True) == blob,
        "ordered_shards": ordered_shards,
        "full_blob_rehash_performed": False,
        "field_provenance": {
            "repository_id": "expected repository argument and HF cache directory name",
            "revision": "snapshots/<revision> path component",
            "snapshot_filename": "snapshot symlink name",
            "snapshot_symlink_target": "os.readlink output",
            "trusted_sha256": "trusted upstream hash matched to HF blob key",
        },
    }


def build_gguf_admission_record(
    path: str | Path,
    *,
    repository_id: str,
    trusted_sha256: str,
    expected_architectures: set[str] | frozenset[str],
    cache_root: str | Path | None = None,
    max_header_bytes: int = DEFAULT_MAX_HEADER_BYTES,
) -> JsonDict:
    """Return one reusable closed admission record for a model or fixture."""

    reasons: list[str] = []
    try:
        metadata = read_gguf_metadata(path, max_header_bytes=max_header_bytes)
    except GgufMetadataError as exc:
        return {
            "repository_id": repository_id,
            "path": str(Path(path).expanduser().absolute()),
            "admitted": False,
            "content_metadata": None,
            "provenance": None,
            "rejection_reasons": [exc.reason],
            "error_detail": exc.detail,
        }
    if metadata["architecture"] not in expected_architectures:
        reasons.append("architecture_mismatch")
    if metadata["is_language_model"] is not True:
        reasons.append("not_language_model")
    try:
        provenance = bind_hf_cache_provenance(
            path,
            repository_id=repository_id,
            cache_root=cache_root,
            trusted_sha256=trusted_sha256,
            content_metadata=metadata,
        )
    except GgufMetadataError as exc:
        provenance = None
        reasons.append(exc.reason)
    return {
        "repository_id": repository_id,
        "path": str(Path(path).expanduser().absolute()),
        "admitted": not reasons,
        "content_metadata": metadata,
        "provenance": provenance,
        "rejection_reasons": reasons,
        "error_detail": "",
    }
