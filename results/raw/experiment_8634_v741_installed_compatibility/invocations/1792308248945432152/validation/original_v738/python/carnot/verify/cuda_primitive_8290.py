"""REQ-VERIFY-8290: isolate driver, runtime and native calls without neural weights.

Each API result remains a primitive. A failed initialization does not explain
its cause, and enumeration does not establish that a context can execute.
"""

from __future__ import annotations

import ctypes as c
import json
import os
from pathlib import Path
import subprocess
import uuid as uuid_module
from typing import Any

from carnot.gpu_lease_phase_journal import proc_start_ticks
from carnot.reporting.current_work_receipt import sha256_file

Json = dict[str, Any]
NATIVE_DEADLINE = 45.0
VARIABLES = (
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "CUDA_HOME",
    "CUDA_PATH",
)


def node_access(paths: list[Path]) -> list[Json]:
    """Read access without changing permissions, so missing nodes remain evidence."""
    return [
        dict(path=str(p), exists=p.exists(), read_write=os.access(p, os.R_OK | os.W_OK))
        for p in paths
    ]


def libraries() -> list[Json]:
    """Bind resolved loaded objects rather than a loader's possibly ambiguous name."""
    paths = {
        line.split()[-1]
        for line in Path("/proc/self/maps").read_text().splitlines()
        if "/" in line and any(s in line for s in ("libcuda", "libfixture", "libggml"))
    }
    return [
        dict(path=str(Path(p).resolve()), sha256=sha256_file(Path(p)))
        for p in sorted(paths)
        if Path(p).is_file()
    ]


def native(binary: str, result: Json) -> None:
    """Keep the native process identity and stderr even when its exit is zero."""
    command = [binary, "--list-devices"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    result.update(
        native_argv=command,
        native_pid=process.pid,
        native_pid_start_ticks=proc_start_ticks(process.pid),
    )
    try:
        out, err = process.communicate(timeout=NATIVE_DEADLINE)
    except subprocess.TimeoutExpired:
        process.kill()
        out, err = process.communicate()
        result["timed_out"] = True
    result.update(
        stdout=out.decode(errors="replace"),
        stderr=err.decode(errors="replace"),
        native_exit=process.returncode,
        native_binary=dict(path=binary, sha256=sha256_file(Path(binary))),
        native_cleanup=process.poll() is not None,
    )
    text = result["stdout"] + result["stderr"]
    result["native_compatible"] = (
        process.returncode == 0
        and not result["timed_out"]
        and "CUDA0" in text
        and "invalid device" not in text.lower()
    )


def probe(layer: str, library: str, uuid: str, binary: str) -> Json:
    """Stop at the first failed dependency while preserving every executed API code."""
    result: Json = dict(
        layer=layer,
        pid=os.getpid(),
        pid_start_ticks=proc_start_ticks(os.getpid()),
        environment={k: os.environ[k] for k in VARIABLES if k in os.environ},
        api_returns={},
        libraries=[],
        local_ordinal=None,
        device_count=None,
        context_copy_ready=False,
        native_compatible=False,
        timed_out=False,
        error=None,
        allocation_bytes=32,
        cleanup_passed=True,
    )
    context, memory = c.c_void_p(), c.c_ulonglong()
    lib: Any = None

    def call(name: str, *args: Any) -> bool:
        """Record the C return directly so errors cannot become inferred passes."""
        function = getattr(lib, name)
        function.restype = c.c_int
        code = int(function(*args))
        result["api_returns"][name] = code
        return code == 0

    try:
        if layer == "native":
            native(binary, result)
            return result
        lib = c.CDLL(library)
        result["libraries"] = libraries()
        if layer == "runtime":
            version, count = c.c_int(), c.c_int()
            call("cudaRuntimeGetVersion", c.byref(version))
            call("cudaGetDeviceCount", c.byref(count))
            result.update(runtime_version=version.value, device_count=count.value)
            return result
        if not call("cuInit", c.c_uint(0)):
            return result
        version, count = c.c_int(), c.c_int()
        call("cuDriverGetVersion", c.byref(version))
        result["driver_version"] = version.value
        if not call("cuDeviceGetCount", c.byref(count)):
            return result
        result["device_count"] = count.value
        selected = None
        result["devices"] = []
        for ordinal in range(count.value):
            device, ident = c.c_int(), (c.c_ubyte * 16)()
            if not call("cuDeviceGet", c.byref(device), c.c_int(ordinal)):
                return result
            if not call("cuDeviceGetUuid", c.byref(ident), device):
                return result
            actual = "GPU-" + str(uuid_module.UUID(bytes=bytes(ident)))
            result["devices"].append(dict(local_ordinal=ordinal, uuid=actual))
            if actual == uuid:
                selected = device
                result["local_ordinal"] = ordinal
        if selected is None:
            result["error"] = "permitted_uuid_not_enumerated"
            return result
        if not call("cuCtxCreate_v2", c.byref(context), c.c_uint(0), selected):
            context = c.c_void_p()
            return result
        if not call("cuMemAlloc_v2", c.byref(memory), c.c_size_t(32)):
            memory = c.c_ulonglong()
            return result
        source = (c.c_ubyte * 32)(*range(32))
        target = (c.c_ubyte * 32)()
        if not call("cuMemcpyHtoD_v2", memory, source, c.c_size_t(32)):
            return result
        if not call("cuMemcpyDtoH_v2", target, memory, c.c_size_t(32)):
            return result
        result.update(
            copy_source_hex=bytes(source).hex(),
            copy_target_hex=bytes(target).hex(),
            byte_copy_parity=bytes(source) == bytes(target),
        )
        result["context_copy_ready"] = result["byte_copy_parity"]
    except (OSError, AttributeError, ValueError) as error:
        result["error"] = str(error)
    finally:
        if memory.value:
            result["cleanup_passed"] &= call("cuMemFree_v2", memory)
        if context.value:
            result["cleanup_passed"] &= call("cuCtxDestroy_v2", context)
        result["context_copy_ready"] &= result["cleanup_passed"]
    return result


def main(argv: list[str]) -> int:
    """Keep diagnostic errors in JSON; a normal receipt can still describe a block."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda-probe", choices=["driver", "runtime", "native"], required=True)
    parser.add_argument("--library", default="libcuda.so.1")
    parser.add_argument("--uuid", default="")
    parser.add_argument("--binary", default="")
    args = parser.parse_args(argv)
    print(json.dumps(probe(args.cuda_probe, args.library, args.uuid, args.binary)), flush=True)
    return 0
