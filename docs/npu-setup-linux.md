# AMD XDNA NPU setup on Linux (Arch / CachyOS / Ubuntu)

This document explains how to get an AMD Ryzen AI XDNA NPU (Phoenix, Hawk
Point, Strix Point, Strix Halo, Krackan) reachable from Python on a Linux
host. It was written after the milestone 2026.04.38 investigation found
that the Exp 511 `NPUEntropyProbe` returned `npu_not_available` on a host
that genuinely *did* have a functional NPU — the probe's one check was
"does `onnxruntime` list a `VitisAIExecutionProvider`" and that provider
is essentially never present in the stock upstream `onnxruntime` on
Linux. Five other layers of the stack were fine; only the ONNX-side
wrapper was missing.

This document walks through all six layers, shows what "fine" looks like
at each, and gives the fix for the common failure modes.

## TL;DR

```bash
# 1. Install the kernel driver userspace (Arch / CachyOS):
sudo pacman -S xrt xrt-plugin-amdxdna

# 2. Raise the memlock limit so XRT can mmap the NPU BAR (64 MiB):
#    AMD's Ubuntu .deb installs an equivalent file automatically.
sudo tee /etc/security/limits.d/xrt.conf <<'EOF'
@render   soft   memlock   unlimited
@render   hard   memlock   unlimited
EOF

# 3. Re-login (PAM only re-reads limits.d on new sessions).

# 4. Verify:
/usr/bin/python3 scripts/verify_npu.py
#    Expected honest_verdict: "npu_fully_available_pyxrt_only"
```

That gets you the `pyxrt`-direct path working. The ONNX+VitisAI path is
intentionally *not* part of the TL;DR because AMD's official Linux wheels
for the VitisAI execution provider are not publicly available as of
2026-Q2; see the [ONNX path](#onnx--vitisai-path) section below for the
current state.

## The six layers of the NPU stack

| Layer | Component | Where it comes from (Arch) | Failure symptom |
|-------|-----------|----------------------------|-----------------|
| 1 | Hardware (PCI XDNA device) | Ryzen AI chip | `lspci` shows no `[1022:17f0]` entry |
| 2 | Kernel driver + firmware | `amdxdna-dkms` (AUR) + `linux-firmware` | `/dev/accel/accel0` missing |
| 3 | XRT userspace libs | `xrt` + `xrt-plugin-amdxdna` (extra repo) | `libxrt_driver_xdna.so.2` missing |
| 4 | `pyxrt` Python bindings | Bundled with `xrt` package | `import pyxrt` fails |
| 5 | RLIMIT_MEMLOCK >= 64 MiB | PAM via `/etc/security/limits.d/xrt.conf` | XRT `mmap` fails with EAGAIN |
| 6 | ONNX+VitisAI provider (optional) | AMD's gated Ryzen AI SW bundle | `onnxruntime.get_available_providers()` lacks `VitisAIExecutionProvider` |

`scripts/verify_npu.py` exercises all six layers independently and emits
`results/npu_stack_verification.json` classifying the state with one of
the following `honest_verdict` values:

- `npu_fully_available` — all six layers pass.
- `npu_fully_available_pyxrt_only` — layers 1-5 pass; only layer 6 (ONNX)
  is absent. This is the realistic "everything works" state on Linux today.
- `npu_blocked_memlock` — 1-4 pass but 5 fails. One-line fix below.
- `npu_userspace_missing` — 1-2 pass, 3-4 fail. Install `xrt` +
  `xrt-plugin-amdxdna`.
- `npu_kernel_missing` — layer 1 passes but layer 2 fails. Driver not
  attached; check `lsmod | grep amdxdna`, `dmesg | grep amdxdna`, and
  firmware presence at `/lib/firmware/amdnpu/*/npu*.sbin`.
- `npu_hardware_absent` — no XDNA PCI device at all. Wrong CPU.

## Per-layer details

### Layer 1: hardware detection

```bash
lspci -nn | grep -i '\[1022:17f0\]'
# Expect: c2:00.1 Signal processing controller [1180]: Advanced Micro Devices, Inc. [AMD] Strix/Krackan/Strix Halo Neural Processing Unit [1022:17f0] (rev 10)
```

Device ID `1022:17f0` is the XDNA2 engine used by Strix Point, Strix Halo,
and Krackan. Earlier Phoenix (XDNA1) parts have a different PCI ID.

### Layer 2: kernel driver + firmware

On a modern Linux kernel (>= 6.11) the `amdxdna` driver is in-tree.
Older kernels need the AUR package `amdxdna-dkms`. Firmware blobs live
in `/lib/firmware/amdnpu/*/npu*.sbin` (shipped with the `linux-firmware`
package in recent Arch releases). On a healthy boot `dmesg` shows:

```
amdxdna 0000:c2:00.1: [drm] Load firmware amdnpu/17f0_10/npu_7.sbin
amdxdna 0000:c2:00.1: enabling device (0000 -> 0002)
[drm] Initialized amdxdna_accel_driver 0.6.0 for 0000:c2:00.1 on minor 0
```

and the device node is at `/dev/accel/accel0` (not `/dev/amdxdna*` —
modern XDNA uses the DRM accel subsystem). Permissions are usually
`crw-rw-rw- root:render`; users in the `render` group get access for
free.

### Layer 3: XRT userspace

```bash
sudo pacman -S xrt xrt-plugin-amdxdna
```

This provides the Xilinx Runtime (`libxrt_core.so`, `libxrt_coreutil.so`,
`libxrt++.so`) plus the XDNA-specific backend (`libxrt_driver_xdna.so.2`).
The command-line tool `xrt-smi` is in the `xrt` package; try:

```bash
sudo xrt-smi examine
```

(`sudo` is temporary — it's only needed until layer 5 is fixed.) Expected
output includes `NPU Firmware Version : 1.1.2.64` and `RyzenAI-npu4`.

### Layer 4: pyxrt Python bindings

The `xrt` Arch package drops `pyxrt.cpython-<abi>-x86_64-linux-gnu.so`
into `/usr/lib/python3.<X>/site-packages/`. Important: it is
ABI-linked to the **system** Python version, so `import pyxrt` only
works from `/usr/bin/python3`, not from a venv that uses a different
Python minor version. The `scripts/verify_npu.py` script uses
`#!/usr/bin/python3` for this reason. If you need `pyxrt` inside a venv,
either (a) create the venv against the same system Python
(`python3 -m venv --system-site-packages .venv`) or (b) rebuild `pyxrt`
from the xrt source tree against the venv's Python.

### Layer 5: memlock limit

This is the most common silent blocker. XRT memory-maps ~64 MiB of the
NPU's BAR region through `/dev/accel/accel0`, and the default PAM
`RLIMIT_MEMLOCK` on Arch is 8 MiB. mmap returns `EAGAIN` / "Resource
temporarily unavailable" and both `xrt-smi examine` and `pyxrt.device(0)`
fail with the same error:

```
mmap(addr=..., len=67108864, prot=3, flags=8209, offset=4294967296)
failed (err=-11): Resource temporarily unavailable
```

AMD's Ubuntu .deb installs `/etc/security/limits.d/xrt.conf` automatically;
the Arch `xrt` package omits it. Install it manually:

```bash
sudo tee /etc/security/limits.d/xrt.conf <<'EOF'
@render   soft   memlock   unlimited
@render   hard   memlock   unlimited
EOF
```

Then **re-login** — PAM only re-reads `limits.d` on new login sessions.
Opening a new terminal in an existing graphical session may or may not
pick up the new limits depending on your display manager (some inherit
from the login that started the session, some don't). The surest way is
`ssh localhost` after applying the file.

Verify with:

```bash
prlimit --memlock
# Expect: UNLIMITED UNLIMITED bytes
```

### Layer 6: ONNX Runtime with VitisAI EP (optional, Linux gap)

AMD's **VitisAI Execution Provider** is a patched build of `onnxruntime`
that calls into XRT through the VOE (Vitis ONNX EP) runtime. It is
shipped as part of AMD's **Ryzen AI Software** bundle. As of 2026-Q2
this bundle is Windows-first; the Linux story is:

- No `onnxruntime-vitisai` package exists on PyPI, Arch repos, or AUR.
- AMD publishes Linux build instructions for `onnxruntime` with
  `--use_vitisai` at
  `github.com/amd/xdna-driver/tree/main/example` but the VOE runtime
  source it depends on is partially closed; expect to hit missing
  headers during configure.
- The conda env AMD publishes at
  `github.com/amd/RyzenAI-SW/tree/main/tutorial` installs only plain
  `onnxruntime` on Linux — no VitisAI EP.

**Practical consequence:** on Linux today you run NPU inference through
`pyxrt` directly (or through IREE's XDNA backend, or through the
`mlir-aie` IRON compiler) rather than through the ONNX graph executor.
`scripts/verify_npu.py` flags this layer as `expected_missing` and
does **not** downgrade the overall verdict when it is absent — layers
1-5 passing is the real success bar.

## ONNX / VitisAI path: blocked on AMD as of 2026-Q2

A direct web audit on 2026-04-19 found that the Linux ONNX+VitisAI path
is not merely inconvenient, it is **structurally blocked and
unsupported**:

- **AMD Ryzen AI Software 1.7.1** (the latest release, April 8, 2026)
  officially targets **Windows only**. The installation page at
  [ryzenai.docs.amd.com/en/latest/inst.html](https://ryzenai.docs.amd.com/en/latest/inst.html)
  opens with "This page covers Ryzen AI installation on Windows" and
  makes zero references to Linux throughout.
- **The `voe` Python package** (Vitis ONNX Execution graph passes)
  that the VitisAI EP requires is **entirely absent** from Linux
  x86_64 distributions. See
  [amd/RyzenAI-SW#341](https://github.com/amd/RyzenAI-SW/issues/341)
  — the reporter documents
  `ModuleNotFoundError: No module named 'voe'`, 100% CPU fallback
  (`[Vitis AI EP] No. of Operators : CPU 30`, zero NPU ops) even when
  the EP appears to initialize, and a pointer bug in the C++ config
  parser. The issue has been open for months with **no AMD response**.
- **Source-building `onnxruntime` with `--use_vitisai`** fails on
  modern toolchains. GCC 13/14 enforces stricter C++17/20 template
  deduction that the Vitis AI EP sources in `element_wise_ops.cc`
  do not satisfy. Documented in
  [amd/xdna-driver#1017](https://github.com/amd/xdna-driver/issues/1017)
  and
  [microsoft/onnxruntime#27097](https://github.com/microsoft/onnxruntime/issues/27097)
  — again with no AMD response, no patches, no ETA. The reporter
  of #1017 specifically frames the situation as a "split state": the
  kernel+hardware layer is fine but the user-space runtime fails to
  build.
- **HuggingFace Optimum-AMD** for Ryzen AI explicitly requires the
  same (missing) Vitis AI EP plus the `vaip_config.json` from the
  VOE package. Same dead-end. See
  [huggingface.co/docs/optimum/.../amd/ryzenai/overview](https://huggingface.co/docs/optimum/v1.27.0/en/amd/ryzenai/overview).
- **Requests for Linux binaries** (e.g.
  [amd/RyzenAI-SW#319](https://github.com/amd/RyzenAI-SW/issues/319),
  asking specifically for 1.6.1 Linux binaries on Krackan/XDNA2) have
  received no official response either.

**Practical consequence:** do not spend engineering time trying to
obtain or build a Linux `onnxruntime-vitisai` wheel until AMD publishes
one. There is no workaround that has been confirmed working by anyone
on any public forum as of this writing.

## Recommended Linux paths going forward

Given the ONNX+VitisAI gap is structural, the two paths that actually
work on Linux today are:

### 1. `pyxrt` direct (what `scripts/verify_npu.py` proves)

AMD's Xilinx Runtime Python bindings let you open the device, allocate
buffer objects, and push compiled kernels directly. It is low-level —
you manage memory and synchronisation yourself — but every call goes
through the supported XRT ABI that AMD ships in the Arch `xrt` and
Ubuntu `xrt-base` packages. Carnot's revised Exp 511 rerun plan uses
this path by adding a `backend="pyxrt"` mode to `NPUEntropyProbe`.

### 2. `mlir-aie` / IRON (AMD's open-source compiler path)

[Xilinx/mlir-aie](https://github.com/Xilinx/mlir-aie) is AMD's
officially-maintained MLIR-based toolchain for AI Engine devices,
including XDNA2 (Strix/Krackan). It is **actively maintained**
(v1.3.1 released March 2026, 2600+ commits on main) and explicitly
targets Ryzen AI NPUs. Two important caveats:

- It is a **close-to-metal toolkit**, not an end-to-end ONNX/PyTorch
  inference runtime. You author AIE designs in MLIR or via their
  IRON Python DSL, not by loading a `.onnx` file.
- The Python package installs as `mlir_aie`, with `llvm-aie` (Peano
  compiler) as a companion on PyPI.

Carnot's Exp 460 scaffolded the IRON install path. The next milestone
should expand that into a runnable kernel and a `backend="iron"` path
on `NPUEntropyProbe` so we have two independent Linux-native NPU
backends — `pyxrt` for direct device control, IRON for compiled
kernels.

### 3. Future: ONNX+VitisAI (revisit when AMD acts)

Track the three GitHub issues above and revisit when any of them
close with a working Linux fix. Signs to watch for:

- AMD publishing a Linux `voe-*.whl` on pypi.org or their own package
  index
- The `element_wise_ops.cc` GCC 14 fix being merged into AMD's
  onnxruntime fork
- A Ryzen AI Software release note explicitly adding Linux to the
  supported OS list

Until then, the ONNX path stays documented-but-not-attempted.

## Reference

- `scripts/verify_npu.py` — end-to-end six-layer check, writes
  `results/npu_stack_verification.json`.
- `/etc/security/limits.d/xrt.conf` — memlock fix (installed manually
  on Arch; automatic on AMD's Ubuntu .deb).
- `xrt-smi examine` — AMD's CLI probe; same mmap path as `pyxrt`, so
  it surfaces the same memlock error if layer 5 is wrong.
- PCI device ID `1022:17f0` corresponds to XDNA2 (Strix Point family).
- `/dev/accel/accel0` is the DRM accel subsystem node; users in the
  `render` group get r/w access by default.
- `openspec/change-proposals/env-hardening-and-reruns.md` — the broader
  conductor environment-hardening proposal that Exp 511's honest state
  feeds into.
