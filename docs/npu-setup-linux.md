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

## ONNX / VitisAI path (if you want it anyway)

If you do need the ONNX path — for instance to run `Exp 511`'s
`NPUEntropyProbe` exactly as written, rather than porting it to
`pyxrt` — the current best-effort recipe is:

1. Clone AMD's xdna-driver userspace examples for current XRT version
   reference: `git clone https://github.com/amd/xdna-driver`.
2. Clone ONNX Runtime source: `git clone --recursive https://github.com/microsoft/onnxruntime`.
3. Configure with `./build.sh --config Release --use_vitisai --build_wheel --parallel`.
4. Expect the configure step to fail looking for VOE headers. You will
   need AMD's Vitis AI runtime source tree alongside — the private
   parts are downloadable for registered developers from
   `ryzenai.docs.amd.com` but require accepting an EULA.

Given the closed bits, the Carnot roadmap explicitly plans to rewrite
`NPUEntropyProbe` in a future milestone to call `pyxrt` directly (and
optionally wrap the result to *look* like an ONNX Runtime session so
the rest of the pipeline code does not change). The ONNX path remains a
documented option but is not a blocker for Phase-2 hardware experiments.

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
