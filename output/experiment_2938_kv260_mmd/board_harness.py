#!/usr/bin/env python3
import glob
import json
import mmap
import os
import struct
import sys
import time

ADDR_CONTROL = 0x0000
ADDR_STATUS = 0x0004
ADDR_SPIN_COUNT = 0x0008
ADDR_BETA_FINAL = 0x001C
ADDR_BIAS_BASE = 0x1000
ADDR_ADJ_BASE = 0x2000
ADDR_COUPL_BASE = 0x6000
ADDR_SPOUT_BASE = 0xA010
STATUS_DONE_MASK = 0x4
DEFAULT_MAP_SIZE = 0x20000
SAMPLER_BASE_ADDR = 0xA0000000
POLL_TIMEOUT_S = 0.250


def _read_text(path, default=""):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError:
        return default


def _parse_int(text, default=0):
    try:
        return int(text, 0)
    except (TypeError, ValueError):
        return default


def _discover_uio_devices():
    devices = []
    for sys_path in sorted(glob.glob("/sys/class/uio/uio*")):
        name = os.path.basename(sys_path)
        devices.append(
            {
                "path": "/dev/" + name,
                "addr": _parse_int(_read_text(os.path.join(sys_path, "maps/map0/addr"))),
                "addr_hex": _read_text(os.path.join(sys_path, "maps/map0/addr")),
                "size": _parse_int(_read_text(os.path.join(sys_path, "maps/map0/size")), DEFAULT_MAP_SIZE),
                "name": _read_text(os.path.join(sys_path, "maps/map0/name")),
            }
        )
    return devices


def _select_sampler_uio(devices):
    for dev in devices:
        if dev["addr"] == SAMPLER_BASE_ADDR:
            return dev
    for dev in devices:
        if "ising" in dev["name"].lower() or "sampler" in dev["name"].lower():
            return dev
    for dev in devices:
        if dev["path"] == "/dev/uio0":
            return dev
    raise RuntimeError("no UIO device candidates found")


def _open_map(dev):
    size = max(dev.get("size") or DEFAULT_MAP_SIZE, DEFAULT_MAP_SIZE)
    fd = os.open(dev["path"], os.O_RDWR | os.O_SYNC)
    try:
        mm = mmap.mmap(fd, size, prot=mmap.PROT_READ | mmap.PROT_WRITE, flags=mmap.MAP_SHARED)
    except Exception:
        os.close(fd)
        raise
    return fd, mm


def _read_u32(mm, offset):
    return struct.unpack_from("<I", mm, offset)[0]


def _write_u32(mm, offset, value):
    struct.pack_into("<I", mm, offset, int(value) & 0xFFFFFFFF)


def _pack_i16(value):
    return int(value) & 0xFFFF


def _upload_problem(mm, problem):
    n = int(problem["n_spins"])
    max_degree = int(problem["upload"]["max_degree"])
    _write_u32(mm, ADDR_CONTROL, 0x2)
    _write_u32(mm, ADDR_CONTROL, 0x0)
    _write_u32(mm, ADDR_SPIN_COUNT, n)
    _write_u32(mm, ADDR_BETA_FINAL, int(problem.get("beta_final_q88", 0x0100)))

    for i, q in enumerate(problem["upload"]["h_q88"]):
        _write_u32(mm, ADDR_BIAS_BASE + 4 * i, _pack_i16(q))

    for i, row in enumerate(problem["upload"]["adjacency"]):
        for k, neighbor in enumerate(row):
            offset = 4 * (i * max_degree + k)
            _write_u32(mm, ADDR_ADJ_BASE + offset, _pack_i16(neighbor))
            _write_u32(mm, ADDR_COUPL_BASE + offset, _pack_i16(problem["upload"]["couplings_q88"][i][k]))


def _read_spins(mm, n):
    words = []
    for word_index in range((n + 31) // 32):
        words.append(_read_u32(mm, ADDR_SPOUT_BASE + 4 * word_index))
    spins = []
    for i in range(n):
        word = words[i // 32]
        spins.append(1 if ((word >> (i % 32)) & 1) else -1)
    return spins


def _energy(j_matrix, h_vector, spins):
    n = len(spins)
    total = 0.0
    for i in range(n):
        total -= float(h_vector[i]) * spins[i]
        for j in range(i + 1, n):
            total -= float(j_matrix[i][j]) * spins[i] * spins[j]
    return total


def _run_energy_trace(mm, problem, n_samples):
    n = int(problem["n_spins"])
    energies = []
    failed = 0
    for _ in range(int(n_samples)):
        _write_u32(mm, ADDR_CONTROL, 0x2)
        _write_u32(mm, ADDR_CONTROL, 0x0)
        _write_u32(mm, ADDR_CONTROL, 0x1)
        deadline = time.perf_counter() + POLL_TIMEOUT_S
        done = False
        while time.perf_counter() < deadline:
            if _read_u32(mm, ADDR_STATUS) & STATUS_DONE_MASK:
                done = True
                break
        if not done:
            failed += 1
            continue
        spins = _read_spins(mm, n)
        energies.append(_energy(problem["j_matrix"], problem["h_vector"], spins))
    if len(energies) != int(n_samples):
        raise RuntimeError(f"completed {len(energies)} of {n_samples} samples; failed={failed}")
    return energies


def main():
    problem_path = sys.argv[1]
    with open(problem_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    started = time.perf_counter()
    print("BOARD_HARNESS_START experiment_2938", flush=True)
    devices = _discover_uio_devices()
    print("UIO_DEVICES " + json.dumps(devices, sort_keys=True), flush=True)
    sampler_uio = _select_sampler_uio(devices)
    print("SELECTED_UIO " + json.dumps(sampler_uio, sort_keys=True), flush=True)
    fd, mm = _open_map(sampler_uio)
    try:
        energies_by_seed = {}
        for problem in payload["problems"]:
            _upload_problem(mm, problem)
            print(f"RUN seed={problem['random_seed']} n_samples={payload['n_samples']}", flush=True)
            energies_by_seed[str(problem["random_seed"])] = _run_energy_trace(
                mm, problem, int(payload["n_samples"])
            )
    finally:
        mm.close()
        os.close(fd)
    print(json.dumps(
        {
            "duration_s": time.perf_counter() - started,
            "selected_uio": sampler_uio["path"],
            "selected_uio_addr_hex": sampler_uio.get("addr_hex", ""),
            "uio_devices": [dev["path"] for dev in devices],
            "energies_by_seed": energies_by_seed,
        },
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
