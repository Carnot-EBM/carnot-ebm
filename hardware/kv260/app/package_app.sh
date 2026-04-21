#!/usr/bin/env bash
#
# Package the Carnot Ising sampler as a Kria "app" for xmutil loadapp.
#
# Why this script exists:
#   xmutil loadapp <name> expects /lib/firmware/xilinx/<name>/ to contain:
#     1. <name>.bit.bin   — the PL bitstream in the bootgen-wrapped binary
#                            format (NOT raw Vivado .bit).  Required because
#                            the in-kernel FPGA manager (fpga_manager) reads
#                            this format via /sys/class/fpga_manager/fpga0/firmware.
#     2. <name>.dtbo      — compiled device-tree overlay adding the new PL
#                            peripherals to /proc/device-tree at load time.
#     3. shell.json       — minimal manifest describing the PL "slot".
#   Raw .bit loads will fail with "Load Error: -1" (as we saw in
#   results/experiment_661_kv260_n64_benchmark.json blocked_on_dfx_mgr_load_failure).
#
# What this script does:
#   1. Wraps hardware/kv260/app/carnot_ising.bif around the bitstream, calls
#      bootgen -arch zynqmp -process_bitstream bin to produce .bit.bin.
#   2. Compiles hardware/kv260/app/carnot_ising.dts with dtc -@ (symbols) to
#      produce .dtbo.
#   3. Stages the three files plus shell.json into a local bundle directory
#      hardware/kv260/app/build/<app_name>/ so they can be scp'd as a unit.
#
# Tools required:
#   - bootgen                   — part of Vivado/Vitis, at
#                                 /tools/Xilinx/2025.2.1/Vivado/bin/bootgen
#   - dtc                       — device-tree compiler.  Locally via
#                                 pacman -S dtc on CachyOS; on kria pre-installed.
#                                 If absent locally, pass CARNOT_DTC_ON_KRIA=1 to
#                                 compile the .dtbo on the Kria instead.
#
# Usage:
#   source /tools/Xilinx/2025.2.1/Vivado/settings64.sh
#   hardware/kv260/app/package_app.sh
#
# Output layout:
#   hardware/kv260/app/build/carnot_ising_v2_n64/
#     carnot_ising_v2_n64.bit.bin
#     carnot_ising_v2_n64.dtbo
#     shell.json
#
# Deploy:
#   scp -r hardware/kv260/app/build/carnot_ising_v2_n64 \
#       kria:/tmp/
#   ssh kria 'sudo mv /tmp/carnot_ising_v2_n64 /lib/firmware/xilinx/'
#   ssh kria 'sudo xmutil loadapp carnot_ising_v2_n64'

set -euo pipefail

APP_NAME="${1:-carnot_ising_v2_n64}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BIT_SRC="${REPO_ROOT}/output/carnot_ising_bd/carnot_ising_bd_wrapper.bit"
APP_SRC_DIR="${REPO_ROOT}/hardware/kv260/app"
BUILD_DIR="${APP_SRC_DIR}/build/${APP_NAME}"
BIF_FILE="${BUILD_DIR}/${APP_NAME}.bif"
BIT_BIN="${BUILD_DIR}/${APP_NAME}.bit.bin"
DTS_SRC="${APP_SRC_DIR}/carnot_ising.dts"
DTBO="${BUILD_DIR}/${APP_NAME}.dtbo"

[[ -f "${BIT_SRC}" ]] || { echo "ERROR: bitstream not found at ${BIT_SRC}" >&2; exit 1; }
[[ -f "${DTS_SRC}" ]] || { echo "ERROR: dts source not found at ${DTS_SRC}" >&2; exit 1; }

echo "=== [1/4] Create build directory ==="
mkdir -p "${BUILD_DIR}"
cp "${APP_SRC_DIR}/shell.json" "${BUILD_DIR}/shell.json"

echo "=== [2/4] Convert .bit -> .bit.bin via bootgen ==="
cat > "${BIF_FILE}" <<BIF
all:
{
    [destination_device = pl] ${BIT_SRC}
}
BIF

if ! command -v bootgen >/dev/null 2>&1; then
    echo "ERROR: bootgen not on PATH.  Run:" >&2
    echo "       source /tools/Xilinx/2025.2.1/Vivado/settings64.sh" >&2
    exit 1
fi

bootgen -arch zynqmp -process_bitstream bin -image "${BIF_FILE}" -w
# bootgen writes <bitname>.bit.bin next to the original .bit (NOT in BUILD_DIR).
# Move it into the build dir so the final bundle is self-contained.
GEN_BIN="${BIT_SRC}.bin"
if [[ -f "${GEN_BIN}" ]]; then
    mv "${GEN_BIN}" "${BIT_BIN}"
fi
[[ -f "${BIT_BIN}" ]] || { echo "ERROR: bootgen did not produce ${BIT_BIN}" >&2; exit 1; }
echo "    produced ${BIT_BIN} ($(stat -c%s "${BIT_BIN}") bytes)"

echo "=== [3/4] Compile .dts -> .dtbo ==="
if [[ "${CARNOT_DTC_ON_KRIA:-0}" == "1" ]]; then
    # Fallback: compile on Kria where dtc is always present.
    echo "    (CARNOT_DTC_ON_KRIA=1; compiling on kria)"
    scp "${DTS_SRC}" "kria:/tmp/carnot_ising.dts"
    ssh kria "dtc -@ -O dtb -o /tmp/${APP_NAME}.dtbo /tmp/carnot_ising.dts"
    scp "kria:/tmp/${APP_NAME}.dtbo" "${DTBO}"
else
    if ! command -v dtc >/dev/null 2>&1; then
        echo "ERROR: dtc not on PATH.  Install (pacman -S dtc) or re-run with" >&2
        echo "       CARNOT_DTC_ON_KRIA=1 to compile on the Kria instead." >&2
        exit 1
    fi
    dtc -@ -O dtb -o "${DTBO}" "${DTS_SRC}"
fi
echo "    produced ${DTBO} ($(stat -c%s "${DTBO}") bytes)"

echo "=== [4/4] Bundle ==="
ls -la "${BUILD_DIR}"
echo ""
echo "Bundle ready at ${BUILD_DIR}"
echo "Deploy commands:"
echo "  scp -r ${BUILD_DIR} kria:/tmp/"
echo "  ssh kria 'sudo mv /tmp/${APP_NAME} /lib/firmware/xilinx/'"
echo "  ssh kria 'sudo xmutil loadapp ${APP_NAME}'"
