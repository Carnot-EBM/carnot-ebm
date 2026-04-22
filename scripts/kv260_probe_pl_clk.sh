#!/usr/bin/env bash
# kv260_probe_pl_clk.sh — Read-only probe of pl_clk0 state on the Kria PS.
#
# **Why this script exists:**
#   Exp 661 wedged the KV260 PS twice on first AXI access.  The Vivado
#   simulator (results/kv260_axi_simulation_v2.json) rejected the
#   "RTL AXI protocol deadlock" hypothesis for the hang.  The next ranked
#   hypothesis is that pl_clk0 is not actually running on-hardware despite
#   xmutil loadapp reporting success — the overlay's fclk_cfg fragment
#   might not have activated the clock, so the PL has no aclk, the slave
#   never ACKs, and the PS AXI master blocks forever.
#
# **What this does (read-only, no AXI access, no power-cycle risk):**
#   1. ssh into kria, confirm reachability.
#   2. Report what app is currently loaded via `xmutil listapps`.
#   3. Read the Zynq UltraScale+ CRL_APB.PL0_REF_CTRL register at
#      0xFF5E00C0 via a small Python mmap stub.  Decode:
#        - CLKACT (bit 24):      is the clock output enabled?
#        - DIVISOR0 (bits 13:8): first divisor
#        - DIVISOR1 (bits 21:16): second divisor
#        - SRCSEL (bits 2:0):    which PLL (0=IOPLL, 2=RPLL, 3=DPLL)
#      Compute the actual output frequency.
#   4. If `--load-app` is passed, load carnot_ising_v2_n64 first THEN probe.
#      Will only succeed if the bundle is present in /lib/firmware/xilinx/.
#      DOES NOT do any AXI access after load, so does not risk wedging.
#
# **Interpretation:**
#   - After boot, k26-starter-kits should be loaded (default Kria app).
#     That app uses pl_clk0 at 100 MHz, so probe should show CLKACT=1,
#     freq ≈ 100 MHz.  This is the sanity-check baseline.
#   - After `sudo xmutil loadapp carnot_ising_v2_n64`, probe should show
#     CLKACT=1, freq ≈ 40 MHz (the target set in build_bd.tcl
#     CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ={40}).
#     If CLKACT=0 OR freq is wildly different (e.g. 0): the overlay's
#     fclk_cfg fragment is the bug — that's the hang root cause and
#     we can fix the overlay.
#
# **Usage:**
#   scripts/kv260_probe_pl_clk.sh                 # probe current state
#   scripts/kv260_probe_pl_clk.sh --load-app      # load app, then probe
#
# Spec: RETRO-074 diagnosis, hypothesis #1 test.  Safe to run repeatedly.

set -euo pipefail

KRIA_HOST="${CARNOT_KRIA_HOST:-kria}"
APP_NAME="carnot_ising_v2_n64"
# Zynq UltraScale+ CRL_APB.PL0_REF_CTRL register.  From UG1085 table 10-3.
PL0_REF_CTRL_ADDR="0xFF5E00C0"

load_app=false
for arg in "$@"; do
    case "$arg" in
        --load-app) load_app=true ;;
        *) echo "unknown arg: $arg" >&2; exit 2 ;;
    esac
done

echo "=== ssh $KRIA_HOST reachability ==="
if ! ssh -o ConnectTimeout=10 -o BatchMode=yes "$KRIA_HOST" "uname -r" >/dev/null 2>&1; then
    echo "ERROR: ssh $KRIA_HOST failed.  Kria may be powered off or unreachable." >&2
    exit 1
fi
echo "OK: ssh $KRIA_HOST reachable"

echo ""
echo "=== currently-loaded PL app ==="
ssh "$KRIA_HOST" 'sudo xmutil listapps 2>&1 | head -15'

if "$load_app"; then
    echo ""
    echo "=== loading $APP_NAME ==="
    # Unload any currently-loaded app first; OK if this errors (no slot loaded).
    ssh "$KRIA_HOST" "sudo xmutil unloadapp 2>&1 || true" >/dev/null
    ssh "$KRIA_HOST" "sudo xmutil loadapp $APP_NAME 2>&1 | head -5"
fi

echo ""
echo "=== CRL_APB.PL0_REF_CTRL probe ($PL0_REF_CTRL_ADDR) ==="

# Inline Python via ssh: mmap /dev/mem at the register page, read the 32-bit
# register, decode fields, compute output frequency.  Requires root on kria
# (mmap of /dev/mem is privileged); we use sudo.
#
# Why Python: devmem2 is not installed on this kria image.  Python's mmap
# with a page-aligned offset is a 20-line equivalent.
ssh "$KRIA_HOST" "sudo python3 - <<'PYEOF'
import mmap, os, struct

REG_ADDR = 0xFF5E00C0
PAGE = 0x1000
page_base = REG_ADDR & ~(PAGE - 1)
page_off  = REG_ADDR - page_base

fd = os.open('/dev/mem', os.O_RDONLY | os.O_SYNC)
mm = mmap.mmap(fd, PAGE, mmap.MAP_SHARED, mmap.PROT_READ, offset=page_base)
raw = struct.unpack('<I', mm[page_off:page_off+4])[0]
mm.close(); os.close(fd)

# Decode per UG1085 table 10-3.
clkact    = (raw >> 24) & 0x1
divisor1  = (raw >> 16) & 0x3F
divisor0  = (raw >>  8) & 0x3F
srcsel    =  raw        & 0x7

# SRCSEL to PLL name.  Assumes typical Kria boot config (source PLL runs
# at 1500 MHz).  Absolute frequency depends on boot-time PLL config; this
# gives a BEST-EFFORT reading.  The binary CLKACT bit is what matters most.
src_name = {0: 'IOPLL', 2: 'RPLL', 3: 'DPLL'}.get(srcsel, f'unknown({srcsel})')
src_freq_mhz = 1500  # Zynq UltraScale+ default for IOPLL/RPLL
if divisor0 and divisor1:
    out_freq_mhz = src_freq_mhz / (divisor0 * divisor1)
else:
    out_freq_mhz = 0.0

print(f'raw_register:        0x{raw:08x}')
print(f'CLKACT (bit 24):     {clkact}   ({'enabled' if clkact else 'DISABLED'})')
print(f'DIVISOR0 (bits 13-8): {divisor0}')
print(f'DIVISOR1 (bits 21-16): {divisor1}')
print(f'SRCSEL (bits 2-0):   {srcsel} ({src_name})')
print(f'estimated_out_mhz:   {out_freq_mhz:.2f}')
print()
if clkact == 0:
    print('>>> CLKACT=0 : pl_clk0 output is DISABLED.  Any AXI access from')
    print('>>> the PS to the PL will block forever because the slave has no')
    print('>>> clock to run its state machine.  This IS the hang root cause.')
elif out_freq_mhz < 1 or out_freq_mhz > 500:
    print(f'>>> Suspicious out_freq_mhz={out_freq_mhz:.2f}, divisors may be set wrong.')
else:
    print(f'>>> pl_clk0 reports enabled at ~{out_freq_mhz:.1f} MHz.  Clock is NOT the hang cause.')
PYEOF"

echo ""
echo "=== probe complete ==="
