# GateMate JTAG wiring via DirtyJTAG (reference for non-Olimex variants)

**Status note (2026-05-18):** The Carnot bench uses the **Olimex
GateMate A1-EVB-2M**, which has an onboard DirtyJTAG-compatible
programmer MCU built into the PCB. **No external wiring is required**
for that board — programming goes over the board's own USB-C cable.
This document is retained as a reference for two situations:

1. You move to a different GateMate evaluation board (non-Olimex
   variant, raw-chip dev board, custom carrier PCB) that exposes
   only a JTAG header without onboard programming.
2. You want to debug or replace the wiring between an external
   DirtyJTAG adapter and a GateMate board.

If you only have the Olimex GateMate A1-EVB-2M plugged in and the
USB-C cable connected, **skip to "Step 3: Verify the chain"** — the
onboard programmer handles everything internally.

## What the Olimex GateMate A1-EVB-2M actually does (current bench)

Verified 2026-05-18 via `/sys/bus/usb/devices/3-2.3/`:

```
manufacturer = Jean THOMAS
product      = DirtyJTAG
bNumInterfaces = 5
  3-2.3:1.0  class=ff      ← vendor-specific (raw JTAG endpoint)
  3-2.3:1.1  class=02 "DirtyJTAG CDC 0"  → /dev/ttyACM0
  3-2.3:1.2  class=0a      ← CDC data
  3-2.3:1.3  class=02 "DirtyJTAG CDC 1"  → /dev/ttyACM1
  3-2.3:1.4  class=0a      ← CDC data
```

The Olimex board ships with an integrated MCU (likely RP2040) flashed
with Jean Thomas's open-source DirtyJTAG firmware. That MCU drives the
GateMate fabric's JTAG pins via PCB traces. The host sees the result
as USB device `1209:c0ca` with the 5-interface composite layout above.

The udev rule at `/etc/udev/rules.d/99-fpga-boards.rules` (installed
2026-05-18) grants the `uucp` group raw USB access so
`openFPGALoader -c dirtyJtag` works without sudo.

---

The rest of this document covers the EXTERNAL-DirtyJTAG case, which
applies only to non-Olimex variants. Skip past it if you only have
the Olimex board.

## Why external wiring would be needed (non-Olimex case)

Other GateMate evaluation boards (raw-chip dev boards, custom carrier
PCBs, some open-hardware variants) do **not** include an onboard
USB-to-JTAG bridge. They expose a raw JTAG header (TCK/TMS/TDI/TDO/GND,
plus optional TRST and VTREF) that must be driven by an external
programmer.

The DirtyJTAG project (https://github.com/jeanthom/DirtyJTAG) provides
a $5–15 microcontroller-based JTAG adapter that exposes the right
pins over USB.

## Step 0: Identify your DirtyJTAG host hardware

DirtyJTAG is firmware, not hardware — it runs on several different MCU
boards, and the JTAG-signal pinout depends on which board the firmware
was built for. The most common variants:

| Host board | DirtyJTAG firmware repo / branch | TCK | TMS | TDI | TDO | TRST (opt) | SRST (opt) |
|---|---|---|---|---|---|---|---|
| STM32 Blue Pill (STM32F103C8) | `jeanthom/DirtyJTAG` `main` | PA0 | PA1 | PA2 | PA3 | PA4 | PA5 |
| Raspberry Pi Pico (RP2040) | `jeanthom/DirtyJTAG-pico` | GP2 | GP3 | GP4 | GP5 | GP6 | GP7 |
| Adafruit Trinkey RP2040 | RP2040 fork (community) | varies | varies | varies | varies | — | — |

**To check which variant you have**:

```bash
# Plug in the DirtyJTAG, then:
udevadm info -n /dev/ttyACM0 | grep -E "ID_MODEL|ID_VENDOR_FROM_DATABASE"
# DirtyJTAG firmware presents as "Jean THOMAS DirtyJTAG" regardless of host MCU,
# so you'll need to physically inspect the board to determine the host.
```

Open the host MCU board's case (if any) and look at the silkscreen or
the PCB markings. Blue Pill boards are blue with an STM32 chip in the
center; Pico boards are smaller with `Pico` printed on top.

## Step 1: Identify the GateMate A1-EVB-2M JTAG header

The GateMate A1-EVB-2M (Olimex-branded variant) has a 10-pin JTAG header
in the **standard ARM 0.1" (2.54mm) pitch** layout. The header is
silkscreened **JTAG** on the PCB. The pinout (looking at the top side of
the board, key/notch on the left):

```
   ┌─────────────────┐
   │  1  VTREF       │ ← typically tied to 3.3V on the GateMate
   │  3  nTRST       │ ← optional; can be left disconnected
   │  5  TDI         │
   │  7  TMS         │
   │  9  TCK         │
   │ 11  RTCK        │ ← optional; not used by DirtyJTAG
   │ 13  TDO         │
   │ 15  nSRST       │ ← optional system reset
   │ 17  N/C         │
   │ 19  N/C         │
   │  2  Vsupply     │
   │  4  GND         │
   │  6  GND         │
   │  8  GND         │
   │ 10  GND         │
   │ 12  GND         │
   │ 14  GND         │
   │ 16  GND         │
   │ 18  GND         │
   │ 20  GND         │
   └─────────────────┘
```

The minimum signals you must wire are **TCK, TMS, TDI, TDO, and GND**.
The other pins can be left floating for a first bring-up.

If your GateMate board has a different JTAG header pitch (e.g. 0.05"
shrouded, or a 6-pin header), consult the Cologne Chip "GateMate A1
Evaluation Board User Manual" §3 (JTAG interface) for your specific
revision.

## Step 2: Wire the 5 essential signals

Use five female-to-female (or whatever matches your DirtyJTAG headers)
jumper wires. Strip nothing — these connectors are just push-on.

Example wiring for a **STM32 Blue Pill** DirtyJTAG:

| DirtyJTAG pin | GateMate JTAG header pin | Signal |
|---|---|---|
| PA0 | 9 | TCK |
| PA1 | 7 | TMS |
| PA2 | 5 | TDI |
| PA3 | 13 | TDO |
| GND | 4 (or any GND) | GND |

Example wiring for a **Raspberry Pi Pico** DirtyJTAG:

| DirtyJTAG pin | GateMate JTAG header pin | Signal |
|---|---|---|
| GP2 | 9 | TCK |
| GP3 | 7 | TMS |
| GP4 | 5 | TDI |
| GP5 | 13 | TDO |
| GND | 4 (or any GND) | GND |

**Important**: do **not** connect `Vsupply` (pin 2) or `VTREF` (pin 1)
from the GateMate to the DirtyJTAG. The DirtyJTAG is bus-powered from
USB; the GateMate is bus-powered from its own USB or barrel jack.
Connecting their rails together can damage one or both boards.

If your DirtyJTAG firmware uses 3.3V signal levels (almost all do) and
your GateMate fabric is also 3.3V (the A1 is), no level shifter is
needed.

## Step 3: Verify the chain

After wiring is complete and both boards are powered:

```bash
# Re-plug the DirtyJTAG once so udev re-applies the rule (only needed
# the very first time, after the udev rule install):
# (no command needed if you've already re-plugged once)

# Then ask openFPGALoader to scan the JTAG chain:
/opt/oss-cad-suite/bin/openFPGALoader -c dirtyJtag --detect
```

A working chain will report something like:

```
Jtag frequency : requested 6000000 Hz -> real 6000000 Hz
index 0:
  idcode 0x20000001
  manufacturer colognechip
  family GateMate Series
  model  GM1Ax
  irlength 6
```

The `0x20000001` IDCODE is the standard GateMate A1 fingerprint. If you
see a different IDCODE, you have wired to a different chip on the
board (the EVB has a small CPLD for the FlashSPI mux that has its own
IDCODE — make sure the GateMate fabric, not the mux, is the target).

If `--detect` returns "fails to open device" the udev rule is not
loaded; re-run `sudo udevadm control --reload && sudo udevadm trigger`
and re-plug the DirtyJTAG once.

If `--detect` returns "no jtag chain found" check the physical wiring,
particularly:
- TCK/TMS/TDI/TDO not swapped (TDO is the only output; TCK/TMS/TDI are
  inputs to the FPGA)
- GND is wired
- Both boards are powered
- The jumper wires are seated (a half-seated jumper looks identical to
  a fully-seated one but conducts nothing)

## Step 4: Flash a test bitstream

The Carnot repo ships a pre-synthesized n=16 Ising tile at
`rtl/gatemate_ising_n16.json`. To flash it (assuming P&R has produced a
`.cfg` bitstream — see the OSS CAD Suite workflow below):

```bash
cd /home/ianblenke/github.com/ianblenke/carnot
nextpnr-himbaechel \
  --device CCGM1A1 \
  --vopt ccf=rtl/gatemate_ising_n16.ccf \
  --json rtl/gatemate_ising_n16.json \
  -o textcfg=rtl/gatemate_ising_n16.cfg

gmpack -i rtl/gatemate_ising_n16.cfg -o rtl/gatemate_ising_n16.bit
openFPGALoader -c dirtyJtag -b olimex_gatemateevb \
  rtl/gatemate_ising_n16.bit
```

(As of 2026-05-14, `nextpnr-himbaechel` 0.10 has a LUT mapping mismatch
with yosys 0.64's GateMate output — yosys emits `CC_LUT3/CC_LUT2/CC_LUT1`
but nextpnr only accepts `CC_LUT4`. The workaround is `synth_gatemate
-abc9` in the yosys synthesis script, or upgrading either tool. See
`ops/hardware-bringup-prep.md` for the latest status.)

## Cross-references

- `ops/hardware-bringup-prep.md` — full bring-up state, including
  what's verified working and what's still TODO
- `/etc/udev/rules.d/99-fpga-boards.rules` — the udev rule that grants
  `uucp` group raw USB access (installed 2026-05-18)
- `rtl/gatemate_ising_n16.v` — n=16 Ising tile RTL (yosys-synthesizable
  to 136 cells, ~0.7% of A1 budget)
- `rtl/gatemate_ising_n16.json` — synthesized netlist
- `python/carnot/experiment_1676_gatemate_flash.py` — Carnot's flash
  experiment script
- DirtyJTAG firmware: https://github.com/jeanthom/DirtyJTAG
- Cologne Chip GateMate documentation: https://www.colognechip.com/programmable-logic/gatemate/
