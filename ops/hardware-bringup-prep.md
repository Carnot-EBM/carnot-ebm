# Hardware bring-up prep — GateMate A1 + PolarFire SoC Discovery Kit

**Filed by:** outer-loop (Claude) 2026-05-14 16:55Z
**Status:** boards physically on bench; software prep documented below
**Affects experiments:** `.166 exp2105 (GateMate smoke), planned `.166/`.167 PolarFire smoke

## What's verified working

| Component | Status | Evidence |
|---|---|---|
| GateMate A1-EVB-2M physical | attached, not auto-enumerating | `lsusb` shows no FT2232H/FT4232H; board needs external JTAG via DirtyJTAG |
| DirtyJTAG adapter | enumerated as `1209:c0ca` on /dev/ttyACM0+1 | `lsusb` + `udevadm info -n /dev/ttyACM0` |
| PolarFire SoC Discovery Kit | enumerated as `1514:2008` Microsemi FlashPro5 | `lsusb` + dmesg shows FT4232H internal bridge → /dev/ttyUSB0 (UART) |
| Ethernet (PolarFire) | connected to switch | operator confirmation |
| microSD in PolarFire | inserted, NOT formatted with boot image | operator note |
| OSS CAD Suite | at /opt/oss-cad-suite, fully populated | yosys 0.64+149, nextpnr-himbaechel 0.10-45, openFPGALoader v1.1.1, ghdl |
| DirtyJTAG cable support in openFPGALoader | yes (`-c dirtyJtag`) | `openFPGALoader --list-cables \| grep dirty` |
| GateMate A1 board profile in openFPGALoader | yes (`-b olimex_gatemateevb`) | `openFPGALoader --list-boards` |
| User serial-port group membership | `ianblenke` in `uucp` group | `id` |

## What's NOT yet working

| Component | Issue | Fix path |
|---|---|---|
| DirtyJTAG USB raw access | `openFPGALoader -c dirtyJtag --detect` fails with "fails to open device" because /dev/bus/usb/003/006 is 664 root:root | Install /tmp/99-fpga-boards.rules (sudo) — see below |
| FlashPro5 USB raw access (for JTAG/flash ops, not UART) | Same — Microsemi 1514:2008 needs udev rule for raw USB | Same udev rule file covers both |
| GateMate JTAG wiring | DirtyJTAG → GateMate JTAG header pins (TCK/TMS/TDI/TDO/GND) not physically connected | Operator at bench: identify GateMate A1-EVB-2M JTAG header, wire 5 jumpers to DirtyJTAG GPIOs (pins depend on DirtyJTAG host firmware — typically PB6/PB7/PB8/PB9/GND on STM32 BluePill DirtyJTAG, or specific GPIOs on RP2040) |
| PolarFire microSD boot image | SD inserted but not formatted with HSS + Linux | Operator at bench: format SD with Microchip's HSS payload + a reference Linux rootfs (Yocto build or Microchip's reference image) |
| PolarFire DIP-switch boot mode | typically defaults to eMMC | Operator at bench: flip DIP switches to SD-boot mode per Microchip documentation |
| yosys ↔ nextpnr-himbaechel LUT mapping mismatch | yosys 0.64 emits CC_LUT3/CC_LUT2/CC_LUT1, nextpnr-himbaechel 0.10 only accepts CC_LUT4 | Workaround pending: try `synth_gatemate -abc9` or upgrade either tool. exp2105 will document; not blocking the rest of the prep |

## Pre-prepared artifacts

| Artifact | Path | Purpose |
|---|---|---|
| Minimal n=16 GateMate Ising tile (RTL) | `rtl/gatemate_ising_n16.v` | exp2105 smoke target |
| yosys synthesis script | `rtl/gatemate_ising_n16.ys` | yosys -p flow documented |
| Synthesized JSON netlist | `rtl/gatemate_ising_n16.json` | verified produces 136 cells (~0.7% of A1 budget); P&R blocked on LUT mapping |
| udev rule file | `/tmp/99-fpga-boards.rules` | NOT YET INSTALLED — needs operator sudo |

## Operator-side action checklist (sudo / bench access needed)

```bash
# 1. Install udev rule for FlashPro5 + DirtyJTAG raw USB access
sudo cp /tmp/99-fpga-boards.rules /etc/udev/rules.d/99-fpga-boards.rules
sudo udevadm control --reload && sudo udevadm trigger
# Then re-plug both USB devices.

# 2. Verify DirtyJTAG access (should report scan results, not "fails to open"):
openFPGALoader -c dirtyJtag --detect

# 3. Verify PolarFire UART (115200 8N1, board power-cycled):
screen /dev/ttyUSB0 115200    # exit with Ctrl-A k y
# Should show U-Boot/HSS banner if the board has a working boot image.
```

### Bench-side (physical actions)

1. **GateMate JTAG wiring**: identify the GateMate A1-EVB-2M JTAG header (typically a 10-pin or 14-pin connector marked "JTAG" on the silkscreen). Wire from the DirtyJTAG's host pins:
   - DirtyJTAG TCK → GateMate JTAG TCK
   - DirtyJTAG TMS → GateMate JTAG TMS
   - DirtyJTAG TDI → GateMate JTAG TDI
   - DirtyJTAG TDO → GateMate JTAG TDO
   - DirtyJTAG GND → GateMate GND
   - (optionally) DirtyJTAG TRST → GateMate JTAG TRST
   Pinout for DirtyJTAG depends on the firmware variant; check the BluePill/Pico firmware's README.

2. **PolarFire microSD**: format with Microchip's reference Linux image. Recommended path:
   - Download Microchip's pre-built reference image from https://github.com/polarfire-soc/polarfire-soc-yocto-bsp (or buildroot variant)
   - `sudo dd if=mpfs_reference.img of=/dev/sdX bs=4M status=progress`
   - Flip DIP switches to SD-boot mode (board-specific — check the Microchip MPFS-DISCO-KIT user guide)
   - Power-cycle. UART should show HSS → U-Boot → Linux boot.

3. **PolarFire factory image verification** (alternative to formatting SD): if you'd rather just confirm the board is alive without committing to a Linux image, the FlashPro5 can read the existing eMMC content via Microchip's SoftConsole (proprietary, ~5GB install) OR via OpenOCD with the right config.

## Next milestone tasks

- `.166 exp2105: GateMate A1 toolchain smoke (RTL synth + P&R; flash conditional on JTAG wiring complete)
- `.166 exp2106 (proposed, pending operator confirmation): PolarFire SoC smoke (UART banner read + boot-mode documentation)
- `.167+: real Carnot Ising sampler on GateMate (replace the minimal n=16 with the verifier-relevant sampler from rtl/ising_sampler_v3.v adapted)

## Why nothing further can be done from this session

Without sudo (udev rule), the DirtyJTAG and FlashPro5 USB raw devices remain root-only. The conductor's experiments would face the same "fails to open device" error. Once the udev rule lands, every subsequent JTAG/flash op should work from non-root.
