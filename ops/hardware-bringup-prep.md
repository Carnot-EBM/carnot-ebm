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

## 2026-05-14 22:15Z note — THERMAL CONSTRAINTS (operator)

**Neither the PolarFire SoC Discovery Kit nor the GateMate A1-EVB-2M has
active cooling.** The KV260 has a small fan + heatsink; these two boards
are bare / passively cooled.

**Implications:**

- **Sustained high-load workloads risk thermal throttling.** RISC-V cores
  on the MPFS095T may down-clock under sustained 100% utilization.
  FPGA fabric on either board may degrade or warp under extended
  high-toggle-rate operation.
- **Benchmark results may not be representative of production sustained
  load.** A 30-minute hot run could give different numbers than a 30-
  second smoke test.
- **Avoid long sustained-100%-load tests** without external cooling
  (small USB fan, etc.) — the boards aren't designed for it.

**Required artifact field for hardware experiments going forward:**

Hardware experiments on PolarFire / GateMate should include in the
artifact:

```
- thermal_note: "passively cooled; no active fan; sustained-load results
  may differ from production with active cooling"
- run_duration_s: <int>      # so retro can assess thermal exposure
- soc_temp_max_c: <float|null>  # if readable via /sys/class/thermal on
                                # PolarFire; null on GateMate (no thermal
                                # sensor exposed)
```

**For sustained workload tests (>5 min wall time):**

- Monitor `/sys/class/thermal/thermal_zone*/temp` on the PolarFire
  (millidegrees C) periodically
- If `soc_temp_max_c > 85`, abort the run and note thermal_aborted in
  the artifact
- For GateMate, watch for unexplained logic failures mid-run as a proxy
  for thermal issues (the fabric doesn't have an integrated thermal sensor
  in the chip)

**Paper-v6 implications:** any "Carnot runs on $130 of open hardware"
sovereignty claim must explicitly disclose the passive-cooling
limitation. Sustained verifier-deployment latencies + accuracies on
these boards represent burst-mode capability, not production sustained
load.

---

## 2026-05-14 18:30Z update — PolarFire SoC Discovery Kit BOOTED + SSH-ACCESSIBLE

| Item | State |
|---|---|
| Boot chain | HSS (2026.04, programmed via FPExpress + .job) → U-Boot → Linux ✓ |
| OS | Microchip Distro 1.0 (Yocto), Linux 6.18.17-linux4microchip-2026.04.1 |
| CPU | SiFive U54-MC, 4 cores, riscv64, sv39 MMU |
| RAM | 545 MiB total, 478 MiB available |
| SD card | 119.3 GiB, mmcblk0; rootfs auto-resized to 119.2 GiB ✓ (sgdisk -e was the fix — pre-extend the GPT before insertion) |
| Network | eth0 via DHCP at 192.168.51.197 (lease should be reserved on router); mDNS works via `mpfs-disco-kit.local` |
| SSH | `ssh root@mpfs-disco-kit.local` works with key-based auth ✓ |
| Local alias | `ssh polarfire` (configured in `~/.ssh/config` on this host) |
| Python | 3.12.12 + pip3 pre-installed |
| Outbound network | works (ICMP to 1.1.1.1: 10.8ms) |

### Working FPExpress flash recipe (for future re-flash)

```bash
# (one-time prerequisites)
sudo cp /tmp/99-fpga-boards.rules /etc/udev/rules.d/ && sudo udevadm control --reload && sudo udevadm trigger
mv /usr/local/microchip/Program_Debug_v2024.1/Program_Debug_Tool/lib64/rhel/libstdc++.so.6 \
   /usr/local/microchip/Program_Debug_v2024.1/Program_Debug_Tool/lib64/rhel/libstdc++.so.6.microchip
# (libstdc++ moves let the system's newer libstdc++ be picked up by ld)

# (per-flash)
mkdir -p ~/fpx_create_job/MPFS
cat > /tmp/flash.tcl <<TCL
create_job_project -job_project_location "\$env(HOME)/fpx_create_job/MPFS" -job_file "/tmp/MPFS_DISCOVERY.job"
TCL
QT_QPA_PLATFORM=offscreen /usr/local/microchip/Program_Debug_v2024.1/Program_Debug_Tool/bin64/FPExpress SCRIPT:/tmp/flash.tcl

# Then read the chip name from $HOME/fpx_create_job/MPFS/MPFS_DISCOVERY/MPFS_DISCOVERY.pro
# (look for <Device type="ACTEL"><Name>...</Name></Device>)
# For Discovery Kit MPFS095T: device name is "MPFS095T"

cat > /tmp/flash_run.tcl <<TCL
open_project -project "\$env(HOME)/fpx_create_job/MPFS/MPFS_DISCOVERY/MPFS_DISCOVERY.pro"
set_programming_action -name "MPFS095T" -action "PROGRAM"
run_selected_actions
TCL
QT_QPA_PLATFORM=offscreen /usr/local/microchip/Program_Debug_v2024.1/Program_Debug_Tool/bin64/FPExpress SCRIPT:/tmp/flash_run.tcl
```

### SD card prep recipe (for future re-flash)

```bash
# Download
cd /tmp
curl -fsSL -O "https://github.com/linux4microchip/meta-mchp/releases/download/linux4microchip-2026.04/mchp-base-image-mpfs-disco-kit.rootfs-20260430114629.wic.gz"
gzip -t mchp-base-image-mpfs-disco-kit.rootfs-*.wic.gz

# Write + extend GPT (run extension BEFORE first U-Boot read, otherwise U-Boot can't find backup GPT and may hang on mmc reads)
gunzip -c mchp-base-image-mpfs-disco-kit.rootfs-*.wic.gz | sudo dd of=/dev/sdb bs=4M status=progress conv=fsync
sudo sgdisk -e /dev/sdb
sync
```

### Open issue: SDCARD #1 (29GB SanDisk SL32G) had U-Boot read failures at UHS SDR104

- `mmc info` worked, `mmc part` failed with "mmc fail to send stop cmd"
- Same image on the new 119GB card works fine
- Likely card-vs-controller signal-integrity issue at 208MHz UHS SDR104
- Not chasing further; the 119GB card is functional

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
