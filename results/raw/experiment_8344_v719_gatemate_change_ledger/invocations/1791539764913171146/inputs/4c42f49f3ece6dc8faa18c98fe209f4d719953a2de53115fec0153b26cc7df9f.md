# Hardware bring-up prep — GateMate A1 + PolarFire SoC Discovery Kit

**Filed by:** outer-loop (Claude) 2026-05-14 16:55Z
**Status:** boards physically on bench; software prep documented below
**Affects experiments:** `.166 exp2105 (GateMate smoke), planned `.166/`.167 PolarFire smoke

## What's verified working

| Component | Status | Evidence |
|---|---|---|
| GateMate A1-EVB-2M physical | attached, **programmer is ONBOARD** | The Olimex variant ships with an integrated DirtyJTAG-firmware MCU on the PCB; programming goes over the board's own USB-C cable. Verified 2026-05-18 via `/sys/bus/usb/devices/3-2.3/` (5-interface composite: vendor JTAG + 2x CDC ACM + 2x CDC data). The earlier note about needing an external DirtyJTAG was misleading — `lsusb` was checked for FTDI VIDs, but the onboard programmer is a DirtyJTAG-firmware MCU at `1209:c0ca`, not FTDI. |
| DirtyJTAG programmer (onboard the GateMate) | enumerated as `1209:c0ca` on /dev/ttyACM0+1 | `udevadm info -n /dev/ttyACM0` → manufacturer=Jean THOMAS, product=DirtyJTAG, serial=1861832311111616 |
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
| FlashPro5 USB raw access (for JTAG/flash ops, not UART) | ~~needs udev rule~~ — INSTALLED 2026-05-18 | `/etc/udev/rules.d/99-fpga-boards.rules` grants `uucp` group raw access to 1209:c0ca + 1514:2008 |
| ~~GateMate JTAG wiring~~ | **CORRECTED 2026-05-18: not needed — programmer is onboard the Olimex GateMate A1-EVB-2M** | The Olimex variant integrates a DirtyJTAG-firmware MCU on the PCB; programming goes over the board's own USB-C, no external jumpers required. Verified by reading 0x20000001 (GM1Ax IDCODE) over `openFPGALoader -c dirtyJtag --detect`. |
| PolarFire microSD boot image | RESOLVED 2026-05-14 | SD imaged + extended; board boots cleanly. |
| PolarFire DIP-switch boot mode | RESOLVED 2026-05-14 | Booting from SD as confirmed by 4+ days uptime under ssh polarfire. |
| yosys ↔ nextpnr-himbaechel LUT mapping mismatch | yosys 0.64 emits CC_LUT3/CC_LUT2/CC_LUT1, nextpnr-himbaechel 0.10 only accepts CC_LUT4 | Workaround pending: try `synth_gatemate -abc9` or upgrade either tool. exp2105 will document; this is now the **last remaining** GateMate blocker for end-to-end bitstream flow. |

## Pre-prepared artifacts

| Artifact | Path | Purpose |
|---|---|---|
| Minimal n=16 GateMate Ising tile (RTL) | `rtl/gatemate_ising_n16.v` | exp2105 smoke target |
| yosys synthesis script | `rtl/gatemate_ising_n16.ys` | yosys -p flow documented |
| Synthesized JSON netlist | `rtl/gatemate_ising_n16.json` | verified produces 136 cells (~0.7% of A1 budget); P&R blocked on LUT mapping |
| udev rule file | `/etc/udev/rules.d/99-fpga-boards.rules` | **INSTALLED 2026-05-18 19:54Z** by outer-loop; verified working (devices now 660 root:uucp) |

## Operator-side action checklist (sudo / bench access needed)

**All items in this checklist are now COMPLETE (2026-05-18):**

```bash
# 1. udev rule install — COMPLETED 2026-05-18 19:54Z
#    sudo cp /tmp/99-fpga-boards.rules /etc/udev/rules.d/99-fpga-boards.rules
#    sudo udevadm control --reload && sudo udevadm trigger
#    Verified: stat /dev/bus/usb/003/006 = 660 root:uucp ✓
#              stat /dev/bus/usb/003/005 = 660 root:uucp ✓

# 2. DirtyJTAG access — VERIFIED 2026-05-18:
#    openFPGALoader -c dirtyJtag --detect
#    → idcode 0x20000001 (GM1Ax) read successfully

# 3. PolarFire SSH — VERIFIED 2026-05-18:
#    ssh polarfire 'uname -a && uptime'
#    → Linux mpfs-disco-kit 6.18.17-linux4microchip-2026.04.1 riscv64
#       uptime: 4 days 5:41
```

**No further bench-side action required for either board** as of 2026-05-18.
The only remaining technical blocker is the yosys↔nextpnr LUT mapping
mismatch (`synth_gatemate -abc9` workaround pending), which is software
not hardware.

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

## KV260
- date: 2026-05-21
- ssh_kria: reachable
- active_overlay: carnot_ising_v2_n64 (= carnot_ising_v4 bitstream)
- uio_devices_present: 5
- latency_mean_us: 3.183
- n_cycles_measured: 100
- kv260_terminal: True
- next_step: GRADUATED: KV260 terminal criteria met
