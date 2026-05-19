# KV260 PYNQ SD-Card Boot Path

This document outlines the procedure to flash and boot the KV260 via PYNQ to load custom bitstreams without requiring a JTAG programmer.

## Is PYNQ Viable?
**Yes.** PYNQ supports loading custom user bitstreams at runtime using the Python `Overlay()` API, provided you supply both the `.bit` file and its corresponding `.hwh` (hardware handoff) file.

## Requirements
*   **MicroSD Card:** Minimum 16GB (32GB recommended)
*   **PYNQ OS:** Ubuntu 22.04 LTS for KV260 with Kria-PYNQ installed over it.
*   **Bitstream & HWH:** `carnot_ising_v4.bit` and `carnot_ising_v4.hwh`

### 1. Generating the `.hwh` File
Currently, our `build_bd_v4.tcl` Vivado script generates the `.bit` file but does not explicitly export the `.hwh` file to the final output directory.
Because our design is packaged as a Block Design (`carnot_ising_v4_bd`), Vivado generates the `.hwh` internally during synthesis/hardware handoff.
To extract it, either:
*   Locate the file within the Vivado project tree after the build:
    `output/carnot_ising_v4_bd/project/carnot_ising_v4.gen/sources_1/bd/carnot_ising_v4_bd/hw_handoff/carnot_ising_v4_bd.hwh`
*   Or append a command to the Tcl script to copy it alongside the bitstream:
    ```tcl
    file copy -force ${project_dir}/${project_name}.gen/sources_1/bd/${bd_name}/hw_handoff/${bd_name}.hwh ${bitstream_dst_dir}/
    ```

### 2. Flashing the PYNQ SD Card
Since there isn't a single pre-built PYNQ image for the KV260, the installation is a two-step process:
1.  **Download Ubuntu Base Image:** Download the official Ubuntu 22.04 LTS image for AMD/Xilinx Kria from [ubuntu.com/download/amd](https://ubuntu.com/download/amd).
2.  **Flash SD Card:** Write the image to the SD card using `dd` or a tool like BalenaEtcher:
    ```bash
    sudo dd if=ubuntu-22.04-preinstalled-desktop-arm64+xilinx-kria.img of=/dev/sdX bs=4M status=progress
    ```

### 3. Installing PYNQ Framework
1.  Insert the SD card into the KV260 and boot (ensure boot mode jumpers are set for SD card).
2.  Connect to the KV260 via SSH or serial console.
3.  Clone the Kria-PYNQ repository and run the installation script:
    ```bash
    git clone https://github.com/Xilinx/Kria-PYNQ.git
    cd Kria-PYNQ/
    sudo bash install.sh -b KV260
    ```
    *(Note: This installation may take up to an hour)*

### 4. Loading the Custom Bitstream via PYNQ
Once PYNQ is installed and running:
1.  Transfer your `.bit` and `.hwh` files to the KV260 (ensure they have the exact same base name):
    ```bash
    scp carnot_ising_v4.bit carnot_ising_v4.hwh ubuntu@<kv260_ip>:~/
    ```
2.  In a Python environment on the KV260, load the overlay:
    ```python
    from pynq import Overlay
    
    # The .bit and .hwh files must share the same base name in the same directory.
    ol = Overlay('/home/ubuntu/carnot_ising_v4.bit')
    
    # Access the axi_gpio peripheral (which exposes the sampler's state)
    # The IP name matches the name in your block design.
    gpio_ip = ol.axi_gpio_0
    
    # Read the 32-bit state
    spin_state = gpio_ip.read(0x0)
    print(f"Current spin state: {spin_state:08x}")
    ```

## Operator Effort Estimate
*   Downloading and flashing Ubuntu: ~30 minutes.
*   Running PYNQ install script on target: ~60 minutes.
*   Extracting `.hwh` and testing the Python script: ~15 minutes.
*   **Total estimated effort:** ~1.5 - 2 hours.
