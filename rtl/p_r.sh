#!/bin/bash
nextpnr-himbaechel --device GateMateA1_32VQ --json rtl/gatemate_ising_n16.json --write rtl/gatemate_ising_n16_routed.json --v dff
p_r -i rtl/gatemate_ising_n16_routed.json -o rtl/gatemate_ising_n16
openFPGALoader -b gatemate_a1_evb rtl/gatemate_ising_n16.bit