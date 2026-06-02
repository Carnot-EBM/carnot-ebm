# PROVENANCE

Repository: https://github.com/alexiglad/EBT
Commit SHA: 19420cbeae655bbf11930219a675ade6897019e8
License: Apache-2.0

Copied components:
- `EBTModelArgs` (from `model/model_utils.py`)
- `init_whole_model_weights` (from `model/model_utils.py`)
- `RMSNorm` (from `model/ar_ebt_default.py`)
- `precompute_freqs_cis` (from `model/ar_ebt_default.py`)
- `reshape_for_broadcast` (from `model/ar_ebt_default.py`)
- `apply_rotary_emb` (from `model/ar_ebt_default.py`)
- `Attention` (from `model/ar_ebt_default.py`)
- `FeedForward` (from `model/ar_ebt_default.py`)
- `TransformerBlock` (from `model/ar_ebt_default.py`)
- `EBTDefault` (from `model/ar_ebt_default.py`)

These components were minimally modified to operate independently of other codebase dependencies for the purpose of a clean subset vendoring.