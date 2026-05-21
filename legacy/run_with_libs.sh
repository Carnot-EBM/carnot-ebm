#!/bin/bash
export LD_LIBRARY_PATH=.venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:.venv/lib/python3.12/site-packages/nvidia/cublas/lib:.venv/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
.venv/bin/python test_gguf.py
