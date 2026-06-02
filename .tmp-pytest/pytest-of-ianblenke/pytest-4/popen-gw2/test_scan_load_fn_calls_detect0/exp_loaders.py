def main():
    model_a = _load_gemma(gpu_index=0)
    model_b = _load_qwen(hf_id='org/Q', gpu_index=0)
