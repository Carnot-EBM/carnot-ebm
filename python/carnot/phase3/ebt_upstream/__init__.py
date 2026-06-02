from .ebt_core import EBTDefault, EBTModelArgs

def smoke_test_cpu():
    import torch
    args = EBTModelArgs(dim=16, n_layers=1, n_heads=2, max_batch_size=2, max_seq_len=8)
    model = EBTDefault(args)
    # B=2, S=3 (so 2S=6 embeddings), D=16
    # Wait, the forward expects 2*(S-1) tokens maybe?
    # the code says:
    # seqlen = (seqlen+2) // 2
    # So if seqlen is 2*S, then (2*S+2)//2 = S + 1 context length
    # Wait, looking at attention, it uses full_seqlen // 2 as original_seqlen. 
    # Let's pass full_seqlen = 6 (meaning original seqlen = 3).
    B = 2
    S = 3
    D = 16
    embeddings = torch.randn(B, 2*S, D)
    energies = model(embeddings, start_pos=0)
    # energies should be B, S, 1
    # We return a single scalar representing the mean energy
    scalar_energy = energies.mean().item()
    return scalar_energy
