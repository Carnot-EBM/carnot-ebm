# Exp 5231 p-bit boundary-exchange timing-ratio plan

Status: future sampler plan only. No speedup claim is allowed from this document.

Future workload: `distributed_sparse_pbit_boundary_exchange_n1024x4`.

partitions: 4 x 256 p-bits. Each partition owns a sparse Ising subgraph and emits a deterministic boundary packet after each exchange interval. The first implementation target is CPU reference plus board-local or simulator parity; million-p-bit and TSU/XTR-0 references remain motivation only until real device evidence exists.

hash/correctness checks:
- Hash the graph CSR, partition map, seeds, exchange interval, and each boundary packet stream.
- Compare CPU reference energy, final state checksum, per-partition boundary packet hashes, and accepted exchange count.
- Record the exact command transcript for each substrate and reject any run with missing hash or correctness fields.

Measurement needed before speedup:
- Same workload, same seeds, same partition map, and same exchange schedule on CPU baseline and hardware candidate.
- End-to-end wall clock including partition setup, boundary exchange, device transfer, sampler steps, and result validation.
- Only a hash/correctness-passing end-to-end workload may support a timing-ratio or speedup statement.
