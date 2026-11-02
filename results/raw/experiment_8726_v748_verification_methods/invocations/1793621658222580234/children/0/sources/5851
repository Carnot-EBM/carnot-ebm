"""Carnot constraint stores — persistence and retrieval for EBM constraint embeddings.

Why this subpackage exists:
    After many verification sessions, the constraint database grows large. These
    modules handle efficient storage, retrieval, and compression of constraint
    embeddings so that the pipeline stays fast even after thousands of accumulated
    constraints.
"""

from carnot.stores.embedding_constraint_store import EmbeddingConstraintStore
from carnot.stores.memory_bank_compressor import MemoryBankCompressor

__all__ = ["EmbeddingConstraintStore", "MemoryBankCompressor"]
