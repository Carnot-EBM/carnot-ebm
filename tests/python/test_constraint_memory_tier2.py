import pytest
import os
import sqlite3
from carnot.learn.constraint_memory import ConstraintMemoryCache

@pytest.fixture
def cache(tmp_path):
    db_path = str(tmp_path / "test_constraint_memory.db")
    c = ConstraintMemoryCache(db_path=db_path)
    yield c
    c.clear()

def test_constraint_memory_store_and_query(cache):
    # Store a violation (real error)
    fact_id1 = cache.store_violation("math", "2+2=4", True)
    assert fact_id1 is not None
    
    # Store another violation for same fact (not a real error)
    fact_id2 = cache.store_violation("math", "2+2=4", False)
    assert fact_id1 == fact_id2 # SQLite won't return lastrowid for update, but our logic doesn't fetch it back actually in store_violation if row exists, wait, let's fix store_violation to return id.
    
def test_constraint_memory_templates(cache):
    cache.store_violation("physics", "F=ma", True)
    cache.store_violation("physics", "E=mc^2", True)
    
    templates = cache.query_templates("physics")
    assert len(templates) == 2
    assert "Template for: F=ma" in templates
    assert "Template for: E=mc^2" in templates
