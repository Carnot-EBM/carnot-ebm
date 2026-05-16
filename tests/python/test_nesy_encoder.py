import jax.numpy as jnp
import pytest
from carnot.embeddings.nesy_encoder import NeSyEncoder

def test_nesy_encoder_equality() -> None:
    # REQ-SYMKAN-2074, SCENARIO-SYMKAN-2074
    encoder = NeSyEncoder()
    # State with 3 variables, vocab size 100
    state = jnp.zeros((3, 100))
    state = state.at[0, 5].set(1.0)
    state = state.at[1, 5].set(1.0)
    state = state.at[2, 10].set(1.0)
    
    # x == y
    energy_fn_eq = encoder.compile_predicate("VAR_0 == VAR_1")
    energy_eq = energy_fn_eq(state)
    assert energy_eq < 1e-5
    
    energy_fn_neq = encoder.compile_predicate("VAR_0 == VAR_2")
    energy_neq = energy_fn_neq(state)
    assert energy_neq > 1.0
    
def test_nesy_encoder_inequality() -> None:
    # REQ-SYMKAN-2074, SCENARIO-SYMKAN-2074
    encoder = NeSyEncoder()
    state = jnp.zeros((3, 100))
    state = state.at[0, 5].set(1.0)
    state = state.at[1, 5].set(1.0)
    state = state.at[2, 10].set(1.0)
    
    # x != y
    energy_fn_neq2 = encoder.compile_predicate("VAR_0 != VAR_1")
    energy_neq2 = energy_fn_neq2(state)
    assert energy_neq2 > 1.0  # High energy because they are equal
    
    energy_fn_eq2 = encoder.compile_predicate("VAR_0 != VAR_2")
    energy_eq2 = energy_fn_eq2(state)
    assert energy_eq2 < 1.0  # Low energy because they are different

def test_nesy_encoder_unsupported() -> None:
    # REQ-SYMKAN-2074
    encoder = NeSyEncoder()
    with pytest.raises(ValueError):
        encoder.compile_predicate("VAR_0 AND VAR_1")
