import numpy as np
from carnot.agentic.arc_latent_registers import compute_latent_registers, AugmentedInducedWorldModel

def test_latent_registers():
    grid0 = np.array([[1, 0], [0, 2]])
    grid1 = np.array([[1, 0], [0, 2]])
    grid2 = np.array([[1, 0], [0, 2]])
    
    akey1 = (6, 0, 0)
    akey2 = (6, 1, 1)
    
    traj = [
        (grid0, akey1, grid1),
        (grid1, akey2, grid2)
    ]
    
    latents = compute_latent_registers(traj)
    assert len(latents) == 3
    assert latents[0] == (0, frozenset())
    assert latents[1] == (1, frozenset([1]))
    assert latents[2] == (2, frozenset([1, 2]))

def test_augmented_world_model():
    grid0 = np.array([[1, 0], [0, 2]])
    grid1 = np.array([[2, 0], [0, 2]])
    
    latent0 = (0, frozenset())
    latent1 = (1, frozenset([1]))
    
    akey = (6, 0, 0)
    
    transitions = [
        (grid0, latent0, akey, grid1, latent1)
    ]
    
    model = AugmentedInducedWorldModel("test")
    model.fit_augmented(transitions)
    
    pred = model.predict_augmented(grid0, latent0, akey)
    assert np.array_equal(pred, grid1)
    
    held_out = [
        (grid0, latent0, akey, grid1, latent1)
    ]
    res = model.consistency_energy_augmented(held_out)
    assert res["energy"] == 0.0
