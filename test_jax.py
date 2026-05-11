import jax
import optax
from carnot.models.kan.symbolic_kan import SymbolicKANConfig, SymbolicRoutingLayer, SymbolicKANParams

config = SymbolicKANConfig()
layer = SymbolicRoutingLayer(config)

def extract_params(p):
    return {
        "projection_weights": p.projection_weights,
        "projection_bias": p.projection_bias,
        "route_logits": p.route_logits,
        "route_scales": p.route_scales,
        "output_bias": p.output_bias
    }

d = extract_params(layer.params)
print(jax.tree_util.tree_flatten(d))
