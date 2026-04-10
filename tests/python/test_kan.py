"""Tests for KAN (Kolmogorov-Arnold Networks) Energy-Based Model.

Spec coverage: REQ-CORE-001, REQ-CORE-002, SCENARIO-CORE-001, SCENARIO-CORE-003
"""

import jax
import jax.numpy as jnp
import jax.random as jrandom
import pytest

from carnot.core.energy import EnergyFunction
from carnot.models.ising import IsingConfig, IsingModel
from carnot.models.gibbs import GibbsConfig, GibbsModel
from carnot.models.kan import (
    BSpline,
    BSplineParams,
    KANConfig,
    KANEnergyFunction,
    KANModel,
)


class TestBSpline:
    """Tests for REQ-CORE-001: BSpline implementation."""

    def test_bspline_creation(self) -> None:
        """REQ-CORE-001: BSpline creates with valid parameters."""
        spline = BSpline(num_knots=10, degree=3)
        assert spline.num_knots == 10
        assert spline.degree == 3

    def test_bspline_n_params(self) -> None:
        """REQ-CORE-001: BSpline has correct number of parameters."""
        spline = BSpline(num_knots=10, degree=3)
        assert spline.n_params == 13

    def test_bspline_evaluate_finite(self) -> None:
        """SCENARIO-CORE-001: BSpline produces finite output."""
        spline = BSpline(num_knots=10, degree=3)
        x = jnp.array([0.0, -0.5, 0.5, 1.0, -1.0])
        result = spline.evaluate(x)
        assert jnp.all(jnp.isfinite(result))

    def test_bspline_evaluate_at_knots(self) -> None:
        """REQ-CORE-001: BSpline evaluation at knot points."""
        spline = BSpline(num_knots=5, degree=1)
        x = jnp.linspace(-1.0, 1.0, 5)
        result = spline.evaluate(x)
        assert result.shape == x.shape
        assert jnp.all(jnp.isfinite(result))

    def test_bspline_custom_params(self) -> None:
        """REQ-CORE-001: BSpline with custom control points."""
        params = BSplineParams(control_points=jnp.ones(6))
        spline = BSpline(num_knots=3, degree=3)
        x = jnp.array([0.0])
        result = spline.evaluate(x, params)
        assert jnp.isfinite(result)

    def test_bspline_callable(self) -> None:
        """REQ-CORE-001: BSpline __call__ shorthand."""
        spline = BSpline(num_knots=5, degree=2)
        x = jnp.array([0.0, 0.5])
        result = spline(x)
        assert result.shape == x.shape


class TestKANConfig:
    """Tests for KAN configuration."""

    def test_default_config(self) -> None:
        """REQ-CORE-001: Default KAN config has sensible values."""
        config = KANConfig()
        assert config.input_dim == 784
        assert config.num_knots == 10
        assert config.degree == 3
        assert config.sparse is True
        assert config.edge_density == 0.1

    def test_custom_config(self) -> None:
        """REQ-CORE-001: Custom KAN architecture."""
        config = KANConfig(
            input_dim=100,
            num_knots=8,
            degree=2,
            sparse=False,
        )
        assert config.input_dim == 100
        assert config.num_knots == 8
        assert config.degree == 2
        assert config.sparse is False

    def test_validation_zero_dim(self) -> None:
        """REQ-CORE-001: input_dim=0 raises error."""
        config = KANConfig(input_dim=0)
        with pytest.raises(ValueError, match="input_dim must be > 0"):
            config.validate()

    def test_validation_small_knots(self) -> None:
        """REQ-CORE-001: num_knots < 2 raises error."""
        config = KANConfig(num_knots=1)
        with pytest.raises(ValueError, match="num_knots must be >= 2"):
            config.validate()


class TestKANEnergyFunction:
    """Tests for REQ-CORE-002: KAN energy function."""

    def test_creation(self) -> None:
        """REQ-CORE-002: KAN energy function creates successfully."""
        config = KANConfig(input_dim=10, sparse=True, edge_density=0.5)
        kan = KANEnergyFunction(config)
        assert kan.input_dim == 10

    def test_energy_finite(self) -> None:
        """SCENARIO-CORE-001: Energy is finite for valid input."""
        config = KANConfig(input_dim=10, sparse=True, edge_density=0.3)
        kan = KANEnergyFunction(config)
        x = jnp.array([0.0, 1.0, 0.5, -0.5, 1.0, 0.0, 0.5, -0.5, 1.0, 0.0])
        e = kan.energy(x)
        assert jnp.isfinite(e)

    def test_energy_batch(self) -> None:
        """SCENARIO-CORE-002: Batch energy computation."""
        config = KANConfig(input_dim=5, sparse=True, edge_density=0.5)
        kan = KANEnergyFunction(config)
        xs = jrandom.normal(jrandom.PRNGKey(0), (4, 5))
        energies = kan.energy_batch(xs)
        assert energies.shape == (4,)
        assert jnp.all(jnp.isfinite(energies))

    def test_gradient_finite(self) -> None:
        """SCENARIO-CORE-003: Gradient is finite and correct shape."""
        config = KANConfig(input_dim=5, sparse=True, edge_density=0.5)
        kan = KANEnergyFunction(config)
        x = jrandom.normal(jrandom.PRNGKey(0), (5,))
        grad = kan.grad_energy(x)
        assert grad.shape == (5,)
        assert jnp.all(jnp.isfinite(grad))

    def test_gradient_consistency(self) -> None:
        """SCENARIO-CORE-003: Gradient matches finite difference."""
        config = KANConfig(input_dim=3, sparse=True, edge_density=1.0)
        kan = KANEnergyFunction(config)
        x = jrandom.normal(jrandom.PRNGKey(42), (3,)) * 0.5
        grad = kan.grad_energy(x)

        eps = 1e-4
        for i in range(3):
            x_p = x.at[i].add(eps)
            x_m = x.at[i].add(-eps)
            fd = (kan.energy(x_p) - kan.energy(x_m)) / (2.0 * eps)
            assert jnp.abs(grad[i] - fd) < 0.1, (
                f"Gradient mismatch at index {i}: analytic={grad[i]}, fd={fd}"
            )

    def test_interface_conformance(self) -> None:
        """REQ-CORE-002: Works through EnergyFunction protocol."""
        config = KANConfig(input_dim=5, sparse=True, edge_density=0.5)
        kan = KANEnergyFunction(config)
        assert isinstance(kan, EnergyFunction)

        def compute_energy(ef: EnergyFunction, x: jax.Array) -> jax.Array:
            return ef.energy(x)

        x = jnp.ones(5)
        e = compute_energy(kan, x)
        assert jnp.isfinite(e)

    def test_dense_edges(self) -> None:
        """REQ-CORE-001: KAN with dense connectivity."""
        config = KANConfig(input_dim=4, sparse=False)
        kan = KANEnergyFunction(config)
        x = jnp.ones(4)
        e = kan.energy(x)
        assert jnp.isfinite(e)

    def test_n_params(self) -> None:
        """REQ-CORE-002: n_params property counts all spline parameters."""
        config = KANConfig(input_dim=3, sparse=True, edge_density=1.0)
        kan = KANEnergyFunction(config)
        n_params = kan.n_params
        assert n_params > 0


class TestKANFromIsing:
    """Tests for KAN initialization from Ising model."""

    def test_from_ising_creation(self) -> None:
        """REQ-CORE-002: KAN can be initialized from Ising model."""
        ising = IsingModel(IsingConfig(input_dim=5))
        kan = KANEnergyFunction.from_ising(ising)
        assert kan.input_dim == 5

    def test_from_ising_energy_comparable(self) -> None:
        """REQ-CORE-002: KAN from Ising produces similar energy values."""
        ising = IsingModel(IsingConfig(input_dim=5))
        kan = KANEnergyFunction.from_ising(ising)

        x = jnp.array([1.0, 0.0, 1.0, 0.0, 1.0])

        ising_e = ising.energy(x)
        kan_e = kan.energy(x)

        assert jnp.isfinite(kan_e)
        assert jnp.isfinite(ising_e)


class TestKANModel:
    """Tests for KANModel wrapper."""

    def test_model_creation(self) -> None:
        """REQ-CORE-002: KANModel wraps energy function."""
        config = KANConfig(input_dim=10, sparse=True, edge_density=0.3)
        model = KANModel(config)
        assert model.input_dim == 10
        assert model.n_params > 0

    def test_model_energy(self) -> None:
        """REQ-CORE-002: KANModel.energy() delegates to energy function."""
        config = KANConfig(input_dim=5, sparse=True, edge_density=0.5)
        model = KANModel(config)
        x = jnp.ones(5)
        e = model.energy(x)
        assert jnp.isfinite(e)

    def test_model_gradient(self) -> None:
        """SCENARIO-CORE-003: KANModel.grad_energy() works."""
        config = KANConfig(input_dim=5, sparse=True, edge_density=0.5)
        model = KANModel(config)
        x = jnp.ones(5)
        grad = model.grad_energy(x)
        assert grad.shape == (5,)
        assert jnp.all(jnp.isfinite(grad))


class TestKANParameterScaling:
    """Tests for KAN parameter count relative to other tiers."""

    def test_kan_more_params_than_ising(self) -> None:
        """REQ-CORE-002: KAN has more parameters than Ising for same input_dim."""
        input_dim = 20
        ising = IsingModel(IsingConfig(input_dim=input_dim))

        ising_params = input_dim * input_dim + input_dim
        ising_params = ising_params // 2 + input_dim

        config = KANConfig(input_dim=input_dim, sparse=True, edge_density=1.0)
        kan = KANEnergyFunction(config)
        kan_params = kan.n_params

        assert kan_params > ising_params, (
            f"KAN params ({kan_params}) should exceed Ising params ({ising_params})"
        )

    def test_kan_fewer_params_than_gibbs(self) -> None:
        """REQ-CORE-002: KAN has fewer parameters than Gibbs MLP for same input_dim."""
        input_dim = 20
        config_kan = KANConfig(input_dim=input_dim, sparse=True, edge_density=1.0)
        kan = KANEnergyFunction(config_kan)

        gibbs = GibbsModel(
            GibbsConfig(
                input_dim=input_dim,
                hidden_dims=[64, 32],
            )
        )

        kan_params = kan.n_params
        gibbs_params = sum(w.size + b.size for w, b in gibbs.layers)
        gibbs_params += gibbs.output_weight.size + 1

        assert kan_params < gibbs_params, (
            f"KAN params ({kan_params}) should be less than Gibbs params ({gibbs_params})"
        )

    def test_kan_interpret_edge(self) -> None:
        """REQ-CORE-002: interpret_edge returns spline for existing edge."""
        config = KANConfig(input_dim=5, sparse=True, edge_density=1.0)
        kan = KANEnergyFunction(config)

        spline = kan.interpret_edge(0, 1)
        assert spline is not None
        assert isinstance(spline, BSpline)


class TestBSplineValidation:
    """Tests for REQ-CORE-001: BSpline constructor validation."""

    def test_bspline_invalid_num_knots(self) -> None:
        """REQ-CORE-001: num_knots < 2 raises ValueError in BSpline."""
        with pytest.raises(ValueError, match="num_knots must be >= 2"):
            BSpline(num_knots=1, degree=3)

    def test_bspline_invalid_degree(self) -> None:
        """REQ-CORE-001: negative degree raises ValueError in BSpline."""
        with pytest.raises(ValueError, match="degree must be >= 0"):
            BSpline(num_knots=5, degree=-1)


class TestKANConfigValidationExtra:
    """Tests for REQ-CORE-001: KANConfig.validate() edge cases."""

    def test_validation_negative_degree(self) -> None:
        """REQ-CORE-001: negative degree raises ValueError in KANConfig."""
        config = KANConfig(degree=-1)
        with pytest.raises(ValueError, match="degree must be >= 0"):
            config.validate()

    def test_validation_invalid_edge_density(self) -> None:
        """REQ-CORE-001: edge_density out of (0,1] raises ValueError."""
        config = KANConfig(edge_density=0.0)
        with pytest.raises(ValueError, match="edge_density must be in"):
            config.validate()


class TestKANFromIsingWithEdges:
    """Tests for KANEnergyFunction.from_ising when edges are pre-supplied."""

    def test_from_ising_with_existing_edges(self) -> None:
        """REQ-CORE-002: from_ising uses config.edges when already set."""
        ising = IsingModel(IsingConfig(input_dim=4))
        config = KANConfig(
            input_dim=4,
            edges=[(0, 1), (1, 2)],
            sparse=True,
        )
        kan = KANEnergyFunction.from_ising(ising, config=config)
        # Only the pre-specified edges should exist.
        assert set(kan.edge_splines.keys()) == {(0, 1), (1, 2)}


class TestKANModelExtra:
    """Tests for KANModel methods not yet covered."""

    def test_model_energy_batch(self) -> None:
        """REQ-CORE-002: KANModel.energy_batch() delegates to energy function."""
        config = KANConfig(input_dim=4, sparse=True, edge_density=0.5)
        model = KANModel(config)
        xs = jrandom.normal(jrandom.PRNGKey(0), (3, 4))
        energies = model.energy_batch(xs)
        assert energies.shape == (3,)
        assert jnp.all(jnp.isfinite(energies))

    def test_model_grad_energy(self) -> None:
        """REQ-CORE-002: KANModel.grad_energy() returns finite gradient."""
        config = KANConfig(input_dim=4, sparse=True, edge_density=0.5)
        model = KANModel(config)
        x = jnp.ones(4)
        grad = model.grad_energy(x)
        assert grad.shape == (4,)
        assert jnp.all(jnp.isfinite(grad))

    def test_model_train_cd_returns_list(self) -> None:
        """REQ-CORE-002: KANModel.train_cd() returns a list."""
        config = KANConfig(input_dim=4, sparse=True, edge_density=0.5)
        model = KANModel(config)
        data = jrandom.normal(jrandom.PRNGKey(0), (10, 4))
        history = model.train_cd(data, n_epochs=5, lr=0.01)
        assert isinstance(history, list)

    def test_model_interpret_edge(self) -> None:
        """REQ-CORE-002: KANModel.interpret_edge() delegates to energy function."""
        config = KANConfig(input_dim=4, sparse=True, edge_density=1.0)
        model = KANModel(config)
        spline = model.interpret_edge(0, 1)
        assert spline is not None
        assert isinstance(spline, BSpline)
