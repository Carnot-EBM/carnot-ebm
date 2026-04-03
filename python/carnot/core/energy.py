"""Energy function protocol and mixins.

**Researcher summary:**
    Defines the core ``EnergyFunction`` protocol that all EBM tiers (Ising, Gibbs,
    Boltzmann) must satisfy: scalar energy, batched energy, gradient, and input
    dimension. Provides ``AutoGradMixin`` for automatic gradient derivation via
    ``jax.grad``.

**Detailed explanation for engineers:**
    An Energy-Based Model (EBM) assigns a scalar energy value to every possible
    input configuration. Low energy = the model considers that configuration likely;
    high energy = unlikely. This is the foundation of the entire Carnot framework.

    Unlike a classifier that outputs probabilities directly, an EBM outputs an
    unnormalized energy. The probability of a configuration x is proportional to
    exp(-E(x)), which is called the Boltzmann distribution. To *sample* from an
    EBM, we need the gradient of the energy function (dE/dx), which tells us
    which direction to move x to lower the energy (i.e., make it more likely).

    This module defines:
    - ``EnergyFunction``: A Python Protocol (like an interface) that all models
      must implement.
    - ``AutoGradMixin``: A convenience mixin that auto-computes ``grad_energy``
      and ``energy_batch`` so that concrete models only need to implement
      ``energy(x)`` for a single input.

    **What is auto-differentiation (autodiff)?**
    Instead of computing gradients by hand (symbolic math) or by finite differences
    (perturbing inputs slightly), JAX can automatically compute exact gradients of
    any Python function. ``jax.grad(f)`` returns a new function that computes
    df/dx. This is central to EBM training and sampling — we never write gradient
    formulas by hand.

Spec: REQ-CORE-002
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp


@runtime_checkable
class EnergyFunction(Protocol):
    """Protocol (interface) that every Energy-Based Model must implement.

    **Researcher summary:**
        Structural subtyping protocol for EBMs. Requires ``energy``,
        ``energy_batch``, ``grad_energy``, and ``input_dim``.

    **Detailed explanation for engineers:**
        A Protocol in Python is like an interface in Java or a trait in Rust.
        Any class that has these four methods/properties with matching signatures
        is considered an EnergyFunction — no explicit inheritance required.

        The ``@runtime_checkable`` decorator means you can use
        ``isinstance(obj, EnergyFunction)`` at runtime to verify an object
        conforms to this protocol.

        Every model tier (Ising, Gibbs, Boltzmann) implements this protocol,
        which allows samplers and trainers to work with any model
        interchangeably.

    For example::

        from carnot.models.ising import IsingModel, IsingConfig

        model = IsingModel(IsingConfig(input_dim=10))
        assert isinstance(model, EnergyFunction)  # True — duck typing

        x = jnp.ones(10)
        e = model.energy(x)       # scalar energy
        g = model.grad_energy(x)  # gradient vector, same shape as x

    Spec: REQ-CORE-002
    """

    def energy(self, x: jax.Array) -> jax.Array:
        """Compute scalar energy for a single input configuration x.

        **Researcher summary:**
            Maps configuration x in R^d to a scalar energy E(x) in R.

        **Detailed explanation for engineers:**
            Given a 1-D array x of shape (input_dim,), returns a single scalar
            (0-D array) representing the energy. Lower energy means the model
            considers this configuration more probable.

        Args:
            x: A 1-D JAX array of shape (input_dim,).

        Returns:
            A scalar JAX array (shape ()) representing E(x).

        Spec: REQ-CORE-002, SCENARIO-CORE-001
        """
        ...

    def energy_batch(self, xs: jax.Array) -> jax.Array:
        """Compute energy for a batch of inputs simultaneously.

        **Researcher summary:**
            Vectorized energy over a batch; returns one scalar per sample.

        **Detailed explanation for engineers:**
            Given a 2-D array xs of shape (batch_size, input_dim), returns a
            1-D array of shape (batch_size,) where each element is the energy
            of the corresponding input. This is typically implemented via
            ``jax.vmap`` for automatic vectorization.

        Args:
            xs: A 2-D JAX array of shape (batch_size, input_dim).

        Returns:
            A 1-D JAX array of shape (batch_size,).

        Spec: REQ-CORE-002, SCENARIO-CORE-002
        """
        ...

    def grad_energy(self, x: jax.Array) -> jax.Array:
        """Gradient of energy with respect to the input x.

        **Researcher summary:**
            Returns nabla_x E(x), the score function (up to sign).

        **Detailed explanation for engineers:**
            Computes dE/dx — how the energy changes as each element of x
            changes. This gradient vector has the same shape as x. It points
            in the direction of steepest energy *increase*. Samplers use the
            *negative* gradient to move toward lower energy (more probable
            configurations).

            In EBM literature, the negative gradient -dE/dx is called the
            "score function" and is central to both sampling (Langevin
            dynamics) and training (score matching).

        Args:
            x: A 1-D JAX array of shape (input_dim,).

        Returns:
            A JAX array of the same shape as x.

        Spec: REQ-CORE-002, SCENARIO-CORE-003
        """
        ...

    @property
    def input_dim(self) -> int:
        """Number of input dimensions (size of the configuration vector).

        For example, an Ising model over 784 spins has input_dim=784.
        """
        ...


class AutoGradMixin:
    """Mixin that auto-derives grad_energy and energy_batch from energy().

    **Researcher summary:**
        Provides default ``energy_batch`` via ``jax.vmap`` and ``grad_energy``
        via ``jax.grad``. Subclasses only need to implement ``energy(x)``.

    **Detailed explanation for engineers:**
        This mixin uses two key JAX primitives:

        1. **jax.vmap** (vectorized map): Takes a function that operates on a
           single input and returns a function that operates on a *batch* of
           inputs. Internally, JAX compiles this into efficient vectorized
           operations — it is NOT a Python for-loop. Think of it like
           numpy broadcasting but for arbitrary functions.

        2. **jax.grad** (automatic differentiation): Takes a scalar-valued
           function f(x) and returns a new function that computes df/dx
           exactly. No finite differences, no symbolic math — JAX traces
           through the computation graph and applies the chain rule
           automatically.

        By inheriting from this mixin, a model class only needs to define
        ``energy(self, x)`` for a single input, and it automatically gets
        correct batch computation and gradient computation for free.

    For example::

        class MyModel(AutoGradMixin):
            def energy(self, x):
                return jnp.sum(x ** 2)  # simple quadratic bowl

            @property
            def input_dim(self):
                return 10

        model = MyModel()
        x = jnp.ones(10)
        grad = model.grad_energy(x)  # returns 2*x automatically

    Spec: REQ-CORE-002
    """

    def energy(self, x: jax.Array) -> jax.Array:
        """Subclasses must override this to compute scalar energy."""
        raise NotImplementedError

    def energy_batch(self, xs: jax.Array) -> jax.Array:
        """Default batched energy via jax.vmap.

        **How jax.vmap works:**
            ``jax.vmap(self.energy)`` transforms ``energy(x)`` (which expects
            a single 1-D input) into a function that accepts a 2-D batch and
            applies ``energy`` to each row independently. JAX compiles this
            into a single vectorized kernel — no Python loop overhead.

        Args:
            xs: A 2-D JAX array of shape (batch_size, input_dim).

        Returns:
            A 1-D JAX array of shape (batch_size,).

        Spec: SCENARIO-CORE-002
        """
        return jax.vmap(self.energy)(xs)

    def grad_energy(self, x: jax.Array) -> jax.Array:
        """Auto-derived gradient via jax.grad.

        **How jax.grad works:**
            ``jax.grad(self.energy)`` returns a *new function* that computes
            the gradient dE/dx. Internally, JAX records all operations
            performed during the forward pass of ``self.energy(x)`` and then
            applies reverse-mode automatic differentiation (backpropagation)
            to compute the gradient in one backward pass.

            This means: if you change ``energy()``, the gradient automatically
            updates — no manual derivative code needed.

        Args:
            x: A 1-D JAX array of shape (input_dim,).

        Returns:
            A JAX array of the same shape as x, containing dE/dx.

        Spec: SCENARIO-CORE-003
        """
        return jax.grad(self.energy)(x)
