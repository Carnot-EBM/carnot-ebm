"""MCMC samplers for Energy Based Models."""

try:
    from .backend import (
        ClutCpuBackend,
        CpuBackend,
        SamplerBackend,
        TsuBackend,
        backend_registry,
        get_backend,
        get_sampler_backend,
    )
    from .adaptive_ising import SelfAdaptiveIsingSampler
    from .dwave_backend import DWaveNealBackend
    from .dwave_sampler import DWaveSampler
    from .discrete_simulated_bifurcation import (
        DiscreteSBConfig,
        DiscreteSBConstraintProblem,
        InertialDiscreteSBConfig,
        run_discrete_sb,
        run_fover_discrete_sb_probe,
        run_fover_inertial_ising_probe,
        run_gibbs_ising_baseline,
        run_inertial_discrete_sb,
    )
    from .fpga_backend import FpgaBackend
    from .fpga_ising import FPGAIsingSampler
    from .equilibrium_matching import EquilibriumMatchingSampler
    from .continuous_gumbel import ContinuousGumbelSampler
    from .casal import CASALSampler, casal_sample
    from .clut_random_variate import ClutLogisticBernoulliSampler
    from .gpu_oim_simulator import (
        GPUOscillatorIsingSimulator,
        JEPARetrainResult,
        OIMSpeedupResult,
    )
    from .hmc import HMCSampler
    from .knuth_yao import KnuthYaoSampler
    from .langevin import LangevinSampler
    from .projected_langevin import ProjectedLangevinSampler
    from .parallel_dense_ising import (
        ParallelDenseIsingConfig,
        ParallelDenseIsingInertia,
    )
    from .parallel_ising import AnnealingSchedule, ParallelIsingSampler
    from .mode_jump_rust_backend import ModeJumpRustBackend
    from .one_axis_rust_backend import OneAxisRustBackend
    from .phase4_sampler import Phase4Sampler
    from .potts_sampler import PottsSampler
    from .synchronous_ising import SynchronousIsingSampler
    from .thrml_backend import ThrmlSamplerBackend
except ModuleNotFoundError as exc:
    if exc.name != "jax":
        raise
    from .potts_sampler import PottsSampler

    __all__ = ["PottsSampler"]
else:
    __all__ = [
        "LangevinSampler",
        "ProjectedLangevinSampler",
        "EquilibriumMatchingSampler",
        "ContinuousGumbelSampler",
        "CASALSampler",
        "ClutLogisticBernoulliSampler",
        "ClutCpuBackend",
        "ModeJumpRustBackend",
        "OneAxisRustBackend",
        "HMCSampler",
        "KnuthYaoSampler",
        "SelfAdaptiveIsingSampler",
        "ParallelIsingSampler",
        "Phase4Sampler",
        "AnnealingSchedule",
        "FPGAIsingSampler",
        "FpgaBackend",
        "SamplerBackend",
        "CpuBackend",
        "TsuBackend",
        "DWaveNealBackend",
        "DWaveSampler",
        "get_backend",
        "get_sampler_backend",
        "backend_registry",
        "GPUOscillatorIsingSimulator",
        "OIMSpeedupResult",
        "JEPARetrainResult",
        "SynchronousIsingSampler",
        "PottsSampler",
        "ThrmlSamplerBackend",
        "ParallelDenseIsingConfig",
        "ParallelDenseIsingInertia",
        "DiscreteSBConfig",
        "DiscreteSBConstraintProblem",
        "InertialDiscreteSBConfig",
        "run_discrete_sb",
        "run_fover_discrete_sb_probe",
        "run_fover_inertial_ising_probe",
        "run_gibbs_ising_baseline",
        "run_inertial_discrete_sb",
        "casal_sample",
    ]
