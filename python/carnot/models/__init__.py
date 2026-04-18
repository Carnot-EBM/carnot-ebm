"""Energy Based Model implementations: Ising, Gibbs, Boltzmann tiers + EBT + EORM."""

from carnot.models.boltzmann import BoltzmannConfig, BoltzmannModel
from carnot.models.ebm_cot_calibrator import EBMCoTCalibrator
from carnot.models.ebm_cot_calibrator_v3 import (
    EBMCoTCalibratorV3,
    EPCouplingUpdate,
    SyntheticCoTPairGenerator,
)
from carnot.models.compliance_checker import (
    ComplianceDomain,
    ComplianceEnergyChecker,
    ComplianceExample,
)
from carnot.models.ebt import EBTConfig, EBTransformer
from carnot.models.eorm import CoTEnergyInput, EORMModel, EORMTrainer
from carnot.models.gibbs import GibbsConfig, GibbsModel
from carnot.models.ising import IsingConfig, IsingModel
from carnot.models.kan import (
    BSpline,
    BSplineParams,
    KANConfig,
    KANEnergyFunction,
    KANModel,
)
from carnot.models.kaem_energy import (
    KAEMEnergy,
    UnivariateKAEMLayer,
    benchmark_kaem_vs_mcmc,
)

__all__ = [
    "BoltzmannConfig",
    "BoltzmannModel",
    "EBMCoTCalibrator",
    "EBMCoTCalibratorV3",
    "EPCouplingUpdate",
    "SyntheticCoTPairGenerator",
    "ComplianceDomain",
    "ComplianceEnergyChecker",
    "ComplianceExample",
    "CoTEnergyInput",
    "EBTConfig",
    "EBTransformer",
    "EORMModel",
    "EORMTrainer",
    "GibbsConfig",
    "GibbsModel",
    "IsingConfig",
    "IsingModel",
    "BSpline",
    "BSplineParams",
    "KANConfig",
    "KANEnergyFunction",
    "KANModel",
    "KAEMEnergy",
    "UnivariateKAEMLayer",
    "benchmark_kaem_vs_mcmc",
]
