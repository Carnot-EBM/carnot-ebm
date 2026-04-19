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
from carnot.models.jepa_retrain_v2 import (
    CoTPairQuality,
    CoTPairQualityFilter,
    JEPAQualityAugmentor,
    JEPARetrainV2Result,
)
from carnot.models.gibbs import GibbsConfig, GibbsModel
from carnot.models.ising import IsingConfig, IsingModel
from carnot.models.kan import (
    BSpline,
    BSplineParams,
    KANConfig,
    KANEnergyFunction,
    KANModel,
)
from carnot.models.jepa_curriculum_diagnostic import CorpusAnalysis, JEPACurriculumDiagnostic
from carnot.models.jepa_curriculum_trainer import (
    CurriculumStageResult,
    JEPACurriculumTrainer,
    JEPARetrainV3Result,
)
from carnot.models.kaem_crossover import KAEMCrossoverResult
from carnot.models.kaem_distribution_benchmark import (
    DistributionFamilyResult,
    KAEMDistributionBenchmark,
)
from carnot.models.kaem_extended_result import KAEMExtendedResult
from carnot.models.kaem_energy import (
    KAEMEnergy,
    UnivariateKAEMLayer,
    benchmark_kaem_vs_mcmc,
)
from carnot.models.cikan_energy import (
    CIKANEnergy,
    CIKANLayer,
    ConstraintBoundary,
)
from carnot.models.lowrank_kaem import (
    LowRankKAEMEnergy,
    LowRankProjector,
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
    "CoTPairQuality",
    "CoTPairQualityFilter",
    "JEPAQualityAugmentor",
    "JEPARetrainV2Result",
    "GibbsConfig",
    "GibbsModel",
    "IsingConfig",
    "IsingModel",
    "BSpline",
    "BSplineParams",
    "KANConfig",
    "KANEnergyFunction",
    "KANModel",
    "CorpusAnalysis",
    "JEPACurriculumDiagnostic",
    "CurriculumStageResult",
    "JEPACurriculumTrainer",
    "JEPARetrainV3Result",
    "KAEMCrossoverResult",
    "DistributionFamilyResult",
    "KAEMDistributionBenchmark",
    "KAEMExtendedResult",
    "KAEMEnergy",
    "UnivariateKAEMLayer",
    "benchmark_kaem_vs_mcmc",
    "CIKANEnergy",
    "CIKANLayer",
    "ConstraintBoundary",
    "LowRankKAEMEnergy",
    "LowRankProjector",
]
