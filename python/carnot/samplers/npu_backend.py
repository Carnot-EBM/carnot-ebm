"""AMD XDNA NPU backend for JEPAViolationPredictor inference.

**Researcher summary:**
    Provides ``NpuJEPAPredictor`` — a drop-in replacement for direct JAX-based
    JEPA inference that routes execution to the AMD Ryzen AI NPU when the
    ``VitisAIExecutionProvider`` is available in onnxruntime. Falls back to
    ``CPUExecutionProvider`` transparently if the NPU software stack is absent.

**Detailed explanation for engineers:**
    The AMD Ryzen AI NPU (Neural Processing Unit) uses AMD's XDNA architecture,
    which is based on AI Engine (AIE) tiles — small, programmable VLIW processors
    arranged in a 2D array on-chip. The XDNA NPU is designed for sustained low-
    power inference of small-to-medium neural networks (a perfect fit for the
    JEPA predictor's 256→64→32→3 MLP).

    **Software stack required for NPU execution:**

    1. ``amdxdna`` kernel module (upstream since Linux 6.11, also in cachyos)
       — registers the NPU as ``/dev/accel*`` DRM device.

    2. AMD Ryzen AI software stack:
       - ``onnxruntime-vitisai`` (conda: ``conda install -c amd onnxruntime-vitisai``)
       - This package extends onnxruntime with ``VitisAIExecutionProvider``
       - Vitis AI compiler is bundled and compiles ONNX graphs to AIE instructions

    3. The ONNX model (exported by Exp 146 to results/jepa_predictor_146.onnx)
       is passed to onnxruntime; the provider compiles it at session creation time
       and caches the compiled form. Subsequent calls skip compilation.

    **Current status (Exp 179):**
        NPU hardware IS present (/dev/accel0, amdxdna module loaded). ✅
        Git symlinks in RyzenAI-SW/linux/onnx/ryzen14/ FIXED. ✅
        VitisAI EP .so files in .venv-npu/ PRESENT but built for Python 3.10. ⚠️
        Running Python 3.12 — EP requires libpython3.10.so.1.0, not found.
        REMAINING BLOCKER: Need AMD VitisAI wheel for Python 3.12.
        Download from: ryzenai.docs.amd.com/en/latest/inst.html
        Provider name CORRECTED: 'VitisAIExecutionProvider' (was wrong in Exp 146).

    **Provider name (CORRECTED in Exp 179):**
        The correct ONNX Runtime provider name is ``VitisAIExecutionProvider``,
        NOT ``AMDXDNAExecutionProvider`` as assumed in Exp 146. AMD's own
        example code (RyzenAI-SW/example/image_classification/utils.py) uses
        ``VitisAIExecutionProvider``.

    **How this class works:**
        - At construction time, attempts to create an onnxruntime session with
          ``VitisAIExecutionProvider``. If that fails (missing provider or no
          device), falls back to ``CPUExecutionProvider`` and logs a warning.
        - ``predict(x)`` accepts a (256,) numpy array and returns domain probs.
        - ``is_high_risk(x, threshold)`` replicates JEPAViolationPredictor's
          boolean gate interface.
        - ``backend_name`` property identifies which execution path is active.

    **Why not use JAX for NPU?**
        JAX targets CUDA/ROCm/XLA backends. AMD's XDNA NPU is not a CUDA or
        ROCm device — it's a separate accelerator with its own ISA (AIE
        instructions). ONNX Runtime with VitisAIExecutionProvider is the
        officially supported inference path for this hardware.

Spec: REQ-JEPA-001 (Tier 3 predictor), research-program.md §"Next Milestone Focus" #5
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Default ONNX model path produced by Experiment 146
_DEFAULT_ONNX_PATH = (
    Path(__file__).parent.parent.parent.parent / "results" / "jepa_predictor_146.onnx"
)

# Domain names must match JEPAViolationPredictor.DOMAINS order
DOMAINS = ["arithmetic", "code", "logic"]
EMBED_DIM = 256


class NpuJEPAPredictor:
    """JEPA Tier-3 predictor backed by AMD XDNA NPU (falls back to CPU).

    **Researcher summary:**
        Wraps the ONNX export of the JEPAViolationPredictor for hardware-
        accelerated inference. When the AMD Ryzen AI software stack is
        installed, routes inference to the NPU for sub-millisecond latency.
        Falls back to CPU transparently when NPU is unavailable.

    **Detailed explanation for engineers:**
        This class mirrors the public API of ``JEPAViolationPredictor``:
        - ``predict(x)`` → dict[str, float] (per-domain violation probabilities)
        - ``is_high_risk(x, threshold)`` → bool
        - ``backend_name`` → "npu" | "cpu_fallback"

        It does NOT implement the full EnergyFunction protocol (no ``energy()``,
        ``energy_batch()``, ``grad_energy()``) because ONNX inference is not
        differentiable through JAX autodiff. Use JAX-based JEPAViolationPredictor
        when gradients are needed (e.g., Langevin sampling).

        **NPU path blocked until VitisAIExecutionProvider available:**
            When ``VitisAIExecutionProvider`` is absent from onnxruntime,
            the constructor logs a warning and silently uses CPUExecutionProvider.
            No user code changes required when the NPU stack is later installed —
            just reinstantiate and the NPU path activates automatically.

    Example::

        from carnot.samplers.npu_backend import NpuJEPAPredictor

        predictor = NpuJEPAPredictor()  # auto-selects NPU or CPU
        print(predictor.backend_name)   # "npu" or "cpu_fallback"
        probs = predictor.predict(embedding)  # {"arithmetic": 0.82, ...}
        if predictor.is_high_risk(embedding):
            restart_generation()

    Spec: REQ-JEPA-001
    """

    def __init__(
        self,
        onnx_path: str | Path | None = None,
        prefer_npu: bool = True,
        vaip_config: str | Path | None = None,
    ) -> None:
        """Load the ONNX model and create an inference session.

        **Detailed explanation for engineers:**
            Session creation strategy:
            1. If ``prefer_npu=True`` (default), first tries to create a session
               with ``["VitisAIExecutionProvider", "CPUExecutionProvider"]``.
               VitisAI EP requires provider_options with config_file pointing to
               vaip_config_npu_2_3.json (AMD VitisAI pass configuration). The
               provider compiles the graph to AIE instructions at session creation
               time (takes ~5-60s on first call, then cached in cacheDir).
            2. If that raises any exception or the provider is absent, falls back
               to ``["CPUExecutionProvider"]`` and sets
               ``_active_backend = "cpu_fallback"``.
            3. If ``prefer_npu=False``, goes straight to CPU.

            **VitisAI EP provider options (Exp 179):**
                - ``config_file``: path to vaip_config_npu_2_3.json
                  (in ~/github.com/amd/RyzenAI-SW/.../ryzen14/)
                - ``cacheDir``: directory for compiled model cache
                - ``cacheKey``: unique key per model

            The ONNX model must be exported first by running:
                ``JAX_PLATFORMS=cpu python scripts/experiment_146_npu.py``

        Args:
            onnx_path: Path to the ONNX model file. Defaults to
                ``results/jepa_predictor_146.onnx`` (created by Exp 146).
            prefer_npu: If True, attempt NPU provider before CPU fallback.
            vaip_config: Path to vaip_config_npu_2_3.json. If None, searched
                in the standard AMD RyzenAI-SW location.
        """
        import onnxruntime as ort  # type: ignore[import]

        model_path = Path(onnx_path) if onnx_path else _DEFAULT_ONNX_PATH
        if not model_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found at {model_path}. "
                "Run: JAX_PLATFORMS=cpu python scripts/experiment_146_npu.py"
            )

        available_providers = ort.get_available_providers()
        self._active_backend = "cpu_fallback"

        # Locate vaip config for VitisAI EP
        _vaip_default = (
            Path.home()
            / "github.com"
            / "amd"
            / "RyzenAI-SW"
            / "Ryzen-AI-CVML-Library"
            / "linux"
            / "onnx"
            / "ryzen14"
            / "vaip_config_npu_2_3.json"
        )
        vaip_config_path = Path(vaip_config) if vaip_config else _vaip_default

        if prefer_npu and "VitisAIExecutionProvider" in available_providers:
            _cache_dir = model_path.parent / "npu_cache"
            _cache_dir.mkdir(exist_ok=True)
            _provider_options = [{
                "config_file": str(vaip_config_path),
                "cacheDir": str(_cache_dir),
                "cacheKey": "jepa_predictor_npu",
            }]
            try:
                self._session = ort.InferenceSession(
                    str(model_path),
                    providers=["VitisAIExecutionProvider", "CPUExecutionProvider"],
                    provider_options=_provider_options,
                )
                self._active_backend = "npu"
                logger.info("NpuJEPAPredictor: VitisAIExecutionProvider active.")
            except Exception as exc:
                # NPU provider present but session creation failed (e.g. device
                # busy, driver error). Fall back gracefully.
                logger.warning(
                    "NpuJEPAPredictor: NPU session creation failed (%s). "
                    "Falling back to CPU.",
                    exc,
                )
                self._session = ort.InferenceSession(
                    str(model_path),
                    providers=["CPUExecutionProvider"],
                )
        else:
            if prefer_npu and "VitisAIExecutionProvider" not in available_providers:
                # NPU path blocked — explain what's missing
                logger.warning(
                    "NpuJEPAPredictor: VitisAIExecutionProvider not found in "
                    "onnxruntime (available: %s). "
                    "NPU path blocked until AMD Ryzen AI software stack is "
                    "installed: conda install -c amd onnxruntime-vitisai. "
                    "See: https://ryzenai.docs.amd.com/en/latest/inst.html. "
                    "Falling back to CPUExecutionProvider.",
                    available_providers,
                )
            self._session = ort.InferenceSession(
                str(model_path),
                providers=["CPUExecutionProvider"],
            )

        self._input_name: str = self._session.get_inputs()[0].name
        logger.info(
            "NpuJEPAPredictor: session ready on backend=%s, model=%s",
            self._active_backend,
            model_path,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def backend_name(self) -> str:
        """Active execution backend: ``"npu"`` or ``"cpu_fallback"``.

        **Detailed explanation for engineers:**
            "npu" means VitisAIExecutionProvider was successfully activated.
            "cpu_fallback" means the NPU provider was unavailable or failed,
            so CPUExecutionProvider is being used instead.
        """
        return self._active_backend

    def predict(self, partial_embedding: np.ndarray) -> dict[str, float]:
        """Predict per-domain violation probabilities via ONNX inference.

        **Detailed explanation for engineers:**
            Reshapes the input to (1, 256) for batch-1 inference, runs the
            ONNX session, and returns a dict mapping domain → probability.
            The ONNX graph includes the sigmoid layer, so output values are
            already in [0, 1].

        Args:
            partial_embedding: 1-D numpy array of shape (256,), dtype float32.
                Produced by ``RandomProjectionEmbedding.encode(partial_text)``.

        Returns:
            Dict mapping domain name → float probability in [0, 1].
            E.g. {"arithmetic": 0.73, "code": 0.12, "logic": 0.08}

        Spec: REQ-JEPA-001, SCENARIO-JEPA-001
        """
        x = np.asarray(partial_embedding, dtype=np.float32).reshape(1, EMBED_DIM)
        outputs = self._session.run(None, {self._input_name: x})
        probs = outputs[0][0]  # shape (3,) after batch-1 squeeze
        return {domain: float(probs[i]) for i, domain in enumerate(DOMAINS)}

    def is_high_risk(
        self,
        partial_embedding: np.ndarray,
        threshold: float = 0.5,
    ) -> bool:
        """Return True if any domain violation probability exceeds threshold.

        **Detailed explanation for engineers:**
            Drop-in replacement for JEPAViolationPredictor.is_high_risk().
            Same semantics: max per-domain probability ≥ threshold → True.
            Lower threshold = stricter gate (more early-exits, fewer missed
            violations).

        Args:
            partial_embedding: 1-D array of shape (256,).
            threshold: Violation probability above which we flag as high-risk.
                Default 0.5.

        Returns:
            True if max(predict(x).values()) >= threshold, else False.

        Spec: REQ-JEPA-001, SCENARIO-JEPA-002
        """
        probs = self.predict(partial_embedding)
        return max(probs.values()) >= threshold

    def __repr__(self) -> str:
        return f"NpuJEPAPredictor(backend={self._active_backend!r})"
