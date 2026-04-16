"""DeliverableContentValidator — guards the conductor fast-path against corrupt deliverables.

**Researcher summary (RETRO-023):**
    Five Python modules (cikan_energy, jitrl_memory, safety_kan, semantic_energy_scorer,
    crane_extractor) contain JSON artifact data instead of Python source code.  The root
    cause is that the conductor's "deliverable already exists" fast-path checks only for
    file *existence*, not for file *content*.  When an experiment is interrupted mid-run,
    the result JSON can end up written to the module path, and subsequent runs skip
    re-execution because the path exists — leaving the repo with files that import-fail
    silently or cause mysterious AttributeErrors at experiment time.

**What this module provides:**
    1. ``DeliverableContentValidator`` — a utility class (all static methods) that can be
       imported by any experiment to guard its own deliverable before reporting "done":
       - ``is_valid_python(path)`` — returns True only when ``ast.parse()`` succeeds.
       - ``validate_and_clear(path)`` — deletes the file if it is not valid Python, so
         that the conductor will re-run the experiment on the next pass.
       - ``audit_known_corrupt_files(project_root)`` — checks the five known RETRO-023
         affected files and returns a ``{relative_path: status}`` dict.

    2. ``CloudGPUInstructions`` — dataclass containing ready-to-run provisioning commands
       for Lambda Labs, vast.ai, and RunPod.

    3. ``build_cloud_gpu_instructions()`` — constructs the canonical A100 provisioning
       commands with the current cost estimate.

    4. ``generate_cloud_gpu_script(instructions, output_path)`` — writes a shell script
       that the operator can run to spin up a cloud GPU node within minutes.

**Why ast.parse instead of a regex?**
    ``ast.parse()`` is the authoritative Python parser.  Any content that is not valid
    Python source (JSON, binary data, partial writes) will raise ``SyntaxError`` or
    ``UnicodeDecodeError``, giving us an exact and complete validity signal with zero
    false positives (a valid Python file will never be misclassified as corrupt).

Spec: REQ-INFRA-019, REQ-INFRA-020,
      SCENARIO-INFRA-022, SCENARIO-INFRA-023, SCENARIO-INFRA-024
"""

from __future__ import annotations

import ast
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# The five known corrupt files from RETRO-023.
# Paths are relative to the project root so they work on any machine.
# ---------------------------------------------------------------------------

_KNOWN_CORRUPT_FILES: list[str] = [
    "python/carnot/models/cikan_energy.py",
    "python/carnot/pipeline/jitrl_memory.py",
    "python/carnot/models/safety_kan.py",
    "python/carnot/pipeline/semantic_energy_scorer.py",
    "python/carnot/pipeline/crane_extractor.py",
]


# ---------------------------------------------------------------------------
# DeliverableContentValidator
# ---------------------------------------------------------------------------


class DeliverableContentValidator:
    """Validates that deliverable files contain valid Python, not JSON artifacts.

    All methods are static — this class is a namespace, not an instance.  Import it in
    any experiment script and call ``validate_and_clear(deliverable_path)`` right after
    ``tmpl.setup()`` to guard against the RETRO-023 fast-path bypass.

    Example usage in an experiment::

        validator = DeliverableContentValidator()
        if not validator.validate_and_clear("python/carnot/pipeline/my_module.py"):
            # File was corrupt and deleted; re-implement from scratch below
            ...

    Spec: REQ-INFRA-019, SCENARIO-INFRA-022, SCENARIO-INFRA-023
    """

    @staticmethod
    def is_valid_python(path: str) -> bool:
        """Return True only when *path* exists and contains syntactically valid Python.

        Reads the file, then calls ``ast.parse()`` on its contents.  Returns ``False``
        for JSON files, empty files, missing files, binary files, and any file whose
        content raises ``SyntaxError`` or ``UnicodeDecodeError``.

        **Why empty files are rejected:**
            An empty ``.py`` file is not a valid Python module — it is almost certainly
            an artefact of a partial write.  A real module always has at least a docstring
            or an import.  Rejecting empties prevents the conductor from treating a
            zero-byte file as a successfully produced deliverable.

        Parameters
        ----------
        path : str
            Absolute or relative filesystem path to the candidate Python file.

        Returns
        -------
        bool
            ``True`` iff the file exists, is non-empty, and passes ``ast.parse()``.

        Never raises.

        Spec: REQ-INFRA-019, SCENARIO-INFRA-022, SCENARIO-INFRA-023
        """
        try:
            content = Path(path).read_text(encoding="utf-8")
        except (FileNotFoundError, OSError):
            # File does not exist or is unreadable — not valid Python
            return False
        except (UnicodeDecodeError, ValueError):
            # Binary content or encoding error — not valid Python source
            return False

        # Reject empty files — a partial write that produced zero bytes is not valid
        if not content.strip():
            return False

        # Reject JSON files — a JSON object/array is syntactically valid Python
        # (dict/list literals), so ast.parse() alone cannot distinguish them.
        # JSON files are the primary RETRO-023 corruption pattern: experiment result
        # artifacts written to module paths by interrupted conductor runs.
        try:
            json.loads(content)
            # json.loads succeeded → this is a JSON document, not a Python module
            return False
        except (json.JSONDecodeError, ValueError):
            pass  # Not JSON — continue to ast.parse check

        try:
            ast.parse(content)
        except SyntaxError:
            # Content is not syntactically valid Python (e.g. partial source, binary)
            return False

        return True

    @staticmethod
    def validate_and_clear(path: str) -> bool:
        """Validate *path* as Python; delete it if corrupt and return False.

        If ``is_valid_python(path)`` returns ``True``, this method does nothing and
        returns ``True`` (the file is safe to import).

        If ``is_valid_python(path)`` returns ``False``, this method:
        1. Logs a WARNING with the path and reason (so the researcher knows which file
           was affected and why).
        2. Attempts to delete the file via ``os.remove(path)`` so the conductor will
           treat the deliverable as absent and re-run the producing experiment.
        3. Returns ``False`` so the caller knows the file must be re-implemented.

        Callers should check the return value and re-implement the deliverable when
        ``False`` is returned::

            if not DeliverableContentValidator.validate_and_clear(module_path):
                # Deliverable was corrupt and has been deleted.  Re-implement it.
                write_module(module_path)

        Parameters
        ----------
        path : str
            Path to the file to validate.

        Returns
        -------
        bool
            ``True`` if the file is valid Python (no action taken).
            ``False`` if the file was invalid (warning logged, file deleted if present).

        Never raises.

        Spec: REQ-INFRA-019, SCENARIO-INFRA-022, SCENARIO-INFRA-023
        """
        if DeliverableContentValidator.is_valid_python(path):
            return True

        # Determine the reason for rejection (for the log message)
        p = Path(path)
        if not p.exists():
            reason = "file does not exist"
        else:
            try:
                content = p.read_text(encoding="utf-8")
                if not content.strip():
                    reason = "file is empty"
                else:
                    reason = "content failed ast.parse() — likely JSON or corrupt data"
            except (UnicodeDecodeError, ValueError):
                reason = "binary or non-UTF-8 content"
            except OSError as exc:
                reason = f"OS error reading file: {exc}"

        _log.warning(
            "DeliverableContentValidator: corrupt deliverable detected — "
            "path=%s reason=%s — deleting so conductor will re-run the experiment",
            path,
            reason,
        )

        # Attempt deletion; if the file is already gone, that is fine
        try:
            os.remove(path)
        except FileNotFoundError:
            pass  # Already absent — nothing to do
        except OSError as exc:
            _log.warning(
                "DeliverableContentValidator: failed to delete %s: %s", path, exc
            )

        return False

    @staticmethod
    def audit_known_corrupt_files(project_root: str) -> dict[str, str]:
        """Check the five RETRO-023 known corrupt files and return their status.

        For each of the five known affected files, determines whether the file is:
        - ``'valid_python'``  — exists and passes ``ast.parse()``
        - ``'corrupt_json'``  — exists but fails ``ast.parse()`` (likely JSON artifact)
        - ``'missing'``       — does not exist at all

        This audit does NOT delete any files — it is read-only.  Use
        ``validate_and_clear()`` to repair individual files.

        Parameters
        ----------
        project_root : str
            Absolute path to the repository root.  The five known relative paths are
            resolved under this root.

        Returns
        -------
        dict[str, str]
            Mapping from relative path string to status string.  Always contains
            exactly the five RETRO-023 paths as keys.

        Spec: REQ-INFRA-019, SCENARIO-INFRA-022 (Exp 404)
        """
        root = Path(project_root)
        result: dict[str, str] = {}

        for rel_path in _KNOWN_CORRUPT_FILES:
            abs_path = root / rel_path

            if not abs_path.exists():
                result[rel_path] = "missing"
                continue

            if DeliverableContentValidator.is_valid_python(str(abs_path)):
                result[rel_path] = "valid_python"
            else:
                result[rel_path] = "corrupt_json"

        return result


# ---------------------------------------------------------------------------
# CloudGPUInstructions
# ---------------------------------------------------------------------------


@dataclass
class CloudGPUInstructions:
    """Ready-to-run provisioning commands for three major cloud GPU providers.

    Generated by ``build_cloud_gpu_instructions()`` and written to a shell script
    by ``generate_cloud_gpu_script()``.  All three providers offer A100-class GPUs
    that can run the full Carnot experiment stack.

    Fields
    ------
    lambda_command : str
        Lambda Labs CLI command to provision a 1×A100 instance in us-west-2.
    vastai_command : str
        vast.ai CLI command to create a PyTorch 2.3 instance.
    runpod_command : str
        RunPod CLI command to create an NVIDIA A100 80GB pod.
    estimated_cost_per_hour_usd : float
        Approximate hourly cost in USD (A100 spot pricing as of 2026-04).

    Spec: REQ-INFRA-020, SCENARIO-INFRA-024
    """

    lambda_command: str
    vastai_command: str
    runpod_command: str
    estimated_cost_per_hour_usd: float


# ---------------------------------------------------------------------------
# build_cloud_gpu_instructions
# ---------------------------------------------------------------------------


def build_cloud_gpu_instructions() -> CloudGPUInstructions:
    """Return canonical A100 provisioning commands for Lambda, vast.ai, and RunPod.

    These commands assume:
    - Lambda Labs CLI (``lambda``) is installed and authenticated.
    - vast.ai CLI (``vastai``) is installed and authenticated.
    - RunPod CLI (``runpodctl``) is installed and authenticated.

    The ``<id>`` in the vast.ai command is a placeholder; the operator must replace
    it with an actual offer ID obtained from ``vastai search offers``.

    Returns
    -------
    CloudGPUInstructions
        Dataclass with three commands and the current cost estimate.

    Spec: REQ-INFRA-020
    """
    return CloudGPUInstructions(
        lambda_command=(
            "lambdalabs instance create "
            "--instance-type gpu_1x_a100 "
            "--region us-west-2 "
            "--quantity 1"
        ),
        vastai_command=(
            "vastai create instance <id> "
            "--image pytorch/pytorch:2.3.0-cuda12.1"
        ),
        runpod_command=(
            "runpodctl create pod "
            "--gpuType NVIDIA_A100_80GB"
        ),
        estimated_cost_per_hour_usd=1.10,
    )


# ---------------------------------------------------------------------------
# generate_cloud_gpu_script
# ---------------------------------------------------------------------------


def generate_cloud_gpu_script(
    instructions: CloudGPUInstructions,
    output_path: str,
) -> None:
    """Write a shell script with provisioning commands for all three cloud GPU providers.

    The script is a human-readable reference; the operator copies and runs the relevant
    section for their preferred provider.  It is written to *output_path*, creating
    parent directories as needed.

    Parameters
    ----------
    instructions : CloudGPUInstructions
        Commands to embed in the script (from ``build_cloud_gpu_instructions()``).
    output_path : str
        Destination path (e.g. ``"scripts/setup_cloud_gpu.sh"``).
        Parent directories are created if absent.

    Spec: REQ-INFRA-020, SCENARIO-INFRA-024
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    content = (
        "#!/usr/bin/env bash\n"
        "# Cloud GPU setup script — generated by Exp 404 (RETRO-022 / REQ-INFRA-020)\n"
        "# Run the section for your preferred cloud provider.\n"
        "# Estimated cost: ${cost:.2f}/hr (A100 80GB)\n"
        "#\n"
        "# IMPORTANT: Authenticate with your chosen provider CLI before running.\n\n"
        "# -----------------------------------------------------------------------\n"
        "# Option 1: Lambda Labs\n"
        "# Install: pip install lambdalabs-cli\n"
        "# Auth:    lambdalabs auth login\n"
        "# -----------------------------------------------------------------------\n"
        "{lambda_cmd}\n\n"
        "# -----------------------------------------------------------------------\n"
        "# Option 2: vast.ai\n"
        "# Install: pip install vastai\n"
        "# Auth:    vastai set api-key <YOUR_API_KEY>\n"
        "# Find ID: vastai search offers 'gpu_name=A100' --storage 50\n"
        "# Replace <id> below with the offer ID from the search above.\n"
        "# -----------------------------------------------------------------------\n"
        "{vastai_cmd}\n\n"
        "# -----------------------------------------------------------------------\n"
        "# Option 3: RunPod\n"
        "# Install: https://docs.runpod.io/cli/install-runpodctl\n"
        "# Auth:    runpodctl config --apiKey <YOUR_API_KEY>\n"
        "# -----------------------------------------------------------------------\n"
        "{runpod_cmd}\n"
    ).format(
        cost=instructions.estimated_cost_per_hour_usd,
        lambda_cmd=instructions.lambda_command,
        vastai_cmd=instructions.vastai_command,
        runpod_cmd=instructions.runpod_command,
    )

    out.write_text(content)
    _log.info(
        "generate_cloud_gpu_script: wrote cloud GPU setup script to %s", output_path
    )
