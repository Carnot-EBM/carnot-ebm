"""Tests for Sudoku-to-Lean4 formal verification bridge.

Spec coverage: REQ-VERIFY-1740, SCENARIO-VERIFY-1740
"""

from __future__ import annotations

import subprocess
from unittest import mock

from carnot.verify.sudoku_lean4 import (
    SudokuLean4Verifier,
    encode_sudoku_solution_as_lean4,
)

# Standard example: puzzle clues and its known correct solution
_CLUES = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9],
]

_SOLUTION = [
    [5, 3, 4, 6, 7, 8, 9, 1, 2],
    [6, 7, 2, 1, 9, 5, 3, 4, 8],
    [1, 9, 8, 3, 4, 2, 5, 6, 7],
    [8, 5, 9, 7, 6, 1, 4, 2, 3],
    [4, 2, 6, 8, 5, 3, 7, 9, 1],
    [7, 1, 3, 9, 2, 4, 8, 5, 6],
    [9, 6, 1, 5, 3, 7, 2, 8, 4],
    [2, 8, 7, 4, 1, 9, 6, 3, 5],
    [3, 4, 5, 2, 8, 6, 1, 7, 9],
]


class TestEncodeSudokuSolutionAsLean4:
    """REQ-VERIFY-1740: Lean 4 code generation from Sudoku solution."""

    def test_output_contains_solution_values(self) -> None:
        """REQ-VERIFY-1740: generated code contains flattened solution digits."""
        code = encode_sudoku_solution_as_lean4(_CLUES, _SOLUTION)
        # First row of solution: 5 3 4 6 7 8 9 1 2
        assert "5, 3, 4, 6, 7, 8, 9, 1, 2" in code

    def test_output_contains_lean_structure_functions(self) -> None:
        """REQ-VERIFY-1740: generated code defines row/col/box helpers."""
        code = encode_sudoku_solution_as_lean4(_CLUES, _SOLUTION)
        assert "def getRow" in code
        assert "def getCol" in code
        assert "def getBox" in code
        assert "def allDistinct" in code
        assert "def structureValid" in code

    def test_output_contains_clue_constraints(self) -> None:
        """REQ-VERIFY-1740: generated code encodes each non-zero clue cell."""
        code = encode_sudoku_solution_as_lean4(_CLUES, _SOLUTION)
        # Clue at (0,0)=5 → flat index 0, value 5
        assert "grid.get! 0 == 5" in code
        # Clue at (0,1)=3 → flat index 1, value 3
        assert "grid.get! 1 == 3" in code

    def test_output_contains_eval_trigger(self) -> None:
        """REQ-VERIFY-1740: generated code has #eval to trigger verification."""
        code = encode_sudoku_solution_as_lean4(_CLUES, _SOLUTION)
        assert "#eval isSudokuValid solution" in code

    def test_output_is_string(self) -> None:
        """REQ-VERIFY-1740: return type is str."""
        code = encode_sudoku_solution_as_lean4(_CLUES, _SOLUTION)
        assert isinstance(code, str)

    def test_no_clues_produces_trivially_true(self) -> None:
        """REQ-VERIFY-1740: empty clue grid produces 'true' as cluesValid body."""
        empty_clues = [[0] * 9 for _ in range(9)]
        code = encode_sudoku_solution_as_lean4(empty_clues, _SOLUTION)
        assert "  true" in code

    def test_solution_list_has_81_elements(self) -> None:
        """REQ-VERIFY-1740: flat list must have exactly 81 comma-separated values."""
        code = encode_sudoku_solution_as_lean4(_CLUES, _SOLUTION)
        # Extract the content of the solution list literal
        start = code.index("def solution : List Nat := [") + len(
            "def solution : List Nat := ["
        )
        end = code.index("]", start)
        values = [v.strip() for v in code[start:end].split(",")]
        assert len(values) == 81


class TestSudokuLean4VerifierAvailability:
    """REQ-VERIFY-1740: lean binary availability detection."""

    @mock.patch("subprocess.run")
    def test_lean4_available_returns_true_when_lean_exits_zero(
        self, mock_run: mock.Mock
    ) -> None:
        """SCENARIO-VERIFY-1740: lean binary present and healthy."""
        mock_run.return_value = mock.Mock(returncode=0)
        verifier = SudokuLean4Verifier()
        assert verifier.lean4_available() is True

    @mock.patch("subprocess.run")
    def test_lean4_available_returns_false_when_lean_exits_nonzero(
        self, mock_run: mock.Mock
    ) -> None:
        """SCENARIO-VERIFY-1740: lean binary present but broken."""
        mock_run.return_value = mock.Mock(returncode=1)
        verifier = SudokuLean4Verifier()
        assert verifier.lean4_available() is False

    @mock.patch("subprocess.run")
    def test_lean4_available_returns_false_when_not_installed(
        self, mock_run: mock.Mock
    ) -> None:
        """SCENARIO-VERIFY-1740: lean binary not on PATH."""
        mock_run.side_effect = FileNotFoundError()
        verifier = SudokuLean4Verifier()
        assert verifier.lean4_available() is False

    @mock.patch("subprocess.run")
    def test_lean4_available_returns_false_on_timeout(
        self, mock_run: mock.Mock
    ) -> None:
        """SCENARIO-VERIFY-1740: lean probe times out."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="lean", timeout=5.0)
        verifier = SudokuLean4Verifier()
        assert verifier.lean4_available() is False


class TestSudokuLean4VerifierSolution:
    """REQ-VERIFY-1740: end-to-end solution verification via Lean4 backend."""

    @mock.patch("carnot.verify.lean4_bridge.Lean4VerifierBackend.verify")
    def test_verify_solution_returns_true_on_lean_success(
        self, mock_verify: mock.Mock
    ) -> None:
        """SCENARIO-VERIFY-1740: lean verifies solution → returns True."""
        mock_verify.return_value = 0.0
        verifier = SudokuLean4Verifier()
        result = verifier.verify_solution(_CLUES, _SOLUTION)
        assert result is True
        mock_verify.assert_called_once()

    @mock.patch("carnot.verify.lean4_bridge.Lean4VerifierBackend.verify")
    def test_verify_solution_returns_false_on_lean_failure(
        self, mock_verify: mock.Mock
    ) -> None:
        """SCENARIO-VERIFY-1740: lean rejects solution → returns False."""
        mock_verify.return_value = float("inf")
        verifier = SudokuLean4Verifier()
        result = verifier.verify_solution(_CLUES, _SOLUTION)
        assert result is False

    @mock.patch("carnot.verify.lean4_bridge.Lean4VerifierBackend.verify")
    def test_verify_solution_passes_full_lean4_program(
        self, mock_verify: mock.Mock
    ) -> None:
        """REQ-VERIFY-1740: backend receives a complete Lean 4 program, not just a snippet."""
        mock_verify.return_value = 0.0
        verifier = SudokuLean4Verifier()
        verifier.verify_solution(_CLUES, _SOLUTION)

        # The code passed to verify must contain the solution definition
        lean_code_arg = mock_verify.call_args[0][0]
        assert "def solution" in lean_code_arg
        assert "def isSudokuValid" in lean_code_arg
        assert "#eval" in lean_code_arg

    @mock.patch("carnot.verify.lean4_bridge.Lean4VerifierBackend.verify")
    def test_verify_solution_returns_false_when_lean_not_installed(
        self, mock_verify: mock.Mock
    ) -> None:
        """SCENARIO-VERIFY-1740: lean not installed → verify_solution is False."""
        # FileNotFoundError inside verify returns float('inf')
        mock_verify.return_value = float("inf")
        verifier = SudokuLean4Verifier()
        result = verifier.verify_solution(_CLUES, _SOLUTION)
        assert result is False

    def test_default_backend_timeout_is_30_seconds(self) -> None:
        """REQ-VERIFY-1740: default backend uses 30-second timeout for lean."""
        verifier = SudokuLean4Verifier()
        assert verifier._backend.timeout_seconds == 30.0
