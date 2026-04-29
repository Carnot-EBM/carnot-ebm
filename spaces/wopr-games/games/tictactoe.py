"""Tic-Tac-Toe cartridge — the literal WarGames game.

In the 1983 film, Joshua (the WOPR) plays itself at Tic-Tac-Toe over
and over and concludes that the game is a draw — which generalizes to
its conclusion that Global Thermonuclear War is also unwinnable. This
cartridge is the cultural anchor: every visitor recognises it, and
"have it play itself" is the on-screen WarGames moment.

Energy formulation:
  E(state) = 0          if game is over
           = -10 * advantage  otherwise

where advantage is determined by perfect-play minimax. Carnot's "step"
is one move by the current player, where each player plays optimally.
With perfect play on both sides, every game ends in a draw — exactly
the lesson Joshua learns.
"""

from __future__ import annotations

from dataclasses import dataclass

from games._base import StepResult, WOPRGame

# Board: list of 9 chars, each ' ', 'X', or 'O'. Index layout:
#   0 1 2
#   3 4 5
#   6 7 8
WIN_LINES: list[tuple[int, int, int]] = [
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),  # rows
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),  # cols
    (0, 4, 8),
    (2, 4, 6),  # diagonals
]


@dataclass
class TicTacToeState:
    board: list[str]  # length 9
    next_player: str  # 'X' or 'O'

    def clone(self) -> TicTacToeState:
        return TicTacToeState(board=self.board[:], next_player=self.next_player)


def winner(board: list[str]) -> str | None:
    """Return 'X', 'O', or None if no winner yet."""
    for a, b, c in WIN_LINES:
        if board[a] != " " and board[a] == board[b] == board[c]:
            return board[a]
    return None


def is_draw(board: list[str]) -> bool:
    return winner(board) is None and " " not in board


def is_terminal(board: list[str]) -> bool:
    return winner(board) is not None or " " not in board


def _minimax(board: list[str], player: str) -> tuple[int, int | None]:
    """Return (score, best_move). Score: +1 X-wins, -1 O-wins, 0 draw.

    Joshua's algorithm: pure minimax with no pruning needed (game is tiny).
    """
    w = winner(board)
    if w == "X":
        return 1, None
    if w == "O":
        return -1, None
    if " " not in board:
        return 0, None

    best_score = -2 if player == "X" else 2
    best_move: int | None = None

    for i in range(9):
        if board[i] != " ":
            continue
        board[i] = player
        opponent = "O" if player == "X" else "X"
        score, _ = _minimax(board, opponent)
        board[i] = " "

        if player == "X" and score > best_score or player == "O" and score < best_score:
            best_score = score
            best_move = i

    return best_score, best_move


def tictactoe_energy(state: TicTacToeState) -> float:
    """Energy = 0 when terminal; otherwise -10 * (score for next player).

    Carnot's energy descends to 0 as the game completes. With perfect
    play on both sides, the score stays at 0 (the WarGames result).
    """
    if is_terminal(state.board):
        return 0.0
    score, _ = _minimax(state.board[:], state.next_player)
    # Negative-leaning energy because we "want" lower energy.
    return -10.0 * abs(score) if state.next_player == "X" else -10.0 * abs(score)


class TicTacToeGame(WOPRGame[TicTacToeState, int]):
    name = "TIC-TAC-TOE"
    description = "JOSHUA PLAYS ITSELF. SHALL WE PLAY A GAME?"
    accent_color = "#39ff14"

    def initial_state(self) -> TicTacToeState:
        return TicTacToeState(board=[" "] * 9, next_player="X")

    def energy(self, state: TicTacToeState) -> float:
        return tictactoe_energy(state)

    def is_solved(self, state: TicTacToeState) -> bool:
        """A "solved" game is a terminal one (won, lost, or drawn).

        Joshua's lesson: with perfect play, every game terminates in a
        draw — the energy goes to 0 not because someone won, but because
        the game is provably unwinnable.
        """
        return is_terminal(state.board)

    def available_actions(self, state: TicTacToeState) -> list[int]:
        return [i for i, v in enumerate(state.board) if v == " "]

    def apply_action(self, state: TicTacToeState, action: int) -> TicTacToeState:
        new_state = state.clone()
        if 0 <= action < 9 and new_state.board[action] == " ":
            new_state.board[action] = new_state.next_player
            new_state.next_player = "O" if new_state.next_player == "X" else "X"
        return new_state

    def carnot_step(self, state: TicTacToeState, iteration: int) -> StepResult[TicTacToeState]:
        """Both players use minimax — Joshua plays itself. The game
        will reach a draw, demonstrating the WarGames lesson on screen.
        """
        if is_terminal(state.board):
            w = winner(state.board)
            if w:
                annotation = f"PLAYER {w} WINS. UNEXPECTED."
            else:
                annotation = "DRAW. THE ONLY WINNING MOVE IS NOT TO PLAY."
            return StepResult(
                state=state,
                energy=0.0,
                iteration=iteration,
                is_solved=True,
                annotation=annotation,
            )

        # Pick the optimal move for the current player
        _, move = _minimax(state.board[:], state.next_player)
        if move is None:
            return StepResult(
                state=state,
                energy=0.0,
                iteration=iteration,
                is_solved=True,
                annotation="NO MOVES.",
            )

        new_state = self.apply_action(state, move)
        new_energy = tictactoe_energy(new_state)
        annotation = f"JOSHUA PLAYS {state.next_player} AT POSITION {move}. ANALYZING NEXT MOVE..."

        return StepResult(
            state=new_state,
            energy=new_energy,
            iteration=iteration,
            is_solved=is_terminal(new_state.board),
            annotation=annotation,
        )

    def visualize(self, state: TicTacToeState, energy: float) -> str:
        rows_html = []
        for r in range(3):
            cells_html = []
            for c in range(3):
                idx = r * 3 + c
                v = state.board[idx]
                color = "#39ff14" if v == "X" else ("#ff3939" if v == "O" else "#1a3a1a")
                cells_html.append(
                    f'<td style="width:60px;height:60px;text-align:center;'
                    f"font-family:JetBrains Mono,monospace;font-size:36px;"
                    f"font-weight:bold;color:{color};"
                    f'border:2px solid #39ff14;background:#000;">'
                    f"{v if v != ' ' else '·'}</td>"
                )
            rows_html.append("<tr>" + "".join(cells_html) + "</tr>")

        table = (
            '<table style="border-collapse:collapse;background:#000;'
            'padding:8px;">' + "".join(rows_html) + "</table>"
        )
        return table
