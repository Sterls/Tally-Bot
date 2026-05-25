import math
import time
import chess
import torch
import numpy as np

from engine.features import board_to_tensor
from engine.moves import legal_policy

C_PUCT = 1.5
DIRICHLET_ALPHA = 0.3
DIRICHLET_EPS = 0.25


class MCTSNode:
    __slots__ = ("parent", "move", "prior", "children", "N", "W", "expanded")

    def __init__(self, parent=None, move=None, prior: float = 0.0):
        self.parent = parent
        self.move = move
        self.prior = prior
        self.children: dict = {}
        self.N = 0
        self.W = 0.0
        self.expanded = False

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N else 0.0

    def ucb(self, n_parent: int) -> float:
        return self.Q + C_PUCT * self.prior * math.sqrt(n_parent) / (1 + self.N)

    def best_child(self) -> "MCTSNode":
        n = self.N
        return max(self.children.values(), key=lambda c: c.ucb(n))


def _infer(board: chess.Board, model) -> tuple:
    """
    Run the network. Returns (value, policy) where value is from the
    current player's perspective (+1 = current player wins).
    """
    device = next(model.parameters()).device
    x = torch.from_numpy(board_to_tensor(board)).unsqueeze(0).float().to(device)
    with torch.no_grad():
        value, logits = model(x)
    v = value.item()
    if board.turn == chess.BLACK:
        v = -v
    return v, legal_policy(board, logits.squeeze(0))


def _terminal_value(board: chess.Board) -> float:
    """Value from the current player's perspective at a terminal position."""
    if board.is_checkmate():
        return -1.0  # player to move was checkmated
    return 0.0       # stalemate / other draw


def search(
    board: chess.Board,
    model,
    n_simulations: int = 50,
    add_noise: bool = True,
) -> dict:
    """
    Run MCTS from board. Returns {move: visit_probability} over legal moves.
    All values in the tree are from each node's current-player perspective.
    """
    if board.is_game_over():
        return {}

    root = MCTSNode()
    v, policy = _infer(board, model)

    if add_noise and policy:
        moves = list(policy.keys())
        noise = np.random.dirichlet([DIRICHLET_ALPHA] * len(moves))
        policy = {
            m: (1 - DIRICHLET_EPS) * policy[m] + DIRICHLET_EPS * float(n)
            for m, n in zip(moves, noise)
        }

    for move, prior in policy.items():
        root.children[move] = MCTSNode(parent=root, move=move, prior=prior)
    root.expanded = True
    root.N = 1
    root.W = v

    for _ in range(n_simulations - 1):
        node = root
        sim_board = board.copy(stack=False)
        path = [node]

        # Selection
        while node.expanded and not sim_board.is_game_over():
            node = node.best_child()
            sim_board.push(node.move)
            path.append(node)

        # Expansion / evaluation
        if sim_board.is_game_over():
            leaf_v = _terminal_value(sim_board)
        else:
            leaf_v, child_policy = _infer(sim_board, model)
            for move, prior in child_policy.items():
                node.children[move] = MCTSNode(parent=node, move=move, prior=prior)
            node.expanded = True

        # Backup — alternate sign going up (parent is the opponent)
        v = leaf_v
        for n in reversed(path):
            n.N += 1
            n.W += v
            v = -v

    total = sum(c.N for c in root.children.values())
    if total == 0:
        return {}
    return {m: c.N / total for m, c in root.children.items()}


def best_move(board: chess.Board, model, n_simulations: int = 50):
    """Select best move by MCTS visit counts (no exploration noise)."""
    policy = search(board, model, n_simulations, add_noise=False)
    return max(policy, key=policy.get) if policy else None


def best_move_timed(board: chess.Board, model, think_ms: int):
    """Run MCTS for think_ms milliseconds and return the best move."""
    if board.is_game_over():
        return None

    root = MCTSNode()
    v, policy = _infer(board, model)

    for move, prior in policy.items():
        root.children[move] = MCTSNode(parent=root, move=move, prior=prior)
    root.expanded = True
    root.N = 1
    root.W = v

    deadline = time.monotonic() + think_ms / 1000
    sims = 0
    while time.monotonic() < deadline:
        node = root
        sim_board = board.copy(stack=False)
        path = [node]

        while node.expanded and not sim_board.is_game_over():
            node = node.best_child()
            sim_board.push(node.move)
            path.append(node)

        if sim_board.is_game_over():
            leaf_v = _terminal_value(sim_board)
        else:
            leaf_v, child_policy = _infer(sim_board, model)
            for move, prior in child_policy.items():
                node.children[move] = MCTSNode(parent=node, move=move, prior=prior)
            node.expanded = True

        v = leaf_v
        for n in reversed(path):
            n.N += 1
            n.W += v
            v = -v
        sims += 1

    print(f"  [{sims} sims in {think_ms}ms]")
    if not root.children:
        return None
    return max(root.children, key=lambda m: root.children[m].N)
