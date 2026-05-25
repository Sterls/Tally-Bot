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
BATCH_SIZE = 8  # leaf nodes to evaluate per GPU call


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
    device = next(model.parameters()).device
    x = torch.from_numpy(board_to_tensor(board)).unsqueeze(0).float().to(device)
    with torch.no_grad():
        value, logits = model(x)
    v = value.item()
    if board.turn == chess.BLACK:
        v = -v
    return v, legal_policy(board, logits.squeeze(0))


def _infer_batch(boards: list, model) -> list:
    """Evaluate multiple boards in one forward pass. Returns list of (value, policy)."""
    device = next(model.parameters()).device
    xs = torch.stack([
        torch.from_numpy(board_to_tensor(b)).float()
        for b in boards
    ]).to(device)
    with torch.no_grad():
        values, logits = model(xs)
    results = []
    for i, board in enumerate(boards):
        v = values[i].item()
        if board.turn == chess.BLACK:
            v = -v
        results.append((v, legal_policy(board, logits[i])))
    return results


def _terminal_value(board: chess.Board) -> float:
    """Value from the current player's perspective at a terminal position."""
    if board.is_checkmate():
        return -1.0  # player to move was checkmated
    return 0.0       # stalemate / other draw


def _select_and_apply_virtual_loss(root, board):
    """
    Tree selection from root. Applies virtual loss (N+=1, W-=1) along the path
    so that subsequent selections in the same batch avoid this path.
    Returns (path, sim_board).
    """
    node = root
    sim_board = board.copy(stack=False)
    path = [node]

    while node.expanded and not sim_board.is_game_over():
        node = node.best_child()
        sim_board.push(node.move)
        path.append(node)

    for n in path:
        n.N += 1
        n.W -= 1

    return path, sim_board


def _backup(path, leaf_v):
    """
    Undo virtual loss and backup leaf_v up the path.
    leaf_v is from the leaf node's current-player perspective.
    Net effect: N stays +1 (from virtual loss), W += leaf_v (alternating sign).
    """
    v = leaf_v
    for n in reversed(path):
        n.W += 1 + v  # undo VL (-1) and add real value
        v = -v


def search(
    board: chess.Board,
    model,
    n_simulations: int = 50,
    add_noise: bool = True,
) -> dict:
    """
    Run MCTS from board. Returns {move: visit_probability} over legal moves.
    Evaluates BATCH_SIZE leaf nodes per GPU call using virtual loss.
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

    sims_done = 1
    while sims_done < n_simulations:
        batch = min(BATCH_SIZE, n_simulations - sims_done)

        nonterminal = []  # (path, sim_board)
        terminal = []     # (path, terminal_v)

        for _ in range(batch):
            path, sim_board = _select_and_apply_virtual_loss(root, board)
            if sim_board.is_game_over():
                terminal.append((path, _terminal_value(sim_board)))
            else:
                nonterminal.append((path, sim_board))

        if nonterminal:
            inferred = _infer_batch([b for _, b in nonterminal], model)
            for (path, _), (leaf_v, child_policy) in zip(nonterminal, inferred):
                node = path[-1]
                for move, prior in child_policy.items():
                    node.children[move] = MCTSNode(parent=node, move=move, prior=prior)
                node.expanded = True
                _backup(path, leaf_v)

        for path, leaf_v in terminal:
            _backup(path, leaf_v)

        sims_done += batch

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
        nonterminal = []
        terminal = []

        for _ in range(BATCH_SIZE):
            if time.monotonic() >= deadline:
                break
            path, sim_board = _select_and_apply_virtual_loss(root, board)
            if sim_board.is_game_over():
                terminal.append((path, _terminal_value(sim_board)))
            else:
                nonterminal.append((path, sim_board))

        if nonterminal:
            inferred = _infer_batch([b for _, b in nonterminal], model)
            for (path, _), (leaf_v, child_policy) in zip(nonterminal, inferred):
                node = path[-1]
                for move, prior in child_policy.items():
                    node.children[move] = MCTSNode(parent=node, move=move, prior=prior)
                node.expanded = True
                _backup(path, leaf_v)

        for path, leaf_v in terminal:
            _backup(path, leaf_v)

        sims += len(nonterminal) + len(terminal)

    print(f"  [{sims} sims in {think_ms}ms]")
    if not root.children:
        return None
    return max(root.children, key=lambda m: root.children[m].N)
