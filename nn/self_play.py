import math
import os
import numpy as np
import torch
import chess

from engine.features import board_to_tensor
from engine.moves import policy_to_tensor
from engine import mcts
from engine.search import evaluate as static_eval

MAX_MOVES = 200
TEMP_THRESHOLD = 20  # sample proportionally for first N moves, then argmax
STATIC_EVAL_SCALE = 600  # centipawns; tanh(600cp/600) ≈ 0.76 for a queen-up position


def play_game(model, n_simulations: int = 50) -> tuple:
    """
    Play one self-play game using MCTS.
    Returns (tensors, policy_targets, outcome).
    outcome: +1 white wins, -1 black wins, 0 draw.
    """
    board = chess.Board()
    tensors = []
    policy_targets = []

    for move_num in range(MAX_MOVES):
        if board.is_game_over():
            break

        policy = mcts.search(board, model, n_simulations, add_noise=True)
        if not policy:
            break

        tensors.append(board_to_tensor(board))
        policy_targets.append(policy_to_tensor(policy))

        moves = list(policy.keys())
        probs = np.array([policy[m] for m in moves], dtype=np.float64)
        probs /= probs.sum()

        if move_num < TEMP_THRESHOLD:
            move = moves[np.random.choice(len(moves), p=probs)]
        else:
            move = moves[probs.argmax()]

        board.push(move)

    result = board.result()
    if result == "1-0":
        outcome = 1.0
    elif result == "0-1":
        outcome = -1.0
    elif result == "*":
        # Truncated by move limit — use material balance as a proxy signal
        outcome = math.tanh(static_eval(board) / STATIC_EVAL_SCALE)
    else:
        outcome = 0.0  # genuine draw (stalemate, 50-move, repetition)
    return tensors, policy_targets, outcome


def play_game_vs(model_white, model_black, n_simulations: int = 25) -> float:
    """Pit two models. Returns outcome from white's perspective (+1/-1/0)."""
    board = chess.Board()

    for _ in range(MAX_MOVES):
        if board.is_game_over():
            break
        model = model_white if board.turn == chess.WHITE else model_black
        move = mcts.best_move(board, model, n_simulations)
        if move is None:
            break
        board.push(move)

    result = board.result()
    if result == "1-0":
        return 1.0
    elif result == "0-1":
        return -1.0
    elif result == "*":
        return math.tanh(static_eval(board) / STATIC_EVAL_SCALE)
    return 0.0


def generate(n_games: int, output_path: str, model, n_simulations: int = 50):
    """Play n_games self-play games and save dataset."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    all_tensors, all_policies, all_outcomes = [], [], []

    for i in range(n_games):
        tensors, policies, outcome = play_game(model, n_simulations)
        all_tensors.extend(tensors)
        all_policies.extend(policies)
        all_outcomes.extend([outcome] * len(tensors))
        print(f"  game {i + 1}/{n_games}  moves={len(tensors)}  outcome={outcome:+.3f}")

    dataset = {
        "tensors": torch.tensor(np.array(all_tensors), dtype=torch.uint8),
        "policies": torch.stack(all_policies),
        "outcomes": torch.tensor(all_outcomes, dtype=torch.float32),
    }
    torch.save(dataset, output_path)
    print(f"Saved {len(all_outcomes)} positions → {output_path}")
    return dataset
