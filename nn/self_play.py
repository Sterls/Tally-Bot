import math
import os
import threading
import numpy as np
import torch
import chess

from engine.features import board_to_tensor
from engine.moves import policy_to_tensor
from engine import mcts
from engine import search as engine_search
from engine.search import evaluate as static_eval

MAX_MOVES = 200
TEMP_THRESHOLD = 20  # sample proportionally for first N moves, then argmax
STATIC_EVAL_SCALE = 600
N_WORKERS = 4


def nn_move_fn(model, n_simulations: int):
    """Returns a (board) -> move callable backed by MCTS."""
    def fn(board: chess.Board):
        return mcts.best_move(board, model, n_simulations)
    return fn


def engine_move_fn(depth: int = 3):
    """Returns a (board) -> move callable backed by alpha-beta search."""
    tt = {}
    def fn(board: chess.Board):
        return engine_search.best_move(board, depth=depth, tt=tt)
    return fn


def _game_outcome(board: chess.Board) -> float:
    """Final outcome from white's perspective (+1/-1/0/*=static eval)."""
    result = board.result()
    if result == "1-0":
        return 1.0
    elif result == "0-1":
        return -1.0
    elif result == "*":
        return math.tanh(static_eval(board) / STATIC_EVAL_SCALE)
    return 0.0


def play_game_vs(white_fn, black_fn) -> float:
    """
    Pit two move functions. Returns outcome from white's perspective (+1/-1/0).
    Each argument is a callable (board) -> move.
    """
    board = chess.Board()
    for _ in range(MAX_MOVES):
        if board.is_game_over():
            break
        fn = white_fn if board.turn == chess.WHITE else black_fn
        move = fn(board)
        if move is None:
            break
        board.push(move)
    return _game_outcome(board)


def play_game(model, n_simulations: int = 50, engine_depth: int = 3) -> tuple:
    """
    Play one game: NN vs alpha-beta engine, collecting NN's positions.
    NN color is white. Call with alternating perspective by flipping the board
    externally, or use generate() which handles alternation across games.

    Returns (tensors, policy_targets, outcome) where outcome is from white's
    perspective so the value head trains consistently regardless of NN color.
    """
    board = chess.Board()
    tensors = []
    policy_targets = []
    tt = {}

    for move_num in range(MAX_MOVES):
        if board.is_game_over():
            break

        if board.turn == chess.WHITE:
            # NN's turn — run MCTS and record the position
            policy = mcts.search(board, model, n_simulations, add_noise=True)
            if not policy:
                break

            tensors.append(board_to_tensor(board))
            policy_targets.append(policy_to_tensor(policy))

            moves = list(policy.keys())
            probs = np.array([policy[m] for m in moves], dtype=np.float64)
            probs /= probs.sum()
            move = (
                moves[np.random.choice(len(moves), p=probs)]
                if move_num < TEMP_THRESHOLD
                else moves[probs.argmax()]
            )
        else:
            # Engine's turn — no data collected
            move = engine_search.best_move(board, depth=engine_depth, tt=tt)
            if move is None:
                break

        board.push(move)

    return tensors, policy_targets, _game_outcome(board)


def _play_game_color(model, n_simulations, engine_depth, nn_is_white):
    """
    Play one game with NN as white or black.
    Always returns outcome from white's perspective.
    """
    if nn_is_white:
        return play_game(model, n_simulations, engine_depth)

    # NN plays black: mirror the board each ply so NN always sees itself as white
    board = chess.Board()
    tensors = []
    policy_targets = []
    tt = {}

    for move_num in range(MAX_MOVES):
        if board.is_game_over():
            break

        if board.turn == chess.BLACK:
            # NN's turn — mirror the board so NN evaluates from white's view
            mirrored = board.mirror()
            policy_mirrored = mcts.search(mirrored, model, n_simulations, add_noise=True)
            if not policy_mirrored:
                break

            tensors.append(board_to_tensor(board))
            # Mirror policy moves back to real board coordinates
            real_policy = {
                chess.Move(chess.square_mirror(m.from_square),
                           chess.square_mirror(m.to_square),
                           m.promotion): p
                for m, p in policy_mirrored.items()
            }
            # Filter to legal moves only
            legal = set(board.legal_moves)
            real_policy = {m: p for m, p in real_policy.items() if m in legal}
            if not real_policy:
                break
            total = sum(real_policy.values())
            real_policy = {m: p / total for m, p in real_policy.items()}
            policy_targets.append(policy_to_tensor(real_policy))

            moves = list(real_policy.keys())
            probs = np.array([real_policy[m] for m in moves], dtype=np.float64)
            probs /= probs.sum()
            move = (
                moves[np.random.choice(len(moves), p=probs)]
                if move_num < TEMP_THRESHOLD
                else moves[probs.argmax()]
            )
        else:
            # Engine plays white
            move = engine_search.best_move(board, depth=engine_depth, tt=tt)
            if move is None:
                break

        board.push(move)

    return tensors, policy_targets, _game_outcome(board)


def generate(n_games: int, output_path: str, model, n_simulations: int = 50,
             engine_depth: int = 3):
    """
    Play n_games of NN vs engine in parallel, alternating NN color each game.
    Collects only the NN's positions so every position has a real opponent.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    game_counter = [0]
    all_results = []
    lock = threading.Lock()

    def worker():
        while True:
            with lock:
                if game_counter[0] >= n_games:
                    return
                idx = game_counter[0]
                game_counter[0] += 1

            nn_is_white = (idx % 2 == 0)
            tensors, policies, outcome = _play_game_color(
                model, n_simulations, engine_depth, nn_is_white
            )
            color = "W" if nn_is_white else "B"

            with lock:
                all_results.append((tensors, policies, outcome))
                n = len(all_results)
                print(f"  game {n}/{n_games} (NN={color})  "
                      f"moves={len(tensors)}  outcome={outcome:+.3f}")

    threads = [threading.Thread(target=worker) for _ in range(N_WORKERS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    all_tensors, all_policies, all_outcomes = [], [], []
    for tensors, policies, outcome in all_results:
        all_tensors.extend(tensors)
        all_policies.extend(policies)
        all_outcomes.extend([outcome] * len(tensors))

    dataset = {
        "tensors": torch.tensor(np.array(all_tensors), dtype=torch.uint8),
        "policies": torch.stack(all_policies),
        "outcomes": torch.tensor(all_outcomes, dtype=torch.float32),
    }
    torch.save(dataset, output_path)
    print(f"Saved {len(all_outcomes)} positions → {output_path}")
    return dataset
