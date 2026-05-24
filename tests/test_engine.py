import chess
import pytest

from engine.search import evaluate, best_move, best_move_timed, INF, MATE, MATE_BOUND


# --- evaluate ---

def test_evaluate_starting_position_is_zero():
    assert evaluate(chess.Board()) == 0

def test_evaluate_white_up_a_queen():
    board = chess.Board("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    assert evaluate(board) > 0

def test_evaluate_black_up_a_queen():
    board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1")
    assert evaluate(board) < 0

def test_evaluate_pst_prefers_centralized_knight():
    # Knight on d4 (central) vs a1 (corner) — PST should make d4 strictly better.
    centered = chess.Board("4k3/8/8/8/3N4/8/8/4K3 w - - 0 1")
    cornered = chess.Board("4k3/8/8/8/8/8/8/N3K3 w - - 0 1")
    assert evaluate(centered) > evaluate(cornered)


# --- best_move ---

def test_best_move_returns_legal_move():
    board = chess.Board()
    move = best_move(board, depth=1)
    assert move in board.legal_moves

def test_best_move_captures_free_piece():
    # White queen on d1 can take undefended black rook on g1.
    board = chess.Board("k7/8/8/8/8/8/8/K2Q2r1 w - - 0 1")
    move = best_move(board, depth=1)
    assert move is not None
    assert board.is_capture(move)

def test_best_move_returns_none_on_game_over():
    # Scholar's mate — black is mated, no legal moves.
    board = chess.Board("r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4")
    assert board.is_checkmate()
    assert best_move(board, depth=1) is None

def test_best_move_as_black():
    board = chess.Board()
    board.push_san("e4")
    move = best_move(board, depth=1)
    assert move in board.legal_moves
    assert board.turn == chess.BLACK


# --- mate detection (search-level, not evaluate-level) ---

def test_best_move_finds_mate_in_one():
    # White rook on a1, black king on g8 boxed in by own pawns. Ra8#.
    board = chess.Board("6k1/5ppp/8/8/8/8/8/R6K w - - 0 1")
    move = best_move(board, depth=2)
    assert move == chess.Move.from_uci("a1a8")

def test_best_move_finds_mate_in_one_for_black():
    # Mirror: black to move with rook on a8, white king on g1 boxed in.
    board = chess.Board("r6k/8/8/8/8/8/5PPP/6K1 b - - 0 1")
    move = best_move(board, depth=2)
    assert move == chess.Move.from_uci("a8a1")


# --- best_move_timed ---

def test_best_move_timed_returns_legal_move():
    board = chess.Board()
    move = best_move_timed(board, think_ms=200)
    assert move in board.legal_moves

def test_best_move_timed_returns_none_on_game_over():
    board = chess.Board("r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4")
    assert best_move_timed(board, think_ms=200) is None

def test_best_move_timed_respects_tight_budget():
    # 1ms budget must still return a legal move (depth 1 is uninterruptible).
    board = chess.Board()
    move = best_move_timed(board, think_ms=1)
    assert move in board.legal_moves


# --- mate-score sanity ---

def test_mate_bound_constants():
    assert MATE > MATE_BOUND > 0
    assert MATE < INF
