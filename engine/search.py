import time
import chess
import chess.polyglot

INF = 10**9
MATE = 100_000
MATE_BOUND = MATE - 1000  # |score| > MATE_BOUND indicates a forced mate score

_PIECE_VAL = {
    chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
    chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20_000,
}

# Piece-square tables, white perspective. Index 0 = A1, 63 = H8 (matches chess.SQUARES).
# Mirror via chess.square_mirror for black.
# Two sets: middlegame (MG, Michniewski) and endgame (EG). Interpolated by phase().
# King PST flips sign between MG (stay castled) and EG (centralize).
# Pawn PST in EG gives big advancement bonus (passed pawns dominate the endgame).

_PST_MG = {
    chess.PAWN: [
         0,  0,  0,  0,  0,  0,  0,  0,
         5, 10, 10,-20,-20, 10, 10,  5,
         5, -5,-10,  0,  0,-10, -5,  5,
         0,  0,  0, 20, 20,  0,  0,  0,
         5,  5, 10, 25, 25, 10,  5,  5,
        10, 10, 20, 30, 30, 20, 10, 10,
        50, 50, 50, 50, 50, 50, 50, 50,
         0,  0,  0,  0,  0,  0,  0,  0,
    ],
    chess.KNIGHT: [
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50,
    ],
    chess.BISHOP: [
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,  5,  0,  0,  0,  0,  5,-10,
        -10, 10, 10, 10, 10, 10, 10,-10,
        -10,  0, 10, 10, 10, 10,  0,-10,
        -10,  5,  5, 10, 10,  5,  5,-10,
        -10,  0,  5, 10, 10,  5,  0,-10,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -20,-10,-10,-10,-10,-10,-10,-20,
    ],
    chess.ROOK: [
          0,  0,  0,  5,  5,  0,  0,  0,
         -5,  0,  0,  0,  0,  0,  0, -5,
         -5,  0,  0,  0,  0,  0,  0, -5,
         -5,  0,  0,  0,  0,  0,  0, -5,
         -5,  0,  0,  0,  0,  0,  0, -5,
         -5,  0,  0,  0,  0,  0,  0, -5,
          5, 10, 10, 10, 10, 10, 10,  5,
          0,  0,  0,  0,  0,  0,  0,  0,
    ],
    chess.QUEEN: [
        -20,-10,-10, -5, -5,-10,-10,-20,
        -10,  0,  5,  0,  0,  0,  0,-10,
        -10,  5,  5,  5,  5,  5,  0,-10,
          0,  0,  5,  5,  5,  5,  0, -5,
         -5,  0,  5,  5,  5,  5,  0, -5,
        -10,  0,  5,  5,  5,  5,  0,-10,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -20,-10,-10, -5, -5,-10,-10,-20,
    ],
    chess.KING: [
         20, 30, 10,  0,  0, 10, 30, 20,
         20, 20,  0,  0,  0,  0, 20, 20,
        -10,-20,-20,-20,-20,-20,-20,-10,
        -20,-30,-30,-40,-40,-30,-30,-20,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
    ],
}

_PST_EG = {
    chess.PAWN: [
          0,   0,   0,   0,   0,   0,   0,   0,
          5,   5,   5,   5,   5,   5,   5,   5,
         10,  10,  10,  10,  10,  10,  10,  10,
         20,  20,  20,  20,  20,  20,  20,  20,
         35,  35,  35,  35,  35,  35,  35,  35,
         60,  60,  60,  60,  60,  60,  60,  60,
        100, 100, 100, 100, 100, 100, 100, 100,
          0,   0,   0,   0,   0,   0,   0,   0,
    ],
    # Knights/bishops/rooks/queens: EG positional preferences resemble MG closely
    # (central control still good, corners still bad). Reuse MG tables — refining
    # these is future work, but the framework is in place.
    chess.KNIGHT: None,  # filled below
    chess.BISHOP: None,
    chess.ROOK:   None,
    chess.QUEEN:  None,
    chess.KING: [
        -50,-30,-30,-30,-30,-30,-30,-50,
        -30,-30,  0,  0,  0,  0,-30,-30,
        -30,-10, 20, 30, 30, 20,-10,-30,
        -30,-10, 30, 40, 40, 30,-10,-30,
        -30,-10, 30, 40, 40, 30,-10,-30,
        -30,-10, 20, 30, 30, 20,-10,-30,
        -30,-20,-10,  0,  0,-10,-20,-30,
        -50,-40,-30,-20,-20,-30,-40,-50,
    ],
}
for _pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
    _PST_EG[_pt] = _PST_MG[_pt]

# Tapered-eval phase: 24 = full middlegame, 0 = pure endgame (only Ks + pawns).
_PHASE_WEIGHT = {chess.KNIGHT: 1, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 4}
_TOTAL_PHASE = 24

EXACT, LOWER, UPPER = 0, 1, 2


class _Timeout(Exception):
    pass


def evaluate(board: chess.Board) -> int:
    """Tapered material + PST eval, white-positive. No terminal handling."""
    mg = 0
    eg = 0
    phase = 0
    for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING):
        mg_table = _PST_MG[pt]
        eg_table = _PST_EG[pt]
        mat = 0 if pt == chess.KING else _PIECE_VAL[pt]
        pw = _PHASE_WEIGHT.get(pt, 0)
        for sq in board.pieces(pt, chess.WHITE):
            mg += mat + mg_table[sq]
            eg += mat + eg_table[sq]
            phase += pw
        for sq in board.pieces(pt, chess.BLACK):
            mirror = chess.square_mirror(sq)
            mg -= mat + mg_table[mirror]
            eg -= mat + eg_table[mirror]
            phase += pw
    phase = min(phase, _TOTAL_PHASE)
    return (mg * phase + eg * (_TOTAL_PHASE - phase)) // _TOTAL_PHASE


def _terminal_score(board: chess.Board, ply: int) -> int:
    """Absolute (white-positive) score for a game-over position."""
    if board.is_checkmate():
        # board.turn = side that has just been mated (to move with no escape)
        return -(MATE - ply) if board.turn == chess.WHITE else (MATE - ply)
    return 0  # stalemate / insufficient material / 50-move / repetition


def _order_moves(board, moves, tt_move=None):
    """TT move → recaptures → captures (MVV-LVA) → escapes → quiet moves."""
    last = board.peek() if board.move_stack else None
    recap_sq = last.to_square if last else None
    opp = not board.turn

    def key(move):
        if tt_move is not None and move == tt_move:
            return (-1, 0, 0)
        victim = board.piece_at(move.to_square)
        aggressor = board.piece_at(move.from_square)
        agg_val = _PIECE_VAL.get(aggressor.piece_type, 0) if aggressor else 0
        if victim:
            vic_val = _PIECE_VAL.get(victim.piece_type, 0)
            tier = 0 if move.to_square == recap_sq else 1
            return (tier, -vic_val, agg_val)
        if aggressor and board.is_attacked_by(opp, move.from_square):
            atk_min = min(
                (_PIECE_VAL.get(board.piece_at(sq).piece_type, 0)
                 for sq in board.attackers(opp, move.from_square)
                 if board.piece_at(sq)),
                default=0,
            )
            if atk_min < agg_val:
                return (2, 0, 0)
        return (3, 0, 0)

    return sorted(moves, key=key)


def _quiescence(board, alpha, beta, maximizing, eval_fn, ply, deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise _Timeout

    if board.is_game_over():
        return _terminal_score(board, ply)

    stand_pat = eval_fn(board)
    if maximizing:
        if stand_pat >= beta:
            return stand_pat
        alpha = max(alpha, stand_pat)
    else:
        if stand_pat <= alpha:
            return stand_pat
        beta = min(beta, stand_pat)

    captures = [m for m in board.legal_moves if board.is_capture(m)]
    for move in _order_moves(board, captures):
        board.push(move)
        try:
            val = _quiescence(board, alpha, beta, not maximizing, eval_fn, ply + 1, deadline)
        finally:
            board.pop()
        if maximizing:
            if val >= beta:
                return val
            alpha = max(alpha, val)
        else:
            if val <= alpha:
                return val
            beta = min(beta, val)
    return alpha if maximizing else beta


def _alphabeta(board, depth, alpha, beta, maximizing, eval_fn, ply, tt, deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise _Timeout

    if board.is_game_over():
        return _terminal_score(board, ply), None

    key = chess.polyglot.zobrist_hash(board)
    tt_move = None
    entry = tt.get(key) if tt is not None else None
    if entry is not None:
        ent_depth, ent_value, ent_flag, ent_move = entry
        tt_move = ent_move
        if ent_depth >= depth:
            if ent_flag == EXACT:
                return ent_value, ent_move
            if ent_flag == LOWER and ent_value >= beta:
                return ent_value, ent_move
            if ent_flag == UPPER and ent_value <= alpha:
                return ent_value, ent_move

    if depth <= 0:
        return _quiescence(board, alpha, beta, maximizing, eval_fn, ply, deadline), None

    orig_alpha, orig_beta = alpha, beta
    best_move_local = None

    if maximizing:
        best = -INF
        for move in _order_moves(board, board.legal_moves, tt_move):
            board.push(move)
            try:
                val, _ = _alphabeta(board, depth - 1, alpha, beta, False,
                                    eval_fn, ply + 1, tt, deadline)
            finally:
                board.pop()
            if val > best:
                best = val
                best_move_local = move
            alpha = max(alpha, val)
            if alpha >= beta:
                break
    else:
        best = INF
        for move in _order_moves(board, board.legal_moves, tt_move):
            board.push(move)
            try:
                val, _ = _alphabeta(board, depth - 1, alpha, beta, True,
                                    eval_fn, ply + 1, tt, deadline)
            finally:
                board.pop()
            if val < best:
                best = val
                best_move_local = move
            beta = min(beta, val)
            if beta <= alpha:
                break

    if tt is not None:
        if best <= orig_alpha:
            flag = UPPER
        elif best >= orig_beta:
            flag = LOWER
        else:
            flag = EXACT
        tt[key] = (depth, best, flag, best_move_local)

    return best, best_move_local


def best_move(board, depth=3, eval_fn=evaluate, tt=None):
    """Iterative deepening alpha-beta up to `depth`."""
    if board.is_game_over():
        return None
    if tt is None:
        tt = {}
    move = None
    for d in range(1, depth + 1):
        _, move = _alphabeta(board, d, -INF, INF,
                             board.turn == chess.WHITE, eval_fn, 0, tt, deadline=None)
    return move


def best_move_timed(board, think_ms, eval_fn=evaluate, max_depth=64, tt=None):
    """Iterative deepening with a wall-clock deadline. Depth 1 always completes."""
    if board.is_game_over():
        return None
    if tt is None:
        tt = {}
    deadline = time.monotonic() + think_ms / 1000

    # Depth 1 uninterruptible — guarantees a thought-through move under time pressure.
    _, best = _alphabeta(board, 1, -INF, INF,
                         board.turn == chess.WHITE, eval_fn, 0, tt, deadline=None)
    last_depth = 1

    for d in range(2, max_depth + 1):
        if time.monotonic() >= deadline:
            break
        try:
            _, move = _alphabeta(board, d, -INF, INF,
                                 board.turn == chess.WHITE, eval_fn, 0, tt, deadline)
            if move is not None:
                best = move
                last_depth = d
        except _Timeout:
            break

    print(f"  [ab depth={last_depth} budget={think_ms}ms]")
    return best or next(iter(board.legal_moves), None)
