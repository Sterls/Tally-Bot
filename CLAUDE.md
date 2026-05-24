# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the bot

```bash
python main.py
```

Requires a `config.py` in the repo root (gitignored) with:
```python
TOKEN = "your_lichess_api_token"
```

## Dependencies

```bash
pip install python-chess berserk
```

## Architecture

The bot connects to Lichess via the [berserk](https://github.com/lichess-org/berserk) client and plays games using a custom chess engine.

**`main.py`** — entry point. Calls `challenge_bots(10)` then `run()`.

**`lichess/bot.py` — `LichessBot`**
- `run()`: streams incoming Lichess events, accepts challenges, dispatches to `play_game()`
- `challenge_bots(n)`: challenges n online bots to rated 3+0 games before entering the event loop
- `play_game(game_id)`: reconstructs the full board from the UCI move history on every state update, then calls the engine on the bot's turn

**`engine/search.py`** — the chess engine
- `evaluate(board)`: pure static eval — material + tapered MG/EG piece-square tables, white-positive. **No terminal handling**; callers must check `board.is_game_over()` first if they care about mate/draw scores. Phase is computed from non-pawn material weights (`_PHASE_WEIGHT`, max `_TOTAL_PHASE = 24`); the score interpolates `_PST_MG` (castled king, central pieces) toward `_PST_EG` (centralized king, advanced pawns) as material comes off.
- `best_move(board, depth=3, tt=None)`: iterative-deepening alpha-beta up to `depth`. Always returns a legal move (or `None` if the position is already game-over).
- `best_move_timed(board, think_ms, tt=None)`: iterative deepening with a wall-clock deadline. Depth 1 always completes, so it returns a thought-through move even under severe time pressure. Pass a persistent `tt` dict across calls in the same game for free move ordering wins.
- Search internals: transposition table with EXACT/LOWER/UPPER bounds keyed by Zobrist hash; quiescence search at the horizon (captures only); MVV-LVA + TT-move-first move ordering.

## Key invariants

- `evaluate()` returns values in `[-INF, INF]` where `INF = 10**9`. Mate scores are produced **inside the search** as `±(MATE - ply)` with `MATE = 100_000`; `|score| > MATE_BOUND` (`MATE - 1000`) indicates a forced mate.
- `board.turn` is `chess.WHITE` (`True`) / `chess.BLACK` (`False`). The engine and bot both rely on this for color detection.
- `play_game()` maintains `board` incrementally across `gameState` events. It tracks the cumulative move count and only pushes new UCI moves; a backwards jump (takeback / out-of-order reconnect) triggers a full resync.
- `run()` spawns a daemon thread per game so the event loop keeps accepting challenges while games are in progress.
