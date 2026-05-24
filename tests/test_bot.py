from datetime import timedelta
from unittest.mock import MagicMock, patch
import chess
import pytest

from lichess.bot import _to_ms, _think_time_ms, GAME_OVER_STATUSES, LichessBot


# --- _to_ms ---

def test_to_ms_int():
    assert _to_ms(180_000) == 180_000

def test_to_ms_timedelta():
    assert _to_ms(timedelta(seconds=180)) == 180_000

def test_to_ms_timedelta_partial():
    assert _to_ms(timedelta(seconds=9, milliseconds=500)) == 9_500


# --- _think_time_ms ---

def test_think_time_clamped_lower():
    # With near-zero clock, must still return at least the 100ms floor.
    assert _think_time_ms(0, 1) == 100

def test_think_time_clamped_upper():
    # With huge clock, must not exceed the 5000ms ceiling.
    assert _think_time_ms(10_000_000, 1) == 5_000

def test_think_time_scales_with_clock():
    early = _think_time_ms(60_000, 1)
    late = _think_time_ms(60_000, 40)
    assert early < late   # fewer moves remaining → larger per-move budget


# --- play_game ---

def _make_bot():
    with patch("lichess.bot.berserk.Client") as mock_client_cls:
        client = mock_client_cls.return_value
        client.account.get.return_value = {"id": "testbot"}
        bot = LichessBot.__new__(LichessBot)
        bot.client = client
        bot.my_id = "testbot"
        bot.model = None
        return bot, client


def test_play_game_stops_on_outoftime():
    bot, client = _make_bot()

    states = [
        {
            "type": "gameFull",
            "white": {"id": "opponent"},
            "black": {"id": "testbot"},
            "state": {"moves": "", "status": "started", "wtime": 180_000, "btime": 180_000},
        },
        {
            "type": "gameState",
            "moves": "",
            "status": "outoftime",
            "wtime": 180_000,
            "btime": 0,
        },
    ]
    client.bots.stream_game_state.return_value = iter(states)

    bot.play_game("abc123")

    client.bots.make_move.assert_not_called()


def test_play_game_stops_on_mate():
    bot, client = _make_bot()

    states = [
        {
            "type": "gameFull",
            "white": {"id": "opponent"},
            "black": {"id": "testbot"},
            "state": {"moves": "", "status": "mate", "wtime": 100_000, "btime": 100_000},
        },
    ]
    client.bots.stream_game_state.return_value = iter(states)

    bot.play_game("abc123")

    client.bots.make_move.assert_not_called()


def test_play_game_skips_opponents_turn():
    bot, client = _make_bot()

    states = [
        {
            "type": "gameFull",
            "white": {"id": "opponent"},
            "black": {"id": "testbot"},
            "state": {"moves": "", "status": "started", "wtime": 180_000, "btime": 180_000},
        },
    ]
    client.bots.stream_game_state.return_value = iter(states)

    bot.play_game("abc123")

    client.bots.make_move.assert_not_called()


def test_play_game_makes_move_on_our_turn():
    bot, client = _make_bot()

    states = [
        {
            "type": "gameFull",
            "white": {"id": "testbot"},
            "black": {"id": "opponent"},
            "state": {"moves": "", "status": "started", "wtime": 180_000, "btime": 180_000},
        },
    ]
    client.bots.stream_game_state.return_value = iter(states)

    bot.play_game("abc123")

    client.bots.make_move.assert_called_once()
    move_uci = client.bots.make_move.call_args[0][1]
    assert chess.Move.from_uci(move_uci) in chess.Board().legal_moves


def test_play_game_incremental_board_after_opponent_move():
    """Second gameState carries the full move history; bot must respond from that position."""
    bot, client = _make_bot()

    states = [
        {
            "type": "gameFull",
            "white": {"id": "opponent"},
            "black": {"id": "testbot"},
            "state": {"moves": "", "status": "started", "wtime": 180_000, "btime": 180_000},
        },
        {
            "type": "gameState",
            "moves": "e2e4",  # opponent (white) played e4 → bot (black) must respond
            "status": "started",
            "wtime": 180_000,
            "btime": 180_000,
        },
    ]
    client.bots.stream_game_state.return_value = iter(states)

    bot.play_game("abc123")

    client.bots.make_move.assert_called_once()
    move_uci = client.bots.make_move.call_args[0][1]
    # Bot's reply must be legal after 1.e4.
    board = chess.Board()
    board.push_uci("e2e4")
    assert chess.Move.from_uci(move_uci) in board.legal_moves


def test_play_game_make_move_error_does_not_crash():
    bot, client = _make_bot()

    states = [
        {
            "type": "gameFull",
            "white": {"id": "testbot"},
            "black": {"id": "opponent"},
            "state": {"moves": "", "status": "started", "wtime": 180_000, "btime": 180_000},
        },
    ]
    client.bots.stream_game_state.return_value = iter(states)
    client.bots.make_move.side_effect = Exception("Game already finished")

    bot.play_game("abc123")   # must not raise
