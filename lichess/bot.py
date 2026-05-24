import threading
import chess
import berserk
from engine.search import best_move_timed as ab_best_move_timed, evaluate

GAME_OVER_STATUSES = {"mate", "resign", "stalemate", "timeout", "draw",
                      "outoftime", "cheat", "noStart", "unknownFinish", "aborted"}


def _to_ms(t) -> int:
    """Normalize berserk clock value to milliseconds (int or timedelta)."""
    return int(t.total_seconds() * 1000) if hasattr(t, "total_seconds") else int(t)


def _think_time_ms(my_time_ms: int, fullmove_number: int) -> int:
    """Allocate thinking time based on remaining clock and move number."""
    moves_left = max(15, 50 - fullmove_number)
    budget = my_time_ms / moves_left * 0.9
    return int(min(max(budget, 100), 5_000))


class LichessBot:
    def __init__(self, token: str, model=None):
        session = berserk.TokenSession(token)
        self.client = berserk.Client(session=session)
        self.my_id = self.client.account.get()["id"]
        self.model = model  # ChessNet or None

    def run(self):
        print("Bot running...")

        for event in self.client.bots.stream_incoming_events():

            if event["type"] == "challenge":
                challenge_id = event["challenge"]["id"]
                try:
                    self.client.bots.accept_challenge(challenge_id)
                    print("Accepted challenge", challenge_id)
                except berserk.exceptions.ResponseError as e:
                    print("Could not accept challenge:", e)
                    try:
                        self.client.bots.decline_challenge(challenge_id)
                    except berserk.exceptions.ResponseError:
                        pass

            elif event["type"] == "gameStart":
                game_id = event["game"]["id"]
                print("Game started:", game_id)
                threading.Thread(
                    target=self._safe_play_game,
                    args=(game_id,),
                    daemon=True,
                    name=f"game-{game_id}",
                ).start()

    def challenge_bots(self, n: int):
        sent = 0
        for bot in self.client.bots.get_online_bots():
            if sent >= n:
                break
            if bot["id"] == self.my_id:
                continue
            try:
                self.client.challenges.create(
                    bot["id"],
                    rated=True,
                    clock_limit=180,
                    clock_increment=0,
                )
                print(f"Challenged {bot['id']}")
                sent += 1
            except berserk.exceptions.ResponseError as e:
                print(f"Could not challenge {bot['id']}:", e)

    def _pick_move(self, board: chess.Board, my_time_ms: int, tt: dict):
        think_ms = _think_time_ms(my_time_ms, board.fullmove_number)
        if self.model is not None:
            from engine.mcts import best_move_timed as mcts_best_move_timed
            return mcts_best_move_timed(board, self.model, think_ms)
        return ab_best_move_timed(board, think_ms, eval_fn=evaluate, tt=tt)

    def _safe_play_game(self, game_id: str):
        try:
            self.play_game(game_id)
        except Exception as e:
            print(f"[{game_id}] game loop crashed: {e}")

    def play_game(self, game_id: str):
        is_white = None
        my_time_ms = 180_000
        board = chess.Board()
        last_move_count = 0
        tt: dict = {}

        for state in self.client.bots.stream_game_state(game_id):

            if state["type"] == "gameFull":
                is_white = (state["white"]["id"] == self.my_id)
                moves_str = state["state"]["moves"]
                status = state["state"].get("status", "started")
                my_time_ms = _to_ms(state["state"]["wtime" if is_white else "btime"])

            elif state["type"] == "gameState":
                moves_str = state["moves"]
                status = state.get("status", "started")
                if is_white is not None:
                    my_time_ms = _to_ms(state["wtime" if is_white else "btime"])

            else:
                continue

            if status in GAME_OVER_STATUSES:
                print(f"[{game_id}] Game over: {status}")
                break

            all_moves = moves_str.split() if moves_str else []
            if len(all_moves) < last_move_count:
                # Resync (takeback or out-of-order reconnect)
                board = chess.Board()
                last_move_count = 0
            for m in all_moves[last_move_count:]:
                board.push_uci(m)
            last_move_count = len(all_moves)

            if is_white is None or board.turn != is_white:
                continue

            move = self._pick_move(board, my_time_ms, tt)
            if not move:
                continue

            print(f"[{game_id}] Playing: {move.uci()}  (time={my_time_ms//1000}s)")
            try:
                self.client.bots.make_move(game_id, move.uci())
            except Exception as e:
                print(f"[{game_id}] make_move failed: {e}")
                break
