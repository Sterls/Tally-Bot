import os
import glob
import tempfile
import torch

from nn.model import ChessNet, load as load_model
from nn.self_play import generate, play_game_vs, nn_move_fn, engine_move_fn
from nn.train import train
from nn import storage

CHECKPOINT_DIR = "nn/checkpoints"
DATA_DIR = "nn/data"
BEST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best.pt")
DEPTH_STATE_FILE    = os.path.join(CHECKPOINT_DIR, "engine_depth.txt")
WIN_RATE_STATE_FILE = os.path.join(CHECKPOINT_DIR, "best_win_rate.txt")


def pit_vs_engine(model, n_games: int = 10, n_simulations: int = 15,
                  engine_depth: int = 1) -> float:
    """
    Pit model against the alpha-beta engine, alternating colors.
    Returns win rate of the NN (wins + 0.5*draws) / n_games.
    """
    wins = draws = losses = 0
    eng = engine_move_fn(engine_depth)

    for i in range(n_games):
        nn = nn_move_fn(model, n_simulations)
        if i % 2 == 0:
            outcome = play_game_vs(nn, eng)
            if outcome > 0:    wins += 1
            elif outcome == 0: draws += 1
            else:              losses += 1
        else:
            outcome = play_game_vs(eng, nn)
            if outcome < 0:    wins += 1
            elif outcome == 0: draws += 1
            else:              losses += 1

    win_rate = (wins + 0.5 * draws) / n_games
    print(f"  Pit vs engine(d={engine_depth}): {wins}W {draws}D {losses}L  "
          f"win_rate={win_rate:.2f}")
    return win_rate


def _prune_old_data(max_positions: int):
    """Remove oldest gen files so the remaining total stays within max_positions."""
    files = sorted(glob.glob(os.path.join(DATA_DIR, "gen*.pt")))
    if not files:
        return

    sizes = [len(torch.load(f, weights_only=True)["outcomes"]) for f in files]

    total = 0
    keep_from = 0
    for i in range(len(files) - 1, -1, -1):
        if total + sizes[i] > max_positions:
            keep_from = i + 1
            break
        total += sizes[i]

    keep_from = min(keep_from, len(files) - 1)  # always keep at least the newest file

    for f in files[:keep_from]:
        os.remove(f)
        print(f"  Pruned {f}")


def _load_depth(default: int) -> int:
    if os.path.exists(DEPTH_STATE_FILE):
        with open(DEPTH_STATE_FILE) as f:
            depth = int(f.read().strip())
        print(f"Resuming at engine depth {depth}")
        return depth
    return default


def _save_depth(depth: int):
    with open(DEPTH_STATE_FILE, "w") as f:
        f.write(str(depth))


def _load_best_win_rate() -> float:
    if os.path.exists(WIN_RATE_STATE_FILE):
        with open(WIN_RATE_STATE_FILE) as f:
            rate = float(f.read().strip())
        print(f"Best win rate so far: {rate:.2f}")
        return rate
    return -1.0  # any result promotes on first run


def _save_best_win_rate(rate: float):
    with open(WIN_RATE_STATE_FILE, "w") as f:
        f.write(str(rate))


def run(
    generations: int = 10,
    games_per_gen: int = 200,
    epochs_per_gen: int = 10,
    n_simulations: int = 50,
    pit_games: int = 10,
    pit_simulations: int = 15,
    engine_depth: int = 1,
    max_engine_depth: int = 5,
    depth_advance_threshold: float = 0.60,
    max_positions: int = 200_000,
):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Syncing from Drive...")
    storage.pull(CHECKPOINT_DIR, "checkpoints")
    storage.pull(DATA_DIR, "data")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    current_depth = _load_depth(engine_depth)
    best_win_rate = _load_best_win_rate()

    existing_gens = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "gen*.pt")))
    start_gen = len(existing_gens)

    if os.path.exists(BEST_CHECKPOINT):
        current_model = load_model(BEST_CHECKPOINT, device=device)
        print(f"Resuming from {BEST_CHECKPOINT}")
    else:
        current_model = ChessNet().to(device).eval()
        print("No checkpoint — bootstrapping with random model")

    for gen in range(start_gen, start_gen + generations):
        print(f"\n{'='*50}")
        print(f"Generation {gen}  [engine depth {current_depth}]")
        print(f"{'='*50}")

        # 1. Self-play vs engine
        data_path = os.path.join(DATA_DIR, f"gen{gen:03d}.pt")
        print(f"Self-play vs engine(d={current_depth}): {games_per_gen} games, "
              f"{n_simulations} sims/move")
        generate(games_per_gen, data_path, current_model, n_simulations, current_depth)

        # 2. Rolling window
        _prune_old_data(max_positions)

        # 3. Train
        data_glob = os.path.join(DATA_DIR, "gen*.pt")
        resume = BEST_CHECKPOINT if os.path.exists(BEST_CHECKPOINT) else None
        print(f"Training: {epochs_per_gen} epochs")
        with tempfile.TemporaryDirectory() as tmpdir:
            new_model = train(
                data_glob,
                checkpoint_dir=tmpdir,
                resume=resume,
                epochs=epochs_per_gen,
            )

        gen_ckpt = os.path.join(CHECKPOINT_DIR, f"gen{gen:03d}.pt")
        torch.save(new_model.state_dict(), gen_ckpt)
        print(f"Saved {gen_ckpt}")

        # 4. Pit vs engine
        new_model = new_model.eval()
        if pit_games == 0:
            win_rate = None
            promote = True
            print("  Auto-promoting (pit disabled)")
        else:
            win_rate = pit_vs_engine(new_model, pit_games, pit_simulations, current_depth)
            promote = win_rate >= best_win_rate

        # 5. Promote if new record against engine
        if promote:
            torch.save(new_model.state_dict(), BEST_CHECKPOINT)
            current_model = new_model
            best_win_rate = win_rate
            _save_best_win_rate(best_win_rate)
            print(f"  Promoted gen{gen:03d} → best.pt  (new best: {best_win_rate:.2f})")
        else:
            print(f"  Not promoted — best win rate still {best_win_rate:.2f}")

        # 6. Advance engine depth if NN is winning consistently
        if (win_rate is not None
                and win_rate >= depth_advance_threshold
                and current_depth < max_engine_depth):
            current_depth += 1
            _save_depth(current_depth)
            print(f"  *** Engine depth advanced to {current_depth} ***")

        # 7. Sync
        print("Syncing to Drive...")
        storage.push(CHECKPOINT_DIR, "checkpoints")
        storage.push(DATA_DIR, "data")

    print("\nRL loop complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--generations",            type=int,   default=10)
    parser.add_argument("--games-per-gen",          type=int,   default=200)
    parser.add_argument("--epochs-per-gen",         type=int,   default=10)
    parser.add_argument("--n-simulations",          type=int,   default=50)
    parser.add_argument("--pit-games",              type=int,   default=10)
    parser.add_argument("--pit-simulations",        type=int,   default=15)
    parser.add_argument("--engine-depth",           type=int,   default=1)
    parser.add_argument("--max-engine-depth",       type=int,   default=5)
    parser.add_argument("--depth-advance-threshold",type=float, default=0.60)
    parser.add_argument("--max-positions",          type=int,   default=200_000)
    args = parser.parse_args()

    run(
        generations=args.generations,
        games_per_gen=args.games_per_gen,
        epochs_per_gen=args.epochs_per_gen,
        n_simulations=args.n_simulations,
        pit_games=args.pit_games,
        pit_simulations=args.pit_simulations,
        engine_depth=args.engine_depth,
        max_engine_depth=args.max_engine_depth,
        depth_advance_threshold=args.depth_advance_threshold,
        max_positions=args.max_positions,
    )
