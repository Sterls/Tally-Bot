import os
import glob
import tempfile
import torch

from nn.model import ChessNet, load as load_model
from nn.self_play import generate, play_game_vs
from nn.train import train
from nn import storage

CHECKPOINT_DIR = "nn/checkpoints"
DATA_DIR = "nn/data"
BEST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best.pt")


def pit(model_new, model_old, n_games: int = 20, n_simulations: int = 25) -> float:
    """
    Play n_games between new and old model, alternating colors.
    Returns win rate of new model (wins + 0.5*draws) / n_games.
    """
    wins = draws = losses = 0

    for i in range(n_games):
        if i % 2 == 0:
            outcome = play_game_vs(model_new, model_old, n_simulations)
            if outcome > 0:    wins += 1
            elif outcome == 0: draws += 1
            else:              losses += 1
        else:
            outcome = play_game_vs(model_old, model_new, n_simulations)
            if outcome < 0:    wins += 1
            elif outcome == 0: draws += 1
            else:              losses += 1

    win_rate = (wins + 0.5 * draws) / n_games
    print(f"  Pit result: {wins}W {draws}D {losses}L  win_rate={win_rate:.2f}")
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


def run(
    generations: int = 10,
    games_per_gen: int = 100,
    epochs_per_gen: int = 10,
    n_simulations: int = 50,
    pit_games: int = 20,
    pit_simulations: int = 25,
    promote_threshold: float = 0.55,
    max_positions: int = 200_000,
):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Syncing from Drive...")
    storage.pull(CHECKPOINT_DIR, "checkpoints")
    storage.pull(DATA_DIR, "data")

    device = "cuda" if torch.cuda.is_available() else "cpu"

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
        print(f"Generation {gen}")
        print(f"{'='*50}")

        # 1. Self-play
        data_path = os.path.join(DATA_DIR, f"gen{gen:03d}.pt")
        print(f"Self-play: {games_per_gen} games, {n_simulations} sims/move")
        generate(games_per_gen, data_path, current_model, n_simulations)

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

        # 4. Pit
        new_model = new_model.eval()
        if pit_games == 0:
            promote = True
            print("  Auto-promoting (pit disabled)")
        else:
            print(f"Pitting new vs current best ({pit_games} games, {pit_simulations} sims)...")
            win_rate = pit(new_model, current_model, pit_games, pit_simulations)
            promote = win_rate >= promote_threshold

        # 5. Promote
        if promote:
            torch.save(new_model.state_dict(), BEST_CHECKPOINT)
            current_model = new_model
            print(f"  Promoted gen{gen:03d} → best.pt")
        else:
            print(f"  Not promoted — keeping previous best")

        # 6. Sync
        print("Syncing to Drive...")
        storage.push(CHECKPOINT_DIR, "checkpoints")
        storage.push(DATA_DIR, "data")

    print("\nRL loop complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--generations",       type=int,   default=10)
    parser.add_argument("--games-per-gen",     type=int,   default=100)
    parser.add_argument("--epochs-per-gen",    type=int,   default=10)
    parser.add_argument("--n-simulations",     type=int,   default=50)
    parser.add_argument("--pit-games",         type=int,   default=20)
    parser.add_argument("--pit-simulations",   type=int,   default=25)
    parser.add_argument("--promote-threshold", type=float, default=0.55)
    parser.add_argument("--max-positions",     type=int,   default=200_000)
    args = parser.parse_args()

    run(
        generations=args.generations,
        games_per_gen=args.games_per_gen,
        epochs_per_gen=args.epochs_per_gen,
        n_simulations=args.n_simulations,
        pit_games=args.pit_games,
        pit_simulations=args.pit_simulations,
        promote_threshold=args.promote_threshold,
        max_positions=args.max_positions,
    )
