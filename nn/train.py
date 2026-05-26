import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from nn.model import ChessNet


class ChessDataset(Dataset):
    def __init__(self, path: str):
        data = torch.load(path, weights_only=True)
        self.tensors = data["tensors"]
        self.policies = data["policies"]
        self.outcomes = data["outcomes"].unsqueeze(1)

    def __len__(self):
        return len(self.outcomes)

    def __getitem__(self, idx):
        return self.tensors[idx].float(), self.policies[idx], self.outcomes[idx]


def load_dataset(data_path: str) -> Dataset:
    paths = sorted(glob.glob(data_path)) if "*" in data_path else [data_path]
    if not paths:
        raise FileNotFoundError(f"No data files matched: {data_path}")
    datasets = [ChessDataset(p) for p in paths]
    print(f"Loaded {sum(len(d) for d in datasets):,} positions from {len(paths)} file(s)")
    return ConcatDataset(datasets)


def train(
    data_path: str,
    checkpoint_dir: str = "nn/checkpoints",
    resume: str = None,
    epochs: int = 10,
    batch_size: int = 512,
    lr: float = 1e-3,
    device: str = None,
) -> ChessNet:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    dataset = load_dataset(data_path)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        pin_memory=(device == "cuda"),
    )

    model = ChessNet().to(device)
    if resume:
        model.load_state_dict(torch.load(resume, map_location=device, weights_only=True))
        print(f"Resumed from {resume}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_v_loss = total_p_loss = 0.0

        for x, policy_target, outcome in loader:
            x = x.to(device)
            policy_target = policy_target.to(device)
            outcome = outcome.to(device)

            optimizer.zero_grad()
            value, policy_logits = model(x)

            v_loss = F.mse_loss(value, outcome)
            log_probs = F.log_softmax(policy_logits, dim=-1)
            p_loss = -(policy_target * log_probs).sum(dim=-1).mean()

            (5 * v_loss + p_loss).backward()
            optimizer.step()

            n = len(outcome)
            total_v_loss += v_loss.item() * n
            total_p_loss += p_loss.item() * n

        N = len(dataset)
        ckpt = os.path.join(checkpoint_dir, f"epoch_{epoch:03d}.pt")
        torch.save(model.state_dict(), ckpt)
        print(
            f"Epoch {epoch:3d}/{epochs}  "
            f"value={total_v_loss/N:.4f}  policy={total_p_loss/N:.4f}  "
            f"saved {ckpt}"
        )

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("--checkpoint-dir", default="nn/checkpoints")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train(
        args.data,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
