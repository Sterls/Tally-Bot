import torch
import torch.nn as nn

from engine.moves import POLICY_SIZE


class ChessNet(nn.Module):
    """
    Input:  (batch, 13, 8, 8)
    Value head:  (batch, 1) in [-1, 1]; +1 = white wins
    Policy head: (batch, POLICY_SIZE) logits over all 64*64 from-to pairs
    """

    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(13, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(128, 1, 1),
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(128, 4, 1),
            nn.Flatten(),
            nn.Linear(256, POLICY_SIZE),
        )

    def forward(self, x: torch.Tensor):
        body = self.conv(x)
        return self.value_head(body), self.policy_head(body)


def load(path: str, device: str = "cpu") -> ChessNet:
    model = ChessNet()
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.to(device).eval()


def make_eval_fn(path_or_model, device: str = "cpu"):
    """Returns eval_fn(board) -> float for alpha-beta compatibility."""
    from engine.features import board_to_tensor
    model = (
        load(path_or_model, device)
        if isinstance(path_or_model, str)
        else path_or_model.to(device).eval()
    )

    def eval_fn(board):
        x = torch.from_numpy(board_to_tensor(board)).unsqueeze(0).float().to(device)
        with torch.no_grad():
            value, _ = model(x)
        return value.item()

    return eval_fn
