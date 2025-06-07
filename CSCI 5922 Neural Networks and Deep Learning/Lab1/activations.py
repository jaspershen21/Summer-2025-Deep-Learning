import torch

class ReLU():
    def forward(x: torch.tensor) -> torch.tensor:
        return torch.maximum(x, torch.tensor(0.0))

    def backward(delta: torch.tensor, x: torch.tensor) -> torch.tensor:
        return delta * (x > 0).float()


class LeakyReLU():
    def forward(x: torch.tensor) -> torch.tensor:
        return torch.maximum(x, 0.1 * x)

    def backward(delta: torch.tensor, x: torch.tensor) -> torch.tensor:
        return delta * torch.where(x > 0, torch.tensor(1.0), torch.tensor(0.1)).float()