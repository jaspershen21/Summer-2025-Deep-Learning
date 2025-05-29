import torch

class ReLU():
    #Complete this class
    def forward(x: torch.tensor) -> torch.tensor:
        return torch.max(x, torch.zeros_like(x))
    
    def backward(delta: torch.tensor, x: torch.tensor) -> torch.tensor:
        # delta * ReLU'(x) where (ReLU' = do/dz)
        dReLU = torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))
        return torch.mul(delta, dReLU)


class LeakyReLU():
    #Complete this class
    def forward(x: torch.tensor) -> torch.tensor:
        return torch.max(x, 0,1*x)
    
    def backward(delta: torch.tensor, x: torch.tensor) -> torch.tensor:
        #implement delta * LeakyReLU'(x) where (LeakyReLU' = do/dz)
        dLReLU = torch.where(x > 0, torch.ones_like(x), 0.1 * torch.ones_like(x))
        return torch.mul(delta, dLReLU)