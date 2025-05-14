import torch

# 2D ground-truth function
def f2(x: torch.Tensor, noise_level: float = 1.0, noise: bool = True) -> torch.Tensor:
    """
    x: Tensor of shape (N,2)
    returns: Tensor of shape (N,1)
    """
    x1, x2 = x[:, 0], x[:, 1]                           # each (N,)
    y1     = x1.pow(4) - 14*x1.pow(2) + 5*x1             # (N,)
    y2     = x2.pow(4) - 14*x2.pow(2) + 5*x2             # (N,)
    y      = -(y1 + y2 + 3*(x1*x2))                      # (N,)
    if noise:
        y = y + torch.normal(0.0, noise_level, size=y.shape, dtype=y.dtype, device=y.device)
    return y.unsqueeze(-1) 