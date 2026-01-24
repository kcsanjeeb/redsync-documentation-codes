import copy
import torch
from collections import deque

def plasticity_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    # a,b: activation tensors (same shape)
    # normalize so scale doesn't dominate
    a = a.detach().float()
    b = b.detach().float()
    denom = (a.pow(2).mean().sqrt() + 1e-8)
    return float(((a - b).pow(2).mean().sqrt() / denom).item())

def linear_slope(xs):
    # slope of y over x=0..n-1 via least squares
    n = len(xs)
    if n < 2:
        return 1e9
    x = torch.arange(n, dtype=torch.float32)
    y = torch.tensor(xs, dtype=torch.float32)
    x_mean = x.mean()
    y_mean = y.mean()
    num = ((x - x_mean) * (y - y_mean)).sum()
    den = ((x - x_mean) ** 2).sum() + 1e-12
    return float((num / den).item())

class FreezeController:
    """
    Maintains plasticity history for the 'frontmost non-frozen module' boundary.
    Freezes when slope < T for enough checks.
    """
    def __init__(self, W=50, slope_T=1e-4, required_hits=3):
        self.W = W
        self.slope_T = slope_T
        self.required_hits = required_hits
        self.hist = deque(maxlen=W)
        self.hit_count = 0

    def update(self, pval: float) -> dict:
        self.hist.append(pval)
        if len(self.hist) < self.W:
            return {"ready": False}

        slope = linear_slope(list(self.hist))
        if abs(slope) < self.slope_T:
            self.hit_count += 1
        else:
            self.hit_count = 0

        freeze_now = (self.hit_count >= self.required_hits)
        return {
            "ready": True,
            "plasticity": pval,
            "slope": slope,
            "hit_count": self.hit_count,
            "freeze_now": freeze_now,
        }

def make_cpu_reference(model: torch.nn.Module) -> torch.nn.Module:
    ref = copy.deepcopy(model).cpu().eval()
    for p in ref.parameters():
        p.requires_grad = False
    return ref
