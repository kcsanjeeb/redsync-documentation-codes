import torch.nn as nn
from typing import List, Tuple
import torch

def iter_blocks(seq: nn.Sequential):
    # seq is layer1/layer2/layer3 each is Sequential of BasicBlock
    return [m for m in seq if isinstance(m, nn.Module)]

def build_resnet56_modules(model: nn.Module):
    """
    Returns a list of modules in the exact order we will freeze:
    [stem, layer1_block0..8, layer2_block0..8, layer3_block0..8, head]
    """
    modules = []

    # stem
    modules.append(nn.Sequential(model.conv1, model.bn1))

    # residual blocks
    for blk in model.layer1:
        modules.append(blk)
    for blk in model.layer2:
        modules.append(blk)
    for blk in model.layer3:
        modules.append(blk)

    # head
    modules.append(nn.Sequential(model.avgpool, nn.Flatten(), model.fc))

    return modules

def freeze_module(mod: nn.Module):
    for p in mod.parameters():
        p.requires_grad = False
    # Important: keep BN stable in frozen modules
    for m in mod.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

def unfreeze_module(mod: nn.Module):
    for p in mod.parameters():
        p.requires_grad = True
    for m in mod.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train()


def forward_through_modules(modules, x, stop_at=None):
    """
    Runs x through modules in order.
    If stop_at is not None, returns (out, activation_at_stop)
    where activation_at_stop is output after modules[stop_at].
    """
    act = None
    out = x
    for i, m in enumerate(modules):
        out = m(out)
        if stop_at is not None and i == stop_at:
            act = out
    return out, act


def forward_modules(modules: List[torch.nn.Module], x: torch.Tensor, start: int = 0, end: int = None) -> torch.Tensor:
    """Run modules[start:end] sequentially."""
    if end is None:
        end = len(modules)
    for i in range(start, end):
        x = modules[i](x)
    return x

def forward_with_frozen_prefix(
    modules: List[torch.nn.Module],
    x: torch.Tensor,
    frozen_upto: int,
) -> torch.Tensor:
    """
    If frozen_upto >= 0, run prefix [0..frozen_upto] under no_grad, then detach boundary activation.
    Then run remaining modules with grad.
    """
    if frozen_upto < 0:
        return forward_modules(modules, x, 0, None)

    # Prefix: no_grad (no autograd graph, no backward)
    with torch.no_grad():
        h = forward_modules(modules, x, 0, frozen_upto + 1)

    # Important: boundary tensor is treated as constant input to suffix
    h = h.detach()

    # Suffix: normal grad
    out = forward_modules(modules, h, frozen_upto + 1, None)
    return out