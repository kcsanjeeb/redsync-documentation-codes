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