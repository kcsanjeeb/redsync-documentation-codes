# src/train_freeze_phase3_trainreuse_async_v1.py
import argparse
import hashlib
import json
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, models, transforms

from src.feature_cache import FeatureCache


# ----------------------------
# Messages
# ----------------------------
@dataclass
class ProbeMsg:
    boundary_idx: int
    epoch: int
    act_bytes: bytes
    act_shape: Tuple[int, ...]
    act_dtype: str


@dataclass
class DecisionMsg:
    epoch: int
    boundary_idx: int
    freeze_idx: Optional[int]
    plasticity: Optional[float]
    slope: Optional[float]
    hit_count: int
    note: str = ""


# ----------------------------
# Utils
# ----------------------------
def set_seed(seed: int = 1337) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str) -> None:
    d = os.path.dirname(p)
    if d:
        os.makedirs(d, exist_ok=True)


def write_jsonl(path: str, obj: Dict[str, Any]) -> None:
    ensure_dir(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()


def cuda_sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def freeze_module_bn_safe(m: nn.Module, freeze_bn_affine: bool = True) -> None:
    m.eval()
    for p in m.parameters(recurse=True):
        p.requires_grad = False
    for sub in m.modules():
        if isinstance(sub, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            sub.eval()
            sub.track_running_stats = True
            if freeze_bn_affine:
                if sub.weight is not None:
                    sub.weight.requires_grad = False
                if sub.bias is not None:
                    sub.bias.requires_grad = False


# ----------------------------
# Dataset that returns indices
# ----------------------------
class WithIndex(Dataset):
    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y = self.base[idx]
        return x, y, idx


def batch_key(indices: torch.Tensor, stage: int) -> str:
    """
    Stable key for a batch. We hash indices so the key stays short.
    """
    # indices: [B] on CPU
    arr = indices.detach().cpu().numpy().astype(np.int64, copy=False)
    h = hashlib.sha1(arr.tobytes()).hexdigest()[:16]
    return f"train/stage={stage}/h={h}/b={arr.shape[0]}"


# ----------------------------
# Build freeze units (kept for controller decisions)
# ----------------------------
def build_freeze_units_resnet18(model: nn.Module) -> List[nn.Module]:
    units: List[nn.Module] = []
    units.extend([model.conv1, model.bn1, model.relu])

    def add_layer(layer: nn.Sequential):
        for blk in layer:
            units.extend([blk.conv1, blk.bn1, blk.relu, blk.conv2, blk.bn2])
            if blk.downsample is not None:
                units.append(blk.downsample)

    add_layer(model.layer1)
    add_layer(model.layer2)
    add_layer(model.layer3)
    add_layer(model.layer4)

    units.extend([model.avgpool, model.fc])
    return units


# ----------------------------
# Split-forward ResNet (stage boundaries)
# ----------------------------
class ResNet18Split(nn.Module):
    """
    Provides stage-wise split forward for torchvision resnet18:

    stage 0: after stem (conv1->bn1->relu->maxpool)
    stage 1: after layer1
    stage 2: after layer2
    stage 3: after layer3
    stage 4: after layer4
    stage 5: after avgpool + flatten  (feature vector)
    stage 6: after fc (logits)  [not used as boundary]
    """

    def __init__(self, base: models.ResNet):
        super().__init__()
        self.base = base

    def forward_prefix(self, x: torch.Tensor, stage: int) -> torch.Tensor:
        b = self.base
        # stem
        x = b.conv1(x)
        x = b.bn1(x)
        x = b.relu(x)
        x = b.maxpool(x)
        if stage == 0:
            return x

        x = b.layer1(x)
        if stage == 1:
            return x

        x = b.layer2(x)
        if stage == 2:
            return x

        x = b.layer3(x)
        if stage == 3:
            return x

        x = b.layer4(x)
        if stage == 4:
            return x

        x = b.avgpool(x)
        x = torch.flatten(x, 1)
        if stage == 5:
            return x

        # stage >= 6 is "logits", but prefix for training reuse shouldn’t go there
        return x

    def forward_suffix(self, act: torch.Tensor, stage: int) -> torch.Tensor:
        b = self.base
        x = act

        if stage <= 0:
            x = b.layer1(x)
        if stage <= 1:
            x = b.layer2(x)
        if stage <= 2:
            x = b.layer3(x)
        if stage <= 3:
            x = b.layer4(x)
        if stage <= 4:
            x = b.avgpool(x)
            x = torch.flatten(x, 1)
        # stage 5 means we already have flattened vector
        logits = b.fc(x)
        return logits


# ----------------------------
# BoundaryModel (for probe hook)
# ----------------------------
class BoundaryModel(nn.Module):
    def __init__(self, base: nn.Module, units: List[nn.Module]):
        super().__init__()
        self.base = base
        self.units = units
        self._last_boundary_act = None

    def forward_with_boundary(self, x: torch.Tensor, boundary_idx: int):
        self._last_boundary_act = None
        handle = None

        if 0 <= boundary_idx < len(self.units):

            def _hook(_m, _inp, out):
                self._last_boundary_act = out

            handle = self.units[boundary_idx].register_forward_hook(_hook)

        logits = self.base(x)

        if handle is not None:
            handle.remove()

        if self._last_boundary_act is None:
            self._last_boundary_act = logits

        return logits, self._last_boundary_act


# ----------------------------
# Controller
# ----------------------------
class FreezeController:
    def __init__(
        self,
        W: int,
        slope_T: float,
        required_hits: int,
        ref_update_every_epochs: int,
        use_relative_plasticity: bool,
        rel_eps: float = 1e-6,
    ):
        self.W = W
        self.slope_T = slope_T
        self.required_hits = required_hits
        self.ref_update_every_epochs = ref_update_every_epochs
        self.use_relative_plasticity = use_relative_plasticity
        self.rel_eps = rel_eps

        self.cur_boundary: Optional[int] = None
        self.ref_feat: Optional[torch.Tensor] = None
        self.history: List[float] = []
        self.hits = 0

    @staticmethod
    def decode_probe(msg: ProbeMsg) -> torch.Tensor:
        arr = np.frombuffer(msg.act_bytes, dtype=np.float32).copy()
        t = torch.from_numpy(arr).reshape(msg.act_shape)
        return t

    @staticmethod
    def featurize(act: torch.Tensor) -> torch.Tensor:
        if act.dim() >= 4:
            dims = [0] + list(range(2, act.dim()))
            feat = act.mean(dim=dims)
        elif act.dim() == 2:
            feat = act.mean(dim=0)
        else:
            feat = act.flatten()
        return feat.detach().to(torch.float32).cpu().contiguous()

    def _reset_for_boundary(self, boundary_idx: int, epoch: int) -> DecisionMsg:
        self.cur_boundary = boundary_idx
        self.ref_feat = None
        self.history = []
        self.hits = 0
        return DecisionMsg(
            epoch=epoch,
            boundary_idx=boundary_idx,
            freeze_idx=None,
            plasticity=None,
            slope=None,
            hit_count=0,
            note=f"reset_boundary:{boundary_idx}",
        )

    def on_probe(self, msg: ProbeMsg) -> DecisionMsg:
        if self.cur_boundary is None or msg.boundary_idx != self.cur_boundary:
            return self._reset_for_boundary(msg.boundary_idx, msg.epoch)

        act = self.decode_probe(msg)
        feat = self.featurize(act)

        do_ref = False
        if self.ref_feat is None:
            do_ref = True
        elif self.ref_update_every_epochs > 0 and (msg.epoch % self.ref_update_every_epochs == 0):
            do_ref = True

        if do_ref:
            self.ref_feat = feat.clone()

        if self.ref_feat is None:
            plast = float("nan")
        else:
            diff = (feat - self.ref_feat).abs().mean().item()
            if self.use_relative_plasticity:
                denom = self.ref_feat.abs().mean().item() + self.rel_eps
                plast = diff / denom
            else:
                plast = diff

        self.history.append(float(plast))
        if len(self.history) > self.W:
            self.history = self.history[-self.W:]

        slope = None
        freeze_idx = None
        note = "warming_history"

        if len(self.history) >= self.W:
            denom = max(1, self.W - 1)
            slope = (self.history[-1] - self.history[0]) / denom

            if abs(slope) < self.slope_T:
                self.hits += 1
            else:
                self.hits = 0

            note = "ok"
            if self.hits >= self.required_hits:
                freeze_idx = msg.boundary_idx

        return DecisionMsg(
            epoch=msg.epoch,
            boundary_idx=msg.boundary_idx,
            freeze_idx=freeze_idx,
            plasticity=float(plast),
            slope=None if slope is None else float(slope),
            hit_count=int(self.hits),
            note=note,
        )


def controller_thread_main(
    probe_q: "queue.Queue[ProbeMsg]",
    decision_q: "queue.Queue[DecisionMsg]",
    stop_event: threading.Event,
    controller: FreezeController,
) -> None:
    while not stop_event.is_set():
        try:
            msg = probe_q.get(timeout=0.25)
        except queue.Empty:
            continue
        try:
            dec = controller.on_probe(msg)
            decision_q.put(dec)
        except Exception as e:
            decision_q.put(
                DecisionMsg(
                    epoch=msg.epoch,
                    boundary_idx=msg.boundary_idx,
                    freeze_idx=None,
                    plasticity=None,
                    slope=None,
                    hit_count=0,
                    note=f"err:{e}",
                )
            )


# ----------------------------
# Eval
# ----------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        x, y = batch[0], batch[1]
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return 100.0 * correct / max(1, total)


# ----------------------------
# Training with reuse
# ----------------------------
def map_frozen_to_stage(frozen_upto: int) -> int:
    """
    Conservative mapping from fine-grained 'frozen_upto' (units index) to a stage boundary.
    This mapping assumes you are freezing early units first; we only reuse if we are confident
    the whole prefix stage is frozen.

    You can tighten later by explicitly tracking which stage modules are frozen.
    """
    # Empirical safe thresholds for your unit construction:
    # units = [conv1,bn1,relu] + layer1 blocks ... + layer4 ... + avgpool + fc
    # We'll map:
    # stage 0 = after stem (needs conv1,bn1,relu + maxpool). We only freeze conv/bn/relu here.
    # stage 1..4 = after layer1..4. stage 5 = after avgpool+flatten.
    #
    # Because maxpool isn't in units list, stage0 reuse is still safe once stem is frozen.
    if frozen_upto < 2:
        return -1  # no reuse yet
    # Once we start freezing inside layer1, allow stage0 reuse
    if frozen_upto >= 2:
        return 0
    return -1


def train_one_epoch_with_reuse(
    base: nn.Module,
    split: ResNet18Split,
    loader: DataLoader,
    device: torch.device,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    fc: Optional[FeatureCache],
    reuse_enabled: bool,
    reuse_stage: int,
) -> Tuple[float, float, Dict[str, Any]]:
    base.train()
    total_loss = 0.0
    total_acc = 0.0
    n = 0

    # epoch metrics
    m: Dict[str, Any] = {
        "train_forward_s": 0.0,
        "train_backward_s": 0.0,
        "train_step_s": 0.0,
        "reuse_hit": 0,
        "reuse_miss": 0,
        "reuse_stage": reuse_stage,
        "reuse_enabled": reuse_enabled,
    }
    cache_over: Dict[str, float] = {
        "cache_d2h_s": 0.0,
        "cache_h2d_s": 0.0,
        "cache_serialize_s": 0.0,
        "cache_deserialize_s": 0.0,
        "cache_compress_s": 0.0,
        "cache_decompress_s": 0.0,
    }
    m["cache_overhead_s"] = cache_over

    for x, y, idx in loader:
        x, y = x.to(device), y.to(device)
        idx_cpu = idx.detach().cpu()

        optimizer.zero_grad(set_to_none=True)

        # --- Forward ---
        t_f0 = time.perf_counter()
        cuda_sync_if_needed(device)

        logits = None
        key = None

        if reuse_enabled and fc is not None and reuse_stage >= 0:
            key = batch_key(idx_cpu, stage=reuse_stage)
            if fc.has(key):
                out = fc.load(key, device=device)
                act = out.pop("act")
                logits = split.forward_suffix(act, stage=reuse_stage)
                m["reuse_hit"] += 1

                # overhead accounting
                for k in list(cache_over.keys()):
                    if k in out:
                        cache_over[k] += float(out[k])
            else:
                # miss: compute prefix+suffix and store
                act = split.forward_prefix(x, stage=reuse_stage)
                logits = split.forward_suffix(act, stage=reuse_stage)
                m["reuse_miss"] += 1

                cap = fc.capture(key, act.detach())
                # overhead accounting
                for k in list(cache_over.keys()):
                    if k in cap:
                        cache_over[k] += float(cap[k])
        else:
            logits = base(x)

        cuda_sync_if_needed(device)
        m["train_forward_s"] += time.perf_counter() - t_f0

        loss = criterion(logits, y)

        # --- Backward ---
        t_b0 = time.perf_counter()
        cuda_sync_if_needed(device)
        loss.backward()
        cuda_sync_if_needed(device)
        m["train_backward_s"] += time.perf_counter() - t_b0

        # --- Step ---
        t_s0 = time.perf_counter()
        cuda_sync_if_needed(device)
        optimizer.step()
        cuda_sync_if_needed(device)
        m["train_step_s"] += time.perf_counter() - t_s0

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(logits.detach(), y) * bs
        n += bs

    return total_loss / max(1, n), 100.0 * total_acc / max(1, n), m


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_dir", type=str, default="./data")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--wd", type=float, default=5e-4)

    ap.add_argument("--bootstrap_epochs", type=int, default=120)
    ap.add_argument("--probe_bs", type=int, default=64)
    ap.add_argument("--probe_every_epochs", type=int, default=1)

    ap.add_argument("--W", type=int, default=10)
    ap.add_argument("--slope_T", type=float, default=8e-4)
    ap.add_argument("--required_hits", type=int, default=3)
    ap.add_argument("--ref_update_every_epochs", type=int, default=20)

    ap.add_argument("--use_relative_plasticity", action="store_true")
    ap.add_argument("--max_freeze_modules", type=int, default=8)
    ap.add_argument("--freeze_bn_affine", action="store_true")

    ap.add_argument("--no_aug", action="store_true")
    ap.add_argument("--cpu_threads", type=int, default=4)

    ap.add_argument("--ckpt_path", type=str, default="runs/checkpoints/egeria.pt")
    ap.add_argument("--save_ckpt_on_freeze_k", type=int, default=0)
    ap.add_argument("--stop_after_freeze_k", type=int, default=0)

    ap.add_argument("--log_path", type=str, default="runs/train.jsonl")
    ap.add_argument("--debug_decisions", action="store_true")

    # ---- Feature cache (for TRAIN reuse) ----
    ap.add_argument("--train_reuse", action="store_true", help="Enable training-time forward reuse.")
    ap.add_argument("--fc_codec", type=str, default="zstd")
    ap.add_argument("--fc_level", type=int, default=3)
    ap.add_argument("--fc_pin_memory", action="store_true")
    ap.add_argument("--fc_max_items", type=int, default=256)
    ap.add_argument("--fc_enable_after_freeze_k", type=int, default=1)

    args = ap.parse_args()

    torch.set_num_threads(max(1, args.cpu_threads))
    set_seed(1337)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    if args.no_aug:
        train_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])

    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_ds0 = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=train_tf)
    test_ds0 = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=test_tf)

    train_ds = WithIndex(train_ds0)
    test_ds = test_ds0  # test doesn't need indices

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)

    # Fixed probe subset
    probe_indices = list(range(min(len(train_ds0), args.probe_bs)))
    probe_loader = DataLoader(Subset(train_ds0, probe_indices), batch_size=args.probe_bs, shuffle=False, num_workers=0)

    # Model
    base = models.resnet18(num_classes=10).to(device)
    units = build_freeze_units_resnet18(base)
    total_units = len(units)
    boundary_model = BoundaryModel(base, units)
    split = ResNet18Split(base)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(base.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)

    # Async controller
    probe_q: "queue.Queue[ProbeMsg]" = queue.Queue(maxsize=8)
    decision_q: "queue.Queue[DecisionMsg]" = queue.Queue(maxsize=64)
    stop_event = threading.Event()

    controller = FreezeController(
        W=args.W,
        slope_T=args.slope_T,
        required_hits=args.required_hits,
        ref_update_every_epochs=args.ref_update_every_epochs,
        use_relative_plasticity=args.use_relative_plasticity,
    )

    t = threading.Thread(
        target=controller_thread_main,
        args=(probe_q, decision_q, stop_event, controller),
        daemon=True,
    )
    t.start()

    # FeatureCache for training reuse
    fc: Optional[FeatureCache] = None
    if args.train_reuse:
        fc = FeatureCache(
            codec=args.fc_codec,
            level=args.fc_level,
            pin_memory=args.fc_pin_memory,
            max_entries=args.fc_max_items,
        )

    frozen_upto = -1
    last_freeze_epoch = -1
    silent_ctr = 0

    def drain_and_apply_freezes(epoch: int) -> Tuple[Optional[DecisionMsg], int]:
        nonlocal frozen_upto, last_freeze_epoch
        latest: Optional[DecisionMsg] = None
        while True:
            try:
                d = decision_q.get_nowait()
            except queue.Empty:
                break
            latest = d

        if latest is None:
            return None, 0

        if args.debug_decisions:
            print(
                f"[CTRL] epoch={latest.epoch} b={latest.boundary_idx} freeze_idx={latest.freeze_idx} "
                f"plast={latest.plasticity} slope={latest.slope} hits={latest.hit_count} note={latest.note}"
            )

        if latest.freeze_idx is None:
            return latest, 0

        target = int(latest.freeze_idx)
        target = min(target, args.max_freeze_modules - 1, total_units - 1)

        applied = 0
        while frozen_upto < target:
            frozen_upto += 1
            freeze_module_bn_safe(units[frozen_upto], freeze_bn_affine=args.freeze_bn_affine)
            applied += 1
            last_freeze_epoch = epoch
            if args.debug_decisions:
                print(f"[APPLY] froze module {frozen_upto} (frozen={frozen_upto+1}/{total_units})")

        return latest, applied

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ---- Training with optional reuse ----
        frozen_k = frozen_upto + 1
        reuse_enabled = bool(args.train_reuse and (fc is not None) and (frozen_k >= args.fc_enable_after_freeze_k))
        reuse_stage = map_frozen_to_stage(frozen_upto)

        train_loss, train_acc, train_metrics = train_one_epoch_with_reuse(
            base=base,
            split=split,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            criterion=criterion,
            fc=fc,
            reuse_enabled=reuse_enabled,
            reuse_stage=reuse_stage,
        )

        test_acc = evaluate(base, test_loader, device)

        # ---- Probe (still used for controller decisions) ----
        probe = False
        froze_idx = None
        ckpt_saved = False
        cache_log: Dict[str, Any] = {}

        if epoch >= args.bootstrap_epochs and args.probe_every_epochs > 0 and epoch % args.probe_every_epochs == 0:
            probe = True
            next_idx = min(frozen_upto + 1, total_units - 1)

            base.eval()
            with torch.no_grad():
                xb, _ = next(iter(probe_loader))
                xb = xb.to(device)
                _, act = boundary_model.forward_with_boundary(xb, boundary_idx=next_idx)
                act_to_send = act.detach().to("cpu", dtype=torch.float32).contiguous()

                msg = ProbeMsg(
                    boundary_idx=int(next_idx),
                    epoch=int(epoch),
                    act_bytes=act_to_send.numpy().tobytes(),
                    act_shape=tuple(act_to_send.shape),
                    act_dtype="float32",
                )
                try:
                    probe_q.put_nowait(msg)
                except queue.Full:
                    pass

        dec, _ = drain_and_apply_freezes(epoch)
        if dec is None:
            silent_ctr += 1
        else:
            silent_ctr = 0
            froze_idx = dec.freeze_idx

        frozen_k = frozen_upto + 1

        # ---- Checkpoint ----
        if args.save_ckpt_on_freeze_k > 0 and frozen_k == args.save_ckpt_on_freeze_k and last_freeze_epoch == epoch:
            ensure_dir(args.ckpt_path)
            torch.save(
                {
                    "epoch": epoch,
                    "model": base.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "frozen_upto": frozen_upto,
                    "units": total_units,
                    "args": vars(args),
                },
                args.ckpt_path,
            )
            ckpt_saved = True

        dt = time.time() - t0

        reuse_hit = int(train_metrics.get("reuse_hit", 0))
        reuse_miss = int(train_metrics.get("reuse_miss", 0))
        reuse_total = reuse_hit + reuse_miss
        reuse_hit_rate = (reuse_hit / reuse_total) if reuse_total > 0 else 0.0

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} train_acc={train_acc:.2f}% | "
            f"test_acc={test_acc:.2f}% | time={dt:.2f}s | frozen={frozen_k}/{total_units} | "
            f"reuse={reuse_enabled} stage={reuse_stage} hit_rate={reuse_hit_rate:.3f} | "
            f"probe={probe} | froze={froze_idx} | ckpt_saved={ckpt_saved}"
        )

        write_jsonl(
            args.log_path,
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "time_sec": dt,
                "frozen_upto": frozen_upto,
                "frozen_k": frozen_k,
                "total_units": total_units,
                "probe": probe,
                "froze_idx": froze_idx,
                "ckpt_saved": ckpt_saved,
                "controller": None
                if dec is None
                else {
                    "epoch": dec.epoch,
                    "boundary_idx": dec.boundary_idx,
                    "freeze_idx": dec.freeze_idx,
                    "plasticity": dec.plasticity,
                    "slope": dec.slope,
                    "hit_count": dec.hit_count,
                    "note": dec.note,
                },
                "train_metrics": train_metrics,
                "reuse_summary": {
                    "enabled": reuse_enabled,
                    "stage": reuse_stage,
                    "hit": reuse_hit,
                    "miss": reuse_miss,
                    "hit_rate": reuse_hit_rate,
                },
                "feature_cache_probe": cache_log if cache_log else None,
            },
        )

        if silent_ctr > 25 and epoch >= args.bootstrap_epochs:
            print("[WARN] No DecisionMsg received for 25 probe epochs (controller falling behind or probes dropped).")

        if args.stop_after_freeze_k > 0 and frozen_k >= args.stop_after_freeze_k:
            break

    stop_event.set()
    try:
        t.join(timeout=1.0)
    except Exception:
        pass


if __name__ == "__main__":
    main()
