# src/train_freeze_phase2_async_fixed_fc_reuse_gated_v4.py
import argparse
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
from torch.utils.data import DataLoader, Subset
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
    act_dtype: str  # "float32"


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
    """
    Tracks plasticity at the current boundary only.
    When boundary changes, state resets.
    """
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
            decision_q.put(DecisionMsg(
                epoch=msg.epoch,
                boundary_idx=msg.boundary_idx,
                freeze_idx=None,
                plasticity=None,
                slope=None,
                hit_count=0,
                note=f"err:{e}",
            ))


# ----------------------------
# Train / eval
# ----------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return 100.0 * correct / max(1, total)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(logits.detach(), y) * bs
        n += bs

    return total_loss / max(1, n), 100.0 * total_acc / max(1, n)


# ----------------------------
# EMA helpers (Phase-3D gating)
# ----------------------------
def ema_update(cur: Optional[float], x: float, alpha: float) -> float:
    if cur is None:
        return float(x)
    return float(alpha * x + (1.0 - alpha) * cur)


def sum_overhead_from_fc_log(log: Dict[str, Any]) -> float:
    """
    FeatureCache returns logs that may include timing fields.
    We conservatively sum known fields if present.
    """
    s = 0.0
    for k in ("d2h_s", "h2d_s", "serialize_s", "deserialize_s", "compress_s", "decompress_s"):
        if k in log and log[k] is not None:
            s += float(log[k])
    # some implementations store nested keys; handled in caller
    return float(s)


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

    # -------- FeatureCache --------
    ap.add_argument("--feature_cache", action="store_true")
    ap.add_argument("--fc_codec", type=str, default="zstd")
    ap.add_argument("--fc_level", type=int, default=3)
    ap.add_argument("--fc_pin_memory", action="store_true")
    ap.add_argument("--fc_max_items", type=int, default=256)
    ap.add_argument("--fc_enable_after_freeze_k", type=int, default=1)

    # -------- Phase-3D gated reuse controls --------
    ap.add_argument("--reuse_gated", action="store_true",
                    help="Enable Phase-3D: gated reuse on cache hit (skip probe forward if break-even).")
    ap.add_argument("--gate_alpha", type=float, default=0.10,
                    help="EMA alpha for p_hit / fwd_ms / overhead_s.")
    ap.add_argument("--gate_margin", type=float, default=0.20,
                    help="Safety margin: require benefit > overhead*(1+margin).")
    ap.add_argument("--gate_min_obs", type=int, default=10,
                    help="Minimum hit/miss observations at boundary before enabling gated reuse.")
    ap.add_argument("--gate_min_phit", type=float, default=0.50,
                    help="Minimum p_hit EMA required before gating can pass (stability).")
    ap.add_argument("--gate_warmup_capture", type=int, default=1,
                    help="On first ever encounter, force a capture path even if key exists (debug).")

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

    train_ds = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=train_tf)
    test_ds = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)

    # Fixed probe subset
    probe_indices = list(range(min(len(train_ds), args.probe_bs)))
    probe_loader = DataLoader(Subset(train_ds, probe_indices), batch_size=args.probe_bs, shuffle=False, num_workers=0)

    # Model
    model = models.resnet18(num_classes=10).to(device)
    units = build_freeze_units_resnet18(model)
    total_units = len(units)
    boundary_model = BoundaryModel(model, units)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)

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

    # FeatureCache
    fc: Optional[FeatureCache] = None
    if args.feature_cache:
        fc = FeatureCache(
            codec=args.fc_codec,
            level=args.fc_level,
            pin_memory=args.fc_pin_memory,
            max_entries=args.fc_max_items,
        )

    # Phase-3D gating state (per boundary)
    p_hit_ema: Dict[int, Optional[float]] = {}
    obs_cnt: Dict[int, int] = {}
    probe_fwd_ms_ema: Dict[int, Optional[float]] = {}
    fc_load_overhead_s_ema: Dict[int, Optional[float]] = {}

    # For debugging / forcing a capture first time
    seen_boundary: Dict[int, int] = {}

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
            print(f"[CTRL] epoch={latest.epoch} b={latest.boundary_idx} freeze_idx={latest.freeze_idx} "
                  f"plast={latest.plasticity} slope={latest.slope} hits={latest.hit_count} note={latest.note}")

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

        train_loss, train_acc = train_one_epoch(model, train_loader, device, optimizer, criterion)
        test_acc = evaluate(model, test_loader, device)

        probe = False
        froze_idx = None
        ckpt_saved = False
        cache_log: Dict[str, Any] = {}

        # ----------------------------
        # PROBE + FeatureCache (+ Phase-3D gated reuse)
        # ----------------------------
        if epoch >= args.bootstrap_epochs and args.probe_every_epochs > 0 and epoch % args.probe_every_epochs == 0:
            probe = True
            next_idx = min(frozen_upto + 1, total_units - 1)

            model.eval()
            with torch.no_grad():
                xb, _ = next(iter(probe_loader))
                xb = xb.to(device)

                frozen_k = frozen_upto + 1
                do_fc = (fc is not None) and (frozen_k >= args.fc_enable_after_freeze_k)

                if do_fc:
                    key = f"probe_b{next_idx}"  # stable key -> hits after first capture
                    cache_log["fc_enabled"] = True
                    cache_log["fc_key"] = key
                    cache_log["fc_hit_rate"] = fc.hit_rate()

                    # Init gate state
                    if next_idx not in p_hit_ema:
                        p_hit_ema[next_idx] = None
                        obs_cnt[next_idx] = 0
                        probe_fwd_ms_ema[next_idx] = None
                        fc_load_overhead_s_ema[next_idx] = None
                        seen_boundary[next_idx] = 0

                    seen_boundary[next_idx] += 1

                    # ---------- Option A: cache hit path ----------
                    if fc.has(key):
                        obs_cnt[next_idx] += 1
                        p_hit_ema[next_idx] = ema_update(p_hit_ema[next_idx], 1.0, args.gate_alpha)

                        # Always measure load overhead on a hit (we need this for the gate)
                        fc_t0 = time.perf_counter()
                        load_log = fc.load(key, device=device)
                        act_loaded = load_log.pop("act")
                        fc_total_s = time.perf_counter() - fc_t0

                        # Extract overhead sum (support both flat and nested logs)
                        overhead_s = 0.0
                        overhead_s += sum_overhead_from_fc_log(load_log)
                        if "timings" in load_log and isinstance(load_log["timings"], dict):
                            overhead_s += sum_overhead_from_fc_log(load_log["timings"])

                        fc_load_overhead_s_ema[next_idx] = ema_update(
                            fc_load_overhead_s_ema[next_idx],
                            overhead_s,
                            args.gate_alpha
                        )

                        # Decide whether to SKIP probe forward
                        gate_ok = False
                        reason = "gate_disabled"
                        if args.reuse_gated:
                            if obs_cnt[next_idx] < args.gate_min_obs:
                                gate_ok = False
                                reason = f"gate_wait_obs:{obs_cnt[next_idx]}/{args.gate_min_obs}"
                            elif (p_hit_ema[next_idx] is not None) and (p_hit_ema[next_idx] < args.gate_min_phit):
                                gate_ok = False
                                reason = f"gate_low_phit:{p_hit_ema[next_idx]:.3f}<{args.gate_min_phit}"
                            elif probe_fwd_ms_ema[next_idx] is None:
                                gate_ok = False
                                reason = "gate_no_fwd_profile"
                            else:
                                # benefit = p_hit * saved_forward
                                p = float(p_hit_ema[next_idx] or 0.0)
                                saved_s = float(probe_fwd_ms_ema[next_idx]) / 1000.0
                                cost_s = float(fc_load_overhead_s_ema[next_idx] or overhead_s)
                                gate_ok = (p * saved_s) > (cost_s * (1.0 + args.gate_margin))
                                reason = "gate_pass" if gate_ok else "gate_fail_break_even"

                        # Optional: force first encounter to be capture-like (debug)
                        if args.gate_warmup_capture > 0 and seen_boundary[next_idx] <= args.gate_warmup_capture:
                            gate_ok = False
                            reason = f"gate_forced_warmup:{seen_boundary[next_idx]}/{args.gate_warmup_capture}"

                        cache_log.update({
                            "fc_mode": "hit_load",
                            "fc_total_probe_path_s": fc_total_s,
                            "fc_load": {k: v for k, v in load_log.items() if k != "act"},
                            "gate": {
                                "enabled": bool(args.reuse_gated),
                                "ok": bool(gate_ok),
                                "reason": reason,
                                "p_hit_ema": p_hit_ema[next_idx],
                                "obs_cnt": obs_cnt[next_idx],
                                "probe_fwd_ms_ema": probe_fwd_ms_ema[next_idx],
                                "fc_load_overhead_s_ema": fc_load_overhead_s_ema[next_idx],
                                "gate_margin": args.gate_margin,
                            }
                        })

                        if gate_ok:
                            # TRUE REUSE: skip any model forward; send cached activation
                            act_to_send = act_loaded.detach().to("cpu", dtype=torch.float32).contiguous()
                        else:
                            # Gate failed -> compute probe forward (like v3) to keep controller honest
                            torch.cuda.synchronize(device) if device.type == "cuda" else None
                            ev0 = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
                            ev1 = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None

                            if ev0 is not None:
                                ev0.record()
                            _, act = boundary_model.forward_with_boundary(xb, boundary_idx=next_idx)
                            if ev1 is not None:
                                ev1.record()
                                torch.cuda.synchronize()
                                fwd_ms = float(ev0.elapsed_time(ev1))
                                probe_fwd_ms_ema[next_idx] = ema_update(probe_fwd_ms_ema[next_idx], fwd_ms, args.gate_alpha)
                                cache_log["probe_forward_ms"] = fwd_ms
                                cache_log["probe_forward_ms_ema"] = probe_fwd_ms_ema[next_idx]

                            act = act.detach()
                            act_to_send = act.to("cpu", dtype=torch.float32).contiguous()

                    # ---------- Option B: cache miss path ----------
                    else:
                        obs_cnt[next_idx] += 1
                        p_hit_ema[next_idx] = ema_update(p_hit_ema[next_idx], 0.0, args.gate_alpha)

                        # Must run probe forward (to get act)
                        torch.cuda.synchronize(device) if device.type == "cuda" else None
                        ev0 = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
                        ev1 = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None

                        if ev0 is not None:
                            ev0.record()
                        _, act = boundary_model.forward_with_boundary(xb, boundary_idx=next_idx)
                        if ev1 is not None:
                            ev1.record()
                            torch.cuda.synchronize()
                            fwd_ms = float(ev0.elapsed_time(ev1))
                            probe_fwd_ms_ema[next_idx] = ema_update(probe_fwd_ms_ema[next_idx], fwd_ms, args.gate_alpha)
                            cache_log["probe_forward_ms"] = fwd_ms
                            cache_log["probe_forward_ms_ema"] = probe_fwd_ms_ema[next_idx]

                        act = act.detach()

                        fc_t0 = time.perf_counter()
                        cap_log = fc.capture(key, act)  # GPU tensor -> measure D2H + serialize + compress
                        fc_total_s = time.perf_counter() - fc_t0

                        cache_log.update({
                            "fc_mode": "miss_capture",
                            "fc_total_probe_path_s": fc_total_s,
                            "fc_capture": {k: v for k, v in cap_log.items() if k != "act"},
                            "gate": {
                                "enabled": bool(args.reuse_gated),
                                "ok": False,
                                "reason": "miss",
                                "p_hit_ema": p_hit_ema[next_idx],
                                "obs_cnt": obs_cnt[next_idx],
                                "probe_fwd_ms_ema": probe_fwd_ms_ema[next_idx],
                            }
                        })

                        act_to_send = act.to("cpu", dtype=torch.float32).contiguous()

                    # cumulative stats snapshot
                    st = fc.stats
                    cache_log["fc_cum"] = {
                        "hit": st.hit,
                        "miss": st.miss,
                        "raw_bytes": st.raw_bytes,
                        "compressed_bytes": st.compressed_bytes,
                        "d2h_s": st.d2h_s,
                        "h2d_s": st.h2d_s,
                        "serialize_s": st.serialize_s,
                        "deserialize_s": st.deserialize_s,
                        "compress_s": st.compress_s,
                        "decompress_s": st.decompress_s,
                    }

                else:
                    # No FeatureCache: just compute act and send
                    cache_log = {"fc_enabled": False}
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

        # Apply freezes
        dec, _ = drain_and_apply_freezes(epoch)
        if dec is None:
            silent_ctr += 1
        else:
            silent_ctr = 0
            froze_idx = dec.freeze_idx

        frozen_k = frozen_upto + 1

        # Optional checkpoint save
        if args.save_ckpt_on_freeze_k > 0 and frozen_k == args.save_ckpt_on_freeze_k and last_freeze_epoch == epoch:
            ensure_dir(args.ckpt_path)
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "frozen_upto": frozen_upto,
                    "units": total_units,
                    "args": vars(args),
                },
                args.ckpt_path,
            )
            ckpt_saved = True

        dt = time.time() - t0

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} train_acc={train_acc:.2f}% | "
            f"test_acc={test_acc:.2f}% | time={dt:.2f}s | frozen={frozen_k}/{total_units} | "
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
                "controller": None if dec is None else {
                    "epoch": dec.epoch,
                    "boundary_idx": dec.boundary_idx,
                    "freeze_idx": dec.freeze_idx,
                    "plasticity": dec.plasticity,
                    "slope": dec.slope,
                    "hit_count": dec.hit_count,
                    "note": dec.note,
                },
                "feature_cache": cache_log if cache_log else None,
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
