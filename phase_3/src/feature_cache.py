# src/feature_cache.py
from __future__ import annotations

import io
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch


def _now() -> float:
    return time.perf_counter()


def _has_cuda() -> bool:
    return torch.cuda.is_available()


def _cuda_sync_if_needed(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _compress_bytes(data: bytes, codec: str, level: int = 3) -> Tuple[bytes, str]:
    """
    codec: "zstd" | "lz4" | "zlib" | "none"
    returns: (compressed_bytes, resolved_codec_name)
    """
    codec = (codec or "zstd").lower()

    if codec == "none":
        return data, "none"

    if codec == "zstd":
        try:
            import zstandard as zstd  # pip install zstandard
            cctx = zstd.ZstdCompressor(level=level)
            return cctx.compress(data), "zstd"
        except Exception:
            # fall back
            codec = "zlib"

    if codec == "lz4":
        try:
            import lz4.frame  # pip install lz4
            return lz4.frame.compress(data, compression_level=level), "lz4"
        except Exception:
            codec = "zlib"

    if codec == "zlib":
        import zlib
        # zlib level 1..9
        lvl = max(1, min(int(level), 9))
        return zlib.compress(data, level=lvl), "zlib"

    # unknown codec -> no compression
    return data, "none"


def _decompress_bytes(data: bytes, codec: str) -> bytes:
    codec = (codec or "none").lower()

    if codec == "none":
        return data

    if codec == "zstd":
        import zstandard as zstd
        dctx = zstd.ZstdDecompressor()
        return dctx.decompress(data)

    if codec == "lz4":
        import lz4.frame
        return lz4.frame.decompress(data)

    if codec == "zlib":
        import zlib
        return zlib.decompress(data)

    # unknown -> treat as raw
    return data


@dataclass
class CacheStats:
    # time breakdown (seconds)
    d2h_s: float = 0.0
    h2d_s: float = 0.0
    serialize_s: float = 0.0
    deserialize_s: float = 0.0
    compress_s: float = 0.0
    decompress_s: float = 0.0

    # sizes (bytes)
    raw_bytes: int = 0
    compressed_bytes: int = 0

    # counters
    hit: int = 0
    miss: int = 0


class FeatureCache:
    """
    Naïve feature cache:
      - capture: GPU tensor -> CPU (D2H) -> torch.save bytes -> compress -> store
      - load: decompress -> torch.load CPU tensor -> GPU (H2D)
    """

    def __init__(
        self,
        codec: str = "zstd",
        level: int = 3,
        pin_memory: bool = False,
        max_entries: int = 8,
    ):
        self.codec = codec
        self.level = level
        self.pin_memory = pin_memory
        self.max_entries = max_entries

        # key -> entry dict
        self._store: Dict[str, Dict[str, Any]] = {}
        self._lru: Dict[str, float] = {}  # key -> last_access timestamp

        self.stats = CacheStats()

    def __len__(self) -> int:
        return len(self._store)

    def has(self, key: str) -> bool:
        return key in self._store

    def _touch(self, key: str):
        self._lru[key] = _now()

    def _evict_if_needed(self):
        if len(self._store) <= self.max_entries:
            return
        # evict least recently used
        oldest_key = min(self._lru.items(), key=lambda kv: kv[1])[0]
        self._store.pop(oldest_key, None)
        self._lru.pop(oldest_key, None)

    def capture(self, key: str, act_gpu: torch.Tensor) -> Dict[str, Any]:
        """
        Store activation under `key`.
        act_gpu: tensor on GPU or CPU (works for both).
        Returns per-call logging dict.
        """
        log: Dict[str, Any] = {}
        device = act_gpu.device

        # Ensure we don't keep graph
        act = act_gpu.detach()

        # D2H
        t0 = _now()
        if device.type == "cuda":
            _cuda_sync_if_needed(device)
        if act.device.type != "cpu":
            act_cpu = act.to("cpu", non_blocking=False)
        else:
            act_cpu = act
        if self.pin_memory and act_cpu.device.type == "cpu":
            try:
                act_cpu = act_cpu.pin_memory()
            except Exception:
                pass
        if device.type == "cuda":
            _cuda_sync_if_needed(device)
        d2h_s = _now() - t0

        # Serialize (torch.save -> bytes)
        t1 = _now()
        buf = io.BytesIO()
        torch.save(act_cpu, buf)
        raw = buf.getvalue()
        serialize_s = _now() - t1

        # Compress
        t2 = _now()
        compressed, used_codec = _compress_bytes(raw, self.codec, self.level)
        compress_s = _now() - t2

        # Store
        self._store[key] = {
            "codec": used_codec,
            "dtype": str(act_cpu.dtype),
            "shape": tuple(act_cpu.shape),
            "bytes": compressed,
        }
        self._touch(key)
        self._evict_if_needed()

        # Update stats
        self.stats.d2h_s += d2h_s
        self.stats.serialize_s += serialize_s
        self.stats.compress_s += compress_s
        self.stats.raw_bytes += len(raw)
        self.stats.compressed_bytes += len(compressed)
        self.stats.miss += 1

        # Per-call log
        log.update({
            "cache_key": key,
            "cache_codec": used_codec,
            "cache_shape": list(act_cpu.shape),
            "cache_dtype": str(act_cpu.dtype),
            "cache_bytes_raw": len(raw),
            "cache_bytes_compressed": len(compressed),
            "cache_d2h_s": d2h_s,
            "cache_serialize_s": serialize_s,
            "cache_compress_s": compress_s,
            "cache_store_entries": len(self._store),
            "reuse_hit": False,
        })
        return log

    def load(self, key: str, device: torch.device) -> Dict[str, Any]:
        """
        Load activation for `key` onto `device`.
        Returns dict with {"act": tensor, ...timings...}.
        """
        if key not in self._store:
            raise KeyError(f"FeatureCache miss for key={key}")

        entry = self._store[key]
        self._touch(key)

        log: Dict[str, Any] = {"cache_key": key, "reuse_hit": True}

        # Decompress
        t0 = _now()
        comp = entry["bytes"]
        codec = entry["codec"]
        raw = _decompress_bytes(comp, codec)
        decompress_s = _now() - t0

        # Deserialize (torch.load CPU tensor)
        t1 = _now()
        buf = io.BytesIO(raw)
        act_cpu = torch.load(buf, map_location="cpu")
        deserialize_s = _now() - t1

        # H2D
        t2 = _now()
        if device.type == "cuda":
            torch.cuda.synchronize()
        act = act_cpu.to(device, non_blocking=False)
        if device.type == "cuda":
            torch.cuda.synchronize()
        h2d_s = _now() - t2

        # Update stats
        self.stats.decompress_s += decompress_s
        self.stats.deserialize_s += deserialize_s
        self.stats.h2d_s += h2d_s
        self.stats.hit += 1

        log.update({
            "cache_codec": codec,
            "cache_bytes_raw": len(raw),
            "cache_bytes_compressed": len(comp),
            "cache_decompress_s": decompress_s,
            "cache_deserialize_s": deserialize_s,
            "cache_h2d_s": h2d_s,
        })
        log["act"] = act
        return log

    def hit_rate(self) -> float:
        total = self.stats.hit + self.stats.miss
        return float(self.stats.hit) / float(total) if total > 0 else 0.0
