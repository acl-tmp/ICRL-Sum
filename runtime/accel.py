from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch
from contextlib import contextmanager, nullcontext

# ------------------------------------------------------------------
# 1) 条件导入 Ascend 自动迁移（按官方 PT_LMTMOG_0014 建议）
#    - PyTorch≤2.4 需:  import torch_npu  +  from torch_npu.contrib import transfer_to_npu
#    - PyTorch≥2.5.1: 无需 import torch_npu，但仍建议导入 transfer_to_npu 以启用迁移
#    - 重复导入是幂等的；若环境无 NPU/无包，则忽略。
# ------------------------------------------------------------------
try:  # PyTorch 2.5.1+ autoload: 无需 import torch_npu
    from torch_npu.contrib import transfer_to_npu  # type: ignore
except Exception:
    try:  # PyTorch≤2.4 需要显式 import torch_npu
        import torch_npu  # noqa: F401
        from torch_npu.contrib import transfer_to_npu  # type: ignore
    except Exception:
        transfer_to_npu = None  # type: ignore


# ---------------------------
# 基础能力探测
# ---------------------------

def _has_npu() -> bool:
    try:
        return bool(getattr(torch, "npu", None)) and torch.npu.is_available()
    except Exception:
        return False


def _has_cuda() -> bool:
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


def device(prefer: Optional[str] = None) -> torch.device:
    """选择设备（优先 NPU→CUDA→CPU），支持环境变量 ACCEL_DEVICE 覆盖。"""
    override = (prefer or os.getenv("ACCEL_DEVICE", "")).strip().lower()
    if override in {"npu", "cuda", "cpu"}:
        if override == "npu" and _has_npu():
            return torch.device("npu")
        if override == "cuda" and _has_cuda():
            return torch.device("cuda")
        if override == "cpu":
            return torch.device("cpu")
    if _has_npu():
        return torch.device("npu")
    if _has_cuda():
        return torch.device("cuda")
    return torch.device("cpu")


def is_npu() -> bool:
    return device().type == "npu"


def is_cuda() -> bool:
    return device().type == "cuda"


def is_cpu() -> bool:
    return device().type == "cpu"


# ---------------------------
# 分布式/设备相关工具
# ---------------------------

def ddp_backend() -> str:
    """返回推荐的分布式后端（NPU=hccl，其它=nccl）。"""
    return "hccl" if is_npu() else "nccl"


def set_device(local_rank: Optional[int] = None):
    """设置当前进程设备。NPU 使用 torch.npu.set_device，CUDA 使用 torch.cuda.set_device。"""
    if local_rank is None:
        return
    if is_npu():
        try:
            torch.npu.set_device(local_rank)  # type: ignore[attr-defined]
        except Exception:
            pass
    elif is_cuda():
        torch.cuda.set_device(local_rank)


@contextmanager
def synchronized():
    """在块首尾做设备同步（最佳努力）。"""
    try:
        if is_npu():
            torch.npu.synchronize()  # type: ignore[attr-defined]
            yield
            torch.npu.synchronize()  # type: ignore[attr-defined]
            return
        if is_cuda():
            torch.cuda.synchronize()
            yield
            torch.cuda.synchronize()
            return
    except Exception:
        pass
    yield


# ---------------------------
# AMP: autocast + GradScaler（NPU/CUDA/CPU 统一接口）
# ---------------------------

def preferred_amp_dtype() -> torch.dtype:
    # NPU/新卡普遍支持 bf16，保守选择 bf16，否则 fp16
    # CUDA: 若后端允许 bf16/tf32，优先 bf16
    try:
        if is_cuda() and (torch.backends.cuda.matmul.allow_bf16 or torch.backends.cuda.matmul.allow_tf32):
            return torch.bfloat16
    except Exception:
        pass
    # NPU bf16 能力探测（若有提供）
    try:
        if is_npu() and hasattr(torch.npu, "is_support_bf16") and torch.npu.is_support_bf16():  # type: ignore[attr-defined]
            return torch.bfloat16
    except Exception:
        pass
    return torch.float16


@contextmanager
def autocast(enabled: Optional[bool] = None, dtype: Optional[torch.dtype] = None):
    """统一的 autocast。CPU 回退为 no-op。"""
    enabled_flag = (enabled if enabled is not None else not is_cpu())
    amp_dtype = dtype or preferred_amp_dtype()

    if not enabled_flag:
        with nullcontext():
            yield
        return

    if is_npu():
        npu_amp = getattr(getattr(torch, "npu", object()), "amp", None)
        if hasattr(npu_amp, "autocast"):
            with npu_amp.autocast(dtype=amp_dtype):
                yield
            return
    if is_cuda():
        with torch.cuda.amp.autocast(dtype=amp_dtype):
            yield
        return
    with nullcontext():
        yield


class GradScaler:
    """统一的 GradScaler。NPU/CUDA 启用，CPU 为 no-op。"""

    def __init__(self, enabled: Optional[bool] = None, **kwargs):
        self.enabled = (enabled if enabled is not None else not is_cpu())
        self._impl = None
        if self.enabled:
            if is_npu():
                try:
                    npu_amp = getattr(getattr(torch, "npu", object()), "amp", None)
                    if npu_amp and hasattr(npu_amp, "GradScaler"):
                        self._impl = npu_amp.GradScaler(**kwargs)
                except Exception:
                    self._impl = None
            elif is_cuda():
                self._impl = torch.cuda.amp.GradScaler(**kwargs)
        if self._impl is None:
            self.enabled = False

    def scale(self, loss: torch.Tensor):
        return self._impl.scale(loss) if self.enabled else loss

    def step(self, optimizer: torch.optim.Optimizer):
        if self.enabled:
            return self._impl.step(optimizer)
        return optimizer.step()

    def update(self):
        if self.enabled:
            return self._impl.update()

    def unscale_(self, optimizer: torch.optim.Optimizer):
        if self.enabled:
            return self._impl.unscale_(optimizer)


# ---------------------------
# 内存与搬运
# ---------------------------

def empty_cache():
    if is_npu():
        try:
            torch.npu.empty_cache()  # type: ignore[attr-defined]
        except Exception:
            pass
    elif is_cuda():
        torch.cuda.empty_cache()


@dataclass
class MemoryStats:
    allocated: int
    reserved: int
    max_allocated: int
    max_reserved: int


def memory_stats() -> MemoryStats:
    if is_npu():
        try:
            mem_alloc = getattr(torch.npu, "memory_allocated", lambda: 0)()
            mem_rsrv = getattr(torch.npu, "memory_reserved", lambda: 0)()
            mem_max_alloc = getattr(torch.npu, "max_memory_allocated", lambda: 0)()
            mem_max_rsrv = getattr(torch.npu, "max_memory_reserved", lambda: 0)()
            return MemoryStats(mem_alloc, mem_rsrv, mem_max_alloc, mem_max_rsrv)
        except Exception:
            return MemoryStats(0, 0, 0, 0)
    if is_cuda():
        return MemoryStats(
            allocated=torch.cuda.memory_allocated(),
            reserved=torch.cuda.memory_reserved(),
            max_allocated=torch.cuda.max_memory_allocated(),
            max_reserved=torch.cuda.max_memory_reserved(),
        )
    return MemoryStats(0, 0, 0, 0)


def move_to(obj: Any, dev: Optional[torch.device] = None, non_blocking: bool = True):
    dev = dev or device()
    if isinstance(obj, torch.nn.Module):
        return obj.to(dev)
    if isinstance(obj, torch.Tensor):
        return obj.to(dev, non_blocking=non_blocking)
    if isinstance(obj, Mapping):
        return {k: move_to(v, dev, non_blocking) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        items = [move_to(x, dev, non_blocking) for x in obj]
        return type(obj)(items) if isinstance(obj, tuple) else items
    return obj


# ---------------------------
# 可重复性与初始化
# ---------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if _has_cuda():
        torch.cuda.manual_seed_all(seed)
    try:
        if _has_npu():
            torch.npu.manual_seed(seed)  # type: ignore[attr-defined]
    except Exception:
        pass


def init_accel(seed: Optional[int] = None, tf32: bool = False, local_rank: Optional[int] = None):
    """一次性初始化：
    - 设置随机种子；
    - 选择/设置设备（含 local_rank）；
    - CUDA 场景可选开启 TF32；
    - 打印设备信息横幅（ACCEL_SILENT=1 可关闭）。
    """
    if seed is not None:
        set_seed(seed)
    set_device(local_rank)

    if _has_cuda():
        try:
            torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
            torch.backends.cudnn.allow_tf32 = bool(tf32)
        except Exception:
            pass

    if os.getenv("ACCEL_SILENT", "0") != "1":
        try:
            if is_npu():
                print("[accel] Using Ascend NPU (backend=hccl)")
            elif is_cuda():
                idx = torch.cuda.current_device()
                print(f"[accel] Using CUDA:{idx} — {torch.cuda.get_device_name(idx)}")
            else:
                print("[accel] Using CPU")
        except Exception:
            pass

# run_train.py
# from runtime.accel import init_accel, device

# def main():
#     init_accel(seed=42, tf32=False)   # 一次性初始化，加速 & 随机种子
#     dev = device()                    # 自动检测 npu / cuda / cpu
#     print(f"Using device: {dev}")

    # 下面正常导入模型、数据等
    #...
# ============================================================
# Quick self-test
# ============================================================
if __name__ == "__main__":
    print("==== [Accel self-test] ====")
    try:
        # 1) 初始化
        init_accel(seed=1234)
        dev = device()
        print(f"[Device] -> {dev}")

        # 2) 检查 AMP 自动上下文
        from torch import nn
        model = nn.Linear(16, 8)
        model = move_to(model, dev)

        x = torch.randn(4, 16).to(dev)
        scaler = GradScaler()

        with autocast():
            y = model(x)
            loss = y.sum()
        print(f"[Autocast] success, output dtype={y.dtype}")

        # 3) 检查 GradScaler
        opt = torch.optim.SGD(model.parameters(), lr=1e-3)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        print("[GradScaler] ran without error")

        # 4) 检查内存状态
        stats = memory_stats()
        print(f"[Memory] allocated={stats.allocated/1e6:.2f} MB, reserved={stats.reserved/1e6:.2f} MB")

        # 5) 清理
        empty_cache()
        print("[Cache] cleared successfully")

        print("==== [Accel self-test: PASS] ====")
    except Exception as e:
        print("==== [Accel self-test: FAIL] ====")
        import traceback; traceback.print_exc()
