"""Compile and load kernel.c."""

from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

_SOURCE = Path(__file__).with_name("kernel.c")


def _library() -> ctypes.CDLL:
    tag = hashlib.sha256(_SOURCE.read_bytes()).hexdigest()[:12]
    cache = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")) / "expertstream"
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"kernel-{tag}.so"
    if not target.exists():
        fd, tmp = tempfile.mkstemp(suffix=".so", dir=cache)
        os.close(fd)
        subprocess.run(
            [os.getenv("CC", "cc"), "-O3", "-march=native", "-shared", "-fPIC",
             "-o", tmp, str(_SOURCE), "-lm"], check=True)
        os.replace(tmp, target)
    return ctypes.CDLL(str(target))


_lib = _library()
_p, _n = ctypes.c_void_p, ctypes.c_int64
_lib.bf16_to_f32.argtypes = [_p, _p, _n]
_lib.bf16_matvec.argtypes = [_p, _n, _n, _p, _p]
_lib.expert_apply.argtypes = [_p, _p, _n, _n, _p, _p, _p]
for _f in (_lib.bf16_to_f32, _lib.bf16_matvec, _lib.expert_apply):
    _f.restype = None


def to_f32(src: np.ndarray) -> np.ndarray:
    out = np.empty(src.shape, np.float32)
    _lib.bf16_to_f32(src.ctypes.data, out.ctypes.data, src.size)
    return out


def matvec(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    out = np.empty(w.shape[0], np.float32)
    _lib.bf16_matvec(w.ctypes.data, w.shape[0], w.shape[1], x.ctypes.data, out.ctypes.data)
    return out


def expert_apply(win: np.ndarray, wout: np.ndarray, x: np.ndarray) -> np.ndarray:
    hidden, inner = wout.shape
    tmp, out = np.empty(3 * inner, np.float32), np.empty(hidden, np.float32)
    _lib.expert_apply(win.ctypes.data, wout.ctypes.data, hidden, inner,
                      x.ctypes.data, tmp.ctypes.data, out.ctypes.data)
    return out
