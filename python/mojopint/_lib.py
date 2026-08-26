"""ctypes bridge to the Mojo conversion and dimensionality kernels."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import ctypes
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")
LIB = os.environ.get("MOJOPINT_LIB") or os.path.join(
    ROOT, "dist", "libmojo-pint.so"
)

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mp_convert_f64": ([I, I, I, F, F], None),
    "mp_convert_f64_serial": ([I, I, I, F, F], None),
    "mp_add_converted_f64": ([I, I, I, I, F, F, F], None),
    "mp_add_converted_f64_serial": ([I, I, I, I, F, F, F], None),
    "mp_combine_dimensions": ([I, I, I, I, F], None),
    "mp_dimensions_equal_many": ([I, I, I, I, F], None),
}
_PARALLEL_THRESHOLD = 1_000_000
_PARALLEL_WORKERS = min(16, os.cpu_count() or 1)
_executor = ThreadPoolExecutor(max_workers=_PARALLEL_WORKERS)


class BuildError(RuntimeError):
    pass


def mojo_command() -> list[str]:
    override = os.environ.get("MOJOPINT_MOJO")
    if override:
        return override.split()
    found = shutil.which("mojo")
    if found:
        return [found]
    pixi = shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
    manifest = os.path.join(ROOT, "pixi.toml")
    if os.path.exists(pixi) and os.path.exists(manifest):
        return [pixi, "run", "--manifest-path", manifest, "mojo"]
    raise BuildError("mojo not found; set MOJOPINT_MOJO=/path/to/mojo")


def build(force: bool = False) -> str:
    if os.environ.get("MOJOPINT_LIB") and os.path.exists(LIB) and not force:
        return LIB
    source = os.path.join(SRC, "kernels.mojo")
    if not os.path.exists(source):
        if os.path.exists(LIB):
            return LIB
        raise BuildError(f"no Mojo source at {source} and no library at {LIB}")
    if (
        not force
        and os.path.exists(LIB)
        and os.path.getmtime(LIB) >= os.path.getmtime(source)
    ):
        return LIB
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    cmd = mojo_command() + ["build", "--emit", "shared-lib", source, "-o", LIB]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0 or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_lib = None


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_lib, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _lib


def addr(array: np.ndarray) -> int:
    return array.ctypes.data


def convert_array(value, scale: float, shift: float) -> np.ndarray:
    original = np.asarray(value)
    if original.dtype.kind in "cO" or original.dtype.itemsize > 8:
        return original * scale + shift
    source = np.ascontiguousarray(original, dtype=np.float64)
    result = np.empty_like(source)
    if source.size == 0:
        return result
    _convert_f64(source, result, scale, shift)
    return result


def _convert_f64(
    source: np.ndarray,
    result: np.ndarray,
    scale: float,
    shift: float,
) -> None:
    if source.dtype != np.float64 or result.dtype != np.float64:
        raise TypeError("conversion buffers must have dtype float64")
    if source.shape != result.shape:
        raise ValueError("conversion buffers must have the same shape")
    if not source.flags.c_contiguous or not result.flags.c_contiguous:
        raise ValueError("conversion buffers must be C-contiguous")
    if not result.flags.writeable:
        raise ValueError("conversion output must be writable")
    if source.size == 0:
        return
    convert = lib().mp_convert_f64_serial
    source_addr = addr(source)
    result_addr = addr(result)
    if source.size < _PARALLEL_THRESHOLD:
        convert(source_addr, result_addr, source.size, scale, shift)
        return
    futures = []
    for worker in range(_PARALLEL_WORKERS):
        start = worker * source.size // _PARALLEL_WORKERS
        end = (worker + 1) * source.size // _PARALLEL_WORKERS
        futures.append(
            _executor.submit(
                convert,
                source_addr + start * 8,
                result_addr + start * 8,
                end - start,
                scale,
                shift,
            )
        )
    for future in futures:
        future.result()


def convert_array_inplace(value, scale: float, shift: float) -> np.ndarray:
    source = np.asarray(value)
    if (
        source.dtype != np.float64
        or not source.flags.c_contiguous
        or not source.flags.writeable
    ):
        return convert_array(source, scale, shift)
    _convert_f64(source, source, scale, shift)
    return source


def add_converted_arrays(lhs, rhs, scale: float, shift: float, sign=1.0):
    left_original = np.asarray(lhs)
    right_original = np.asarray(rhs)
    if (
        left_original.dtype.kind in "cO"
        or right_original.dtype.kind in "cO"
        or left_original.dtype.itemsize > 8
        or right_original.dtype.itemsize > 8
    ):
        return left_original + sign * (right_original * scale + shift)
    left = np.ascontiguousarray(left_original, dtype=np.float64)
    right = np.ascontiguousarray(right_original, dtype=np.float64)
    if left.shape != right.shape:
        return left + sign * (right * scale + shift)
    result = np.empty_like(left)
    if left.size == 0:
        return result
    add = lib().mp_add_converted_f64_serial
    left_addr = addr(left)
    right_addr = addr(right)
    result_addr = addr(result)
    if left.size < _PARALLEL_THRESHOLD:
        add(
            left_addr,
            right_addr,
            result_addr,
            left.size,
            scale,
            shift,
            sign,
        )
        return result
    futures = []
    for worker in range(_PARALLEL_WORKERS):
        start = worker * left.size // _PARALLEL_WORKERS
        end = (worker + 1) * left.size // _PARALLEL_WORKERS
        futures.append(
            _executor.submit(
                add,
                left_addr + start * 8,
                right_addr + start * 8,
                result_addr + start * 8,
                end - start,
                scale,
                shift,
                sign,
            )
        )
    for future in futures:
        future.result()
    return result


def combine_dimension_arrays(lhs, rhs, sign: float = 1.0) -> np.ndarray:
    a = np.ascontiguousarray(lhs, dtype=np.float64).reshape(-1, 7)
    b = np.ascontiguousarray(rhs, dtype=np.float64).reshape(-1, 7)
    if a.shape != b.shape:
        raise ValueError("dimension arrays must have the same shape")
    result = np.empty_like(a)
    if len(a) == 0:
        return result
    lib().mp_combine_dimensions(addr(a), addr(b), addr(result), len(a), sign)
    return result


def dimensions_equal_many(lhs, rhs, tolerance: float = 1e-12) -> np.ndarray:
    a = np.ascontiguousarray(lhs, dtype=np.float64).reshape(-1, 7)
    b = np.ascontiguousarray(rhs, dtype=np.float64).reshape(-1, 7)
    if a.shape != b.shape:
        raise ValueError("dimension arrays must have the same shape")
    result = np.empty(len(a), dtype=np.bool_)
    if len(a) == 0:
        return result
    lib().mp_dimensions_equal_many(
        addr(a), addr(b), addr(result), len(a), tolerance
    )
    return result


def main() -> int:
    print(build(force="--force" in sys.argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
