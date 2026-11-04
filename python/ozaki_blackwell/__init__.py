"""FP64, double-double and triple-double matrix products from the INT8 tensor cores of NVIDIA GPUs (sm_80 and newer).

The Ozaki scheme II (Ozaki, Uchino, Imamura) multiplies scaled integer residues exactly on INT8 tensor cores and rebuilds
the product by the Chinese remainder theorem. On consumer and laptop GPUs, whose FP64 units run at 1/64 of FP32, this
gives FP64 products 10-20x faster than the GPU's own FP64, and precisions no GPU has in hardware: double-double (~106
bits) and triple-double (~159 bits). Paper and code: https://git.spacecode.tech/shaban/ozaki-blackwell

    import numpy as np, ozaki_blackwell as ob
    C = ob.matmul(A, B)                         # FP64; NumPy arrays in and out (copied to the GPU and back)
    Ch, Cl = ob.matmul_dd((Ah, Al), (Bh, Bl))    # double-double: each value is hi + lo
    C0, C1, C2 = ob.matmul_td(A3, B3)           # triple-double: tuples of three arrays
CUDA arrays (PyTorch tensors, CuPy arrays, anything with __cuda_array_interface__) stay on the GPU; results are new
arrays of the same library, or `out=`. Arrays are float64 and row-major (C-contiguous); other inputs are converted.

License: Ozaki-Blackwell Restricted Source License 1.0 -- free for private study (90 days) and open academic research,
every other use needs a commercial license. See the LICENSE file in this package.
"""
import ctypes
import os

import numpy as np

__version__ = "0.1.0"
__all__ = ["matmul", "matmul_dd", "matmul_td", "device", "release", "bench", "OzakiError"]

_lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "libozaki_blackwell.so"))
_P = ctypes.c_void_p
_lib.ozb_error.restype = ctypes.c_char_p
_lib.ozb_error.argtypes = [ctypes.c_int]
_lib.ozb_version.restype = ctypes.c_char_p
_lib.ozb_device.argtypes = [ctypes.c_char_p, ctypes.c_int] + [ctypes.POINTER(ctypes.c_int)] * 3
_lib.ozb_dgemm.argtypes = [ctypes.c_int] * 3 + [_P] * 3 + [ctypes.c_int] * 2
_lib.ozb_ddgemm.argtypes = [ctypes.c_int] * 3 + [_P] * 6 + [ctypes.c_int]
_lib.ozb_tdgemm.argtypes = [ctypes.c_int] * 3 + [_P] * 9 + [ctypes.c_int]
_lib.ozb_bench.argtypes = [ctypes.c_int] * 3 + [ctypes.POINTER(ctypes.c_double)]
_lib.ozb_release.restype = None


class OzakiError(RuntimeError):
    pass


def _check(code):
    if code:
        raise OzakiError(_lib.ozb_error(code).decode())


def _is_cuda(x):
    return hasattr(x, "__cuda_array_interface__")


def _cuda_ptr(x, what):
    ai = x.__cuda_array_interface__
    if ai["typestr"] != "<f8" or len(ai["shape"]) != 2:
        raise TypeError(f"{what}: a 2-D float64 CUDA array is needed, got {ai['typestr']} with shape {ai['shape']}")
    if ai.get("strides") not in (None, (ai["shape"][1] * 8, 8)):
        raise TypeError(f"{what}: the CUDA array must be C-contiguous (call .contiguous() / cupy.ascontiguousarray)")
    return ai["data"][0], tuple(ai["shape"])


def _sync(x):
    """Wait for the stream that produced x (the products run on the legacy default stream)."""
    mod = type(x).__module__.split(".")[0]
    if mod == "torch":
        import torch
        torch.cuda.current_stream(x.device).synchronize()
    elif mod == "cupy":
        import cupy
        cupy.cuda.get_current_stream().synchronize()


def _empty_like_cuda(x, shape):
    mod = type(x).__module__.split(".")[0]
    if mod == "torch":
        return x.new_empty(shape)
    if mod == "cupy":
        import cupy
        return cupy.empty(shape, dtype=cupy.float64)
    raise TypeError("for this CUDA array type pass out= (C-contiguous float64 arrays of the result's shape)")


def _shapes(a_parts, b_parts):
    (m, k), (k2, n) = a_parts[0].shape, b_parts[0].shape
    if k != k2:
        raise ValueError(f"inner dimensions differ: {a_parts[0].shape} @ {b_parts[0].shape}")
    for x in a_parts:
        if x.shape != (m, k):
            raise ValueError("all parts of A must have one shape")
    for x in b_parts:
        if x.shape != (k, n):
            raise ValueError("all parts of B must have one shape")
    return m, n, k


def _run(fn, a_parts, b_parts, nout, extra, out):
    if any(_is_cuda(x) for x in a_parts + b_parts):
        if not all(_is_cuda(x) for x in a_parts + b_parts):
            raise TypeError("mix of CUDA and host arrays: move all of them to one side")
        ptrs, shapes = zip(*(_cuda_ptr(x, "input") for x in a_parts + b_parts))
        m, k = shapes[0]
        n = shapes[len(a_parts)][1]
        if any(s != (m, k) for s in shapes[:len(a_parts)]) or any(s != (k, n) for s in shapes[len(a_parts):]):
            raise ValueError(f"shapes do not match: {shapes}")
        outs = list(out) if out is not None else [_empty_like_cuda(a_parts[0], (m, n)) for _ in range(nout)]
        optr = [_cuda_ptr(o, "out")[0] for o in outs]
        for x in a_parts + b_parts + tuple(outs):
            _sync(x)
        _check(fn(m, n, k, *ptrs, *optr, *extra, 0))
        return outs
    a_parts = tuple(np.ascontiguousarray(x, dtype=np.float64) for x in a_parts)
    b_parts = tuple(np.ascontiguousarray(x, dtype=np.float64) for x in b_parts)
    if a_parts[0].ndim != 2 or b_parts[0].ndim != 2:
        raise ValueError("A and B must be 2-D")
    m, n, k = _shapes(a_parts, b_parts)
    outs = list(out) if out is not None else [np.empty((m, n), dtype=np.float64) for _ in range(nout)]
    for o in outs:
        if not (isinstance(o, np.ndarray) and o.dtype == np.float64 and o.shape == (m, n) and o.flags.c_contiguous):
            raise TypeError("out= must hold C-contiguous float64 arrays of the result's shape")
    p = [x.ctypes.data for x in a_parts + b_parts + tuple(outs)]
    _check(fn(m, n, k, *p, *extra, 1))
    return outs


def matmul(a, b, moduli=16, out=None):
    """C = A B in FP64 from INT8 tensor cores. 16 moduli: componentwise error below native FP64's for exponents spread
    up to 2^+-16 (the paper's Table 5); fewer moduli are faster and less accurate."""
    return _run(_lib.ozb_dgemm, (a,), (b,), 1, (int(moduli),), None if out is None else (out,))[0]


def matmul_dd(a, b, out=None):
    """(Ch, Cl) = (Ah + Al)(Bh + Bl) in double-double (~106-bit significands), 32 moduli. a and b are (hi, lo) pairs."""
    return tuple(_run(_lib.ozb_ddgemm, tuple(a), tuple(b), 2, (), out))


def matmul_td(a, b, out=None):
    """(C0, C1, C2) = (A0 + A1 + A2)(B0 + B1 + B2) in triple-double (~159-bit significands), 49 moduli."""
    return tuple(_run(_lib.ozb_tdgemm, tuple(a), tuple(b), 3, (), out))


def device():
    """Name, compute capability (e.g. 120), L2 size in KiB and SM count of the current CUDA device."""
    name = ctypes.create_string_buffer(256)
    sm, l2, sms = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
    _check(_lib.ozb_device(name, 256, ctypes.byref(sm), ctypes.byref(l2), ctypes.byref(sms)))
    return {"name": name.value.decode(), "sm": sm.value, "l2_kib": l2.value, "sms": sms.value}


def release():
    """Free the device memory of the cached plans (one per precision, for the last shape used)."""
    _lib.ozb_release()


def bench(n=4096, kind="fp64", reps=5):
    """Median GPU time (ms) and FP64-equivalent TFLOPS of an n x n x n product on the device, transfers excluded."""
    kinds = {"fp64": 0, "dd": 1, "td": 2}
    ms = ctypes.c_double()
    _check(_lib.ozb_bench(int(n), kinds[kind], int(reps), ctypes.byref(ms)))
    return ms.value, 2.0 * n ** 3 / (ms.value * 1e-3) / 1e12
