# ozaki-blackwell

FP64, double-double (~106-bit) and triple-double (~159-bit) matrix products from the **INT8 tensor cores** of NVIDIA
GPUs, by the Ozaki scheme II: the inputs are scaled to integers, their residues modulo up to 49 coprime moduli are
multiplied exactly on the INT8 tensor cores, and the product is rebuilt by the Chinese remainder theorem.

Consumer and laptop GPUs run FP64 at 1/64 of their FP32 rate. On a laptop RTX PRO 3000 Blackwell (sm_120), measured:

| product | n = 4096 | against the GPU's own FP64 (0.449 TFLOPS peak) |
|---|---|---|
| FP64, 16 moduli | 6.2–6.8 TFLOPS | 14–15× |
| double-double, 32 moduli | 3.81 TFLOPS | 8.5×, with twice the bits |
| triple-double, 49 moduli | 2.03 TFLOPS | a precision no GPU has in hardware |

For plain FP64, use NVIDIA's own emulation in cuBLAS 13 or RIKEN's GEMMul8: on the same laptop both are faster and at
least as accurate. What this package adds is double-double and triple-double from the same INT8 units.

```
pip install ozaki-blackwell
python -m ozaki_blackwell          # self-test against exact rational sums, then the speed of your GPU
```

```python
import numpy as np, ozaki_blackwell as ob

C = ob.matmul(A, B)                             # FP64; NumPy in, NumPy out (copied to the GPU and back)
Ch, Cl = ob.matmul_dd((Ah, Al), (Bh, Bl))       # double-double: every value is hi + lo
C0, C1, C2 = ob.matmul_td((A0, A1, A2), (B0, B1, B2))
ob.bench(4096, "dd")                            # (ms, TFLOPS) on the device, transfers excluded
```

PyTorch tensors, CuPy arrays and anything else with `__cuda_array_interface__` stay on the GPU (float64, C-contiguous).

**Requirements.** Linux x86_64 (WSL2 included), an NVIDIA GPU with compute capability 8.0 or newer (Ampere, Ada,
Hopper, Blackwell), an NVIDIA driver for CUDA 12. Nothing to compile: the wheel carries the GPU code for sm_80, 86, 89,
90, 100 and 120, and PTX for newer GPUs. Inner dimension K < 131 072.

**Accuracy.** With 16 moduli the componentwise error is below native FP64's for exponents spread up to 2^±16 (not at
2^±32); double-double and triple-double stay within a few units of 2^-106 and 2^-159 on the paper's tests. The self-test
checks sampled outputs against exact rational sums.

**Paper and code.** Shaban Ejupi, *Precision as a software choice: FP64, double-double, triple-double and beyond from
the INT8 tensor cores of a laptop GPU* (2026), with every measurement script:
https://git.spacecode.tech/shaban/ozaki-blackwell (mirror: https://github.com/ShabanEjupi/ozaki-blackwell).

**License.** Ozaki-Blackwell Restricted Source License 1.0 (the LICENSE file in the package). This is not an
open-source license: free for private study (90 days) and for open academic research; every other use, including any use
by or for an organization, needs a commercial license.

## Shqip

Shumëzim matricash me saktësi FP64, double-double (~106 bit) dhe triple-double (~159 bit) mbi **tensor core-t INT8** të
kartelave NVIDIA (sm_80 e më të reja), me skemën Ozaki II. Te një laptop RTX PRO 3000 Blackwell: FP64 6,2–6,8 TFLOPS
(14–15× FP64-ja e vetë kartelës), double-double 3,81, triple-double 2,03 TFLOPS te n = 4096. Për FP64 të thjeshtë,
emulimi i cuBLAS 13 dhe GEMMul8 janë më të shpejtë; vlera e kësaj pakete është double-double dhe triple-double.
Instalimi: `pip install ozaki-blackwell`, prova: `python -m ozaki_blackwell`. Licenca: falas për studim privat (90
ditë) dhe kërkim akademik të hapur; çdo përdorim tjetër kërkon licencë tregtare.
