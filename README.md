# ozaki-blackwell

**EN** — Full-precision FP64 matrix multiplication on a *laptop* Blackwell GPU (RTX PRO 3000, sm_120), built from its
INT8 tensor cores (Ozaki scheme II: exact integer products + Chinese remainder theorem), written in CUDA C and inline PTX.
Goal: make a consumer GPU behave like a data-center FP64 part.

**SQ** — Shumëzim matricash me saktësi të plotë FP64 mbi një GPU *laptopi* Blackwell (RTX PRO 3000, sm_120), i ndërtuar nga
tensor core-t INT8 (skema Ozaki II: prodhime të sakta me numra të plotë + teorema kineze e mbetjeve), në CUDA C dhe PTX.
Qëllimi: një GPU e zakonshme të punojë si një kartelë FP64 e qendrave të të dhënave.

## Measured / Matur (15-10-2026, `bench/peak.cu`)

| Unit | Throughput |
|---|---|
| FP64 FMA, native | 0.449 TFLOPS |
| FP32 FMA, native | 28.2 TFLOPS |
| INT8 `mma.sync.m16n8k32` (PTX) | 232 TOPS |
| **INT8 / FP64** | **517×** |

## Run / Nisja
`./wsl.sh 'nvcc -O3 -arch=sm_120 -o out_peak bench/peak.cu && ./out_peak'` — builds and runs inside the laptop's WSL2
(Smart App Control refuses unsigned Windows binaries).

## FP64 from INT8 — results / rezultatet (15-10-2026, `bench/ozaki_test.cu`)

Error: max componentwise |C − C_ref| / (|A||B|), in units of 2⁻⁵³, against an 80-bit CPU reference. M = N = 256.
Gabimi: maksimumi për element kundrejt |A||B|, në njësi 2⁻⁵³, kundër një reference 80-bitëshe në CPU.

| K | magnitudes | native FP64 | 12 moduli | 14 moduli | **16 moduli** |
|---|---|---|---|---|---|
| 512 | 2^±0 | 4.71 | 2 660 | 10.9 | **0.99** |
| 512 | 2^±8 | 24.1 | 94 600 | 395 | **3.1** |
| 4096 | 2^±0 | 4.15 | 2 400 | 10.5 | **0.25** |
| 4096 | 2^±8 | 25.1 | 42 200 | 203 | **1.87** |

Speed, n = 4096 (FP64-equivalent TFLOPS; native FP64 peak measured at 0.449):
Shpejtësia, n = 4096 (TFLOPS FP64-ekuivalentë; piku vendas FP64 i matur 0,449):

| | 12 moduli | 14 moduli | **16 moduli** |
|---|---|---|---|
| TFLOPS | 7.55 | 6.46 | **5.56 — 12.4× the native peak, and more accurate** |

Time split at 16 moduli / ndarja e kohës: scales 1.0 ms · residues 3.0 ms · 16 INT8 GEMMs 16.8 ms · CRT 3.6 ms.
The INT8 GEMM runs at ~57 % of the measured tensor-core peak — the next gains are there.

## Energy / Energjia (n = 4096, 12 s each, `bench/energy.cu`, nvidia-smi every 200 ms)

| | GFLOPS | mean W | GFLOPS/W |
|---|---|---|---|
| native FP64, peak kernel | 423 | 56.8 | 7.4 |
| **Ozaki-II, 16 moduli** | **5 068** | 94.6 | **53.6 (7.2×)** |
| Ozaki-II, 12 moduli | 6 580 | 94.6 | 69.5 |

Under load the laptop holds ~95 W and the SM clock drops to ~1.8 GHz (2.76 GHz on the FP64 units alone).

## Next / Më tej
Quad-precision-class GEMM (~30 moduli, ~2^240 range) from the same INT8 tensor cores: a unit no GPU has in hardware.
GEMM me precizion të klasës katërfishe nga të njëjtat tensor core INT8: një njësi që asnjë GPU s'e ka në harduer.

## Double-double (~106-bit) GEMM from INT8 — a precision no GPU has in hardware (`bench/dd_test.cu`)

Error: max componentwise |C − C_ref| / (|A||B|), in units of **2⁻¹⁰⁶**, against a `__float128` CPU reference. M = N = 128.

| K | magnitudes | 24 moduli | 28 moduli | **32 moduli** |
|---|---|---|---|---|
| 256 | 2^±0 | 6.2e5 | 27.1 | **1.09** |
| 256 | 2^±8 | 2.0e7 | 1 160 | **46.9** |
| 1024 | 2^±0 | 4.7e5 | 31 | **0.49** |
| 1024 | 2^±8 | 1.1e7 | 713 | **12.4** |

Speed at n = 4096: **0.65 TFLOPS of double-double with 32 moduli — faster than this GPU's own plain FP64 (0.449), with twice
the significand bits.** Ndërtohet nga të njëjtat tensor core INT8; asnjë GPU s'ka double-double/FP128 në harduer.
The 32 INT8 GEMMs take ~35 of 211 ms; residues (fmod) and the O(32²) Garner step are the rest — the next target.
