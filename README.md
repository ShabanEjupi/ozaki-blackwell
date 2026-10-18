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
The INT8 GEMM runs at 67 % of the measured tensor-core peak (157 TOPS, three-stage cp.async). With real data the laptop is
**power-bound** (~95 W, SM 1.8 GHz): a faster kernel did not move the end-to-end numbers, so the next gains are in energy
per INT8 operation (fewer moduli, fewer bit flips), not in latency hiding.

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

Speed at n = 4096: **1.76 TFLOPS of double-double with 32 moduli (2.34 with 24) — 3.9× this GPU's own plain FP64 (0.449), with twice
the significand bits.** Ndërtohet nga të njëjtat tensor core INT8; asnjë GPU s'ka double-double/FP128 në harduer.
Residues split into two int64 limbs instead of fmod on 2^107 values: 211 → 78 ms. The 32 INT8 GEMMs are ~35 of those 78.

## Related work / Puna e mëparshme (checked 15-10-2026)
- Ozaki, Uchino, Imamura — Ozaki Scheme II (arXiv 2504.08009): FP64 from INT8 via CRT, 7.4–9.8 TFLOPS on an RTX 4090. Public code: RIKEN GEMMul8.
- Ozaki-II SGEMM incl. power efficiency on RTX 5080 (arXiv 2508.03984); FP8 variant (2603.10634); FP4 limbs (2608.06812, 2609.24519);
  EmuGEMM fused kernels, 81 % of INT8 peak on Blackwell (2606.25453); complex CRT GEMM (2512.08321).
- **Not found:** a double-double / >FP64 GEMM built this way. Ozaki-II mentions such an extension as possible; no implementation or
  measurement turned up. That — and the power-bound laptop regime — is where this work can claim something new.
  FP64 alone is **not** new; it is the baseline.

## A real job: Ax = b at κ = 10²⁰ — Një punë reale: Ax = b me κ = 10²⁰ (18-10-2026, `bench/solve.cu`)

A = Q₁ diag(σ) Q₂ (orthogonal DST-I, the second permuted and sign-flipped; σ from 1 to 10⁻²⁰) — not fixable by scaling.
Blocked double-double LU: panels and U₁₂ on the CPU, trailing updates on the INT8 tensor cores. Error = max |x − x_ref| / max |x_ref|.
LU-ja double-double me blloqe: panelet te CPU-ja, përditësimet e mbetura (2/3 n³) te tensor core-t INT8.

| n = 4096, 24 CPU threads | seconds | err vs x_true | err vs `__float128` |
|---|---|---|---|
| FP64 LU | 2.69 | 193 | 193 |
| FP64 LU + 10 refinement steps, double-double residuals | | 1.9·10¹⁰ | 647 (best) |
| **double-double LU, INT8 tensor cores** | **5.03** (GPU 1.10) | **6.6·10⁻¹¹** | **3.3·10⁻¹¹** |
| `__float128` LU, all cores (reference) | 50.6 | 7.1·10⁻¹¹ | 0 |

- FP64 has no correct digit, and textbook refinement diverges (it needs κ·u < 1). Control at κ = 10¹², n = 512: refinement
  converges to 3·10⁻²⁰, so the baseline is implemented right. / Kontrolli me κ = 10¹²: rafinimi konvergjon.
- n = 2048: 0.34 s vs 5.98 s for `__float128` (17.6×), 2.2·10⁻¹⁰. At κ = 10²⁴ both reach ~4–7·10⁻⁷ (κ·u_dd).
- Error vs x_true is floored by rounding b to double-double (κ·2⁻¹⁰⁶), for the quad solve too.
- The CPU part (panels, U₁₂, A₂₂ −= P on the host) is now 3.9 of 5.0 s → next: keep A on the GPU, subtract there.
