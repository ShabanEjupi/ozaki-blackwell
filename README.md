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

### All on the GPU — Gjithçka te GPU-ja (22-10-2026)
n = 4096, κ = 10²⁰, nb = 256. Panel, L11⁻¹, solves as double-double kernels on the FP64 units; U12 = L11⁻¹A12 and the
trailing update as double-double GEMMs on the INT8 tensor cores. Only A, b in and x out cross PCIe.
Paneli, L11⁻¹ dhe zgjidhjet si kernelë double-double; U12 dhe përditësimi si GEMM double-double te tensor core-t INT8.

| solver | s | err vs x_true | err vs `__float128` |
|---|---|---|---|
| FP64 LU (CPU) | 2.95 | 193 | 193 |
| double-double LU, hybrid (panels on the CPU) | 7.41 | 9.5e-11 | 2.1e-11 |
| **double-double LU, all on GPU** | **0.68** | 1.1e-10 | 2.5e-11 |
| `__float128` LU, 24 CPU threads | 55.6 | 8.4e-11 | 0 |

**82× faster than `__float128` at the same accuracy** (was 10×). Phases (`OZ_PROF=1`): INT8 GEMMs 0.22 s, panel 0.28 s
(4 launches per column: the next target), L11⁻¹ + U12 0.04 s, blocked solve 0.03 s. `OZ_TRSM=1` / `OZ_SOLVE=1` select the
column-by-column versions (0.42 s / 0.13 s). A one-block pivot+swap+divide kernel was slower (0.44 s) and was dropped.

### Recursive panel, CRT on the integer units — 499× at n = 8192 (22-10-2026, evening)
Same problem (κ = 10²⁰), `OZ_SKIPCPU=1 ./out_solve n 1024 20 1`; GPU times after one warm-up solve (the GPU clocks idle
during the CPU reference). Error = max |x − x_ref| / max |x_ref|.

| n | double-double LU, all on GPU | `__float128` LU, 24 threads | speedup | err vs x_true (ours / quad) |
|---|---|---|---|---|
| 2048 | 0.113 s | 7.65 s | 67× | 1.6e-11 / 2.4e-11 |
| 4096 | 0.283 s | 55.6 s | ~197× | — |
| 6144 | 0.572 s | 200.8 s | **351×** | 3.4e-10 / 2.9e-10 |
| 8192 | 0.956 s | 476.9 s | **499×** | 2.1e-11 / 2.3e-11 |

At n = 8192 the double-double answer is *closer* to x_true than the `__float128` one, 500× sooner. The 0.956 s is from the
same run as the quad reference (after 8 minutes of all-core load); alone on a cool GPU the solve takes 0.855 s.
Në n = 8192 përgjigjja double-double është më afër x_true se ajo `__float128`, 500 herë më shpejt.

What changed / çfarë ndryshoi:
- **CRT by direct sum** (`crt_dd`): x = Σ rₗWₗ − qM over eight 32-bit limbs on the integer units, q from 40-bit fixed-point
  fractions, then five limbs → double-double. Garner + a double-double Horner was 41.5 of the 47 ms of a 4096² dd GEMM:
  31 dd steps of FP64, which runs at 1/64 rate here. **dd GEMM 2.8 → 0.68 ns per output element (4.1×)**; the cost is per
  output, nearly independent of K. Checked on the host (`bench/crt_check.cpp`, 10⁵ random 200-bit integers, 0 wrong) and
  against a long-double sum (`OZ_MMBENCH=2`).
- **Recursive panel**: 32-column leaves as one cooperative launch (a thread per row holding the row in registers; rows are
  renamed instead of swapped; one grid sync per column), halves joined by dd GEMMs on the tensor cores, and L11⁻¹ assembled
  from the halves' inverses. nb grows to 1024, so the trailing update gets 4× cheaper too.
- **One refinement step** in double-double on the GPU (`OZ_NOREF=1` turns it off): large blocks left the LU error
  wandering between 1e-11 and 1e-10; with the step it sits at the quad solver's level.
- Solves: left-looking (one dot product per row instead of a dd reduction per row per block), inverted 64×64 diagonal blocks.
- All 32 moduli in one `gemm_i8` launch; the dd GEMM reads blocks in place and subtracts into the matrix; residue planes
  sized to the free memory, wider C done in column chunks — n = 8192 fits in 12 GB.

Where the 0.855 s go at n = 8192 (`OZ_PROF=1`): panel 0.45 (leaves 0.22 — FP64-bound: ~10 GFLOPS per SM on this chip),
trailing GEMMs 0.15, solve + refinement 0.09, U12 0.04. A kernel launch costs ~7 µs under WSL2 (`bench/launch.cu`;
0.83 µs per node in a CUDA graph), a grid sync ~1 µs.

### Energy per solve / Energjia për zgjidhje (n = 4096, κ = 10²⁰, `bench/energy_solve.sh`)
CPU package power from the Windows RAPL counters (`\Energy Meter(RAPL_Package0_PKG)\Power`, 1 s), GPU power from
nvidia-smi (100 ms); 20 back-to-back GPU solves after a warm-up. DRAM power is not exposed on this laptop (reads 0).

| | time | CPU package | GPU | energy per solve |
|---|---|---|---|---|
| `__float128` LU, 24 threads | 55.0 s | 106.5 W | 8.4 W (idle) | **6.3 kJ** |
| double-double LU on the GPU | 0.238 s | 28.2 W (driving) | 68.3 W | **23 J** |

**~270× less energy per solve, CPU and GPU both counted** (GPU alone vs CPU package alone: 360×). Idle: CPU 15.4 W.

## Triple-double (~159-bit) GEMM from INT8 — precision as a software choice (22-10-2026, `bench/td_test.cu`)
`src/ozaki_td.cuh`: 49 moduli (the 32 above plus the 17 primes 103…29 that are not yet factors; product ~2³⁴²); each of the
three parts of an input split exactly into 53-bit int64 pieces for the residues; CRT as a direct sum over twelve 32-bit
limbs on the integer units; the result peeled off the integer as three doubles (round down, round down, round to nearest,
each exact from the integer), then made canonical. No floating-point arithmetic touches the 342-bit value.

Error: max_ij |C − C_exact| / (|A||B|)_ij in units of **2⁻¹⁵⁹**, against an exact 600-bit MPFR sum. M = N = 128.

| K | magnitudes | 41 moduli | 45 moduli | **49 moduli** |
|---|---|---|---|---|
| 256 | 2^±0 | 1.7e4 | 5.2 | **0.24** |
| 256 | 2^±8 | 1.2e6 | 343 | **10.7** |
| 1024 | 2^±0 | 1.8e4 | 4.6 | **0.13** |
| 1024 | 2^±8 | 4.3e5 | 110 | **2.1** |

Speed at n = 4096, every competitor on this laptop (CPU idle for the CPU runs; CPU numbers measured at n = 512 / 256 and
scaled by n³):

| ~159-bit GEMM, n = 4096 | time | TFLOPS | GPU speedup |
|---|---|---|---|
| **INT8 tensor cores, 49 moduli** | **0.114 s** | **1.21** | — |
| triple-double arithmetic on 24 CPU threads (QD-style, AVX, `omp simd`) | ~16 s | 0.0086 | **141×** |
| MPFR at 160 bits, `mpfr_fma`, 24 threads | ~230 s | 0.0006 | **~2 000×** |

Triple-double GEMM runs at 2.7× this GPU's own plain-FP64 peak (0.449 TFLOPS) while carrying three times the bits.

### A real job for triple-double: Ax = b at κ = 10³⁶ (22-10-2026, `bench/solve_td.cu`)
κ·u is 10²⁰ for FP64 and 10⁴ for double-double: neither has a correct digit. A = Q₁ diag(σ) Q₂ built with MPFR (σ down to
10⁻³⁶) and the product on the GPU as a triple-double GEMM; b = A x_true summed exactly. The GPU solver is solve.cu's
structure in three parts: recursive panel with 16-column cooperative leaves, td GEMMs in place, inverted diagonal
blocks, one refinement step. Competitors on the same laptop, CPU otherwise idle (the campaign stopped):

| n | **td LU, all on GPU** | td LU on 24 CPU threads (AVX, same arithmetic) | MPFR LU, 160 bits, 24 threads | FP64 LU |
|---|---|---|---|---|
| 1024 | **0.082 s**, 1.3e-14 | 0.10 s, 3.6e-14 (1.2×) | 1.32 s, 3.3e-14 (16×) | 128 |
| 2048 | **0.212 s**, 2.1e-14 | 1.04 s, 7.6e-14 (5×) | 10.2 s, 5.6e-14 (48×) | 103 |
| 4096 | **0.563 s**, 2.2e-14 | 18.0 s, 1.2e-13 (**32×**) | 81.9 s, 9.6e-14 (**145×**) | 86 |

(error = max |x − x_true| / max |x_true|; speedups of the GPU in parentheses.) The GPU answer is also the most accurate.
Honest note: at n ≤ 1024 a well-vectorized triple-double LU on 24 cores keeps up; the gap opens with n, as the GPU's
work moves into the tensor-core GEMMs. Phases at n = 4096: panels 0.38 s (leaves 0.14), U12 + trailing 0.08, solves 0.08.

## Beyond triple-double: arbitrary precision from INT8 (22-10-2026, `src/ozaki_mp.cuh`, `bench/mp_test.cu`)
The 49 moduli are every pairwise-coprime 8-bit set there is (all primes < 256 are used), so ~2³⁴² caps one Ozaki-II GEMM
at ~160-bit inputs. Beyond: balanced 160-bit slices, one 49-modulus GEMM per slice pair that reaches the result, exact
wide-integer accumulation, Q doubles peeled. Errors vs an exact 2400-bit MPFR sum (K = 256): 212 bits 0.13 ulp,
318 bits 0.25, 424 bits 0.08. Speed, CPU idle (campaign stopped), CPU numbers measured at n = 256/512 and scaled by n³:

| precision | GPU (INT8 tensor cores) | best CPU library, 24 threads | MPFR, 24 threads |
|---|---|---|---|
| 159 bits (td), n = 4096 | **87 ms, 1.58 TFLOPS** | triple-double AVX ~10.7 s (**123×**) | ~208 s (**2 390×**) |
| 212 bits (Q = 4), n = 2048 | **36 ms, 0.47 TFLOPS** | QD `qd_real` ~16.2 s (**445×**) | ~35 s (**960×**) |
| 318 bits (Q = 6), n = 2048 | **52 ms, 0.33 TFLOPS** | — | ~37 s (**713×**) |
| 424 bits (Q = 8), n = 2048 | **80 ms, 0.22 TFLOPS** | — | ~41 s (**514×**) |

Residues cut once into 16-bit chunks (32-bit multiply-adds per modulus): td residues 57 → 19 ms at n = 4096.
Honest limits: the INT8 GEMMs are power-bound on this laptop (~95 W); cost grows ~p² with precision, as MPFR's does at
these sizes, so the ratio to MPFR does not grow without bound — it is largest against software emulation of each width.

## Exact big-integer GEMM, thousands of bits / GEMM e saktë me numra të plotë mijëra-bitësh (23-10-2026, `src/ozaki_big.cuh`, `bench/big_test.cu`)

**EN** — The 49 moduli ≤ 256 cap one Ozaki-II GEMM at ~160-bit inputs, and slicing beyond that costs ~S² GEMMs. Here the
moduli are primes < 65 280 (up to 4 352 of them, M ≈ 2^66 464): each balanced residue is split into two bytes, so A·B mod p
is four INT8 GEMMs, and **the GEMM work grows linearly with the bits**. Both conversions are INT8 GEMMs too (residues =
digits × (2^8j mod p); CRT = t_l × the bytes of M/p_l, then q·M taken off per element), so everything runs on the tensor
cores. Every result is **bit-exact** (256 sampled elements per run against GMP). CPU = 24 threads, the same exact product.
FLINT's `fmpz_mat_mul` is itself multimodular with BLAS kernels: the strongest baseline we know of.

**SQ** — Modulet ≤ 256 e kufizojnë një GEMM Ozaki-II në ~160 bit. Këtu modulet janë prime < 65 280 (deri në 4 352), çdo mbetje
ndahet në dy bajte → 4 GEMM INT8 për prim: puna rritet *linearisht* me bit-et. Edhe konvertimet janë GEMM INT8. Çdo rezultat
del i saktë bit për bit kundrejt GMP-së.

| entry bits | n | primes | GPU | GMP mpz (24 thr) | FLINT fmpz_mat_mul (24 thr) | vs FLINT |
|---|---|---|---|---|---|---|
| 1 015 | 1024 | 192 | **71 ms** | 23.0 s (324×) | 1.31 s | **18×** |
| 4 087 | 512 | 576 | **62 ms** | 16.4 s (265×) | 1.07 s | **17×** |
| 16 375 | 256 | 2 112 | **90 ms** | 14.0 s (156×) | 1.69 s | **19×** |
| 32 759 | 256 | 4 352 | **316 ms** | 36.1 s (114×) | 4.56 s | **14×** |
| 65 527 | 128 | 5 760 (3 bytes) | **313 ms** | 10.6 s (34×) | 3.45 s | **11×** |
| 131 063 | 128 | 11 520 (3 bytes) | **983 ms** | 29.6 s (30×) | 9.76 s | **10×** |
| 4 087 | 1024 | 640 | **341 ms** | 125 s (368×) | 6.10 s | **18×** |
| 16 375 | 512 | 2 112 | **363 ms** (residues fused, below) | 104 s (287×) | 6.33 s | **17×** |

Doubling n at fixed bits does not raise the ratio (16 375 bits: 19× at n = 256, 15× at n = 512): even there the two
conversion GEMMs (~1.9·10¹³ INT8 ops at 16 375 bits, n = 512) dwarf the 8 704 product GEMMs (~2.3·10¹² ops). With the residue reduction fused into the GEMM epilogue (a prime's two weight rows are rows g and g + 8 of one m16 tile, so
one thread holds both sums and no int32 matrix goes to memory; `OZ_UNFUSED=1` turns it off): 16 375 bits, n = 512
403 → 363 ms; 1 015 bits, n = 1024 83 → 73 ms (the GPU is power-bound; runs vary by ~5 %). The conversions
are the whole remaining cost; a two-level residue system (subquadratic in P) is the next step.

Above ~60 000 bits of M the residues take three bytes and the primes run up to 2^23 (9 product GEMMs per prime, ~23 bits
each): 131 063-bit entries (~39 450 decimal digits) are still bit-exact. The ratio to FLINT falls slowly (19× → 10×) because
the two conversions are n²·P² GEMMs; at n = 128 they dominate. Mbi ~60 000 bit mbetjet marrin 3 bajte (prime < 2^23).

GMP times are a 96² (48² above 8 000 bits) corner of the outputs, scaled; FLINT runs the full product. At 1 015 bits the
phases are residues ~37 ms · 768 product GEMMs + t ~25 ms · CRT ~20 ms: the conversions (n²·P² work), not the n³ part,
dominate at these sizes — next: residues and CRT reads/writes fully coalesced, larger n.

Scale, honestly: one part in a googol (10^-100) is ~333 bits, already inside the 424-bit format above; a «googol-tuple»
number (10^100 doubles) cannot exist (more than the atoms in the universe). What does scale is cost: linear in bits here.

### Energy per exact product / Energjia për prodhim të saktë (`OZ_ENERGY=1 out_big`, RAPL package 1 s + nvidia-smi 100 ms)
Both chips counted in both columns (CPU package + GPU board), means over the busy segments of the power traces.
Të dy çipat numërohen te të dyja kolonat.

| entry bits, n | GPU path: time · GPU + CPU W · energy | FLINT, 24 threads: time · CPU + GPU W · energy | less energy |
|---|---|---|---|
| 1 015, 1024 | 80 ms · 88.1 + 19.2 W · **8.6 J** | 1.45 s · 81.3 + 9.2 W · 131 J | **15×** |
| 16 375, 256 | 66 ms · 95.3 + 23.3 W · **7.9 J** | 1.62 s · 81.4 + 6.5 W · 142 J | **18×** |

The GPU runs at its ~95 W cap; the host thread that feeds it costs ~20 W of CPU package.

## Exact integer determinant and exact Ax = b / Përcaktori dhe zgjidhja e saktë (`src/ozaki_det.cuh`, `src/ozaki_detbound.cuh`, `bench/det.cu`)

The big-integer GEMM above stops scaling near 131 000 bits because its two conversions are n²·P² GEMMs. Here the problem
is turned around so the conversions disappear: the input is int8, the output one residue per prime (or a vector of
p-adic digits), and the big-number work is a few scalar or vector steps on the host.

- **LU mod thousands of primes** (method 0): recursive LU (Toledo) without pivoting for a batch of primes < 65 280 at
  once; every O(n³) step is X W mod p = [Xlo Xhi][Wlo; Vlo] + 256 [Xlo Xhi][Whi; Vhi], V = 256 W mod p: two INT8 GEMMs of
  depth 2K, one reduction in the epilogue. 64 × 64 diagonal blocks are inverted in shared memory. A prime with a vanishing
  leading minor (~n/p) is dropped. Structured matrices (a zero leading minor over Z) get U·A·L with random unit triangular
  U, L, entries in [0, 127] (Kaltofen–Saunders), the same residues for every prime.
- **Divisor by lifting** (method 1): A⁻¹ mod p from the LU of [A I]; Dixon lifting on the GPU (per step one 16-bit A⁻¹
  matvec + one int8 residual matvec); rational reconstruction of 4 entries (GMP) gives d | det; det / d from a few primes.
- **A proved bound tight to one bit** (`ozaki_detbound.cuh`, n ≥ 4096): Hadamard leaves ~0.72 n bits of slack on a random
  matrix (5 900 at n = 8192 = hundreds of primes for det / d). FP32 LU with pivoting (cuSOLVER) → integer triangular
  X_L = round(2^s L⁻¹), X_U = round(2^s U⁻¹ diag(u)) (determinants = products of diagonals, exactly); B = X_L·PA·X_U
  **exactly** (the same modular INT8 GEMMs, int64 operands reduced while packing, Garner to 128 bits); Hadamard on B with
  upward rounding. Floating point decides only how tight it is, never whether it holds. Measured: 47 021 vs |det| 47 021
  bits (n = 4096), 98 140 vs 98 139 (8192), 204 491 vs 204 480 (16 384). It runs on its own thread and stream, overlapping
  the lifting. In spirit close to Ogita's verified determinants.
- **Exact rational solve** (method 2): the same lifting, all n entries to integers (Horner by halves, 24 threads), one
  common denominator.

Random int8 entries in [-127, 127]; FLINT 3.0.1 on 24 CPU threads; CPU idle during every FLINT run unless noted.

| n | \|det\| bits | GPU method 0 (all primes) | **GPU method 1** | FLINT `fmpz_mat_det` | vs FLINT |
|---|---|---|---|---|---|
| 1 024 | 10 736 | 0.25 s | **0.073 s** | 2.89 s | **40×** |
| 2 048 | 22 485 | 1.96–2.4 s | **0.293 s** | 24.9 s | **85×** |
| 4 096 | 47 021 | 21.3 s | **1.15 s** | 257 s | **224×** |
| 8 192 | 98 139 | — (primes < 65 280 run out) | **8.05 s** | 2 861 s ¹ | **355×** |
| 16 384 | 204 480 | — | **65.9 s** | not run (~6 h by n³) | |
| **32 768** | **425 348** | — | **594 s** ² | not run (~2 days by n³) | |

¹ while the GPU runs of this table shared the CPU now and then; the leading hex digits agree (`4d609fe9397eb725`).
² 26-10-2026, `OZ_CHECK=1 out_det 32768 32 0 1`: 12 GB of VRAM. A⁻¹ 15 s, lifting 56 145 steps 497 s (the read
  bandwidth again), the proved bound 238 s on its own thread hidden under the lifting, det / d 39 primes 71 s; checked modulo 4
  primes used nowhere else (all ok), the same leading digits `-98f922d047a31fc` in three runs. The bound is 244 bits above
  |det| here (FP32 LU at n = 32 768; `OZ_S=26` gives the same), so det / d takes 39 primes instead of 6.
All rows bit-exact against FLINT; n = 8192 and 16 384 also checked, and 32 768 checked modulo 4 primes used nowhere else (`OZ_CHECK=1`).
`fmpz_mat_det` is already FLINT's accelerated path (a Dixon divisor, then primes); its plain multimodular
`fmpz_mat_det_modular` takes 347 s at n = 2048 (177× method 0, the same algorithm).

| exact Ax = b, n | numerators (all) | GPU method 2 | FLINT `fmpz_mat_solve_dixon_den` | vs FLINT |
|---|---|---|---|---|
| 1 024 | 11.0 Mbit | **0.064 s** | 2.33 s | **36×** |
| 2 048 | 46.0 Mbit | **0.145 s** | 18.9 s | **130×** |
| 4 096 | 192.6 Mbit | **1.62 s** | 145 s | **90×** |
| 8 192 | 804 Mbit | **7.52 s** | not run | |

Energy per determinant, n = 2048 (`bench/energy_det_host.sh`: RAPL package from Windows 1 s, nvidia-smi 100 ms, both
chips in both columns): 20 GPU determinants in 11.05 s, process start-up included, at 53.7 W GPU + 19.5 W CPU =
**~40 J each** (~23 J counting only the 0.31 s of computation); FLINT 25.9 s at 72.1 W CPU + 6.5 W GPU = **~2 030 J**:
**~50× less energy** (~90× for the computation alone).

Where the time goes at n = 8192: lifting 6.9 s = 13 013 steps reading A⁻¹ (16 bit) and A (8 bit), at ~100 % of the
measured read bandwidth (~410 GB/s, `cudaMemcpy`/read kernel). That limit is physical for digit-by-digit lifting; the way
past it is lifting by matrix products (Storjohann's high-order lifting), the next step.

**SQ** — Problemi u kthye përmbys: asnjë konvertim numrash të mëdhenj. Ngritja p-adike te GPU-ja jep d | det, një kufi i
provuar (i ngushtë brenda 1 biti) e lë det/d me 6 prime. n = 4096: 1,15 s kundrejt 257 s të FLINT-it (224×); n = 8192:
8,05 s kundrejt 2 861 s (355×); n = 16 384: përcaktor 204 480 bit në 66 s. Zgjidhja e saktë racionale e Ax = b: deri 130×
FLINT. Energjia: ~50× më pak. Saktë bit për bit.

**Past the bandwidth wall — what the numbers say (read at the source: Pauderis & Storjohann, ISSAC 2012, DoublePlusOneLift).**
High-order lifting replaces the 13 013 GEMVs by ~log₂(m) levels of three n × n products of small residues
(‖R‖ < 0.6 n‖A‖ ≈ 2^22, X ≥ 3.61 n²‖A‖ ≈ 2^35), which are tensor-core GEMMs. For int8 inputs at n = 8192 that is
~13 × 3 × n³ × (4 primes × 4 byte products) ≈ 3·10¹⁴ INT8 MACs, ~3 s at 100 TOPS, against 6.4 s for Dixon at full bandwidth:
~2×, not an order of magnitude, because here m ≈ 1.6 n. It wins big when m ≫ n log n, i.e. for large entries. The other
levers are small: 31-bit lifting primes save ~14 % of the bytes per bit, tighter numerator bounds ~6 % of the steps.
For int8 matrices the digit-by-digit lifting at full bandwidth is close to what this machine allows; the next limits to
break were size (n = 32 768, done below) and large entries (where high-order lifting pays).

**n = 32 768 in 12 GB** (26-10-2026). Three things did not fit: the byte-packing buffers of one modular product
((2n + 128) · 2n bytes each, 13 GB for three), the proved bound (three n × n float matrices and two int64 ones, 30 GB), and
[A I] beside a 16-bit A⁻¹. Now: products deeper than `kc` = 2048 run in K-chunks (R = X₁W₁, R += X₂W₂, … — a third
epilogue mode), the bound builds X_L and X_U in 2048-column panels (trsm against identity panels) kept in host RAM,
T = X_L·PA mod q in row panels, B = T·X_U in column panels to the host as 16-bit residues, Garner and the row sums in row
panels, the L1 norms exact in 128 bits; the lifting keeps only the 4 entries per digit it reconstructs; A⁻¹ goes to 16 bits
through the host. Peak: ~11 GB of VRAM, 29 GB of host RAM. Every piece is checked bit-exact against FLINT at n = 1024–4096
with the chunking and the panels forced small (`OZ_KCAP=192`, `OZ_PANEL=1`); at n = 4096 the panelled bound gives the same
47 021 bits as the whole-matrix one.

## Large entries / Hyrje të mëdha (26-10-2026, `OZ_PLANES=L out_det n 32 1 {0,1}`)

Entries of L bytes, stored as L planes of signed bytes (entry = Σ_j 256^j A_j, random). The residues come from the planes
by Horner (`det_init`), so method 0 and the LU of [A I] are unchanged; the lifting reads all L planes per step
(`lift_r_planes`: L int8 matvecs, the residual kept as L + 4 base-256 digits per row, normalized and divided by p exactly
by one lane per row); only the entries whose D·x_j mod Q is not already small are rationally reconstructed. Hadamard
stands in for the proved bound (which reads int8 A). Every row bit-exact against FLINT `fmpz_mat_det`, 24 threads.

| n | entry bits | \|det\| bits | GPU method 0 | GPU method 1, one prime | **GPU method 1, S primes** | FLINT | vs FLINT |
|---|---|---|---|---|---|---|---|
| 256 | 64 | 16 765 | 0.049 s | 0.075 s | **0.048 s** (S = 4) | 0.17 s | 3.5× |
| 256 | 128 | 33 148 | 0.097 s | 0.149 s | **0.089 s** (8) | 0.40 s | 4.5× |
| 256 | 256 | 65 914 | 0.205 s | 0.377 s | **0.242 s** (16) | 1.08 s | 4.5× |
| 512 | 64 | 33 776 | 0.240 s | 0.166 s | **0.102 s** (4) | 1.17 s | **11×** |
| 512 | 128 | 66 550 | 0.511 s | 0.429 s | **0.314 s** (8) | 2.66 s | **8.5×** |
| 1 024 | 64 | 68 073 | 1.55 s | 0.364 s | **0.247 s** (4) | 12.39 s | **50×** |
| 1 024 | 128 | 133 615 | — (primes run out) | 0.894 s | **0.692 s** (8) | 30.98 s | **45×** |
| 2 048 | 32 | 71 638 | 7.57 s | 0.612 s | **0.563 s** (2) | 46.70 s | **83×** |
| 2 048 | 64 | 137 173 | — | 2.79 s | **2.03 s** (4) | 106.89 s | **53×** |

GPU times from runs without FLINT beside them, FLINT the fastest of its runs: on this laptop CPU and GPU share one
power budget, and a GPU run right after 24 FLINT threads is up to 2× slower (measured; the first version of this table
had those numbers). **S primes at once** (`liftm_x`, `liftm_r`, S = the largest of 2 3 4 6 8 12 16 not above L / 2, or
`OZ_LIFTS`): Dixon modulo M = p₀…p_{S−1}; x_i = A⁻¹r_i mod M by CRT from S 16-bit residues (Garner, then S base-2¹⁶
digits), A·x_i = Σ 256^(j+2u) A_j x_{i,u} lands on byte-aligned digits of the residual, which is divided by p₀, …, p_{S−1}
in turn (32-bit divisions after normalizing). A step gains 16S bits for 2S bytes of A⁻¹ residues + L bytes of A per
entry: the lifting falls 1.3–2.9× (n = 1024, 128-bit: 0.579 → 0.388 s). S is a template parameter: with run-time S the
per-digit arrays went to local memory and a step took 860 µs instead of 185.

What it says: FLINT is far stronger on large entries than on int8 (its multimodular part scales with the bits, its
Dixon divisor too), and our digit-by-digit lifting costs ∝ L² (L times the steps, each reading L planes): the one-prime
lifting fell from 40× to 17× (throttled) as the entries went from 64 to 128 bits at n = 1024; S primes at once keep
50× and 45×, but each step still reads all L planes of A, so the cost still grows faster than the bits. This is the regime where lifting by matrix products
(high-order lifting, Pauderis–Storjohann ISSAC 2012) changes the exponent — the next step, now with its baseline measured.

**SQ** — Hyrje 32–256 bit (L plane bajtesh), ngritje me S prime njëherësh: 50× FLINT te n = 1024 me 64 bit, 45× me
128 bit, 83× te n = 2048 me 32 bit, 53× me 64 bit (përcaktor 137 173 bit), bit për bit e saktë. Raporti bie me bitët (ngritja ∝ L²) —
pikërisht aty ku ngritja e rendit të lartë (HOL) e ndryshon eksponentin: hapi i radhës, tani me bazën e matur.

## High-order lifting on tensor cores / Ngritja e rendit të lartë (27-10-2026, `src/ozaki_hol.cuh`, `OZ_HOL=1 OZ_PLANES=L out_det n 32 1 1`)

Dixon gains one X-digit per step and reads A and A⁻¹ each time; for L-byte entries that is ~L² and memory-bound. High-order
lifting (Storjohann 2005, J. Complexity 21:609–650, §8 "sparse inverse expansion"; row form, y·A = b) squares the residue
instead, so the precision doubles per level: ~log₂ N levels of n³ products instead of N matrix-vector steps.

- **X = p₀…p_{s−1}**, distinct 16-bit primes with X ≥ 6.25 n²‖A‖; B₀ = A⁻¹ mod X from the LU of [A I] modulo each (the same
  batched LU as before, s primes at once). R₀ = (I − A B₀)/X; level i: R̄ = Rᵢ², Mᵢ = B₀R̄ mod X ∈ [0, X),
  R_{i+1} = (R̄ − A Mᵢ)/X exactly, and ‖Rᵢ‖ ≤ 1.25 n‖A‖ at every level. With C₀ = B₀, C_{i+1} = Cᵢ(I + X^m Rᵢ) + X^{2m} Mᵢ,
  m_{i+1} = 2mᵢ + 1: A⁻¹ = Cᵢ + X^{mᵢ}A⁻¹Rᵢ, so cᵢ = b Cᵢ ≡ b A⁻¹ (mod X^{mᵢ}).
- **The residue chain in RNS**, no big number anywhere: Rᵢ modulo the s primes of X and w more primes q (∏q > 2·2^zbits);
  Mᵢ from its s residues by Garner; R_{i+1} mod q = (R̄ − A Mᵢ)·X⁻¹ mod q, its exact value by Garner over the q (sign from the
  mixed-radix digits against (q−1)/2), checked against its bound. 2(s + w) modular GEMMs per level (64-bit entries at
  n = 2048: s = 6, w = 5). Every matrix is kept **transposed** (Z = Rᵀ, B₀ᵀ, Aᵀ, Mᵀ), so the vector chain reads K-contiguous
  rows; B₀ᵀ and Aᵀ are packed into GEMM operands once.
- **The vector chain as INT8 GEMMs**: cᵢ is stored as balanced digits in a *periodic* mixed radix (position e has radix
  p_{e mod s}), so X^m is a shift by s·m positions and the Garner digits of Mᵢ are digits of this radix already. cᵢ·Rᵢ is one
  GEMM: rows = digits of cᵢ as [lo | hi] bytes (K = 2n), columns = the byte planes of Z as [Z_v | Z_{v−1}], so column v
  collects 256^v; the planes are combined per (position, entry) in parallel into NL 32-bit limbs, and the carries run down
  the positions one thread per entry (floor division by p with a 64-bit reciprocal). Only cᵢ mod X^{N−m} matters, and cᵢ
  has ~s·mᵢ nonzero digits, so the rows sum to ~s·N over all levels.
- **Around it**: the proved bound now takes byte planes (float LU of A/256^{L−1} — only an approximation, B = X_L·PA·X_U is
  exact from the planes; Garner over up to 16 primes with |B_ij| rounded up from the mixed-radix digits, no 128-bit limit):
  at n = 2048 with 64-bit entries it gives 137 174 bits for |det| = 137 173 bits, so det/d takes 6 primes instead of 128.
  The rational reconstruction is FLINT's `_fmpq_reconstruct_fmpz_2` (half-gcd above a cutoff, FLINT 3.0.1 source): 0.493 →
  0.047 s at that size (`OZ_GMPRR=1` keeps the quadratic Euclid).

| n | entry bits | \|det\| bits | Dixon, S primes (lift) | **HOL** (lift: residue + vector chain) | FLINT | **vs FLINT** |
|---|---|---|---|---|---|---|
| 256 | 64 | 16 765 | 0.041 s (0.015) | **0.039 s** (0.013) | 0.17 s | 4.4× |
| 256 | 128 | 33 148 | 0.059 s (0.034) | **0.039 s** (0.018) | 0.40 s | 10× |
| 256 | 256 | 65 914 | 0.139 s (0.103) | **0.063 s** (0.033) | 1.08 s | 17× |
| 512 | 64 | 33 776 | 0.074 s (0.040) | **0.052 s** (0.022) | 1.17 s | 22× |
| 512 | 128 | 66 550 | 0.147 s (0.105) | **0.074 s** (0.036) | 2.66 s | 36× |
| 512 | 256 | 132 082 | 0.399 s (0.343) | **0.139 s** (0.079) | 8.82 s | **63×** |
| 1 024 | 64 | 68 073 | 0.177 s (0.120) | **0.126 s** (0.064) | 12.39 s | **98×** |
| 1 024 | 128 | 133 615 | 0.434 s (0.361) | **0.219 s** (0.132) | 30.98 s | **141×** |
| 2 048 | 32 | 71 638 | 0.358 s (0.279) | **0.288 s** (0.204) | 46.70 s | **162×** |
| 2 048 | 64 | 137 173 | 1.091 s (0.974) | **0.453 s** (0.329) | 106.89 s | **236×** |
| 2 048 | 128 | 268 249 | 2.946 s (2.774) | **0.856 s** (0.652) | 261.65 s | **306×** |
| 4 096 | 32 | 145 330 | 3.425 s (3.217) | **1.934 s** (1.666) | 416.46 s | **215×** |
| 4 096 | 64 | 276 398 | 7.443 s (7.050) | **3.273 s** (2.712) | 867.68 s | **265×** |

GPU: best of two runs, without FLINT beside it (`bench/hol_table.sh`, measured right after the FLINT runs, ~10 % slower than the
first runs of the night). Both GPU columns now use the proved bound for planes and FLINT's reconstruction, so the Dixon column is faster than the
table above. FLINT: `fmpz_mat_det`, 24 threads; the 512 × 256-bit, 2048 × 128-bit and 4096-row references are new
(27-10, `/opt/ozaki/flint/hol_flint.log`). At 4096 × 64-bit the residue chain is 2.08 s of the 2.71 s lifting
(247 modular GEMMs of n³, ~8 ms each: power-bound), the vector chain 0.44 s.

What changed: the ratio to FLINT now **rises** with the entry size (63× at 512 × 256-bit, 306× at 2048 × 128-bit) where
digit-by-digit lifting made it fall — the lifting is 2.6–4.3× faster than S-prime Dixon and grows like the bits, not their
square.

Checked: bit-exact against FLINT `fmpz_mat_det` in every row; a zero leading minor (preconditioned) and a singular matrix
(det = 0, FLINT 0) through the same path; the bounded-residue check never fired.

🚫 **Moduli below 256 for X do not work** (measured): the LU without pivoting fails modulo p with probability ~1 − e^{−n/p}
(n = 1024, p = 251: ~2 % success), and preconditioning cannot help (its randomness lives in F_p). Only the q (which need no
A⁻¹) could be 8-bit: ~20 % of the residue chain, not done.

**SQ** — Ngritja e rendit të lartë (Storjohann 2005) te bërthamat tensor: mbetja ngrihet në katror, saktësia dyfishohet për
nivel; zinxhiri i mbetjeve në RNS (2(s+w) GEMM modulare për nivel, matricat të transpozuara), zinxhiri i vektorit si GEMM INT8
mbi shifra me bazë të përzier periodike. Bashkë me kufirin e provuar për plane (1 bit mbi |det| te 2048 × 64 bit) dhe
rindërtimin half-gcd të FLINT-it: 236× FLINT te 2048 × 64 bit, 306× te 2048 × 128 bit, 265× te 4096 × 64 bit (867,7 s → 3,27 s),
bit për bit e saktë. Raporti tani RRITET me bitët. Modulet < 256 për X nuk bëjnë (LU pa pivotim mod p dështon ~1−e^(−n/p)).
