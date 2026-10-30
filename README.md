# ozaki-blackwell

**EN** — Full-precision FP64 matrix multiplication on a *laptop* Blackwell GPU (RTX PRO 3000, sm_120), built from its
INT8 tensor cores (Ozaki scheme II: exact integer products + Chinese remainder theorem), written in CUDA C and inline PTX.
Goal: make a consumer GPU behave like a data-center FP64 part.

**SQ** — Shumëzim matricash me saktësi të plotë FP64 mbi një GPU *laptopi* Blackwell (RTX PRO 3000, sm_120), i ndërtuar nga
tensor core-t INT8 (skema Ozaki II: prodhime të sakta me numra të plotë + teorema kineze e mbetjeve), në CUDA C dhe PTX.
Qëllimi: një GPU e zakonshme të punojë si një kartelë FP64 e qendrave të të dhënave.

> **License / Licenca** — source-available under the [Ozaki-Blackwell Restricted Source License 1.0](LICENSE), **not open
> source**: free only for private study (90 days) and open academic research; every other use -- any company or organization,
> any product or service, publishing trained adapters or models -- needs a paid license. The prohibited uses of section 4
> (interest, gambling, alcohol, images of living beings, music, weapons, surveillance...) bind every licensee.
> Kod me burim të dukshëm, **jo open source**: falas vetëm për studim privat (90 ditë) dhe kërkim akademik të hapur; çdo
> përdorim tjetër kërkon licencë me pagesë; përdorimet e ndaluara të nenit 4 vlejnë për këdo.

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

## Past the classical ceiling: Strassen over Z/p on the tensor cores / Përtej tavanit klasik (29-10-2026, `bench/strassen_p.cu`)

An Ozaki-II modulus multiplies residues mod p < 256. The sum of two residues reduced mod p is again a residue, so Strassen's
operand sums stay int8 at every level (over the integers each level adds a bit). Effective rate = 2n³ / time, the classical
operation count; every run is compared byte for byte with the classical product (64 entries of that checked on the CPU, plus
a mutant byte that must be counted). n = 32768, residues uniform in [−125, 125]:

| | ms | effective TOPS |
|---|---|---|
| classical, 128×128 tile (before today) | 553 | 127.1 |
| classical, 128×256 tile, memory clock 9001 MHz | 443–470 | 149.7–158.7 |
| Strassen L = 2, level by level | 456 | 154.1 |
| **Strassen L = 2, two levels in one pass, 128×256 tile, 9001 MHz** | **375** | **187.8** |

- **What the hardware gives at this power**: 46 SMs × 2048 INT8 ops/clock × 2195 MHz (measured under load, 100 W) =
  206.8 TOPS. The 232 TOPS of the spec need ~2.46 GHz, which the laptop does not hold. Classical reaches 72–77 % of the
  206.8; Strassen mod p reaches **91 %** of it, measured in classical operations.
- **Two levels in one pass**: the two levels are one bilinear algorithm with 49 products whose coefficients are the
  Kronecker products U⊗U, V⊗V, W⊗W. The 16 sub-blocks are read once, the 49 operands written once, the 49 products run as
  one batched launch, and C's 16 sub-blocks are written once and never read: ~8 h² bytes per node instead of ~39 h².
  n = 32768, L = 3: 125.9 → 168.5 TOPS. Table entries are template arguments; a runtime index had put the 16 sub-blocks in
  local memory.
- **The power cap is an energy budget (measured)**: a pure DRAM copy draws 84 W at 460 GB/s, the GEMM ~100–110 W. Putting
  the memory passes on their own streams beside the products gained nothing twice (L = 1–3 pipelined: ±1 %; L = 4 with the
  child passes overlapped: 433.7 vs 433.2 ms). Under a power cap time ≈ energy / P, so only fewer joules help. Lowering the
  memory clock to 9001 MHz moves watts to the SMs (2001 → 2126 MHz): classical +10 %, Strassen +6 %.
- **L = 4** (two passes of two levels): the 2401 products alone take 272–288 ms = **245 effective TOPS, above the 232 of the
  spec**, but the memory passes (~180 ms at ~84 W) take it back: 164 TOPS end to end. They are the next target (operand sums
  fused into the leaf GEMM's loads).
- Fewer toggling bits draw fewer watts: operands in [0, 15] instead of [−125, 125] gave +7 % (SM 1910 → 2062 MHz), not
  enough to pay for a smaller modulus (that would need +60 %).
- **Literature (checked 29-10-2026)**: no work found that combines Strassen with Ozaki-II / CRT emulation; the Ozaki-I
  line sets Strassen aside for floating-point stability. Over Z/m the question does not arise: Strassen is exact in any
  commutative ring, so every modulus of Ozaki-II (256, 255, 253, ... composite ones too) can use it and the CRT result is
  bit-identical to the classical one. It pays from n ≈ 16384 on this card, where 16 residue planes (8.6 GB) and the FP64
  matrices (6 GB) no longer fit 12 GB together: the moduli stream in pairs (next section).
- The power limit itself: `nvidia-smi -pl` is refused on this HP ZBook (the OEM firmware sets it; the vBIOS allows 120 W,
  Dynamic Boost gives ~100–113 W with the CPU idle), and NVML reports a V/F offset range of [0, 0] (`tools/vf-offset.ps1`).

**SQ** — Strassen mbi Z/p te bërthamat tensor: shumat e operandëve mbeten mbetje int8 në çdo nivel. Dy nivele në një kalim
(49 produkte me koeficientë Kronecker, një lëshim) + pllaka 128×256 + memorja 9001 MHz: **187,8 TOPS efektive te n = 32768
= 91 % e asaj që GPU-ja jep me këtë fuqi** (206,8 TOPS te 2195 MHz), ndërsa klasiku arrin 72–77 %. Kufiri i fuqisë është
buxhet energjie: kalimet e kujtesës (84 W) s'fshihen pas GEMM-it; te L = 4 vetë produktet japin 245 TOPS efektive (> 232),
por kalimet e marrin mbrapsht.

## Strassen inside Ozaki-II: FP64 DGEMM measured in joules / FP64 i matur në xhaul (29-10-2026, `bench/oz_strassen.cu`, `src/strassen_mod.cuh`)

Every Ozaki-II modular product now goes through two Strassen levels in one pass, for any modulus ≤ 256 (`src/strassen_mod.cuh`:
the reduction mod m is `v − m·rint(v/m)` in float, exact for |v| < 2²¹, so 256 and the composites work too). n = 16384,
C = A·Bᵀ, Gaussian entries; error = max over 1024 sampled entries of |C − C_ref| / (|A||Bᵀ|) in units of 2⁻⁵³, against a
double-double reference; energy = NVML power samples every 2 ms, integrated over ≥ 8 s of repeated DGEMMs; clocks set with
`nvidia-smi -lgc/-lmc` (`tools/vf_sweep.sh`, `tools/vf_mem.sh`, `tools/oz_campaign.sh`).

| | SM / memory MHz | ms | FP64 TFLOPS | W | **J per DGEMM** | GFLOPS/W | error |
|---|---|---|---|---|---|---|---|
| the GPU's own FP64 | default | 17 324 | 0.508 | 82.5 | 1 441.5 | 6.10 | 2.82 |
| the GPU's own FP64 at its best point | 1500 / 9001 | 32 173 | 0.273 | 31.9 | 1 025.7 | 8.58 | 2.82 |
| Ozaki-II, 15 moduli (max scale), classical | default | 1 009 | 8.72 | 119.4 | 119.4 | 73.7 | 0.98 |
| + scale from the row 2-norm: 14 moduli | default | 942 | 9.34 | 119.8 | 114.4 | 76.9 | 2.03 |
| + Strassen, two levels in one pass | default | 854 | 10.30 | 120.2 | 103.3 | 85.2 | 2.03 |
| **+ the operating point** | **1350 / 9001** | 1 376 | 6.39 | 53.4 | **73.6** | **119.4** | **2.03** |

- **19.6× fewer joules than the GPU's own FP64 at default clocks, 13.9× fewer than it at its own best point, 20.3× faster at
  default clocks, and more accurate** (2.03 against 2.82; with magnitudes spread over 2^±8: 6.89 against 17.14). 15 moduli
  at 1350/9001: 81.6 J, error 0.98. Two runs of the best point: 73.36 and 73.64 J.
- **Static and dynamic joules**: the GPU idles at ~17 W at every clock point. At 1350/9001 the DGEMM is 23.5 J static + 50.1 J
  dynamic; the GPU's own FP64 at 1500/9001 is 557 J static + 469 J dynamic. Dynamic alone: 9.4× fewer joules; static: 23.7×,
  because it is done 23× sooner.
- **Where the time goes** (`OZ_PROF=1`, 1350/9001): products 1051 ms, operand sums 120, residues 91, CRT 60, output sums 45,
  scales 28. A's residues for all 14 moduli come from one read of A into C's planes (the result of modulus l overwrites plane
  l), Bt's two moduli per read (residues 142 → 91 ms); the CRT ends in 128-bit integer Horner, one rounding instead of 13 in
  FP64 at 1/64 rate (75 → 60 ms).
- **Bit-identical**: the residue planes of C with Strassen equal the classical ones byte for byte (0 of 3 758 096 384 differ),
  so every FP64 entry is identical; a flipped byte (mutant) is seen. Over Z/m there is no stability question. The same at
  n = 8192 and 12288 with 14 and 16 moduli; 14 moduli stay below the native error at every size (1.45 vs 2.65, 1.34 vs 2.09,
  2.03 vs 2.82). Below n = 16384 Strassen saves joules, not time (n = 12288: 409 → 407 ms, 45.1 → 39.5 J).
- **Scale from the 2-norm** (`scales_norm`): |C′ᵢⱼ| ≤ ‖a′ᵢ‖·‖b′ⱼ‖ (Cauchy–Schwarz) < P/2 when every row norm, rounding
  included, is below 2^β, β = ⌊(log₂P − 1.001)/2⌋. K leaves the bound and a row spread over K entries keeps
  log₂(‖aᵢ‖/max|aᵢₖ|) more bits (~3 at K = 16384): 14 moduli reach the error 15 had with the max scale. Exact for any data.
- **The operating point**: under the power cap the clock is where the watts go; below it the energy per operation falls with
  the voltage (1500 MHz: 61 W, 1800: 78 W, default ~120 W). Below ~1350 MHz the fixed power of the board and memory
  (~30 W) takes over and the joules rise again (900 MHz: 94.9 J). Memory at 9001 instead of 14001 MHz saves 8–11 %.
- **Why Strassen gives 9 % of the time here, not the 20 % of the bare product**: at n = 16384 the 12 GB card holds A, Bᵀ, C (6.4 GB) and
  14 int8 planes of C (3.8 GB); the residues of the current pair of moduli and the Strassen buffers live inside C's FP64
  buffer until the CRT overwrites it, which forces operand chunks of 23 of 49 (the bare product runs unchunked). Strassen
  cuts the joules more than the time (14.2 % at 1350/9001: 85.8 → 73.6 J): fewer MACs, and the extra passes are cheap at the
  low memory clock.
- **For scale** (datasheet, rated peak over rated power, not measured): H100 SXM 67 TFLOPS FP64 Tensor Core at up to 700 W
  (96 GFLOPS/W), H100 NVL 60 TFLOPS at 350–400 W (150–171). A laptop GPU whose own FP64 gives 8.6 GFLOPS/W at best delivers
  FP64-accurate GEMM at 119.4 GFLOPS/W, inside the range of a datacenter FP64 tensor core.
- NVML's energy counter on this laptop advances in steps of ~1 s of energy: over 1–8 s windows it read 20–150 % high against
  the integrated samples (296.8 W "average" at 112.3 W sampled). The samples are the measurement; the counter is printed
  only as a cross-check.

**SQ** — Çdo produkt modular i Ozaki-II kalon tani nga dy nivele Strassen në një kalim, për çdo modul ≤ 256. Te n = 16384:
**73,6 J për një DGEMM FP64 (119,4 GFLOPS/W) — 19,6× më pak energji se FP64 i vetë GPU-së, 13,9× më pak se ai te pika e vet më e
mirë, 20,3× më shpejt dhe më i saktë** (2,03 kundër 2,82). Shtresat: INT8 në vend të FP64 → shkallëzimi me normën-2 (14 module në
vend të 15, kufi i garantuar me Cauchy–Schwarz) → Strassen mbi Z/m (bit për bit si klasiku: 0 bajtë të ndryshëm nga 3,76
miliardë) → mbetjet e A-së nga një lexim, CRT me Horner 128-bit → pika e punës 1350 MHz / memorja 9001 MHz. Pjesa dinamike e
xhauleve: 9,4× më pak; ajo statike (~17 W bordi): 23,7× më pak, sepse mbaron 23× më shpejt. Brenda brezit të një H100 (96–171 GFLOPS/W sipas datasheet-it), në laptop.

## Next / Më tej (27-10-2026)
- Kimi K3 end to end on this laptop needs 1.56 TB of storage (669 GB free; the office link gave 2.9 MB/s): an external NVMe
  and the K3 attention (KDA linear attention + MLA) on the exact products; then the whole model streams at ~27 s per token.
- Real systems above n ≈ 6000 run by Dixon (bench/real_dixon.sh); HOL by column blocks would be 2.6–4.3× faster there.
- cry10000 (n = 10 000) solved exactly in 122 s (D 574 553 bits); the SPEX column for the larger systems fills in the night
  runs (`/opt/ozaki/real/batch.log`; `bench/real_table.py [--tex]`).
- ✅ 27-10: Nemotron 3 Ultra (560 B, open weights; its main pre-training data gated, internal training only) runs whole and
  exact from disk (`llm/ultra.cu`, below), 19.6 s per token with read-ahead. Next: the NVFP4 prefill tiled
  through shared memory, and the fully open models — OLMo 2 32B traced into its whole training data, K2-Think-V2 (`llm/dense.cu`).
- The decode kernel reads 94 GB/s of packed weights, below what the GPU memory gives (irrelevant while the weights stream from disk).
- An 8-bit X for HOL was measured and dropped (below); the proved bound with row scaling and n = 32768 remain as before.

**SQ** — Radha: K3 i plotë kërkon 1,56 TB disk (SSD i jashtëm) dhe vëmendjen e K3; n > 6000 me Dixon (HOL në blloqe do
ishte 2,6–4,3× më shpejt); kolona SPEX plotësohet natën.

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
(n = 1024, p = 251: ~2 % success), and preconditioning cannot help (its randomness lives in F_p). Everything else of the residue chain can be
8-bit, and now is (next section).

**SQ** — Ngritja e rendit të lartë (Storjohann 2005) te bërthamat tensor: mbetja ngrihet në katror, saktësia dyfishohet për
nivel; zinxhiri i mbetjeve në RNS (2(s+w) GEMM modulare për nivel, matricat të transpozuara), zinxhiri i vektorit si GEMM INT8
mbi shifra me bazë të përzier periodike. Bashkë me kufirin e provuar për plane (1 bit mbi |det| te 2048 × 64 bit) dhe
rindërtimin half-gcd të FLINT-it: 236× FLINT te 2048 × 64 bit, 306× te 2048 × 128 bit, 265× te 4096 × 64 bit (867,7 s → 3,27 s),
bit për bit e saktë. Raporti tani RRITET me bitët. Modulet < 256 për X nuk bëjnë (LU pa pivotim mod p dështon ~1−e^(−n/p)).

### The residue chain over 8-bit moduli (27-10-2026, `hol8_setup`, `bench/rchain_bench.cu`, `bench/hol8_check.sh`)

Measured first, on random data (the power budget depends on the bits that toggle): a 16-bit modular product (two INT8 GEMMs of
depth 2n, two passes) runs at 95 TOPS at n = 4096, a product modulo a prime < 256 (one INT8 GEMM of depth n, balanced residues
as int8, reduced in the epilogue) at 134 TOPS — **2.8× less time per bit of modulus**. So only M = B₀·R̄ mod X keeps 16-bit
products (B₀ = A⁻¹ mod p needs the LU, see above); the rest moved to 8-bit moduli (odd primes 17…251 and 243, 169, 125, 121, 49):

- **Z² exactly**, in an RNS of W moduli with ∏ > 2^{2·zbits + log₂n + 10} (W = 23 at 64-bit entries, 42 at 128-bit), one
  INT8 GEMM each; then **base extension** to the 16-bit p_t by approximate CRT (x = Σ ξ_u R/r_u − αR, α = round(Σ ξ_u/r_u)
  in float, exact while |x| ≤ 2⁻⁹R), written straight into modgemm's packed [lo | hi] operand.
- **A·M only modulo the first WQ moduli** (∏ > 2^{zbits+2}: WQ = 10 at 64-bit), and **Z from them by approximate CRT**
  (no Garner): its residues modulo all W moduli and its balanced byte planes accumulated signed from the bytes of Q/q_r.
- Barrett reduction (`redm`, 4 instructions) instead of `%` in every Garner, and the kernels templated on s so the arrays stay
  in registers (they spilled 240 B before).

| n × entry bits | 16-bit chain (residue chain) | **8-bit chain** (residue chain) | FLINT | **vs FLINT** |
|---|---|---|---|---|
| 1 024 × 64 | 0.129 s (0.032) | **0.110 s** (0.017) | 12.39 s | 113× |
| 1 024 × 128 | 0.201 s (0.056) | **0.183 s** (0.041) | 30.98 s | 169× |
| 2 048 × 64 | 0.468 s (0.211) | **0.362 s** (0.124) | 106.89 s | **295×** |
| 2 048 × 128 | 0.838 s (0.365) | **0.725 s** (0.259) | 261.65 s | **361×** |
| 4 096 × 64 | 3.221 s (2.050) | **2.284 s** (1.141) | 867.68 s | **380×** |

Same heads as the 16-bit chain (`OZ_HOL16=1`) at 256…4096 × 64/128 bits, also with every new buffer poisoned (`OZ_POISON=1`);
n % 128 ≠ 0 or > ~360 bits of Z² fall back to the 16-bit chain by themselves. At 4096 × 64 the residue chain is now
Z² 0.27 s · M (16-bit) 0.39 · A·M 0.15 · Z 0.09 · Garner of M 0.07 · base extension 0.04 · transposes 0.02 (`OZ_PROF=1`):
M is the largest part left, and it needs A⁻¹ modulo small primes, i.e. an LU with pivoting.

**SQ** — Zinxhiri i mbetjeve tani me module 8-bit: Z² saktë në RNS me 23–42 module < 256 (një GEMM INT8 secili, 2,8× më lirë
për bit se moduli 16-bit), zgjerim baze me CRT të përafërt, A·M vetëm mod 10 moduleve të para. 4096 × 64 bit: 3,22 → 2,28 s
(380× FLINT), bit për bit i njëjtë me zinxhirin 16-bit.

## Real systems, exact / Sisteme reale, saktë (27-10-2026, `bench/mm2oz.py`, `OZ_FILE`, `bench/spex_solve.c`, `bench/real_run.sh`)

The question an engineer asks: **how wrong is FP64 on the hardest real matrices, and can the exact answer be had in seconds?**
The 13 most ill-conditioned square real matrices with 900 ≤ n ≤ 8192 of NIST Matrix Market (the first source: the original
Harwell-Boeing, SPARSKIT/FIDAP and NEP collections with NIST's condition estimates; SuiteSparse refuses scripted downloads),
from fluid dynamics (FIDAP, LNS), magnetohydrodynamics (MHD), a power network (Quebec Hydro), crystal growth, oil reservoirs
and chemical plants.

- **No rounding anywhere**: every double is m·2^e exactly, row i is multiplied by 2^{k_i} (its largest denominator), so
  (DA)x = Db has integer entries — up to 154 bits — and the same x. Rows are put in the order of FP64 partial pivoting (the GPU
  LU does not pivot; the order changes nothing in x), n is padded to a multiple of 128 with I. b = D⁻¹c, c random in [−127, 127].
- **The whole solution**: high-order lifting (above) for y·(PDA)ᵀ = c, all n entries converted by halves (X^{2^l}), one
  common denominator found in a few rounds of half-gcd reconstruction, then **checked**: y·M = c modulo two primes used nowhere
  else. The reconstruction is unique by the Hadamard/Cramer bounds, so the solution is proved, not just tested.
- **Against SPEX Left LU** (SuiteSparse 7.6, SPEX 2.3: exact sparse REF LU with COLAMD, GMP, one CPU thread — the dedicated
  tool for exact sparse solving) on the very same integers; both print a digest Σ x_j r^j mod 2⁶¹−1 of the solution.
- **FP64**: LAPACK `dgesv` (dense, partial pivoting) and SuperLU (scipy `spsolve`) on the original doubles, against the exact x.

| matrix | n | nnz | kappa (NIST est.) | entry bits | denominator bits | GPU s | SPEX s | | same x | y M = c mod 2 primes | FP64 LAPACK digits (entries with none) | FP64 SuperLU digits (entries with none) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:-:|:-:|---:|---:|
| west0989 | 989 | 3518 | 5.7e12 | 75 | 28383 | 0.17 | 1.5 | 9x | yes | ok | 11.7 (0) | 11.8 (0) |
| qh1484 | 1484 | 6110 | 5.6e17 | 96 | 82985 | 0.55 | 12.4 | 23x | yes | ok | 11.2 (0) | 12.2 (0) |
| fidap032 | 1159 | 11047 | 5.2e20 | 60 | 58651 | 0.28 | 54.5 | 193x | yes | ok | 0.0 (568) | 0.0 (572) |
| fidap006 | 1651 | 48968 | 3.5e21 | 154 | 137635 | 1.08 | 963.7 | 895x | yes | ok | 0.0 (887) | 0.0 (693) |
| cry2500 | 2500 | 12349 | 4.4e17 | 70 | 142772 | 1.41 | 699.3 | 497x | yes | ok | 6.4 (0) | 5.1 (0) |
| fidapm07 | 2065 | 45184 | 6.8e19 | 118 | 161950 | 1.19 | > 7200 |  | -- | ok | 0.0 (430) | 0.0 (290) |
| fidap026 | 2163 | 74195 | 4.7e18 | 153 | 194835 | 2.12 | > 7200 |  | -- | ok | 0.0 (1568) | 0.0 (1530) |
| mhd3200a | 3200 | 68026 | 6.5e39 | 130 | 186119 | 4.02 | 1290.8 | 321x | yes | ok | 0.0 (2876) | 0.0 (2768) |
| fidap014 | 3251 | 65747 | 2.6e16 | 124 | 256431 | 4.74 | > 7200 |  | -- | ok | 7.6 (0) | 8.0 (0) |
| lns_3937 | 3937 | 25407 | 1.0e17 | 79 | 195969 | 4.31 | -- |  | -- | ok | 10.1 (2) | 7.4 (1) |
| fidap012 | 3973 | 79017 | 2.2e20 | 119 | 271165 | 7.44 | -- |  | -- | ok | 0.0 (483) | 0.0 (482) |
| mhd4800a | 4800 | 102252 | 3.2e47 | 131 | 281770 | 34.39 | -- |  | -- | ok | 0.0 (3923) | 0.0 (4754) |
| sherman3 | 5005 | 20033 | 6.9e16 | 102 | 170131 | 14.25 | -- |  | -- | ok | 11.8 (0) | 12.4 (0) |
| fidapm15 | 9287 | 96099 | 1.36e20 | 63 | 510845 | 78.66 | -- |  | -- | ok | 0.0 (2251) | 0.0 (5260) |
| cry10000 | 10000 | 49699 | 6.9e20 | 76 | 574553 | 115.02 | -- |  | -- | ok | 4.2 (66) | 1.6 (923) |

GPU s: A⁻¹ + lifting + reconstruction + all numerators (the determinant's det/d step apart; where it needs more than the
~78 000 bits the 16-bit primes give, it is reported as not computed — the solution does not depend on it). SPEX ran in the
night after the GPU pass (CPU alone), limit 2 h each.

What the table says: on **7 of 13 real systems FP64 has not one correct digit** — LAPACK and SuperLU alike, with the right
order of magnitude only by accident (mhd3200a / mhd4800a: normwise error 10⁸, 82–99 % of the entries without a correct digit) —
while NIST's condition estimate (itself FP64) does not tell which ones (qh1484 at 5.6·10¹⁷ keeps 11 digits). The exact
solution, certified, takes 0.2–34 s on the laptop; SPEX gives the same x bit for bit where it finishes, 23× to ~900× slower.

**SQ** — 13 matricat reale më të këqija të NIST Matrix Market (n 989–5005, κ deri 3·10⁴⁷): zgjidhja e SAKTË, e certifikuar,
për 0,2–34 s te laptopi; SPEX (vegla e dedikuar për zgjidhje të sakta të rralla) jep të njëjtën x bit për bit, 23–900× më ngadalë.
Te 7 nga 13 sisteme FP64 (LAPACK dhe SuperLU) s'ka asnjë shifër të saktë.

## Exact arithmetic for the largest open model / Aritmetikë e saktë për modelin më të madh të hapur (27-10-2026, `llm/`)

**Kimi K3** (moonshotai/Kimi-K3 on Hugging Face, read from the model card and the safetensors headers: **2 779 931 837 184
parameters**, 96 shards, 1.56 TB) is the largest open-weights model; after it come Qwen3.8-2.4T-A95B and DeepSeek-V4-Pro
(1.65 T). Its routed experts (2.72 T of the parameters: 93 layers × 896 experts, 16 used per token, latent width 3584) are stored
MXFP4: an E2M1 value per weight and one E8M0 scale per 32. The rest (57 G) is BF16.

An MXFP4 group of 32 is exactly the depth of one `mma.m16n8k32` INT8 step, and 2·E2M1 is an integer ≤ 12. So the product of
MXFP4 weights and BF16 activations can be computed **exactly** with INT8 tensor cores and rounded **once** — every output is
the correctly rounded value of Σ_k w_ik x_kj (`llm/mxexact.cu`):

- weights: q' = 2·E2M1 · 2^{(s − s_min) mod 4} is an int8 (≤ 96); a row keeps one scale per band of four binades (on K3 every
  row of the 48 matrices tested spans ≤ 3 binades: one band);
- activations: each token is aligned to its largest exponent, A = m·2^{D_c − (E − e)} with D_c = 7S − 8, cut into S int8
  digits; elements more than 2^{D_c} below the top form the next tier with their own exponent;
- every (band, tier, digit) is a plain INT8 GEMM whose int32 sums cannot overflow (|P| ≤ K·96·128 < 2^{26.4} for K ≤ 7168);
  the few terms of an output are added in a 128-bit integer and rounded once to FP32 (round-to-nearest-even, subnormals
  included). A correctly rounded result cannot depend on the order of summation, the tiling, the batch or the machine.

The baseline is what inference engines do: dequantize to BF16 (exact) and multiply on BF16 tensor cores with FP32
accumulation, the split of K chosen by shape (`bf16_gemm` + `reduce_k`); for decode, an FP4 GEMV with FP32 multiply-add.

Real weights: layer 5, experts 0–15, w1/w2/w3 (48 matrices, 281 MB fetched with HTTP range requests by `llm/k3_fetch.py`;
the rest of the 16 GB shard never travels). Activations: Student-t (3 degrees of freedom: heavy tails like real hidden states)
times the layer's real `routed_expert_norm` weight.

| | result |
|---|---|
| outputs certified correctly rounded (48 matrices, 512 tokens; expert 0 w1 also 4096 tokens) | **all** (0 uncertified of 79.7 M) |
| independent check: Python integers/`Fraction` from the raw bytes (`llm/mx_check.py`), 1500 random outputs | 1500 / 1500 |
| same outputs for batch 1, 3, 17, 64, 200 and the 4096 batch | bit-identical, every matrix |
| decode kernel (packed 4-bit read once, 1–8 tokens) vs the prefill path | bit-identical, every matrix |
| BF16 tensor cores + FP32 accumulation, split-K 1 / 2 / 4 / 8: outputs not correctly rounded | 81.5 / 67.1 / 55.6 / 51.2 % (expert 0 w1); 75–83 % for split-K 1 over all 48 |
| BF16 split-K 1 vs 4: outputs whose bits differ (4096 tokens × 3072) | 8 963 796 (71 %) |

Speed (expert 0 w1, 3072 × 3584):

| | exact (this work) | baseline |
|---|---:|---:|
| 4096 tokens (prefill), 4 digits | 8.0 ms: split 1.3 + 8 INT8 GEMMs 4.5 (153–162 TOPS each) + rounding 1.9 | BF16 tensor cores 2.9 ms (our kernel, 31 TFLOPS) |
| 1 token (decode) | 0.062 ms = 94 GB/s of packed weights | FP4 GEMV, FP32: 0.125 ms |
| version 1 (per-group 128-bit accumulator, commit 9b77801), 4096 tokens | 34.7 ms | |

**Where exactness is free.** Running K3 on this laptop is a streaming problem: per token the 16 routed experts of 92 layers
(25.8 GB of MXFP4) and the dense BF16 part (111.2 GB: attention 72.4, shared experts 24.3, gates, latent projections and norms 12.1, lm_head 2.4 -- summed from the 96 safetensors headers) must come from storage: 137.0 GB per text token. Measured here: NVMe 5.1 GB/s (`dd`
O_DIRECT), pinned RAM → GPU 45 GB/s (PCIe 5 ×16, `llm/h2d.cu`). The exact decode kernel consumes 94 GB/s — 18× what the disk
delivers — so on a laptop **a bit-reproducible trillion-parameter model costs the same time as a non-reproducible one**:
~27 s per token, all of it reading. What it needs and this laptop lacks: 1.56 TB of storage (669 GB free) — the office link
gave 2.9 MB/s from Hugging Face, i.e. six days for the weights.

**SQ** — Kimi K3 (2,78 T parametra, modeli më i madh me pesha të hapura): prodhimi MXFP4 × BF16 i ekspertëve, i
rrumbullakosur SAKTË, mbi tensor core INT8. Mbi peshat e vërteta (48 matrica): çdo dalje e certifikuar, e njëjtë për çdo
batch dhe te decode; rruga e zakonshme BF16+FP32 gabon te 51–83 % e daljeve dhe ndryshon bitët sipas split-K. Prefill 8 ms
(2,8× BF16), decode 94 GB/s — 18× më shpejt se disku, pra për një model që rrjedh nga disku saktësia s'kushton asgjë.

### The router: where rounding picks the expert (`llm/router.cu`, `llm/exact.cuh`)

The gate of a K3 MoE layer is logits = x·W_gᵀ in FP32 (W_g: 896 × 7168 BF16, real, layer 5), sigmoid + the learned
`e_score_correction_bias`, top 16 of 896. `llm/exact.cuh` generalizes the exact product to any BF16 × BF16 (both sides split
in tiers of 4 int8 digits: 64 INT8 GEMMs for the gate); inputs are Student-t hidden states through RMSNorm with the layer's
real `post_attention_layernorm` weight. The routing step is the same CPU code for every set of logits.

| | 4096 tokens (seed 1) | 16 384 (seed 2) | 16 384 (seed 3) |
|---|---:|---:|---:|
| exact logits: not certified | 0 | 0 | 0 |
| ordinary logits (BF16 TC, FP32 acc.) not correctly rounded | 98.6–99.8 % | 98.6–99.8 % | 98.6–99.8 % |
| tokens sent to a different expert set than with exact logits, split-K 1 / 2 / 4 / 8 | 1 / 1 / 2 / 1 | 2 / 2 / 0 / 0 | 0 / 0 / 2 / 0 |
| median / 1st percentile gap between the 16th and 17th score | 1.1e-3 / 1.7e-5 | 1.1e-3 / 1.5e-5 | 1.1e-3 / 1.7e-5 |

11 swaps in 147 456 (token, split) pairs: **~1 token in 13 000 per layer** changes an expert when only the rounding of its
logits changes; over the 92 MoE layers of K3, ~1 token in 140 runs at least one different expert somewhere. With exact
logits the choice is a function of the input alone.

**SQ** — Te ruteri real i K3, vetëm rrumbullakimi i logit-ëve (split-K) e dërgon ~1 token në 13 000 te një ekspert tjetër për
shtresë, ~1 në 140 tokenë në 92 shtresa. Me logit-ë të saktë zgjedhja varet vetëm nga hyrja.

### Measured and dropped: an 8-bit X for the HOL residue chain (27-10-2026, `bench/lu8_test.cu`)

M = B₀·Zb mod X is the last 16-bit part of the 8-bit residue chain; with X a product of primes < 256 it would be one INT8
GEMM of depth n per prime instead of modgemm's two of depth 2n. B₀ = A⁻¹ mod p then needs LU **with** pivoting (a leading
minor vanishes with probability ~1/p): `panel_lu_piv` factors the whole tall leaf panel with row swaps over the whole row
(the L to the left, the columns still waiting for their updates -- every update is row-wise, so a swap commutes with it --
and the identity of [A I]). Checked: (A mod p)·A⁻¹ = I on every tested row, no prime singular; n = 4096, 12 primes < 256:
0.357 s (16-bit primes without pivoting: 0.119 s, one of six singular; with pivoting 0.236 s).

End to end it does not pay: same determinant head, but 4096 × 64-bit **2.33 s → 4.58 s** (1024 × 64: 0.122 → 0.159), because
an X of 8-bit primes doubles the digit positions of the vector chain (lifting 1.79 → 3.78 s) and Garner grows with the square
of the primes; at 128-bit entries (23 primes) the chain left its bound. The wiring was removed; the pivoted LU stays
(`DetWork.piv`), and so does the 16-bit X.

**SQ** — X me prime < 256: LU me pivotim funksionon (A·A⁻¹ = I, asnjë prime singular), por HOL-i del 2× më i ngadaltë
(2,33 → 4,58 s), sepse shifrat e zinxhirit vektorial dyfishohen. U mat dhe u hoq; mbetet X 16-bit.

### A whole MoE block of K3, exact (`llm/moe.cu`)

Layer 5 end to end, real weights (62 routed experts fetched on demand by `llm/k3_fetch.py`, the gate, the latent down/up
projections, the norm, the two shared experts): router → 16 experts (w1, w3, SiTU, w2) → weighted sum → RMSNorm → up
projection → + shared experts. Exact path: every matrix product correctly rounded to BF16 (the router to FP32), the
weighted sum and RMSNorm in FP64 in a fixed order (deterministic, not exact), then rounded to BF16. Engine path: BF16 tensor cores with FP32 accumulation,
results cast to BF16, float reductions. 4 tokens (Student-t hidden states through the real post-attention norm):

| | |
|---|---|
| exact products not certified | 0 |
| exact: each token alone vs in the batch of 4 | 0 of 28 672 outputs differ |
| engine, split-K 1 vs split-K 4 | 14 041 of 28 672 outputs differ (49.0 %) |
| engine split-K 1 vs exact | 14 256 differ (49.7 %) |

The only parts not covered by correct rounding are the transcendental functions of SiTU and the sigmoid (FP64 here, rounded
once): reproducible with the same libm, and correctly rounded except in rare hard cases.

**SQ** — Një bllok i tërë MoE i K3 (shtresa 5, peshat reale): rruga e saktë jep të njëjtët bitë për çdo token, vetëm ose
në batch; motori me BF16+FP32 ndryshon 49 % të daljeve vetëm nga split-K.

### The largest model with a published data recipe: Nemotron 3 Ultra, NVFP4 W4A4 exact (`llm/nvexact.cu`, `llm/nv_check.py`)

K3 publishes its weights, not its data. Measured 27-10 on the Hugging Face API, the bulk of the data Nemotron 3 Ultra lists
(Nemotron-CC v2/v2.1, the code and SFT pre-training sets) is gated by manual approval under an agreement for internal
training only; Legal-v1, Specialized-v1.x and Code-v3 are free (CC-BY-4.0). With that caveat, the largest model that
publishes **weights, recipe and a list of its pre-training data** is NVIDIA
Nemotron 3 Ultra (Hugging Face metadata of the BF16 checkpoint: **560 524 578 816 parameters**, 55B active; model card:
226 datasets, 14.8T tokens, 53.8 TiB, a few third-party sets private). It was pre-trained in NVFP4 and its routed experts ship
as NVFP4 for weights *and* activations (ModelOpt group 16: E2M1 values, one E4M3 scale per 16, an FP32 global scale; the
activation global scale is static, `input_scale`). The NVFP4 checkpoint is 352.3 GB (summed from its 113 safetensors headers):
unlike K3, **the whole model fits on this laptop's free 669 GB**. Per token it reads 37.8 GB of dense weights (Mamba-2,
attention, latent projections, router, shared experts, lm_head) and 13.4 GB of routed experts (22 of 512 in 48 layers):
51.2 GB. With the dense part in RAM (PCIe 45 GB/s, measured) and the experts from the NVMe (5.1 GB/s), ~3.5 s per token,
bit-reproducible — 8× sooner than K3 on the same laptop.

Exactness here is cheaper than for MXFP4: one NVFP4 block of 16 is exactly one `mma.m16n8k16` INT8 (2·E2M1 is an int8,
|block sum| ≤ 2304), the two E4M3 scales multiply to an integer ≤ 225 times 2^(εW+εA), and with a shift of 18 every term is
below 2^47, so the K/16 ≤ 320 terms of an output add **in an int64 without loss**; gW·gA is a 48-bit integer, and the
104-bit product is rounded once. Nibble order checked against the BF16 checkpoint (low nibble = even k: correlation 0.996;
the other order 0.03).

One routed expert end to end, real weights (expert 0 of MoE layers 49 and 99; 4096 tokens: h ~ N(0,1) → the real
`fc1_latent_proj` → NVFP4 → up_proj → relu² → NVFP4 → down_proj):

| | L49 up | L49 down | L99 up | L99 down |
|---|---:|---:|---:|---:|
| exact: tensor-core path vs integer loops vs 32 tokens alone | 0 / 0 differ | 0 / 0 | 0 / 0 | 0 / 0 |
| independent check (`nv_check.py`, Python rationals from the raw bytes), 300 outputs | 0 wrong | 0 wrong | | |
| FP32 accumulation, alpha = fp32(gW·gA) (emulated, one block term at a time): not correctly rounded | 18.5 % | 36.1 % | 30.3 % | 49.0 % |
| **native NVFP4 tensor core** (`mma.m16n8k64 kind::mxf4nvf4.block_scale.scale_vec::4X`, sm_120a, FP32 accumulation — what engines run): not correctly rounded | **18.5 %** | **36.6 %** | **30.3 %** | **49.1 %** |
| native, split-K 1 vs 4 differ | 0 % | 0.81 % | 0 % | 0.28 % |
| native speed (the same plain tiling as the exact kernel) | 49.5 TOPS | 55.0 | 50.3 | 54.8 |
| CPU (numpy BLAS + Python integers, `nv_cpu.py`), every output vs the GPU tensor cores | 0 of 20 971 520 differ | 0 of 8 388 608 | | |
| energy, 10 s each at ~94 W (`llm/energy_nv.sh`): exact / native, GOP per joule | 194 / 447 | 204 / 473 | | |
| the same FP32 sum with an exact epilogue (summation alone) | 0 % | 0.88 % | 0 % | 0.35 % |
| split-K 1 vs 4 differ | 0 % | 0.30 % | 0 % | 0.12 % |
| exact prefill (INT8 MMA + int64 epilogue, shared-memory tiles) | 20.5 TOPS | 23.2 | 20.5 | 23.2 |
| **exact decode**, 1 token / 8 tokens (packed nibbles, dp4a + int64), GB/s of weights | **471 / 138** | 405 / 184 | 450 / 136 | 487 / 181 |
| FP32 GEMV (our plain engine-style kernel), 1 / 8 tokens, GB/s; outputs that differ from exact | 144 / 45; 18 % | 131 / 42; 36 % | 144 / 45; 29 % | 131 / 42; 49 % |

So for this model most of the "engine" error is not the order of summation but the epilogue: the global scales are
multiplied in FP32 (alpha rounded, then the product rounded again). Decode, the step that matters when a model streams,
runs exactly at L2/memory speed (one expert is 5.9 MB and stays in L2 across the timed repeats). Prefill is epilogue bound
(16 int64 shift-adds per MMA): a __constant__ scale table indexed differently by each lane is serialized and gave
10–12.6 TOPS; scales stored transposed and decoded with bit operations give 16–17.5; shared-memory tiles (cp.async, 128 x 64
per block, warps 32 x 32) give 20.5–23.2. Without the int64 epilogue the tiled kernel runs 25.6 TOPS: exactness itself costs
~20 %, the rest is that an NVFP4 block needs a k = 16 INT8 MMA, four instructions where the FP4 unit issues one k = 64.
The native FP4 MMA with a plain tiling runs 49–55 TOPS (energy 2.2–2.3× the exact path). The honest ceiling is the
instruction peak (`bench/peak_fp4.cu`, registers only): `mma.m16n8k16` INT8 116 TOPS, `mma.m16n8k64 mxf4nvf4` 465 TOPS —
with both kernels tuned to peak, exactness costs 4× in prefill; our kernels reach 18–20 % and 11 % of their peaks. In decode
both read the same packed bytes, and exactness costs nothing. The native instruction rounds 18–49 % of the outputs wrongly and changes up to 0.81 % with
the split, exactly like the emulation above: the error is the format’s FP32 epilogue and accumulation, not a kernel bug. Layer 1 is
different: its experts are 99.5 % (near) zero in the BF16 checkpoint as well (median |w| 6.6·10⁻¹⁰), so 92 % of its exact
outputs are 0.

**SQ** — Modeli më i madh me pesha të hapura dhe recetë të publikuar të dhënash (Nemotron 3 Ultra, 560 miliardë parametra,
14,8T tokenë; shumica e tyre me qasje të kufizuar: miratim me dorë, vetëm për trajnim të brendshëm) llogaritet saktë në formatin e vet NVFP4: një bllok 16 = një MMA INT8, shkallët si shift në int64, një
rrumbullakim. 0 gabime (edhe te kontrolli i pavarur); motori FP32 gabon 18–49 %, kryesisht te shkallët globale. Decode i
saktë: 405–487 GB/s. I gjithë modeli NVFP4 (352 GB) hyn në diskun e laptopit; një token lexon 51,2 GB → ~3,5 s/token, saktë.


#### The Ultra router (`llm/router_nv.cu`)

The gate of Nemotron 3 Ultra is FP32 (512 × 8192; sigmoid + `e_score_correction_bias`, top-22). An FP32 weight is exactly
the sum of three BF16 numbers (checked for all 4 194 304), so x·Wᵀ = [x x x]·[W₁ W₂ W₃]ᵀ is **one** correctly rounded
BF16 × BF16 product with K = 24 576 on the existing machinery (`llm/exact.cuh`): 2 097 152 logits in 91 ms, all certified.
Student-t hidden states through the layer's real RMSNorm weight, 4096 tokens, layers 49 and 99, seeds 1 and 2:

| | |
|---|---|
| FP32 FMA logits not correctly rounded (split-K 1 / 2 / 4 / 8) | 96.8 / 95.4 / 93.3 / 90.4 % |
| token-layer decisions routed to another expert set (FP32 split-K 1 vs exact) | 1 of 16 384 |
| the same, split-K 1 vs 4 | 1 of 16 384 |
| gap between the 22nd and 23rd score, median / 1st percentile | 0.0017–0.0024 / 1.9–3.1·10⁻⁵ (ties occur: smallest 0) |

Almost every logit is rounded differently, yet routing moves for about 1 decision in 16 000 — over 48 MoE layers, roughly
one token in 340 goes through at least one different expert. The FP32 gate keeps Ultra about as stable as K3 (1 in 13 000
per layer with a BF16 gate); exact ties are broken by index here, and an engine's top-k may break them otherwise.

**SQ** — Ruteri i Ultra-s: porta FP32 ndahet saktë në 3 BF16 dhe logit-et dalin të rrumbullakosura saktë në 91 ms. Logit-et
FP32 të zakonshme janë 90–97 % jo të sakta, por rrugëzimi ndryshon vetëm 1 herë në 16 384 vendime (≈ 1 token në 340 në
gjithë 48 shtresat).

#### A whole routed MoE block of Ultra, exact (`llm/moe_nv.cu`)

Layer 49, real weights, experts fetched on demand: exact router (FP32 gate as three BF16) → top-22 → `fc1_latent_proj`
(correctly rounded to BF16) → each expert's NVFP4 up → relu² → down (correctly rounded) → weighted sum in FP64 in expert
order → `fc2_latent_proj` (correctly rounded to BF16). Engine path: FP32 router, BF16 tensor cores with FP32 accumulation,
the native NVFP4 instruction, FP32 sum, split-K 1 or 4. 4 tokens (Student-t through the real norm), 75 experts: exact
products not certified 0; **each token alone vs in the batch: 0 of 32 768 BF16 outputs differ**; engine split-K 1 vs 4:
35 differ (0.1 %); engine vs exact: 37 (0.1 %). The shared expert (FP8 W8A8) is not in it yet.

#### The whole of Nemotron 3 Ultra, token by token, every linear map exact (`llm/ultra.cu`)

All 560 B parameters (the 352 GB NVFP4 checkpoint, sha256-checked) stream from the laptop's NVMe for every token. One GEMV
covers the checkpoint's four formats — BF16 (attention, latent projections, lm_head), FP32 (router), FP8 E4M3 W8A8 (Mamba
in/out, shared experts), NVFP4 W4A4 (routed experts): once the activation is quantized as its format prescribes, every
element is an integer times a power of two; a warp sums its row in a 128-bit fixed point anchored at the row's largest
possible term, multiplies by the exact product of the FP32 global scales and rounds **once** to BF16 (FP32 for router and
logits), certified (terms below the 125-bit window are counted; the rounding stands when their interval rounds to one
value). The rest — Mamba-2 recurrence (FP32 state), causal conv, attention (no positional encoding), sigmoid top-22
routing, RMSNorms, residuals — runs on the host in FP64, rounded once where the model keeps BF16. Semantics from
transformers v5.3.0 `modeling_nemotron_h.py` (production `dt_limit (0, ∞)`, not the eager path's clamp at 1e-3).

- `out_ultra --selftest`: 6 656 outputs of all four formats against an independent 2 560-bit host superaccumulator (narrow
  and 2^±60 ranges, heavy cancellation, exact ties): 0 differ, 0 uncertified. Mutants caught: a 40-bit window (3 459
  differ), ties away from zero (188). The tie test caught a real bug first: squeezing to 78 bits widened the interval even
  when no bit was lost, so every exact tie came out uncertified and on the wrong side.
- One sequence reads 48 GB per token: 33–37 s per token; with read-ahead (`OZ_AHEAD=3`: a thread reads the next tensors,
  in the order they will be used, into a 3 GB pinned ring) **19.5–19.7 s**, digest unchanged (`5b5a0167bdff8394`); the
  FP32 engine path with read-ahead 20.3–20.9 s — **exactness costs nothing at decode**, both wait on the disk (measured
  with the K2 download paused). "The capital of Kosovo is" → **" Pristina"**.
- Three prompts × {exact, engine split-K 1, engine split-K 4} = nine sequences, 45 steps of 57–71 s (70–86 GB read per
  step, no read-ahead yet): **1.54 billion correctly rounded outputs, 0 uncertified**. Shadow (the engine GEMV on the
  exact path's own inputs): FP32 router logits 40.6–47.3 % rounded differently, FP8 and NVFP4 only 0.003 % (41 914 of
  1.42 billion). Counted apart (one sequence, 7 steps; the first count lumped the logits in with BF16): BF16 projections
  **0.01 %** (561 of 4.86 M), FP32 output logits 73–83 %. What is rounded to BF16 is almost always exact under FP32
  accumulation; what stays in FP32 (router, logits) is not — and the router's logits set the expert weights.
- Downstream the engine's top logit drifts up to **1.22** from the exact one (mean 0.17–0.31) while tokens agree; the exact
  top-two margin falls to 0.017. Albanian: exact "Kryeqyteti i Kosovës është Prishtina." then "Kosova është vend i ri i
  pavarur."; split-K 1 parts at the 13th generated token ("Kosova është vend i pavarur që ka deklaruar pavarësinë më 17
  shkurt 2008."), split-K 4 at the 5th (indentation). English: parts at the 29th (" and" vs " of").
  `123456789 * 987654321 =` → **121932631112635269** on all three paths (correct).
- `ultra_summary.py` prints this from the log; the drivers print an FNV-1a 64 digest of every residual stream and logit
  vector of the exact path (from the next run on), so a third party with the checkpoint can check a run bit for bit.

**Full access** (measured 27-10 on the HF API and model cards): Ultra's main data is gated; Jais-3-375B-A23B (379 B,
Apache-2.0) beats Ultra on most rows of its own card but publishes a 10 % sample of its data; the strongest model whose
data is wholly public is IFM's K2-Think-V2 (72.6 B, TxT360 ODC-BY); OLMo-2-0325-32B-Instruct is the one model whose whole
training data (4.6 T tokens) is indexed for public lookup (infini-gram). `llm/dense.cu` runs both dense models exact;
`llm/olmo_trace.py` finds every maximal span of an output in that data ("The capital of Kosovo is Pristina": 224 times,
a DCLM page from graphicmaps.com); `--index v4_dclm-baseline_llama` looks Ultra's text up in the public web it was drawn
from: in Ultra's exact English continuation the longest span found verbatim is 9 words, "Pristina is the capital and
largest city of Kosovo." (11 times; one source albbnb.com), with nine shorter ones of 5–6 words.

OLMo 2 32B, same three prompts (`llm/chain_open.sh`, 39 steps of 30 s, 64 GB each, 0 uncertified): the exact path writes
"The municipalities are divided into **10,908** settlements", both engines "**2,994** settlements" (parting at the 29th
generated token); in all 4.6 T training tokens "10,908 settlements" and "2,994 settlements" occur **0** times — both made
up, differently. Its product is wrong on every path (`12345678987654321`), with an explanation stitched from training
phrases ("This is a special case of a more general phenomenon: the", 11 words, 22 times). Albanian: same text on all
paths ("Kryeqyteti i Kosovës është një qyteti në Kosovë. It is the capital and largest city of Kosovo."). Trace:
`/opt/ozaki/chain/olmo_trace.txt`.

K2-Think-V2 72.6 B, the same nine sequences from disk (39 steps of 93–106 s, 274 GB read per step, 0 uncertified; the
engine rounds 0.01–0.02 % of the BF16 outputs and 69–81 % of the FP32 logits differently). Both engines agree with each
other and part from the exact path: Albanian at the 8th generated token (exact "Prishtina, **qyteti më i madh në vend**",
engines "Prishtina, **qytet i vendosur në pjesën qendrore të vendit**"), English at the 27th ("confirmation **or
correction?**" against "confirmation **or maybe they are stating a fact**"). Both texts are true; the rounding picks which
one. The product prompt is still reasoning after 30 generated tokens, identically on all paths. Log:
`/opt/ozaki/chain/k2.{log,txt}`.

**SQ** — Nemotron 3 Ultra i tërë (560 miliardë parametra) punon në laptop nga disku, token pas tokeni, me çdo prodhim
matricë-vektor të rrumbullakosur saktë (një rrumbullakim i certifikuar, i provuar kundër një akumuluesi 2 560-bit; prova e
barazimeve gjeti dhe ndreqi një defekt). 19,6 s për token me lexim përpara — aq sa motori FP32: saktësia s'kushton asgjë.
Motori me akumulim FP32 ndryshon 42–49 % të logit-ëve të ruterit (projeksionet BF16 vetëm 0,01 %), logit-i largohet deri në
1,22, dhe teksti ndahet: shqip pas tokenit të 5-të / 13-të, anglisht pas të 29-tit. Te OLMo 2 rruga e saktë shkruan «10,908»
vendbanime, motori «2,994» — asnjëri s'ndodhet te 4,6T tokenat e trajnimit: të dy të shpikur, ndryshe. 123456789 × 987654321 = 121932631112635269 — saktë në të tri rrugët. Te K2-Think-V2 (72,6 B, nga disku, 0 të pacertifikuara) të dy motorët bien dakord mes tyre dhe ndahen nga rruga e saktë: shqip pas tokenit të 8-të (e sakta «qyteti më i madh në vend», motorët «qytet i vendosur në pjesën qendrore të vendit»), anglisht pas të 27-tit — të dyja të vërteta, rrumbullakimi zgjedh cilën. Për qasje të plotë te të
dhënat: OLMo 2 32B (të gjitha të dhënat e trajnimit të kërkueshme), K2-Think-V2 (më i forti plotësisht i hapur); asnjë model
plotësisht i hapur s'e kalon Ultra-n sot.
