# Tensor Index

## ID Scheme

Tensor IDs use 4-digit numbers.  The last two digits encode the circuit/structure within a family, while the leading digits encode the source and representation variant.

| Series | Format | Description |
|--------|--------|-------------|
| `01xx` | `(n, n, n)` | AlphaTensor-Quantum benchmarks |
| `03xx` | `(k, k, 2k-1)` | Polynomial multiplication (convolution), base |
| `05xx` | `(k, k, k)` | GF(2^k) multiplication (mod P(x)), base |
| `07xx` | `(3k, 3k, 3k)` | GF(2^k) multiplication, topp (phase tensor) |
| `08xx` | `(n, n, n)` | Circuit benchmarks (from VivienVandaele/quantum-circuit-optimization) |
| `2xxx` | `(n, n, n)` | Planted random benchmarks |

For the `xx` suffix in GF multiplication families: `xx = k + 23` (k=2 gives 25, k=10 gives 33).

Circuits that appear in both sources have two IDs: `08xx` (from circuit compilation via `setup_circuits_todd.py`) and `01xx` (from AlphaTensor-Quantum via `setup_circuits_atq.py`).  The underlying tensors are identical.

## Circuit Benchmarks

Best known Toffoli-count (CP rank) and T-count (Waring rank).

Columns `08xx` and `01xx` show the tensor IDs from the two sources.  Circuits marked `--` in the `01xx` column are only available via circuit compilation; those marked `--` in the `08xx` column are only available from AlphaTensor-Quantum.  The `data/paper/` column shows which tensor variants have verified decompositions achieving the claimed ranks (see `scripts/verify_paper_schemes.py`).

| 08xx | 01xx | Circuit | n | nnz | CP | CP method | Waring | `data/paper/` |
|-----:|-----:|---------|--:|----:|---:|-----------|-------:|---------------|
| 0800 | -- | Adder_8 | 61 | 3268 | 27 | BCO+SGE | 117 | 08 |
| 0801 | 0101 | Barenco Tof_3 | 8 | 4 | 2 | BCO | 13 | 08 01 |
| 0802 | 0102 | Barenco Tof_4 | 14 | 10 | 4 | BCO | 23 | 08 01 |
| 0803 | 0103 | Barenco Tof_5 | 20 | 18 | 6 | BCO | 33 | 08 01 |
| 0804 | 0104 | Barenco Tof_10 | 50 | 63 | 16 | BCO | 83 | 08 01 |
| 0805 | 0105 | CSLA MUX_3 | 21 | 38 | 8 | BCO | 39 | 08 01 |
| 0806 | 0106 | CSUM MUX_9 | 42 | 86 | 14 | BCO | 71 | 08 01 |
| 0807 | -- | Grover_5 | 77 | 6163 | 25 | BCO | 143 | 08 |
| 0808 | 0108 | Ham15 (low) | 46 | 1008 | 17 | BCO | 73 | 08 01 |
| 0809 | -- | Ham15 (med) | 71 | 635 | 33 | BCO+SGE | 137 | 08 |
| 0810 | -- | Ham15 (high) | 351 | 18777 | 155 | BCO+SGE | 643 | 08 |
| 0811 | 0111 | HWB_6 | 27 | 276 | 10 | BCO+SGE | 51 | 08 01 |
| 0812 | 0112 | Mod 5_4 | 5 | 5 | 1 | BCO | 7 | 08 01 |
| 0813 | -- | Mod Adder_1024 | 332 | 129405 | 128 | BCO | 573 | 08 |
| 0814 | 0114 | Mod Mult_55 | 12 | 9 | 3 | BCO | 17 | 08 01 |
| 0815 | 0115 | Mod Red_21 | 28 | 74 | 11 | BCO | 51 | 08 01 |
| 0816 | -- | QCLA Adder_10 | 61 | 127 | 24 | BCO | 107 | 08 |
| 0817 | 0117 | QCLA Com_7 | 42 | 75 | 12 | BCO | 59 | 08 01 |
| 0818 | -- | QCLA Mod_7 | 84 | 2144 | 37 | BCO | 153 | 08 |
| 0819 | 0119 | RC Adder_6 | 24 | 21 | 6 | BCO | 37 | 08 01 |
| 0820 | 0120 | Tof_3 | 7 | 3 | 2 | BCO | 13 | 08 01 |
| 0821 | 0121 | Tof_4 | 11 | 8 | 3 | BCO | 19 | 08 01 |
| 0822 | 0122 | Tof_5 | 15 | 15 | 4 | BCO | 25 | 08 01 |
| 0823 | 0123 | Tof_10 | 35 | 60 | 9 | BCO | 55 | 08 01 |
| 0824 | 0124 | VBE Adder_3 | 14 | 40 | 3 | BCO | 19 | 08 01 |

## GF(2^k) Multiplication

Best known Toffoli-count (CP rank) and T-count (Waring rank).

Three tensor representations exist for each field size k = 2..10:

| Series | Dims | Description |
|--------|------|-------------|
| `03xx` | `(k, k, 2k-1)` | Polynomial multiplication (convolution) |
| `05xx` | `(k, k, k)` | GF(2^k) multiplication mod P(x) |
| `07xx` | `(3k, 3k, 3k)` | Phase tensor (topp embedding of `05xx`) |

ATQ ID `01xx` corresponds to the topp tensor `07xx` for the same field.

| 07xx | 05xx | 03xx | 01xx | Circuit | k | n | nnz | CP | Waring | `data/paper/` |
|-----:|-----:|-----:|-----:|---------|--:|--:|----:|---:|-------:|---------------|
| 0725 | 0525 | 0325 | 0125 | GF(2^2) Mult | 2 | 6 | 5 | 3 | 17 | 07 |
| 0726 | 0526 | 0326 | 0126 | GF(2^3) Mult | 3 | 9 | 12 | 6 | 29 | 07 |
| 0727 | 0527 | 0327 | 0127 | GF(2^4) Mult | 4 | 12 | 22 | 9 | 39 | 07 |
| 0728 | 0528 | 0328 | 0128 | GF(2^5) Mult | 5 | 15 | 36 | 13 | 59 | 07 01 |
| 0729 | 0529 | 0329 | 0129 | GF(2^6) Mult | 6 | 18 | 51 | 15 | 77 | 07 |
| 0730 | 0530 | 0330 | 0130 | GF(2^7) Mult | 7 | 21 | 70 | 22 | 101 | 07 |
| 0731 | 0531 | 0331 | 0131 | GF(2^8) Mult | 8 | 24 | 150 | 24 | 123 | 07 |
| 0732 | 0532 | 0332 | 0132 | GF(2^9) Mult | 9 | 27 | 123 | 30 | 147 | 07 |
| 0733 | 0533 | 0333 | 0133 | GF(2^10) Mult | 10 | 30 | 148 | 33 | 173 | 07 |

The `n` and `nnz` columns refer to the topp tensor (`07xx`).  CP and Waring ranks are best overall results from the paper.

## Additional ATQ Benchmarks

Circuits from AlphaTensor-Quantum not in the standard benchmark set above.

### Binary Addition (Cuccaro Adder)

| ID | Circuit | n | nnz | CP | CP method | Waring |
|----|---------|--:|----:|---:|-----------|-------:|
| 0134 | Cuccaro Adder n=3 | 8 | 6 | 2 | BCO | 13 |
| 0135 | Cuccaro Adder n=4 | 12 | 21 | 3 | BCO | 19 |
| 0136 | Cuccaro Adder n=5 | 16 | 55 | 4 | BCO | 25 |
| 0137 | Cuccaro Adder n=6 | 20 | 130 | 5 | BCO | 31 |
| 0138 | Cuccaro Adder n=7 | 24 | 134 | 6 | BCO | 37 |
| 0139 | Cuccaro Adder n=8 | 28 | 242 | 7 | BCO | 43 |
| 0140 | Cuccaro Adder n=9 | 32 | 394 | 8 | BCO | 49 |
| 0141 | Cuccaro Adder n=10 | 36 | 422 | 9 | BCO | 55 |

### Quantum Chemistry (Basis Change)

| ID | Circuit | n | nnz | CP | CP method | Waring |
|----|---------|--:|----:|---:|-----------|-------:|
| 0142 | Basis Change p=4 o=3 | 27 | 530 | 8 | BCO+SGE | 41* |
| 0143 | Basis Change p=4 o=4 | 39 | 1965 | 12 | BCO+SGE | 61* |
| 0144 | Basis Change p=4 o=5 | 51 | 5170 | 16 | BCO+SGE | 81 |
| 0145 | Basis Change p=5 o=3 | 39 | 2129 | 12 | BCO+SGE | 61* |
| 0146 | Basis Change p=5 o=4 | 56 | 3767 | 18 | BCO+SGE | 91* |
| 0147 | Basis Change p=6 o=3 | 51 | 3899 | 16 | BCO+SGE | 85* |
| 0148 | Basis Change p=7 o=3 | 63 | 11098 | 20 | BCO+SGE | 105* |

### Hamming Weight / Phase Gradient

| ID | Circuit | n | nnz | CP | CP method | Waring |
|----|---------|--:|----:|---:|-----------|-------:|
| 0149 | Hamming Weight n=4 | 9 | 20 | 3 | BCO | 19 |
| 0150 | Hamming Weight n=5 | 10 | 24 | 3 | BCO | 19 |
| 0151 | Hamming Weight n=6 | 12 | 18 | 4 | BCO | 25 |
| 0152 | Hamming Weight n=7 | 13 | 35 | 4 | BCO | 25 |
| 0153 | Hamming Weight n=8 | 21 | 84 | 7 | BCO | 43 |
| 0154 | Hamming Weight n=9 | 22 | 74 | 7 | BCO | 43 |
| 0155 | Hamming Weight n=10 | 24 | 86 | 8 | BCO | 49 |
| 0156 | Hamming Weight n=11 | 25 | 135 | 8 | BCO | 49 |
| 0157 | Hamming Weight n=12 | 30 | 183 | 10 | BCO | 61 |
| 0158 | Hamming Weight n=13 | 31 | 171 | 10 | BCO | 61 |
| 0159 | Hamming Weight n=14 | 33 | 112 | 11 | BCO | 67 |
| 0160 | Hamming Weight n=15 | 34 | 152 | 11 | BCO | 67 |
| 0161 | Hamming Weight n=16 | 45 | 514 | 15 | BCO | 91 |
| 0162 | Hamming Weight n=17 | 46 | 919 | 15 | BCO | 91 |
| 0163 | Hamming Weight n=18 | 48 | 327 | 16 | BCO | 97 |
| 0164 | Hamming Weight n=19 | 49 | 380 | 16 | BCO | 97 |
| 0165 | Hamming Weight n=20 | 54 | 781 | 18 | BCO | 109 |

### Unary Iteration

| ID | Circuit | n | nnz | CP | CP method | Waring |
|----|---------|--:|----:|---:|-----------|-------:|
| 0166 | Unary Iteration n=3 | 20 | 86 | 7 | BCO | 31 |
| 0167 | Unary Iteration n=4 | 38 | 215 | 15 | BCO | 63 |
| 0168 | Unary Iteration n=5 | 72 | 404 | 31 | BCO+SGE | 127 |

## Planted Random Benchmarks

512 tensors with known CP decomposition rank, for benchmarking search algorithms.

ID = `2000 + i*64 + j*4 + k`, where i indexes n, j indexes r, k = 0..3 (repetitions).

| i | n | ID range |
|--:|--:|----------|
| 0 | 6 | 2000-2063 |
| 1 | 8 | 2064-2127 |
| 2 | 10 | 2128-2191 |
| 3 | 12 | 2192-2255 |
| 4 | 14 | 2256-2319 |
| 5 | 16 | 2320-2383 |
| 6 | 18 | 2384-2447 |
| 7 | 20 | 2448-2511 |

For each n: r = 1, 2, ..., 16 with 4 repetitions each (64 tensors).
Ground-truth decompositions are stored in `data/other/cpd-rnd/topp/`.

## File Format

Tensors are stored as NumPy `.npy` files with shape `(nnz+1, 3)` and dtype `int64`:

- Row 0: dimensions `(n1, n2, n3)`
- Rows 1+: nonzero entry coordinates `(i, j, k)`

For symmetric tensors (circuit benchmarks, topp), triples satisfy `i < j < k`.
For bilinear tensors (base GF multiplication), triples are unsorted.

## Generation Scripts

| Script | Tensors | Description |
|--------|---------|-------------|
| `setup_circuits_todd.py` | `08xx` | Clone circuit repo, compile to phase polynomials |
| `setup_circuits_atq.py` | `01xx` | Clone AlphaTensor-Quantum, reconstruct Waring tensors |
| `make_tensors_gfmult.py` | `03xx`, `05xx`, `07xx` | Polynomial and GF(2^k) multiplication tensors |
| `make_cpd_gf.py` | (CPD schemes) | Known CPD decompositions for GF(2^k) |
| `make_rnd_bench.py` | `2xxx` | Planted random benchmarks |
