# Python Scripts

## Shared utilities

### `gf2.py`

Shared library module (not executable). Provides GF(2) linear algebra: matrix inversion, rank computation, Gauss-Jordan elimination (Numba-accelerated). Also includes tensor I/O helpers, basis change application, and bitvector decoding.

Used by: `make_cpd_gf.py`, `verify_paper_schemes.py`, `collect_paper_schemes.py`.

Requires: `numpy`, `numba`.

## Tensor generation

### `make_tensors_gfmult.py`

Generates GF(2^k) multiplication tensors for k = 2..10 in three representations:

| Series | Dims | Description |
|--------|------|-------------|
| `03xx` | `(k, k, 2k-1)` | Polynomial multiplication (convolution) |
| `05xx` | `(k, k, k)` | GF(2^k) multiplication mod irreducible polynomial |
| `07xx` | `(3k, 3k, 3k)` | Phase tensor (topp embedding of `05xx`) |

Outputs 27 tensors to `data/tensors/` (IDs 0325-0333, 0525-0533, 0725-0733). Uses hardcoded primitive polynomials; no external input needed.

### `make_cpd_gf.py`

Constructs known-rank CP decompositions for GF(2^k) multiplication tensors (k = 2..10) using algebraic tower-field decompositions with explicit isomorphisms.

Outputs to `data/cpd/base/` and `data/cpd/topp/`. Optionally verifies against pre-existing tensors in `data/tensors/`.

### `make_rnd_bench.py`

Generates 512 planted random benchmark tensors with known CP rank (ground truth). Parameters: 8 dimensions (6, 8, ..., 20) × 16 ranks (1..16) × 4 repetitions.

Outputs tensors (IDs 2000-2511) to `data/tensors/` and ground-truth CPD files to `data/other/cpd-rnd/topp/`.

> **Note:** Uses random generation; re-running produces different tensors. Pre-generated tensors are included in the repository.

## Circuit import

### `setup_circuits_atq.py`

Imports AlphaTensor-Quantum benchmark circuits. Clones the [alphatensor_quantum](https://github.com/google-deepmind/alphatensor_quantum) repo, reconstructs symmetric tensors from Waring decomposition factors.

Outputs 66 tensors (IDs 0101-0168) to `data/tensors/`.

Requires: `git`, internet access.

### `setup_circuits_todd.py`

Imports quantum circuits from the [quantum-circuit-optimization](https://github.com/VivienVandaele/quantum-circuit-optimization) repository. Copies `.qc` files to `data/circuits/raw/` with numeric IDs, optionally compiles them to extract phase polynomials and tensors.

Outputs `.qc` files and tensors (IDs 0800-0876) to `data/circuits/raw/` and `data/tensors/`.

Requires: `git`, internet access. Optional: `bin/compile` (for tensor extraction).

## Verification

### `verify_paper_schemes.py`

Verifies all CP and Waring decompositions in `data/paper/` against original tensors. For each tensor, loads the BCO transform, reconstructs the dense tensor from the decomposition, and checks it matches the original.

Reads from `data/tensors/`, `data/paper/transform/`, `data/paper/cpd/topp/`, `data/paper/waring/`. Prints PASS/FAIL for each tensor; exits with status 1 on any failure.

## Internal (not needed for reproduction)

### `collect_paper_schemes.py`

Collects best optimization results from `data/tmp/ncgm02/` (intermediate experimental output) into `data/paper/`. This is an internal script used during paper preparation — the collected results are already included in the repository.
