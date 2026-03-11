# Reproducing Results

## Prerequisites

- C++20 compiler (g++ recommended), pthreads
- Python 3.10+ with `numpy`, `matplotlib`
- `git` and internet access (for `setup_circuits_atq.py`, `setup_circuits_todd.py`)

## Tensor setup

Tensors are pre-generated in `data/tensors/`. To regenerate or verify:

```bash
python scripts/setup_circuits_atq.py      # 01xx (ATQ benchmarks, requires git)
python scripts/setup_circuits_todd.py     # 08xx (VV benchmarks, requires git + bin/compile)
python scripts/make_tensors_gfmult.py     # 03xx, 05xx, 07xx (GF multiplication)
python scripts/make_cpd_gf.py             # known CPDs for GF multiplication
python scripts/make_rnd_bench.py          # 2xxx (planted random, non-deterministic)
```

## Verification

```bash
python scripts/verify_paper_schemes.py    # verify all decompositions in data/paper/
```

## Reproduce scripts

Each script has configuration parameters at the top. Default values are reduced for quick testing; paper values are noted in comments.

| Script | Paper | Output | Time (default) |
|--------|-------|--------|----------------|
| `reproduce_tab1.py` | Table 1 (Toffoli) | stdout | ~minutes |
| `reproduce_tab1_waring.py` | Table 1 (T-count) | stdout | ~minutes |
| `reproduce_tab2.py` | Table 2 | stdout | ~10 min |
| `reproduce_fig2.py` | Figure 2 | `data/fig2.tsv`, `data/fig2.png` | ~hours |
| `reproduce_fig3.py` | Figure 3 | `data/fig3.tsv`, `data/fig3.png` | ~hours |

### Table 1: Circuit benchmarks

```bash
python scripts/reproduce_tab1.py          # BCO + SGE → Toffoli counts
python scripts/reproduce_tab1_waring.py   # FGS + Waring → Toffoli + T-counts
```

`reproduce_tab1_waring.py` depends on `reproduce_tab1.py` (uses its BCO-transformed tensors).

### Table 2: GF(2^k) multiplication

```bash
python scripts/reproduce_tab2.py
```

### Figure 2: Planted CP benchmark

```bash
python scripts/reproduce_fig2.py
```

Requires planted tensors (`2xxx`) from `make_rnd_bench.py`.

### Figure 3: Waring on planted benchmark

```bash
python scripts/reproduce_fig3.py
```

Requires `reproduce_fig2.py` to have been run first (BCO tensors + CPD files).

## Key parameters

| Parameter | Default | Paper |
|-----------|---------|-------|
| BCO beam width | 8 | 1024 |
| SGE beam width | 8 | 1024 |
| FGS pool size (`-s`) | 40 | 1000 |
| FGS path limit (`-f`) | 10^6 | 10^6 |
| FGS plus limit (`-p`) | 5×10^4 | 5×10^4 |
| Waring beam width | 64 | 1024 |
| NUM_REPS (fig2, fig3) | 1 | 4 |
| NUM_RUNS (tab2) | 2 | 10 |
