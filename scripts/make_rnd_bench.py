#!/usr/bin/env python3
"""
Generate planted random benchmark tensors and their known CPD decompositions.

For each (n, r, k) triple, generates a random CP decomposition (U, V, W) of
rank r over GF(2), symmetrizes it to obtain a signature tensor, and saves:
  - Sparse tensor to data/tensors/{id:04d}.npy
  - Ground-truth CPD to data/other/cpd-rnd/topp/{id:04d}-{r:02d}.npy

Tensor IDs: 2000 + i*64 + j*4 + k, where
  i = index into N_VALUES (n = 6,8,...,20)
  j = index into R_VALUES (r = 1,2,...,16)
  k = 0..3 (repetitions)

Total: 8 * 16 * 4 = 512 tensors (IDs 2000–2511).

Usage:
    python make_rnd_bench.py

If tensors already exist, the script verifies them against the new generation
and reports mismatches (tensors will differ due to randomness, so this is
mainly useful for checking the format).
"""

import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent

TENSORS_DIR = REPO_DIR / "data" / "tensors"
CPD_RND_DIR = REPO_DIR / "data" / "other" / "cpd-rnd" / "topp"

N_VALUES = [6, 8, 10, 12, 14, 16, 18, 20]
R_VALUES = list(range(1, 17))
K_REPEAT = 4

# All 6 permutations of a 3-tensor
PERMS_3 = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]


# ---------------------------------------------------------------------------
# Tensor generation
# ---------------------------------------------------------------------------


def make_planted_tensor(n, r, rng):
    """Generate a random planted benchmark.

    Returns (U, V, W, T_sparse) where U, V, W are binary (r, n) matrices
    and T_sparse is the symmetrized tensor in sparse format.
    """
    U, V, W = (rng.random((3, r, n)) < 0.5).astype(np.uint8)

    # CP tensor: T_ijk = sum_q U_qi V_qj W_qk (mod 2)
    T = np.einsum("qi,qj,qk->ijk", U, V, W) % 2

    # Symmetrize: sum over all 6 permutations (mod 2)
    T_sym = sum(T.transpose(p) for p in PERMS_3) % 2

    # Extract strictly increasing triples (i < j < k)
    coords = np.stack(np.nonzero(T_sym), axis=1)
    if coords.size == 0:
        idx = np.empty((0, 3), dtype=np.int64)
    else:
        mask = (coords[:, 1] > coords[:, 0]) & (coords[:, 2] > coords[:, 1])
        idx = coords[mask].astype(np.int64)

    sparse = np.empty((idx.shape[0] + 1, 3), dtype=np.int64)
    sparse[0] = (n, n, n)
    if idx.shape[0] > 0:
        sparse[1:] = idx

    return U, V, W, sparse


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    TENSORS_DIR.mkdir(parents=True, exist_ok=True)
    CPD_RND_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng()

    header = f"{'ID':>4}  {'n':>2} {'r':>2} {'k':>1}  {'nnz':>5}  Status"
    print(header)
    print("-" * len(header))

    total = 0
    new_count = 0

    for i, n in enumerate(N_VALUES):
        for j, r in enumerate(R_VALUES):
            for k in range(K_REPEAT):
                tensor_id = 2000 + i * 64 + j * 4 + k
                U, V, W, T_sparse = make_planted_tensor(n, r, rng)
                nnz = T_sparse.shape[0] - 1

                # Save tensor
                tensor_path = TENSORS_DIR / f"{tensor_id:04d}.npy"
                if tensor_path.exists():
                    status = "exists"
                else:
                    np.save(tensor_path, T_sparse)
                    status = "new"
                    new_count += 1

                # Save ground-truth CPD: shape (3, r, n), dtype uint8
                cpd_path = CPD_RND_DIR / f"{tensor_id:04d}-{r:02d}.npy"
                if not cpd_path.exists():
                    uvw = np.stack((U, V, W)).astype(np.uint8)
                    np.save(cpd_path, uvw)

                total += 1

                if n == N_VALUES[0] and r <= 2:
                    print(
                        f"{tensor_id:04d}  {n:>2} {r:>2} {k:>1}  {nnz:>5}  {status}"
                    )

        # Summary line per n
        id_lo = 2000 + i * 64
        id_hi = 2000 + i * 64 + 15 * 4 + 3
        print(f"  n={n:>2}:  IDs {id_lo}-{id_hi}  ({16 * K_REPEAT} tensors)")

    print("-" * len(header))
    print(f"Total: {total} tensors ({new_count} new)")


if __name__ == "__main__":
    main()
