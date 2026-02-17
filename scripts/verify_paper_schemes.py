"""
Verify CP and Waring decompositions in data/paper/ against original tensors.

For each tensor:
  1. Load original tensor and BCO transform
  2. Compute the BCO-transformed tensor (reference)
  3. Verify CP decomposition by reconstructing the dense tensor
  4. Verify Waring decomposition by reconstructing the dense tensor

Usage:
    python verify_paper_schemes.py
"""

import glob
import time
from pathlib import Path

import numpy as np

from gf2 import (
    apply_inverse_transform,
    bitvecs_to_dense,
    gf2_inv,
    load_tensor_dense,
    upper_triples,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent

TENSORS_DIR = REPO_DIR / "data" / "tensors"
TRANSFORM_DIR = REPO_DIR / "data" / "paper" / "transform"
CPD_DIR = REPO_DIR / "data" / "paper" / "cpd" / "topp"
WARING_DIR = REPO_DIR / "data" / "paper" / "waring"


# ---------------------------------------------------------------------------
# BCO tensor computation
# ---------------------------------------------------------------------------


def compute_bco_tensor(tensor_id):
    """Compute BCO-transformed dense tensor. Returns (n, T_dense) or (None, None)."""
    n, T = load_tensor_dense(TENSORS_DIR / f"{tensor_id:04d}.npy")
    if n is None:
        return None, None
    A = np.load(TRANSFORM_DIR / f"{tensor_id:04d}.npy")
    A_inv = gf2_inv(A)
    return n, apply_inverse_transform(T, A_inv)


# ---------------------------------------------------------------------------
# CP decomposition verification
# ---------------------------------------------------------------------------


def verify_cpd(cpd_path, n, ref_dense):
    """Verify CP decomposition by building dense tensor from rank-1 terms.

    Each rank-1 term (u, v, w) contributes Sym(u tensor v tensor w).
    All terms are decoded into matrices U, V, W of shape (rank, n) and
    the tensor is reconstructed via a single batched einsum + axis permutations.

    Returns (rank, ok).
    """
    data = np.load(cpd_path)
    row = data[0]
    vec_words = (n + 63) // 64
    rank = len(row) // (3 * vec_words)

    # Decode all 3*rank bitvectors at once
    all_vecs = bitvecs_to_dense(
        row.reshape(3 * rank, vec_words), n
    ).astype(np.int32)
    U = all_vecs[0::3]  # (rank, n)
    V = all_vecs[1::3]
    W = all_vecs[2::3]

    # T_uvw[i,j,k] = sum_t U[t,i] V[t,j] W[t,k]
    T_uvw = np.einsum("ti,tj,tk->ijk", U, V, W)
    # Symmetrize: sum over all 6 permutations of axes
    T_rec = (T_uvw + T_uvw.transpose(0, 2, 1) + T_uvw.transpose(1, 0, 2) + T_uvw.transpose(1, 2, 0) + T_uvw.transpose(2, 0, 1) + T_uvw.transpose(2, 1, 0)) % 2

    ok = upper_triples(T_rec) == upper_triples(ref_dense)
    return rank, ok


# ---------------------------------------------------------------------------
# Waring decomposition verification
# ---------------------------------------------------------------------------


def verify_waring(waring_path, n, ref_dense):
    """Verify Waring decomposition by building dense tensor from parities.

    Each parity vector p contributes p^{tensor 3}.  All parities are loaded
    as a matrix A of shape (tcount, n) and the tensor is reconstructed via
    a single batched einsum: T[i,j,k] = sum_q A[q,i] A[q,j] A[q,k] mod 2.

    Returns (tcount, ok).
    """
    A = np.load(waring_path).astype(np.int32)  # (tcount, n)
    tcount = A.shape[0]

    T_rec = np.einsum("qi,qj,qk->ijk", A, A, A) % 2

    ok = upper_triples(T_rec) == upper_triples(ref_dense)
    return tcount, ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def find_file(directory, tensor_id):
    """Find file matching {id:04d}-*.npy in directory."""
    pattern = str(directory / f"{tensor_id:04d}-*.npy")
    matches = glob.glob(pattern)
    return Path(matches[0]) if matches else None


def parse_rank_from_path(path):
    return int(path.stem.split("-")[1])


def main():
    t0 = time.time()

    transform_ids = sorted(
        int(f.stem) for f in TRANSFORM_DIR.iterdir() if f.suffix == ".npy"
    )

    print(f"Verifying {len(transform_ids)} tensors from data/paper/\n")
    print(f"{'ID':>6s}  {'n':>4s}  {'CP':>4s}  {'CP ok':>7s}  {'Waring':>7s}  {'War ok':>7s}")
    print("-" * 46)

    cp_pass = cp_fail = cp_skip = 0
    war_pass = war_fail = war_skip = 0

    for tid in transform_ids:
        n, T_dense = compute_bco_tensor(tid)
        if n is None:
            print(f"{tid:06d}  {'?':>4s}  {'?':>4s}  {'skip':>7s}  {'?':>7s}  {'skip':>7s}")
            cp_skip += 1
            war_skip += 1
            continue

        # CP verification
        cpd_path = find_file(CPD_DIR, tid)
        if cpd_path is not None:
            cp_rank = parse_rank_from_path(cpd_path)
            _, cp_ok = verify_cpd(cpd_path, n, T_dense)
            cp_str = "PASS" if cp_ok else "FAIL"
            if cp_ok:
                cp_pass += 1
            else:
                cp_fail += 1
        else:
            cp_rank = None
            cp_str = "-"
            cp_skip += 1

        # Waring verification
        war_path = find_file(WARING_DIR, tid)
        if war_path is not None:
            war_rank = parse_rank_from_path(war_path)
            _, war_ok = verify_waring(war_path, n, T_dense)
            war_str = "PASS" if war_ok else "FAIL"
            if war_ok:
                war_pass += 1
            else:
                war_fail += 1
        else:
            war_rank = None
            war_str = "-"
            war_skip += 1

        cp_r = f"{cp_rank:4d}" if cp_rank is not None else "   -"
        war_r = f"{war_rank:7d}" if war_rank is not None else "      -"
        print(f"{tid:06d}  {n:4d}  {cp_r}  {cp_str:>7s}  {war_r}  {war_str:>7s}")

    elapsed = time.time() - t0
    print(f"\nCP:     {cp_pass} PASS, {cp_fail} FAIL, {cp_skip} skip")
    print(f"Waring: {war_pass} PASS, {war_fail} FAIL, {war_skip} skip")
    print(f"Time:   {elapsed:.0f}s")
    return 1 if (cp_fail > 0 or war_fail > 0) else 0


if __name__ == "__main__":
    exit(main())
