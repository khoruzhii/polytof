#!/usr/bin/env python3
"""
Generate base tensors for polynomial and GF(2^n) multiplication.

Creates three families of tensors for n = 2..10:

  03xx  (n, n, 2n-1)   Polynomial multiplication (convolution)
  05xx  (n, n, n)       GF(2^n) multiplication (mod irreducible polynomial)
  07xx  (3n, 3n, 3n)    GF(2^n) multiplication, phase tensor (topp embedding)

where xx = n + 23  (so n=2 -> 25, n=10 -> 33).

If a tensor already exists, it is verified; otherwise it is created.

Usage:
    python make_tensors_gfmult.py
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
TENSORS_DIR = REPO_DIR / "data" / "tensors"

# Primitive polynomials over GF(2), as lists of exponents (descending).
# P(x) = x^n + (lower terms), e.g. [2,1,0] means x^2 + x + 1.
PRIMITIVE_POLYS = {
    2: [2, 1, 0],
    3: [3, 1, 0],
    4: [4, 1, 0],
    5: [5, 2, 0],
    6: [6, 1, 0],
    7: [7, 1, 0],
    8: [8, 4, 3, 2, 0],
    9: [9, 4, 0],
    10: [10, 3, 0],
}


# ---------------------------------------------------------------------------
# Polynomial multiplication (convolution) tensor
# ---------------------------------------------------------------------------


def poly_mult_tensor(n):
    """Convolution tensor: c_k = sum_{i+j=k} a_i b_j, dims (n, n, 2n-1).

    Every (i, j) pair with i,j < n contributes triple (i, j, i+j).
    """
    triples = [(i, j, i + j) for i in range(n) for j in range(n)]
    dims = (n, n, 2 * n - 1)
    return dims, np.array(triples, dtype=np.int64)


# ---------------------------------------------------------------------------
# GF(2^n) multiplication tensor
# ---------------------------------------------------------------------------


def _build_reduction_table(n, exponents):
    """Build reduction rules: x^k -> set of exponents < n, for k = n .. 2n-2."""
    lower = set(e for e in exponents if e < n)
    reduction = {n: lower.copy()}
    for k in range(n + 1, 2 * n - 1):
        result = set()
        for e in reduction.get(k - 1, {k - 1}):
            new_exp = e + 1
            if new_exp >= n:
                result ^= reduction.get(new_exp, {new_exp})
            else:
                result ^= {new_exp}
        reduction[k] = result
    return reduction


def gf_mult_tensor(n):
    """GF(2^n) multiplication tensor: c_k = T_ijk a_i b_j (mod P(x)), dims (n, n, n).

    Polynomial multiplication followed by reduction modulo the irreducible polynomial.
    """
    reduction = _build_reduction_table(n, PRIMITIVE_POLYS[n])

    raw_triples = []
    for i in range(n):
        for j in range(n):
            m = i + j
            if m < n:
                raw_triples.append((i, j, m))
            else:
                for k in reduction[m]:
                    raw_triples.append((i, j, k))

    # Reduce mod 2 (pairs cancel)
    cnt = Counter(raw_triples)
    triples = sorted(t for t, c in cnt.items() if c % 2 == 1)

    dims = (n, n, n)
    return dims, np.array(triples, dtype=np.int64)


# ---------------------------------------------------------------------------
# Topp (phase tensor) embedding
# ---------------------------------------------------------------------------


def topp_embedding(triples, n):
    """Embed bilinear tensor (n, n, n) into symmetric phase tensor (3n, 3n, 3n).

    Maps (i, j, k) -> (i, n+j, 2n+k) and extracts strictly increasing triples
    (a < b < c) from the symmetrization.
    """
    m = 3 * n
    # The embedding places nonzeros at (i, n+j, 2n+k) for each (i,j,k).
    # After symmetrization over S_3, the 6 permutations produce distinct triples
    # since i < n <= n+j < 2n <= 2n+k, so all 6 permutations are distinct.
    # The strictly increasing triple from each is (i, n+j, 2n+k).
    embedded = [(i, n + j, 2 * n + k) for i, j, k in triples]
    dims = (m, m, m)
    return dims, np.array(embedded, dtype=np.int64)


# ---------------------------------------------------------------------------
# Sparse format I/O
# ---------------------------------------------------------------------------


def to_sparse(dims, triples):
    """Pack (dims, triples) into sparse .npy format: first row = dims, rest = triples."""
    out = np.empty((len(triples) + 1, 3), dtype=np.int64)
    out[0] = dims
    if len(triples) > 0:
        out[1:] = triples
    return out


def save_or_verify(tensor_id, dims, triples):
    """Save tensor if new, verify if existing. Returns status string."""
    data = to_sparse(dims, triples)
    path = TENSORS_DIR / f"{tensor_id:04d}.npy"

    if path.exists():
        existing = np.load(path)
        if np.array_equal(existing, data):
            return "match"
        else:
            return "MISMATCH"
    else:
        np.save(path, data)
        return "new"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    TENSORS_DIR.mkdir(parents=True, exist_ok=True)

    header = f"{'ID':>4}  {'Type':<10}  {'dims':<16}  {'nnz':>5}  Status"
    print(header)
    print("-" * len(header))

    errors = []

    for n in range(2, 11):
        xx = n + 23  # 25..33

        # --- 03xx: polynomial multiplication ---
        tid_poly = 300 + xx
        dims_p, triples_p = poly_mult_tensor(n)
        status = save_or_verify(tid_poly, dims_p, triples_p)
        print(
            f"{tid_poly:04d}  {'poly':<10}  "
            f"{str(dims_p):<16}  {len(triples_p):>5}  {status}"
        )
        if status == "MISMATCH":
            errors.append((tid_poly, "tensor mismatch"))

        # --- 05xx: GF(2^n) multiplication ---
        tid_gf = 500 + xx
        dims_g, triples_g = gf_mult_tensor(n)
        status = save_or_verify(tid_gf, dims_g, triples_g)
        print(
            f"{tid_gf:04d}  {'gf base':<10}  "
            f"{str(dims_g):<16}  {len(triples_g):>5}  {status}"
        )
        if status == "MISMATCH":
            errors.append((tid_gf, "tensor mismatch"))

        # --- 07xx: topp embedding ---
        tid_topp = 700 + xx
        dims_t, triples_t = topp_embedding(triples_g, n)
        status = save_or_verify(tid_topp, dims_t, triples_t)
        print(
            f"{tid_topp:04d}  {'gf topp':<10}  "
            f"{str(dims_t):<16}  {len(triples_t):>5}  {status}"
        )
        if status == "MISMATCH":
            errors.append((tid_topp, "tensor mismatch"))

    print("-" * len(header))
    print(f"Total: {3 * 9} tensors")

    if errors:
        print(f"\n{len(errors)} error(s):")
        for id_, err in errors:
            print(f"  {id_:04d}: {err}")
        sys.exit(1)

    print("Done.")


if __name__ == "__main__":
    main()
