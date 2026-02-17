"""
GF(2) linear algebra and tensor utilities shared across scripts.
"""

import numpy as np
from numba import njit


# ---------------------------------------------------------------------------
# GF(2) linear algebra
# ---------------------------------------------------------------------------


@njit(cache=True)
def gf2_inv(A):
    """Compute inverse of matrix A over GF(2) using Gauss-Jordan elimination."""
    n = A.shape[0]
    M = np.zeros((n, 2 * n), dtype=np.uint8)
    for i in range(n):
        for j in range(n):
            M[i, j] = A[i, j] & 1
        M[i, n + i] = 1
    for col in range(n):
        pivot = -1
        for row in range(col, n):
            if M[row, col] == 1:
                pivot = row
                break
        if pivot == -1:
            raise ValueError("Singular matrix over GF(2)")
        if pivot != col:
            for j in range(2 * n):
                M[col, j], M[pivot, j] = M[pivot, j], M[col, j]
        for row in range(n):
            if row != col and M[row, col] == 1:
                for j in range(2 * n):
                    M[row, j] = M[row, j] ^ M[col, j]
    return M[:, n:].copy()


@njit(cache=True)
def gf2_rank(M):
    """Rank of a binary matrix over GF(2)."""
    nr, nc = M.shape
    A = np.empty((nr, nc), dtype=np.uint8)
    for i in range(nr):
        for j in range(nc):
            A[i, j] = M[i, j] & 1
    r = 0
    for c in range(nc):
        pivot = -1
        for i in range(r, nr):
            if A[i, c]:
                pivot = i
                break
        if pivot == -1:
            continue
        if pivot != r:
            for j in range(nc):
                A[r, j], A[pivot, j] = A[pivot, j], A[r, j]
        for i in range(nr):
            if i != r and A[i, c]:
                for j in range(nc):
                    A[i, j] = A[i, j] ^ A[r, j]
        r += 1
        if r == nr:
            break
    return r


# ---------------------------------------------------------------------------
# Tensor I/O
# ---------------------------------------------------------------------------


def load_tensor_triples(path):
    """Load tensor from .npy path as (n, triples_array).

    Returns (None, None) if the file does not exist.
    """
    if not path.exists():
        return None, None
    data = np.load(path)
    n = int(data[0, 0])
    triples = data[1:]
    return n, triples


def build_dense_tensor(n, triples):
    """Build dense symmetric (n, n, n) tensor from sparse triples over GF(2).

    Triples are stored with i < j < k.  All 6 permutations are set.
    """
    T = np.zeros((n, n, n), dtype=np.uint8)
    idx = triples.astype(np.intp)
    T[idx[:, 0], idx[:, 1], idx[:, 2]] = 1
    T[idx[:, 0], idx[:, 2], idx[:, 1]] = 1
    T[idx[:, 1], idx[:, 0], idx[:, 2]] = 1
    T[idx[:, 1], idx[:, 2], idx[:, 0]] = 1
    T[idx[:, 2], idx[:, 0], idx[:, 1]] = 1
    T[idx[:, 2], idx[:, 1], idx[:, 0]] = 1
    return T


def load_tensor_dense(path):
    """Load tensor from .npy path as dense (n, n, n) symmetric GF(2) array.

    Returns (None, None) if the file does not exist.
    """
    n, triples = load_tensor_triples(path)
    if n is None:
        return None, None
    return n, build_dense_tensor(n, triples)


# ---------------------------------------------------------------------------
# Basis change
# ---------------------------------------------------------------------------


def apply_inverse_transform(T, A_inv):
    """Apply A_inv on all three modes of dense tensor, mod 2.

    Computes T'[i',j',k'] = sum_{a,b,c} A_inv[i',a] A_inv[j',b] A_inv[k',c] T[a,b,c].
    """
    A = A_inv.astype(np.int32)
    return np.einsum("ia,jb,kc,abc->ijk", A, A, A, T.astype(np.int32),
                     optimize=True) % 2


# ---------------------------------------------------------------------------
# Triple extraction
# ---------------------------------------------------------------------------


def upper_triples(T):
    """Extract set of (i, j, k) with i < j < k and T[i,j,k] odd."""
    coords = np.argwhere(T % 2)
    if len(coords) == 0:
        return set()
    mask = (coords[:, 0] < coords[:, 1]) & (coords[:, 1] < coords[:, 2])
    return set(map(tuple, coords[mask].tolist()))


# ---------------------------------------------------------------------------
# Bitvector decoding
# ---------------------------------------------------------------------------


def bitvecs_to_dense(word_matrix, n):
    """Convert array of uint64 bitvectors to dense binary matrix.

    word_matrix: shape (m, vec_words) of uint64
    Returns: shape (m, n) of uint8, each row is a binary vector of length n.
    """
    pos = np.arange(n, dtype=np.uint64)
    return ((word_matrix[:, pos // 64] >> (pos % 64)) & 1).astype(np.uint8)
