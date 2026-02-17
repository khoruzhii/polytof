#!/usr/bin/env python3
"""
Build known-rank CPD schemes for GF(2^n) field multiplication (n = 2..10).

Each scheme is a CP decomposition of the field multiplication tensor into
binary rank-1 terms.  The schemes are constructed from algebraic tower-field
decompositions and converted to the standard polynomial basis via explicit
field isomorphisms.

Usage:
    python make_cpd_gf.py       # build all n = 2..10

Output:
    data/cpd/base/{BASE_ID}-{RANK}.npy   (n x n x n bilinear tensor CPD)
    data/cpd/topp/{TOPP_ID}-{RANK}.npy   (3n x 3n x 3n phase tensor CPD)

Tensor IDs follow the convention tid = n + 23:
    base_id = 500 + tid,  topp_id = 700 + tid.
"""

import numpy as np
from pathlib import Path

from gf2 import gf2_inv, gf2_rank

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
TENSOR_DIR = REPO_DIR / "data" / "tensors"
CPD_BASE_DIR = REPO_DIR / "data" / "cpd" / "base"
CPD_TOPP_DIR = REPO_DIR / "data" / "cpd" / "topp"

# ---------------------------------------------------------------------------
# Primitive polynomials (exponent lists, including n and 0)
# ---------------------------------------------------------------------------

PRIMITIVE_POLYS = {
    2:  [2, 1, 0],          # x^2  + x   + 1
    3:  [3, 1, 0],          # x^3  + x   + 1
    4:  [4, 1, 0],          # x^4  + x   + 1
    5:  [5, 2, 0],          # x^5  + x^2 + 1
    6:  [6, 1, 0],          # x^6  + x   + 1
    7:  [7, 1, 0],          # x^7  + x   + 1
    8:  [8, 4, 3, 2, 0],    # x^8  + x^4 + x^3 + x^2 + 1  (AES)
    9:  [9, 4, 0],          # x^9  + x^4 + 1
    10: [10, 3, 0],         # x^10 + x^3 + 1
}

# Expected ranks from multiplicative complexity bounds
EXPECTED_RANKS = {
    2: 3, 3: 6, 4: 9, 5: 13, 6: 15, 7: 22, 8: 24, 9: 30, 10: 33,
}

# ===========================================================================
#  GF(2) helpers
# ===========================================================================

def vecN_to_bits(v, n):
    """n-bit integer -> length-n binary vector (LSB first)."""
    return np.array([(v >> i) & 1 for i in range(n)], dtype=np.uint8)


def mat_cols_from_ints(ints, n):
    """List of n-bit integers -> n x len matrix (columns = bit vectors, LSB first)."""
    return np.column_stack([vecN_to_bits(x, n) for x in ints]).astype(np.uint8)


# ===========================================================================
#  GF(2^n) standard arithmetic (polynomial basis)
# ===========================================================================

def _mod_poly_int(n):
    poly = 0
    for e in PRIMITIVE_POLYS[n]:
        poly |= 1 << e
    return poly


def gf2n_mul(a, b, n):
    """Multiply a, b in GF(2^n) using PRIMITIVE_POLYS[n]."""
    mod = _mod_poly_int(n)
    red = mod ^ (1 << n)
    mask = (1 << n) - 1
    a &= mask
    b &= mask
    res = 0
    for _ in range(n):
        if b & 1:
            res ^= a
        carry = a & (1 << (n - 1))
        a = (a << 1) & mask
        if carry:
            a ^= red
        b >>= 1
    return res & mask


def poly_eval_gf2n(coeffs, x, n):
    """Evaluate sum_i coeffs[i] * x^i in GF(2^n)."""
    res = 0
    xp = 1
    for c in coeffs:
        if c & 1:
            res ^= xp
        xp = gf2n_mul(xp, x, n)
    return res


def reduction_matrix(n):
    """n x (2n-1) GF(2) matrix for x^k mod h(x), k = 0..2n-2."""
    exps = PRIMITIVE_POLYS[n]
    lower = {e for e in exps if e < n}
    red = {n: lower.copy()}
    for k in range(n + 1, 2 * n - 1):
        red[k] = set()
        for e in red[k - 1]:
            e1 = e + 1
            if e1 < n:
                red[k] ^= {e1}
            else:
                red[k] ^= red.get(e1, {e1})
    R = np.eye(n, 2 * n - 1, dtype=np.uint8)
    for k in range(n, 2 * n - 1):
        for m in red[k]:
            R[m, k] = 1
    return R


# ===========================================================================
#  GF(4) arithmetic (2-bit):  w^2 = w + 1
# ===========================================================================

W4 = 2       # w
W4_2 = 3     # w^2 = w + 1


def gf4_mul(a, b):
    a0, a1 = a & 1, (a >> 1) & 1
    b0, b1 = b & 1, (b >> 1) & 1
    c0 = a0 & b0
    c1 = (a0 & b1) ^ (a1 & b0)
    c2 = a1 & b1
    return ((c0 ^ c2) | ((c1 ^ c2) << 1)) & 3


GF4_MUL = np.zeros((4, 4), dtype=np.uint8)
for _a in range(4):
    for _b in range(4):
        GF4_MUL[_a, _b] = gf4_mul(_a, _b)


def gf4_add(a, b):
    return (a ^ b) & 3


def gf4_const_mat(c):
    """2 x 2 binary matrix for multiplication by constant c in GF(4)."""
    if c == 0:
        return np.zeros((2, 2), dtype=np.uint8)
    if c == 1:
        return np.eye(2, dtype=np.uint8)
    if c == W4:
        return np.array([[0, 1], [1, 1]], dtype=np.uint8)
    if c == W4_2:
        return np.array([[1, 1], [1, 0]], dtype=np.uint8)
    raise ValueError(c)


# ===========================================================================
#  GF(8) arithmetic (3-bit):  x^3 + x + 1
# ===========================================================================

GF8_MOD = 0xB  # 0b1011


def gf8_add(a, b):
    return (a ^ b) & 7


def gf8_mul(a, b):
    a &= 7
    b &= 7
    res = 0
    aa, bb = a, b
    for _ in range(6):
        if bb & 1:
            res ^= aa
        bb >>= 1
        aa <<= 1
        if aa & 0b1000:
            aa ^= GF8_MOD
    return res & 7


def gf8_pow(a, e):
    r = 1
    for _ in range(e):
        r = gf8_mul(r, a)
    return r


def gf8_const_mat(c):
    """3 x 3 binary matrix for multiplication by constant c in GF(8)."""
    c &= 7
    M = np.zeros((3, 3), dtype=np.uint8)
    for j in range(3):
        w = gf8_mul(c, 1 << j)
        M[:, j] = vecN_to_bits(w, 3)
    return M


# ===========================================================================
#  Generic field isomorphism:  tower basis A -> standard GF(2^n)
# ===========================================================================

def _find_degree_n_element(mulA, one, n):
    """Find g in field A such that {1, g, ..., g^{n-1}} spans GF(2)^n."""
    for g in range(2, 1 << n):
        powers = [one]
        cur = one
        for _ in range(1, n):
            cur = mulA(cur, g)
            powers.append(cur)
        P = mat_cols_from_ints(powers, n)
        if gf2_rank(P) == n:
            return g
    raise RuntimeError("Degree-n element not found.")


def _minimal_polynomial(mulA, one, n, g):
    """Minimal polynomial of g in field A as [c0, ..., cn] with cn = 1."""
    powers = [one]
    cur = one
    for _ in range(1, n + 1):
        cur = mulA(cur, g)
        powers.append(cur)
    P_A = mat_cols_from_ints(powers[:n], n)
    invP_A = gf2_inv(P_A)
    rhs = vecN_to_bits(powers[n], n)
    coeffs = (invP_A @ rhs) % 2
    return [int(coeffs[i]) for i in range(n)] + [1]


def build_isomorphism(mulA, one, n):
    """
    Return n x n binary matrix M:  vec_standard = M @ vec_A.

    mulA(a, b) -> int:  multiplication in field A (both inputs/output are
                         n-bit packed integers).
    one:                 multiplicative identity in A.
    """
    gA = _find_degree_n_element(mulA, one, n)
    f = _minimal_polynomial(mulA, one, n, gA)

    roots = [x for x in range(1 << n) if poly_eval_gf2n(f, x, n) == 0]
    if not roots:
        raise RuntimeError("No root found in standard field.")
    gB = roots[0]

    powA = [one]
    cur = one
    for _ in range(1, n):
        cur = mulA(cur, gA)
        powA.append(cur)

    powB = [1]
    cur = 1
    for _ in range(1, n):
        cur = gf2n_mul(cur, gB, n)
        powB.append(cur)

    P_A = mat_cols_from_ints(powA, n)
    P_B = mat_cols_from_ints(powB, n)
    return (P_B @ gf2_inv(P_A)) % 2


# ===========================================================================
#  Inner Karatsuba helpers
# ===========================================================================

# Karatsuba GF(4) -> GF(2): rank 3
INNER_U_GF4 = [
    np.array([1, 0], dtype=np.uint8),
    np.array([0, 1], dtype=np.uint8),
    np.array([1, 1], dtype=np.uint8),
]
INNER_V_GF4 = INNER_U_GF4
INNER_W_GF4 = [
    np.array([1, 1], dtype=np.uint8),  # m0
    np.array([1, 0], dtype=np.uint8),  # m1
    np.array([0, 1], dtype=np.uint8),  # m2
]

# GF(8) -> GF(2): rank 6 (same as scheme_n3 from n2_to_n7)
INNER_U_GF8 = [
    np.array([1, 0, 0], dtype=np.uint8),
    np.array([0, 1, 0], dtype=np.uint8),
    np.array([0, 0, 1], dtype=np.uint8),
    np.array([1, 1, 0], dtype=np.uint8),
    np.array([1, 0, 1], dtype=np.uint8),
    np.array([0, 1, 1], dtype=np.uint8),
]
INNER_V_GF8 = INNER_U_GF8
INNER_W_GF8 = [
    np.array([1, 1, 1], dtype=np.uint8),
    np.array([1, 0, 1], dtype=np.uint8),
    np.array([1, 0, 0], dtype=np.uint8),
    np.array([0, 1, 0], dtype=np.uint8),
    np.array([0, 0, 1], dtype=np.uint8),
    np.array([1, 1, 0], dtype=np.uint8),
]


def _expand_tower(n_outer, inner_rank, inner_u, inner_v, inner_w,
                  const_mat_fn, inner_dim, Xs, Ys, out_const, n_out_coeffs):
    """
    Expand an outer-rank scheme over an extension field into binary CPD.

    Parameters
    ----------
    n_outer      : number of outer products (e.g. 3 for Karatsuba)
    inner_rank   : rank of inner GF(q)->GF(2) multiplication (e.g. 3 for GF(4))
    inner_u/v/w  : inner factor vectors
    const_mat_fn : function(c) -> binary matrix for multiplication by scalar c
    inner_dim    : dimension of inner field over GF(2) (e.g. 2 for GF(4))
    Xs, Ys       : list of inner_dim x total_dim binary matrices (evaluation)
    out_const    : n_out_coeffs x n_outer matrix of extension-field scalars
    n_out_coeffs : number of output coefficients

    Returns (U, V, W) binary matrices, shape (n_outer * inner_rank) x total_dim.
    """
    total_dim = Xs[0].shape[1]
    U_rows, V_rows, W_rows = [], [], []
    for i in range(n_outer):
        Xi, Yi = Xs[i], Ys[i]
        for t in range(inner_rank):
            urow = (inner_u[t] @ Xi) & 1
            vrow = (inner_v[t] @ Yi) & 1
            wrow = np.zeros(n_out_coeffs * inner_dim, dtype=np.uint8)
            for j in range(n_out_coeffs):
                c = out_const[j][i]
                if c == 0:
                    continue
                contrib = (const_mat_fn(c) @ inner_w[t].reshape(-1, 1))[:, 0] & 1
                wrow[inner_dim * j: inner_dim * (j + 1)] ^= contrib
            U_rows.append(urow)
            V_rows.append(vrow)
            W_rows.append(wrow)
    return (np.vstack(U_rows).astype(np.uint8),
            np.vstack(V_rows).astype(np.uint8),
            np.vstack(W_rows).astype(np.uint8))


# ===========================================================================
#  Scheme builders:  n = 2 .. 10
# ===========================================================================

def scheme_n2():
    """Karatsuba, rank 3, mod x^2 + x + 1."""
    U = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.uint8)
    V = U.copy()
    W = np.array([[1, 1], [1, 0], [0, 1]], dtype=np.uint8)
    return U, V, W


def scheme_n3():
    """Rank 6, mod x^3 + x + 1."""
    U = np.array([
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 0], [1, 0, 1], [0, 1, 1],
    ], dtype=np.uint8)
    V = U.copy()
    W = np.array([
        [1, 1, 1], [1, 0, 1], [1, 0, 0],
        [0, 1, 0], [0, 0, 1], [1, 1, 0],
    ], dtype=np.uint8)
    return U, V, W


def scheme_n4():
    """Rank 9 via GF(4)[z]/(z^2 + z + w), outer rank 3, inner rank 3."""
    n = 4

    def lin2x4(coeffs):
        X = np.zeros((2, n), dtype=np.uint8)
        for i, c in enumerate(coeffs):
            if c:
                X[:, 2 * i: 2 * i + 2] ^= gf4_const_mat(int(c))
        return X

    Xs = [lin2x4([1, 0]), lin2x4([0, 1]), lin2x4([1, 1])]
    Ys = Xs
    out_const = [
        [1, W4, 0],  # c0
        [1, 0, 1],   # c1
    ]

    U_A, V_A, W_A = _expand_tower(
        3, 3, INNER_U_GF4, INNER_V_GF4, INNER_W_GF4,
        gf4_const_mat, 2, Xs, Ys, out_const, 2)

    # Multiplication in tower field A = GF(4)[z]/(z^2 + z + w)
    def mulA(a, b):
        a0, a1 = (a >> 0) & 3, (a >> 2) & 3
        b0, b1 = (b >> 0) & 3, (b >> 2) & 3
        m0 = gf4_mul(a0, b0)
        m1 = gf4_mul(a1, b1)
        cross = gf4_add(gf4_mul(a0, b1), gf4_mul(a1, b0))
        c0 = gf4_add(m0, gf4_mul(W4, m1))
        c1 = gf4_add(cross, m1)
        return (c0 & 3) | ((c1 & 3) << 2)

    M = build_isomorphism(mulA, 1, n)
    Minv = gf2_inv(M)
    U = (U_A @ Minv) % 2
    V = (V_A @ Minv) % 2
    W = (W_A @ M.T) % 2
    return U.astype(np.uint8), V.astype(np.uint8), W.astype(np.uint8)


def scheme_n5():
    """M2(5) = 13, polynomial multiplication then reduction mod x^5 + x^2 + 1."""
    forms = [
        ([1,0,0,0,0], [1,0,0,0,0]),
        ([0,1,0,0,0], [0,1,0,0,0]),
        ([0,0,1,0,0], [0,0,1,0,0]),
        ([0,0,0,1,0], [0,0,0,1,0]),
        ([0,0,0,0,1], [0,0,0,0,1]),
        ([1,1,0,0,0], [1,1,0,0,0]),
        ([1,0,1,0,0], [1,0,1,0,0]),
        ([0,0,1,0,1], [0,0,1,0,1]),
        ([0,0,0,1,1], [0,0,0,1,1]),
        ([1,0,1,1,0], [1,0,1,1,0]),
        ([0,1,1,0,1], [0,1,1,0,1]),
        ([1,1,0,1,1], [1,1,0,1,1]),
        ([1,1,1,1,1], [1,1,1,1,1]),
    ]
    U = np.array([u for u, _ in forms], dtype=np.uint8)
    V = np.array([v for _, v in forms], dtype=np.uint8)

    c_sets = [
        {1},
        {6, 1, 2},
        {7, 1, 3, 2},
        {1, 13, 12, 10, 8, 3, 5, 4},
        {6, 1, 2, 13, 10, 11, 9, 5, 4},
        {7, 1, 3, 2, 13, 11, 12, 5},
        {8, 3, 5, 4},
        {9, 4, 5},
        {5},
    ]
    W_full = np.zeros((13, 9), dtype=np.uint8)
    for k, s in enumerate(c_sets):
        for mi in s:
            W_full[mi - 1, k] ^= 1

    R = reduction_matrix(5)
    W = (W_full @ R.T) % 2
    return U, V, W.astype(np.uint8)


def scheme_n6():
    """Rank 15 via GF(4)[z]/(z^3 + z + 1), outer rank 5, inner rank 3."""
    n = 6

    def lin2x6(coeffs):
        X = np.zeros((2, n), dtype=np.uint8)
        for i, c in enumerate(coeffs):
            if c:
                X[:, 2 * i: 2 * i + 2] ^= gf4_const_mat(int(c))
        return X

    points = [0, 1, W4, W4_2, "inf"]

    def coeffs_for_t(t):
        if t == "inf":
            return [0, 0, 1]
        return [1, t, gf4_mul(t, t)]

    Xs = [lin2x6(coeffs_for_t(t)) for t in points]
    Ys = Xs
    out_const = [
        [0, 1, 1, 1, 0],
        [1, 0, W4, W4_2, 0],
        [0, 1, W4, W4_2, 1],
    ]

    U_A, V_A, W_A = _expand_tower(
        5, 3, INNER_U_GF4, INNER_V_GF4, INNER_W_GF4,
        gf4_const_mat, 2, Xs, Ys, out_const, 3)

    # Multiplication in A = GF(4)[z]/(z^3 + z + 1)
    def mulA(a, b):
        A_c = [(a >> (2 * i)) & 3 for i in range(3)]
        B_c = [(b >> (2 * i)) & 3 for i in range(3)]
        prod = [0] * 5
        for i, ai in enumerate(A_c):
            for j, bj in enumerate(B_c):
                prod[i + j] = gf4_add(prod[i + j], gf4_mul(ai, bj))
        for k in range(4, 2, -1):
            coeff = prod[k]
            if coeff:
                shift = k - 3
                prod[shift] = gf4_add(prod[shift], coeff)
                prod[shift + 1] = gf4_add(prod[shift + 1], coeff)
            prod.pop()
        return (prod[0] & 3) | ((prod[1] & 3) << 2) | ((prod[2] & 3) << 4)

    M = build_isomorphism(mulA, 1, n)
    Minv = gf2_inv(M)
    U = (U_A @ Minv) % 2
    V = (V_A @ Minv) % 2
    W = (W_A @ M.T) % 2
    return U.astype(np.uint8), V.astype(np.uint8), W.astype(np.uint8)


def scheme_n7():
    """M2(7) = 22, CRT scheme, then reduction mod x^7 + x + 1."""
    def uvec(*idxs):
        v = np.zeros(7, dtype=np.uint8)
        for i in idxs:
            v[i] ^= 1
        return v

    forms = [
        (uvec(0), uvec(0)),
        (uvec(1), uvec(1)),
        (uvec(0, 1), uvec(0, 1)),
        (uvec(0,1,2,3,4,5,6), uvec(0,1,2,3,4,5,6)),
        (uvec(1,3,5), uvec(1,3,5)),
        (uvec(0,2,4,6), uvec(0,2,4,6)),
        (uvec(0,2,3,5,6), uvec(0,2,3,5,6)),
        (uvec(1,2,4,5), uvec(1,2,4,5)),
        (uvec(0,1,3,4,6), uvec(0,1,3,4,6)),
        (uvec(0,3,5,6), uvec(0,3,5,6)),
        (uvec(1,3,4,5), uvec(1,3,4,5)),
        (uvec(2,4,5,6), uvec(2,4,5,6)),
        (uvec(0,1,4,6), uvec(0,1,4,6)),
        (uvec(0,2,3,4), uvec(0,2,3,4)),
        (uvec(1,2,3,6), uvec(1,2,3,6)),
        (uvec(0,3,4,5), uvec(0,3,4,5)),
        (uvec(1,4,5,6), uvec(1,4,5,6)),
        (uvec(2,3,4,6), uvec(2,3,4,6)),
        (uvec(0,1,3,6), uvec(0,1,3,6)),
        (uvec(0,2,5,6), uvec(0,2,5,6)),
        (uvec(1,2,3,5), uvec(1,2,3,5)),
        (uvec(6), uvec(6)),
    ]
    U = np.vstack([u for u, _ in forms]).astype(np.uint8)
    V = np.vstack([v for _, v in forms]).astype(np.uint8)

    def e(i):
        v = np.zeros(22, dtype=np.uint8)
        v[i - 1] = 1
        return v

    R0 = e(1);  R1 = e(1) ^ e(2) ^ e(3)
    S0 = e(4);  S1 = e(4) ^ e(5) ^ e(6)
    T0 = e(7) ^ e(8);  T1 = e(7) ^ e(9)

    p0 = e(10);  p1 = e(13) ^ e(10) ^ e(11)
    p2 = e(14) ^ e(10) ^ e(11) ^ e(12)
    p3 = e(15) ^ e(11) ^ e(12);  p4 = e(12)
    U0 = p0 ^ p3;  U1 = p1 ^ p3 ^ p4;  U2 = p2 ^ p4

    q0 = e(16);  q1 = e(19) ^ e(16) ^ e(17)
    q2 = e(20) ^ e(16) ^ e(17) ^ e(18)
    q3 = e(21) ^ e(17) ^ e(18);  q4 = e(18)
    V0 = q0 ^ q3 ^ q4;  V1 = q1 ^ q4;  V2 = q2 ^ q3 ^ q4

    L = e(22)

    c = [
        R0,
        R1,
        S0 ^ S1 ^ T0 ^ U1 ^ U2 ^ V1 ^ L,
        R0 ^ S1 ^ T0 ^ T1 ^ U0 ^ U2 ^ V0 ^ V2,
        R1 ^ S0 ^ S1 ^ T1 ^ U1 ^ V0 ^ V1,
        S0 ^ U1 ^ V0 ^ V2 ^ L,
        S0 ^ U2 ^ V0 ^ V1,
        R0 ^ S0 ^ U0 ^ V0 ^ V1 ^ V2,
        R1 ^ S0 ^ U0 ^ U1 ^ V1 ^ V2,
        S1 ^ T0 ^ U0 ^ V1 ^ V2 ^ L,
        R0 ^ S0 ^ S1 ^ T0 ^ T1 ^ U0 ^ U1 ^ V2,
        R1 ^ S1 ^ T1 ^ U0 ^ U1 ^ U2 ^ V0,
        L,
    ]

    W_full = np.zeros((22, 13), dtype=np.uint8)
    for k in range(13):
        W_full[:, k] = c[k]

    R = reduction_matrix(7)
    W = (W_full @ R.T) % 2
    return U, V, W.astype(np.uint8)


def scheme_n8():
    """Rank 24 via GF(4)^4 tower, outer rank 8, inner rank 3."""
    n = 8

    const_x = [
        [W4, W4_2, 0,   0],
        [0,  W4,   0,   W4],
        [0,  W4,   W4,  0],
        [1,  0,    0,   1],
        [1,  0,    1,   0],
        [0,  0,    1,   0],
        [0,  0,    0,   1],
        [W4, W4,   1,   W4_2],
    ]

    out_const = [[0] * 8 for _ in range(4)]
    # c0 = w p2 + p3 + p6 + p7 + w^2 p8
    out_const[0][1] = W4;  out_const[0][2] = 1
    out_const[0][5] = 1;   out_const[0][6] = 1;  out_const[0][7] = W4_2
    # c1 = p5 + w p6 + w p7 + w^2 p8
    out_const[1][4] = 1;   out_const[1][5] = W4
    out_const[1][6] = W4;  out_const[1][7] = W4_2
    # c2 = w^2 p1 + w p2 + w^2 p3 + w^2 p4 + w p5 + w p6
    out_const[2][0] = W4_2; out_const[2][1] = W4;  out_const[2][2] = W4_2
    out_const[2][3] = W4_2; out_const[2][4] = W4;  out_const[2][5] = W4
    # c3 = p1 + w^2 p2 + w p3 + w^2 p4 + p7 + w^2 p8
    out_const[3][0] = 1;    out_const[3][1] = W4_2; out_const[3][2] = W4
    out_const[3][3] = W4_2; out_const[3][6] = 1;    out_const[3][7] = W4_2

    def build_lin2x8(row):
        X = np.zeros((2, n), dtype=np.uint8)
        for k, c in enumerate(row):
            if c:
                X[:, 2 * k: 2 * k + 2] ^= gf4_const_mat(c)
        return X

    Xs = [build_lin2x8(row) for row in const_x]
    Ys = Xs

    U_A, V_A, W_A = _expand_tower(
        8, 3, INNER_U_GF4, INNER_V_GF4, INNER_W_GF4,
        gf4_const_mat, 2, Xs, Ys, out_const, 4)

    # Multiplication in tower field A = GF(4)^4
    def unpack(x):
        return [(x >> (2 * k)) & 3 for k in range(4)]

    def pack(v):
        return sum((int(c) & 3) << (2 * k) for k, c in enumerate(v)) & 0xFF

    def mul_A_vec(a, b):
        a0, a1, a2, a3 = a
        b0, b1, b2, b3 = b
        x1 = gf4_add(GF4_MUL[W4, a0], GF4_MUL[W4_2, a1])
        x2 = gf4_add(GF4_MUL[W4, a1], GF4_MUL[W4, a3])
        x3 = gf4_add(GF4_MUL[W4, a1], GF4_MUL[W4, a2])
        x4 = gf4_add(a0, a3)
        x5 = gf4_add(a0, a2)
        x6, x7 = a2, a3
        x8 = gf4_add(gf4_add(GF4_MUL[W4, a0], GF4_MUL[W4, a1]),
                      gf4_add(a2, GF4_MUL[W4_2, a3]))

        y1 = gf4_add(GF4_MUL[W4, b0], GF4_MUL[W4_2, b1])
        y2 = gf4_add(GF4_MUL[W4, b1], GF4_MUL[W4, b3])
        y3 = gf4_add(GF4_MUL[W4, b1], GF4_MUL[W4, b2])
        y4 = gf4_add(b0, b3)
        y5 = gf4_add(b0, b2)
        y6, y7 = b2, b3
        y8 = gf4_add(gf4_add(GF4_MUL[W4, b0], GF4_MUL[W4, b1]),
                      gf4_add(b2, GF4_MUL[W4_2, b3]))

        xs = [x1, x2, x3, x4, x5, x6, x7, x8]
        ys = [y1, y2, y3, y4, y5, y6, y7, y8]
        ps = [GF4_MUL[xs[i], ys[i]] for i in range(8)]
        p1, p2, p3, p4, p5, p6, p7, p8 = ps

        c0 = gf4_add(gf4_add(gf4_add(gf4_add(
            GF4_MUL[W4, p2], p3), p6), p7), GF4_MUL[W4_2, p8])
        c1 = gf4_add(gf4_add(p5, GF4_MUL[W4, p6]),
                      gf4_add(GF4_MUL[W4, p7], GF4_MUL[W4_2, p8]))
        c2 = gf4_add(
            gf4_add(gf4_add(gf4_add(gf4_add(
                GF4_MUL[W4_2, p1], GF4_MUL[W4, p2]),
                GF4_MUL[W4_2, p3]), GF4_MUL[W4_2, p4]),
                GF4_MUL[W4, p5]),
            GF4_MUL[W4, p6])
        c3 = gf4_add(
            gf4_add(gf4_add(gf4_add(p1, GF4_MUL[W4_2, p2]),
                            GF4_MUL[W4, p3]), GF4_MUL[W4_2, p4]),
            gf4_add(p7, GF4_MUL[W4_2, p8]))
        return [c0, c1, c2, c3]

    def mulA(a, b):
        return pack(mul_A_vec(unpack(a), unpack(b)))

    # Find identity in A
    eid = None
    for e_cand in range(256):
        ok = True
        for a in range(256):
            if mulA(e_cand, a) != a or mulA(a, e_cand) != a:
                ok = False
                break
        if ok:
            eid = e_cand
            break
    if eid is None:
        raise RuntimeError("Identity not found in field A (n=8).")

    M = build_isomorphism(mulA, eid, n)
    Minv = gf2_inv(M)
    U = (U_A @ Minv) % 2
    V = (V_A @ Minv) % 2
    W = (W_A @ M.T) % 2
    return U.astype(np.uint8), V.astype(np.uint8), W.astype(np.uint8)


def scheme_n9():
    """Rank 30 via GF(8)[z]/(z^3 + alpha*z + 1), outer rank 5, inner rank 6."""
    n = 9
    ALPHA = 2  # primitive element in GF(8)

    def pack3(c):
        return sum((int(v) & 7) << (3 * i) for i, v in enumerate(c)) & 0x1FF

    def unpack3(x):
        return [(x >> (3 * i)) & 7 for i in range(3)]

    def reduce_Q(poly):
        """Reduce mod Q(z) = z^3 + alpha*z + 1."""
        poly = poly[:]
        while len(poly) > 3:
            k = len(poly) - 1
            coeff = poly[k]
            if coeff:
                shift = k - 3
                poly[shift] = gf8_add(poly[shift], coeff)
                poly[shift + 1] = gf8_add(poly[shift + 1], gf8_mul(coeff, ALPHA))
            poly.pop()
        poly += [0] * (3 - len(poly))
        return poly

    def mulA(a, b):
        ac = unpack3(a)
        bc = unpack3(b)
        prod = [0] * 5
        for i, ai in enumerate(ac):
            for j, bj in enumerate(bc):
                prod[i + j] = gf8_add(prod[i + j], gf8_mul(ai, bj))
        return pack3(reduce_Q(prod))

    # Evaluation at t in {0, 1, alpha, alpha^2, inf}
    T2 = ALPHA
    T3 = gf8_mul(ALPHA, ALPHA)
    points = [0, 1, T2, T3, "inf"]

    # Reconstruction constants
    K = [
        [7, 7, 5, 4, 7],  # r0
        [4, 7, 3, 0, 7],  # r1
        [4, 4, 7, 7, 7],  # r2
    ]

    def eval_lin_matrix(t):
        X = np.zeros((3, n), dtype=np.uint8)
        if t == "inf":
            X[:, 6:9] = np.eye(3, dtype=np.uint8)
            return X
        t2 = gf8_mul(t, t)
        X[:, 0:3] ^= np.eye(3, dtype=np.uint8)
        X[:, 3:6] ^= gf8_const_mat(t)
        X[:, 6:9] ^= gf8_const_mat(t2)
        return X

    Xs = [eval_lin_matrix(t) for t in points]
    Ys = Xs

    U_A, V_A, W_A = _expand_tower(
        5, 6, INNER_U_GF8, INNER_V_GF8, INNER_W_GF8,
        gf8_const_mat, 3, Xs, Ys, K, 3)

    one = pack3([1, 0, 0])
    M = build_isomorphism(mulA, one, n)
    Minv = gf2_inv(M)
    U = (U_A @ Minv) % 2
    V = (V_A @ Minv) % 2
    W = (W_A @ M.T) % 2
    return U.astype(np.uint8), V.astype(np.uint8), W.astype(np.uint8)


def scheme_n10():
    """Rank 33 via GF(4)[z]/(P(z)), P = z^5 + w z^4 + z^3 + w^2 z^2 + w z + 1,
    outer rank 11, inner rank 3."""
    n = 10

    P_COEFFS = [1, W4, W4_2, 1, W4, 1]  # P(z) coefficients, low to high

    def pack5(c):
        out = 0
        for i, v in enumerate(c):
            out |= ((int(v) & 1) << (2 * i)) | ((int(v) >> 1 & 1) << (2 * i + 1))
        return out & 0x3FF

    def unpack5(x):
        return [((x >> (2 * k)) & 1) | (((x >> (2 * k + 1)) & 1) << 1) for k in range(5)]

    def poly_mod_P(poly):
        poly = poly[:]
        while len(poly) > 5:
            k = len(poly) - 1
            coeff = poly[k]
            if coeff:
                shift = k - 5
                for i in range(5):
                    poly[i + shift] = gf4_add(poly[i + shift], gf4_mul(coeff, P_COEFFS[i]))
            poly.pop()
        poly += [0] * (5 - len(poly))
        return poly

    def mulA(a, b):
        ac = unpack5(a)
        bc = unpack5(b)
        prod = [0] * (len(ac) + len(bc) - 1)
        for i, ai in enumerate(ac):
            for j, bj in enumerate(bc):
                prod[i + j] = gf4_add(prod[i + j], gf4_mul(ai, bj))
        return pack5(poly_mod_P(prod))

    X_coeffs = [
        [1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [1, W4, W4_2, 1, W4],
        [1, W4_2, W4, 1, W4_2],
        [0, 0, 0, 0, 1],
        [1, 0, W4, W4, 1],
        [0, 1, 1, W4_2, 1],
        [1, 1, W4_2, 1, 0],
        [1, 0, W4_2, W4_2, 1],
        [0, 1, 1, W4, 1],
        [1, 1, W4, 1, 0],
    ]

    out_const = [
        [W4_2, 1,  W4,  1,  W4,  0,  1,  W4_2, 1,  W4,  W4],
        [0,    0,  0,   1,  0,   1,  1,  W4,   W4, 1,   0],
        [W4_2, W4_2, 1, 1,  1,   0,  W4, 1,    W4, W4_2, W4_2],
        [0,    1,  W4_2, W4, 1, W4_2, W4, W4,  W4, W4_2, W4_2],
        [W4,   W4_2, 1, 1,  W4_2, W4_2, 1, 0,  0,  1,   W4],
    ]

    def build_lin2x10(row):
        X = np.zeros((2, n), dtype=np.uint8)
        for i, c in enumerate(row):
            if c:
                X[:, 2 * i: 2 * i + 2] ^= gf4_const_mat(c)
        return X

    Xs = [build_lin2x10(row) for row in X_coeffs]
    Ys = Xs

    U_A, V_A, W_A = _expand_tower(
        11, 3, INNER_U_GF4, INNER_V_GF4, INNER_W_GF4,
        gf4_const_mat, 2, Xs, Ys, out_const, 5)

    one = pack5([1, 0, 0, 0, 0])
    M = build_isomorphism(mulA, one, n)
    Minv = gf2_inv(M)
    U = (U_A @ Minv) % 2
    V = (V_A @ Minv) % 2
    W = (W_A @ M.T) % 2
    return U.astype(np.uint8), V.astype(np.uint8), W.astype(np.uint8)


# ===========================================================================
#  Registry
# ===========================================================================

SCHEMES = {
    2:  scheme_n2,
    3:  scheme_n3,
    4:  scheme_n4,
    5:  scheme_n5,
    6:  scheme_n6,
    7:  scheme_n7,
    8:  scheme_n8,
    9:  scheme_n9,
    10: scheme_n10,
}


# ===========================================================================
#  I/O helpers
# ===========================================================================

def matrix_to_bits(M):
    """Binary matrix -> array of uint64 bitmasks."""
    return (M.astype(np.uint64) << np.arange(M.shape[1], dtype=np.uint64)).sum(axis=1)


def save_single_scheme(U, V, W, tensor_id, out_dir):
    """Save CPD scheme in interleaved format: {id:04d}-{rank:05d}.npy."""
    out_dir.mkdir(parents=True, exist_ok=True)
    u_bits = matrix_to_bits(U)
    v_bits = matrix_to_bits(V)
    w_bits = matrix_to_bits(W)
    row = np.column_stack([u_bits, v_bits, w_bits]).ravel()
    path = out_dir / f"{tensor_id:04d}-{U.shape[0]:05d}.npy"
    np.save(path, np.array([row], dtype=np.uint64))
    return path


def base_to_topp(U, V, W, p):
    """Embed (p x p x p) factors into (3p x 3p x 3p) phase-tensor factors."""
    m = 3 * p
    U_t = np.zeros((U.shape[0], m), dtype=np.uint8)
    V_t = np.zeros((V.shape[0], m), dtype=np.uint8)
    W_t = np.zeros((W.shape[0], m), dtype=np.uint8)
    U_t[:, :p] = U
    V_t[:, p: 2 * p] = V
    W_t[:, 2 * p: 3 * p] = W
    return U_t, V_t, W_t


def load_tensor_dense(path):
    """Load sparse tensor .npy as dense binary array."""
    arr = np.load(path)
    dims = tuple(arr[0])
    T = np.zeros(dims, dtype=np.uint8)
    for i, j, k in arr[1:]:
        T[i, j, k] = 1
    return T


def verify_cpd(U, V, W, T):
    """Check T_ijk = sum_q U_qi V_qj W_qk (mod 2)."""
    T_rec = np.einsum("qi,qj,qk->ijk", U, V, W) % 2
    return np.array_equal(T_rec, T)


# ===========================================================================
#  Build one field
# ===========================================================================

def build_field(p, verbose=True):
    """Build and save CPD for GF(2^p) multiplication.  Returns (base_id, topp_id, rank)."""
    rank = EXPECTED_RANKS[p]
    builder = SCHEMES[p]

    tid = p + 23
    base_id = 500 + tid
    topp_id = 700 + tid

    if verbose:
        print(f"GF(2^{p}):  rank {rank},  base {base_id:04d},  topp {topp_id:04d}")

    U, V, W = builder()
    assert U.shape == (rank, p) and V.shape == (rank, p) and W.shape == (rank, p), \
        f"Shape mismatch: expected ({rank}, {p}), got U={U.shape}"

    path_b = save_single_scheme(U, V, W, base_id, CPD_BASE_DIR)
    U_t, V_t, W_t = base_to_topp(U, V, W, p)
    path_t = save_single_scheme(U_t, V_t, W_t, topp_id, CPD_TOPP_DIR)

    if verbose:
        print(f"  saved {path_b}")
        print(f"  saved {path_t}")

    # Optional verification against pre-existing tensors
    for tid_v, Uc, Vc, Wc, label in [
        (base_id, U, V, W, "base"),
        (topp_id, U_t, V_t, W_t, "topp"),
    ]:
        tpath = TENSOR_DIR / f"{tid_v:04d}.npy"
        if tpath.exists():
            T = load_tensor_dense(tpath)
            ok = verify_cpd(Uc, Vc, Wc, T)
            if verbose:
                print(f"  verify {label} {tid_v:04d}: {'PASS' if ok else 'FAIL'}")
            if not ok:
                raise RuntimeError(f"Verification FAILED for {label} {tid_v:04d}")
        else:
            if verbose:
                print(f"  verify {label} {tid_v:04d}: tensor not found, skipped")

    return base_id, topp_id, rank


# ===========================================================================
#  Main
# ===========================================================================

def main():
    for p in range(2, 11):
        build_field(p)
    print("Done.")


if __name__ == "__main__":
    main()
