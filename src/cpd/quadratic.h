// cpd/quadratic.h
// Factorization of quadratic forms over GF(2) via alternating matrix decomposition.
// Given p pairs (u_q, v_q) defining a quadratic form sum_q u_q[i] * v_q[j],
// we factorize its symmetric part B = A + A^T into minimal rank representation.

#pragma once

#include "core/bit_arr.h"
#include <vector>
#include <utility>
#include <algorithm>

namespace cpd {

// GF(2) bit matrix: each row is a BitArr, matrix is n x n
using BitMatrix = std::vector<BitArr>;

// Build alternating matrix B[i,j] = sum_q (u[q,i]*v[q,j] ^ u[q,j]*v[q,i]) mod 2
// B is symmetric with zero diagonal, size n x n
inline BitMatrix build_alternating(const std::vector<BitArr>& u, const std::vector<BitArr>& v, int n) {
    BitMatrix B(n);
    const int p = static_cast<int>(u.size());
    
    for (int q = 0; q < p; ++q) {
        for (int i = 0; i < n; ++i) {
            if (!u[q].test(i) && !v[q].test(i)) continue;
            bool ui = u[q].test(i), vi = v[q].test(i);
            
            for (int j = i + 1; j < n; ++j) {
                bool uj = u[q].test(j), vj = v[q].test(j);
                bool t = (ui & vj) ^ (uj & vi);
                if (t) {
                    B[i].flip(j);
                    B[j].flip(i);
                }
            }
        }
    }
    return B;
}

// Swap rows i and j in matrix M (in-place)
inline void swap_rows(BitMatrix& M, int i, int j) {
    std::swap(M[i], M[j]);
}

// Swap columns i and j in matrix M (in-place)
inline void swap_cols(BitMatrix& M, int i, int j) {
    for (auto& row : M) {
        bool ti = row.test(i), tj = row.test(j);
        row.set(i, tj);
        row.set(j, ti);
    }
}

// XOR row src into row dst
inline void xor_row(BitMatrix& M, int dst, int src) {
    M[dst] ^= M[src];
}

// XOR column src into column dst
inline void xor_col(BitMatrix& M, int dst, int src) {
    for (auto& row : M) {
        if (row.test(src)) row.flip(dst);
    }
}

// Bring alternating matrix B to canonical block-diagonal form.
// Returns (B_canonical, S, r) where:
//   - B_canonical = S^T * B * S has r blocks [[0,1],[1,0]] on diagonal
//   - rank(B) = 2*r
inline std::tuple<BitMatrix, BitMatrix, int> alternating_canonical_form(BitMatrix B) {
    const int n = static_cast<int>(B.size());
    
    // Initialize S as identity
    BitMatrix S(n);
    for (int i = 0; i < n; ++i) S[i].set(i);
    
    int i = 0, r = 0;
    
    while (i < n - 1) {
        // Find pivot: first (p, q) with p >= i, q > p, B[p][q] = 1
        int pi = -1, qi = -1;
        for (int p = i; p < n && pi < 0; ++p) {
            for (int q = p + 1; q < n; ++q) {
                if (B[p].test(q)) { pi = p; qi = q; break; }
            }
        }
        if (pi < 0) break;  // No more pivots
        
        // Move pivot to (i, i+1)
        if (pi != i) {
            swap_rows(B, i, pi); swap_cols(B, i, pi);
            swap_cols(S, i, pi);
        }
        if (qi != i + 1) {
            swap_rows(B, i + 1, qi); swap_cols(B, i + 1, qi);
            swap_cols(S, i + 1, qi);
        }
        
        // Now B[i][i+1] = 1. Clear B[i][k] and B[i+1][k] for k > i+1
        for (int k = i + 2; k < n; ++k) {
            bool a = B[i].test(k), b = B[i + 1].test(k);
            if (!a && !b) continue;
            
            if (a && !b) {
                xor_row(B, k, i + 1); xor_col(B, k, i + 1);
                xor_col(S, k, i + 1);
            } else if (!a && b) {
                xor_row(B, k, i); xor_col(B, k, i);
                xor_col(S, k, i);
            } else {
                xor_row(B, k, i); xor_col(B, k, i);
                xor_col(S, k, i);
                xor_row(B, k, i + 1); xor_col(B, k, i + 1);
                xor_col(S, k, i + 1);
            }
        }
        
        i += 2;
        ++r;
    }
    
    return {B, S, r};
}

// Invert matrix M over GF(2) using Gauss-Jordan elimination.
// Uses two separate n×n matrices instead of augmented [M|I] to avoid overflow.
inline BitMatrix gf2_inverse(const BitMatrix& M) {
    const int n = static_cast<int>(M.size());
    
    // Work matrix A (copy of M) and result matrix Inv (starts as identity)
    BitMatrix A = M;
    BitMatrix Inv(n);
    for (int i = 0; i < n; ++i) Inv[i].set(i);
    
    // Forward elimination with partial pivoting
    for (int col = 0; col < n; ++col) {
        // Find pivot
        int pivot = -1;
        for (int row = col; row < n; ++row) {
            if (A[row].test(col)) { pivot = row; break; }
        }
        if (pivot < 0) continue;  // Singular
        
        if (pivot != col) {
            std::swap(A[col], A[pivot]);
            std::swap(Inv[col], Inv[pivot]);
        }
        
        // Eliminate
        for (int row = 0; row < n; ++row) {
            if (row != col && A[row].test(col)) {
                A[row] ^= A[col];
                Inv[row] ^= Inv[col];
            }
        }
    }
    
    return Inv;
}

// Factorize quadratic form over GF(2).
//
// Input: p pairs (u[q], v[q]) as bit vectors of dimension n
// Output: (u_small, v_small) with r <= p pairs such that
//         the symmetric part of sum_q u[q] ⊗ v[q] equals
//         the symmetric part of sum_k u_small[k] ⊗ v_small[k]
//
// The reduction is optimal: 2*r = rank of the alternating matrix B = A + A^T.
inline std::pair<std::vector<BitArr>, std::vector<BitArr>> factorize_quadratic(
    const std::vector<BitArr>& u_vecs,
    const std::vector<BitArr>& v_vecs,
    int n
) {
    
    if (u_vecs.empty() || n == 0) return {{}, {}};
    
    // Build alternating matrix B
    BitMatrix B = build_alternating(u_vecs, v_vecs, n);
    
    // Canonical form: B_can = S^T * B * S with r blocks [[0,1],[1,0]]
    auto [B_can, S, r] = alternating_canonical_form(std::move(B));
    
    if (r == 0) return {{}, {}};
    
    // S^{-1} rows give the factorization
    BitMatrix S_inv = gf2_inverse(S);
    
    std::vector<BitArr> u_small(r), v_small(r);
    for (int k = 0; k < r; ++k) {
        // Extract first n bits from rows 2k and 2k+1 of S_inv
        for (int j = 0; j < n; ++j) {
            u_small[k].set(j, S_inv[2 * k].test(j));
            v_small[k].set(j, S_inv[2 * k + 1].test(j));
        }
    }
    
    return {u_small, v_small};
}

} // namespace cpd::topp