// cpd/gauss.h
// GF(2) Gaussian elimination for finding linear dependencies.

#pragma once

#include <vector>
#include <cassert>

#include "core/bit_arr.h"
#include "core/bit_vec.h"

namespace cpd {

// Row echelon form over GF(2) with dependency tracking.
// V[i] = row vectors (modified in place)
// I[i] = identity tracking (modified in place), I[i].test(j) means row j contributed
// Returns index of first dependent row (V[result].none()), or -1 if all independent.
// After call, I[result] contains the dependency mask.
inline int ref_find_dependency(std::vector<BitArr>& V, std::vector<BitVec>& I) {
    const int m = static_cast<int>(V.size());
    assert(I.size() == static_cast<std::size_t>(m));
    if (m == 0) return -1;

    const int max_col = static_cast<int>(BitArr::max_bits());
    int pivot_row = 0;

    for (int col = 0; col < max_col && pivot_row < m; ++col) {
        // Find pivot in column
        int found = -1;
        for (int row = pivot_row; row < m; ++row) {
            if (V[row].test(col)) {
                found = row;
                break;
            }
        }
        if (found < 0) continue;

        // Swap to pivot position
        if (found != pivot_row) {
            std::swap(V[pivot_row], V[found]);
            std::swap(I[pivot_row], I[found]);
        }

        // Eliminate below
        for (int row = pivot_row + 1; row < m; ++row) {
            if (V[row].test(col)) {
                V[row] ^= V[pivot_row];
                I[row] ^= I[pivot_row];

                // Check if row became zero
                if (V[row].none()) return row;
            }
        }
        ++pivot_row;
    }

    return -1;
}

// Initialize identity matrix for dependency tracking
inline void init_identity(std::vector<BitVec>& I, int m) {
    I.clear();
    I.reserve(m);
    for (int i = 0; i < m; ++i) {
        BitVec row(m);
        row.set(i);
        I.push_back(std::move(row));
    }
}

} // namespace cpd::base