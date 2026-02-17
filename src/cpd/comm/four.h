// cpd/comm/four.h
// GF(2) commutative quadratic form decomposition utilities.
// Four-set: [u, v, u^v, w] where (u,v) defines an antisymmetric form.

#pragma once

#include "core/bit_arr.h"

#include <array>
#include <vector>
#include <utility>

namespace cpd::comm {

// Returns true iff (u,v,w) contributes to the quadratic part over GF(2).
// Requires: u != 0, v != 0, u != v, w != 0
inline bool is_live_quad(const BitArr& u, const BitArr& v, const BitArr& w) {
    if (u.none() || v.none() || w.none()) return false;
    if (u == v) return false;
    return true;
}

// Build four-set: [u, v, u^v, w]
// Indices 0,1,2 form the three-set (uv-part), index 3 is w
inline std::array<BitArr, 4> make_four(const BitArr& u, const BitArr& v, const BitArr& w) {
    return {u, v, u ^ v, w};
}

// Create unit vector with single bit set
inline BitArr vec_from_bit(int bit) {
    BitArr v{};
    v.set(bit);
    return v;
}

// -----------------------------------------------------------------------------
// Three-set operations (indices 0,1,2 of four-set)
// -----------------------------------------------------------------------------

// Count intersection between two three-sets (uv-parts of four-sets).
// Returns 0, 1, or 3.
inline int three_intersection_count(const std::array<BitArr, 4>& n1,
                                    const std::array<BitArr, 4>& n2) {
    int cnt = 0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            if (n1[i] == n2[j]) ++cnt;
    return cnt;
}

// Collect all pairs (i,j) with n1[i] == n2[j] for i,j in [0,2].
inline void three_collect_pairs(const std::array<BitArr, 4>& n1,
                                const std::array<BitArr, 4>& n2,
                                std::vector<std::pair<int, int>>& out) {
    out.clear();
    out.reserve(3);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            if (n1[i] == n2[j]) out.emplace_back(i, j);
}

// Find first matching pair (i,j) where n1[i] == n2[j].
// Returns {-1,-1} if no match.
inline std::pair<int, int> three_find_common(const std::array<BitArr, 4>& n1,
                                              const std::array<BitArr, 4>& n2) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            if (n1[i] == n2[j]) return {i, j};
    return {-1, -1};
}

// Pick an index in [0,2] that is not equal to `used`.
// For three-set, any element determines the other two up to swap.
inline int three_pick_outside(int used) {
    return (used + 1) % 3;
}

// Pick an index in [0,2] not in the used set.
// Returns -1 if all are used.
inline int three_pick_outside(const std::vector<int>& used) {
    bool taken[3] = {};
    for (int idx : used)
        if (0 <= idx && idx < 3) taken[idx] = true;
    for (int i = 0; i < 3; ++i)
        if (!taken[i]) return i;
    return -1;
}

// Maps code (0,1,2) to the other element for forming a basis pair.
// three-set: [u, v, u^v] at indices [0, 1, 2]
// For code c, returns index of a partner that with c forms a basis.
// code 0 (u)   -> partner 1 (v)
// code 1 (v)   -> partner 0 (u)
// code 2 (u^v) -> partner 0 (u) or 1 (v), we pick 0
inline int three_partner(int code) {
    static constexpr int partner[3] = {1, 0, 0};
    return partner[code];
}

} // namespace cpd::comm