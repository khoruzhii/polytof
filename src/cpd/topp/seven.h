// cpd/topp/seven.h
#pragma once

#include "core/bit_arr.h"

#include <array>
#include <vector>
#include <algorithm>
#include <utility>

namespace cpd::topp {

// Returns true iff (u,v,w) contributes to the cubic part over GF(2).
inline bool cubic_triple(const BitArr& u, const BitArr& v, const BitArr& w) {
    if (u.none() || v.none() || w.none()) return false;
    if (u == v || u == w || v == w) return false;
    if ((u ^ v ^ w).none()) return false;
    return true;
}

// Fixed order: [u, v, w, u^v, u^w, v^w, u^v^w].
inline std::array<BitArr, 7> make_seven(const BitArr& u, const BitArr& v, const BitArr& w) {
    return {u, v, w, u ^ v, u ^ w, v ^ w, u ^ v ^ w};
}

// Simple O(7*7) intersection count between two seven-sets.
inline int seven_intersection_count(const std::array<BitArr, 7>& s1,
                                    const std::array<BitArr, 7>& s2) {
    int cnt = 0;
    for (int i = 0; i < 7; ++i)
        for (int j = 0; j < 7; ++j)
            if (s1[i] == s2[j]) ++cnt;
    return cnt;
}

// Collect all pairs (i,j) with s1[i] == s2[j] in "first match" order.
inline void seven_collect_pairs(const std::array<BitArr, 7>& s1,
                                const std::array<BitArr, 7>& s2,
                                std::vector<std::pair<int, int>>& out_pairs) {
    out_pairs.clear();
    out_pairs.reserve(7);
    for (int i = 0; i < 7; ++i)
        for (int j = 0; j < 7; ++j)
            if (s1[i] == s2[j]) out_pairs.emplace_back(i, j);
}

// Returns the first index in [0..6] not listed in marks; -1 if none.
inline int seven_pick_outside(const std::vector<int>& marks) {
    bool used[7] = {};
    for (int idx : marks)
        if (0 <= idx && idx < 7) used[idx] = true;
    for (int i = 0; i < 7; ++i)
        if (!used[i]) return i;
    return -1;
}

// In-place sort of three BitArr values to keep rows canonical.
inline void sort3(BitArr& a, BitArr& b, BitArr& c) {
    if (b < a) std::swap(a, b);
    if (c < b) std::swap(b, c);
    if (b < a) std::swap(a, b);
}

} // namespace cpd::topp