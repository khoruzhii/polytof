// waring/prototype.h
// Common types for phase polynomial optimization pipeline.
// Used by prototype.cpp for experimental bco8 → ctc → todd pipeline.

#pragma once

#include <array>
#include <vector>
#include <cstdint>
#include <algorithm>
#include "unordered_dense.h"
#include "waring/tensor.h"

namespace waring::proto {

// -----------------------------------------------------------------------------
// Packed triple representation (9 bits per index, n ≤ 512)
// -----------------------------------------------------------------------------

using Triple = std::uint32_t;
constexpr Triple INVALID = 0xFFFFFFFF;

inline Triple pack(int i, int j, int k) {
    return (std::uint32_t(i) << 18) | (std::uint32_t(j) << 9) | std::uint32_t(k);
}
inline int ti(Triple t) { return int(t >> 18); }
inline int tj(Triple t) { return int((t >> 9) & 0x1FF); }
inline int tk(Triple t) { return int(t & 0x1FF); }

inline int degree(Triple t) {
    int i = ti(t), j = tj(t), k = tk(t);
    if (i == j && j == k) return 1;
    if (i == j && j < k) return 2;
    return 3;
}

// Sort and pack cubic triple (returns INVALID if not distinct)
inline Triple sorted3(int a, int b, int c) {
    if (a > b) std::swap(a, b);
    if (b > c) std::swap(b, c);
    if (a > b) std::swap(a, b);
    if (a == b || b == c) return INVALID;
    return pack(a, b, c);
}

// Make canonical quadratic triple (i,i,k) with i < k
inline Triple make_quad(int a, int c) {
    if (a == c) return INVALID;
    return (a < c) ? pack(a, a, c) : pack(c, c, a);
}

// -----------------------------------------------------------------------------
// PolyState: phase polynomial with contains-index for efficient transvections
// Used by bco8 algorithm
// -----------------------------------------------------------------------------

class PolyState {
public:
    int n;
    ankerl::unordered_dense::set<Triple> triples;
    std::vector<ankerl::unordered_dense::set<Triple>> contains;

    PolyState(const SymTensor& T) : n(T.n), contains(T.n) {
        for (auto [i, j, k] : T.triples) {
            Triple t = pack(i, j, k);
            triples.insert(t);
            contains[i].insert(t);
            if (j != i) contains[j].insert(t);
            if (k != j) contains[k].insert(t);
        }
    }

    std::int64_t cost(int c1, int c2, int c3) const {
        std::int64_t c = 0;
        int cs[] = {0, c1, c2, c3};
        for (Triple t : triples) c += cs[degree(t)];
        return c;
    }

    void count(std::size_t& n1, std::size_t& n2, std::size_t& n3) const {
        n1 = n2 = n3 = 0;
        for (Triple t : triples) {
            int d = degree(t);
            if (d == 1) ++n1; else if (d == 2) ++n2; else ++n3;
        }
    }

    void collect_changes(int i, int j, ankerl::unordered_dense::map<Triple, int>& changes) const {
        for (Triple t : contains[i]) {
            int a = ti(t), b = tj(t), c = tk(t);
            int deg = degree(t);

            if (deg == 1) {
                changes[pack(j, j, j)] ^= 1;
                if (i != j) changes[make_quad(i, j)] ^= 1;
            } else if (deg == 2) {
                int other = (a == i) ? c : a;
                if (j != other) changes[make_quad(j, other)] ^= 1;
                Triple casc = sorted3(i, j, other);
                if (casc != INVALID) changes[casc] ^= 1;
            } else {
                int cnt = (a == i) + (b == i) + (c == i);
                std::array<int, 3> idx = {a, b, c}, pos;
                int pi = 0;
                for (int p = 0; p < 3; ++p)
                    if (idx[p] == i) pos[pi++] = p;

                for (int mask = 1; mask < (1 << cnt); ++mask) {
                    auto nidx = idx;
                    for (int bit = 0; bit < cnt; ++bit)
                        if (mask & (1 << bit)) nidx[pos[bit]] = j;
                    Triple nt = sorted3(nidx[0], nidx[1], nidx[2]);
                    if (nt != INVALID) changes[nt] ^= 1;
                }
            }
        }
    }

    std::int64_t compute_delta(int i, int j, int c1, int c2, int c3) const {
        ankerl::unordered_dense::map<Triple, int> changes;
        collect_changes(i, j, changes);
        int cs[] = {0, c1, c2, c3};
        std::int64_t delta = 0;
        for (auto [t, parity] : changes)
            if (parity) delta += cs[degree(t)] * (triples.contains(t) ? -1 : 1);
        return delta;
    }

    void apply(int i, int j) {
        ankerl::unordered_dense::map<Triple, int> changes;
        collect_changes(i, j, changes);

        for (auto [t, parity] : changes) {
            if (!parity) continue;
            int a = ti(t), b = tj(t), c = tk(t);
            if (triples.contains(t)) {
                triples.erase(t);
                contains[a].erase(t);
                if (b != a) contains[b].erase(t);
                if (c != b) contains[c].erase(t);
            } else {
                triples.insert(t);
                contains[a].insert(t);
                if (b != a) contains[b].insert(t);
                if (c != b) contains[c].insert(t);
            }
        }
    }

    SymTensor export_tensor() const {
        SymTensor T;
        T.n = n;
        T.triples.reserve(triples.size());
        for (Triple t : triples)
            T.triples.push_back({ti(t), tj(t), tk(t)});
        std::sort(T.triples.begin(), T.triples.end());
        return T;
    }

    std::vector<int> active_indices() const {
        std::vector<int> active;
        for (int i = 0; i < n; ++i)
            if (!contains[i].empty()) active.push_back(i);
        return active;
    }
};

// -----------------------------------------------------------------------------
// PhaseState: phase polynomial with separate f1, f2, f3 sets
// Used by ctc algorithm
// -----------------------------------------------------------------------------

class PhaseState {
public:
    int n;
    ankerl::unordered_dense::set<int> f1;       // linear: x_i
    ankerl::unordered_dense::set<Triple> f2;    // quadratic: (i,i,k) canonical
    ankerl::unordered_dense::set<Triple> f3;    // cubic: (i,j,k) with i<j<k

    PhaseState(const SymTensor& T) : n(T.n) {
        for (auto [i, j, k] : T.triples) {
            int deg = degree(pack(i, j, k));
            if (deg == 1) f1.insert(i);
            else if (deg == 2) f2.insert(pack(i, j, k));
            else f3.insert(pack(i, j, k));
        }
    }

    std::int64_t cost(int c1, int c2, int c3) const {
        return c1 * std::int64_t(f1.size()) + c2 * std::int64_t(f2.size()) + c3 * std::int64_t(f3.size());
    }

    std::size_t gates() const {
        return f1.size() + 3 * f2.size() + 7 * f3.size();
    }

    // Delta from insertion T(i,j): adds x_i + x_j + 2·x_i·x_j
    std::int64_t insertion_delta(int i, int j, int c1, int c2) const {
        if (i == j) return 0;
        std::int64_t delta = c1;  // insertion cost
        delta += f1.contains(i) ? -c1 : c1;
        delta += f1.contains(j) ? -c1 : c1;
        Triple q = make_quad(i, j);
        delta += f2.contains(q) ? -c2 : c2;
        return delta;
    }

    void apply_insertion(int i, int j) {
        if (i == j) return;
        auto toggle = [](auto& s, auto x) {
            if (s.contains(x)) s.erase(x);
            else s.insert(x);
        };
        toggle(f1, i);
        toggle(f1, j);
        toggle(f2, make_quad(i, j));
    }
};

} // namespace waring::proto