// cpd/topp/tensor.h
// GF(2) Third Order Phase Polynomial (TOPP) tensor representation and verification.
// Stores symmetric 3-tensor as distinct-index triples (i < j < k).

#pragma once

#include <array>
#include <vector>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <bit>

#include "cnpy.h"
#include "core/bit_arr.h"
#include "cpd/topp/seven.h"

namespace cpd::topp {

// Symmetric GF(2) 3-tensor, stored as sorted triples (i < j < k)
struct Tensor {
    int n{0};
    std::vector<std::array<int, 3>> triples;
    std::array<int, 3> get_dims() const { return {n, n, n}; }
};

// -----------------------------------------------------------------------------
// I/O: .npy with shape (nnz+1, 3), row 0 = (n,n,n), rest = sorted triples
// -----------------------------------------------------------------------------

inline Tensor load_tensor(const std::string& path) {
    cnpy::NpyArray arr = cnpy::npy_load(path);
    if (arr.shape.size() != 2 || arr.shape[1] != 3)
        throw std::runtime_error("Tensor .npy must have shape (rows, 3)");
    if (arr.shape[0] < 1)
        throw std::runtime_error("Tensor .npy must have at least one row");

    const std::size_t rows = arr.shape[0];

    auto read = [&](auto* ptr) -> Tensor {
        using T = std::remove_pointer_t<decltype(ptr)>;
        Tensor t;

        if (ptr[0] != ptr[1] || ptr[1] != ptr[2])
            throw std::runtime_error("First row must be (n,n,n)");
        if (ptr[0] < 0 || static_cast<std::size_t>(ptr[0]) > BitArr::max_bits())
            throw std::runtime_error("n out of range");

        t.n = static_cast<int>(ptr[0]);
        t.triples.reserve(rows - 1);

        std::array<int, 3> prev{-1, -1, -1};
        for (std::size_t r = 1; r < rows; ++r) {
            int i = static_cast<int>(ptr[r * 3]);
            int j = static_cast<int>(ptr[r * 3 + 1]);
            int k = static_cast<int>(ptr[r * 3 + 2]);

            if (i < 0 || j < 0 || k < 0 || i >= t.n || j >= t.n || k >= t.n)
                throw std::runtime_error("Index out of range");
            if (!(i < j && j < k))
                throw std::runtime_error("Triple must satisfy i < j < k");

            std::array<int, 3> cur{i, j, k};
            if (prev[0] != -1 && !(prev < cur))
                throw std::runtime_error("Triples must be strictly increasing");

            prev = cur;
            t.triples.push_back(cur);
        }
        return t;
    };

    switch (arr.word_size) {
        case 1: return read(arr.data<int8_t>());
        case 2: return read(arr.data<int16_t>());
        case 4: return read(arr.data<int32_t>());
        case 8: return read(arr.data<int64_t>());
        default: throw std::runtime_error("Unsupported word size");
    }
}

// -----------------------------------------------------------------------------
// Trivial decomposition: one rank-1 term per triple
// -----------------------------------------------------------------------------

inline std::vector<BitArr> trivial_decomposition(const Tensor& t) {
    std::vector<BitArr> data;
    data.reserve(t.triples.size() * 3);
    for (const auto& [i, j, k] : t.triples) {
        data.push_back(vec_from_bit(i));
        data.push_back(vec_from_bit(j));
        data.push_back(vec_from_bit(k));
    }
    return data;
}

// -----------------------------------------------------------------------------
// Rank computation
// -----------------------------------------------------------------------------

inline int get_rank(const std::vector<BitArr>& data) {
    int cnt = 0;
    for (std::size_t i = 0; i < data.size(); i += 3)
        if (data[i].any()) ++cnt;
    return cnt;
}

inline int get_rank_cubic(const std::vector<BitArr>& data) {
    int cnt = 0;
    for (std::size_t i = 0; i + 2 < data.size(); i += 3)
        if (cubic_triple(data[i], data[i + 1], data[i + 2])) ++cnt;
    return cnt;
}

// -----------------------------------------------------------------------------
// Primordial reduce: canonical grouping by (u,v), XOR-accumulate w
// -----------------------------------------------------------------------------

inline std::vector<BitArr> primordial_reduce(const std::vector<BitArr>& data) {
    if (data.size() % 3 != 0) return {};
    const int m = static_cast<int>(data.size() / 3);
    if (m == 0) return {};

    // Collect and canonicalize cubic triples
    std::vector<std::array<BitArr, 3>> sorted;
    sorted.reserve(m);

    for (int t = 0; t < m; ++t) {
        BitArr a = data[3 * t], b = data[3 * t + 1], c = data[3 * t + 2];
        if (!cubic_triple(a, b, c)) continue;
        sort3(a, b, c);
        sorted.push_back({a, b, c});
    }
    if (sorted.empty()) return {};

    std::sort(sorted.begin(), sorted.end());

    // Group by (u,v), XOR w
    std::vector<std::array<BitArr, 3>> out;
    out.reserve(sorted.size());

    BitArr gu = sorted[0][0], gv = sorted[0][1], gw = sorted[0][2];

    for (std::size_t t = 1; t < sorted.size(); ++t) {
        const auto& [u, v, w] = sorted[t];
        if (u == gu && v == gv) {
            gw ^= w;
        } else {
            if (gw.any() && cubic_triple(gu, gv, gw)) {
                BitArr a = gu, b = gv, c = gw;
                sort3(a, b, c);
                out.push_back({a, b, c});
            }
            gu = u; gv = v; gw = w;
        }
    }

    // Final group
    if (gw.any() && cubic_triple(gu, gv, gw)) {
        BitArr a = gu, b = gv, c = gw;
        sort3(a, b, c);
        out.push_back({a, b, c});
    }

    std::sort(out.begin(), out.end());

    // Flatten
    std::vector<BitArr> result;
    result.reserve(out.size() * 3);
    for (const auto& [a, b, c] : out) {
        result.push_back(a);
        result.push_back(b);
        result.push_back(c);
    }
    return result;
}

// -----------------------------------------------------------------------------
// Verification: check if decomposition reconstructs tensor
// -----------------------------------------------------------------------------

namespace detail {

// Cubic term contribution at point x: a*b*c ⊕ (ab)*c ⊕ (ac)*b ⊕ (bc)*a
inline int cubic_term_value(U64 u, U64 v, U64 w, U64 x) {
    auto p = [](U64 val) { return std::popcount(val) & 1; };
    int a = p(u & x), b = p(v & x), c = p(w & x);
    int ab = p((u & v) & x), ac = p((u & w) & x), bc = p((v & w) & x);
    return (a & b & c) ^ (ab & c) ^ (ac & b) ^ (bc & a);
}

// Phi-table: 9-bit pattern -> cubic coefficient contribution
inline const std::array<uint8_t, 512>& phi_table() {
    static const auto tbl = [] {
        std::array<uint8_t, 512> t{};
        for (int p = 0; p < 512; ++p) {
            U64 u = ((p >> 0) & 1) | (((p >> 1) & 1) << 1) | (((p >> 2) & 1) << 2);
            U64 v = ((p >> 3) & 1) | (((p >> 4) & 1) << 1) | (((p >> 5) & 1) << 2);
            U64 w = ((p >> 6) & 1) | (((p >> 7) & 1) << 1) | (((p >> 8) & 1) << 2);
            int coeff = 0;
            for (U64 x = 0; x < 8; ++x)
                coeff ^= cubic_term_value(u, v, w, x);
            t[p] = static_cast<uint8_t>(coeff & 1);
        }
        return t;
    }();
    return tbl;
}

} // namespace detail

inline bool verify(const std::vector<BitArr>& scheme, const Tensor& t) {
    // Filter cubic-live terms
    std::vector<BitArr> cubic;
    cubic.reserve(scheme.size());

    for (std::size_t i = 0; i + 2 < scheme.size(); i += 3) {
        const auto &u = scheme[i], &v = scheme[i + 1], &w = scheme[i + 2];
        if (cubic_triple(u, v, w)) {
            cubic.push_back(u);
            cubic.push_back(v);
            cubic.push_back(w);
        }
    }

    const int rc = static_cast<int>(cubic.size() / 3);
    if (rc == 0) return t.triples.empty();

    const auto& phi = detail::phi_table();
    std::vector<std::array<int, 3>> result;
    result.reserve(t.triples.size());

    for (int i = 0; i < t.n; ++i) {
        for (int j = i + 1; j < t.n; ++j) {
            for (int k = j + 1; k < t.n; ++k) {
                unsigned bit = 0;
                for (int q = 0; q < rc; ++q) {
                    const auto &u = cubic[3 * q], &v = cubic[3 * q + 1], &w = cubic[3 * q + 2];
                    unsigned pat = 0;
                    pat |= static_cast<unsigned>(u.test(i));
                    pat |= static_cast<unsigned>(u.test(j)) << 1;
                    pat |= static_cast<unsigned>(u.test(k)) << 2;
                    pat |= static_cast<unsigned>(v.test(i)) << 3;
                    pat |= static_cast<unsigned>(v.test(j)) << 4;
                    pat |= static_cast<unsigned>(v.test(k)) << 5;
                    pat |= static_cast<unsigned>(w.test(i)) << 6;
                    pat |= static_cast<unsigned>(w.test(j)) << 7;
                    pat |= static_cast<unsigned>(w.test(k)) << 8;
                    bit ^= phi[pat];
                }
                if (bit & 1U) result.push_back({i, j, k});
            }
        }
    }

    return result == t.triples;
}

} // namespace cpd::topp