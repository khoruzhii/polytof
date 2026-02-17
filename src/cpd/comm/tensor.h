// cpd/comm/tensor.h
// GF(2) commutative quadratic tensor representation and verification.
// Stores tensor T_{ijk} for quadratic form c_k = sum_{i<j} T_{ijk} y_i y_j.
// Triples (i, j, k) with i < j.

#pragma once

#include <array>
#include <vector>
#include <string>
#include <algorithm>
#include <stdexcept>

#include "cnpy.h"
#include "core/bit_arr.h"
#include "cpd/comm/four.h"

namespace cpd::comm {

// Commutative quadratic tensor of shape (n, n, nw).
// Stored as triples (i, j, k) with i < j.
struct Tensor {
    int n{0};   // dimension for i, j (quadratic variables)
    int nw{0};  // dimension for k (output)
    std::vector<std::array<int, 3>> triples;
    std::array<int, 3> get_dims() const { return {n, n, nw}; }
};

// -----------------------------------------------------------------------------
// I/O: .npy with shape (nnz+1, 3), row 0 = (n, n, nw), rest = (i, j, k) triples
// -----------------------------------------------------------------------------

inline Tensor load_tensor(const std::string& path) {
    cnpy::NpyArray arr = cnpy::npy_load(path);
    if (arr.shape.size() != 2 || arr.shape[1] != 3)
        throw std::runtime_error("Tensor .npy must have shape (rows, 3)");
    if (arr.shape[0] < 1)
        throw std::runtime_error("Tensor .npy must have at least one row");

    const std::size_t rows = arr.shape[0];

    auto read = [&](auto* ptr) -> Tensor {
        Tensor t;

        // First row: (n, n, nw)
        if (ptr[0] != ptr[1])
            throw std::runtime_error("First row must be (n, n, nw)");
        if (ptr[0] < 0 || static_cast<std::size_t>(ptr[0]) > BitArr::max_bits())
            throw std::runtime_error("n out of range");
        if (ptr[2] < 0 || static_cast<std::size_t>(ptr[2]) > BitArr::max_bits())
            throw std::runtime_error("nw out of range");

        t.n = static_cast<int>(ptr[0]);
        t.nw = static_cast<int>(ptr[2]);
        t.triples.reserve(rows - 1);

        for (std::size_t r = 1; r < rows; ++r) {
            int i = static_cast<int>(ptr[r * 3]);
            int j = static_cast<int>(ptr[r * 3 + 1]);
            int k = static_cast<int>(ptr[r * 3 + 2]);

            // Normalize to i < j (antisymmetric part)
            if (i == j) continue;  // diagonal contributes only to linear part
            if (i > j) std::swap(i, j);

            if (i < 0 || j >= t.n || k < 0 || k >= t.nw)
                throw std::runtime_error("Index out of range");

            t.triples.push_back({i, j, k});
        }

        // Sort and deduplicate (XOR cancellation)
        std::sort(t.triples.begin(), t.triples.end());

        std::vector<std::array<int, 3>> unique;
        unique.reserve(t.triples.size());

        for (std::size_t idx = 0; idx < t.triples.size(); ) {
            std::size_t end = idx + 1;
            while (end < t.triples.size() && t.triples[end] == t.triples[idx])
                ++end;
            // Keep if odd count (XOR)
            if ((end - idx) & 1)
                unique.push_back(t.triples[idx]);
            idx = end;
        }

        t.triples = std::move(unique);
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
    for (std::size_t i = 0; i + 2 < data.size(); i += 3)
        if (is_live_quad(data[i], data[i + 1], data[i + 2])) ++cnt;
    return cnt;
}

// -----------------------------------------------------------------------------
// Verification: check if decomposition reconstructs tensor
// T_{ijk} = XOR_q (u_q[i] * v_q[j] + u_q[j] * v_q[i]) * w_q[k]  for i < j
// -----------------------------------------------------------------------------

inline bool verify(const std::vector<BitArr>& scheme, const Tensor& t) {
    // Filter live terms
    std::vector<std::array<const BitArr*, 3>> live;
    live.reserve(scheme.size() / 3);

    for (std::size_t idx = 0; idx + 2 < scheme.size(); idx += 3) {
        const auto &u = scheme[idx], &v = scheme[idx + 1], &w = scheme[idx + 2];
        if (is_live_quad(u, v, w))
            live.push_back({&u, &v, &w});
    }

    if (live.empty()) return t.triples.empty();

    // Reconstruct tensor
    std::vector<std::array<int, 3>> result;
    result.reserve(t.triples.size());

    for (int i = 0; i < t.n; ++i) {
        for (int j = i + 1; j < t.n; ++j) {
            for (int k = 0; k < t.nw; ++k) {
                unsigned bit = 0;
                for (const auto& [u, v, w] : live) {
                    // (u[i]*v[j] + u[j]*v[i]) * w[k]
                    unsigned ui = u->test(i), uj = u->test(j);
                    unsigned vi = v->test(i), vj = v->test(j);
                    unsigned wk = w->test(k);
                    bit ^= ((ui & vj) ^ (uj & vi)) & wk;
                }
                if (bit & 1U) result.push_back({i, j, k});
            }
        }
    }

    // Compare (both should be sorted)
    return result == t.triples;
}

} // namespace cpd::comm