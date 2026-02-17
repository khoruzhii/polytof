// cpd/base/tensor.h
// GF(2) bilinear tensor representation and verification.
// Stores 3-tensor T_{ijk} as sparse list of nonzero (i,j,k) entries.

#pragma once

#include <algorithm>
#include <array>
#include <vector>
#include <string>
#include <stdexcept>

#include "cnpy.h"
#include "core/bit_arr.h"

namespace cpd::base {

// GF(2) 3-tensor of shape (n0, n1, n2), stored as nonzero (i,j,k) triples
struct Tensor {
    std::array<int, 3> dims{0, 0, 0};  // n0, n1, n2
    std::vector<std::array<int, 3>> triples;
    std::array<int, 3> get_dims() const { return dims; }
};

// -----------------------------------------------------------------------------
// I/O: .npy with shape (nnz+1, 3), row 0 = (n0,n1,n2), rest = (i,j,k) triples
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
        t.dims[0] = static_cast<int>(ptr[0]);
        t.dims[1] = static_cast<int>(ptr[1]);
        t.dims[2] = static_cast<int>(ptr[2]);

        for (int d = 0; d < 3; ++d) {
            if (t.dims[d] <= 0 || static_cast<std::size_t>(t.dims[d]) > BitArr::max_bits())
                throw std::runtime_error("Dimension out of range");
        }

        t.triples.reserve(rows - 1);
        for (std::size_t r = 1; r < rows; ++r) {
            int i = static_cast<int>(ptr[r * 3 + 0]);
            int j = static_cast<int>(ptr[r * 3 + 1]);
            int k = static_cast<int>(ptr[r * 3 + 2]);

            if (i < 0 || i >= t.dims[0] ||
                j < 0 || j >= t.dims[1] ||
                k < 0 || k >= t.dims[2])
                throw std::runtime_error("Index out of range");

            t.triples.push_back({i, j, k});
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

inline BitArr vec_from_bit(int bit) {
    BitArr v{};
    v.set(bit);
    return v;
}

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

// -----------------------------------------------------------------------------
// Verification: check if decomposition reconstructs tensor
// T_{ijk} = XOR_q (u_q[i] * v_q[j] * w_q[k])
// -----------------------------------------------------------------------------

inline bool verify(const std::vector<BitArr>& scheme, const Tensor& t) {
    const int n0 = t.dims[0], n1 = t.dims[1], n2 = t.dims[2];
    const int num_terms = static_cast<int>(scheme.size() / 3);

    // Collect live terms
    std::vector<std::array<const BitArr*, 3>> live;
    live.reserve(num_terms);
    for (int q = 0; q < num_terms; ++q) {
        if (scheme[3*q].any())
            live.push_back({&scheme[3*q], &scheme[3*q+1], &scheme[3*q+2]});
    }

    // Build set of nonzero entries from decomposition
    std::vector<std::array<int, 3>> result;
    result.reserve(t.triples.size());

    for (int i = 0; i < n0; ++i) {
        for (int j = 0; j < n1; ++j) {
            for (int k = 0; k < n2; ++k) {
                unsigned bit = 0;
                for (const auto& [u, v, w] : live)
                    bit ^= (u->test(i) & v->test(j) & w->test(k));
                if (bit & 1U)
                    result.push_back({i, j, k});
            }
        }
    }

    // Compare with original tensor (sort both for comparison)
    auto sorted_orig = t.triples;
    std::sort(sorted_orig.begin(), sorted_orig.end());
    std::sort(result.begin(), result.end());

    return result == sorted_orig;
}

} // namespace cpd::base