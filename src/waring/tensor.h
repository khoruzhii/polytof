// waring/tensor.h
// Symmetric 3-tensor over GF(2) and parity table I/O.

#pragma once

#include <array>
#include <vector>
#include <string>
#include <cstdint>
#include <stdexcept>
#include <filesystem>
#include <regex>
#include <optional>
#include <algorithm>
#include "cnpy.h"
#include "core/bit_arr.h"
#include "core/paths.h"

namespace waring {

namespace fs = std::filesystem;

// -----------------------------------------------------------------------------
// Symmetric 3-tensor over GF(2)
// Stored as sorted triples (i ≤ j ≤ k) where A_ijk = 1.
// Supports all degrees: linear (i,i,i), quadratic (i,i,k), cubic (i,j,k).
// -----------------------------------------------------------------------------

struct SymTensor {
    int n = 0;
    std::vector<std::array<int, 3>> triples;

    // Build tensor from parity columns: A_ijk = |{p : p[i] ∧ p[j] ∧ p[k]}| mod 2
    // Uses canonical form: i==j==k (linear), i==j<k (quadratic), i<j<k (cubic)
    static SymTensor from_parities(const std::vector<BitArr>& parities, int n) {
        SymTensor T;
        T.n = n;
        for (int i = 0; i < n; ++i) {
            for (int j = i; j < n; ++j) {
                for (int k = j; k < n; ++k) {
                    if (i < j && j == k) continue;  // skip non-canonical quadratic
                    std::size_t cnt = 0;
                    for (const auto& p : parities)
                        if (p.test(i) && p.test(j) && p.test(k)) ++cnt;
                    if (cnt & 1) T.triples.push_back({i, j, k});
                }
            }
        }
        return T;
    }

    bool operator==(const SymTensor& o) const {
        return n == o.n && triples == o.triples;
    }
};

// -----------------------------------------------------------------------------
// Tensor I/O: .npy with shape (nnz+1, 3), row 0 = (n,n,n), rest = sorted triples
// -----------------------------------------------------------------------------

inline SymTensor load_tensor(const std::string& path) {
    cnpy::NpyArray arr = cnpy::npy_load(path);
    if (arr.shape.size() != 2 || arr.shape[1] != 3)
        throw std::runtime_error("Tensor .npy must have shape (rows, 3)");
    if (arr.shape[0] < 1)
        throw std::runtime_error("Tensor .npy must have at least one row");

    const std::size_t rows = arr.shape[0];

    auto read = [&](auto* ptr) -> SymTensor {
        SymTensor t;

        if (ptr[0] != ptr[1] || ptr[1] != ptr[2])
            throw std::runtime_error("First row must be (n,n,n)");
        if (ptr[0] < 0 || static_cast<std::size_t>(ptr[0]) > BitArr::max_bits())
            throw std::runtime_error("n out of range");

        t.n = static_cast<int>(ptr[0]);
        t.triples.reserve(rows - 1);

        for (std::size_t r = 1; r < rows; ++r) {
            int i = static_cast<int>(ptr[r * 3]);
            int j = static_cast<int>(ptr[r * 3 + 1]);
            int k = static_cast<int>(ptr[r * 3 + 2]);

            if (i < 0 || j < 0 || k < 0 || i >= t.n || j >= t.n || k >= t.n)
                throw std::runtime_error("Index out of range");
            if (!(i <= j && j <= k))
                throw std::runtime_error("Triple must satisfy i <= j <= k");

            t.triples.push_back({i, j, k});
        }

        std::sort(t.triples.begin(), t.triples.end());
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

inline void save_tensor(const SymTensor& t, const std::string& path) {
    std::vector<int32_t> data((t.triples.size() + 1) * 3);
    data[0] = data[1] = data[2] = t.n;
    for (std::size_t r = 0; r < t.triples.size(); ++r) {
        data[(r + 1) * 3 + 0] = t.triples[r][0];
        data[(r + 1) * 3 + 1] = t.triples[r][1];
        data[(r + 1) * 3 + 2] = t.triples[r][2];
    }
    fs::create_directories(fs::path(path).parent_path());
    cnpy::npy_save(path, data.data(), {t.triples.size() + 1, 3});
}

// -----------------------------------------------------------------------------
// Parity I/O
// -----------------------------------------------------------------------------

// Load parities from .npy matrix [n × rank] (column-major style)
inline std::vector<BitArr> load_parities(const std::string& path, int& n) {
    cnpy::NpyArray arr = cnpy::npy_load(path);
    if (arr.shape.size() != 2)
        throw std::runtime_error("Expected 2D array, got " + std::to_string(arr.shape.size()) + "D");

    n = static_cast<int>(arr.shape[0]);
    std::size_t rank = arr.shape[1];

    if (static_cast<std::size_t>(n) > BitArr::max_bits())
        throw std::runtime_error("n=" + std::to_string(n) + 
            " exceeds BitArr::max_bits()=" + std::to_string(BitArr::max_bits()));

    std::vector<BitArr> parities(rank);
    const uint8_t* data = arr.data<uint8_t>();
    for (std::size_t r = 0; r < rank; ++r)
        for (int q = 0; q < n; ++q)
            if (data[q * rank + r]) parities[r].set(q);
    return parities;
}

// Load waring parities from .npy matrix [rank × n] (row-major, each row is a parity)
inline std::vector<BitArr> load_waring_parities(const std::string& path, int& n) {
    cnpy::NpyArray arr = cnpy::npy_load(path);
    if (arr.shape.size() != 2)
        throw std::runtime_error("Expected 2D array, got " + std::to_string(arr.shape.size()) + "D");

    std::size_t rank = arr.shape[0];
    n = static_cast<int>(arr.shape[1]);

    if (static_cast<std::size_t>(n) > BitArr::max_bits())
        throw std::runtime_error("n=" + std::to_string(n) + 
            " exceeds BitArr::max_bits()=" + std::to_string(BitArr::max_bits()));

    std::vector<BitArr> parities(rank);
    const uint8_t* data = arr.data<uint8_t>();
    for (std::size_t r = 0; r < rank; ++r)
        for (int c = 0; c < n; ++c)
            if (data[r * n + c]) parities[r].set(c);
    return parities;
}

// Save parities to .npy matrix [rank × n] (row-major)
inline void save_parities(const std::vector<BitArr>& parities, int n, const std::string& path) {
    std::size_t rank = parities.size();
    std::vector<uint8_t> data(rank * n, 0);
    for (std::size_t r = 0; r < rank; ++r)
        for (int c = 0; c < n; ++c)
            data[r * n + c] = parities[r].test(c) ? 1 : 0;

    fs::create_directories(fs::path(path).parent_path());
    cnpy::npy_save(path, data.data(), {rank, static_cast<std::size_t>(n)}, "w");
}

// -----------------------------------------------------------------------------
// Waring file discovery
// -----------------------------------------------------------------------------

struct WaringFileInfo {
    std::string id;
    int rank;
    fs::path path;
};

inline std::string waring_filename_regex() {
    std::ostringstream oss;
    oss << R"((\d{)" << paths::ID_WIDTH << R"(})-(\d{)" << paths::RANK_WIDTH << R"(}))";
    if constexpr (paths::HAS_RUN)
        oss << "-" << std::setw(3) << std::setfill('0') << paths::RUN_VAL;
    oss << R"(\.npy)";
    return oss.str();
}

inline std::optional<WaringFileInfo> parse_waring_filename(const fs::path& p) {
    const std::string fname = p.filename().string();
    static const std::regex pat(waring_filename_regex());
    std::smatch m;
    if (!std::regex_match(fname, m, pat)) return std::nullopt;
    return WaringFileInfo{m[1].str(), std::stoi(m[2].str()), p};
}

inline std::vector<WaringFileInfo> find_waring_files(const std::string& id) {
    const std::string idp = paths::pad_id(id);
    std::vector<WaringFileInfo> result;
    const fs::path dir(paths::WARING_DIR);
    if (!fs::exists(dir)) return result;

    for (const auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        if (auto info = parse_waring_filename(entry.path()); info && info->id == idp)
            result.push_back(*info);
    }
    std::sort(result.begin(), result.end(),
              [](const auto& a, const auto& b) { return a.rank < b.rank; });
    return result;
}

inline std::optional<int> find_min_waring_rank(const std::string& id) {
    auto files = find_waring_files(id);
    return files.empty() ? std::nullopt : std::optional{files.front().rank};
}

inline std::optional<WaringFileInfo> find_best_waring_file(const std::string& id) {
    auto files = find_waring_files(id);
    return files.empty() ? std::nullopt : std::optional{files.front()};
}

// -----------------------------------------------------------------------------
// Waring loading
// -----------------------------------------------------------------------------

struct LoadedWaring {
    std::vector<BitArr> parities;
    int n;
    int rank;
    std::string path;
};

inline std::optional<LoadedWaring> load_waring(const std::string& id, int rank = -1) {
    std::optional<WaringFileInfo> info;
    if (rank < 0) {
        info = find_best_waring_file(id);
    } else {
        for (const auto& f : find_waring_files(id))
            if (f.rank == rank) { info = f; break; }
    }
    if (!info) return std::nullopt;

    LoadedWaring result;
    result.parities = load_waring_parities(info->path.string(), result.n);
    result.rank = info->rank;
    result.path = fs::absolute(info->path).string();
    return result;
}

// -----------------------------------------------------------------------------
// Parity generators for different term degrees
// -----------------------------------------------------------------------------

// Degree of sorted triple (i,j,k) with i ≤ j ≤ k
inline int triple_degree(int i, int j, int k) {
    if (i == j && j == k) return 1;  // linear: (i,i,i)
    if (i == j || j == k) return 2;  // quadratic: (i,i,k) or (i,k,k)
    return 3;                         // cubic: (i,j,k) distinct
}

// Linear term (degree 1): single basis vector e_i
inline BitArr make_one(int i) {
    BitArr p{};
    p.set(i);
    return p;
}

// Quadratic term (degree 2): {e_a, e_b, e_a + e_b} for a != b
inline std::array<BitArr, 3> make_three(int a, int b) {
    BitArr ea{}, eb{}, eab{};
    ea.set(a);
    eb.set(b);
    eab.set(a);
    eab.set(b);
    return {ea, eb, eab};
}

// Cubic term (degree 3): 7 parities from outer product
inline std::array<BitArr, 7> make_seven(const BitArr& u, const BitArr& v, const BitArr& w) {
    return {u, v, w, u ^ v, u ^ w, v ^ w, u ^ v ^ w};
}

// -----------------------------------------------------------------------------
// Tensor to Waring expansion (supports all degrees)
// -----------------------------------------------------------------------------

inline std::vector<BitArr> tensor_to_waring(const SymTensor& T) {
    std::vector<BitArr> parities;
    parities.reserve(T.triples.size() * 7);
    
    for (const auto& triple : T.triples) {
        int i = triple[0], j = triple[1], k = triple[2];
        int deg = triple_degree(i, j, k);
        
        if (deg == 1) {
            parities.push_back(make_one(i));
        } else if (deg == 2) {
            int a = (i == j) ? i : j;
            int b = (i == j) ? k : i;
            auto three = make_three(a, b);
            for (const auto& p : three) parities.push_back(p);
        } else {
            BitArr u{}, v{}, w{};
            u.set(i); v.set(j); w.set(k);
            auto seven = make_seven(u, v, w);
            for (const auto& p : seven) parities.push_back(p);
        }
    }
    return parities;
}

} // namespace waring