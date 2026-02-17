// cpd/pool.h
// Utilities for saving/loading GF(2) tensor scheme pools.
// Filename format: {id:04d}-{rank:05d}.npy (no RUN) or {id:04d}-{rank:05d}-{run:03d}.npy (with -DRUN=run)

#pragma once

#include <string>
#include <vector>
#include <stdexcept>
#include <filesystem>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <system_error>
#include <regex>
#include <optional>

#include "cnpy.h"
#include "core/paths.h"
#include "core/bit_arr.h"
#include "cpd/config.h"

namespace cpd {

// -----------------------------------------------------------------------------
// Path helpers
// -----------------------------------------------------------------------------

inline void ensure_dir_exists(const std::filesystem::path& dir) {
    std::error_code ec;
    if (!std::filesystem::exists(dir) && !std::filesystem::create_directories(dir, ec))
        throw std::runtime_error("Failed to create directory: " + dir.string());
}

inline std::filesystem::path schemes_dir() {
    return std::filesystem::path(cpd::SCHEMES_DIR);
}

inline std::filesystem::path init_dir() {
    return std::filesystem::path(paths::INIT_DIR);
}

inline std::filesystem::path scheme_file_path(const std::string& id, int rank) {
    std::ostringstream oss;
    oss << paths::pad_id(id) << "-" << std::setw(5) << std::setfill('0') << rank;
    paths::append_run_suffix(oss);
    oss << ".npy";
    return schemes_dir() / oss.str();
}

inline std::filesystem::path previous_best_path(const std::string& id, int new_rank) {
    return scheme_file_path(id, new_rank + 1);
}

// Init scheme path (no RUN suffix)
inline std::filesystem::path init_scheme_file_path(const std::string& id, int rank) {
    std::ostringstream oss;
    oss << paths::pad_id(id) << "-" << std::setw(5) << std::setfill('0') << rank << ".npy";
    return init_dir() / oss.str();
}

// -----------------------------------------------------------------------------
// BitArr <-> U64 serialization
// -----------------------------------------------------------------------------

inline void flatten_vec(const BitArr& v, std::vector<U64>& out) {
    for (std::size_t i = 0; i < BitArr::word_count(); ++i)
        out.push_back(v[i]);
}

inline void flatten_scheme(const std::vector<BitArr>& scheme, std::vector<U64>& out) {
    out.clear();
    out.reserve(scheme.size() * BitArr::word_count());
    for (const auto& v : scheme)
        flatten_vec(v, out);
}

inline BitArr unflatten_vec(const U64* ptr) {
    BitArr v{};
    for (std::size_t i = 0; i < BitArr::word_count(); ++i)
        v[i] = ptr[i];
    return v;
}

inline void unflatten_scheme(const U64* ptr, std::size_t num_vecs, std::vector<BitArr>& out) {
    out.clear();
    out.reserve(num_vecs);
    constexpr std::size_t W = BitArr::word_count();
    for (std::size_t i = 0; i < num_vecs; ++i)
        out.push_back(unflatten_vec(ptr + i * W));
}

// -----------------------------------------------------------------------------
// Scheme file discovery and loading
// -----------------------------------------------------------------------------

struct SchemeFileInfo {
    std::string id;
    int rank;
    int run = -1;  // -1 means no run suffix
    std::filesystem::path path;
};

inline std::optional<SchemeFileInfo> parse_scheme_filename(const std::filesystem::path& p) {
    const std::string fname = p.filename().string();
    static const std::regex pat(paths::scheme_filename_regex());
    std::smatch m;
    if (!std::regex_match(fname, m, pat)) return std::nullopt;
    SchemeFileInfo info;
    info.id = m[1].str();
    info.rank = std::stoi(m[2].str());
    info.path = p;
    return info;
}

// Parse scheme filename with optional run suffix: {id:04d}-{rank:05d}[-{run:03d}].npy
inline std::optional<SchemeFileInfo> parse_scheme_filename_any_run(const std::filesystem::path& p) {
    const std::string fname = p.filename().string();
    static const std::regex pat(R"((\d{4})-(\d{5})(?:-(\d{3}))?\.npy)");
    std::smatch m;
    if (!std::regex_match(fname, m, pat)) return std::nullopt;
    SchemeFileInfo info;
    info.id = m[1].str();
    info.rank = std::stoi(m[2].str());
    info.run = m[3].matched ? std::stoi(m[3].str()) : -1;
    info.path = p;
    return info;
}

// Parse init filename (no RUN suffix): {id:04d}-{rank:05d}.npy
inline std::optional<SchemeFileInfo> parse_init_filename(const std::filesystem::path& p) {
    const std::string fname = p.filename().string();
    static const std::regex pat(R"((\d{4})-(\d{5})\.npy)");
    std::smatch m;
    if (!std::regex_match(fname, m, pat)) return std::nullopt;
    SchemeFileInfo info;
    info.id = m[1].str();
    info.rank = std::stoi(m[2].str());
    info.path = p;
    return info;
}

inline std::vector<SchemeFileInfo> find_scheme_files(const std::string& id) {
    const std::string idp = paths::pad_id(id);
    std::vector<SchemeFileInfo> result;
    const auto dir = schemes_dir();
    if (!std::filesystem::exists(dir)) return result;

    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        if (auto info = parse_scheme_filename(entry.path()); info && info->id == idp)
            result.push_back(*info);
    }
    std::sort(result.begin(), result.end(),
              [](const auto& a, const auto& b) { return a.rank < b.rank; });
    return result;
}

inline std::optional<SchemeFileInfo> find_best_scheme_file(const std::string& id) {
    auto files = find_scheme_files(id);
    return files.empty() ? std::nullopt : std::optional{files.front()};
}

// Find all scheme files for given (id, rank) across all runs
inline std::vector<SchemeFileInfo> find_all_files_for_rank(const std::string& id, int rank) {
    const std::string idp = paths::pad_id(id);
    std::vector<SchemeFileInfo> result;
    const auto dir = schemes_dir();
    if (!std::filesystem::exists(dir)) return result;

    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        auto info = parse_scheme_filename_any_run(entry.path());
        if (info && info->id == idp && info->rank == rank)
            result.push_back(*info);
    }
    // Sort: no-run first, then by run id
    std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) {
        if (a.run < 0 && b.run >= 0) return true;
        if (a.run >= 0 && b.run < 0) return false;
        return a.run < b.run;
    });
    return result;
}

// Find init scheme file for given id (exactly one file expected)
inline std::optional<SchemeFileInfo> find_init_scheme_file(const std::string& id) {
    const std::string idp = paths::pad_id(id);
    const auto dir = init_dir();
    if (!std::filesystem::exists(dir)) return std::nullopt;

    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        if (auto info = parse_init_filename(entry.path()); info && info->id == idp)
            return info;
    }
    return std::nullopt;
}

// Find minimum rank available for given id
inline std::optional<int> find_min_rank(const std::string& id) {
    auto files = find_scheme_files(id);
    if (files.empty()) return std::nullopt;
    return files.front().rank;
}

// Loads pool from .npy file. Shape: (pool_size, 3 * rank * VEC_WORDS).
inline std::vector<std::vector<BitArr>> load_pool_from_npy(const std::filesystem::path& npy_path) {
    cnpy::NpyArray arr = cnpy::npy_load(npy_path.string());
    if (arr.shape.size() != 2)
        throw std::runtime_error("Pool .npy must be 2D.");

    const std::size_t pool_size = arr.shape[0];
    const std::size_t row_len = arr.shape[1];
    constexpr std::size_t W = BitArr::word_count();

    if (row_len % (3 * W) != 0)
        throw std::runtime_error("Pool row length must be divisible by 3*VEC_WORDS.");

    const std::size_t num_vecs = row_len / W;

    std::vector<std::vector<BitArr>> pool;
    pool.reserve(pool_size);

    auto read_pool = [&](auto* ptr) {
        std::vector<U64> row_u64(row_len);
        for (std::size_t r = 0; r < pool_size; ++r) {
            for (std::size_t c = 0; c < row_len; ++c)
                row_u64[c] = static_cast<U64>(ptr[r * row_len + c]);
            std::vector<BitArr> row;
            unflatten_scheme(row_u64.data(), num_vecs, row);
            pool.push_back(std::move(row));
        }
    };

    switch (arr.word_size) {
        case 1: read_pool(arr.data<uint8_t>()); break;
        case 2: read_pool(arr.data<uint16_t>()); break;
        case 4: read_pool(arr.data<uint32_t>()); break;
        case 8: read_pool(arr.data<uint64_t>()); break;
        default: throw std::runtime_error("Unsupported word size in pool .npy.");
    }
    return pool;
}

struct LoadedPool {
    std::vector<std::vector<BitArr>> pool;
    int rank;
    std::string path;
};

inline std::optional<LoadedPool> load_best_pool(const std::string& id) {
    auto best = find_best_scheme_file(id);
    if (!best) return std::nullopt;

    LoadedPool result;
    result.pool = load_pool_from_npy(best->path);
    result.rank = best->rank;
    result.path = std::filesystem::absolute(best->path).string();
    return result;
}

// Load init scheme (single scheme from data/init/)
inline std::optional<LoadedPool> load_init_scheme(const std::string& id) {
    auto info = find_init_scheme_file(id);
    if (!info) return std::nullopt;

    LoadedPool result;
    result.pool = load_pool_from_npy(info->path);
    result.rank = info->rank;
    result.path = std::filesystem::absolute(info->path).string();
    return result;
}

struct MergedPoolInfo {
    std::vector<std::vector<BitArr>> pool;
    std::size_t num_files = 0;
    std::vector<std::string> source_files;
};

// Load and merge pools from all files for (id, rank) across all runs
inline std::optional<MergedPoolInfo> load_merged_pool(const std::string& id, int rank) {
    auto files = find_all_files_for_rank(id, rank);
    if (files.empty()) return std::nullopt;

    MergedPoolInfo result;
    result.num_files = files.size();
    result.source_files.reserve(files.size());

    for (const auto& info : files) {
        auto pool = load_pool_from_npy(info.path);
        result.source_files.push_back(info.path.filename().string());
        for (auto& scheme : pool)
            result.pool.push_back(std::move(scheme));
    }
    return result;
}

// -----------------------------------------------------------------------------
// Scheme helpers
// -----------------------------------------------------------------------------

inline int count_live_terms(const std::vector<BitArr>& scheme) {
    const int r = static_cast<int>(scheme.size() / 3);
    int cnt = 0;
    for (int t = 0; t < r; ++t)
        if (scheme[3 * t].any()) ++cnt;
    return cnt;
}

inline void take_first_k_live_terms(const std::vector<BitArr>& scheme, int rank, std::vector<BitArr>& out) {
    out.clear();
    out.reserve(static_cast<std::size_t>(3 * rank));
    const int r = static_cast<int>(scheme.size() / 3);
    int taken = 0;
    for (int t = 0; t < r && taken < rank; ++t) {
        if (scheme[3 * t].none()) continue;
        out.push_back(scheme[3 * t + 0]);
        out.push_back(scheme[3 * t + 1]);
        out.push_back(scheme[3 * t + 2]);
        ++taken;
    }
    if (taken != rank)
        throw std::runtime_error("take_first_k_live_terms: fewer live terms than requested.");
}

inline void pack_pool_rows(const std::vector<std::vector<BitArr>>& pool,
                           int rank,
                           std::vector<U64>& flat_out) {
    if (pool.empty()) {
        flat_out.clear();
        return;
    }

    constexpr std::size_t W = BitArr::word_count();
    const std::size_t row_len = static_cast<std::size_t>(3 * rank) * W;
    flat_out.resize(pool.size() * row_len);

    std::vector<BitArr> rowbuf;
    std::vector<U64> flatbuf;

    for (std::size_t r = 0; r < pool.size(); ++r) {
        const auto& sch = pool[r];
        if (static_cast<int>(sch.size()) == 3 * rank && count_live_terms(sch) == rank)
            rowbuf = sch;
        else
            take_first_k_live_terms(sch, rank, rowbuf);

        flatten_scheme(rowbuf, flatbuf);
        if (flatbuf.size() != row_len)
            throw std::runtime_error("pack_pool_rows: internal row length mismatch.");

        std::copy(flatbuf.begin(), flatbuf.end(), flat_out.begin() + r * row_len);
    }
}

inline void atomic_npy_save(const std::filesystem::path& dst_path,
                            const U64* data,
                            const std::vector<std::size_t>& shape) {
    const std::filesystem::path dir = dst_path.parent_path();
    ensure_dir_exists(dir);

    const std::filesystem::path tmp = dst_path.string() + ".tmp";
    cnpy::npy_save(tmp.string(), data, shape);

    std::error_code ec;
    std::filesystem::rename(tmp, dst_path, ec);
    if (ec) {
        std::filesystem::remove(dst_path, ec);
        ec.clear();
        std::filesystem::rename(tmp, dst_path, ec);
        if (ec) {
            cnpy::npy_save(dst_path.string(), data, shape);
            std::filesystem::remove(tmp, ec);
        }
    }
}

// -----------------------------------------------------------------------------
// Saving pools
// -----------------------------------------------------------------------------

inline std::string save_pool(const std::vector<std::vector<BitArr>>& pool,
                             int rank,
                             const std::string& id) {
    if (pool.empty()) return {};

    std::vector<U64> flat;
    pack_pool_rows(pool, rank, flat);

    constexpr std::size_t W = BitArr::word_count();
    const std::filesystem::path out_path = scheme_file_path(id, rank);
    const std::vector<std::size_t> shape{pool.size(), static_cast<std::size_t>(3 * rank) * W};

    atomic_npy_save(out_path, flat.data(), shape);
    return std::filesystem::absolute(out_path).string();
}

inline std::string save_best_pool(const std::vector<std::vector<BitArr>>& pool,
                                  int rank,
                                  const std::string& id) {
    if (pool.empty()) return {};

    const std::string new_path = save_pool(pool, rank, id);

    const std::filesystem::path prev = previous_best_path(id, rank);
    const std::filesystem::path newp = scheme_file_path(id, rank);
    if (prev != newp) {
        std::error_code ec;
        std::filesystem::remove(prev, ec);
    }
    return new_path;
}

} // namespace cpd