// bco.cpp
// Beam search to minimize nnz of cubic GF(2) tensor via transvections.
// Packed triples as uint32_t for faster hashing.
// Logs progress to data/logs/bco-{id_in}-{id_out}.json

#include "core/paths.h"
#include "core/bit_vec.h"

#include "CLI11.hpp"
#include "cnpy.h"
#include "picojson.h"
#include "unordered_dense.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <string>
#include <filesystem>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <random>
#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <thread>

namespace fs = std::filesystem;

// Packed Triple: 9 bits per index (n <= 512)
using Triple = std::uint32_t;
constexpr Triple INVALID = 0xFFFFFFFF;

inline Triple pack(int i, int j, int k) {
    return (std::uint32_t(i) << 18) | (std::uint32_t(j) << 9) | std::uint32_t(k);
}
inline int ti(Triple t) { return int(t >> 18); }
inline int tj(Triple t) { return int((t >> 9) & 0x1FF); }
inline int tk(Triple t) { return int(t & 0x1FF); }

inline Triple sorted(int a, int b, int c) {
    if (a > b) std::swap(a, b);
    if (b > c) std::swap(b, c);
    if (a > b) std::swap(a, b);
    return (a == b || b == c) ? INVALID : pack(a, b, c);
}

static fs::path tensor_path(const std::string& id) {
    return fs::path(paths::TENSORS_DIR) / (id + ".npy");
}

static fs::path transform_path(const std::string& id_in, const std::string& id_out) {
    return fs::path(paths::TRANSFORM_DIR) / (id_in + "-" + id_out + ".npy");
}

struct TensorData {
    int n;
    std::vector<std::array<int, 3>> triples;
};

static TensorData load_tensor(const std::string& path) {
    cnpy::NpyArray arr = cnpy::npy_load(path);
    if (arr.shape.size() != 2 || arr.shape[1] != 3 || arr.shape[0] < 1)
        throw std::runtime_error("Invalid tensor format");

    auto read = [&](auto* ptr) -> TensorData {
        TensorData t;
        t.n = int(ptr[0]);
        for (std::size_t r = 1; r < arr.shape[0]; ++r)
            t.triples.push_back({int(ptr[r*3]), int(ptr[r*3+1]), int(ptr[r*3+2])});
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

class TensorState {
public:
    int n;
    ankerl::unordered_dense::set<Triple> triples;
    std::vector<ankerl::unordered_dense::set<Triple>> contains;

    TensorState(const TensorData& T) : n(T.n), contains(T.n) {
        for (auto [i, j, k] : T.triples) {
            Triple t = pack(i, j, k);
            triples.insert(t);
            contains[i].insert(t);
            contains[j].insert(t);
            contains[k].insert(t);
        }
    }

    std::size_t nnz() const { return triples.size(); }

    int compute_delta(int i, int j) const {
        ankerl::unordered_dense::map<Triple, int> changes;

        for (Triple t : contains[i]) {
            int a = ti(t), b = tj(t), c = tk(t);
            int cnt = (a == i) + (b == i) + (c == i);
            std::array<int, 3> idx = {a, b, c}, pos;
            int pi = 0;
            for (int p = 0; p < 3; ++p)
                if (idx[p] == i) pos[pi++] = p;

            for (int mask = 1; mask < (1 << cnt); ++mask) {
                auto nidx = idx;
                for (int bit = 0; bit < cnt; ++bit)
                    if (mask & (1 << bit)) nidx[pos[bit]] = j;
                Triple nt = sorted(nidx[0], nidx[1], nidx[2]);
                if (nt != INVALID) changes[nt] ^= 1;
            }
        }

        int delta = 0;
        for (auto [t, parity] : changes)
            if (parity) delta += triples.contains(t) ? -1 : 1;
        return delta;
    }

    void apply(int i, int j) {
        ankerl::unordered_dense::map<Triple, int> changes;

        for (Triple t : contains[i]) {
            int a = ti(t), b = tj(t), c = tk(t);
            int cnt = (a == i) + (b == i) + (c == i);
            std::array<int, 3> idx = {a, b, c}, pos;
            int pi = 0;
            for (int p = 0; p < 3; ++p)
                if (idx[p] == i) pos[pi++] = p;

            for (int mask = 1; mask < (1 << cnt); ++mask) {
                auto nidx = idx;
                for (int bit = 0; bit < cnt; ++bit)
                    if (mask & (1 << bit)) nidx[pos[bit]] = j;
                Triple nt = sorted(nidx[0], nidx[1], nidx[2]);
                if (nt != INVALID) changes[nt] ^= 1;
            }
        }

        for (auto [t, parity] : changes) {
            if (!parity) continue;
            int a = ti(t), b = tj(t), c = tk(t);
            if (triples.contains(t)) {
                triples.erase(t);
                contains[a].erase(t);
                contains[b].erase(t);
                contains[c].erase(t);
            } else {
                triples.insert(t);
                contains[a].insert(t);
                contains[b].insert(t);
                contains[c].insert(t);
            }
        }
    }

    std::vector<std::array<int, 3>> export_triples() const {
        std::vector<std::array<int, 3>> result;
        result.reserve(triples.size());
        for (Triple t : triples) result.push_back({ti(t), tj(t), tk(t)});
        std::sort(result.begin(), result.end());
        return result;
    }

    std::uint64_t hash() const {
        auto exp = export_triples();
        return ankerl::unordered_dense::detail::wyhash::hash(exp.data(), exp.size() * 12);
    }

    std::vector<int> active_indices() const {
        std::vector<int> active;
        for (int i = 0; i < n; ++i)
            if (!contains[i].empty()) active.push_back(i);
        return active;
    }
};

class TransformMatrix {
public:
    int n;
    std::vector<BitVec> cols;

    TransformMatrix(int n_) : n(n_), cols(n_) {
        for (int i = 0; i < n; ++i) {
            cols[i].resize_bits(n);
            cols[i].set(i);
        }
    }

    void apply(int i, int j) { cols[i] ^= cols[j]; }

    std::vector<uint8_t> export_row_major() const {
        std::vector<uint8_t> data(n * n);
        for (int r = 0; r < n; ++r)
            for (int c = 0; c < n; ++c)
                data[r * n + c] = cols[c].test(r) ? 1 : 0;
        return data;
    }
};

struct BeamState {
    TensorState tensor;
    TransformMatrix transform;
    BeamState(const TensorData& T) : tensor(T), transform(T.n) {}
};

struct Candidate {
    int beam_idx, i, j;
    std::size_t new_nnz;
    bool operator<(const Candidate& o) const { return new_nnz < o.new_nnz; }
};

// -----------------------------------------------------------------------------
// Verification
// -----------------------------------------------------------------------------

static bool verify(const TensorData& orig, const std::vector<std::array<int, 3>>& new_triples,
                   const TransformMatrix& A) {
    ankerl::unordered_dense::set<Triple> reconstructed;
    for (auto [ip, jp, kp] : new_triples) {
        auto si = A.cols[ip].ones(A.n);
        auto sj = A.cols[jp].ones(A.n);
        auto sk = A.cols[kp].ones(A.n);
        for (auto a : si)
            for (auto b : sj)
                for (auto c : sk) {
                    Triple t = sorted(int(a), int(b), int(c));
                    if (t == INVALID) continue;
                    if (reconstructed.contains(t)) reconstructed.erase(t);
                    else reconstructed.insert(t);
                }
    }
    if (reconstructed.size() != orig.triples.size()) return false;
    for (auto [i, j, k] : orig.triples)
        if (!reconstructed.contains(pack(i, j, k))) return false;
    return true;
}

// -----------------------------------------------------------------------------
// Save functions
// -----------------------------------------------------------------------------

static void ensure_dir(const fs::path& dir) {
    std::error_code ec;
    if (!fs::exists(dir)) fs::create_directories(dir, ec);
}

static void save_tensor(const std::vector<std::array<int, 3>>& triples, int n,
                        const fs::path& path) {
    ensure_dir(path.parent_path());
    std::vector<int32_t> data((triples.size() + 1) * 3);
    data[0] = data[1] = data[2] = n;
    for (std::size_t r = 0; r < triples.size(); ++r) {
        data[(r+1)*3] = triples[r][0];
        data[(r+1)*3+1] = triples[r][1];
        data[(r+1)*3+2] = triples[r][2];
    }
    cnpy::npy_save(path.string(), data.data(), {triples.size() + 1, 3});
}

static void save_transform(const TransformMatrix& A, const fs::path& path) {
    ensure_dir(path.parent_path());
    auto data = A.export_row_major();
    cnpy::npy_save(path.string(), data.data(), {std::size_t(A.n), std::size_t(A.n)});
}

// -----------------------------------------------------------------------------
// Candidate generation
// -----------------------------------------------------------------------------

static std::vector<Candidate> generate_candidates(const std::vector<BeamState>& beam, 
                                                   int n, int threads) {
    std::vector<std::vector<Candidate>> local(threads);
    std::vector<std::thread> pool;

    for (int t = 0; t < threads; ++t)
        pool.emplace_back([&, t] {
            for (int b = t; b < int(beam.size()); b += threads) {
                const auto& state = beam[b].tensor;
                std::size_t base = state.nnz();
                auto active = state.active_indices();
                local[t].reserve(local[t].size() + active.size() * n);

                for (int i : active) {
                    for (int j = 0; j < n; ++j) {
                        if (i == j) continue;
                        int delta = state.compute_delta(i, j);
                        local[t].push_back({b, i, j, std::size_t(std::ptrdiff_t(base) + delta)});
                    }
                }
            }
        });

    for (auto& t : pool) t.join();

    std::vector<Candidate> result;
    for (auto& v : local)
        result.insert(result.end(), std::make_move_iterator(v.begin()), 
                      std::make_move_iterator(v.end()));
    return result;
}

// -----------------------------------------------------------------------------
// JSON Logger
// -----------------------------------------------------------------------------

struct BcoLogger {
    bool enabled = false;
    fs::path out_path;
    picojson::object root;
    picojson::array progress;

    static std::string iso8601_utc_now() {
        auto now = std::chrono::system_clock::now();
        std::time_t t = std::chrono::system_clock::to_time_t(now);
        std::tm tm{};
#if defined(_WIN32)
        gmtime_s(&tm, &t);
#else
        tm = *std::gmtime(&t);
#endif
        std::ostringstream oss;
        oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
        return oss.str();
    }

    static std::string make_run_id() {
        std::mt19937_64 rng{std::random_device{}()};
        std::ostringstream oss;
        oss << std::hex << ((rng() ^ (rng() << 13)) & 0xffffffffffULL);
        return oss.str();
    }

    void begin(const std::string& id_in, const std::string& id_out,
               int beam_width, int threads, int patience,
               int n, std::size_t initial_nnz, bool enable_logging) {
        enabled = enable_logging;
        if (!enabled) return;

        out_path = fs::path(paths::LOGS_DIR) / ("bco-" + id_in + "-" + id_out + ".json");
        ensure_dir(out_path.parent_path());

        root["id_in"] = picojson::value(id_in);
        root["id_out"] = picojson::value(id_out);

        picojson::object tensor;
        tensor["n"] = picojson::value(static_cast<double>(n));
        tensor["initial_nnz"] = picojson::value(static_cast<double>(initial_nnz));
        root["tensor"] = picojson::value(tensor);

        picojson::object cfg;
        cfg["beam_width"] = picojson::value(static_cast<double>(beam_width));
        cfg["threads"] = picojson::value(static_cast<double>(threads));
        cfg["patience"] = picojson::value(static_cast<double>(patience));
        root["config"] = picojson::value(cfg);

        picojson::object run;
        run["run_id"] = picojson::value(make_run_id());
        run["started_at"] = picojson::value(iso8601_utc_now());
        run["status"] = picojson::value("running");
        run["last_update"] = picojson::value(iso8601_utc_now());
        root["run"] = picojson::value(run);

        write_();
    }

    void add_iteration(int iter, std::size_t nnz, bool improved, double time_sec) {
        if (!enabled) return;

        picojson::object rec;
        rec["iter"] = picojson::value(static_cast<double>(iter));
        rec["nnz"] = picojson::value(static_cast<double>(nnz));
        rec["improved"] = picojson::value(improved);
        rec["time_sec"] = picojson::value(time_sec);
        progress.push_back(picojson::value(rec));

        root["run"].get<picojson::object>()["last_update"] = picojson::value(iso8601_utc_now());
        write_();
    }

    void finish(std::size_t initial_nnz, std::size_t final_nnz, int iterations,
                double total_time, bool verified, bool saved,
                const std::string& tensor_path, const std::string& transform_path) {
        if (!enabled) return;

        picojson::object& run = root["run"].get<picojson::object>();
        run["status"] = picojson::value("finished");
        run["last_update"] = picojson::value(iso8601_utc_now());

        picojson::object result;
        result["initial_nnz"] = picojson::value(static_cast<double>(initial_nnz));
        result["final_nnz"] = picojson::value(static_cast<double>(final_nnz));
        result["reduction"] = picojson::value(static_cast<double>(initial_nnz - final_nnz));
        result["reduction_pct"] = picojson::value(100.0 * (1.0 - double(final_nnz) / initial_nnz));
        result["iterations"] = picojson::value(static_cast<double>(iterations));
        result["total_time_sec"] = picojson::value(total_time);
        result["verified"] = picojson::value(verified);
        result["saved"] = picojson::value(saved);
        if (saved) {
            result["tensor_path"] = picojson::value(tensor_path);
            result["transform_path"] = picojson::value(transform_path);
        }
        root["result"] = picojson::value(result);

        write_();
    }

    fs::path path() const { return out_path; }

private:
    void write_() {
        root["progress"] = picojson::value(progress);
        std::ofstream ofs(out_path);
        if (ofs) ofs << picojson::value(root).serialize(true);
    }
};

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

int main(int argc, char** argv) {
    CLI::App app{"bco: Beam search to minimize nnz of GF(2) cubic tensor"};

    int tensor_id = 0;
    int output_id = -1;
    int beam_width = 1;
    int patience = 5;
    int threads = 8;
    bool do_save = false;
    bool do_verify = false;
    bool do_log = false;
    bool verbose = false;

    app.add_option("tensor_id", tensor_id, "Input tensor ID")->required();
    app.add_option("-o,--output", output_id, "Output tensor ID (default: input + 1000)");
    app.add_option("-b,--beam", beam_width, "Beam width")->default_val(1);
    app.add_option("--patience", patience, "Stop after N iterations without improvement")->default_val(5);
    app.add_option("-t,--threads", threads, "Number of threads")->default_val(8);
    app.add_flag("--save", do_save, "Save output tensor and transform");
    app.add_flag("--verify", do_verify, "Verify result");
    app.add_flag("--log", do_log, "Write JSON log to data/logs/");
    app.add_flag("-v,--verbose", verbose, "Print progress each iteration");

    CLI11_PARSE(app, argc, argv);

    threads = std::max(1, threads);
    if (output_id < 0) output_id = tensor_id + 1000;

    const std::string id_in = paths::pad_id(tensor_id);
    const std::string id_out = paths::pad_id(output_id);

    auto path = tensor_path(id_in);
    if (!fs::exists(path)) {
        std::cerr << "Error: " << path << " not found\n";
        return 1;
    }

    std::cout << "Loading tensor " << id_in << "\n";
    TensorData T = load_tensor(path.string());
    std::cout << "n=" << T.n << " nnz=" << T.triples.size()
              << " beam=" << beam_width << " threads=" << threads << "\n\n";

    // Init logger
    BcoLogger logger;
    logger.begin(id_in, id_out, beam_width, threads, patience, 
                 T.n, T.triples.size(), do_log);

    std::vector<BeamState> beam;
    beam.emplace_back(T);

    std::size_t initial_nnz = beam[0].tensor.nnz();
    std::size_t best_nnz = initial_nnz;
    auto t_start = std::chrono::steady_clock::now();

    int iter = 0, no_improve = 0;
    while (no_improve < patience) {
        ++iter;
        auto t0 = std::chrono::steady_clock::now();

        auto candidates = generate_candidates(beam, T.n, threads);
        std::size_t take = std::min<std::size_t>(beam_width * 2, candidates.size());
        std::partial_sort(candidates.begin(), candidates.begin() + take, candidates.end());

        std::vector<BeamState> new_beam;
        new_beam.reserve(beam_width);
        ankerl::unordered_dense::set<std::uint64_t> seen;

        for (std::size_t c = 0; c < candidates.size() && int(new_beam.size()) < beam_width; ++c) {
            auto& cand = candidates[c];
            BeamState ns = beam[cand.beam_idx];
            ns.tensor.apply(cand.i, cand.j);
            ns.transform.apply(cand.i, cand.j);

            auto h = ns.tensor.hash();
            if (seen.contains(h)) continue;
            seen.insert(h);
            new_beam.push_back(std::move(ns));
        }

        std::size_t iter_best = new_beam[0].tensor.nnz();
        for (auto& s : new_beam) iter_best = std::min(iter_best, s.tensor.nnz());

        double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        bool improved = iter_best < best_nnz;

        if (improved) {
            best_nnz = iter_best;
            no_improve = 0;
        } else {
            ++no_improve;
        }

        logger.add_iteration(iter, iter_best, improved, sec);

        if (verbose) {
            std::cout << "  Iter " << iter << ": nnz=" << iter_best;
            if (improved) std::cout << " *";
            std::cout << " (" << std::fixed << std::setprecision(2) << sec << "s)\n";
        }

        beam = std::move(new_beam);
    }

    double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();

    // Find best in final beam
    std::size_t final_nnz = beam[0].tensor.nnz();
    int best_idx = 0;
    for (int b = 0; b < int(beam.size()); ++b)
        if (beam[b].tensor.nnz() < final_nnz) { 
            final_nnz = beam[b].tensor.nnz(); 
            best_idx = b; 
        }

    auto new_triples = beam[best_idx].tensor.export_triples();

    // Verify
    bool verified = false;
    if (do_verify)
        verified = verify(T, new_triples, beam[best_idx].transform);

    // Save
    std::string saved_tensor_path, saved_transform_path;
    if (do_save) {
        auto tp = tensor_path(id_out);
        auto xp = transform_path(id_in, id_out);
        save_tensor(new_triples, T.n, tp);
        save_transform(beam[best_idx].transform, xp);
        saved_tensor_path = fs::absolute(tp).string();
        saved_transform_path = fs::absolute(xp).string();
    }

    logger.finish(initial_nnz, final_nnz, iter, elapsed, verified, do_save,
                  saved_tensor_path, saved_transform_path);

    // Output
    std::cout << "\n=== BCO " << id_in << " -> " << id_out << " ===\n"
              << "Initial:   " << initial_nnz << "\n"
              << "Final:     " << final_nnz << (do_verify ? (verified ? " (ok)" : " (FAIL)") : "") << "\n"
              << "Reduction: " << (initial_nnz - final_nnz) << " ("
              << std::fixed << std::setprecision(1)
              << 100.0 * (1.0 - double(final_nnz) / initial_nnz) << "%)\n"
              << "Iters:     " << iter << "\n"
              << "Time:      " << std::setprecision(2) << elapsed << "s\n";

    if (do_save) {
        std::cout << "Tensor:    " << saved_tensor_path << "\n"
                  << "Transform: " << saved_transform_path << "\n";
    }

    if (do_log)
        std::cout << "Log:       " << fs::absolute(logger.path()).string() << "\n";

    return 0;
}