// waring.cpp
// Optimize T-count via Waring decomposition and TODD algorithm.
//
// Three modes:
//   Default:    Load tensor → trivial decomposition → TODD
//   --cpd:      Load CPD from pool → Waring → TODD
//   --continue: Load existing Waring decomposition → TODD
//
// Logs progress to data/logs/waring-{id}.json.

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <cmath>
#include <limits>
#include <random>

#include "CLI11.hpp"
#include "picojson.h"
#include "cnpy.h"
#include "fmt.h"
#include "core/paths.h"
#include "core/bit_arr.h"
#include "cpd/topp/seven.h"
#include "cpd/pool.h"
#include "waring/tensor.h"
#include "waring/todd.h"

namespace fs = std::filesystem;

// -----------------------------------------------------------------------------
// Tensor loading
// -----------------------------------------------------------------------------

static fs::path tensor_file_path(const std::string& id) {
    return fs::path(paths::TENSORS_DIR) / (paths::pad_id(id) + ".npy");
}

// -----------------------------------------------------------------------------
// CPD to Waring expansion
// -----------------------------------------------------------------------------

static std::vector<BitArr> cpd_to_waring(const std::vector<BitArr>& cpd) {
    std::vector<BitArr> parities;
    std::size_t num_triples = cpd.size() / 3;
    parities.reserve(num_triples * 7);

    for (std::size_t t = 0; t < num_triples; ++t) {
        const BitArr& u = cpd[3 * t + 0];
        const BitArr& v = cpd[3 * t + 1];
        const BitArr& w = cpd[3 * t + 2];
        if (u.none()) continue;

        auto seven = cpd::topp::make_seven(u, v, w);
        for (const auto& p : seven) parities.push_back(p);
    }
    return parities;
}

// -----------------------------------------------------------------------------
// Utilities
// -----------------------------------------------------------------------------

static std::size_t total_nnz(const std::vector<BitArr>& parities) {
    std::size_t nnz = 0;
    for (const auto& p : parities) nnz += p.count();
    return nnz;
}

static std::size_t cpd_nnz(const std::vector<BitArr>& cpd) {
    std::size_t nnz = 0;
    for (const auto& v : cpd) nnz += v.count();
    return nnz;
}

static fs::path waring_file_path(const std::string& id, std::size_t rank) {
    std::ostringstream oss;
    oss << paths::pad_id(id) << "-" << std::setfill('0') << std::setw(5) << rank;
    paths::append_run_suffix(oss);
    oss << ".npy";
    return fs::path(paths::WARING_DIR) / oss.str();
}

static std::string save_waring(const std::vector<BitArr>& w, int n, const std::string& id) {
    if (w.empty()) return {};
    const fs::path out_path = waring_file_path(id, w.size());
    cpd::ensure_dir_exists(out_path.parent_path());
    waring::save_parities(w, n, out_path.string());
    return fs::absolute(out_path).string();
}

static void remove_previous_waring(const std::string& id, std::size_t old_rank, std::size_t new_rank) {
    if (old_rank == new_rank || old_rank == std::numeric_limits<std::size_t>::max()) return;
    std::error_code ec;
    fs::remove(waring_file_path(id, old_rank), ec);
}

struct Stats {
    std::size_t min_val, max_val;
    double avg, std_dev;

    static Stats compute(const std::vector<std::size_t>& v) {
        if (v.empty()) return {0, 0, 0.0, 0.0};
        std::size_t min_val = *std::min_element(v.begin(), v.end());
        std::size_t max_val = *std::max_element(v.begin(), v.end());
        double sum = std::accumulate(v.begin(), v.end(), 0.0);
        double avg = sum / v.size();
        double sq_sum = 0;
        for (auto x : v) sq_sum += (x - avg) * (x - avg);
        return {min_val, max_val, avg, std::sqrt(sq_sum / v.size())};
    }
};

// -----------------------------------------------------------------------------
// JSON logger
// -----------------------------------------------------------------------------

struct WaringLogger {
    std::string id;
    bool enabled = false;
    picojson::object root;
    picojson::array progress;
    fs::path out_path;

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
        return fmt::format("{:010x}", (rng() ^ (rng() << 13)) & 0xffffffffffULL);
    }

    void begin(const std::string& id_, const std::string& tensor_path_, 
               const std::string& mode, int cpd_rank, int waring_rank,
               std::size_t beam_width, std::size_t threads, bool use_tohpe,
               int num, bool do_verify, bool save_all,
               const std::vector<std::string>& sources, std::size_t total_schemes,
               bool enable_logging) {
        enabled = enable_logging;
        if (!enabled) return;

        id = id_;
        out_path = fs::path(paths::LOGS_DIR) / fmt::format("waring-{}.json", paths::pad_id(id));
        cpd::ensure_dir_exists(out_path.parent_path());

        root["id"] = picojson::value(paths::pad_id(id));
        root["tensor"] = picojson::value(tensor_path_);
        root["mode"] = picojson::value(mode);
        if (cpd_rank > 0)
            root["cpd_rank"] = picojson::value(static_cast<double>(cpd_rank));
        if (waring_rank > 0)
            root["waring_rank"] = picojson::value(static_cast<double>(waring_rank));

        if (!sources.empty()) {
            picojson::array src_arr;
            for (const auto& s : sources) src_arr.push_back(picojson::value(s));
            root["source_files"] = picojson::value(src_arr);
            root["source_schemes"] = picojson::value(static_cast<double>(total_schemes));
        }

        picojson::object cfg;
        cfg["num"] = picojson::value(static_cast<double>(num));
        cfg["beam_width"] = picojson::value(static_cast<double>(beam_width));
        cfg["threads"] = picojson::value(static_cast<double>(threads));
        cfg["use_tohpe"] = picojson::value(use_tohpe);
        cfg["do_verify"] = picojson::value(do_verify);
        cfg["save_all"] = picojson::value(save_all);
        root["config"] = picojson::value(cfg);

        picojson::object run;
        run["run_id"] = picojson::value(make_run_id());
        run["started_at"] = picojson::value(iso8601_utc_now());
        run["status"] = picojson::value("running");
        run["last_update"] = picojson::value(iso8601_utc_now());
        root["run"] = picojson::value(run);

        write_();
    }

    void add_scheme_record(int idx, std::size_t cpd_nnz_val, std::size_t initial_rank,
                           std::size_t waring_rank, std::size_t nnz, double time_sec,
                           bool is_best, bool saved, int verified,
                           const picojson::array& iterations) {
        if (!enabled) return;

        picojson::object rec;
        rec["idx"] = picojson::value(static_cast<double>(idx));
        rec["cpd_nnz"] = picojson::value(static_cast<double>(cpd_nnz_val));
        rec["initial_rank"] = picojson::value(static_cast<double>(initial_rank));
        rec["waring_rank"] = picojson::value(static_cast<double>(waring_rank));
        rec["nnz"] = picojson::value(static_cast<double>(nnz));
        rec["time_sec"] = picojson::value(time_sec);
        rec["is_best"] = picojson::value(is_best);
        if (saved) rec["saved"] = picojson::value(true);
        if (verified >= 0) rec["verified"] = picojson::value(verified == 1);
        if (!iterations.empty()) rec["iterations"] = picojson::value(iterations);
        progress.push_back(picojson::value(rec));

        root["run"].get<picojson::object>()["last_update"] = picojson::value(iso8601_utc_now());
        write_();
    }

    void finish(std::size_t best_rank, std::size_t best_nnz_val, int best_idx,
                const std::string& best_path, double total_time, bool verified,
                const Stats& stats) {
        if (!enabled) return;

        auto& run = root["run"].get<picojson::object>();
        run["status"] = picojson::value("finished");
        run["last_update"] = picojson::value(iso8601_utc_now());

        picojson::object result;
        result["best_rank"] = picojson::value(static_cast<double>(best_rank));
        result["best_nnz"] = picojson::value(static_cast<double>(best_nnz_val));
        result["best_idx"] = picojson::value(static_cast<double>(best_idx));
        result["best_path"] = picojson::value(best_path);
        result["total_time_sec"] = picojson::value(total_time);
        result["verified"] = picojson::value(verified);
        root["result"] = picojson::value(result);

        picojson::object st;
        st["min"] = picojson::value(static_cast<double>(stats.min_val));
        st["max"] = picojson::value(static_cast<double>(stats.max_val));
        st["avg"] = picojson::value(stats.avg);
        st["std_dev"] = picojson::value(stats.std_dev);
        root["stats"] = picojson::value(st);

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

int main(int argc, char* argv[]) {
    CLI::App app{"Waring decomposition optimizer via FastTODD"};

    int tensor_id = -1;
    int cpd_rank = -1;
    int num = 1;
    int beam_width = 1;
    int threads = 8;
    bool use_cpd = false;
    bool use_continue = false;
    bool no_tohpe = false;
    bool do_verify = false;
    bool do_save = false;
    bool save_all = false;
    bool do_log = false;

    app.add_option("id", tensor_id, "Tensor numeric id")->required();
    app.add_option("--cpd", cpd_rank, "Load CPD from pool (optional: specify rank, default: min available)")
       ->expected(0, 1)->default_val(-1);
    app.add_flag("--continue", use_continue, "Continue from existing Waring decomposition");
    app.add_option("-n,--num", num, "Number of CPD schemes to try (--cpd mode only)")->default_val(1);
    app.add_option("-b,--beam", beam_width, "Beam width (1 = greedy)")->default_val(1);
    app.add_option("-t,--threads", threads, "Number of threads")->default_val(4);
    app.add_flag("--no-tohpe", no_tohpe, "Disable TOHPE preprocessing");
    app.add_flag("--verify", do_verify, "Verify result against original tensor");
    app.add_flag("--save", do_save, "Save best Waring decomposition at the end");
    app.add_flag("--save-all", save_all, "Save after each improvement");
    app.add_flag("--log", do_log, "Write JSON log to data/logs/");

    CLI11_PARSE(app, argc, argv);

    // Check if --cpd was provided
    use_cpd = app.count("--cpd") > 0;

    // Validate mutually exclusive flags
    if (use_cpd && use_continue) {
        fmt::print(stderr, "Error: --cpd and --continue are mutually exclusive\n");
        return 1;
    }

    const std::string id = paths::pad_id(tensor_id);
    const fs::path tensor_path = tensor_file_path(id);

    // Load tensor (needed for all modes for verification and n)
    if (!fs::exists(tensor_path)) {
        fmt::print(stderr, "Tensor file not found: {}\n", tensor_path.string());
        return 1;
    }
    waring::SymTensor T = waring::load_tensor(tensor_path.string());
    int n = T.n;
    
    fmt::print("Tensor {}: n={}, nnz={}\n", id, n, T.triples.size());

    bool use_tohpe = !no_tohpe;
    std::string mode;
    std::vector<std::string> source_files;
    std::size_t total_schemes = 0;
    int loaded_waring_rank = -1;
    
    // Prepare schemes to process
    std::vector<std::vector<BitArr>> schemes;
    std::vector<std::size_t> scheme_nnz;  // CPD/source nnz for each scheme

    if (use_continue) {
        // --continue mode: load existing waring decomposition
        mode = "continue";
        
        auto loaded = waring::load_waring(id);
        if (!loaded) {
            fmt::print(stderr, "No Waring files found for id={}\n", id);
            return 1;
        }
        
        if (loaded->n != n) {
            fmt::print(stderr, "Waring dimension mismatch: expected {}, got {}\n", n, loaded->n);
            return 1;
        }
        
        loaded_waring_rank = loaded->rank;
        source_files.push_back(fs::path(loaded->path).filename().string());
        
        fmt::print("Loaded Waring from {} (rank {})\n", loaded->path, loaded->rank);
        
        if (num > 1)
            fmt::print(stderr, "Warning: -n/--num ignored in --continue mode\n");
        
        scheme_nnz.push_back(total_nnz(loaded->parities));
        schemes.push_back(std::move(loaded->parities));
        
    } else if (use_cpd) {
        // --cpd mode: load from CPD pool
        mode = "cpd";
        
        if (cpd_rank < 0) {
            auto min_rank = cpd::find_min_rank(id);
            if (!min_rank) {
                fmt::print(stderr, "No CPD files found for id={}\n", id);
                return 1;
            }
            cpd_rank = *min_rank;
            fmt::print("Using minimum available CPD rank: {}\n", cpd_rank);
        }

        auto merged = cpd::load_merged_pool(id, cpd_rank);
        if (!merged || merged->pool.empty()) {
            fmt::print(stderr, "No CPD files found for id={} rank={}\n", id, cpd_rank);
            return 1;
        }

        source_files = merged->source_files;
        total_schemes = merged->pool.size();

        fmt::print("Loaded {} CPD schemes from {} file(s)\n", total_schemes, merged->num_files);

        int to_try = std::min(num, static_cast<int>(merged->pool.size()));
        for (int i = 0; i < to_try; ++i) {
            scheme_nnz.push_back(cpd_nnz(merged->pool[i]));
            schemes.push_back(cpd_to_waring(merged->pool[i]));
        }
    } else {
        // Default mode: trivial decomposition from tensor
        mode = "trivial";
        
        if (num > 1)
            fmt::print(stderr, "Warning: -n/--num ignored in default mode (only one trivial decomposition)\n");

        scheme_nnz.push_back(T.triples.size());
        schemes.push_back(waring::tensor_to_waring(T));
        
        fmt::print("Using trivial decomposition (rank={})\n", schemes[0].size());
    }

    int to_try = static_cast<int>(schemes.size());

    // Init logger
    WaringLogger logger;
    logger.begin(id, tensor_path.string(), mode, cpd_rank, loaded_waring_rank,
                 beam_width, threads, use_tohpe, to_try,
                 do_verify, save_all, source_files, total_schemes, do_log);

    // Global best tracking
    std::vector<BitArr> global_best_parities;
    std::size_t global_best_rank = std::numeric_limits<std::size_t>::max();
    std::size_t global_best_nnz = std::numeric_limits<std::size_t>::max();
    int global_best_idx = -1;
    std::string global_best_path;

    std::vector<std::size_t> results;
    results.reserve(to_try);

    auto t_total_start = std::chrono::steady_clock::now();

    for (int i = 0; i < to_try; ++i) {
        auto t_start = std::chrono::steady_clock::now();

        auto parities = todd::remove_duplicates(std::move(schemes[i]));
        std::size_t initial_rank = parities.size();

        // Track best for this scheme
        std::size_t scheme_best_rank = initial_rank;
        picojson::array iterations;

        // Progress callback
        auto on_progress = [&](const char* phase, std::size_t iter,
                               std::size_t rank, const std::vector<BitArr>& table) -> bool {
            if (do_log) {
                picojson::object it;
                it["phase"] = picojson::value(phase);
                it["iter"] = picojson::value(static_cast<double>(iter));
                it["rank"] = picojson::value(static_cast<double>(rank));
                iterations.push_back(picojson::value(it));
            }

            if (rank < scheme_best_rank) {
                scheme_best_rank = rank;
                std::size_t nnz = total_nnz(table);

                if (rank < global_best_rank || (rank == global_best_rank && nnz < global_best_nnz)) {
                    std::size_t old_rank = global_best_rank;
                    global_best_rank = rank;
                    global_best_nnz = nnz;
                    global_best_parities = table;
                    global_best_idx = i;

                    if (save_all) {
                        remove_previous_waring(id, old_rank, rank);
                        global_best_path = save_waring(table, n, id);
                        fmt::print("  [{} iter {}] rank {} -> saved\n", phase, iter, rank);
                    }
                }
            }
            return true;
        };

        // Run FastTODD
        parities = todd::toddpp(std::move(parities), n, beam_width, threads, 
                                use_tohpe, on_progress);

        auto t_end = std::chrono::steady_clock::now();
        double iter_time = std::chrono::duration<double>(t_end - t_start).count();

        std::size_t final_rank = parities.size();
        std::size_t final_nnz = total_nnz(parities);
        results.push_back(final_rank);

        // Final check
        bool is_best = final_rank < global_best_rank ||
                       (final_rank == global_best_rank && final_nnz < global_best_nnz);
        bool saved = false;
        int verified = -1;

        if (is_best) {
            std::size_t old_rank = global_best_rank;
            global_best_rank = final_rank;
            global_best_nnz = final_nnz;
            global_best_parities = std::move(parities);
            global_best_idx = i;

            if (save_all) {
                remove_previous_waring(id, old_rank, final_rank);
                global_best_path = save_waring(global_best_parities, n, id);
                saved = true;
            }

            if (do_verify) {
                auto reconstructed = waring::SymTensor::from_parities(global_best_parities, n);
                verified = (reconstructed == T) ? 1 : 0;
            }
        }

        logger.add_scheme_record(i, scheme_nnz[i], initial_rank, final_rank, final_nnz,
                                 iter_time, is_best, saved, verified, iterations);

        fmt::print("[{}/{}] {} -> {} (nnz={}, t={:.2f}s){}\n",
                   i + 1, to_try, initial_rank, final_rank, final_nnz, iter_time,
                   is_best ? " *" : "");
    }

    auto t_total_end = std::chrono::steady_clock::now();
    double total_time = std::chrono::duration<double>(t_total_end - t_total_start).count();

    Stats stats = Stats::compute(results);

    // Final save
    if (do_save && !save_all && !global_best_parities.empty())
        global_best_path = save_waring(global_best_parities, n, id);

    // Final verify
    bool final_verified = true;
    if (do_verify && global_best_idx >= 0 && !save_all) {
        auto reconstructed = waring::SymTensor::from_parities(global_best_parities, n);
        final_verified = (reconstructed == T);
    }

    logger.finish(global_best_rank, global_best_nnz, global_best_idx,
                  global_best_path, total_time, final_verified, stats);

    // Console output
    std::string method = "FastTODD";
    if (beam_width > 1) method += fmt::format(" (beam={})", beam_width);
    if (!use_tohpe) method += " [no TOHPE]";

    fmt::print("\nWARING {}\n", id);
    fmt::print("  Mode    {}", mode);
    if (use_cpd) fmt::print(" (cpd-rank={}, {} schemes)", cpd_rank, to_try);
    if (use_continue) fmt::print(" (waring-rank={})", loaded_waring_rank);
    fmt::print("\n");
    fmt::print("  Method  {}, {} threads\n", method, threads);
    fmt::print("  Result  {:.1f} +- {:.1f} (min {}, max {}, nnz {})\n",
               stats.avg, stats.std_dev, stats.min_val, stats.max_val, global_best_nnz);
    fmt::print("  Time    {:.2f}s", total_time);
    if (to_try > 1) fmt::print(" ({:.3f}s/scheme)", total_time / to_try);
    fmt::print("\n");

    if (!global_best_path.empty())
        fmt::print("  Saved   {}\n", global_best_path);

    if (do_verify)
        fmt::print("  Verify  {}\n", final_verified ? "PASS" : "FAIL");

    if (do_log)
        fmt::print("  Log     {}\n", fs::absolute(logger.path()).string());

    return (do_verify && !final_verified) ? 1 : 0;
}