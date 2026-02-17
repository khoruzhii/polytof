// cpd.cpp
// Pool-based GF(2) tensor flip graph search.
//
// Compile variants:
//   default or -DTOPP  → topp (symmetric 3-tensor)
//   -DBASE             → base (general 3-tensor)  
//   -DCOMM             → comm (commutative variant)
//
// Use -DRUN=N for parallel runs with isolated outputs.

#include "core/types.h"
#include "core/paths.h"
#include "cpd/config.h"
#include "cpd/pool.h"
#include "cpd/walk.h"
#include "cpd/logger.h"

#ifdef TOPP
#include "cpd/topp/sge.h"
#endif

#include "CLI11.hpp"

#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <sstream>
#include <iomanip>
#include <random>
#include <chrono>
#include <array>

namespace fs = std::filesystem;

static std::string dims_str(const std::array<int, 3>& d) {
    std::ostringstream oss;
    oss << d[0] << "x" << d[1] << "x" << d[2];
    return oss.str();
}

static fs::path tensor_path(const std::string& id) {
    return fs::path(paths::TENSORS_DIR) / (paths::pad_id(id) + ".npy");
}

int main(int argc, char* argv[]) {
    CLI::App app{"CPD Flip Graph Search"};

    int tensor_id = -1;
    app.add_option("id", tensor_id, "Tensor numeric id")->required();

    cpd::WalkParams P{};
    bool do_save = false;
    bool do_continue = false;
    bool do_reduce = false;
    bool do_sge = false;
    bool do_fgs = false;
    bool do_log = false;
    int sge_beam = 1;

    // Preprocessing flags (mutually exclusive)
    app.add_flag("--reduce", do_reduce, "Apply simple reduce loop");
    app.add_flag("--sge", do_sge, "Apply SGE beam search (TOPP only)");
    app.add_option("-b,--beam", sge_beam, "SGE beam width")->default_val(1);

    // Main search flag
    app.add_flag("--fgs", do_fgs, "Run flip graph search");

    // FGS parameters
    app.add_option("-f,--path-limit", P.path_limit, "Path length limit")->default_val(P.path_limit);
    app.add_option("-s,--pool-size", P.pool_size, "Pool size limit")->default_val(P.pool_size);
    app.add_option("-p,--plus-lim", P.plus_lim, "Flips before plus transition")->default_val(P.plus_lim);
    app.add_option("-r,--reduce-interval", P.reduce_interval, "Flips between reduce (0=off)")->default_val(P.reduce_interval);
    app.add_option("-t,--threads", P.threads, "Worker threads")->default_val(P.threads);
    app.add_option("-m,--max-attempts", P.max_attempts, "Max attempts per epoch")->default_val(P.max_attempts);
    app.add_flag("--plus", P.use_plus, "Enable plus transitions");
    app.add_flag("--verify", P.do_verify, "Verify candidates");

    // Other flags
    app.add_flag("--save", do_save, "Save pool on improvement");
    app.add_flag("--continue", do_continue, "Resume from saved scheme");
    app.add_flag("--log", do_log, "Write JSON log");

    CLI11_PARSE(app, argc, argv);

    // Validation
    if (do_sge && do_reduce) {
        std::cerr << "Error: --sge and --reduce are mutually exclusive\n";
        return 1;
    }

#ifndef TOPP
    if (do_sge) {
        std::cerr << "Error: --sge is only available for TOPP variant\n";
        return 1;
    }
#endif

    if (!do_sge && !do_reduce && !do_fgs) {
        std::cerr << "Error: need at least one of --sge, --reduce, --fgs\n";
        return 1;
    }

    sge_beam = std::max(1, sge_beam);

    const std::string id = paths::pad_id(tensor_id);
    const fs::path tpath = tensor_path(id);

    if (!fs::exists(tpath)) {
        std::cerr << "Tensor not found: " << tpath << "\n";
        return 1;
    }

    // Random seed
    std::random_device rd;
    U64 seed = (static_cast<U64>(rd()) << 32) ^ static_cast<U64>(rd());

    // Load tensor
    cpd::Tensor T = cpd::load_tensor(tpath.string());
    const auto dims = T.get_dims();

    std::cout << "=== CPD ===\n"
              << "Tensor: " << id;
    if constexpr (paths::HAS_RUN) std::cout << " (run=" << paths::RUN_VAL << ")";
    std::cout << "\nPath: " << tpath.string()
              << "\nDims: " << dims_str(dims) << ", nnz: " << T.triples.size()
              << "\nSeed: " << seed
              << "\nMode:";
    if (do_reduce) std::cout << " reduce";
    if (do_sge) std::cout << " sge(beam=" << sge_beam << ")";
    if (do_fgs) std::cout << " fgs";
    std::cout << "\n\n";

    // Init pool
    std::vector<cpd::SchemeData> pool;
    int current_rank = 0;
    bool continued = false;
    std::string loaded_path;

    if (do_continue) {
        if (auto loaded = cpd::load_best_pool(id); loaded && !loaded->pool.empty()) {
            continued = true;
            current_rank = loaded->rank;
            loaded_path = loaded->path;
            pool.reserve(loaded->pool.size());
            for (auto& sch : loaded->pool) {
                int r = cpd::get_rank(sch);
                pool.emplace_back(std::move(sch), r, dims, 0);
            }
            std::cout << "Continued from " << loaded_path << " (rank " << current_rank << ")\n\n";
        } else {
            std::cout << "No saved schemes found, starting fresh\n\n";
        }
    }

    if (!continued) {
        auto data = cpd::trivial_decomposition(T);
        std::size_t trivial_rank = data.size() / 3;
        current_rank = cpd::get_rank(data);
        pool.emplace_back(std::move(data), current_rank, dims, 0);
        std::cout << "Trivial rank: " << trivial_rank << "\n\n";
    }

    // Logger
    cpd::RunLogger logger;
    logger.begin(id, tpath.string(), dims, T.triples.size(), P,
                 do_save, continued, do_reduce, do_sge, sge_beam, do_fgs, do_log);

    int best_rank = current_rank;
    std::size_t best_pool_size = pool.size();
    std::string best_path = continued ? loaded_path : "";

    // === Preprocessing phase ===
    if (do_reduce || do_sge) {
        std::cout << "=== Preprocess ===\n";

        if (do_reduce) {
            auto t0 = std::chrono::steady_clock::now();
            cpd::Scheme scheme(pool[0].data, dims, seed);
            int initial_rank = scheme.get_rank();
            int iter = 0;
            while (scheme.reduce()) ++iter;
            int final_rank = scheme.get_rank();
            double time_sec = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();

            current_rank = final_rank;
            pool.clear();
            pool.emplace_back(scheme.release_data(), current_rank, dims, 0);

            std::cout << "Reduce: " << initial_rank << " -> " << final_rank
                      << " (iters=" << iter << ") ["
                      << std::fixed << std::setprecision(3) << time_sec << "s]";
            if (P.do_verify)
                std::cout << (cpd::verify(pool[0].data, T) ? " ok" : " FAIL");
            std::cout << "\n\n";

            logger.add_preprocess("reduce", 0, initial_rank, final_rank, iter, time_sec);
        }

#ifdef TOPP
        if (do_sge) {
            auto res = cpd::topp::run_sge(pool[0].data, dims, seed, sge_beam, P.threads);

            current_rank = res.final_rank;
            pool.clear();
            pool.emplace_back(std::move(res.data), current_rank, dims, 0);

            std::cout << "SGE (beam=" << sge_beam << "): "
                      << res.initial_rank << " -> " << res.final_rank
                      << " (iters=" << res.iterations << ") ["
                      << std::fixed << std::setprecision(3) << res.time_sec << "s]";
            if (P.do_verify)
                std::cout << (cpd::verify(pool[0].data, T) ? " ok" : " FAIL");
            std::cout << "\n\n";

            logger.add_preprocess("sge", sge_beam, res.initial_rank, res.final_rank,
                                  res.iterations, res.time_sec);
        }
#endif

        best_rank = current_rank;
        best_pool_size = pool.size();

        if (do_save) {
            std::vector<std::vector<BitArr>> rows;
            rows.reserve(pool.size());
            for (const auto& s : pool) rows.push_back(s.data);
            best_path = cpd::save_best_pool(rows, best_rank, id);
        }
    }

    // === FGS phase ===
    if (do_fgs && current_rank > 0 && !pool.empty()) {
        cpd::WalkExecutor executor(P, seed);

        if (do_save && best_path.empty()) {
            std::vector<std::vector<BitArr>> rows;
            rows.reserve(pool.size());
            for (const auto& s : pool) rows.push_back(s.data);
            best_path = cpd::save_best_pool(rows, best_rank, id);
        }

        while (current_rank > 0 && !pool.empty()) {
            std::cout << "=== Rank " << current_rank << " ===\n";

            auto out = executor.run(current_rank, pool, T);
            bool improved = !out.next_pool.empty() && out.result_rank < current_rank;

            std::cout << std::fixed << std::setprecision(1)
                      << "Collect: " << out.t_collect_sec << "s, attempts: " << out.attempts_made << "\n";

            if (P.do_verify)
                std::cout << "Verify: " << out.t_verify_sec << "s, ok: " 
                          << out.verified_ok << "/" << out.verified_total << "\n";

            std::cout << std::setprecision(2)
                      << "Flips: " << out.flips_mps_total << " M/s * " << out.avg_inner_sec << "s\n"
                      << std::setprecision(0)
                      << "Avg: flips=" << out.avg_flips_per_attempt
                      << std::setprecision(1)
                      << ", plus=" << out.avg_plus_ok_per_attempt << "|" << out.avg_plus_fail_per_attempt << "\n";

            if (P.reduce_interval > 0) {
                double pct = out.total_reduce_calls > 0 
                    ? 100.0 * out.total_reduce_ok / out.total_reduce_calls : 0.0;
                double rps = out.total_reduce_ns > 0
                    ? 1e9 * out.total_reduce_calls / out.total_reduce_ns : 0.0;
                std::cout << std::setprecision(2)
                          << "Reduce: " << out.total_reduce_ok << "/" << out.total_reduce_calls
                          << " (" << pct << "%), " << std::setprecision(1) << rps << "/s\n";
            }

            std::cout << "Collected: " << out.next_pool.size() << " @ rank " << out.result_rank << "\n\n";

            bool saved = false;
            if (improved && do_save) {
                std::vector<std::vector<BitArr>> rows;
                rows.reserve(out.next_pool.size());
                for (const auto& s : out.next_pool) rows.push_back(s.data);
                best_path = cpd::save_best_pool(rows, out.result_rank, id);
                saved = true;
            }

            if (improved) {
                best_rank = out.result_rank;
                best_pool_size = out.next_pool.size();
            }

            logger.add_epoch(out, saved, best_rank, best_path);

            current_rank = out.result_rank;
            pool = std::move(out.next_pool);

            if (!improved) break;
        }
    }

    // Final output
    std::cout << "=== Result ===\n"
              << "Best rank: " << best_rank << "\n"
              << "Pool size: " << best_pool_size << "\n";
    if (!best_path.empty())
        std::cout << "Saved: " << best_path << "\n";
    if (do_log)
        std::cout << "Log: " << fs::absolute(logger.path()).string() << "\n";

    logger.finish(best_rank, best_pool_size, best_path);

    return 0;
}