// cpd/topp/sge.h
// SGE (Symmetric Greedy Elimination) beam search for TOPP tensor decomposition.

#pragma once

#include "core/types.h"
#include "core/bit_arr.h"
#include "cpd/topp/scheme.h"
#include "unordered_dense.h"

#include <vector>
#include <array>
#include <optional>
#include <algorithm>
#include <thread>
#include <chrono>

namespace cpd::topp {

struct SgeResult {
    std::vector<BitArr> data;
    int initial_rank;
    int final_rank;
    int iterations;
    double time_sec;
};

inline SgeResult run_sge(const std::vector<BitArr>& input,
                         std::array<int, 3> dims,
                         U64 seed,
                         int beam_width,
                         int threads) {
    auto t0 = std::chrono::steady_clock::now();

    std::vector<Scheme> beam;
    beam.reserve(beam_width);
    beam.emplace_back(input, dims, seed);
    int initial_rank = beam[0].get_rank();
    int iter = 0;

    // Single-beam fast path
    if (beam_width == 1) {
        while (beam[0].reduce_best()) ++iter;
        int final_rank = beam[0].get_rank();
        double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return {beam[0].release_data(), initial_rank, final_rank, iter, t};
    }

    struct Candidate { int beam_idx; BitArr z; int new_rank; };

    while (true) {
        int prev_best = beam[0].get_rank();
        
        // 1. Parallel candidate generation
        std::vector<std::vector<Candidate>> locals(threads);
        {
            std::vector<std::thread> workers;
            for (int t = 0; t < threads; ++t) {
                workers.emplace_back([&, t] {
                    auto& local = locals[t];
                    local.reserve(128);
                    for (std::size_t b = t; b < beam.size(); b += threads) {
                        int rank = beam[b].get_rank();
                        beam[b].for_each_reduce_candidate([&](const BitArr& z, int impr) {
                            local.push_back({static_cast<int>(b), z, rank - impr});
                        });
                    }
                });
            }
            for (auto& w : workers) w.join();
        }

        // Merge
        std::size_t total = 0;
        for (auto& v : locals) total += v.size();
        if (total == 0) break;

        std::vector<Candidate> candidates;
        candidates.reserve(total);
        for (auto& v : locals)
            for (auto& c : v) candidates.push_back(std::move(c));
        ++iter;

        // 2. Partial sort — top beam_width
        std::size_t take = std::min<std::size_t>(beam_width, candidates.size());
        std::partial_sort(candidates.begin(), candidates.begin() + take, candidates.end(),
            [](const Candidate& a, const Candidate& b) { return a.new_rank < b.new_rank; });
        candidates.resize(take);

        // 3. Parallel clone + apply + hash
        struct Applied {
            std::optional<Scheme> scheme;
            std::uint64_t hash = 0;
        };
        std::vector<Applied> applied(candidates.size());
        {
            std::vector<std::thread> workers;
            for (int t = 0; t < threads; ++t) {
                workers.emplace_back([&, t] {
                    for (std::size_t i = t; i < candidates.size(); i += threads) {
                        auto& a = applied[i];
                        a.scheme = beam[candidates[i].beam_idx].clone();
                        a.scheme->reduce_with_z(candidates[i].z);
                        a.hash = a.scheme->hash();
                    }
                });
            }
            for (auto& w : workers) w.join();
        }

        // 4. Build new beam — deduplicate by hash
        std::vector<Scheme> new_beam;
        new_beam.reserve(beam_width);
        ankerl::unordered_dense::set<std::uint64_t> seen;

        for (auto& a : applied) {
            if (seen.contains(a.hash)) continue;
            seen.insert(a.hash);
            new_beam.push_back(std::move(*a.scheme));
        }

        if (new_beam.empty() || new_beam[0].get_rank() >= prev_best) break;
        beam = std::move(new_beam);
    }

    // Find best in final beam
    int best_idx = 0, final_rank = beam[0].get_rank();
    for (std::size_t b = 1; b < beam.size(); ++b) {
        int r = beam[b].get_rank();
        if (r < final_rank) { final_rank = r; best_idx = static_cast<int>(b); }
    }

    double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return {beam[best_idx].release_data(), initial_rank, final_rank, iter, t};
}

} // namespace cpd::topp