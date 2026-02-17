// waring/ctc.h
// Greedy T-count optimization via CNOT-T-CNOT insertions.
// Insertion T(i,j): adds x_i + x_j + 2·x_i·x_j to phase polynomial.
// Profitable when x_i, x_j, x_i·x_j all present: removes 2·c1 + c2 at cost c1.

#pragma once

#include <vector>
#include <utility>
#include <chrono>
#include "core/bit_arr.h"
#include "waring/prototype.h"

namespace waring::ctc {

using namespace proto;

// -----------------------------------------------------------------------------
// Build Waring decomposition from PhaseState + applied insertions
// -----------------------------------------------------------------------------

inline std::vector<BitArr> build_waring(const PhaseState& state,
                                        const std::vector<std::pair<int,int>>& insertions) {
    std::vector<BitArr> parities;
    parities.reserve(state.f1.size() + insertions.size() + 3 * state.f2.size() + 7 * state.f3.size());

    // Remaining linear terms: e_i
    for (int i : state.f1) {
        BitArr p{};
        p.set(i);
        parities.push_back(p);
    }

    // Insertions: e_i + e_j
    for (auto [i, j] : insertions) {
        BitArr p{};
        p.set(i);
        p.set(j);
        parities.push_back(p);
    }

    // Remaining quadratic terms: {e_i, e_k, e_i + e_k}
    for (Triple t : state.f2) {
        int i = ti(t), k = tk(t);
        BitArr ei{}, ek{}, eik{};
        ei.set(i);
        ek.set(k);
        eik.set(i);
        eik.set(k);
        parities.push_back(ei);
        parities.push_back(ek);
        parities.push_back(eik);
    }

    // Remaining cubic terms: 7 parities
    for (Triple t : state.f3) {
        int i = ti(t), j = tj(t), k = tk(t);
        BitArr ei{}, ej{}, ek{};
        ei.set(i);
        ej.set(j);
        ek.set(k);
        parities.push_back(ei);
        parities.push_back(ej);
        parities.push_back(ek);
        parities.push_back(ei ^ ej);
        parities.push_back(ei ^ ek);
        parities.push_back(ej ^ ek);
        parities.push_back(ei ^ ej ^ ek);
    }

    return parities;
}

// -----------------------------------------------------------------------------
// Greedy ctc: minimize cost via CNOT-T-CNOT insertions
// -----------------------------------------------------------------------------

struct CtcResult {
    std::vector<BitArr> parities;
    std::vector<std::pair<int,int>> insertions;
    std::int64_t initial_cost;
    std::int64_t final_cost;
    std::size_t initial_f1, initial_f2, initial_f3;
    std::size_t final_f1, final_f2, final_f3;
    std::size_t final_rank;
    int iterations;
    double time_sec;
};

inline CtcResult greedy_ctc(const SymTensor& T, int c1, int c2, int c3) {
    auto t_start = std::chrono::steady_clock::now();

    PhaseState state(T);
    std::vector<std::pair<int,int>> insertions;

    std::int64_t initial_cost = state.cost(c1, c2, c3);
    std::size_t initial_f1 = state.f1.size();
    std::size_t initial_f2 = state.f2.size();
    std::size_t initial_f3 = state.f3.size();
    int iter = 0;

    while (true) {
        ++iter;
        int best_i = -1, best_j = -1;
        std::int64_t best_delta = 0;

        for (int i = 0; i < T.n; ++i) {
            for (int j = 0; j < T.n; ++j) {
                if (i == j) continue;
                std::int64_t d = state.insertion_delta(i, j, c1, c2);
                if (d < best_delta) {
                    best_delta = d;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        if (best_delta >= 0) break;

        state.apply_insertion(best_i, best_j);
        insertions.emplace_back(best_i, best_j);
    }

    auto parities = build_waring(state, insertions);
    std::size_t final_rank = parities.size();

    auto t_end = std::chrono::steady_clock::now();

    return {
        std::move(parities),
        std::move(insertions),
        initial_cost,
        state.cost(c1, c2, c3),
        initial_f1, initial_f2, initial_f3,
        state.f1.size(), state.f2.size(), state.f3.size(),
        final_rank,
        iter,
        std::chrono::duration<double>(t_end - t_start).count()
    };
}

} // namespace waring::ctc