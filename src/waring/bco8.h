// waring/bco8.h
// Beam search minimization of mod-8 phase polynomial cost via basis changes.
// Transform: x_i <- x_i ⊕ x_j

#pragma once

#include <vector>
#include <chrono>
#include <algorithm>
#include "core/bit_vec.h"
#include "waring/prototype.h"

namespace waring::bco8 {

using namespace proto;

// -----------------------------------------------------------------------------
// Transform matrix: tracks basis change A where x' = Ax
// -----------------------------------------------------------------------------

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

    TransformMatrix(const TransformMatrix&) = default;
    TransformMatrix& operator=(const TransformMatrix&) = default;
    TransformMatrix(TransformMatrix&&) = default;
    TransformMatrix& operator=(TransformMatrix&&) = default;

    void apply(int i, int j) { cols[i] ^= cols[j]; }
};

// -----------------------------------------------------------------------------
// Verification
// -----------------------------------------------------------------------------

inline bool verify(const SymTensor& orig, const SymTensor& transformed, const TransformMatrix& A) {
    ankerl::unordered_dense::set<Triple> rec;

    for (auto [pi, pj, pk] : transformed.triples) {
        auto Si = A.cols[pi].ones(A.n);
        auto Sj = A.cols[pj].ones(A.n);
        auto Sk = A.cols[pk].ones(A.n);
        int deg = degree(pack(pi, pj, pk));

        auto toggle = [&rec](Triple t) {
            if (t == INVALID) return;
            if (rec.contains(t)) rec.erase(t);
            else rec.insert(t);
        };

        if (deg == 1) {
            for (auto a : Si) toggle(pack(a, a, a));
            for (auto a : Si) for (auto b : Si)
                if (a < b) toggle(pack(a, a, b));
            for (auto a : Si) for (auto b : Si) for (auto c : Si)
                if (a < b && b < c) toggle(pack(a, b, c));
        } else if (deg == 2) {
            for (auto a : Si) for (auto c : Sk)
                if (a != c) toggle(make_quad(a, c));
            for (auto a : Si) for (auto c : Sk) for (auto d : Sk)
                if (c < d) toggle(sorted3(a, c, d));
            for (auto a : Si) for (auto b : Si) for (auto c : Sk)
                if (a < b) toggle(sorted3(a, b, c));
        } else {
            for (auto a : Si) for (auto b : Sj) for (auto c : Sk)
                toggle(sorted3(a, b, c));
        }
    }

    if (rec.size() != orig.triples.size()) return false;
    for (auto [i, j, k] : orig.triples)
        if (!rec.contains(pack(i, j, k))) return false;
    return true;
}

// -----------------------------------------------------------------------------
// Beam state and candidate
// -----------------------------------------------------------------------------

struct BeamState {
    PolyState poly;
    TransformMatrix transform;

    BeamState(const SymTensor& T) : poly(T), transform(T.n) {}
    BeamState(const BeamState&) = default;
    BeamState& operator=(const BeamState&) = default;
    BeamState(BeamState&&) = default;
    BeamState& operator=(BeamState&&) = default;

    std::uint64_t hash() const {
        auto triples = poly.export_tensor().triples;
        return ankerl::unordered_dense::detail::wyhash::hash(triples.data(), triples.size() * 12);
    }
};

struct Candidate {
    int beam_idx, i, j;
    std::int64_t new_cost;
    bool operator<(const Candidate& o) const { return new_cost < o.new_cost; }
};

// -----------------------------------------------------------------------------
// Beam search bco8
// -----------------------------------------------------------------------------

struct Bco8Result {
    SymTensor tensor;
    TransformMatrix transform;
    std::int64_t initial_cost;
    std::int64_t final_cost;
    std::size_t initial_f1, initial_f2, initial_f3;
    std::size_t final_f1, final_f2, final_f3;
    int iterations;
    double time_sec;
};

inline Bco8Result beam_bco8(const SymTensor& T, int c1, int c2, int c3,
                            int beam_width = 1, int patience = 10) {
    auto t_start = std::chrono::steady_clock::now();

    std::vector<BeamState> beam;
    beam.emplace_back(T);

    std::int64_t initial_cost = beam[0].poly.cost(c1, c2, c3);
    std::size_t initial_f1, initial_f2, initial_f3;
    beam[0].poly.count(initial_f1, initial_f2, initial_f3);
    
    std::int64_t best_cost = initial_cost;
    int iter = 0, no_improve = 0;

    while (no_improve < patience) {
        ++iter;

        // Generate candidates from all beam states
        std::vector<Candidate> candidates;
        for (int b = 0; b < int(beam.size()); ++b) {
            const auto& state = beam[b].poly;
            std::int64_t base = state.cost(c1, c2, c3);
            for (int i : state.active_indices()) {
                for (int j = 0; j < state.n; ++j) {
                    if (i == j) continue;
                    std::int64_t d = state.compute_delta(i, j, c1, c2, c3);
                    candidates.push_back({b, i, j, base + d});
                }
            }
        }

        if (candidates.empty()) break;

        // Sort and take top candidates
        std::size_t take = std::min<std::size_t>(beam_width * 2, candidates.size());
        std::partial_sort(candidates.begin(), candidates.begin() + take, candidates.end());

        // Build new beam with unique states
        std::vector<BeamState> new_beam;
        ankerl::unordered_dense::set<std::uint64_t> seen;

        for (const auto& cand : candidates) {
            if (int(new_beam.size()) >= beam_width) break;

            BeamState ns = beam[cand.beam_idx];
            ns.poly.apply(cand.i, cand.j);
            ns.transform.apply(cand.i, cand.j);

            auto h = ns.hash();
            if (seen.contains(h)) continue;
            seen.insert(h);
            new_beam.push_back(std::move(ns));
        }

        if (new_beam.empty()) break;

        // Find best cost in new beam
        std::int64_t iter_best = new_beam[0].poly.cost(c1, c2, c3);
        for (const auto& s : new_beam)
            iter_best = std::min(iter_best, s.poly.cost(c1, c2, c3));

        if (iter_best < best_cost) {
            best_cost = iter_best;
            no_improve = 0;
        } else {
            ++no_improve;
        }

        beam = std::move(new_beam);
    }

    // Find best state in final beam
    int best_idx = 0;
    std::int64_t final_cost = beam[0].poly.cost(c1, c2, c3);
    for (int b = 1; b < int(beam.size()); ++b) {
        auto c = beam[b].poly.cost(c1, c2, c3);
        if (c < final_cost) {
            final_cost = c;
            best_idx = b;
        }
    }

    auto t_end = std::chrono::steady_clock::now();

    std::size_t final_f1, final_f2, final_f3;
    beam[best_idx].poly.count(final_f1, final_f2, final_f3);

    return {
        beam[best_idx].poly.export_tensor(),
        std::move(beam[best_idx].transform),
        initial_cost,
        final_cost,
        initial_f1, initial_f2, initial_f3,
        final_f1, final_f2, final_f3,
        iter,
        std::chrono::duration<double>(t_end - t_start).count()
    };
}

} // namespace waring::bco8