// todd.h
// T-count optimization via TODD algorithm with beam search.
// Based on: "Lower T-count with faster algorithms" by Vivien Vandaele
//
// Main function: toddpp() — beam search over FastTODD actions
// Setting beam_width=1 gives greedy behavior (equivalent to old fast_todd)

#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <functional>
#include <optional>
#include <thread>
#include <vector>

#include "core/bit_vec.h"
#include "core/bit_arr.h"
#include "unordered_dense.h"

namespace todd {

// Progress callback: (phase, iteration, rank, table) -> continue?
using ProgressFn = std::function<bool(const char*, std::size_t, std::size_t, 
                                       const std::vector<BitArr>&)>;

// ============================================================================
// Utilities
// ============================================================================

// Extend parity vector with quadratic terms for kernel computation
inline BitVec extend_parity(const BitArr& p, std::size_t n) {
    std::size_t ext_size = n + n * (n - 1) / 2;
    BitVec ext(ext_size);
    
    for (std::size_t i = 0; i < n; ++i)
        ext.set(i, p.test(i));
    
    std::size_t idx = n;
    for (std::size_t a = n; a-- > 0; ) {
        for (std::size_t b = 0; b < a; ++b)
            ext.set(idx++, p.test(a) && p.test(b));
    }
    return ext;
}

// Remove duplicate and zero parities
inline std::vector<BitArr> remove_duplicates(std::vector<BitArr> table) {
    ankerl::unordered_dense::map<BitArr, std::size_t> seen;
    std::vector<std::size_t> to_remove;
    
    for (std::size_t i = 0; i < table.size(); ++i) {
        if (table[i].none()) {
            to_remove.push_back(i);
            continue;
        }
        auto it = seen.find(table[i]);
        if (it != seen.end()) {
            to_remove.push_back(it->second);
            to_remove.push_back(i);
            seen.erase(it);
        } else {
            seen[table[i]] = i;
        }
    }
    
    std::sort(to_remove.begin(), to_remove.end(), std::greater<std::size_t>());
    to_remove.erase(std::unique(to_remove.begin(), to_remove.end()), to_remove.end());
    
    for (std::size_t idx : to_remove) {
        if (idx < table.size()) {
            table[idx] = table.back();
            table.pop_back();
        }
    }
    return table;
}

// Incremental Gaussian elimination to find kernel vector
inline std::optional<BitVec> kernel(
    std::vector<BitVec>& matrix,
    std::vector<BitVec>& augmented,
    ankerl::unordered_dense::map<std::size_t, std::size_t>& pivots
) {
    std::size_t m = matrix.size();
    
    for (std::size_t i = 0; i < m; ++i) {
        if (pivots.contains(i)) continue;
        
        for (auto& [row, col] : pivots) {
            if (matrix[i].test(col)) {
                matrix[i] ^= matrix[row];
                augmented[i] ^= augmented[row];
            }
        }
        
        std::size_t first = matrix[i].first_one();
        if (first < matrix[i].capacity_bits() && matrix[i].test(first)) {
            for (auto& [row, col] : pivots) {
                if (matrix[row].test(first)) {
                    matrix[row] ^= matrix[i];
                    augmented[row] ^= augmented[i];
                }
            }
            pivots[i] = first;
        } else {
            return augmented[i];
        }
    }
    return std::nullopt;
}

// Hash table state for deduplication
inline std::uint64_t hash_table(const std::vector<BitArr>& table) {
    std::vector<BitArr> sorted = table;
    std::sort(sorted.begin(), sorted.end());
    return ankerl::unordered_dense::detail::wyhash::hash(
        sorted.data(), sorted.size() * sizeof(BitArr));
}

// Apply (z, y) action to table
inline std::vector<BitArr> apply_action(
    std::vector<BitArr> table, const BitArr& z, const BitVec& y
) {
    for (std::size_t i = 0; i < table.size(); ++i)
        if (y.test(i)) table[i] ^= z;
    
    if (y.count() % 2 == 1)
        table.push_back(z);
    
    return remove_duplicates(std::move(table));
}

// ============================================================================
// TOHPE: Third Order Homogeneous Polynomials Elimination (Algorithm 2)
// Complexity: O(n² m³)
// ============================================================================

inline std::vector<BitArr> tohpe(
    std::vector<BitArr> table, 
    std::size_t n,
    ProgressFn on_progress = nullptr
) {
    if (table.empty()) return table;
    
    std::size_t iter = 0;
    
    while (true) {
        std::size_t m = table.size();
        if (m <= 1) break;
        
        std::vector<BitVec> matrix(m);
        std::vector<BitVec> augmented(m);
        
        for (std::size_t i = 0; i < m; ++i) {
            matrix[i] = extend_parity(table[i], n);
            augmented[i].resize_bits(m);
            augmented[i].set(i);
        }
        
        ankerl::unordered_dense::map<std::size_t, std::size_t> pivots;
        auto maybe_y = kernel(matrix, augmented, pivots);
        
        if (!maybe_y.has_value()) break;
        
        BitVec& y = *maybe_y;
        if (y.none()) break;
        
        std::size_t y_cnt = y.count();
        bool y_is_all_ones = (y_cnt == m);
        bool y_parity_even = (y_cnt % 2 == 0);
        
        if (y_is_all_ones && !y_parity_even) break;
        
        ankerl::unordered_dense::map<BitArr, int> score_map;
        
        if (y_parity_even) {
            for (std::size_t i = 0; i < m; ++i)
                if (y.test(i)) score_map[table[i]] = 1;
        } else {
            for (std::size_t i = 0; i < m; ++i)
                if (!y.test(i)) score_map[table[i]] = 1;
        }
        
        for (std::size_t i = 0; i < m; ++i) {
            if (!y.test(i)) continue;
            for (std::size_t j = 0; j < m; ++j) {
                if (y.test(j)) continue;
                score_map[table[i] ^ table[j]] += 2;
            }
        }
        
        int max_score = 0;
        BitArr best_z{};
        for (auto& [z, score] : score_map) {
            if (score > max_score || (score == max_score && z < best_z)) {
                max_score = score;
                best_z = z;
            }
        }
        
        if (max_score <= 0) break;
        
        for (std::size_t i = 0; i < m; ++i)
            if (y.test(i)) table[i] ^= best_z;
        
        if (!y_parity_even) table.push_back(best_z);
        
        table = remove_duplicates(std::move(table));
        ++iter;
        
        if (on_progress && !on_progress("TOHPE", iter, table.size(), table))
            break;
    }
    
    return table;
}

// ============================================================================
// Beam search internals
// ============================================================================

namespace detail {

struct BeamCandidate {
    std::size_t beam_idx;
    BitArr z;
    BitVec y;
    int score;
    
    bool operator<(const BeamCandidate& o) const { return score > o.score; }
};

inline std::pair<std::size_t, std::size_t> linear_to_pair(std::size_t k, std::size_t m) {
    double dm = static_cast<double>(2 * m - 1);
    double disc = dm * dm - 8.0 * static_cast<double>(k);
    std::size_t i = static_cast<std::size_t>((dm - std::sqrt(disc)) / 2.0);
    while (i > 0 && i * (2 * m - i - 1) / 2 > k) --i;
    while ((i + 1) * (2 * m - i - 2) / 2 <= k) ++i;
    std::size_t base = i * (2 * m - i - 1) / 2;
    std::size_t j = k - base + i + 1;
    return {i, j};
}

// Find all (z, y, score) candidates for a single table state
inline std::vector<BeamCandidate> find_candidates(
    const std::vector<BitArr>& table,
    std::size_t n,
    std::size_t beam_idx,
    std::size_t num_threads
) {
    std::size_t m = table.size();
    if (m <= 1) return {};
    
    std::size_t ext_bits = n + n * (n - 1) / 2;
    
    // Build matrix and pivots
    std::vector<BitVec> matrix(m);
    std::vector<BitVec> augmented(m);
    
    for (std::size_t i = 0; i < m; ++i) {
        matrix[i] = extend_parity(table[i], n);
        augmented[i].resize_bits(m);
        augmented[i].set(i);
    }
    
    ankerl::unordered_dense::map<std::size_t, std::size_t> pivots;
    kernel(matrix, augmented, pivots);
    
    ankerl::unordered_dense::map<std::size_t, std::size_t> col_to_row;
    for (auto& [row, col] : pivots)
        col_to_row[col] = row;
    
    ankerl::unordered_dense::map<BitArr, std::size_t> parity_index;
    for (std::size_t i = 0; i < m; ++i)
        parity_index[table[i]] = i;
    
    // Thread-local candidates
    std::vector<std::vector<BeamCandidate>> local(num_threads);
    std::size_t total_pairs = m * (m - 1) / 2;
    std::atomic<std::size_t> pair_idx{0};
    
    auto worker = [&](std::size_t tid) {
        std::vector<BitVec> r_mat;
        std::vector<BitVec> aug_r_mat;
        r_mat.reserve(n + 1);
        aug_r_mat.reserve(n + 1);
        std::vector<bool> z_bits(n);
        
        while (true) {
            std::size_t pk = pair_idx.fetch_add(1, std::memory_order_relaxed);
            if (pk >= total_pairs) break;
            
            auto [i, j] = linear_to_pair(pk, m);
            BitArr z = table[i] ^ table[j];
            
            for (std::size_t b = 0; b < n; ++b)
                z_bits[b] = z.test(b);
            
            r_mat.clear();
            aug_r_mat.clear();
            
            // Build R matrix for this z
            for (std::size_t k = 0; k < n; ++k) {
                BitVec col(ext_bits);
                BitVec aug_col(m);
                
                std::size_t idx = n;
                for (std::size_t a = n; a-- > 0; ) {
                    for (std::size_t b = 0; b < a; ++b) {
                        if ((a == k && z_bits[b]) || (b == k && z_bits[a])) {
                            col.flip(idx);
                            auto pit = col_to_row.find(idx);
                            if (pit != col_to_row.end()) {
                                col ^= matrix[pit->second];
                                aug_col ^= augmented[pit->second];
                            }
                        }
                        ++idx;
                    }
                }
                r_mat.push_back(std::move(col));
                aug_r_mat.push_back(std::move(aug_col));
            }
            
            {
                BitVec col(ext_bits);
                BitVec aug_col(m);
                
                std::size_t idx = n;
                for (std::size_t a = n; a-- > 0; ) {
                    for (std::size_t b = 0; b < a; ++b) {
                        if (z_bits[a] && z_bits[b]) {
                            col.flip(idx);
                            auto pit = col_to_row.find(idx);
                            if (pit != col_to_row.end()) {
                                col ^= matrix[pit->second];
                                aug_col ^= augmented[pit->second];
                            }
                        }
                        ++idx;
                    }
                    if (z_bits[a]) {
                        col.flip(a);
                        auto pit = col_to_row.find(a);
                        if (pit != col_to_row.end()) {
                            col ^= matrix[pit->second];
                            aug_col ^= augmented[pit->second];
                        }
                    }
                }
                r_mat.push_back(std::move(col));
                aug_r_mat.push_back(std::move(aug_col));
            }
            
            // Gaussian elimination on R matrix, collect all valid y vectors
            for (std::size_t k = 0; k < r_mat.size(); ++k) {
                std::size_t first = r_mat[k].first_one();
                if (first < r_mat[k].capacity_bits() && r_mat[k].test(first)) {
                    for (std::size_t l = k + 1; l < r_mat.size(); ++l) {
                        if (r_mat[l].test(first)) {
                            r_mat[l] ^= r_mat[k];
                            aug_r_mat[l] ^= aug_r_mat[k];
                        }
                    }
                } else {
                    if (aug_r_mat[k].test(i) != aug_r_mat[k].test(j)) {
                        BitVec& y = aug_r_mat[k];
                        
                        int score = 0;
                        for (std::size_t l = 0; l < m; ++l) {
                            if (y.test(l)) {
                                auto it = parity_index.find(table[l] ^ z);
                                if (it != parity_index.end() && !y.test(it->second))
                                    score += 2;
                            }
                        }
                        
                        if (y.count() % 2 == 1)
                            score += parity_index.contains(z) ? 1 : -1;
                        
                        if (score > 0)
                            local[tid].push_back({beam_idx, z, y, score});
                    }
                }
            }
        }
    };
    
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (std::size_t t = 0; t < num_threads; ++t)
        threads.emplace_back(worker, t);
    for (auto& th : threads)
        th.join();
    
    // Merge results
    std::vector<BeamCandidate> result;
    for (auto& v : local)
        result.insert(result.end(), 
            std::make_move_iterator(v.begin()), 
            std::make_move_iterator(v.end()));
    
    return result;
}

} // namespace detail

// ============================================================================
// toddpp: Beam search TODD (main entry point)
// ============================================================================

inline std::vector<BitArr> toddpp(
    std::vector<BitArr> table,
    std::size_t n,
    std::size_t beam_width,
    std::size_t threads,
    bool use_tohpe = true,
    ProgressFn on_progress = nullptr
) {
    if (table.empty()) return table;
    
    threads = std::max<std::size_t>(1, threads);
    beam_width = std::max<std::size_t>(1, beam_width);
    
    // Initialize beam with single state
    std::vector<std::vector<BitArr>> beam;
    beam.push_back(remove_duplicates(std::move(table)));
    
    std::size_t iter = 0;
    
    while (true) {
        ++iter;
        
        // Optionally run TOHPE on each state
        if (use_tohpe) {
            for (auto& state : beam)
                state = tohpe(std::move(state), n);
        }
        
        // Collect candidates from all beam states
        std::vector<detail::BeamCandidate> all_candidates;
        
        for (std::size_t b = 0; b < beam.size(); ++b) {
            auto candidates = detail::find_candidates(beam[b], n, b, threads);
            all_candidates.insert(all_candidates.end(),
                std::make_move_iterator(candidates.begin()),
                std::make_move_iterator(candidates.end()));
        }
        
        // No more improvements possible
        if (all_candidates.empty()) break;
        
        // Select top candidates
        std::size_t take = std::min(beam_width * 2, all_candidates.size());
        std::partial_sort(all_candidates.begin(), 
                          all_candidates.begin() + take, 
                          all_candidates.end());
        
        // Apply actions and deduplicate
        std::vector<std::vector<BitArr>> new_beam;
        new_beam.reserve(beam_width);
        ankerl::unordered_dense::set<std::uint64_t> seen;
        
        for (std::size_t c = 0; c < all_candidates.size() && 
             new_beam.size() < beam_width; ++c) {
            auto& cand = all_candidates[c];
            auto new_table = apply_action(beam[cand.beam_idx], cand.z, cand.y);
            
            auto h = hash_table(new_table);
            if (seen.contains(h)) continue;
            seen.insert(h);
            
            new_beam.push_back(std::move(new_table));
        }
        
        if (new_beam.empty()) break;
        
        beam = std::move(new_beam);
        
        // Progress callback with best state
        if (on_progress) {
            auto best = std::min_element(beam.begin(), beam.end(),
                [](const auto& a, const auto& b) { return a.size() < b.size(); });
            if (!on_progress("FastTODD", iter, best->size(), *best))
                break;
        }
    }
    
    // Return best state (smallest table)
    auto best = std::min_element(beam.begin(), beam.end(),
        [](const auto& a, const auto& b) { return a.size() < b.size(); });
    
    return std::move(*best);
}

} // namespace todd