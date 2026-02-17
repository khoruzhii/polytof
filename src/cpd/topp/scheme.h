// cpd/topp/scheme.h
// GF(2) symmetric decomposition driven by seven-sets, with inverted index.
// Liveness: node is alive iff S[0].any().

#pragma once

#include <vector>
#include <array>
#include <tuple>
#include <cassert>
#include <utility>
#include <algorithm>

#include "unordered_dense.h"
#include "small_vector.hpp"
#include "core/random.h"
#include "core/bit_arr.h"
#include "core/types.h"
#include "cpd/topp/seven.h"
#include "cpd/quadratic.h"

#ifndef CAPACITY
#define CAPACITY 8
#endif

namespace cpd::topp {

namespace aud = ankerl::unordered_dense;

// Compute quadratic rank without full factorization
inline int quadratic_rank(const std::vector<BitArr>& u, const std::vector<BitArr>& v, int n) {
    if (u.empty()) return 0;
    BitMatrix B = build_alternating(u, v, n);
    auto [B_can, S, r] = alternating_canonical_form(std::move(B));
    return r;
}

// -----------------------------------------------------------------------------
// Scheme: GF(2) symmetric decomposition driven by seven-sets,
// with inverted index (value -> list of positions).
// -----------------------------------------------------------------------------
class Scheme {
public:
    explicit Scheme(const std::vector<BitArr>& initial, std::array<int, 3> dims, U64 seed = 42)
        : m(0), n(dims[0]), live_count(0), rng(seed) {
        assert(dims[0] == dims[1] && dims[1] == dims[2] && "topp requires symmetric dims");

        const int num_vecs = static_cast<int>(initial.size());
        assert(num_vecs % 3 == 0 && "initial size must be divisible by 3");
        m = num_vecs / 3;

        nodes.resize(m);
        positions.reserve(1ull << 14);
        tmp_pairs.reserve(7);

        for (int t = 0; t < m; ++t) {
            const BitArr& u = initial[3 * t + 0];
            const BitArr& v = initial[3 * t + 1];
            const BitArr& w = initial[3 * t + 2];

            if (!cubic_triple(u, v, w)) continue;

            nodes[t].S = make_seven(u, v, w);
            ++live_count;

            for (int code = 0; code < 7; ++code)
                index_add(t, code, nodes[t].S[code]);
        }
    }

    bool flip() {
        int t1, c1, t2, c2;
        if (!sample_two_positions(t1, c1, t2, c2)) return false;
        if (try_reduce_by_seven(t1, t2)) return true;
        return flip_rank5_pair(t1, c1, t2, c2);
    }

    bool plus() {
        if (live_count < 2 || m < 2) return false;

        for (int attempt = 0; attempt < 64; ++attempt) {
            U64 r = rng.next_u64();
            int t1 = static_cast<int>(r % m);
            int t2 = static_cast<int>((r >> 16) % (m - 1));
            if (t2 >= t1) ++t2;

            if (alive(t1) && alive(t2) && try_plus_pair_rank6(t1, t2))
                return true;
        }
        return false;
    }

    // Reduce with lexicographically first z that gives improvement
    bool reduce() {
        if (flippable.empty()) return false;

        std::vector<std::pair<BitArr, std::size_t>> candidates;
        candidates.reserve(flippable.size());

        for (const BitArr& z : flippable) {
            auto it = positions.find(z);
            if (it == positions.end()) continue;
            std::size_t cnt = it->second.size();
            if (cnt >= 2) candidates.emplace_back(z, cnt);
        }

        if (candidates.empty()) return false;

        std::sort(candidates.begin(), candidates.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });

        for (const auto& [z, cnt] : candidates)
            if (reduce_with_z(z)) return true;

        return false;
    }

    // Reduce with z that gives maximum improvement
    bool reduce_best() {
        if (flippable.empty()) return false;

        BitArr best_z;
        int best_improvement = 0;

        for (const BitArr& z : flippable) {
            int impr = compute_improvement(z);
            if (impr > best_improvement) {
                best_improvement = impr;
                best_z = z;
            }
        }

        if (best_improvement <= 0) return false;
        return reduce_with_z(best_z);
    }

    // Compute improvement for reduce with z (without applying)
    int compute_improvement(const BitArr& z) const {
        auto it = positions.find(z);
        if (it == positions.end()) return 0;

        std::vector<std::pair<int, int>> tc_pairs;
        tc_pairs.reserve(it->second.size());

        for (U32 enc : it->second) {
            int t = decode_t(enc), code = decode_code(enc);
            if (!alive(t)) continue;
            bool dup = false;
            for (auto& [pt, pc] : tc_pairs)
                if (pt == t) { dup = true; break; }
            if (!dup) tc_pairs.emplace_back(t, code);
        }

        const int p = static_cast<int>(tc_pairs.size());
        if (p < 2) return 0;

        std::vector<BitArr> v1_vecs(p), v2_vecs(p);
        for (int q = 0; q < p; ++q) {
            auto [t, code] = tc_pairs[q];
            auto [i1, i2] = kPickXY[code];
            v1_vecs[q] = nodes[t].S[i1];
            v2_vecs[q] = nodes[t].S[i2];
        }

        int r = quadratic_rank(v1_vecs, v2_vecs, n);
        return p - r;
    }

    // Apply reduce with specific z
    bool reduce_with_z(const BitArr& z) {
        auto it = positions.find(z);
        if (it == positions.end()) return false;

        std::vector<std::pair<int, int>> tc_pairs;
        tc_pairs.reserve(it->second.size());

        for (U32 enc : it->second) {
            int t = decode_t(enc), code = decode_code(enc);
            if (!alive(t)) continue;
            bool dup = false;
            for (auto& [pt, pc] : tc_pairs)
                if (pt == t) { dup = true; break; }
            if (!dup) tc_pairs.emplace_back(t, code);
        }

        const int p = static_cast<int>(tc_pairs.size());
        if (p < 2) return false;

        std::vector<BitArr> v1_vecs(p), v2_vecs(p);
        for (int q = 0; q < p; ++q) {
            auto [t, code] = tc_pairs[q];
            auto [i1, i2] = kPickXY[code];
            v1_vecs[q] = nodes[t].S[i1];
            v2_vecs[q] = nodes[t].S[i2];
        }

        auto [u_small, v_small] = factorize_quadratic(v1_vecs, v2_vecs, n);
        const int r = static_cast<int>(u_small.size());
        if (r >= p) return false;

        for (auto [t, code] : tc_pairs) kill_node(t);

        for (int k = 0; k < r; ++k) {
            if (!cubic_triple(u_small[k], v_small[k], z)) continue;
            int t_new = alloc_node_index();
            set_node_from_triple(t_new, u_small[k], v_small[k], z);
        }

        return true;
    }

    // Hash for deduplication in beam search
    std::uint64_t hash() const {
        std::vector<std::array<BitArr, 3>> triples;
        triples.reserve(nodes.size());
        for (const auto& node : nodes) {
            if (!node.S[0].any()) continue;
            std::array<BitArr, 3> triple = {node.S[0], node.S[1], node.S[2]};
            std::sort(triple.begin(), triple.end());
            triples.push_back(triple);
        }
        std::sort(triples.begin(), triples.end());
        return aud::detail::wyhash::hash(triples.data(), triples.size() * sizeof(std::array<BitArr, 3>));
    }

    // Iterate over reduce candidates (z values with potential improvement)
    template<typename Func>
    void for_each_reduce_candidate(Func&& func) const {
        for (const BitArr& z : flippable) {
            int impr = compute_improvement(z);
            if (impr > 0) func(z, impr);
        }
    }

    std::vector<BitArr> get_sevens_flat() const {
        std::vector<BitArr> out;
        out.reserve(static_cast<std::size_t>(m) * 7);
        for (int t = 0; t < m; ++t)
            for (int k = 0; k < 7; ++k)
                out.push_back(nodes[t].S[k]);
        return out;
    }

    std::vector<BitArr> get_data() const {
        std::vector<BitArr> out;
        out.reserve(static_cast<std::size_t>(m) * 3);
        for (int t = 0; t < m; ++t) {
            out.push_back(nodes[t].S[0]);
            out.push_back(nodes[t].S[1]);
            out.push_back(nodes[t].S[2]);
        }
        return out;
    }

    int get_rank() const { return live_count; }
    int get_n() const { return n; }
    std::array<int, 3> get_dims() const { return {n, n, n}; }

    std::vector<BitArr> release_sevens_flat() {
        auto out = get_sevens_flat();
        clear_all();
        return out;
    }

    std::vector<BitArr> release_data() {
        auto out = get_data();
        clear_all();
        return out;
    }

    Scheme clone() const {
        Scheme copy;
        copy.m = m;
        copy.n = n;
        copy.live_count = live_count;
        copy.nodes = nodes;
        copy.rng = rng;
        copy.positions = positions;
        copy.flip_idx = flip_idx;
        copy.flippable = flippable;
        copy.tmp_pairs.reserve(7);
        return copy;
    }

    void enumerate_flips(std::vector<std::tuple<int,int,int,int>>& out) const {
        out.clear();
        for (const BitArr& z : flippable) {
            auto it = positions.find(z);
            if (it == positions.end()) continue;
            const PosBitArr& vec = it->second;
            const std::size_t len = vec.size();
            if (len < 2) continue;
            for (std::size_t i = 0; i < len; ++i) {
                for (std::size_t j = i + 1; j < len; ++j) {
                    int t1 = decode_t(vec[i]), c1 = decode_code(vec[i]);
                    int t2 = decode_t(vec[j]), c2 = decode_code(vec[j]);
                    if (t1 != t2 && alive(t1) && alive(t2))
                        out.emplace_back(t1, c1, t2, c2);
                }
            }
        }
    }

    bool apply_flip(int t1, int c1, int t2, int c2) {
        return flip_rank5_pair(t1, c1, t2, c2);
    }

private:
    struct Node { std::array<BitArr, 7> S{}; };
    using PosBitArr = itlib::small_vector<U32, CAPACITY>;

    Scheme() = default;

    int m{0};
    int n{0};
    int live_count{0};
    std::vector<Node> nodes;
    Rng rng;

    aud::map<BitArr, PosBitArr> positions;
    aud::map<BitArr, int> flip_idx;
    std::vector<BitArr> flippable;
    std::vector<std::pair<int,int>> tmp_pairs;

    static constexpr std::array<std::pair<int,int>, 7> kPickXY = {{
        {1,2}, {0,2}, {0,1}, {0,2}, {0,1}, {1,0}, {0,1}
    }};

    static U32 encode(int t, int code) { return (static_cast<U32>(t) << 3) | (code & 7); }
    static int decode_t(U32 enc) { return static_cast<int>(enc >> 3); }
    static int decode_code(U32 enc) { return enc & 7; }

    bool alive(int t) const { return nodes[t].S[0].any(); }

    int alloc_node_index() {
        nodes.push_back(Node{});
        return m++;
    }

    void index_add(int t, int code, const BitArr& value) {
        if (value.none()) return;
        U32 enc = encode(t, code);
        auto [it, inserted] = positions.try_emplace(value);
        PosBitArr& vec = it->second;
        vec.push_back(enc);

        if (vec.size() == 2 && flip_idx.find(value) == flip_idx.end()) {
            flip_idx.emplace(value, static_cast<int>(flippable.size()));
            flippable.push_back(value);
        }
    }

    void index_del(int t, int code, const BitArr& value) {
        if (value.none()) return;
        auto it = positions.find(value);
        if (it == positions.end()) return;

        PosBitArr& vec = it->second;
        U32 enc = encode(t, code);
        auto pos = std::find(vec.begin(), vec.end(), enc);
        if (pos == vec.end()) return;

        if (vec.size() == 1) {
            positions.erase(it);
            return;
        }
        
        if (vec.size() == 2) {
            auto fit = flip_idx.find(value);
            if (fit != flip_idx.end()) {
                int idx = fit->second;
                int last_idx = static_cast<int>(flippable.size()) - 1;
                if (idx != last_idx) {
                    flippable[idx] = flippable[last_idx];
                    flip_idx[flippable[idx]] = idx;
                }
                flippable.pop_back();
                flip_idx.erase(fit);
            }
        }

        *pos = vec.back();
        vec.pop_back();
    }

    void kill_node(int t) {
        if (t < 0 || t >= m || !alive(t)) return;
        Node& N = nodes[t];
        for (int code = 0; code < 7; ++code) {
            if (N.S[code].any()) {
                index_del(t, code, N.S[code]);
                N.S[code] = BitArr{};
            }
        }
        --live_count;
    }

    void set_node_from_triple(int t, const BitArr& u, const BitArr& v, const BitArr& w) {
        if (t < 0 || t >= m) return;
        if (!cubic_triple(u, v, w)) { kill_node(t); return; }

        Node& N = nodes[t];
        bool was_alive = alive(t);

        if (was_alive)
            for (int code = 0; code < 7; ++code)
                if (N.S[code].any()) index_del(t, code, N.S[code]);

        N.S = make_seven(u, v, w);
        if (!was_alive) ++live_count;

        for (int code = 0; code < 7; ++code)
            if (N.S[code].any()) index_add(t, code, N.S[code]);
    }

    bool sample_two_positions(int& t1, int& code1, int& t2, int& code2) {
        if (flippable.empty()) return false;

        for (int attempt = 0; attempt < 64; ++attempt) {
            U64 r = rng.next_u64();
            std::size_t idx = r % flippable.size();
            const BitArr& value = flippable[idx];

            auto it = positions.find(value);
            if (it == positions.end()) continue;

            const PosBitArr& vec = it->second;
            std::size_t len = vec.size();
            if (len < 2) continue;

            std::size_t i1 = (r >> 16) % len;
            std::size_t i2 = (r >> 32) % (len - 1);
            if (i2 >= i1) ++i2;

            U32 enc1 = vec[i1], enc2 = vec[i2];
            t1 = decode_t(enc1); code1 = decode_code(enc1);
            t2 = decode_t(enc2); code2 = decode_code(enc2);

            if (t1 != t2 && alive(t1) && alive(t2)) return true;
        }
        return false;
    }

    bool reduce_two_nodes_rank4(int t1, int t2) {
        if (t1 == t2 || !alive(t1) || !alive(t2)) return false;

        const auto& S1 = nodes[t1].S;
        const auto& S2 = nodes[t2].S;

        seven_collect_pairs(S1, S2, tmp_pairs);
        if (tmp_pairs.size() != 3) return false;

        const int i1 = tmp_pairs[0].first;
        const int j2 = tmp_pairs[1].second;

        std::vector<int> idx1, idx2;
        idx1.reserve(3); idx2.reserve(3);
        for (auto& p : tmp_pairs) {
            idx1.push_back(p.first);
            idx2.push_back(p.second);
        }

        const int k1 = seven_pick_outside(idx1);
        const int k2 = seven_pick_outside(idx2);
        if (k1 < 0 || k2 < 0) return false;

        BitArr u_new = S1[i1], v_new = S2[j2], w_new = S1[k1] ^ S2[k2];

        if (cubic_triple(u_new, v_new, w_new))
            set_node_from_triple(t1, u_new, v_new, w_new);
        else
            kill_node(t1);

        kill_node(t2);
        return true;
    }

    bool flip_rank5_pair(int t1, int code1, int t2, int code2) {
        if (t1 == t2 || !alive(t1) || !alive(t2)) return false;

        const auto& S1 = nodes[t1].S;
        const auto& S2 = nodes[t2].S;

        if (S1[code1] != S2[code2]) return false;
        BitArr z = S1[code1];

        auto [xi1, yi1] = kPickXY[code1];
        auto [xi2, yi2] = kPickXY[code2];

        BitArr x1 = S1[xi1], y1 = S1[yi1];
        BitArr x2 = S2[xi2], y2 = S2[yi2];

        if (!cubic_triple(x1, y1, z) || !cubic_triple(x2, y2, z)) return false;

        BitArr a1 = x1 ^ x2, b1 = y1, c1 = z;
        BitArr a2 = x2, b2 = y1 ^ y2, c2 = z;

        if (!cubic_triple(a1, b1, c1) || !cubic_triple(a2, b2, c2)) return false;

        set_node_from_triple(t1, a1, b1, c1);
        set_node_from_triple(t2, a2, b2, c2);
        return true;
    }

    bool try_reduce_by_seven(int t1, int t2) {
        if (t1 == t2 || !alive(t1) || !alive(t2)) return false;

        seven_collect_pairs(nodes[t1].S, nodes[t2].S, tmp_pairs);
        const int c = static_cast<int>(tmp_pairs.size());

        if (c == 7) { kill_node(t1); kill_node(t2); return true; }
        if (c == 3) return reduce_two_nodes_rank4(t1, t2);
        return false;
    }

    bool try_plus_pair_rank6(int t1, int t2) {
        if (t1 == t2 || !alive(t1) || !alive(t2)) return false;

        const auto& S1 = nodes[t1].S;
        const auto& S2 = nodes[t2].S;

        seven_collect_pairs(S1, S2, tmp_pairs);
        if (!tmp_pairs.empty()) return false;

        BitArr u1 = S1[0], v1 = S1[1], w1 = S1[2];
        BitArr u2 = S2[0], v2 = S2[1], w2 = S2[2];

        BitArr w2p = w2 ^ w1;
        BitArr t2_u = u2, t2_v = v2, t2_w = w2p;
        BitArr t3_u = u2, t3_v = v2, t3_w = w1;

        if (!cubic_triple(t2_u, t2_v, t2_w)) return false;
        if (!cubic_triple(t3_u, t3_v, t3_w)) return false;

        BitArr t1_new_u = u1 ^ u2, t1_new_v = v1, t1_new_w = w1;
        BitArr t3_new_u = u2, t3_new_v = v1 ^ v2, t3_new_w = w1;

        if (!cubic_triple(t1_new_u, t1_new_v, t1_new_w)) return false;
        if (!cubic_triple(t3_new_u, t3_new_v, t3_new_w)) return false;

        int t3 = alloc_node_index();
        set_node_from_triple(t2, t2_u, t2_v, t2_w);
        set_node_from_triple(t3, t3_u, t3_v, t3_w);

        [[maybe_unused]] bool ok = flip_rank5_pair(t1, 2, t3, 2);
        assert(ok && "flip_rank5_pair failed in plus");
        return true;
    }

    void clear_all() {
        m = 0;
        live_count = 0;
        nodes.clear();
        positions.clear();
        flip_idx.clear();
        flippable.clear();
        tmp_pairs.clear();
    }
};

} // namespace cpd::topp