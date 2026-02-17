// cpd/comm/scheme.h
// GF(2) commutative quadratic decomposition with inverted index.
// Four-set: [u, v, u^v, w] where u,v define antisymmetric form.
// Liveness: node is alive iff S[3].any() (w is nonzero).

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
#include "core/bit_vec.h"
#include "core/types.h"
#include "cpd/comm/four.h"
#include "cpd/gauss.h"
#include "cpd/quadratic.h"

#ifndef CAPACITY
#define CAPACITY 8
#endif

namespace cpd::comm {

namespace aud = ankerl::unordered_dense;

using cpd::ref_find_dependency;
using cpd::init_identity;
using cpd::factorize_quadratic;

class Scheme {
public:
    explicit Scheme(const std::vector<BitArr>& initial, std::array<int, 3> dims, U64 seed = 42)
        : m(0), n(dims[0]), nw(dims[2]), live_count(0), rng(seed) {
        assert(dims[0] == dims[1] && "comm requires dims[0] == dims[1]");

        const int num_vecs = static_cast<int>(initial.size());
        assert(num_vecs % 3 == 0 && "initial size must be divisible by 3");
        m = num_vecs / 3;

        nodes.resize(m);
        idx_w.reserve(1ull << 12);
        idx_uv.reserve(1ull << 14);

        for (int t = 0; t < m; ++t) {
            const BitArr& u = initial[3 * t + 0];
            const BitArr& v = initial[3 * t + 1];
            const BitArr& w = initial[3 * t + 2];

            if (!is_live_quad(u, v, w)) continue;

            nodes[t].S = make_four(u, v, w);
            ++live_count;

            index_w_add(t, w);
            for (int c = 0; c < 3; ++c)
                index_uv_add(t, c, nodes[t].S[c]);
        }
    }

    bool flip() {
        int t1, c1, t2, c2;
        if (!sample_pair(t1, c1, t2, c2)) return false;
        if (try_reduce_pair(t1, t2)) return true;
        return flip_pair(t1, c1, t2, c2);
    }

    bool plus() {
        if (live_count < 2 || m < 2) return false;

        for (int attempt = 0; attempt < 64; ++attempt) {
            U64 r = rng.next_u64();
            int t1 = static_cast<int>(r % m);
            int t2 = static_cast<int>((r >> 16) % (m - 1));
            if (t2 >= t1) ++t2;

            if (alive(t1) && alive(t2) && try_plus_pair(t1, t2))
                return true;
        }
        return false;
    }

    bool reduce() {
        if (try_reduce_w()) return true;
        if (try_reduce_uv()) return true;
        return false;
    }

    std::vector<BitArr> get_data() const {
        std::vector<BitArr> out;
        out.reserve(static_cast<std::size_t>(m) * 3);
        for (int t = 0; t < m; ++t) {
            out.push_back(nodes[t].S[0]);
            out.push_back(nodes[t].S[1]);
            out.push_back(nodes[t].S[3]);
        }
        return out;
    }

    int get_rank() const { return live_count; }
    std::array<int, 3> get_dims() const { return {n, n, nw}; }

    std::vector<BitArr> release_data() {
        auto out = get_data();
        clear_all();
        return out;
    }

    Scheme clone() const {
        Scheme copy;
        copy.m = m;
        copy.n = n;
        copy.nw = nw;
        copy.live_count = live_count;
        copy.nodes = nodes;
        copy.rng = rng;
        copy.idx_w = idx_w;
        copy.idx_uv = idx_uv;
        copy.flip_idx_w = flip_idx_w;
        copy.flip_idx_uv = flip_idx_uv;
        copy.flippable_w = flippable_w;
        copy.flippable_uv = flippable_uv;
        return copy;
    }

    void enumerate_flips_w(std::vector<std::tuple<int, int>>& out) const {
        out.clear();
        for (const BitArr& w : flippable_w) {
            auto it = idx_w.find(w);
            if (it == idx_w.end()) continue;
            const auto& vec = it->second;
            for (std::size_t i = 0; i < vec.size(); ++i)
                for (std::size_t j = i + 1; j < vec.size(); ++j)
                    if (alive(vec[i]) && alive(vec[j]))
                        out.emplace_back(vec[i], vec[j]);
        }
    }

    void enumerate_flips_uv(std::vector<std::tuple<int, int, int, int>>& out) const {
        out.clear();
        for (const BitArr& z : flippable_uv) {
            auto it = idx_uv.find(z);
            if (it == idx_uv.end()) continue;
            const auto& vec = it->second;
            for (std::size_t i = 0; i < vec.size(); ++i) {
                for (std::size_t j = i + 1; j < vec.size(); ++j) {
                    int t1 = decode_t(vec[i]), c1 = decode_c(vec[i]);
                    int t2 = decode_t(vec[j]), c2 = decode_c(vec[j]);
                    if (t1 != t2 && alive(t1) && alive(t2))
                        out.emplace_back(t1, c1, t2, c2);
                }
            }
        }
    }

private:
    struct Node { std::array<BitArr, 4> S{}; };
    using PosListW = itlib::small_vector<int, CAPACITY>;
    using PosListUV = itlib::small_vector<U32, CAPACITY>;

    Scheme() = default;

    int m{0};
    int n{0};
    int nw{0};
    int live_count{0};
    std::vector<Node> nodes;
    Rng rng;

    aud::map<BitArr, PosListW> idx_w;
    aud::map<BitArr, PosListUV> idx_uv;
    aud::map<BitArr, int> flip_idx_w;
    aud::map<BitArr, int> flip_idx_uv;
    std::vector<BitArr> flippable_w;
    std::vector<BitArr> flippable_uv;

    static U32 encode_uv(int t, int c) { return (static_cast<U32>(t) << 2) | (c & 3); }
    static int decode_t(U32 e) { return static_cast<int>(e >> 2); }
    static int decode_c(U32 e) { return e & 3; }

    bool alive(int t) const { return nodes[t].S[3].any(); }

    int alloc_node() {
        nodes.push_back(Node{});
        return m++;
    }

    // -------------------------------------------------------------------------
    // Index operations
    // -------------------------------------------------------------------------

    void index_w_add(int t, const BitArr& w) {
        if (w.none()) return;
        auto [it, ins] = idx_w.try_emplace(w);
        auto& vec = it->second;
        vec.push_back(t);
        if (vec.size() == 2 && flip_idx_w.find(w) == flip_idx_w.end()) {
            flip_idx_w.emplace(w, static_cast<int>(flippable_w.size()));
            flippable_w.push_back(w);
        }
    }

    void index_w_del(int t, const BitArr& w) {
        if (w.none()) return;
        auto it = idx_w.find(w);
        if (it == idx_w.end()) return;
        auto& vec = it->second;
        auto pos = std::find(vec.begin(), vec.end(), t);
        if (pos == vec.end()) return;

        if (vec.size() == 1) { idx_w.erase(it); return; }
        if (vec.size() == 2) {
            auto fit = flip_idx_w.find(w);
            if (fit != flip_idx_w.end()) {
                int idx = fit->second, last = static_cast<int>(flippable_w.size()) - 1;
                if (idx != last) { flippable_w[idx] = flippable_w[last]; flip_idx_w[flippable_w[idx]] = idx; }
                flippable_w.pop_back();
                flip_idx_w.erase(fit);
            }
        }
        *pos = vec.back();
        vec.pop_back();
    }

    void index_uv_add(int t, int c, const BitArr& z) {
        if (z.none()) return;
        U32 enc = encode_uv(t, c);
        auto [it, ins] = idx_uv.try_emplace(z);
        auto& vec = it->second;
        vec.push_back(enc);
        if (vec.size() == 2 && flip_idx_uv.find(z) == flip_idx_uv.end()) {
            flip_idx_uv.emplace(z, static_cast<int>(flippable_uv.size()));
            flippable_uv.push_back(z);
        }
    }

    void index_uv_del(int t, int c, const BitArr& z) {
        if (z.none()) return;
        auto it = idx_uv.find(z);
        if (it == idx_uv.end()) return;
        auto& vec = it->second;
        U32 enc = encode_uv(t, c);
        auto pos = std::find(vec.begin(), vec.end(), enc);
        if (pos == vec.end()) return;

        if (vec.size() == 1) { idx_uv.erase(it); return; }
        if (vec.size() == 2) {
            auto fit = flip_idx_uv.find(z);
            if (fit != flip_idx_uv.end()) {
                int idx = fit->second, last = static_cast<int>(flippable_uv.size()) - 1;
                if (idx != last) { flippable_uv[idx] = flippable_uv[last]; flip_idx_uv[flippable_uv[idx]] = idx; }
                flippable_uv.pop_back();
                flip_idx_uv.erase(fit);
            }
        }
        *pos = vec.back();
        vec.pop_back();
    }

    // -------------------------------------------------------------------------
    // Node lifecycle
    // -------------------------------------------------------------------------

    void kill_node(int t) {
        if (t < 0 || t >= m || !alive(t)) return;
        Node& N = nodes[t];
        index_w_del(t, N.S[3]);
        for (int c = 0; c < 3; ++c) index_uv_del(t, c, N.S[c]);
        N.S = {};
        --live_count;
    }

    void set_node(int t, const BitArr& u, const BitArr& v, const BitArr& w) {
        if (t < 0 || t >= m) return;
        if (!is_live_quad(u, v, w)) { kill_node(t); return; }

        Node& N = nodes[t];
        bool was_alive = alive(t);

        if (was_alive) {
            index_w_del(t, N.S[3]);
            for (int c = 0; c < 3; ++c) index_uv_del(t, c, N.S[c]);
        }

        N.S = make_four(u, v, w);
        if (!was_alive) ++live_count;

        index_w_add(t, N.S[3]);
        for (int c = 0; c < 3; ++c) index_uv_add(t, c, N.S[c]);
    }

    // -------------------------------------------------------------------------
    // Sampling
    // -------------------------------------------------------------------------

    bool sample_pair(int& t1, int& c1, int& t2, int& c2) {
        bool has_w = !flippable_w.empty();
        bool has_uv = !flippable_uv.empty();
        if (!has_w && !has_uv) return false;

        for (int attempt = 0; attempt < 64; ++attempt) {
            U64 r = rng.next_u64();
            bool use_w = has_w && (!has_uv || (r & 1));

            if (use_w) {
                std::size_t idx = (r >> 1) % flippable_w.size();
                auto it = idx_w.find(flippable_w[idx]);
                if (it == idx_w.end() || it->second.size() < 2) continue;

                const auto& vec = it->second;
                std::size_t i1 = (r >> 16) % vec.size();
                std::size_t i2 = (r >> 32) % (vec.size() - 1);
                if (i2 >= i1) ++i2;

                t1 = vec[i1]; t2 = vec[i2];
                c1 = c2 = 3;
            } else {
                std::size_t idx = (r >> 1) % flippable_uv.size();
                auto it = idx_uv.find(flippable_uv[idx]);
                if (it == idx_uv.end() || it->second.size() < 2) continue;

                const auto& vec = it->second;
                std::size_t i1 = (r >> 16) % vec.size();
                std::size_t i2 = (r >> 32) % (vec.size() - 1);
                if (i2 >= i1) ++i2;

                t1 = decode_t(vec[i1]); c1 = decode_c(vec[i1]);
                t2 = decode_t(vec[i2]); c2 = decode_c(vec[i2]);
                if (t1 == t2) continue;
            }

            if (alive(t1) && alive(t2)) return true;
        }
        return false;
    }

    // -------------------------------------------------------------------------
    // Flip
    // -------------------------------------------------------------------------

    bool try_reduce_pair(int t1, int t2) {
        const auto& S1 = nodes[t1].S;
        const auto& S2 = nodes[t2].S;
        bool same_w = (S1[3] == S2[3]);
        int isect = three_intersection_count(S1, S2);

        // Kill: same w, same three-set
        if (same_w && isect == 3) {
            kill_node(t1);
            kill_node(t2);
            return true;
        }

        // Merge three-sets
        if (same_w && isect == 1) {
            auto [i1, i2] = three_find_common(S1, S2);
            BitArr z = S1[i1];
            BitArr a = S1[three_pick_outside(i1)] ^ S2[three_pick_outside(i2)];
            BitArr w = S1[3];
            kill_node(t2);
            if (is_live_quad(z, a, w)) set_node(t1, z, a, w);
            else kill_node(t1);
            return true;
        }

        // Merge w
        if (!same_w && isect == 3) {
            BitArr u = S1[0], v = S1[1];
            BitArr w = S1[3] ^ S2[3];
            kill_node(t2);
            if (is_live_quad(u, v, w)) set_node(t1, u, v, w);
            else kill_node(t1);
            return true;
        }

        return false;
    }

    bool flip_pair(int t1, int c1, int t2, int c2) {
        if (nodes[t1].S[3] == nodes[t2].S[3])
            return flip_common_w(t1, t2);
        else
            return flip_common_uv(t1, c1, t2, c2);
    }

    bool flip_common_w(int t1, int t2) {
        const auto& S1 = nodes[t1].S;
        const auto& S2 = nodes[t2].S;
        BitArr w = S1[3];

        BitArr new1_u = S1[0] ^ S2[0], new1_v = S1[1];
        BitArr new2_u = S2[0], new2_v = S1[1] ^ S2[1];

        set_node(t1, new1_u, new1_v, w);
        set_node(t2, new2_u, new2_v, w);
        return true;
    }

    bool flip_common_uv(int t1, int c1, int t2, int c2) {
        const auto& S1 = nodes[t1].S;
        const auto& S2 = nodes[t2].S;
        BitArr z = S1[c1];

        BitArr a1 = S1[three_partner(c1)], w1 = S1[3];
        BitArr a2 = S2[three_partner(c2)], w2 = S2[3];

        BitArr new1_a = a1 ^ a2;
        BitArr new2_w = w1 ^ w2;

        set_node(t1, z, new1_a, w1);
        set_node(t2, z, a2, new2_w);
        return true;
    }

    // -------------------------------------------------------------------------
    // Reduce
    // -------------------------------------------------------------------------

    bool try_reduce_w() {
        if (flippable_w.empty()) return false;

        std::vector<std::pair<std::size_t, BitArr>> cands;
        cands.reserve(flippable_w.size());
        for (const BitArr& w : flippable_w) {
            auto it = idx_w.find(w);
            if (it != idx_w.end() && it->second.size() >= 2)
                cands.emplace_back(it->second.size(), w);
        }
        std::sort(cands.begin(), cands.end(), std::greater<>());

        for (auto& [_, w] : cands)
            if (try_reduce_with_w(w)) return true;
        return false;
    }

    bool try_reduce_with_w(const BitArr& w) {
        auto it = idx_w.find(w);
        if (it == idx_w.end()) return false;

        std::vector<int> terms;
        for (int t : it->second) if (alive(t)) terms.push_back(t);
        const int p = static_cast<int>(terms.size());
        if (p < 2) return false;

        std::vector<BitArr> us(p), vs(p);
        for (int i = 0; i < p; ++i) {
            us[i] = nodes[terms[i]].S[0];
            vs[i] = nodes[terms[i]].S[1];
        }

        auto [u_new, v_new] = factorize_quadratic(us, vs, n);
        const int r = static_cast<int>(u_new.size());
        if (r >= p) return false;

        for (int t : terms) kill_node(t);
        for (int k = 0; k < r; ++k) {
            if (!is_live_quad(u_new[k], v_new[k], w)) continue;
            set_node(alloc_node(), u_new[k], v_new[k], w);
        }
        return true;
    }

    bool try_reduce_uv() {
        if (flippable_uv.empty()) return false;

        std::vector<std::pair<std::size_t, BitArr>> cands;
        cands.reserve(flippable_uv.size());
        for (const BitArr& z : flippable_uv) {
            auto it = idx_uv.find(z);
            if (it != idx_uv.end() && it->second.size() >= 2)
                cands.emplace_back(it->second.size(), z);
        }
        std::sort(cands.begin(), cands.end(), std::greater<>());

        for (auto& [_, z] : cands)
            if (try_reduce_with_z(z)) return true;
        return false;
    }

    bool try_reduce_with_z(const BitArr& z) {
        auto it = idx_uv.find(z);
        if (it == idx_uv.end()) return false;

        std::vector<std::pair<int, int>> tc;
        for (U32 e : it->second) {
            int t = decode_t(e), c = decode_c(e);
            if (alive(t)) tc.emplace_back(t, c);
        }
        std::sort(tc.begin(), tc.end());
        tc.erase(std::unique(tc.begin(), tc.end(),
            [](auto& a, auto& b) { return a.first == b.first; }), tc.end());

        const int k = static_cast<int>(tc.size());
        if (k < 2) return false;

        // Try by a (extended: also check xA = z)
        {
            std::vector<BitArr> A(k + 1);
            std::vector<BitVec> I;
            init_identity(I, k + 1);
            for (int i = 0; i < k; ++i)
                A[i] = nodes[tc[i].first].S[three_partner(tc[i].second)];
            A[k] = z;

            int dep = ref_find_dependency(A, I);
            if (dep >= 0 && dep < k) {
                auto bits = I[dep].ones(k);
                if (bits.size() >= 2)
                    return apply_uv_reduce(tc, bits, z, true);
            } else if (dep == k) {
                // xA = z solvable: switch one a_j to a_j ^ z
                auto bits = I[k].ones(k);
                if (bits.size() >= 2)
                    return apply_uv_reduce(tc, bits, z, true, 0);
            }
        }

        // Try by w
        {
            std::vector<BitArr> W(k);
            std::vector<BitVec> I;
            init_identity(I, k);
            for (int i = 0; i < k; ++i)
                W[i] = nodes[tc[i].first].S[3];

            int dep = ref_find_dependency(W, I);
            if (dep >= 0) {
                auto bits = I[dep].ones(k);
                if (bits.size() >= 2)
                    return apply_uv_reduce(tc, bits, z, false);
            }
        }

        return false;
    }

    // switch_pos: if >= 0, XOR z into a at position switch_pos in bits (for xA=z case)
    bool apply_uv_reduce(const std::vector<std::pair<int, int>>& tc,
                        const std::vector<U64>& bits,
                        const BitArr& z, bool by_a,
                        int switch_pos = -1) {
        const std::size_t nb = bits.size();
        if (nb < 2) return false;

        struct D { int t; BitArr a, w; };
        std::vector<D> data(nb);
        for (std::size_t i = 0; i < nb; ++i) {
            int idx = static_cast<int>(bits[i]);
            int t = tc[idx].first, c = tc[idx].second;
            BitArr a = nodes[t].S[three_partner(c)];
            if (static_cast<int>(i) == switch_pos) a ^= z;
            data[i] = {t, a, nodes[t].S[3]};
        }

        BitArr kill_a = data[nb - 1].a, kill_w = data[nb - 1].w;
        int kill_t = data[nb - 1].t;

        for (std::size_t i = 0; i + 1 < nb; ++i) {
            int t = data[i].t;
            BitArr a = data[i].a, w = data[i].w;
            if (by_a) {
                BitArr new_w = w ^ kill_w;
                if (is_live_quad(z, a, new_w)) set_node(t, z, a, new_w);
                else kill_node(t);
            } else {
                BitArr new_a = a ^ kill_a;
                if (is_live_quad(z, new_a, w)) set_node(t, z, new_a, w);
                else kill_node(t);
            }
        }
        kill_node(kill_t);
        return true;
    }

    // -------------------------------------------------------------------------
    // Plus
    // -------------------------------------------------------------------------

    bool try_plus_pair(int t1, int t2) {
        const auto& S1 = nodes[t1].S;
        const auto& S2 = nodes[t2].S;

        if (S1[3] == S2[3]) return false;
        if (three_intersection_count(S1, S2) > 0) return false;

        BitArr u1 = S1[0], v1 = S1[1], w1 = S1[3];
        BitArr u2 = S2[0], v2 = S2[1], w2 = S2[3];
        BitArr w2p = w1 ^ w2;

        if (!is_live_quad(u2, v2, w1)) return false;
        if (!is_live_quad(u2, v2, w2p)) return false;

        BitArr new_t1_u = u1 ^ u2, new_t1_v = v1;
        BitArr new_t3_u = u2, new_t3_v = v1 ^ v2;

        if (!is_live_quad(new_t1_u, new_t1_v, w1)) return false;
        if (!is_live_quad(new_t3_u, new_t3_v, w1)) return false;

        set_node(t2, u2, v2, w2p);
        int t3 = alloc_node();
        set_node(t3, new_t3_u, new_t3_v, w1);
        set_node(t1, new_t1_u, new_t1_v, w1);
        return true;
    }

    void clear_all() {
        m = live_count = 0;
        nodes.clear();
        idx_w.clear(); idx_uv.clear();
        flip_idx_w.clear(); flip_idx_uv.clear();
        flippable_w.clear(); flippable_uv.clear();
    }
};

} // namespace cpd::comm