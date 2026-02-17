// cpd/base/scheme.h
// GF(2) bilinear CPD scheme with inverted index.
// Liveness: term is alive iff data[3*t].any().

#pragma once

#include <vector>
#include <array>
#include <tuple>
#include <cassert>
#include <algorithm>

#include "unordered_dense.h"
#include "small_vector.hpp"
#include "core/random.h"
#include "core/bit_arr.h"
#include "core/bit_vec.h"
#include "core/types.h"
#include "cpd/gauss.h"

#ifndef CAPACITY
#define CAPACITY 8
#endif

namespace cpd::base {

namespace aud = ankerl::unordered_dense;

// -----------------------------------------------------------------------------
// Scheme: GF(2) bilinear CPD with inverted index (value -> list of term indices).
// Liveness: term is alive iff data[3*t].any().
// -----------------------------------------------------------------------------
class Scheme {
public:
    explicit Scheme(const std::vector<BitArr>& initial, std::array<int, 3> dims, U64 seed = 42)
        : m(0), dims(dims), live_count(0), rng(seed) {

        const int num_vecs = static_cast<int>(initial.size());
        assert(num_vecs % 3 == 0 && "initial size must be divisible by 3");
        m = num_vecs / 3;

        data.resize(m * 3);

        for (int t = 0; t < m; ++t) {
            const BitArr& u = initial[3*t + 0];
            const BitArr& v = initial[3*t + 1];
            const BitArr& w = initial[3*t + 2];

            if (!is_live_triple(u, v, w)) continue;

            data[3*t + 0] = u;
            data[3*t + 1] = v;
            data[3*t + 2] = w;
            ++live_count;

            for (int c = 0; c < 3; ++c)
                index_add(t, c, data[3*t + c]);
        }
    }

    bool flip() {
        int t1, t2, comp;
        if (!sample_flippable_pair(t1, t2, comp)) return false;
        return do_flip(t1, t2, comp);
    }

    bool plus() {
        if (live_count < 2 || m < 2) return false;

        for (int attempt = 0; attempt < 64; ++attempt) {
            int t1, t2;
            if (!sample_any_pair(t1, t2)) continue;
            if (try_plus_pair(t1, t2)) return true;
        }
        return false;
    }

    bool reduce() {
        // Try all 6 combinations: (shared_comp, check_comp)
        // shared_comp = which component is shared
        // check_comp = which component we check for linear dependency
        static constexpr std::array<std::pair<int,int>, 6> combos = {{
            {0, 1}, {0, 2}, {1, 0}, {1, 2}, {2, 0}, {2, 1}
        }};

        for (auto [shared, check] : combos)
            if (try_reduce_component(shared, check)) return true;

        return false;
    }

    const std::vector<BitArr>& get_data() const { return data; }
    int get_rank() const { return live_count; }
    std::array<int, 3> get_dims() const { return dims; }

    std::vector<BitArr> release_data() {
        auto out = data;
        clear_all();
        return out;
    }

    Scheme clone() const {
        Scheme copy;
        copy.m = m;
        copy.dims = dims;
        copy.live_count = live_count;
        copy.data = data;
        copy.rng = rng;
        for (int c = 0; c < 3; ++c) {
            copy.index[c] = index[c];
            copy.flip_idx[c] = flip_idx[c];
            copy.flippable[c] = flippable[c];
        }
        return copy;
    }

    // Enumerate all valid flip pairs: (t1, t2, comp)
    void enumerate_flips(std::vector<std::tuple<int,int,int>>& out) const {
        out.clear();
        for (int c = 0; c < 3; ++c) {
            for (const BitArr& val : flippable[c]) {
                auto it = index[c].find(val);
                if (it == index[c].end()) continue;
                const auto& vec = it->second;
                if (vec.size() < 2) continue;
                for (std::size_t i = 0; i < vec.size(); ++i) {
                    for (std::size_t j = i + 1; j < vec.size(); ++j) {
                        int t1 = vec[i], t2 = vec[j];
                        if (alive(t1) && alive(t2))
                            out.emplace_back(t1, t2, c);
                    }
                }
            }
        }
    }

    // Apply deterministic flip
    bool apply_flip(int t1, int t2, int comp) {
        return do_flip(t1, t2, comp);
    }

private:
    using PosList = itlib::small_vector<int, CAPACITY>;
    using IndexMap = aud::map<BitArr, PosList>;

    Scheme() = default;  // for clone()

    std::vector<BitArr> data;  // [u0, v0, w0, u1, v1, w1, ...]
    std::array<IndexMap, 3> index;
    std::array<aud::map<BitArr, int>, 3> flip_idx;
    std::array<std::vector<BitArr>, 3> flippable;

    int m{0};
    std::array<int, 3> dims{};
    int live_count{0};
    Rng rng;

    static constexpr int kNext[3] = {1, 2, 0};
    static constexpr int kPrev[3] = {2, 0, 1};

    // ----- Helpers -----

    static bool is_live_triple(const BitArr& u, const BitArr& v, const BitArr& w) {
        return u.any() && v.any() && w.any();
    }

    bool alive(int t) const { return data[3*t].any(); }

    int alloc_term() {
        data.resize(data.size() + 3);
        return m++;
    }

    // ----- Inverted index -----

    void index_add(int t, int comp, const BitArr& val) {
        if (val.none()) return;
        auto [it, inserted] = index[comp].try_emplace(val);
        PosList& vec = it->second;
        vec.push_back(t);

        if (vec.size() == 2 && flip_idx[comp].find(val) == flip_idx[comp].end()) {
            flip_idx[comp].emplace(val, static_cast<int>(flippable[comp].size()));
            flippable[comp].push_back(val);
        }
    }

    void index_del(int t, int comp, const BitArr& val) {
        if (val.none()) return;
        auto it = index[comp].find(val);
        if (it == index[comp].end()) return;

        PosList& vec = it->second;
        auto pos = std::find(vec.begin(), vec.end(), t);
        if (pos == vec.end()) return;

        if (vec.size() == 1) {
            index[comp].erase(it);
            return;
        }

        if (vec.size() == 2) {
            auto fit = flip_idx[comp].find(val);
            if (fit != flip_idx[comp].end()) {
                int idx = fit->second;
                int last_idx = static_cast<int>(flippable[comp].size()) - 1;
                if (idx != last_idx) {
                    flippable[comp][idx] = flippable[comp][last_idx];
                    flip_idx[comp][flippable[comp][idx]] = idx;
                }
                flippable[comp].pop_back();
                flip_idx[comp].erase(fit);
            }
        }

        *pos = vec.back();
        vec.pop_back();
    }

    // ----- Term operations -----

    void set_component(int t, int comp, const BitArr& new_val) {
        BitArr old_val = data[3*t + comp];
        if (old_val == new_val) return;

        index_del(t, comp, old_val);
        data[3*t + comp] = new_val;
        index_add(t, comp, new_val);
    }

    void kill_term(int t) {
        if (t < 0 || t >= m || !alive(t)) return;
        for (int c = 0; c < 3; ++c) {
            if (data[3*t + c].any()) {
                index_del(t, c, data[3*t + c]);
                data[3*t + c] = BitArr{};
            }
        }
        --live_count;
    }

    void set_term(int t, const BitArr& u, const BitArr& v, const BitArr& w) {
        if (t < 0 || t >= m) return;
        if (!is_live_triple(u, v, w)) { kill_term(t); return; }

        bool was_alive = alive(t);

        if (was_alive)
            for (int c = 0; c < 3; ++c)
                if (data[3*t + c].any()) index_del(t, c, data[3*t + c]);

        data[3*t + 0] = u;
        data[3*t + 1] = v;
        data[3*t + 2] = w;

        if (!was_alive) ++live_count;

        for (int c = 0; c < 3; ++c)
            if (data[3*t + c].any()) index_add(t, c, data[3*t + c]);
    }

    int add_term(const BitArr& u, const BitArr& v, const BitArr& w) {
        if (!is_live_triple(u, v, w)) return -1;

        // Find dead slot or allocate
        int t = -1;
        for (int i = 0; i < m; ++i) {
            if (!alive(i)) { t = i; break; }
        }
        if (t < 0) t = alloc_term();

        set_term(t, u, v, w);
        return t;
    }

    // ----- Sampling -----

    bool sample_flippable_pair(int& t1, int& t2, int& comp) {
        int s0 = static_cast<int>(flippable[0].size());
        int s1 = static_cast<int>(flippable[1].size());
        int s2 = static_cast<int>(flippable[2].size());
        int total = s0 + s1 + s2;

        if (total == 0) return false;

        for (int attempt = 0; attempt < 64; ++attempt) {
            U64 r = rng.next_u64();
            int x = static_cast<int>(r % total);

            if (x < s0) {
                comp = 0;
            } else if (x < s0 + s1) {
                comp = 1;
                x -= s0;
            } else {
                comp = 2;
                x -= s0 + s1;
            }

            const BitArr& val = flippable[comp][x];
            auto it = index[comp].find(val);
            if (it == index[comp].end()) continue;

            const PosList& vec = it->second;
            std::size_t len = vec.size();
            if (len < 2) continue;

            std::size_t i1 = (r >> 16) % len;
            std::size_t i2 = (r >> 32) % (len - 1);
            if (i2 >= i1) ++i2;

            t1 = vec[i1];
            t2 = vec[i2];

            if (t1 != t2 && alive(t1) && alive(t2)) return true;
        }
        return false;
    }

    bool sample_any_pair(int& t1, int& t2) {
        if (live_count < 2) return false;

        U64 r = rng.next_u64();
        t1 = static_cast<int>(r % m);
        t2 = static_cast<int>((r >> 16) % (m - 1));
        if (t2 >= t1) ++t2;

        return alive(t1) && alive(t2);
    }

    // ----- Flip operation -----

    // Count matching components between two terms
    int count_matches(int t1, int t2) const {
        int cnt = 0;
        for (int c = 0; c < 3; ++c)
            if (data[3*t1 + c] == data[3*t2 + c]) ++cnt;
        return cnt;
    }

    // Find which component differs (assumes exactly one differs)
    int find_diff_component(int t1, int t2) const {
        for (int c = 0; c < 3; ++c)
            if (data[3*t1 + c] != data[3*t2 + c]) return c;
        return -1;
    }

    bool do_flip(int t1, int t2, int comp) {
        if (t1 == t2 || !alive(t1) || !alive(t2)) return false;
        if (data[3*t1 + comp] != data[3*t2 + comp]) return false;

        int matches = count_matches(t1, t2);

        // All 3 match -> annihilate both
        if (matches == 3) {
            kill_term(t1);
            kill_term(t2);
            return true;
        }

        // 2 match -> merge into one term
        if (matches == 2) {
            int diff_c = find_diff_component(t1, t2);
            BitArr merged = data[3*t1 + diff_c] ^ data[3*t2 + diff_c];

            if (merged.none()) {
                // Merged to zero -> kill both
                kill_term(t1);
                kill_term(t2);
            } else {
                // Update t1, kill t2
                set_component(t1, diff_c, merged);
                kill_term(t2);
            }
            return true;
        }

        // 1 match -> standard flip
        int cn = kNext[comp];
        int cp = kPrev[comp];

        BitArr n1 = data[3*t1 + cn];
        BitArr n2 = data[3*t2 + cn];
        BitArr p1 = data[3*t1 + cp];
        BitArr p2 = data[3*t2 + cp];

        BitArr new_n1 = n1 ^ n2;
        BitArr new_p2 = p1 ^ p2;

        // Check if either term dies
        bool t1_dies = new_n1.none();
        bool t2_dies = new_p2.none();

        if (t1_dies && t2_dies) {
            kill_term(t1);
            kill_term(t2);
        } else if (t1_dies) {
            kill_term(t1);
            set_component(t2, cp, new_p2);
        } else if (t2_dies) {
            set_component(t1, cn, new_n1);
            kill_term(t2);
        } else {
            set_component(t1, cn, new_n1);
            set_component(t2, cp, new_p2);
        }

        return true;
    }

    // ----- Reduce operation -----

    bool try_reduce_component(int shared, int check) {
        // shared = component that must be equal
        // check = component to test for linear dependence
        // third = component to XOR when reducing

        for (const BitArr& shared_val : flippable[shared]) {
            auto it = index[shared].find(shared_val);
            if (it == index[shared].end()) continue;

            const PosList& terms = it->second;
            if (terms.size() < 2) continue;

            // Collect alive terms
            std::vector<int> alive_terms;
            alive_terms.reserve(terms.size());
            for (int t : terms)
                if (alive(t)) alive_terms.push_back(t);

            if (alive_terms.size() < 2) continue;

            if (try_reduce_group(shared, check, alive_terms))
                return true;
        }
        return false;
    }

    bool try_reduce_group(int shared, int check, const std::vector<int>& terms) {
        const int p = static_cast<int>(terms.size());
        const int third = 3 - shared - check;  // the remaining component

        // Build V matrix (check component) and I identity
        std::vector<BitArr> V(p);
        std::vector<BitVec> I;

        for (int i = 0; i < p; ++i)
            V[i] = data[3*terms[i] + check];

        init_identity(I, p);

        // Find dependency
        int dep = ref_find_dependency(V, I);
        if (dep < 0) return false;

        // I[dep] contains the dependency mask
        const BitVec& mask = I[dep];
        auto bits = mask.ones(p);

        if (bits.size() < 2) return false;

        // Proposition 3: pick one term to delete, XOR its w into all others in dependency
        int kill_idx = static_cast<int>(bits.back());
        int kill_t = terms[kill_idx];
        BitArr w_kill = data[3*kill_t + third];

        // Update all other terms in dependency: w_i ^= w_kill
        for (std::size_t i = 0; i + 1 < bits.size(); ++i) {
            int t = terms[bits[i]];
            BitArr new_w = data[3*t + third] ^ w_kill;
            if (new_w.none()) {
                kill_term(t);
            } else {
                set_component(t, third, new_w);
            }
        }

        // Kill the selected term
        kill_term(kill_t);

        return true;
    }

    // ----- Plus operation -----

    bool try_plus_pair(int t1, int t2) {
        if (t1 == t2 || !alive(t1) || !alive(t2)) return false;

        // Check no shared components (otherwise flip is better)
        if (count_matches(t1, t2) > 0) return false;

        BitArr u1 = data[3*t1 + 0], v1 = data[3*t1 + 1], w1 = data[3*t1 + 2];
        BitArr u2 = data[3*t2 + 0], v2 = data[3*t2 + 1], w2 = data[3*t2 + 2];

        // Try all 3 orientations
        U64 r = rng.next_u64();
        int variant = static_cast<int>(r % 3);

        for (int v = 0; v < 3; ++v) {
            int orient = (variant + v) % 3;

            BitArr t1_u, t1_v, t1_w;
            BitArr t2_u, t2_v, t2_w;
            BitArr t3_u, t3_v, t3_w;

            if (orient == 0) {
                // A-oriented
                t1_u = u1 ^ u2; t1_v = v1;      t1_w = w1;
                t2_u = u2;      t2_v = v2;      t2_w = w2 ^ w1;
                t3_u = u2;      t3_v = v2 ^ v1; t3_w = w1;
            } else if (orient == 1) {
                // B-oriented
                t1_u = u1;      t1_v = v1 ^ v2; t1_w = w1;
                t2_u = u2 ^ u1; t2_v = v2;      t2_w = w2;
                t3_u = u1;      t3_v = v2;      t3_w = w2 ^ w1;
            } else {
                // C-oriented
                t1_u = u1;      t1_v = v1;      t1_w = w1 ^ w2;
                t2_u = u2;      t2_v = v2 ^ v1; t2_w = w2;
                t3_u = u2 ^ u1; t3_v = v1;      t3_w = w2;
            }

            // Check all terms are live
            if (!is_live_triple(t1_u, t1_v, t1_w)) continue;
            if (!is_live_triple(t2_u, t2_v, t2_w)) continue;
            if (!is_live_triple(t3_u, t3_v, t3_w)) continue;

            // Apply transformation
            set_term(t1, t1_u, t1_v, t1_w);
            set_term(t2, t2_u, t2_v, t2_w);
            add_term(t3_u, t3_v, t3_w);

            return true;
        }

        return false;
    }

    void clear_all() {
        m = 0;
        live_count = 0;
        data.clear();
        for (int c = 0; c < 3; ++c) {
            index[c].clear();
            flip_idx[c].clear();
            flippable[c].clear();
        }
    }
};

} // namespace cpd::base