// t_merge.h
// T-gate merging: merges pairs of T gates acting on the same Pauli image.

#pragma once

#include <vector>
#include <cstddef>
#include <algorithm>
#include "unordered_dense.h"
#include "circuit/circuit.h"
#include "circuit/tableau.h"
#include "circuit/pauli.h"

namespace t_merge {
namespace {

// Diagonalize Pauli rotation at column col. Returns true if X-part was non-trivial.
bool diagonalize_pauli_rotation(Tableau& tab, std::size_t col) {
    auto it = std::find_if(tab.x.begin(), tab.x.end(),
        [col](const BitVec& bv) { return bv.test(col); });
    if (it == tab.x.end()) return false;

    std::size_t pivot = static_cast<std::size_t>(it - tab.x.begin());
    for (std::size_t j = 0; j < tab.n; ++j) {
        if (tab.x[j].test(col) && j != pivot)
            tab.append_cx(pivot, j);
    }
    if (tab.z[pivot].test(col)) tab.append_s(pivot);
    tab.append_h(pivot);
    return true;
}

// Diagonalize Toffoli/CCZ (7 T-gates). Returns vector of 7 bools.
std::vector<bool> diagonalize_tof(Tableau& tab, const itlib::small_vector<std::size_t, 3>& cols, bool h_gate) {
    std::vector<bool> v;
    v.push_back(diagonalize_pauli_rotation(tab, cols[0]));
    v.push_back(diagonalize_pauli_rotation(tab, cols[1]));
    v.push_back(diagonalize_pauli_rotation(tab, cols[2] + tab.n * (h_gate ? 1 : 0)));
    for (int i = 0; i < 4; ++i) v.push_back(false);
    return v;
}

// Build tableau by reverse pass through circuit, diagonalizing T/Tof/CCZ.
Tableau reverse_diagonalization(const Circuit& c) {
    Tableau tab(c.n);

    // Forward pass: apply Cliffords
    for (const auto& [name, q] : c.gates) {
        if (name == "h") tab.prepend_h(q[0]);
        else if (name == "x") tab.prepend_x(q[0]);
        else if (name == "z") tab.prepend_z(q[0]);
        else if (name == "s") { tab.prepend_s(q[0]); tab.prepend_z(q[0]); }
        else if (name == "cx") tab.prepend_cx(q[0], q[1]);
        // skip t, tof, ccz
    }

    // Reverse pass: diagonalize rotations
    for (auto it = c.gates.rbegin(); it != c.gates.rend(); ++it) {
        const auto& [name, q] = *it;
        if (name == "h") tab.prepend_h(q[0]);
        else if (name == "x") tab.prepend_x(q[0]);
        else if (name == "z") tab.prepend_z(q[0]);
        else if (name == "s") tab.prepend_s(q[0]);
        else if (name == "cx") tab.prepend_cx(q[0], q[1]);
        else if (name == "t") diagonalize_pauli_rotation(tab, q[0]);
        else if (name == "tof") diagonalize_tof(tab, q, true);
        else if (name == "ccz") diagonalize_tof(tab, q, false);
    }
    return tab;
}

// Compute rank vector: for each T position, whether it increases matrix rank.
std::vector<bool> rank_vector(const Circuit& c) {
    Tableau tab = reverse_diagonalization(c);
    std::vector<bool> v;

    for (const auto& [name, q] : c.gates) {
        if (name == "h") tab.prepend_h(q[0]);
        else if (name == "x") tab.prepend_x(q[0]);
        else if (name == "z") tab.prepend_z(q[0]);
        else if (name == "s") { tab.prepend_s(q[0]); tab.prepend_z(q[0]); }
        else if (name == "cx") tab.prepend_cx(q[0], q[1]);
        else if (name == "t") v.push_back(diagonalize_pauli_rotation(tab, q[0]));
        else if (name == "tof") {
            auto vt = diagonalize_tof(tab, q, true);
            v.insert(v.end(), vt.begin(), vt.end());
        }
        else if (name == "ccz") {
            auto vt = diagonalize_tof(tab, q, false);
            v.insert(v.end(), vt.begin(), vt.end());
        }
    }
    return v;
}

}  // anonymous namespace

// Merge T gates with same Pauli image. Can merge through already-merged gates.
inline Circuit merge(const Circuit& c_in) {
    Circuit c_dec = c_in.decompose_tof();
    std::size_t n = c_dec.n;
    auto v = rank_vector(c_in);
    auto w = v;  // working copy, updated during merges

    std::vector<int> r(v.size(), 1);
    std::vector<Pauli> paulis;
    ankerl::unordered_dense::map<PauliKey, std::vector<std::pair<std::size_t, bool>>> map;

    TableauCol tab(n);
    std::size_t t = 0;

    for (const auto& [name, q] : c_dec.gates) {
        if (name == "h") tab.prepend_h(q[0]);
        else if (name == "x") tab.prepend_x(q[0]);
        else if (name == "z") tab.prepend_z(q[0]);
        else if (name == "s") { tab.prepend_s(q[0]); tab.prepend_z(q[0]); }
        else if (name == "cx") tab.prepend_cx(q[0], q[1]);
        else if (name == "t") {
            Pauli p = tab.stabs[q[0]];
            PauliKey key(p, n);

            bool merge = false;
            auto it = map.find(key);
            if (it != map.end() && !it->second.empty()) {
                auto [idx, sign] = it->second.back();
                it->second.pop_back();

                merge = true;
                for (std::size_t i = idx + 1; i < t && merge; ++i) {
                    if (v[i] && !p.commutes_with(paulis[i])) {
                        if (r[i] == 1) {
                            merge = false;
                        } else {
                            // Check if any later unmerged gate blocks
                            for (std::size_t j = i + 1; j < t && merge; ++j) {
                                if (w[j] && r[j] == 1 && !p.commutes_with(paulis[j]))
                                    merge = false;
                            }
                        }
                        // break;
                    }
                }

                if (merge) {
                    if (v[idx]) {
                        for (std::size_t i = idx + 1; i < t; ++i)
                            w[i] = true;
                    }
                    w[idx] = false;
                    r[idx] = 0;
                    r[t] = (sign == p.sign) ? 2 : 0;
                    if (sign == p.sign) tab.prepend_s(q[0]);
                } else {
                    it->second.push_back({idx, sign});
                }
            }

            if (!merge) map[key].push_back({t, p.sign});
            paulis.push_back(p);
            ++t;
        }
    }

    Circuit c_out(n);
    std::size_t ri = 0;
    for (const auto& [name, q] : c_dec.gates) {
        if (name == "t") {
            if (r[ri] == 1) c_out.add("t", {q[0]});
            else if (r[ri] == 2) c_out.add("s", {q[0]});
            ++ri;
        } else {
            c_out.add({name, q});
        }
    }
    return c_out;
}

}  // namespace t_merge