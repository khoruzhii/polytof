// tableau.h
// Gottesman-Knill tableau for Clifford circuit simulation.
// Two representations: row-major (Tableau) and column-major (TableauCol).

#pragma once

#include <vector>
#include <string>
#include <algorithm>
#include <optional>
#include "core/bit_vec.h"
#include "circuit/pauli.h"
#include "small_vector.hpp"

// Gate representation: (name, qubits)
using Gate = std::pair<std::string, itlib::small_vector<std::size_t, 3>>;
using GateList = std::vector<Gate>;

// Row-major tableau: z[qubit], x[qubit] are BitVecs over 2n columns (stabs|destabs).
// Used in h_opt for efficient column operations.
struct Tableau {
    std::size_t n;          // number of qubits
    std::vector<BitVec> z;  // z[i] = Z-part of Paulis for qubit i
    std::vector<BitVec> x;  // x[i] = X-part of Paulis for qubit i
    BitVec signs;           // signs for 2n columns

    explicit Tableau(std::size_t n_qubits) : n(n_qubits), signs(2 * n_qubits) {
        z.reserve(n); x.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            BitVec bz(2 * n), bx(2 * n);
            bz.set(i);           // stab[i] = Z_i
            bx.set(i + n);       // destab[i] = X_i
            z.push_back(std::move(bz));
            x.push_back(std::move(bx));
        }
    }

    // --- Append operations (gate applied to state, right-multiply) ---

    void append_x(std::size_t q) { signs ^= z[q]; }
    void append_z(std::size_t q) { signs ^= x[q]; }

    void append_v(std::size_t q) {  // V = sqrt(X) = HSH
        BitVec a = ~x[q]; a &= z[q];
        signs ^= a;
        x[q] ^= z[q];
    }

    void append_s(std::size_t q) {  // S = sqrt(Z)
        BitVec a = z[q]; a &= x[q];
        signs ^= a;
        z[q] ^= x[q];
    }

    void append_h(std::size_t q) {
        append_s(q); append_v(q); append_s(q);
    }

    void append_cx(std::size_t ctrl, std::size_t targ) {
        BitVec a = ~z[ctrl]; a ^= x[targ]; a &= z[targ]; a &= x[ctrl];
        signs ^= a;
        z[ctrl] ^= z[targ];
        x[targ] ^= x[ctrl];
    }

    void append_cz(std::size_t q0, std::size_t q1) {
        append_s(q0); append_s(q1);
        append_cx(q0, q1);
        append_s(q1); append_z(q1);
        append_cx(q0, q1);
    }

    // --- Extract/insert Pauli from column ---

    Pauli extract(std::size_t col) const {
        Pauli p(n);
        for (std::size_t i = 0; i < n; ++i) {
            p.z.set(i, z[i].test(col));
            p.x.set(i, x[i].test(col));
        }
        p.sign = signs.test(col);
        return p;
    }

    void insert(const Pauli& p, std::size_t col) {
        for (std::size_t i = 0; i < n; ++i) {
            z[i].set(col, p.z.test(i));
            x[i].set(col, p.x.test(i));
        }
        signs.set(col, p.sign);
    }

    // --- Prepend operations (gate applied to operators, left-multiply) ---

    void prepend_x(std::size_t q) { signs.flip(q); }
    void prepend_z(std::size_t q) { signs.flip(q + n); }

    void prepend_s(std::size_t q) {
        Pauli stab = extract(q);
        Pauli destab = extract(q + n);
        destab *= stab;
        insert(destab, q + n);
    }

    void prepend_h(std::size_t q) {
        Pauli stab = extract(q);
        Pauli destab = extract(q + n);
        insert(destab, q);
        insert(stab, q + n);
    }

    void prepend_cx(std::size_t ctrl, std::size_t targ) {
        Pauli stab_c = extract(ctrl);
        Pauli stab_t = extract(targ);
        Pauli destab_c = extract(ctrl + n);
        Pauli destab_t = extract(targ + n);
        stab_t *= stab_c;
        destab_c *= destab_t;
        insert(stab_t, targ);
        insert(destab_c, ctrl + n);
    }

    // Synthesize Clifford circuit from tableau.
    // If inverse=true, returns circuit that maps computational basis to tableau.
    // If inverse=false, returns circuit that maps tableau to computational basis.
    GateList to_gates(bool inverse) const {
        Tableau tab = *this;
        GateList gates;

        for (std::size_t i = 0; i < n; ++i) {
            // Find pivot with X in column i
            auto it = std::find_if(tab.x.begin(), tab.x.end(),
                [i](const BitVec& bv) { return bv.test(i); });
            if (it != tab.x.end()) {
                std::size_t pivot = static_cast<std::size_t>(it - tab.x.begin());
                for (std::size_t j = i + 1; j < n; ++j) {
                    if (tab.x[j].test(i) && j != pivot) {
                        tab.append_cx(pivot, j);
                        gates.push_back({"cx", {pivot, j}});
                    }
                }
                if (tab.z[pivot].test(i)) {
                    tab.append_s(pivot);
                    gates.push_back({"s", {pivot}});
                }
                tab.append_h(pivot);
                gates.push_back({"h", {pivot}});
            }

            // Ensure Z diagonal
            if (!tab.z[i].test(i)) {
                auto it2 = std::find_if(tab.z.begin(), tab.z.end(),
                    [i](const BitVec& bv) { return bv.test(i); });
                std::size_t idx = static_cast<std::size_t>(it2 - tab.z.begin());
                tab.append_cx(i, idx);
                gates.push_back({"cx", {i, idx}});
            }

            // Clear Z column
            for (std::size_t j = 0; j < n; ++j) {
                if (tab.z[j].test(i) && j != i) {
                    tab.append_cx(j, i);
                    gates.push_back({"cx", {j, i}});
                }
            }

            // Clear X in destabilizer column
            for (std::size_t j = 0; j < n; ++j) {
                if (tab.x[j].test(i + n) && j != i) {
                    tab.append_cx(i, j);
                    gates.push_back({"cx", {i, j}});
                }
            }

            // Clear Z in destabilizer column (with CZ via CX-S-CX)
            for (std::size_t j = 0; j < n; ++j) {
                if (tab.z[j].test(i + n) && j != i) {
                    tab.append_cx(i, j);
                    gates.push_back({"cx", {i, j}});
                    tab.append_s(j);
                    gates.push_back({"s", {j}});
                    tab.append_cx(i, j);
                    gates.push_back({"cx", {i, j}});
                }
            }

            if (tab.z[i].test(i + n)) {
                tab.append_s(i);
                gates.push_back({"s", {i}});
            }
            if (tab.signs.test(i)) {
                tab.append_x(i);
                gates.push_back({"x", {i}});
            }
            if (tab.signs.test(i + n)) {
                tab.append_z(i);
                gates.push_back({"z", {i}});
            }
        }

        if (!inverse) {
            GateList result;
            for (auto it = gates.rbegin(); it != gates.rend(); ++it) {
                result.push_back(*it);
                if (it->first == "s") result.push_back({"z", it->second});
            }
            return result;
        }
        return gates;
    }
};

// Column-major tableau: stores n stabs and n destabs as Pauli objects.
// Used in t_merge for efficient Pauli tracking.
struct TableauCol {
    std::size_t n;
    std::vector<Pauli> stabs;
    std::vector<Pauli> destabs;

    explicit TableauCol(std::size_t n_qubits) : n(n_qubits) {
        stabs.reserve(n); destabs.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            Pauli s(n), d(n);
            s.z.set(i);  // stab[i] = Z_i
            d.x.set(i);  // destab[i] = X_i
            stabs.push_back(std::move(s));
            destabs.push_back(std::move(d));
        }
    }

    // --- Prepend operations ---

    void prepend_x(std::size_t q) { stabs[q].sign ^= true; }
    void prepend_z(std::size_t q) { destabs[q].sign ^= true; }

    void prepend_v(std::size_t q) { stabs[q] *= destabs[q]; }
    void prepend_s(std::size_t q) { destabs[q] *= stabs[q]; }

    void prepend_h(std::size_t q) {
        prepend_s(q); prepend_v(q); prepend_s(q);
    }

    void prepend_cx(std::size_t ctrl, std::size_t targ) {
        Pauli sc = stabs[ctrl];
        Pauli dt = destabs[targ];
        stabs[targ] *= sc;
        destabs[ctrl] *= dt;
    }

    // Synthesize Clifford circuit from tableau.
    GateList to_gates(bool inverse) const {
        TableauCol tab = *this;
        GateList gates;

        for (std::size_t i = 0; i < n; ++i) {
            // Find pivot with X in position i
            auto it = std::find_if(tab.stabs.begin(), tab.stabs.end(),
                [i](const Pauli& p) { return p.x.test(i); });
            if (it != tab.stabs.end()) {
                std::size_t pivot = static_cast<std::size_t>(it - tab.stabs.begin());
                for (std::size_t j = i + 1; j < n; ++j) {
                    if (tab.stabs[j].x.test(i) && j != pivot) {
                        tab.prepend_cx(pivot, j);
                        gates.push_back({"cx", {pivot, j}});
                    }
                }
                if (tab.destabs[pivot].x.test(i)) {
                    tab.prepend_s(pivot);
                    gates.push_back({"s", {pivot}});
                }
                tab.prepend_h(pivot);
                gates.push_back({"h", {pivot}});
            }

            // Ensure X diagonal in destabs
            if (!tab.destabs[i].x.test(i)) {
                auto it2 = std::find_if(tab.destabs.begin(), tab.destabs.end(),
                    [i](const Pauli& p) { return p.x.test(i); });
                std::size_t idx = static_cast<std::size_t>(it2 - tab.destabs.begin());
                tab.prepend_cx(i, idx);
                gates.push_back({"cx", {i, idx}});
            }

            // Clear X in destabs column
            for (std::size_t j = 0; j < n; ++j) {
                if (tab.destabs[j].x.test(i) && j != i) {
                    tab.prepend_cx(j, i);
                    gates.push_back({"cx", {j, i}});
                }
            }

            // Clear Z in stabs
            for (std::size_t j = 0; j < n; ++j) {
                if (tab.stabs[j].z.test(i) && j != i) {
                    tab.prepend_cx(i, j);
                    gates.push_back({"cx", {i, j}});
                }
            }

            // Clear Z in destabs (with CZ)
            for (std::size_t j = 0; j < n; ++j) {
                if (tab.destabs[j].z.test(i) && j != i) {
                    tab.prepend_cx(i, j);
                    gates.push_back({"cx", {i, j}});
                    tab.prepend_s(j);
                    gates.push_back({"s", {j}});
                    tab.prepend_cx(i, j);
                    gates.push_back({"cx", {i, j}});
                }
            }

            if (tab.destabs[i].z.test(i)) {
                tab.prepend_s(i);
                gates.push_back({"s", {i}});
            }
            if (tab.stabs[i].sign) {
                tab.prepend_x(i);
                gates.push_back({"x", {i}});
            }
            if (tab.destabs[i].sign) {
                tab.prepend_z(i);
                gates.push_back({"z", {i}});
            }
        }

        std::reverse(gates.begin(), gates.end());

        if (!inverse) {
            GateList result;
            for (auto it = gates.rbegin(); it != gates.rend(); ++it) {
                result.push_back(*it);
                if (it->first == "s") result.push_back({"z", it->second});
            }
            return result;
        }
        return gates;
    }
};
