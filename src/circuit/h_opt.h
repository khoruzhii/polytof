// h_opt.h
// Internal Hadamard gate optimization.
// Minimizes H-count between first and last T gate.

#pragma once

#include <vector>
#include <cstddef>
#include <algorithm>
#include "circuit/circuit.h"
#include "circuit/tableau.h"
#include "circuit/pauli.h"

namespace h_opt {

// Implement Z-rotation from Pauli product using CNOT ladder.
inline GateList impl_z_rotation(const Pauli& p, std::size_t n) {
    GateList gates;
    auto indices = p.z.ones(n);
    if (indices.empty()) return gates;

    std::size_t pivot = indices[0];

    // CNOT ladder to XOR all bits into pivot
    for (std::size_t i = 1; i < indices.size(); ++i)
        gates.push_back({"cx", {indices[i], pivot}});

    gates.push_back({"t", {pivot}});
    if (p.sign) {
        gates.push_back({"s", {pivot}});
        gates.push_back({"z", {pivot}});
    }

    // Undo CNOT ladder
    for (std::size_t i = 1; i < indices.size(); ++i)
        gates.push_back({"cx", {indices[i], pivot}});

    return gates;
}

// Implement Z-rotation from tableau column.
inline GateList impl_z_rotation_col(const Tableau& tab, std::size_t col) {
    GateList gates;

    // Find pivot with Z bit set
    auto it = std::find_if(tab.z.begin(), tab.z.end(),
        [col](const BitVec& bv) { return bv.test(col); });
    if (it == tab.z.end()) return gates;

    std::size_t pivot = static_cast<std::size_t>(it - tab.z.begin());

    // CNOT ladder
    for (std::size_t j = 0; j < tab.n; ++j)
        if (tab.z[j].test(col) && j != pivot)
            gates.push_back({"cx", {j, pivot}});

    gates.push_back({"t", {pivot}});
    if (tab.signs.test(col)) {
        gates.push_back({"s", {pivot}});
        gates.push_back({"z", {pivot}});
    }

    // Undo CNOT ladder
    for (std::size_t j = 0; j < tab.n; ++j)
        if (tab.z[j].test(col) && j != pivot)
            gates.push_back({"cx", {j, pivot}});

    return gates;
}

// Implement general Pauli rotation: diagonalize X-part, then Z-rotation.
inline GateList impl_pauli_rotation(Tableau& tab, std::size_t col) {
    GateList gates;

    // Find pivot with X bit set
    auto it = std::find_if(tab.x.begin(), tab.x.end(),
        [col](const BitVec& bv) { return bv.test(col); });

    if (it != tab.x.end()) {
        std::size_t pivot = static_cast<std::size_t>(it - tab.x.begin());

        // Clear X column with CNOTs
        for (std::size_t j = 0; j < tab.n; ++j) {
            if (tab.x[j].test(col) && j != pivot) {
                tab.append_cx(pivot, j);
                gates.push_back({"cx", {pivot, j}});
            }
        }

        // S if needed
        if (tab.z[pivot].test(col)) {
            tab.append_s(pivot);
            gates.push_back({"s", {pivot}});
        }

        // H to diagonalize
        tab.append_h(pivot);
        gates.push_back({"h", {pivot}});
    }

    // Now implement Z-rotation
    auto zrot = impl_z_rotation_col(tab, col);
    gates.insert(gates.end(), zrot.begin(), zrot.end());

    return gates;
}

// Implement Toffoli/CCZ (7 rotations): 3 single + 4 composite.
inline GateList impl_tof(Tableau& tab, const itlib::small_vector<std::size_t, 3>& cols, bool h_gate) {
    GateList gates;
    std::size_t c2 = cols[2] + tab.n * (h_gate ? 1 : 0);

    // First 3 rotations
    auto g0 = impl_pauli_rotation(tab, cols[0]);
    auto g1 = impl_pauli_rotation(tab, cols[1]);
    auto g2 = impl_pauli_rotation(tab, c2);
    gates.insert(gates.end(), g0.begin(), g0.end());
    gates.insert(gates.end(), g1.begin(), g1.end());
    gates.insert(gates.end(), g2.begin(), g2.end());

    // Extract Paulis for composite rotations
    Pauli p0 = tab.extract(cols[0]);
    Pauli p1 = tab.extract(cols[1]);
    Pauli p2 = tab.extract(c2);

    // 4 composite rotations with XOR combinations
    // p0 ^= p1
    p0.z ^= p1.z;
    p0.sign ^= p1.sign ^ true;
    auto g3 = impl_z_rotation(p0, tab.n);
    gates.insert(gates.end(), g3.begin(), g3.end());

    // p0 ^= p2
    p0.z ^= p2.z;
    p0.sign ^= p2.sign ^ true;
    auto g4 = impl_z_rotation(p0, tab.n);
    gates.insert(gates.end(), g4.begin(), g4.end());

    // p0 ^= p1 (back)
    p0.z ^= p1.z;
    p0.sign ^= p1.sign ^ true;
    auto g5 = impl_z_rotation(p0, tab.n);
    gates.insert(gates.end(), g5.begin(), g5.end());

    // p1 ^= p2
    p1.z ^= p2.z;
    p1.sign ^= p2.sign ^ true;
    auto g6 = impl_z_rotation(p1, tab.n);
    gates.insert(gates.end(), g6.begin(), g6.end());

    return gates;
}

// Build reverse tableau for H optimization.
inline Tableau h_opt_reverse(const Circuit& c) {
    Tableau tab(c.n);

    // Forward pass: Cliffords only
    for (const auto& [name, q] : c.gates) {
        if (name == "h") tab.prepend_h(q[0]);
        else if (name == "x") tab.prepend_x(q[0]);
        else if (name == "z") tab.prepend_z(q[0]);
        else if (name == "s") { tab.prepend_s(q[0]); tab.prepend_z(q[0]); }
        else if (name == "cx") tab.prepend_cx(q[0], q[1]);
        // skip t, tof, ccz
    }

    // Reverse pass: implement rotations
    for (auto it = c.gates.rbegin(); it != c.gates.rend(); ++it) {
        const auto& [name, q] = *it;
        if (name == "h") tab.prepend_h(q[0]);
        else if (name == "x") tab.prepend_x(q[0]);
        else if (name == "z") tab.prepend_z(q[0]);
        else if (name == "s") tab.prepend_s(q[0]);
        else if (name == "cx") tab.prepend_cx(q[0], q[1]);
        else if (name == "t") impl_pauli_rotation(tab, q[0]);
        else if (name == "tof") impl_tof(tab, q, true);
        else if (name == "ccz") impl_tof(tab, q, false);
    }

    return tab;
}

// Internal H optimization: minimize H gates between first and last T.
inline Circuit internal_h_opt(const Circuit& c_in) {
    Tableau tab = h_opt_reverse(c_in);

    Circuit c(c_in.n);
    c.append(tab.to_gates(false));

    // Forward pass: implement rotations
    for (const auto& [name, q] : c_in.gates) {
        if (name == "h") tab.prepend_h(q[0]);
        else if (name == "x") tab.prepend_x(q[0]);
        else if (name == "z") tab.prepend_z(q[0]);
        else if (name == "s") { tab.prepend_s(q[0]); tab.prepend_z(q[0]); }
        else if (name == "cx") tab.prepend_cx(q[0], q[1]);
        else if (name == "t") c.append(impl_pauli_rotation(tab, q[0]));
        else if (name == "tof") c.append(impl_tof(tab, q, true));
        else if (name == "ccz") c.append(impl_tof(tab, q, false));
    }

    c.append(tab.to_gates(true));
    return c;
}

}  // namespace h_opt