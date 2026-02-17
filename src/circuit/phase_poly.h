// phase_poly.h
// Phase polynomial: set of Z-rotations by π/4.
// Each term is a parity (XOR of qubit subset) that gets a T gate.

#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include "core/bit_vec.h"
#include "circuit/tableau.h"
#include "circuit/circuit.h"

struct PhasePoly {
    std::size_t n = 0;            // number of qubits
    std::vector<BitVec> terms;    // each term = parity for one T gate

    PhasePoly() = default;
    explicit PhasePoly(std::size_t n_qubits) : n(n_qubits) {}

    // Synthesize circuit: for each parity, CNOT ladder -> T -> CNOT ladder
    GateList to_gates() const {
        GateList gates;
        for (const auto& z : terms) {
            auto indices = z.ones(n);
            if (indices.empty()) continue;

            std::size_t pivot = indices[0];
            // CNOT ladder: XOR all other bits into pivot
            for (std::size_t i = 1; i < indices.size(); ++i)
                gates.push_back({"cx", {indices[i], pivot}});
            gates.push_back({"t", {pivot}});
            // Undo CNOT ladder
            for (std::size_t i = 1; i < indices.size(); ++i)
                gates.push_back({"cx", {indices[i], pivot}});
        }
        return gates;
    }

    // Compute Clifford correction between original and optimized polynomials.
    // Returns tableau whose circuit compensates the difference in phases.
    // Phase difference for pair (i,j): count how many terms have both bits set.
    // Phase difference for single i: count how many terms have bit i set.
    Tableau clifford_correction(const std::vector<BitVec>& original) const {
        Tableau tab(n);

        // CZ corrections for pairs
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = i + 1; j < n; ++j) {
                std::size_t cnt_orig = 0, cnt_opt = 0;
                for (const auto& t : original)
                    if (t.test(i) && t.test(j)) ++cnt_orig;
                for (const auto& t : terms)
                    if (t.test(i) && t.test(j)) ++cnt_opt;
                // Difference mod 8, then divide by 2 for number of CZ gates
                std::size_t diff = ((cnt_orig - cnt_opt) % 8 + 8) % 8;
                for (std::size_t k = 0; k < diff / 2; ++k)
                    tab.append_cz(i, j);
            }
        }

        // S corrections for singles
        for (std::size_t i = 0; i < n; ++i) {
            std::size_t cnt_orig = 0, cnt_opt = 0;
            for (const auto& t : original)
                if (t.test(i)) ++cnt_orig;
            for (const auto& t : terms)
                if (t.test(i)) ++cnt_opt;
            std::size_t diff = ((cnt_orig - cnt_opt) % 8 + 8) % 8;
            for (std::size_t k = 0; k < diff / 2; ++k)
                tab.append_s(i);
        }

        return tab;
    }
};

// Sliced circuit: init_circuit | [PP_0 | Tab_0] | [PP_1 | Tab_1] | ...
// Slices are separated by H gates.
struct SlicedCircuit {
    std::size_t n = 0;
    Circuit init;                         // gates before first T
    std::vector<PhasePoly> phase_polys;   // phase polynomials between H layers
    std::vector<TableauCol> tableaus;     // Clifford between slices

    SlicedCircuit() = default;
    explicit SlicedCircuit(std::size_t n_qubits) : n(n_qubits), init(n_qubits) {}

    // Build sliced representation from circuit
    static SlicedCircuit from_circuit(const Circuit& c) {
        SlicedCircuit sc(c.n);
        sc.init.ancillas = c.ancillas;

        // Find first T gate
        std::size_t first_t = 0;
        for (std::size_t i = 0; i < c.gates.size(); ++i) {
            if (c.gates[i].first == "t") { first_t = i; break; }
            sc.init.add(c.gates[i]);
        }

        TableauCol tab(c.n);
        PhasePoly pp(c.n);

        for (std::size_t i = first_t; i < c.gates.size(); ++i) {
            const auto& [name, q] = c.gates[i];

            if (name == "h") {
                if (!pp.terms.empty()) {
                    sc.phase_polys.push_back(std::move(pp));
                    pp = PhasePoly(c.n);
                }
                tab.prepend_h(q[0]);
            }
            else if (name == "x") { tab.prepend_x(q[0]); }
            else if (name == "z") { tab.prepend_z(q[0]); }
            else if (name == "s") { tab.prepend_s(q[0]); tab.prepend_z(q[0]); }
            else if (name == "cx") { tab.prepend_cx(q[0], q[1]); }
            else if (name == "t") {
                // Start new slice if needed
                if (pp.terms.empty() && !sc.phase_polys.empty()) {
                    sc.tableaus.push_back(tab);
                    tab = TableauCol(c.n);
                }
                // Add parity term from current stabilizer
                pp.terms.push_back(tab.stabs[q[0]].z);
                // Compensate sign if needed
                if (tab.stabs[q[0]].sign) {
                    tab.prepend_s(q[0]);
                    tab.prepend_z(q[0]);
                }
            }
        }

        if (!pp.terms.empty()) sc.phase_polys.push_back(std::move(pp));
        sc.tableaus.push_back(std::move(tab));

        return sc;
    }

    // Reconstruct circuit after optimization.
    // optimizer(terms, n) -> optimized terms
    template<typename Optimizer>
    Circuit to_circuit(Optimizer&& optimize) {
        Circuit c = init;

        for (std::size_t i = 0; i < phase_polys.size(); ++i) {
            auto original = phase_polys[i].terms;
            phase_polys[i].terms = optimize(original, n);

            // Clifford correction + phase poly + tableau
            auto correction = phase_polys[i].clifford_correction(original);
            c.append(correction.to_gates(false));
            c.append(phase_polys[i].to_gates());
            if (i < tableaus.size())
                c.append(tableaus[i].to_gates(true));
        }

        return c;
    }

    // Get P matrix as row-major uint8 array [t_count × n]
    // Each row is a parity vector. For use with cnpy and waring.
    // Only valid after hadamard_gadgetization (single phase poly).
    std::vector<uint8_t> p_matrix() const {
        if (phase_polys.empty()) return {};
        const auto& terms = phase_polys[0].terms;
        std::size_t t = terms.size();
        std::vector<uint8_t> P(t * n, 0);
        for (std::size_t r = 0; r < t; ++r)
            for (std::size_t c = 0; c < n; ++c)
                P[r * n + c] = terms[r].test(c) ? 1 : 0;
        return P;
    }

    std::size_t t_count() const {
        std::size_t cnt = 0;
        for (const auto& pp : phase_polys) cnt += pp.terms.size();
        return cnt;
    }

    // Pre-Clifford circuit (init gates)
    Circuit pre_circuit() const { return init; }

    // Post-Clifford circuit (final tableau)
    Circuit post_circuit() const {
        Circuit c(n);
        c.ancillas = init.ancillas;
        if (!tableaus.empty())
            c.append(tableaus.back().to_gates(true));
        return c;
    }
};