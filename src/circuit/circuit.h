// circuit.h
// Quantum circuit representation and transformations.

#pragma once

#include <vector>
#include <string>
#include <cstddef>
#include <algorithm>
#include "unordered_dense.h"
#include "small_vector.hpp"
#include "circuit/tableau.h"

struct Circuit {
    std::size_t n = 0;  // number of qubits
    GateList gates;
    ankerl::unordered_dense::map<std::size_t, std::size_t> ancillas;  // ancilla -> parent

    Circuit() = default;
    explicit Circuit(std::size_t n_qubits) : n(n_qubits) {}

    void add(const std::string& name, std::initializer_list<std::size_t> qubits) {
        gates.push_back({name, {qubits.begin(), qubits.end()}});
    }

    void add(const Gate& g) { gates.push_back(g); }

    void append(const GateList& other) {
        gates.insert(gates.end(), other.begin(), other.end());
    }

    void append(const Circuit& other) { append(other.gates); }

    // Circuit statistics
    struct Stats {
        std::size_t h_count = 0;
        std::size_t internal_h = 0;  // H gates between first and last T
        std::size_t t_count = 0;
    };

    Stats stats() const {
        Stats s;
        bool seen_t = false;
        for (const auto& [name, _] : gates) {
            if (name == "h") {
                ++s.h_count;
                if (seen_t) ++s.internal_h;
            }
            if (name == "t") { ++s.t_count; seen_t = true; }
        }
        // Subtract trailing H gates after last T
        if (seen_t) {
            for (auto it = gates.rbegin(); it != gates.rend(); ++it) {
                if (it->first == "h") --s.internal_h;
                if (it->first == "t") break;
            }
        }
        return s;
    }

    // Decompose Toffoli/CCZ into 7 T-gates
    Circuit decompose_tof() const {
        Circuit c(n);
        for (const auto& [name, q] : gates) {
            if ((name == "ccz" || name == "tof") && q.size() == 3) {
                if (name == "tof") c.add("h", {q[2]});
                c.add("t", {q[0]}); c.add("t", {q[1]}); c.add("t", {q[2]});
                c.add("cx", {q[1], q[0]});
                c.add("x", {q[0]}); c.add("t", {q[0]}); c.add("x", {q[0]});
                c.add("cx", {q[2], q[0]});
                c.add("t", {q[0]});
                c.add("cx", {q[1], q[0]});
                c.add("x", {q[0]}); c.add("t", {q[0]}); c.add("x", {q[0]});
                c.add("cx", {q[2], q[0]});
                c.add("cx", {q[2], q[1]});
                c.add("x", {q[1]}); c.add("t", {q[1]}); c.add("x", {q[1]});
                c.add("cx", {q[2], q[1]});
                if (name == "tof") c.add("h", {q[2]});
            } else {
                c.add({name, q});
            }
        }
        return c;
    }

    // Replace internal H gates with ancilla gadgets.
    // Returns circuit with additional ancilla qubits.
    Circuit hadamard_gadgetization() const {
        // Find last T gate index
        std::size_t last_t = 0;
        for (std::size_t i = 0; i < gates.size(); ++i)
            if (gates[i].first == "t") last_t = i;

        std::vector<std::size_t> parent(n);
        for (std::size_t i = 0; i < n; ++i) parent[i] = i;

        Circuit anc_init(n);  // H gates on ancillas
        Circuit main(n);
        bool seen_t = false;

        for (std::size_t i = 0; i < gates.size(); ++i) {
            const auto& [name, q] = gates[i];
            if (name == "t") seen_t = true;

            if (name == "h" && i < last_t && seen_t) {
                std::size_t a = anc_init.n;  // new ancilla index
                anc_init.add("h", {a});
                main.add("s", {a});
                main.add("s", {q[0]});
                main.add("cx", {q[0], a});
                main.add("s", {a});
                main.add("z", {a});
                main.add("cx", {a, q[0]});
                main.add("cx", {q[0], a});
                anc_init.ancillas[a] = parent[q[0]];
                parent[q[0]] = a;
                ++anc_init.n;
            } else {
                main.add({name, q});
            }
        }

        // Output: anc_init | main | anc_init†
        Circuit out(anc_init.n);
        out.ancillas = anc_init.ancillas;
        out.append(anc_init);
        out.append(main);
        out.append(anc_init.gates);  // H is self-inverse
        return out;
    }
};