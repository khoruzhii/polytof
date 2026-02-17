// parser_qc.h
// Parser for .qc circuit format (RevKit style).

#pragma once

#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cctype>
#include "unordered_dense.h"
#include "circuit/circuit.h"

struct QcFile {
    Circuit circuit;
    std::string header;  // original header lines for round-trip
    ankerl::unordered_dense::map<std::size_t, std::string> qubit_names;  // index -> original name

    static QcFile parse(const std::string& filename) {
        std::ifstream file(filename);
        if (!file) throw std::runtime_error("Cannot open file: " + filename);

        QcFile qc;
        ankerl::unordered_dense::map<std::string, std::size_t> name_to_idx;
        std::string line;

        while (std::getline(file, line)) {
            line = trim(line);
            if (line.empty() || line[0] == '#') continue;

            auto tokens = split(line);
            if (tokens.empty()) continue;

            // Header directives
            if (tokens[0][0] == '.') {
                qc.header += line + "\n";
                if (tokens[0] == ".v") {
                    for (std::size_t i = 1; i < tokens.size(); ++i) {
                        std::size_t idx = qc.circuit.n++;
                        name_to_idx[tokens[i]] = idx;
                        qc.qubit_names[idx] = tokens[i];
                    }
                }
                continue;
            }

            if (tokens[0] == "BEGIN" || tokens[0] == "END") continue;

            // Parse gate
            std::string gate = tokens[0];
            itlib::small_vector<std::size_t, 3> qubits;
            for (std::size_t i = 1; i < tokens.size(); ++i) {
                auto it = name_to_idx.find(tokens[i]);
                if (it == name_to_idx.end())
                    throw std::runtime_error("Unknown qubit: " + tokens[i]);
                qubits.push_back(it->second);
            }

            // Normalize gate names
            add_normalized_gate(qc.circuit, gate, qubits);
        }
        return qc;
    }

    void write(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file) throw std::runtime_error("Cannot create file: " + filename);

        // Build name map including ancillas
        auto names = qubit_names;
        std::size_t next_val = names.size();
        for (const auto& [anc, _] : circuit.ancillas) {
            while (has_name_value(names, std::to_string(next_val))) ++next_val;
            names[anc] = std::to_string(next_val++);
        }

        // Write header with ancillas added to .v line
        std::istringstream hdr(header);
        std::string line;
        while (std::getline(hdr, line)) {
            file << line;
            auto tokens = split(line);
            if (!tokens.empty() && tokens[0] == ".v") {
                for (const auto& [anc, _] : circuit.ancillas)
                    file << " " << names.at(anc);
            }
            file << "\n";
        }

        file << "BEGIN\n";
        for (const auto& [gate, q] : circuit.gates) {
            if (gate == "h") file << "H " << names.at(q[0]) << "\n";
            else if (gate == "x") file << "X " << names.at(q[0]) << "\n";
            else if (gate == "z") file << "Z " << names.at(q[0]) << "\n";
            else if (gate == "s") file << "S " << names.at(q[0]) << "\n";
            else if (gate == "t") file << "T " << names.at(q[0]) << "\n";
            else if (gate == "cx") file << "cnot " << names.at(q[0]) << " " << names.at(q[1]) << "\n";
            else if (gate == "ccx" || gate == "tof")
                file << "tof " << names.at(q[0]) << " " << names.at(q[1]) << " " << names.at(q[2]) << "\n";
            else if (gate == "ccz")
                file << "Z " << names.at(q[0]) << " " << names.at(q[1]) << " " << names.at(q[2]) << "\n";
            else throw std::runtime_error("Unknown gate for QC export: " + gate);
        }
        file << "END";
    }

private:
    static std::string trim(const std::string& s) {
        auto start = s.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) return "";
        auto end = s.find_last_not_of(" \t\r\n");
        return s.substr(start, end - start + 1);
    }

    static std::vector<std::string> split(const std::string& s) {
        std::vector<std::string> tokens;
        std::istringstream iss(s);
        std::string token;
        while (iss >> token) tokens.push_back(token);
        return tokens;
    }

    static bool has_name_value(const ankerl::unordered_dense::map<std::size_t, std::string>& m,
                               const std::string& val) {
        for (const auto& [_, v] : m) if (v == val) return true;
        return false;
    }

    static void add_normalized_gate(Circuit& c, const std::string& gate,
                                    const itlib::small_vector<std::size_t, 3>& q) {
        std::size_t nq = q.size();

        if ((gate == "tof") && nq == 3) { c.gates.push_back({"tof", q}); return; }
        if ((gate == "Zd" || gate == "Z") && nq == 3) { c.gates.push_back({"ccz", q}); return; }
        if ((gate == "cnot" || (gate == "tof" && nq == 2))) { c.gates.push_back({"cx", q}); return; }
        if ((gate == "H") && nq == 1) { c.gates.push_back({"h", q}); return; }
        if ((gate == "X") && nq == 1) { c.gates.push_back({"x", q}); return; }
        if ((gate == "Z") && nq == 1) { c.gates.push_back({"z", q}); return; }
        if ((gate == "S" || gate == "P") && nq == 1) { c.gates.push_back({"s", q}); return; }
        if ((gate == "S*" || gate == "P*") && nq == 1) {
            c.gates.push_back({"z", q});
            c.gates.push_back({"s", q});
            return;
        }
        if ((gate == "T") && nq == 1) { c.gates.push_back({"t", q}); return; }
        if ((gate == "T*") && nq == 1) {
            c.gates.push_back({"z", q});
            c.gates.push_back({"s", q});
            c.gates.push_back({"t", q});
            return;
        }
        // Already normalized lowercase
        if (gate == "h" || gate == "x" || gate == "z" || gate == "s" || gate == "t" ||
            gate == "cx" || gate == "ccz" || gate == "tof") {
            c.gates.push_back({gate, q});
            return;
        }
        throw std::runtime_error("Unknown gate: " + gate);
    }
};