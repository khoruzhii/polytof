// compile.cpp
// Quantum circuit compiler: .qc -> optimized .qc + phase polynomial matrix + tensor

#include <iostream>
#include <fstream>
#include <filesystem>
#include "CLI11.hpp"
#include "cnpy.h"
#include "picojson.h"
#include "core/paths.h"
#include "circuit/circuit.h"
#include "circuit/parser_qc.h"
#include "circuit/t_merge.h"
#include "circuit/h_opt.h"
#include "circuit/phase_poly.h"
#include "waring/tensor.h"

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    CLI::App app{"Quantum circuit compiler"};

    int id = -1;
    bool save_idx = false;

    app.add_option("id", id, "Circuit numeric id")->required();
    app.add_flag("--idx", save_idx, "Save tensor with nnz indices");

    CLI11_PARSE(app, argc, argv);

    // Paths
    std::string input_path  = paths::circuit_raw(id);
    std::string output_qc   = paths::circuit_opt(id);
    std::string output_pre  = paths::circuit_pre(id);
    std::string output_post = paths::circuit_post(id);
    std::string output_json = paths::circuit_meta(id);

    fs::create_directories(paths::CIRCUITS_OPT);

    // Parse input
    QcFile qc;
    try {
        qc = QcFile::parse(input_path);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    std::size_t n_original = qc.circuit.n;
    std::size_t gates_original = qc.circuit.gates.size();
    
    // Stats after Toffoli decomposition (to get real T-count)
    auto c_decomposed = qc.circuit.decompose_tof();
    auto stats_original = c_decomposed.stats();

    std::cout << paths::pad_id(id) << "\n";
    std::cout << "  input:  " << n_original << " qubits, "
              << gates_original << " gates, T=" << stats_original.t_count
              << ", H=" << stats_original.internal_h << "\n";

    Circuit c = qc.circuit;

    // T-merge
    c = t_merge::merge(c);
    std::size_t t_merged = c.stats().t_count;
    std::cout << "  merge:  T=" << t_merged << "\n";

    // Internal H optimization
    c = h_opt::internal_h_opt(c);

    // Hadamard gadgetization (removes all internal H)
    c = c.hadamard_gadgetization();

    std::size_t n_total = c.n;
    std::size_t n_ancillas = n_total - n_original;
    auto stats = c.stats();

    std::cout << "  output: " << n_total << " qubits (" << n_ancillas
              << " ancillas), P=[" << stats.t_count << "x" << n_total << "]\n";

    // Extract phase polynomial
    auto sc = SlicedCircuit::from_circuit(c);
    auto P = sc.p_matrix();
    std::size_t t_count = sc.t_count();

    // Save optimized .qc
    qc.circuit = c;
    qc.write(output_qc);

    // Save P matrix [t_count × n] to waring dir
    fs::create_directories(paths::WARING_DIR);
    std::string output_npy = paths::waring_matrix(id, static_cast<int>(t_count));
    cnpy::npy_save(output_npy, P.data(), {t_count, sc.n}, "w");

    // Save tensor with --idx
    if (save_idx) {
        // Compute T_ijk = |{t : P[t,i] ∧ P[t,j] ∧ P[t,k]}| mod 2
        waring::SymTensor T;
        T.n = static_cast<int>(n_total);
        std::size_t f1 = 0, f2 = 0, f3 = 0;
        for (int i = 0; i < T.n; ++i) {
            for (int j = i; j < T.n; ++j) {
                for (int k = j; k < T.n; ++k) {
                    if (i < j && j == k) continue;  // skip non-canonical
                    std::size_t cnt = 0;
                    for (std::size_t t = 0; t < t_count; ++t)
                        if (P[t * n_total + i] && P[t * n_total + j] && P[t * n_total + k]) ++cnt;
                    if (cnt & 1) {
                        T.triples.push_back({i, j, k});
                        if (i == j && j == k) ++f1;
                        else if (i == j || j == k) ++f2;
                        else ++f3;
                    }
                }
            }
        }
        fs::create_directories(paths::TENSORS_DIR);
        waring::save_tensor(T, paths::tensor_file(id));
        std::cout << "  tensor: " << T.triples.size() << " nnz (|f1|=" << f1
                  << ", |f2|=" << f2 << ", |f3|=" << f3 << ")\n";
    }

    // Save pre-circuit
    QcFile pre_qc;
    pre_qc.circuit = sc.pre_circuit();
    pre_qc.header = qc.header;
    pre_qc.qubit_names = qc.qubit_names;
    pre_qc.write(output_pre);

    // Save post-circuit
    QcFile post_qc;
    post_qc.circuit = sc.post_circuit();
    post_qc.header = qc.header;
    post_qc.qubit_names = qc.qubit_names;
    post_qc.write(output_post);

    // Save JSON metadata
    picojson::object meta;
    meta["id"] = picojson::value(static_cast<double>(id));
    meta["n_original"] = picojson::value(static_cast<double>(n_original));
    meta["n_total"] = picojson::value(static_cast<double>(n_total));
    meta["t_count"] = picojson::value(static_cast<double>(t_count));

    picojson::object anc_obj;
    for (const auto& [anc, parent] : c.ancillas)
        anc_obj[std::to_string(anc)] = picojson::value(static_cast<double>(parent));
    meta["ancillas"] = picojson::value(anc_obj);

    picojson::object names_obj;
    for (const auto& [idx, nm] : qc.qubit_names)
        names_obj[std::to_string(idx)] = picojson::value(nm);
    meta["qubit_names"] = picojson::value(names_obj);

    std::ofstream json_file(output_json);
    json_file << picojson::value(meta).serialize(true);

    return 0;
}