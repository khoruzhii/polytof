// prototype.cpp
// Experimental phase polynomial optimization pipeline.
// Combines bco8 (basis change), ctc (CNOT-T-CNOT insertions), and todd (parity optimization).
//
// Pipeline: tensor → [bco8] → [ctc] → [todd] → waring decomposition

#include <iostream>
#include <filesystem>
#include <chrono>

#include "CLI11.hpp"
#include "fmt.h"
#include "cnpy.h"
#include "core/paths.h"
#include "core/bit_arr.h"
#include "cpd/pool.h"
#include "waring/tensor.h"
#include "waring/prototype.h"
#include "waring/bco8.h"
#include "waring/ctc.h"
#include "waring/todd.h"

namespace fs = std::filesystem;

// -----------------------------------------------------------------------------
// Path helpers
// -----------------------------------------------------------------------------

static fs::path tensor_file_path(const std::string& id) {
    return fs::path(paths::TENSORS_DIR) / (paths::pad_id(id) + ".npy");
}

static fs::path waring_file_path(const std::string& id, std::size_t rank) {
    return fs::path(paths::WARING_DIR) / fmt::format("{}-{:05}.npy", paths::pad_id(id), rank);
}

static fs::path transform_file_path(const std::string& id_in, const std::string& id_out) {
    return fs::path(paths::TRANSFORM_DIR) / fmt::format("{}-{}.npy", id_in, id_out);
}

static std::string save_waring(const std::vector<BitArr>& w, int n, const std::string& id) {
    if (w.empty()) return {};
    const fs::path out_path = waring_file_path(id, w.size());
    cpd::ensure_dir_exists(out_path.parent_path());
    waring::save_parities(w, n, out_path.string());
    return fs::absolute(out_path).string();
}

static std::string save_transform(const waring::bco8::TransformMatrix& A,
                                   const std::string& id_in, const std::string& id_out) {
    const fs::path out_path = transform_file_path(id_in, id_out);
    cpd::ensure_dir_exists(out_path.parent_path());

    std::vector<uint8_t> data(A.n * A.n);
    for (int r = 0; r < A.n; ++r)
        for (int c = 0; c < A.n; ++c)
            data[r * A.n + c] = A.cols[c].test(r) ? 1 : 0;

    cnpy::npy_save(out_path.string(), data.data(), {std::size_t(A.n), std::size_t(A.n)});
    return fs::absolute(out_path).string();
}

static std::string save_tensor_file(const waring::SymTensor& T, const std::string& id) {
    const fs::path out_path = fs::path(paths::TENSORS_DIR) / (id + ".npy");
    waring::save_tensor(T, out_path.string());
    return fs::absolute(out_path).string();
}

static std::size_t total_nnz(const std::vector<BitArr>& parities) {
    std::size_t nnz = 0;
    for (const auto& p : parities) nnz += p.count();
    return nnz;
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    CLI::App app{"Prototype: experimental phase polynomial optimization pipeline"};

    int tensor_id = -1;
    int output_id = -1;
    int c1 = 1, c2 = 3, c3 = 7;
    int beam_width = 1;
    int patience = 10;
    bool do_bco = false;
    bool do_ctc = false;
    bool do_todd = false;
    bool do_save = false;
    bool do_verify = false;

    app.add_option("id", tensor_id, "Input tensor ID")->required();
    app.add_option("-o,--output", output_id, "Output tensor ID for bco8 transform");
    app.add_option("--c1", c1, "Cost for linear (T gate)")->default_val(1);
    app.add_option("--c2", c2, "Cost for quadratic (CS gate)")->default_val(3);
    app.add_option("--c3", c3, "Cost for cubic (CCZ gate)")->default_val(7);
    app.add_option("-b,--beam", beam_width, "Beam width for BCO8")->default_val(1);
    app.add_option("--patience", patience, "BCO8 patience (iters without improvement)")->default_val(10);
    app.add_flag("--bco", do_bco, "Run BCO8 basis optimization");
    app.add_flag("--ctc", do_ctc, "Run CTC insertion optimization");
    app.add_flag("--todd", do_todd, "Run TODD parity optimization");
    app.add_flag("--save", do_save, "Save results");
    app.add_flag("--verify", do_verify, "Verify results");

    CLI11_PARSE(app, argc, argv);

    if (!do_bco && !do_ctc && !do_todd) {
        fmt::print(stderr, "Error: need at least one of --bco, --ctc, --todd\n");
        return 1;
    }

    const std::string id_in = paths::pad_id(tensor_id);
    const std::string id_out = paths::pad_id(output_id >= 0 ? output_id : tensor_id + 1000);
    const fs::path tensor_path = tensor_file_path(id_in);

    if (!fs::exists(tensor_path)) {
        fmt::print(stderr, "Tensor file not found: {}\n", tensor_path.string());
        return 1;
    }

    // Load tensor
    waring::SymTensor T = waring::load_tensor(tensor_path.string());
    int n = T.n;

    std::size_t n1, n2, n3;
    {
        waring::proto::PolyState tmp(T);
        tmp.count(n1, n2, n3);
    }
    std::int64_t initial_cost = c1 * n1 + c2 * n2 + c3 * n3;

    fmt::print("=== PROTOTYPE ===\n");
    fmt::print("Tensor {}: n={}, nnz={}\n", id_in, n, T.triples.size());
    fmt::print("Initial: cost={} ({}*{} + {}*{} + {}*{})\n", 
               initial_cost, n1, c1, n2, c2, n3, c3);
    fmt::print("Pipeline:");
    if (do_bco) fmt::print(" bco8");
    if (do_ctc) fmt::print(" ctc");
    if (do_todd) fmt::print(" todd");
    fmt::print("\n\n");

    auto t_total_start = std::chrono::steady_clock::now();

    waring::SymTensor current_tensor = T;
    waring::bco8::TransformMatrix transform(n);
    std::vector<BitArr> parities;
    std::string saved_tensor_path, saved_transform_path, saved_waring_path;

    // === BCO8 phase ===
    if (do_bco) {
        fmt::print("--- BCO8 ---\n");

        auto res = waring::bco8::beam_bco8(current_tensor, c1, c2, c3, beam_width, patience);

        current_tensor = std::move(res.tensor);
        transform = std::move(res.transform);

        fmt::print("  f1:   {} -> {}\n", res.initial_f1, res.final_f1);
        fmt::print("  f2:   {} -> {}\n", res.initial_f2, res.final_f2);
        fmt::print("  f3:   {} -> {}\n", res.initial_f3, res.final_f3);
        fmt::print("  cost: {} -> {}\n", res.initial_cost, res.final_cost);
        fmt::print("  iters={}, time={:.3f}s", res.iterations, res.time_sec);
        if (beam_width > 1) fmt::print(", beam={}", beam_width);
        fmt::print("\n");

        if (do_verify) {
            bool ok = waring::bco8::verify(T, current_tensor, transform);
            fmt::print("  verify: {}\n", ok ? "PASS" : "FAIL");
            if (!ok) return 1;
        }

        if (do_save) {
            saved_tensor_path = save_tensor_file(current_tensor, id_out);
            saved_transform_path = save_transform(transform, id_in, id_out);
            fmt::print("  saved tensor: {}\n", saved_tensor_path);
            fmt::print("  saved transform: {}\n", saved_transform_path);
        }

        fmt::print("\n");
    }

    // === CTC phase ===
    if (do_ctc) {
        fmt::print("--- CTC ---\n");

        auto res = waring::ctc::greedy_ctc(current_tensor, c1, c2, c3);
        parities = std::move(res.parities);

        fmt::print("  f1:   {} -> {}\n", res.initial_f1, res.final_f1);
        fmt::print("  f2:   {} -> {}\n", res.initial_f2, res.final_f2);
        fmt::print("  f3:   {} -> {} (unchanged)\n", res.initial_f3, res.final_f3);
        fmt::print("  cost: {} -> {}\n", res.initial_cost, res.final_cost);
        fmt::print("  insertions={}, rank={}, iters={}, time={:.3f}s\n", 
                   res.insertions.size(), res.final_rank, res.iterations, res.time_sec);
        fmt::print("\n");
    }

    // === TODD phase ===
    if (do_todd) {
        fmt::print("--- TODD ---\n");

        // If no CTC, create trivial waring from current tensor
        if (parities.empty()) {
            parities = waring::tensor_to_waring(current_tensor);
        }

        parities = todd::remove_duplicates(std::move(parities));
        std::size_t initial_rank = parities.size();

        auto t_todd_start = std::chrono::steady_clock::now();
        parities = todd::toddpp(std::move(parities), n, beam_width, 4, true, nullptr);
        auto t_todd_end = std::chrono::steady_clock::now();
        double todd_time = std::chrono::duration<double>(t_todd_end - t_todd_start).count();

        std::size_t final_rank = parities.size();
        std::size_t final_nnz = total_nnz(parities);

        fmt::print("  rank: {} -> {}\n", initial_rank, final_rank);
        fmt::print("  nnz={}, time={:.3f}s", final_nnz, todd_time);
        if (beam_width > 1) fmt::print(", beam={}", beam_width);
        fmt::print("\n\n");
    }

    auto t_total_end = std::chrono::steady_clock::now();
    double total_time = std::chrono::duration<double>(t_total_end - t_total_start).count();

    // === Final verify & save for parities ===
    if (!parities.empty()) {
        if (do_verify) {
            auto reconstructed = waring::SymTensor::from_parities(parities, n);
            bool ok = (reconstructed == current_tensor);
            fmt::print("Waring verify: {}\n", ok ? "PASS" : "FAIL");
            if (!ok) return 1;
        }

        if (do_save) {
            saved_waring_path = save_waring(parities, n, id_in);
            fmt::print("Saved waring: {}\n", saved_waring_path);
        }
    }

    // === Summary ===
    fmt::print("\n=== SUMMARY ===\n");
    fmt::print("Input:  {} (n={}, cost={})\n", id_in, n, initial_cost);
    if (!parities.empty()) {
        fmt::print("Output: rank={}, nnz={}\n", parities.size(), total_nnz(parities));
    }
    fmt::print("Time:   {:.3f}s\n", total_time);

    return 0;
}