// core/paths.h
// Centralized filesystem locations.

#pragma once

#include <string>
#include <sstream>
#include <iomanip>

namespace paths {

// Base directories
constexpr const char* WARING_DIR      = "data/waring";
constexpr const char* TENSORS_DIR     = "data/tensors";
constexpr const char* LOGS_DIR        = "data/logs";
constexpr const char* CIRCUITS_RAW    = "data/circuits/raw";
constexpr const char* CIRCUITS_OPT    = "data/circuits/opt";
constexpr const char* INIT_DIR        = "data/init";
constexpr const char* CPD_TOPP_DIR    = "data/cpd/topp";
constexpr const char* CPD_BASE_DIR    = "data/cpd/base";
constexpr const char* CPD_COMM_DIR    = "data/cpd/comm";
constexpr const char* TRANSFORM_DIR   = "data/transform";

// Tensor ID width (4 digits: 0000-9999)
constexpr int ID_WIDTH = 4;
// Rank/t_count width (5 digits)
constexpr int RANK_WIDTH = 5;

// RUN macro support: -DRUN=N adds suffix "-NNN" to output files
#ifdef RUN
  constexpr bool HAS_RUN = true;
  constexpr int  RUN_VAL = RUN;
#else
  constexpr bool HAS_RUN = false;
  constexpr int  RUN_VAL = -1;
#endif

inline void append_run_suffix(std::ostream& os) {
    if constexpr (HAS_RUN)
        os << "-" << std::setw(3) << std::setfill('0') << RUN_VAL;
}

inline std::string run_suffix_str() {
    if constexpr (!HAS_RUN) return "";
    std::ostringstream oss;
    oss << "-" << std::setw(3) << std::setfill('0') << RUN_VAL;
    return oss.str();
}

// Pad tensor id to ID_WIDTH digits
inline std::string pad_id(int id) {
    std::ostringstream oss;
    oss << std::setw(ID_WIDTH) << std::setfill('0') << id;
    return oss.str();
}

inline std::string pad_id(const std::string& id) {
    if (id.size() >= ID_WIDTH) return id;
    return std::string(ID_WIDTH - id.size(), '0') + id;
}

// Pad rank/t_count to RANK_WIDTH digits
inline std::string pad_rank(int rank) {
    std::ostringstream oss;
    oss << std::setw(RANK_WIDTH) << std::setfill('0') << rank;
    return oss.str();
}

inline std::string scheme_filename_regex() {
    std::ostringstream oss;
    oss << R"((\d{)" << ID_WIDTH << R"(})-(\d{5}))";
    if constexpr (HAS_RUN)
        oss << "-" << std::setw(3) << std::setfill('0') << RUN_VAL;
    oss << R"(\.npy)";
    return oss.str();
}

// --- Path builders ---

// data/circuits/raw/{id:04d}.qc
inline std::string circuit_raw(int id) {
    return std::string(CIRCUITS_RAW) + "/" + pad_id(id) + ".qc";
}

// data/circuits/opt/{id:04d}.qc
inline std::string circuit_opt(int id) {
    return std::string(CIRCUITS_OPT) + "/" + pad_id(id) + ".qc";
}

// data/circuits/opt/{id:04d}.pre.qc
inline std::string circuit_pre(int id) {
    return std::string(CIRCUITS_OPT) + "/" + pad_id(id) + ".pre.qc";
}

// data/circuits/opt/{id:04d}.post.qc
inline std::string circuit_post(int id) {
    return std::string(CIRCUITS_OPT) + "/" + pad_id(id) + ".post.qc";
}

// data/circuits/opt/{id:04d}.json
inline std::string circuit_meta(int id) {
    return std::string(CIRCUITS_OPT) + "/" + pad_id(id) + ".json";
}

// data/waring/{id:04d}-{t_count:05d}.npy
inline std::string waring_matrix(int id, int t_count) {
    return std::string(WARING_DIR) + "/" + pad_id(id) + "-" + pad_rank(t_count) + ".npy";
}

// data/tensors/{id:04d}.npy
inline std::string tensor_file(int id) {
    return std::string(TENSORS_DIR) + "/" + pad_id(id) + ".npy";
}

} // namespace paths