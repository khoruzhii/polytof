// cpd/logger.h
// JSON run logger for CPD flip graph search.

#pragma once

#include "core/paths.h"
#include "cpd/walk.h"
#include "picojson.h"

#include <string>
#include <array>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <random>
#include <system_error>

namespace cpd {

namespace fs = std::filesystem;

struct RunLogger {
    bool enabled = false;
    fs::path out_path;
    picojson::object root;
    picojson::array progress;

    static std::string iso8601_utc_now() {
        auto now = std::chrono::system_clock::now();
        std::time_t t = std::chrono::system_clock::to_time_t(now);
        std::tm tm{};
#if defined(_WIN32)
        gmtime_s(&tm, &t);
#else
        tm = *std::gmtime(&t);
#endif
        std::ostringstream oss;
        oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
        return oss.str();
    }

    static std::string make_run_id() {
        std::mt19937_64 rng{std::random_device{}()};
        std::ostringstream oss;
        oss << std::hex << ((rng() ^ (rng() << 13)) & 0xffffffffffULL);
        return oss.str();
    }

    static fs::path log_path_for(const std::string& id) {
        std::ostringstream oss;
        oss << "cpd-" << paths::pad_id(id);
        paths::append_run_suffix(oss);
        oss << ".json";
        return fs::path(paths::LOGS_DIR) / oss.str();
    }

    void begin(const std::string& id,
               const std::string& tensor_path,
               const std::array<int, 3>& dims,
               std::size_t nnz,
               const WalkParams& P,
               bool save_pools,
               bool continued,
               bool do_reduce,
               bool do_sge,
               int sge_beam,
               bool do_fgs,
               bool enable_logging) {
        enabled = enable_logging;
        if (!enabled) return;

        out_path = log_path_for(id);
        ensure_dir_(out_path.parent_path());

        // Try to load existing log if continuing
        if (continued && fs::exists(out_path)) {
            std::ifstream ifs(out_path);
            if (ifs) {
                std::string content((std::istreambuf_iterator<char>(ifs)),
                                     std::istreambuf_iterator<char>());
                ifs.close();
                picojson::value parsed;
                std::string err = picojson::parse(parsed, content);
                if (err.empty() && parsed.is<picojson::object>()) {
                    root = parsed.get<picojson::object>();
                    if (root.count("progress") && root["progress"].is<picojson::array>())
                        progress = root["progress"].get<picojson::array>();
                    if (root.count("run") && root["run"].is<picojson::object>())
                        root["run"].get<picojson::object>()["status"] = picojson::value("running");
                    write_();
                    return;
                }
            }
        }

        // Create new log
        picojson::object tensor_obj;
        tensor_obj["n0"] = picojson::value(static_cast<double>(dims[0]));
        tensor_obj["n1"] = picojson::value(static_cast<double>(dims[1]));
        tensor_obj["n2"] = picojson::value(static_cast<double>(dims[2]));
        tensor_obj["nnz"] = picojson::value(static_cast<double>(nnz));

        picojson::object cfg;
        cfg["do_reduce"] = picojson::value(do_reduce);
        cfg["do_sge"] = picojson::value(do_sge);
        cfg["sge_beam"] = picojson::value(static_cast<double>(sge_beam));
        cfg["do_fgs"] = picojson::value(do_fgs);

        if (do_fgs) {
            cfg["path_limit"] = picojson::value(static_cast<double>(P.path_limit));
            cfg["pool_size"] = picojson::value(static_cast<double>(P.pool_size));
            cfg["plus_lim"] = picojson::value(static_cast<double>(P.plus_lim));
            cfg["reduce_interval"] = picojson::value(static_cast<double>(P.reduce_interval));
            cfg["threads"] = picojson::value(static_cast<double>(P.threads));
            cfg["max_attempts"] = picojson::value(static_cast<double>(P.max_attempts));
            cfg["use_plus"] = picojson::value(P.use_plus);
            cfg["do_verify"] = picojson::value(P.do_verify);
        }

        cfg["save_pools"] = picojson::value(save_pools);
        cfg["continued"] = picojson::value(continued);

        picojson::object run;
        run["run_id"] = picojson::value(make_run_id());
        run["started_at"] = picojson::value(iso8601_utc_now());
        run["status"] = picojson::value("running");
        run["last_update"] = picojson::value(iso8601_utc_now());

        root["id"] = picojson::value(paths::pad_id(id));
        root["tensor_path"] = picojson::value(tensor_path);
        root["tensor"] = picojson::value(tensor_obj);
        root["config"] = picojson::value(cfg);
        root["run"] = picojson::value(run);

        if constexpr (paths::HAS_RUN)
            root["run_suffix"] = picojson::value(static_cast<double>(paths::RUN_VAL));

        write_();
    }

    void add_preprocess(const std::string& method, int beam,
                        int initial_rank, int final_rank,
                        int iterations, double time_sec) {
        if (!enabled) return;

        picojson::object pre;
        pre["method"] = picojson::value(method);
        if (method == "sge")
            pre["beam"] = picojson::value(static_cast<double>(beam));
        pre["initial_rank"] = picojson::value(static_cast<double>(initial_rank));
        pre["final_rank"] = picojson::value(static_cast<double>(final_rank));
        pre["iterations"] = picojson::value(static_cast<double>(iterations));
        pre["time_sec"] = picojson::value(time_sec);

        root["preprocess"] = picojson::value(pre);
        write_();
    }

    void add_epoch(const WalkOutcome& out, bool best_updated, 
                   int best_rank, const std::string& best_path) {
        if (!enabled) return;

        picojson::object e;
        e["from_rank"] = picojson::value(static_cast<double>(out.current_rank));
        e["result_rank"] = picojson::value(static_cast<double>(out.result_rank));
        e["attempts"] = picojson::value(static_cast<double>(out.attempts_made));
        e["collected"] = picojson::value(static_cast<double>(out.collected_total));
        e["t_collect_sec"] = picojson::value(out.t_collect_sec);

        picojson::object v;
        v["ok"] = picojson::value(static_cast<double>(out.verified_ok));
        v["total"] = picojson::value(static_cast<double>(out.verified_total));
        v["t_sec"] = picojson::value(out.t_verify_sec);
        e["verify"] = picojson::value(v);

        picojson::object perf;
        perf["total_flips"] = picojson::value(static_cast<double>(out.total_flips));
        perf["total_loop_ns"] = picojson::value(static_cast<double>(out.total_loop_ns));
        perf["flips_mps_total"] = picojson::value(out.flips_mps_total);
        perf["avg_inner_sec"] = picojson::value(out.avg_inner_sec);
        perf["total_plus_ok"] = picojson::value(static_cast<double>(out.total_plus_ok));
        perf["total_plus_fail"] = picojson::value(static_cast<double>(out.total_plus_fail));
        perf["total_reduce_calls"] = picojson::value(static_cast<double>(out.total_reduce_calls));
        perf["total_reduce_ok"] = picojson::value(static_cast<double>(out.total_reduce_ok));
        perf["total_reduce_ns"] = picojson::value(static_cast<double>(out.total_reduce_ns));

        picojson::object reduce_by_rank;
        for (const auto& [rank, count] : out.reduce_ok_by_rank)
            reduce_by_rank[std::to_string(rank)] = picojson::value(static_cast<double>(count));
        perf["reduce_ok_by_rank"] = picojson::value(reduce_by_rank);

        perf["avg_flips_per_attempt"] = picojson::value(out.avg_flips_per_attempt);
        perf["avg_plus_ok_per_attempt"] = picojson::value(out.avg_plus_ok_per_attempt);
        perf["avg_plus_fail_per_attempt"] = picojson::value(out.avg_plus_fail_per_attempt);
        perf["avg_reduce_calls_per_attempt"] = picojson::value(out.avg_reduce_calls_per_attempt);
        perf["avg_reduce_ok_per_attempt"] = picojson::value(out.avg_reduce_ok_per_attempt);
        e["perf"] = picojson::value(perf);

        e["best_updated"] = picojson::value(best_updated);
        e["best_rank"] = picojson::value(static_cast<double>(best_rank));
        e["best_path"] = picojson::value(best_path);

        progress.push_back(picojson::value(e));
        write_();
    }

    void finish(int best_rank, std::size_t pool_size, const std::string& best_path) {
        if (!enabled) return;

        root["run"].get<picojson::object>()["status"] = picojson::value("finished");
        root["run"].get<picojson::object>()["last_update"] = picojson::value(iso8601_utc_now());

        picojson::object result;
        result["best_rank"] = picojson::value(static_cast<double>(best_rank));
        result["pool_size"] = picojson::value(static_cast<double>(pool_size));
        result["best_path"] = picojson::value(best_path);
        root["result"] = picojson::value(result);

        write_();
    }

    fs::path path() const { return out_path; }

private:
    static void ensure_dir_(const fs::path& dir) {
        std::error_code ec;
        if (!fs::exists(dir)) fs::create_directories(dir, ec);
    }

    void write_() {
        root["run"].get<picojson::object>()["last_update"] = picojson::value(iso8601_utc_now());
        root["progress"] = picojson::value(progress);

        const fs::path tmp = out_path.string() + ".tmp";
        {
            std::ofstream ofs(tmp, std::ios::binary | std::ios::trunc);
            if (!ofs) return;
            std::string json = picojson::value(root).serialize(true);
            ofs.write(json.data(), static_cast<std::streamsize>(json.size()));
        }

        std::error_code ec;
        fs::rename(tmp, out_path, ec);
        if (ec) {
            fs::remove(out_path, ec);
            fs::rename(tmp, out_path, ec);
        }
    }
};

} // namespace cpd