// cpd/walk.h
// Generic flip graph walk executor for pool-based GF(2) tensor decomposition search.
// Collects schemes with rank < current_rank, stops when any bin reaches pool_size.
// Select variant via -DTOPP or -DBASE at compile time.

#pragma once

#include <vector>
#include <map>
#include <array>
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>
#include <condition_variable>
#include <algorithm>
#include <optional>
#include <cstdint>

#include "core/types.h"
#include "core/bit_arr.h"
#include "core/random.h"
#include "cpd/config.h"

namespace cpd {

// -----------------------------------------------------------------------------
// Data structures
// -----------------------------------------------------------------------------

struct SchemeData {
    std::vector<BitArr> data;
    int rank = 0;
    std::array<int, 3> dims{};
    U32 seed = 0;

    SchemeData() = default;
    SchemeData(std::vector<BitArr> d, int r, std::array<int, 3> dm, U32 s)
        : data(std::move(d)), rank(r), dims(dm), seed(s) {}
};

struct alignas(64) WalkStats {
    U64 flips = 0;
    U64 loop_ns = 0;
    U64 plus_ok = 0;
    U64 plus_fail = 0;
    U64 reduce_calls = 0;
    U64 reduce_ok = 0;
    U64 reduce_ns = 0;

    void merge(const WalkStats& other) {
        flips += other.flips;
        loop_ns += other.loop_ns;
        plus_ok += other.plus_ok;
        plus_fail += other.plus_fail;
        reduce_calls += other.reduce_calls;
        reduce_ok += other.reduce_ok;
        reduce_ns += other.reduce_ns;
    }
};

struct WalkParams {
    int path_limit = 1'000'000;
    int pool_size = 200;
    int plus_lim = 50'000;
    int reduce_interval = 0;
    int threads = 4;
    int max_attempts = 1000;
    bool use_plus = false;
    bool do_verify = false;
};

struct WalkOutcome {
    std::vector<SchemeData> next_pool;
    int result_rank = 0;
    int current_rank = 0;
    int attempts_made = 0;
    std::size_t collected_total = 0;
    std::size_t verified_ok = 0;
    std::size_t verified_total = 0;
    double t_collect_sec = 0.0;
    double t_verify_sec = 0.0;

    // Aggregated stats
    U64 total_flips = 0;
    U64 total_loop_ns = 0;
    U64 total_plus_ok = 0;
    U64 total_plus_fail = 0;
    U64 total_reduce_calls = 0;
    U64 total_reduce_ok = 0;
    U64 total_reduce_ns = 0;
    std::map<int, U64> reduce_ok_by_rank;

    // Derived metrics
    double flips_mps_total = 0.0;
    double avg_inner_sec = 0.0;
    double avg_flips_per_attempt = 0.0;
    double avg_plus_ok_per_attempt = 0.0;
    double avg_plus_fail_per_attempt = 0.0;
    double avg_reduce_calls_per_attempt = 0.0;
    double avg_reduce_ok_per_attempt = 0.0;
};

// -----------------------------------------------------------------------------
// Walk executor
// -----------------------------------------------------------------------------

class WalkExecutor {
public:
    explicit WalkExecutor(const WalkParams& p, U64 global_seed)
        : params_(p), global_seed_(global_seed) {}

    WalkOutcome run(int current_rank, const std::vector<SchemeData>& pool, const Tensor& T) {
        WalkOutcome out;
        out.current_rank = current_rank;

        if (current_rank <= 1 || pool.empty()) {
            out.result_rank = current_rank;
            return out;
        }

        auto t0 = std::chrono::steady_clock::now();

        init_bins(current_rank);
        reset_state();
        thread_stats_.assign(params_.threads, WalkStats{});
        active_workers_ = params_.threads;

        // Launch workers
        std::vector<std::thread> workers;
        workers.reserve(params_.threads);
        for (int i = 0; i < params_.threads; ++i)
            workers.emplace_back(&WalkExecutor::worker_loop, this, i, current_rank, std::cref(pool));

        // Wait for completion or full bin
        {
            std::unique_lock<std::mutex> lk(cv_mutex_);
            cv_.wait(lk, [&] {
                return bin_full_.load(std::memory_order_relaxed) || active_workers_ == 0;
            });
        }

        stop_.store(true, std::memory_order_relaxed);
        cv_.notify_all();
        for (auto& w : workers) w.join();

        out.result_rank = find_best_rank(current_rank);
        out.attempts_made = attempts_.load(std::memory_order_relaxed);

        // Aggregate stats
        WalkStats total;
        for (const auto& s : thread_stats_) total.merge(s);

        out.total_flips = total.flips;
        out.total_loop_ns = total.loop_ns;
        out.total_plus_ok = total.plus_ok;
        out.total_plus_fail = total.plus_fail;
        out.total_reduce_calls = total.reduce_calls;
        out.total_reduce_ok = total.reduce_ok;
        out.total_reduce_ns = total.reduce_ns;

        {
            std::lock_guard<std::mutex> lk(reduce_mutex_);
            out.reduce_ok_by_rank = reduce_by_rank_;
        }

        compute_derived_metrics(out);

        auto t1 = std::chrono::steady_clock::now();
        out.t_collect_sec = std::chrono::duration<double>(t1 - t0).count();

        // Extract collected schemes
        std::vector<SchemeData> collected;
        {
            std::lock_guard<std::mutex> lk(bins_mutex_);
            if (bins_.count(out.result_rank))
                collected = std::move(bins_[out.result_rank]);
        }
        out.collected_total = collected.size();

        // Verify and finalize
        out.next_pool = verify_and_trim(collected, T, out);
        return out;
    }

private:
    // Search state for a single path
    struct SearchState {
        U64 flips = 0;
        U64 plus_ok = 0;
        U64 plus_fail = 0;
        U64 reduce_calls = 0;
        U64 reduce_ok = 0;
        U64 reduce_ns = 0;
        std::chrono::steady_clock::time_point t0;

        void begin() { t0 = std::chrono::steady_clock::now(); }

        void commit(WalkStats& out) {
            auto t1 = std::chrono::steady_clock::now();
            out.flips += flips;
            out.loop_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            out.plus_ok += plus_ok;
            out.plus_fail += plus_fail;
            out.reduce_calls += reduce_calls;
            out.reduce_ok += reduce_ok;
            out.reduce_ns += reduce_ns;
        }
    };

    std::optional<SchemeData> search_path(const SchemeData& start, U64 seed, WalkStats& stats,
                                          std::map<int, U64>& reduce_by_rank) {
        if (start.rank <= 1) return std::nullopt;

        Scheme scheme(start.data, start.dims, seed);
        const int start_rank = start.rank;
        int flips_since_plus = 0, flips_since_reduce = 0;

        SearchState st;
        st.begin();

        for (int step = 0; step < params_.path_limit; ++step) {
            ++st.flips;

            if (!scheme.flip()) {
                if (params_.use_plus) {
                    if (scheme.plus()) { ++st.plus_ok; flips_since_plus = 0; }
                    else { ++st.plus_fail; break; }
                }
            }

            int rank = scheme.get_rank();
            if (rank < start_rank) {
                st.commit(stats);
                return SchemeData(scheme.release_data(), rank, start.dims, static_cast<U32>(seed));
            }

            ++flips_since_plus;
            ++flips_since_reduce;

            // Periodic reduce
            if (params_.reduce_interval > 0 && flips_since_reduce >= params_.reduce_interval) {
                ++st.reduce_calls;
                auto tr0 = std::chrono::steady_clock::now();
                bool ok = scheme.reduce();
                auto tr1 = std::chrono::steady_clock::now();
                st.reduce_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(tr1 - tr0).count();

                if (ok) {
                    ++st.reduce_ok;
                    rank = scheme.get_rank();
                    ++reduce_by_rank[rank];
                    if (rank < start_rank) {
                        st.commit(stats);
                        return SchemeData(scheme.release_data(), rank, start.dims, static_cast<U32>(seed));
                    }
                }
                flips_since_reduce = 0;
            }

            // Periodic plus
            if (params_.use_plus && flips_since_plus >= params_.plus_lim) {
                if (scheme.plus()) { ++st.plus_ok; flips_since_plus = 0; }
                else ++st.plus_fail;
            }
        }

        st.commit(stats);
        return std::nullopt;
    }

    void worker_loop(int tid, int current_rank, const std::vector<SchemeData>& pool) {
        struct OnExit {
            int& active; std::mutex& m; std::condition_variable& cv;
            ~OnExit() { std::lock_guard<std::mutex> lk(m); --active; cv.notify_all(); }
        } on_exit{active_workers_, cv_mutex_, cv_};

        std::map<int, U64> local_reduce;
        WalkStats& my_stats = thread_stats_[tid];

        U64 stream = (static_cast<U64>(current_rank) << 32) | tid;
        Rng rng(derive_seed(global_seed_, stream));

        while (!stop_.load(std::memory_order_relaxed)) {
            int attempt = attempts_.fetch_add(1, std::memory_order_relaxed);
            if (attempt >= params_.max_attempts || pool.empty()) break;

            std::size_t idx = rng.next_u64() % pool.size();
            U64 scheme_seed = derive_seed(global_seed_, (static_cast<U64>(current_rank) << 32) | attempt);

            if (auto res = search_path(pool[idx], scheme_seed, my_stats, local_reduce))
                publish(*res);
        }

        if (!local_reduce.empty()) {
            std::lock_guard<std::mutex> lk(reduce_mutex_);
            for (auto& [r, c] : local_reduce) reduce_by_rank_[r] += c;
        }
    }

    void publish(SchemeData s) {
        std::lock_guard<std::mutex> lk(bins_mutex_);
        auto& bin = bins_[s.rank];
        if (bin.size() < static_cast<std::size_t>(params_.pool_size)) {
            bin.push_back(std::move(s));
            if (bin.size() >= static_cast<std::size_t>(params_.pool_size)) {
                bin_full_.store(true, std::memory_order_relaxed);
                cv_.notify_one();
            }
        }
    }

    void init_bins(int current_rank) {
        std::lock_guard<std::mutex> lk(bins_mutex_);
        for (auto it = bins_.begin(); it != bins_.end(); )
            it = (it->first >= current_rank) ? bins_.erase(it) : ++it;
        for (int r = 0; r < current_rank; ++r)
            if (!bins_.count(r)) bins_[r].reserve(params_.pool_size);
    }

    void reset_state() {
        bin_full_.store(false, std::memory_order_relaxed);
        attempts_.store(0, std::memory_order_relaxed);
        stop_.store(false, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lk(reduce_mutex_);
        reduce_by_rank_.clear();
    }

    int find_best_rank(int current_rank) {
        std::lock_guard<std::mutex> lk(bins_mutex_);
        int best = current_rank;
        // First: find full bin with minimum rank
        for (auto& [r, bin] : bins_)
            if (bin.size() >= static_cast<std::size_t>(params_.pool_size) && r < best)
                best = r;
        // Fallback: any non-empty bin
        if (best == current_rank)
            for (auto& [r, bin] : bins_)
                if (!bin.empty() && r < best) best = r;
        return best;
    }

    void compute_derived_metrics(WalkOutcome& out) {
        double inner_sec = out.total_loop_ns * 1e-9;
        out.avg_inner_sec = (params_.threads > 0) ? inner_sec / params_.threads : 0.0;
        if (out.avg_inner_sec > 0.0)
            out.flips_mps_total = (out.total_flips / out.avg_inner_sec) / 1e6;

        if (out.attempts_made > 0) {
            double d = out.attempts_made;
            out.avg_flips_per_attempt = out.total_flips / d;
            out.avg_plus_ok_per_attempt = out.total_plus_ok / d;
            out.avg_plus_fail_per_attempt = out.total_plus_fail / d;
            out.avg_reduce_calls_per_attempt = out.total_reduce_calls / d;
            out.avg_reduce_ok_per_attempt = out.total_reduce_ok / d;
        }
    }

    std::vector<SchemeData> verify_and_trim(std::vector<SchemeData>& collected,
                                            const Tensor& T, WalkOutcome& out) {
        std::vector<SchemeData> result;

        if (params_.do_verify && !collected.empty()) {
            auto t0 = std::chrono::steady_clock::now();
            std::size_t N = collected.size();
            std::vector<std::uint8_t> ok(N, 0);

            // Parallel verification
            std::vector<std::thread> threads;
            threads.reserve(params_.threads);
            std::size_t chunk = (N + params_.threads - 1) / params_.threads;

            for (int i = 0; i < params_.threads; ++i) {
                threads.emplace_back([&, i] {
                    std::size_t lo = i * chunk, hi = std::min(lo + chunk, N);
                    for (std::size_t j = lo; j < hi; ++j)
                        if (verify(collected[j].data, T)) ok[j] = 1;
                });
            }
            for (auto& t : threads) t.join();

            out.verified_total = N;
            for (std::size_t i = 0; i < N; ++i)
                if (ok[i]) { ++out.verified_ok; result.push_back(std::move(collected[i])); }

            out.t_verify_sec = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
        } else {
            result = std::move(collected);
            out.verified_total = out.verified_ok = result.size();
        }

        // Shuffle and trim
        if (result.size() > static_cast<std::size_t>(params_.pool_size)) {
            Rng rng(derive_seed(global_seed_, out.current_rank ^ 0x123456789abcdefULL));
            for (std::size_t i = result.size(); i > 1; --i)
                std::swap(result[i - 1], result[rng.next_u64() % i]);
            result.resize(params_.pool_size);
        }

        return result;
    }

    // Config
    WalkParams params_;
    U64 global_seed_;

    // Synchronization
    std::mutex cv_mutex_;
    std::condition_variable cv_;
    std::atomic<bool> stop_{false};
    std::atomic<bool> bin_full_{false};
    std::atomic<int> attempts_{0};
    int active_workers_ = 0;

    // Results
    std::mutex bins_mutex_;
    std::map<int, std::vector<SchemeData>> bins_;

    std::mutex reduce_mutex_;
    std::map<int, U64> reduce_by_rank_;

    std::vector<WalkStats> thread_stats_;
};

} // namespace cpd