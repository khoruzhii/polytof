#!/usr/bin/env python3
"""
Reproduce Table 2: GF(2^k) multiplication circuit optimization.

Runs SGE + FGS on three tensor variants:
  - topp  (07xx, 3k x 3k x 3k phase tensor)
  - GF    (05xx, k x k x k field multiplication, base binary)
  - conv  (03xx, k x k x (2k-1) convolution, base binary)

Then optionally runs Waring (TODD) for T-count optimization:
  - Auto: initialized from computed FGS decompositions
  - +MC:  initialized from known multiplicative complexity constructions (make_cpd_gf.py)

Usage:
    python reproduce_tab2.py
"""

import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
BIN_DIR = REPO_DIR / "bin"
SRC_DIR = REPO_DIR / "src"
THIRD_PARTY_DIR = REPO_DIR / "third_party"
TENSORS_DIR = REPO_DIR / "data" / "tensors"
CPD_BASE_DIR = REPO_DIR / "data" / "cpd" / "base"
CPD_TOPP_DIR = REPO_DIR / "data" / "cpd" / "topp"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

THREADS = 8
RECOMPILE = True
RUN_TOFFOLI = True
RUN_T_COUNT = True
NUM_RUNS = 2              # paper: 10   (each run compiled with -DRUN=i for isolated outputs)
SGE_BEAM = 8              # paper: 1024    (-b, SGE beam width)
FGS_POOL_SIZE = 40        # paper: 1000    (-s)
FGS_PATH_LIMIT = 1000000  # paper: 1000000  (-f)
FGS_PLUS_LIM = 50000      # paper: 50000   (-p)
FGS_MAX_ATTEMPTS = 200    # paper: 1000    (-m)
WARING_NUM = 1            # paper: 1000
WARING_BEAM = 1           # paper: 1
USE_MC = True             # include multiplicative complexity constructions in CPD pool

CXX = "g++"
CXXFLAGS = ["-Ofast", "-std=c++20", "-march=native", "-s", "-pthread"]

# ---------------------------------------------------------------------------
# GF(2^k) circuit registry
#
# Tensor IDs:  tid = k + 23
#   topp = 700 + tid,  base_gf = 500 + tid,  conv = 300 + tid
#
# Paper values from Table 2 ("This work" columns):
#   paper_gen: General (SGE+FGS on topp)
#   paper_bil: Bilinear (best of GF and conv)
#   paper_auto: T-count Auto (TODD from FGS CPDs)
#   paper_mc: T-count +MC (TODD from cenk_2010 constructions)
# ---------------------------------------------------------------------------

GF_CIRCUITS = [
    # (k, paper_gen, paper_bil, paper_auto, paper_mc)
    ( 2,   3,   3,   17,   17),
    ( 3,   6,   6,   29,   29),
    ( 4,   9,   9,   39,   45),
    ( 5,  13,  13,   59,   63),
    ( 6,  15,  15,   77,   81),
    ( 7,  23,  22,  101,  103),
    ( 8,  34,  26,  123,  127),
    ( 9,  51,  32,  153,  147),
    (10,  66,  39,  185,  173),
]


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------


def compile_binaries():
    """Compile topp, base, and waring binaries.

    When NUM_RUNS > 1, compiles per-run variants with -DRUN=i suffix
    (e.g. topp1_000, topp1_001, ...) for isolated output files.
    """
    os.makedirs(BIN_DIR, exist_ok=True)

    # Waring doesn't need per-run variants
    fixed_targets = [
        ("waring1", "waring.cpp", ["-DVEC_WORDS=1"]),
    ]
    # CPD binaries get per-run variants when NUM_RUNS > 1
    cpd_targets = [
        ("topp1",   "cpd.cpp",    ["-DVEC_WORDS=1"]),
        ("base1",   "cpd.cpp",    ["-DVEC_WORDS=1", "-DBASE"]),
    ]

    print("=== Compilation ===")

    for name, src, extra in fixed_targets:
        out = BIN_DIR / name
        cmd = (
            [CXX] + extra + CXXFLAGS
            + [f"-I{THIRD_PARTY_DIR}", f"-I{SRC_DIR}"]
            + [str(SRC_DIR / src), "-o", str(out)]
        )
        print(f"  {name}...", end=" ", flush=True)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("FAILED")
            print(result.stderr)
            sys.exit(1)
        print("ok")

    for name, src, extra in cpd_targets:
        if NUM_RUNS <= 1:
            out = BIN_DIR / name
            cmd = (
                [CXX] + extra + CXXFLAGS
                + [f"-I{THIRD_PARTY_DIR}", f"-I{SRC_DIR}"]
                + [str(SRC_DIR / src), "-o", str(out)]
            )
            print(f"  {name}...", end=" ", flush=True)
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print("FAILED")
                print(result.stderr)
                sys.exit(1)
            print("ok")
        else:
            for run_i in range(NUM_RUNS):
                run_name = f"{name}_{run_i:03d}"
                out = BIN_DIR / run_name
                cmd = (
                    [CXX] + extra + [f"-DRUN={run_i}"] + CXXFLAGS
                    + [f"-I{THIRD_PARTY_DIR}", f"-I{SRC_DIR}"]
                    + [str(SRC_DIR / src), "-o", str(out)]
                )
                print(f"  {run_name}...", end=" ", flush=True)
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print("FAILED")
                    print(result.stderr)
                    sys.exit(1)
                print("ok")

    print()


# ---------------------------------------------------------------------------
# ID helpers
# ---------------------------------------------------------------------------


def topp_id(k):
    return 700 + k + 23

def base_gf_id(k):
    return 500 + k + 23

def conv_id(k):
    return 300 + k + 23


# ---------------------------------------------------------------------------
# Bilinear CPD → topp conversion
# ---------------------------------------------------------------------------

PRIMITIVE_POLYS = {
    2:  [2, 1, 0],
    3:  [3, 1, 0],
    4:  [4, 1, 0],
    5:  [5, 2, 0],
    6:  [6, 1, 0],
    7:  [7, 1, 0],
    8:  [8, 4, 3, 2, 0],
    9:  [9, 4, 0],
    10: [10, 3, 0],
}


def reduction_matrix(k):
    """k x (2k-1) GF(2) reduction matrix: maps polynomial product back to GF(2^k)."""
    R = np.zeros((k, 2 * k - 1), dtype=np.uint8)
    R[:k, :k] = np.eye(k, dtype=np.uint8)
    # x^k mod p(x)
    red = np.zeros(k, dtype=np.uint8)
    for e in PRIMITIVE_POLYS[k]:
        if e < k:
            red[e] = 1
    prev = red.copy()
    R[:, k] = prev
    for i in range(k + 1, 2 * k - 1):
        carry = prev[k - 1]
        nxt = np.zeros(k, dtype=np.uint8)
        nxt[1:] = prev[:-1]
        if carry:
            nxt = (nxt + red) % 2
        prev = nxt
        R[:, i] = prev
    return R


def find_cpd_files(tensor_id, cpd_dir):
    """Find all CPD files for tensor_id, grouped by rank. Returns {rank: [paths]}."""
    pattern = re.compile(rf"^{tensor_id:04d}-(\d{{5}})(?:-\d{{3}})?\.npy$")
    by_rank = {}
    if not cpd_dir.exists():
        return by_rank
    for f in cpd_dir.iterdir():
        m = pattern.match(f.name)
        if m:
            rank = int(m.group(1))
            by_rank.setdefault(rank, []).append(f)
    return by_rank


def _convert_row_to_topp(row, k, is_conv, R):
    """Convert one base/conv CPD row (interleaved uint64) to topp format."""
    n_terms = len(row) // 3
    topp_row = np.zeros(3 * n_terms, dtype=np.uint64)
    w_dim = 2 * k - 1 if is_conv else k
    for t in range(n_terms):
        u_val = int(row[3 * t])
        v_val = int(row[3 * t + 1])
        w_val = int(row[3 * t + 2])
        topp_row[3 * t] = np.uint64(u_val)
        topp_row[3 * t + 1] = np.uint64(v_val) << np.uint64(k)
        if is_conv:
            w_vec = np.array([(w_val >> i) & 1 for i in range(w_dim)], dtype=np.uint8)
            w_red = R @ w_vec % 2
            w_bits = int(sum(int(b) << i for i, b in enumerate(w_red)))
        else:
            w_bits = w_val
        topp_row[3 * t + 2] = np.uint64(w_bits) << np.uint64(2 * k)
    return topp_row


def _matrices_to_row(U, V, W):
    """Factor matrices (rank x dim, uint8) → interleaved uint64 row."""
    def to_bits(M):
        return (M.astype(np.uint64) << np.arange(M.shape[1], dtype=np.uint64)).sum(axis=1)
    return np.column_stack([to_bits(U), to_bits(V), to_bits(W)]).ravel().astype(np.uint64)


def merge_cpds_to_topp(k):
    """Merge converted base/conv CPDs (and optionally MC) into topp pool files."""
    CPD_TOPP_DIR.mkdir(parents=True, exist_ok=True)
    tid_topp = topp_id(k)

    # Collect new schemes grouped by rank
    new_schemes = {}  # {rank: [uint64 row, ...]}

    # Convert base_gf and conv CPDs
    for base_tid, is_conv in [(base_gf_id(k), False), (conv_id(k), True)]:
        by_rank = find_cpd_files(base_tid, CPD_BASE_DIR)
        if not by_rank:
            continue
        best_rank = min(by_rank)
        R = reduction_matrix(k) if is_conv else None
        for cpd_path in by_rank[best_rank]:
            data = np.load(cpd_path)
            for row_idx in range(data.shape[0]):
                converted = _convert_row_to_topp(data[row_idx], k, is_conv, R)
                new_schemes.setdefault(best_rank, []).append(converted)

    # MC construction (inline, from make_cpd_gf)
    if USE_MC:
        try:
            sys.path.insert(0, str(SCRIPT_DIR))
            from make_cpd_gf import SCHEMES as MC_SCHEMES
            U, V, W = MC_SCHEMES[k]()
            mc_rank = U.shape[0]
            m = 3 * k
            U_t = np.zeros((mc_rank, m), dtype=np.uint8)
            V_t = np.zeros((mc_rank, m), dtype=np.uint8)
            W_t = np.zeros((mc_rank, m), dtype=np.uint8)
            U_t[:, :k] = U
            V_t[:, k:2*k] = V
            W_t[:, 2*k:3*k] = W
            mc_row = _matrices_to_row(U_t, V_t, W_t)
            new_schemes.setdefault(mc_rank, []).append(mc_row)
        except Exception:
            pass

    # Save: for each rank, concatenate with existing topp file
    for rank, rows in new_schemes.items():
        out_path = CPD_TOPP_DIR / f"{tid_topp:04d}-{rank:05d}.npy"
        new_data = np.vstack([r.reshape(1, -1) for r in rows])
        if out_path.exists():
            existing = np.load(out_path)
            if existing.shape[1] == new_data.shape[1]:
                new_data = np.vstack([existing, new_data])
        np.save(out_path, new_data)


# ---------------------------------------------------------------------------
# SGE+FGS, Waring runners
# ---------------------------------------------------------------------------


def run_sge_fgs(tensor_id, binary, use_sge=True):
    """Run SGE + FGS on a tensor. When NUM_RUNS > 1, runs all variants and returns best.

    Args:
        use_sge: Include --sge preprocessing (only for topp binaries, not base).

    Returns (cp_rank, elapsed) or None.
    """
    tensor_path = TENSORS_DIR / f"{tensor_id:04d}.npy"
    if not tensor_path.exists():
        return None

    binaries = (
        [f"{binary}_{i:03d}" for i in range(NUM_RUNS)]
        if NUM_RUNS > 1 else [binary]
    )

    best_rank = None
    total_elapsed = 0.0

    for bin_name in binaries:
        cmd = [str(BIN_DIR / bin_name), str(tensor_id)]
        if use_sge:
            cmd += ["--sge", "-b", str(SGE_BEAM)]
        cmd += [
            "--fgs", "--plus",
            "-s", str(FGS_POOL_SIZE),
            "-f", str(FGS_PATH_LIMIT),
            "-p", str(FGS_PLUS_LIM),
            "-m", str(FGS_MAX_ATTEMPTS),
            "-t", str(THREADS),
            "--save", "--verify",
        ]
        t0 = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_DIR))
        total_elapsed += time.time() - t0

        m_rank = re.search(r"Best rank:\s+(\d+)", result.stdout)
        if m_rank:
            rank = int(m_rank.group(1))
            if best_rank is None or rank < best_rank:
                best_rank = rank
        else:
            print(f"\n  WARNING: SGE+FGS {tensor_id:04d} ({bin_name}) failed")
            if result.stdout:
                print(result.stdout[:300])

    if best_rank is None:
        return None
    return best_rank, total_elapsed


def run_waring(tensor_id):
    """Run Waring (TODD) with CPD initialization. Returns (waring_rank, elapsed) or None."""
    tensor_path = TENSORS_DIR / f"{tensor_id:04d}.npy"
    if not tensor_path.exists():
        return None

    cmd = [
        str(BIN_DIR / "waring1"), str(tensor_id),
        "--cpd",
        "-n", str(WARING_NUM),
        "-b", str(WARING_BEAM),
        "-t", str(THREADS),
        "--save", "--verify",
    ]
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_DIR))
    elapsed = time.time() - t0

    m = re.search(r"\(min (\d+),", result.stdout)
    if not m:
        # Check if waring found no CPDs
        if "No CPD files" in result.stderr or "No CPD files" in result.stdout:
            return None
        print(f"\n  WARNING: Waring {tensor_id:04d} failed")
        if result.stdout:
            print(result.stdout[:300])
        return None

    return int(m.group(1)), elapsed


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------


def fmt(val, width=5):
    if val is None:
        return "--".rjust(width)
    return str(val).rjust(width)


# Toffoli table
TOF_HDR1 = "                         This try             Paper"
TOF_HDR2 = "   k  Circuit           Gen GF Bil GF Bil C  Gen GF  Bil   t,s"
TOF_SEP  = "----  ----------------  ------ ------ -----  ------ -----  -----"

# T-count table
WAR_HDR1 = "                       This try      Paper"
WAR_HDR2 = "   k  Circuit           T-cnt    Auto   +MC   t,s"
WAR_SEP  = "----  ----------------  -----   ----- -----  -----"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    if RECOMPILE:
        compile_binaries()

    # ------------------------------------------------------------------
    # Phase 1: Toffoli minimization
    # ------------------------------------------------------------------

    if RUN_TOFFOLI:
        print(f"=== Table 2: Toffoli Minimization "
              f"(SGE beam={SGE_BEAM}, FGS pool={FGS_POOL_SIZE}) ===\n")
        print(TOF_HDR1)
        print(TOF_HDR2)
        print(TOF_SEP)
        total_time = 0.0

        for k, paper_gen, paper_bil, _, _ in GF_CIRCUITS:

            tid_topp = topp_id(k)
            tid_gf = base_gf_id(k)
            tid_conv = conv_id(k)
            elapsed = 0.0
            gen_rank = gf_rank = conv_rank = None

            # General: SGE+FGS on topp
            fgs_res = run_sge_fgs(tid_topp, "topp1")
            if fgs_res is not None:
                gen_rank, t = fgs_res
                elapsed += t

            # Bilinear GF: FGS on base (no SGE for base variant)
            fgs_res = run_sge_fgs(tid_gf, "base1", use_sge=False)
            if fgs_res is not None:
                gf_rank, t = fgs_res
                elapsed += t

            # Bilinear conv: FGS on base (no SGE for base variant)
            fgs_res = run_sge_fgs(tid_conv, "base1", use_sge=False)
            if fgs_res is not None:
                conv_rank, t = fgs_res
                elapsed += t

            # Merge bilinear (+ MC) CPDs into topp pool for waring
            merge_cpds_to_topp(k)

            total_time += elapsed

            name = f"GF(2^{k}) Mult"
            t_str = fmt(round(elapsed), 5)
            print(
                f"  {k:>2d}  {name:<16s}"
                f"  {fmt(gen_rank, 6)} {fmt(gf_rank, 6)} {fmt(conv_rank, 5)}"
                f"  {fmt(paper_gen, 6)} {fmt(paper_bil, 5)}"
                f"  {t_str}"
            )

        print(TOF_SEP)
        print(f"  Total time: {total_time:.0f}s\n")

    # ------------------------------------------------------------------
    # Phase 2: T-count minimization
    # ------------------------------------------------------------------

    if RUN_T_COUNT:
        print(f"=== Table 2: T-count Minimization (Waring) ===\n")
        print(WAR_HDR1)
        print(WAR_HDR2)
        print(WAR_SEP)
        total_time = 0.0

        for k, _, _, paper_auto, paper_mc in GF_CIRCUITS:

            tid_topp = topp_id(k)
            elapsed = 0.0
            t_rank = None

            war_res = run_waring(tid_topp)
            if war_res is not None:
                t_rank, t = war_res
                elapsed += t

            total_time += elapsed

            name = f"GF(2^{k}) Mult"
            t_str = fmt(round(elapsed), 5)
            print(
                f"  {k:>2d}  {name:<16s}"
                f"  {fmt(t_rank, 5)}"
                f"   {fmt(paper_auto, 5)} {fmt(paper_mc, 5)}"
                f"  {t_str}"
            )

        print(WAR_SEP)
        print(f"  Total time: {total_time:.0f}s\n")


if __name__ == "__main__":
    main()
