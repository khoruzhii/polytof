#!/usr/bin/env python3
"""
Reproduce Table 1 (FGS + Waring): Toffoli and T-count optimization.

Runs the full pipeline on circuit benchmarks:
  1. BCO  (only for large circuits that need re-optimization)
  2. FGS  (flip graph search -> CP rank = Toffoli count)
  3. TODD (Waring decomposition -> Waring rank = T-count)

Assumes BCO tensors (1xxx) already exist from reproduce_tab1.py.
For Ham15 (high) and Mod Adder_1024, re-runs BCO with BCO_BEAM.

Usage:
    python reproduce_tab1_waring.py
"""

import os
import re
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
BIN_DIR = REPO_DIR / "bin"
SRC_DIR = REPO_DIR / "src"
THIRD_PARTY_DIR = REPO_DIR / "third_party"
TENSORS_DIR = REPO_DIR / "data" / "tensors"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

THREADS = 8
RECOMPILE = False
RUN_ADDITIONAL = False
BCO_BEAM = 8             # paper: 1024, only for large circuits: Ham15 high, Mod Adder 1024
FGS_POOL_SIZE = 40       # paper: 1000    (-s)
FGS_PATH_LIMIT = 1000000 # paper: 1000000  (-f)
FGS_PLUS_LIM = 50000     # paper: 50000   (-p)
FGS_MAX_ATTEMPTS = 200   # paper: 100000    (-m)
WARING_NUM = 10000       # paper: all schemes
WARING_BEAM = 1          # paper: 1

CXX = "g++"
CXXFLAGS = ["-Ofast", "-std=c++20", "-march=native", "-s", "-pthread"]

# ---------------------------------------------------------------------------
# Circuit registry: (suffix, name, paper_cp, paper_waring)
#
# suffix s -> tensor IDs: ATQ = 100+s, VV = 800+s
# paper_cp / paper_waring: best known from data/paper/ (see docs/tensors.md)
# ---------------------------------------------------------------------------

CIRCUITS_MAIN = [
    ( 0, "Adder_8",          27, 117),
    ( 1, "Barenco Tof_3",     2,  13),
    ( 2, "Barenco Tof_4",     4,  23),
    ( 3, "Barenco Tof_5",     6,  33),
    ( 4, "Barenco Tof_10",   16,  83),
    ( 5, "CSLA MUX_3",        8,  39),
    ( 6, "CSUM MUX_9",       14,  71),
    ( 7, "Grover_5",         25, 143),
    ( 8, "Ham15 (low)",      17,  73),
    ( 9, "Ham15 (med)",      33, 137),
    (10, "Ham15 (high)",    155, 643),
    (11, "HWB_6",            10,  51),
    (12, "Mod 5_4",           1,   7),
    (13, "Mod Adder_1024",  128, 573),
    (14, "Mod Mult_55",       3,  17),
    (15, "Mod Red_21",       11,  51),
    (16, "QCLA Adder_10",   24, 107),
    (17, "QCLA Com_7",       12,  59),
    (18, "QCLA Mod_7",       37, 153),
    (19, "RC Adder_6",        6,  37),
    (20, "Tof_3",             2,  13),
    (21, "Tof_4",             3,  19),
    (22, "Tof_5",             4,  25),
    (23, "Tof_10",            9,  55),
    (24, "VBE Adder_3",       3,  19),
]

# Additional ATQ-only benchmarks (01xx series)
CIRCUITS_ADDITIONAL = [
    # Binary Addition (Cuccaro Adder)
    (34, "Cuccaro n=3",    2,  13),
    (35, "Cuccaro n=4",    3,  19),
    (36, "Cuccaro n=5",    4,  25),
    (37, "Cuccaro n=6",    5,  31),
    (38, "Cuccaro n=7",    6,  37),
    (39, "Cuccaro n=8",    7,  43),
    (40, "Cuccaro n=9",    8,  49),
    (41, "Cuccaro n=10",   9,  55),
    # Quantum Chemistry (Basis Change)
    (42, "BasisCh p4o3",   8,  41),
    (43, "BasisCh p4o4",  12,  61),
    (44, "BasisCh p4o5",  16,  81),
    (45, "BasisCh p5o3",  12,  61),
    (46, "BasisCh p5o4",  18,  91),
    (47, "BasisCh p6o3",  16,  85),
    (48, "BasisCh p7o3",  20, 105),
    # Hamming Weight / Phase Gradient
    (49, "HammWt n=4",     3,  19),
    (50, "HammWt n=5",     3,  19),
    (51, "HammWt n=6",     4,  25),
    (52, "HammWt n=7",     4,  25),
    (53, "HammWt n=8",     7,  43),
    (54, "HammWt n=9",     7,  43),
    (55, "HammWt n=10",    8,  49),
    (56, "HammWt n=11",    8,  49),
    (57, "HammWt n=12",   10,  61),
    (58, "HammWt n=13",   10,  61),
    (59, "HammWt n=14",   11,  67),
    (60, "HammWt n=15",   11,  67),
    (61, "HammWt n=16",   15,  91),
    (62, "HammWt n=17",   15,  91),
    (63, "HammWt n=18",   16,  97),
    (64, "HammWt n=19",   16,  97),
    (65, "HammWt n=20",   18, 109),
    # Unary Iteration
    (66, "UnaryIt n=3",    7,  31),
    (67, "UnaryIt n=4",   15,  63),
    (68, "UnaryIt n=5",   31, 127),
]

# Circuits that need BCO re-run with higher beam (large tensors)
LARGE_CIRCUITS = {10, 13}  # Ham15 (high), Mod Adder_1024


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------


def compile_binaries():
    """Compile BCO, FGS (topp) and Waring binaries."""
    os.makedirs(BIN_DIR, exist_ok=True)
    targets = [
        ("bco",     "bco.cpp",    []),
        ("topp1",   "cpd.cpp",    ["-DVEC_WORDS=1"]),
        ("topp2",   "cpd.cpp",    ["-DVEC_WORDS=2"]),
        ("topp6",   "cpd.cpp",    ["-DVEC_WORDS=6"]),
        ("waring1", "waring.cpp", ["-DVEC_WORDS=1"]),
        ("waring2", "waring.cpp", ["-DVEC_WORDS=2"]),
        ("waring6", "waring.cpp", ["-DVEC_WORDS=6"]),
    ]
    print("=== Compilation ===")
    for name, src, extra in targets:
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
    print()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def topp_binary(n):
    """Select topp binary based on tensor dimension n."""
    if n <= 64:
        return "topp1"
    elif n <= 128:
        return "topp2"
    else:
        return "topp6"


def waring_binary(n):
    """Select waring binary based on tensor dimension n."""
    if n <= 64:
        return "waring1"
    elif n <= 128:
        return "waring2"
    else:
        return "waring6"


def get_tensor_n(tensor_id):
    """Read tensor dimension n from the .npy file header."""
    import numpy as np
    path = TENSORS_DIR / f"{tensor_id:04d}.npy"
    if not path.exists():
        return None
    data = np.load(path)
    return int(data[0, 0])


# ---------------------------------------------------------------------------
# BCO, FGS, Waring runners
# ---------------------------------------------------------------------------


def run_bco(tensor_id):
    """Run BCO on a tensor. Returns (n, nnz_after, elapsed) or None."""
    tensor_path = TENSORS_DIR / f"{tensor_id:04d}.npy"
    if not tensor_path.exists():
        return None

    cmd = [
        str(BIN_DIR / "bco"), str(tensor_id),
        "-b", str(BCO_BEAM), "-t", str(THREADS), "--save", "--verify",
    ]
    t0 = time.time()
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(REPO_DIR)
    )
    elapsed = time.time() - t0

    m_n = re.search(r"n=(\d+)", result.stdout)
    m_final = re.search(r"Final:\s+(\d+)", result.stdout)
    if not (m_n and m_final):
        print(f"\n  WARNING: BCO {tensor_id:04d} failed")
        if result.stdout:
            print(result.stdout[:200])
        return None

    return int(m_n.group(1)), int(m_final.group(1)), elapsed


def run_fgs(tensor_id, n):
    """Run SGE + FGS on BCO-optimized tensor (id+1000). Returns (cp_rank, elapsed) or None."""
    bco_id = tensor_id + 1000
    tensor_path = TENSORS_DIR / f"{bco_id:04d}.npy"
    if not tensor_path.exists():
        return None

    binary = topp_binary(n)
    cmd = [
        str(BIN_DIR / binary), str(bco_id),
        "--fgs", "--plus",
        "-s", str(FGS_POOL_SIZE),
        "-f", str(FGS_PATH_LIMIT),
        "-p", str(FGS_PLUS_LIM),
        "-m", str(FGS_MAX_ATTEMPTS),
        "-t", str(THREADS),
        "--save", "--verify",
    ]
    t0 = time.time()
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(REPO_DIR)
    )
    elapsed = time.time() - t0

    m_rank = re.search(r"Best rank:\s+(\d+)", result.stdout)
    if not m_rank:
        print(f"\n  WARNING: FGS {bco_id:04d} failed")
        if result.stdout:
            print(result.stdout[:300])
        return None

    return int(m_rank.group(1)), elapsed


def run_waring(tensor_id, n):
    """Run Waring (TODD) with CPD initialization. Returns (waring_rank, elapsed) or None."""
    bco_id = tensor_id + 1000
    tensor_path = TENSORS_DIR / f"{bco_id:04d}.npy"
    if not tensor_path.exists():
        return None

    binary = waring_binary(n)
    cmd = [
        str(BIN_DIR / binary), str(bco_id),
        "--cpd",
        "-n", str(WARING_NUM),
        "-b", str(WARING_BEAM),
        "-t", str(THREADS),
        "--save", "--verify",
    ]
    t0 = time.time()
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(REPO_DIR)
    )
    elapsed = time.time() - t0

    # Parse: "Result  12.3 +- 1.2 (min 11, max 14, nnz 42)"
    m = re.search(r"\(min (\d+),", result.stdout)
    if not m:
        print(f"\n  WARNING: Waring {bco_id:04d} failed")
        if result.stdout:
            print(result.stdout[:300])
        return None

    return int(m.group(1)), elapsed


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------


def fmt(val, width=5):
    """Format a value for table display."""
    if val is None:
        return "--".rjust(width)
    return str(val).rjust(width)


HDR1 = "                          Toffoli (FGS)   T-count (Waring)       Paper"
HDR2 = "  #  Circuit              ATQ    VV       ATQ    VV        CP   War   t,s"
SEP  = "---- -------------------  ----- -----    ----- -----     ----- -----  -----"


def print_header():
    print(HDR1)
    print(HDR2)
    print(SEP)


def print_row(suffix, name, fgs_atq, fgs_vv, war_atq, war_vv,
              paper_cp, paper_war, elapsed):
    t_str = fmt(round(elapsed), 5)
    print(
        f"  {suffix:02d} {name:<20s}"
        f" {fmt(fgs_atq, 5)} {fmt(fgs_vv, 5)}"
        f"   {fmt(war_atq, 5)} {fmt(war_vv, 5)}"
        f"    {fmt(paper_cp, 5)} {fmt(paper_war, 5)}"
        f"  {t_str}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def process_circuits(circuits):
    """Run FGS + Waring for a list of circuits and print results."""
    print_header()
    total_time = 0.0

    for i, (suffix, name, paper_cp, paper_war) in enumerate(circuits):
        print(
            f"\r  Processing {i + 1}/{len(circuits)}: {name:<20s}",
            end="", flush=True, file=sys.stderr,
        )

        fgs_atq = fgs_vv = war_atq = war_vv = None
        elapsed = 0.0

        for prefix, label in [(100, "atq"), (800, "vv")]:
            tid = prefix + suffix
            tensor_path = TENSORS_DIR / f"{tid:04d}.npy"
            if not tensor_path.exists():
                continue

            # Re-run BCO for large circuits
            if suffix in LARGE_CIRCUITS:
                bco_res = run_bco(tid)
                if bco_res is not None:
                    n, _, t = bco_res
                    elapsed += t
                else:
                    continue
            else:
                # Use existing BCO tensor
                n = get_tensor_n(tid + 1000)
                if n is None:
                    n = get_tensor_n(tid)
                    if n is None:
                        continue

            # FGS
            fgs_res = run_fgs(tid, n)
            if fgs_res is not None:
                cp_rank, t = fgs_res
                elapsed += t
                if label == "atq":
                    fgs_atq = cp_rank
                else:
                    fgs_vv = cp_rank

                # Waring (TODD) initialized from FGS decompositions
                war_res = run_waring(tid, n)
                if war_res is not None:
                    war_rank, t = war_res
                    elapsed += t
                    if label == "atq":
                        war_atq = war_rank
                    else:
                        war_vv = war_rank

        total_time += elapsed
        # Clear progress line
        print("\r" + " " * 60 + "\r", end="", file=sys.stderr)
        print_row(suffix, name, fgs_atq, fgs_vv, war_atq, war_vv,
                  paper_cp, paper_war, elapsed)

    return total_time


def main():
    if RECOMPILE:
        compile_binaries()

    print(f"=== Table 1: FGS + Waring (pool={FGS_POOL_SIZE}) ===\n")
    t = process_circuits(CIRCUITS_MAIN)
    print(SEP)
    print(f"  Total time: {t:.0f}s\n")

    if RUN_ADDITIONAL:
        print(f"=== Additional Benchmarks (FGS + Waring) ===\n")
        t = process_circuits(CIRCUITS_ADDITIONAL)
        print(SEP)
        print(f"  Total time: {t:.0f}s\n")


if __name__ == "__main__":
    main()
