#!/usr/bin/env python3
"""
Reproduce Table 1: Toffoli-count minimization via BCO + SGE.

Runs BCO (Basis Change Optimization) and SGE (Symplectic Gaussian Elimination)
on circuit benchmarks from AlphaTensor-Quantum (ATQ, 01xx) and
Vandaele et al. (VV, 08xx) tensor series.

Usage:
    python reproduce_tab1.py
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

BEAM_WIDTH = 8        # paper: 1024
THREADS = 8
RUN_ADDITIONAL = True
RECOMPILE = False

CXX = "g++"
CXXFLAGS = ["-Ofast", "-std=c++20", "-march=native", "-s", "-pthread"]

# ---------------------------------------------------------------------------
# Circuit registry: (suffix, name, paper_cp)
#
# suffix s -> tensor IDs: ATQ = 100+s, VV = 800+s
# paper_cp: best known CP rank from data/paper/ (see docs/tensors.md)
# ---------------------------------------------------------------------------

CIRCUITS_MAIN = [
    ( 0, "Adder_8",          27),
    ( 1, "Barenco Tof_3",     2),
    ( 2, "Barenco Tof_4",     4),
    ( 3, "Barenco Tof_5",     6),
    ( 4, "Barenco Tof_10",   16),
    ( 5, "CSLA MUX_3",        8),
    ( 6, "CSUM MUX_9",       14),
    ( 7, "Grover_5",         25),
    ( 8, "Ham15 (low)",      17),
    ( 9, "Ham15 (med)",      33),
    (10, "Ham15 (high)",    155),
    (11, "HWB_6",            10),
    (12, "Mod 5_4",           1),
    (13, "Mod Adder_1024",  128),
    (14, "Mod Mult_55",       3),
    (15, "Mod Red_21",       11),
    (16, "QCLA Adder_10",   24),
    (17, "QCLA Com_7",       12),
    (18, "QCLA Mod_7",       37),
    (19, "RC Adder_6",        6),
    (20, "Tof_3",             2),
    (21, "Tof_4",             3),
    (22, "Tof_5",             4),
    (23, "Tof_10",            9),
    (24, "VBE Adder_3",       3),
]

# Additional ATQ-only benchmarks (01xx series)
CIRCUITS_ADDITIONAL = [
    # Binary Addition (Cuccaro Adder)
    (34, "Cuccaro n=3",    2),
    (35, "Cuccaro n=4",    3),
    (36, "Cuccaro n=5",    4),
    (37, "Cuccaro n=6",    5),
    (38, "Cuccaro n=7",    6),
    (39, "Cuccaro n=8",    7),
    (40, "Cuccaro n=9",    8),
    (41, "Cuccaro n=10",   9),
    # Quantum Chemistry (Basis Change)
    (42, "BasisCh p4o3",   8),
    (43, "BasisCh p4o4",  12),
    (44, "BasisCh p4o5",  16),
    (45, "BasisCh p5o3",  12),
    (46, "BasisCh p5o4",  18),
    (47, "BasisCh p6o3",  16),
    (48, "BasisCh p7o3",  20),
    # Hamming Weight / Phase Gradient
    (49, "HammWt n=4",     3),
    (50, "HammWt n=5",     3),
    (51, "HammWt n=6",     4),
    (52, "HammWt n=7",     4),
    (53, "HammWt n=8",     7),
    (54, "HammWt n=9",     7),
    (55, "HammWt n=10",    8),
    (56, "HammWt n=11",    8),
    (57, "HammWt n=12",   10),
    (58, "HammWt n=13",   10),
    (59, "HammWt n=14",   11),
    (60, "HammWt n=15",   11),
    (61, "HammWt n=16",   15),
    (62, "HammWt n=17",   15),
    (63, "HammWt n=18",   16),
    (64, "HammWt n=19",   16),
    (65, "HammWt n=20",   18),
    # Unary Iteration
    (66, "UnaryIt n=3",    7),
    (67, "UnaryIt n=4",   15),
    (68, "UnaryIt n=5",   31),
]


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------


def compile_binaries():
    """Compile BCO and SGE (topp) binaries."""
    os.makedirs(BIN_DIR, exist_ok=True)
    targets = [
        ("bco",   "bco.cpp", []),
        ("topp1", "cpd.cpp", ["-DVEC_WORDS=1"]),
        ("topp2", "cpd.cpp", ["-DVEC_WORDS=2"]),
        ("topp6", "cpd.cpp", ["-DVEC_WORDS=6"]),
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
# BCO and SGE runners
# ---------------------------------------------------------------------------


def topp_binary(n):
    """Select topp binary based on tensor dimension n."""
    if n <= 64:
        return "topp1"
    elif n <= 128:
        return "topp2"
    else:
        return "topp6"


def run_bco(tensor_id, beam_width):
    """Run BCO on a tensor. Returns (n, nnz_after, elapsed) or None."""
    tensor_path = TENSORS_DIR / f"{tensor_id:04d}.npy"
    if not tensor_path.exists():
        return None

    cmd = [
        str(BIN_DIR / "bco"), str(tensor_id),
        "-b", str(beam_width), "-t", str(THREADS), "--save", "--verify",
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


def run_sge(tensor_id, n):
    """Run SGE on BCO-optimized tensor (id+1000). Returns (cp_rank, elapsed) or None."""
    bco_id = tensor_id + 1000
    tensor_path = TENSORS_DIR / f"{bco_id:04d}.npy"
    if not tensor_path.exists():
        return None

    binary = topp_binary(n)
    cmd = [
        str(BIN_DIR / binary), str(bco_id),
        "--sge", "-b", str(BEAM_WIDTH), "-t", str(THREADS),
        "--save", "--verify",
    ]
    t0 = time.time()
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(REPO_DIR)
    )
    elapsed = time.time() - t0

    m_rank = re.search(r"Best rank:\s+(\d+)", result.stdout)
    if not m_rank:
        print(f"\n  WARNING: SGE {bco_id:04d} failed")
        if result.stdout:
            print(result.stdout[:200])
        return None

    return int(m_rank.group(1)), elapsed


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------


def fmt(val, width=5):
    """Format a value for table display."""
    if val is None:
        return "--".rjust(width)
    return str(val).rjust(width)


HDR1 = "                          BCO         BCO+SGE"
HDR2 = "  #  Circuit              ATQ    VV   ATQ    VV  Paper  t,s"
SEP  = "---- -------------------  ---- -----  ---- -----  -----  ----"


def print_header():
    print(HDR1)
    print(HDR2)
    print(SEP)


def print_row(suffix, name, bco_atq, bco_vv, sge_atq, sge_vv, paper_cp, elapsed):
    t_str = fmt(round(elapsed), 4) if elapsed > 0 else fmt(0, 4)
    print(
        f"  {suffix:02d} {name:<20s}"
        f" {fmt(bco_atq, 4)} {fmt(bco_vv, 5)}"
        f"  {fmt(sge_atq, 4)} {fmt(sge_vv, 5)}"
        f"  {fmt(paper_cp, 5)}"
        f"  {t_str}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def process_circuits(circuits, beam_width):
    """Run BCO + SGE for a list of circuits and print results."""
    print_header()
    total_time = 0.0

    for i, (suffix, name, paper_cp) in enumerate(circuits):
        print(
            f"\r  Processing {i + 1}/{len(circuits)}: {name:<20s}",
            end="", flush=True, file=sys.stderr,
        )

        bco_atq = bco_vv = sge_atq = sge_vv = None
        elapsed = 0.0

        # ATQ (01xx)
        atq_id = 100 + suffix
        bco_res = run_bco(atq_id, beam_width)
        if bco_res is not None:
            n_atq, bco_atq, t = bco_res
            elapsed += t
            sge_res = run_sge(atq_id, n_atq)
            if sge_res is not None:
                sge_atq, t = sge_res
                elapsed += t

        # VV (08xx)
        vv_id = 800 + suffix
        bco_res = run_bco(vv_id, beam_width)
        if bco_res is not None:
            n_vv, bco_vv, t = bco_res
            elapsed += t
            sge_res = run_sge(vv_id, n_vv)
            if sge_res is not None:
                sge_vv, t = sge_res
                elapsed += t

        total_time += elapsed
        # Clear progress line
        print("\r" + " " * 60 + "\r", end="", file=sys.stderr)
        print_row(suffix, name, bco_atq, bco_vv, sge_atq, sge_vv, paper_cp, elapsed)

    return total_time


def main():
    if RECOMPILE:
        compile_binaries()

    print(f"=== Table 1: Circuit Benchmarks (beam width {BEAM_WIDTH}) ===\n")
    t = process_circuits(CIRCUITS_MAIN, BEAM_WIDTH)
    print(SEP)
    print(f"  Total time: {t:.0f}s\n")

    if RUN_ADDITIONAL:
        print(f"=== Additional Benchmarks (beam width {BEAM_WIDTH}) ===\n")
        t = process_circuits(CIRCUITS_ADDITIONAL, BEAM_WIDTH)
        print(SEP)
        print(f"  Total time: {t:.0f}s\n")


if __name__ == "__main__":
    main()
