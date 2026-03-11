#!/usr/bin/env python3
"""
Reproduce Figure 3: Waring decomposition on planted CP benchmarks.

Three heatmaps (x=n, y=r, color=log Waring rank):
    Greedy | Beam | CP init

Requires reproduce_fig2.py to have been run first (BCO tensors + CPD files).

Waring runs are isolated via -DRUN to prevent file contamination:
    RUN=0: greedy (beam=1, trivial init)
    RUN=1: beam (beam>1, trivial init)
    RUN=4: CP init (beam=1, --cpd) — reuses fig2's BCO+SGE+FGS RUN

Outputs:
    data/fig3.tsv   — tab-separated results
    data/fig3.png   — figure

Usage:
    python reproduce_fig3.py
"""

import os
import re
import subprocess
import sys
import time
from pathlib import Path
import numpy as np


# Configuration
THREADS = 8
RECOMPILE = True
NUM_REPS = 1                # paper: 4   (1..4, repetitions per (n, r) pair)
WARING_BEAM = 64            # paper: 1024
WARING_NUM = 64             # paper: 1000  (-n, CPD schemes to try)
CXX = "g++"
CXXFLAGS = ["-Ofast", "-std=c++20", "-march=native", "-s", "-pthread"]

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
BIN_DIR = REPO_DIR / "bin"
SRC_DIR = REPO_DIR / "src"
THIRD_PARTY_DIR = REPO_DIR / "third_party"
TENSORS_DIR = REPO_DIR / "data" / "tensors"

# Benchmark grid (same as fig2)
N_VALUES = [6, 8, 10, 12, 14, 16, 18, 20]
R_VALUES = list(range(1, 17))

# RUN assignments for waring isolation
RUN_GREEDY = 0
RUN_BEAM = 1
RUN_CPD_INIT = 4            # same as fig2 BCO+SGE+FGS


def tensor_id(i, j, k):
    """Tensor ID for n=N_VALUES[i], r=R_VALUES[j], repetition k."""
    return 2000 + i * 64 + j * 4 + k


# Compilation


def compile_binaries():
    os.makedirs(BIN_DIR, exist_ok=True)
    targets = [
        ("waring1_r0", "waring.cpp", ["-DVEC_WORDS=1", f"-DRUN={RUN_GREEDY}"]),
        ("waring1_r1", "waring.cpp", ["-DVEC_WORDS=1", f"-DRUN={RUN_BEAM}"]),
        ("waring1_r4", "waring.cpp", ["-DVEC_WORDS=1", f"-DRUN={RUN_CPD_INIT}"]),
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


# Runner


def run_waring(binary, tensor_id, beam=1, use_cpd=False):
    """Run Waring on a tensor. Returns (waring_rank, num_schemes, elapsed) or None."""
    tensor_path = TENSORS_DIR / f"{tensor_id:04d}.npy"
    if not tensor_path.exists():
        return None

    cmd = [
        str(BIN_DIR / binary), str(tensor_id),
        "-b", str(beam),
        "-t", str(THREADS),
        "--save", "--verify",
    ]
    if use_cpd:
        cmd += ["--cpd", "-n", str(WARING_NUM)]

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_DIR))
    elapsed = time.time() - t0

    m = re.search(r"\(min (\d+),", result.stdout)
    if not m:
        return None

    num_schemes = 0
    m2 = re.search(r"Loaded (\d+) CPD schemes", result.stdout)
    if m2:
        loaded = int(m2.group(1))
        num_schemes = min(loaded, WARING_NUM)

    return int(m.group(1)), num_schemes, elapsed


# Main


def main():
    if RECOMPILE:
        compile_binaries()

    total = len(N_VALUES) * len(R_VALUES) * NUM_REPS
    missing = sum(
        1 for i in range(len(N_VALUES))
        for j in range(len(R_VALUES))
        for k in range(NUM_REPS)
        if not (TENSORS_DIR / f"{tensor_id(i, j, k) + 1000:04d}.npy").exists()
    )
    if missing > 0:
        print(f"WARNING: {missing}/{total} BCO tensors missing. "
              f"Run reproduce_fig2.py first.\n")

    nn = len(N_VALUES)
    nr = len(R_VALUES)

    greedy_rank = np.full((nn, nr), np.nan)
    beam_rank = np.full((nn, nr), np.nan)
    cpd_rank = np.full((nn, nr), np.nan)

    print(f"=== Figure 3: Waring on Planted CP Benchmark "
          f"(beam={WARING_BEAM}, num={WARING_NUM}, reps={NUM_REPS}) ===\n")

    t_total = time.time()
    count = 0
    all_schemes = []

    for i, n in enumerate(N_VALUES):
        for j, r in enumerate(R_VALUES):
            acc_greedy = []
            acc_beam = []
            acc_cpd = []
            acc_schemes = []

            for k_rep in range(NUM_REPS):
                bco_tid = tensor_id(i, j, k_rep) + 1000
                count += 1
                print(f"\r  [{count}/{total}] n={n:2d} r={r:2d}",
                      end="", flush=True)

                # Greedy (beam=1, trivial init)
                res = run_waring("waring1_r0", bco_tid, beam=1)
                if res is not None:
                    acc_greedy.append(res[0])

                # Beam (beam>1, trivial init)
                res = run_waring("waring1_r1", bco_tid, beam=WARING_BEAM)
                if res is not None:
                    acc_beam.append(res[0])

                # CP init (beam=1, --cpd)
                res = run_waring("waring1_r4", bco_tid, beam=1, use_cpd=True)
                if res is not None:
                    acc_cpd.append(res[0])
                    acc_schemes.append(res[1])

            for acc, arr in [(acc_greedy, greedy_rank),
                             (acc_beam, beam_rank),
                             (acc_cpd, cpd_rank)]:
                if acc:
                    arr[i, j] = np.mean(acc)
            all_schemes.extend(acc_schemes)

    elapsed = time.time() - t_total
    print(f"\n\n  Total time: {elapsed:.0f}s")
    if all_schemes:
        print(f"  CP init: {np.mean(all_schemes):.1f} schemes/tensor avg "
              f"(min {min(all_schemes)}, max {max(all_schemes)}, num={WARING_NUM})")
    print()

    # ------------------------------------------------------------------
    # Save TSV
    # ------------------------------------------------------------------

    tsv_path = REPO_DIR / "data" / "fig3.tsv"
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = [greedy_rank, beam_rank, cpd_rank]
    col_names = ["greedy", "beam", "cpd_init"]
    with open(tsv_path, "w") as f:
        f.write("n\tr\t" + "\t".join(col_names) + "\n")
        for i, n in enumerate(N_VALUES):
            for j, r in enumerate(R_VALUES):
                vals = [str(n), str(r)]
                for arr in arrays:
                    v = arr[i, j]
                    vals.append(f"{v:.1f}" if not np.isnan(v) else "")
                f.write("\t".join(vals) + "\n")
    print(f"  Saved {tsv_path}")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
    except ImportError:
        print("  matplotlib not available, skipping plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    heatmaps = [
        ("Greedy", greedy_rank),
        ("Beam", beam_rank),
        ("CP init", cpd_rank),
    ]

    all_vals = np.concatenate([a.ravel() for _, a in heatmaps])
    all_vals = all_vals[~np.isnan(all_vals)]
    vmin = max(1, np.min(all_vals)) if len(all_vals) > 0 else 1
    vmax = np.max(all_vals) if len(all_vals) > 0 else 100

    for col, (title, data) in enumerate(heatmaps):
        ax = axes[col]
        masked = np.where(np.isnan(data), vmin, data)
        masked = np.maximum(masked, vmin)
        im = ax.imshow(
            masked.T, aspect="auto", origin="lower",
            norm=LogNorm(vmin=vmin, vmax=vmax),
            cmap="berlin",
            extent=[N_VALUES[0] - 1, N_VALUES[-1] + 1,
                    R_VALUES[0] - 0.5, R_VALUES[-1] + 0.5],
        )
        ax.set_title(title)
        ax.set_xlabel("Dimension n")
        ax.set_xticks(N_VALUES)
        ax.set_yticks(R_VALUES[::2])
        if col == 0:
            ax.set_ylabel("Planted rank r")

    fig.colorbar(im, ax=list(axes), label="log Waring rank", shrink=0.8, pad=0.02)

    png_path = REPO_DIR / "data" / "fig3.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"  Saved {png_path}")


if __name__ == "__main__":
    main()
