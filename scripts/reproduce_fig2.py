"""
Reproduce Figure 2: Performance on planted CP benchmarks.

Row 1 (4 heatmaps, x=n, y=r, color=log recovered rank):
    BCO | SGE | FGS | BCO+SGE+FGS

Row 2 (3 scatter plots):
    BCO (beam) vs BCO (greedy) | SGE (beam) vs SGE (greedy) | BCO+SGE+FGS vs BCO+SGE

CPD runs are isolated via -DRUN to prevent pool contamination:
    RUN=0: SGE greedy on tid
    RUN=1: SGE beam on tid
    RUN=2: FGS on tid
    RUN=3: BCO+SGE beam on tid+1000
    RUN=4: BCO+SGE+FGS on tid+1000 (results used by reproduce_fig3.py)

BCO greedy saves to tid+2000 (-o), BCO beam saves to tid+1000 (-o).

Outputs:
    data/fig2.tsv   — tab-separated results
    data/fig2.png   — figure

Usage:
    python reproduce_fig2.py
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
BCO_BEAM = 8                # paper: 1024
SGE_BEAM = 8                # paper: 1024  (-b)
FGS_POOL_SIZE = 40          # paper: 1000  (-s)
FGS_PATH_LIMIT = 1000000    # paper: 1000000  (-f)
FGS_PLUS_LIM = 50000        # paper: 50000   (-p)
FGS_MAX_ATTEMPTS = 200      # paper: 100000  (-m)
CXX = "g++"
CXXFLAGS = ["-Ofast", "-std=c++20", "-march=native", "-s", "-pthread"]

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
BIN_DIR = REPO_DIR / "bin"
SRC_DIR = REPO_DIR / "src"
THIRD_PARTY_DIR = REPO_DIR / "third_party"
TENSORS_DIR = REPO_DIR / "data" / "tensors"

# Benchmark grid
N_VALUES = [6, 8, 10, 12, 14, 16, 18, 20]
R_VALUES = list(range(1, 17))

# RUN assignments for CPD isolation
RUN_SGE_GREEDY = 0
RUN_SGE_BEAM = 1
RUN_FGS = 2
RUN_BCO_SGE = 3
RUN_BCO_SGE_FGS = 4


def tensor_id(i, j, k):
    """Tensor ID for n=N_VALUES[i], r=R_VALUES[j], repetition k."""
    return 2000 + i * 64 + j * 4 + k


# Compilation


def compile_binaries():
    os.makedirs(BIN_DIR, exist_ok=True)
    targets = [
        ("bco", "bco.cpp", []),
        ("topp1_r0", "cpd.cpp", ["-DVEC_WORDS=1", f"-DRUN={RUN_SGE_GREEDY}"]),
        ("topp1_r1", "cpd.cpp", ["-DVEC_WORDS=1", f"-DRUN={RUN_SGE_BEAM}"]),
        ("topp1_r2", "cpd.cpp", ["-DVEC_WORDS=1", f"-DRUN={RUN_FGS}"]),
        ("topp1_r3", "cpd.cpp", ["-DVEC_WORDS=1", f"-DRUN={RUN_BCO_SGE}"]),
        ("topp1_r4", "cpd.cpp", ["-DVEC_WORDS=1", f"-DRUN={RUN_BCO_SGE_FGS}"]),
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


# Runners


def run_bco(tid, beam, output_id):
    """Run BCO. Returns (nnz_after, elapsed) or None."""
    tensor_path = TENSORS_DIR / f"{tid:04d}.npy"
    if not tensor_path.exists():
        return None

    cmd = [
        str(BIN_DIR / "bco"), str(tid),
        "-o", str(output_id),
        "-b", str(beam), "-t", str(THREADS), "--save", "--verify",
    ]
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_DIR))
    elapsed = time.time() - t0

    m = re.search(r"Final:\s+(\d+)", result.stdout)
    if not m:
        return None
    return int(m.group(1)), elapsed


def run_cpd(binary, tid, sge_beam=0, do_fgs=False):
    """Run CPD with given binary. Returns (cp_rank, elapsed) or None."""
    tensor_path = TENSORS_DIR / f"{tid:04d}.npy"
    if not tensor_path.exists():
        return None
    if sge_beam <= 0 and not do_fgs:
        return None

    cmd = [str(BIN_DIR / binary), str(tid)]
    if sge_beam > 0:
        cmd += ["--sge", "-b", str(sge_beam)]
    if do_fgs:
        cmd += [
            "--fgs", "--plus",
            "-s", str(FGS_POOL_SIZE),
            "-f", str(FGS_PATH_LIMIT),
            "-p", str(FGS_PLUS_LIM),
            "-m", str(FGS_MAX_ATTEMPTS),
        ]
    cmd += ["-t", str(THREADS), "--save", "--verify"]

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_DIR))
    elapsed = time.time() - t0

    m = re.search(r"Best rank:\s+(\d+)", result.stdout)
    if not m:
        return None
    return int(m.group(1)), elapsed


# Main


def main():
    if RECOMPILE:
        compile_binaries()

    total = len(N_VALUES) * len(R_VALUES) * NUM_REPS
    missing = sum(
        1 for i in range(len(N_VALUES))
        for j in range(len(R_VALUES))
        for k in range(NUM_REPS)
        if not (TENSORS_DIR / f"{tensor_id(i, j, k):04d}.npy").exists()
    )
    if missing > 0:
        print(f"WARNING: {missing}/{total} tensors missing. "
              f"Run make_rnd_bench.py first.\n")

    nn = len(N_VALUES)
    nr = len(R_VALUES)

    # Row 1 heatmap data
    bco_nnz = np.full((nn, nr), np.nan)         # BCO beam nnz
    sge_rank = np.full((nn, nr), np.nan)         # SGE beam on tid
    fgs_rank = np.full((nn, nr), np.nan)         # FGS on tid
    full_rank = np.full((nn, nr), np.nan)        # BCO+SGE+FGS on tid+1000

    # Row 2 comparison data
    bco_greedy_nnz = np.full((nn, nr), np.nan)   # BCO greedy nnz
    sge_greedy_rank = np.full((nn, nr), np.nan)  # SGE greedy on tid
    bco_sge_rank = np.full((nn, nr), np.nan)     # BCO+SGE beam on tid+1000

    print(f"=== Figure 2: Planted CP Benchmark "
          f"(BCO beam={BCO_BEAM}, SGE beam={SGE_BEAM}, "
          f"FGS pool={FGS_POOL_SIZE}, reps={NUM_REPS}) ===\n")

    t_total = time.time()
    count = 0

    for i, n in enumerate(N_VALUES):
        for j, r in enumerate(R_VALUES):
            acc_bco = []
            acc_bco_g = []
            acc_sge = []
            acc_sge_g = []
            acc_fgs = []
            acc_bco_sge = []
            acc_full = []

            for k_rep in range(NUM_REPS):
                tid = tensor_id(i, j, k_rep)
                bco_tid = tid + 1000
                count += 1
                print(f"\r  [{count}/{total}] n={n:2d} r={r:2d}",
                      end="", flush=True)

                # 1. BCO greedy → tid+2000
                res = run_bco(tid, beam=1, output_id=tid + 2000)
                if res is None:
                    continue
                acc_bco_g.append(res[0])

                # 2. BCO beam → tid+1000
                res = run_bco(tid, beam=BCO_BEAM, output_id=bco_tid)
                if res is None:
                    continue
                acc_bco.append(res[0])

                # 3. SGE greedy on tid (r0)
                res = run_cpd("topp1_r0", tid, sge_beam=1)
                if res is not None:
                    acc_sge_g.append(res[0])

                # 4. SGE beam on tid (r1)
                res = run_cpd("topp1_r1", tid, sge_beam=SGE_BEAM)
                if res is not None:
                    acc_sge.append(res[0])

                # 5. FGS on tid (r2)
                res = run_cpd("topp1_r2", tid, do_fgs=True)
                if res is not None:
                    acc_fgs.append(res[0])

                # 6. BCO+SGE beam on tid+1000 (r3)
                res = run_cpd("topp1_r3", bco_tid, sge_beam=SGE_BEAM)
                if res is not None:
                    acc_bco_sge.append(res[0])

                # 7. BCO+SGE+FGS on tid+1000 (r4, for fig3)
                res = run_cpd("topp1_r4", bco_tid,
                              sge_beam=SGE_BEAM, do_fgs=True)
                if res is not None:
                    acc_full.append(res[0])

            for acc, arr in [(acc_bco, bco_nnz), (acc_sge, sge_rank),
                             (acc_fgs, fgs_rank), (acc_full, full_rank),
                             (acc_bco_g, bco_greedy_nnz),
                             (acc_sge_g, sge_greedy_rank),
                             (acc_bco_sge, bco_sge_rank)]:
                if acc:
                    arr[i, j] = np.mean(acc)

    elapsed = time.time() - t_total
    print(f"\n\n  Total time: {elapsed:.0f}s\n")

    # ------------------------------------------------------------------
    # Save TSV
    # ------------------------------------------------------------------

    tsv_path = REPO_DIR / "data" / "fig2.tsv"
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = [bco_nnz, sge_rank, fgs_rank, full_rank,
              bco_greedy_nnz, sge_greedy_rank, bco_sge_rank]
    col_names = ["bco", "sge", "fgs", "full",
                 "bco_greedy", "sge_greedy", "bco_sge"]
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

    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.3)

    # --- Row 1: 4 heatmaps (x=n, y=r, color=log rank) ---
    heatmaps = [
        ("BCO", bco_nnz),
        ("SGE", sge_rank),
        ("FGS", fgs_rank),
        ("BCO+SGE+FGS", full_rank),
    ]

    all_vals = np.concatenate([a.ravel() for _, a in heatmaps])
    all_vals = all_vals[~np.isnan(all_vals)]
    vmin = max(1, np.min(all_vals)) if len(all_vals) > 0 else 1
    vmax = np.max(all_vals) if len(all_vals) > 0 else 100

    ax_row1 = []
    for col, (title, data) in enumerate(heatmaps):
        ax = fig.add_subplot(gs[0, col])
        ax_row1.append(ax)
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

    fig.colorbar(im, ax=ax_row1, label="Recovered rank", shrink=0.8, pad=0.02)

    # --- Row 2: 3 scatter comparisons ---
    comparisons = [
        ("BCO (beam) vs BCO (greedy)", bco_nnz, bco_greedy_nnz),
        ("SGE (beam) vs SGE (greedy)", sge_rank, sge_greedy_rank),
        ("BCO+SGE+FGS vs BCO+SGE", full_rank, bco_sge_rank),
    ]

    for col, (title, y_arr, x_arr) in enumerate(comparisons):
        ax = fig.add_subplot(gs[1, col])
        x_vals = x_arr.ravel()
        y_vals = y_arr.ravel()
        mask = ~(np.isnan(x_vals) | np.isnan(y_vals))
        xp = x_vals[mask]
        yp = y_vals[mask]

        if len(xp) > 0:
            ax.scatter(xp, yp, s=12, alpha=0.6, edgecolors="none")
            lo = min(xp.min(), yp.min()) * 0.8
            hi = max(xp.max(), yp.max()) * 1.2
            lo = max(lo, 0.5)
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")

    png_path = REPO_DIR / "data" / "fig2.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"  Saved {png_path}")


if __name__ == "__main__":
    main()
