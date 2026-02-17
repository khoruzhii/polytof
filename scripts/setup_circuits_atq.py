"""
Import AlphaTensor-Quantum benchmark tensors.

Clones the alphatensor_quantum repository (if needed) to data/external/,
reconstructs the signature tensor S_ijk = sum_t f_ti f_tj f_tk (mod 2) from
each decomposition, and saves sparse tensors to data/tensors/{id:04d}.npy.

If a tensor already exists, it is verified against the reconstruction.

Usage:
    python setup_circuits_atq.py

Prerequisites:
    git (for cloning the ATQ repository)
    numpy
"""

import subprocess
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent

EXTERNAL_DIR = REPO_DIR / "data" / "external"
ATQ_REPO = EXTERNAL_DIR / "alphatensor_quantum"
ATQ_REPO_URL = "https://github.com/google-deepmind/alphatensor_quantum"
ATQ_DECOMP_DIR = ATQ_REPO / "decompositions"

TENSORS_DIR = REPO_DIR / "data" / "tensors"

# ---------------------------------------------------------------------------
# Circuit ID assignments
# ---------------------------------------------------------------------------

# fmt: off
SAVE_INFO = [
    (101, "barenco_toff_3",       "benchmarks_gadgets"),
    (102, "barenco_toff_4",       "benchmarks_gadgets"),
    (103, "barenco_toff_5",       "benchmarks_gadgets"),
    (104, "barenco_toff_10",      "benchmarks_gadgets"),
    (105, "csla_mux_3",           "benchmarks_gadgets"),
    (106, "csum_mux_9",           "benchmarks_gadgets"),
    (108, "hamming_15_low",       "benchmarks_gadgets"),
    (111, "hwb_6",                "benchmarks_gadgets"),
    (112, "mod_5_4",              "benchmarks_gadgets"),
    (114, "mod_mult_55",          "benchmarks_gadgets"),
    (115, "mod_red_21",           "benchmarks_gadgets"),
    (117, "qcla_com_7",           "benchmarks_gadgets"),
    (119, "rc_adder_6",           "benchmarks_gadgets"),
    (120, "nc_toff_3",            "benchmarks_gadgets"),
    (121, "nc_toff_4",            "benchmarks_gadgets"),
    (122, "nc_toff_5",            "benchmarks_gadgets"),
    (123, "nc_toff_10",           "benchmarks_gadgets"),
    (124, "vbe_adder_3",          "benchmarks_gadgets"),
    (125, "gf_2pow2_mult_comp2",  "multiplication_finite_fields_gadgets"),
    (126, "gf_2pow3_mult_comp2",  "multiplication_finite_fields_gadgets"),
    (127, "gf_2pow4_mult_comp2",  "multiplication_finite_fields_gadgets"),
    (128, "gf_2pow5_mult_comp2",  "multiplication_finite_fields_gadgets"),
    (129, "gf_2pow6_mult_comp2",  "multiplication_finite_fields_gadgets"),
    (130, "gf_2pow7_mult_comp2",  "multiplication_finite_fields_gadgets"),
    (131, "gf_2pow8_mult_comp2",  "multiplication_finite_fields_gadgets"),
    (132, "gf_2pow9_mult_comp2",  "multiplication_finite_fields_gadgets"),
    (133, "gf_2pow10_mult_comp2", "multiplication_finite_fields_gadgets"),
    (134, "cuccaro_adder_n3",     "binary_addition"),
    (135, "cuccaro_adder_n4",     "binary_addition"),
    (136, "cuccaro_adder_n5",     "binary_addition"),
    (137, "cuccaro_adder_n6",     "binary_addition"),
    (138, "cuccaro_adder_n7",     "binary_addition"),
    (139, "cuccaro_adder_n8",     "binary_addition"),
    (140, "cuccaro_adder_n9",     "binary_addition"),
    (141, "cuccaro_adder_n10",    "binary_addition"),
    (142, "basis_change_p4_o3",   "quantum_chemistry"),
    (143, "basis_change_p4_o4",   "quantum_chemistry"),
    (144, "basis_change_p4_o5",   "quantum_chemistry"),
    (145, "basis_change_p5_o3",   "quantum_chemistry"),
    (146, "basis_change_p5_o4",   "quantum_chemistry"),
    (147, "basis_change_p6_o3",   "quantum_chemistry"),
    (148, "basis_change_p7_o3",   "quantum_chemistry"),
    (149, "hamming_weight_n4",    "hamming_weight_phase_gradient"),
    (150, "hamming_weight_n5",    "hamming_weight_phase_gradient"),
    (151, "hamming_weight_n6",    "hamming_weight_phase_gradient"),
    (152, "hamming_weight_n7",    "hamming_weight_phase_gradient"),
    (153, "hamming_weight_n8",    "hamming_weight_phase_gradient"),
    (154, "hamming_weight_n9",    "hamming_weight_phase_gradient"),
    (155, "hamming_weight_n10",   "hamming_weight_phase_gradient"),
    (156, "hamming_weight_n11",   "hamming_weight_phase_gradient"),
    (157, "hamming_weight_n12",   "hamming_weight_phase_gradient"),
    (158, "hamming_weight_n13",   "hamming_weight_phase_gradient"),
    (159, "hamming_weight_n14",   "hamming_weight_phase_gradient"),
    (160, "hamming_weight_n15",   "hamming_weight_phase_gradient"),
    (161, "hamming_weight_n16",   "hamming_weight_phase_gradient"),
    (162, "hamming_weight_n17",   "hamming_weight_phase_gradient"),
    (163, "hamming_weight_n18",   "hamming_weight_phase_gradient"),
    (164, "hamming_weight_n19",   "hamming_weight_phase_gradient"),
    (165, "hamming_weight_n20",   "hamming_weight_phase_gradient"),
    (166, "unary_iteration_n3",   "unary_iteration_gadgets"),
    (167, "unary_iteration_n4",   "unary_iteration_gadgets"),
    (168, "unary_iteration_n5",   "unary_iteration_gadgets"),
]
# fmt: on

# ---------------------------------------------------------------------------
# Tensor reconstruction and sparse format
# ---------------------------------------------------------------------------


def waring_tensor_dense(factors):
    """Reconstruct Waring tensor S_ijk = sum_t f_ti f_tj f_tk (mod 2).

    The tensor is symmetric because each rank-1 term uses the same vector.
    """
    return np.einsum("qi,qj,qk->ijk", factors, factors, factors) % 2


def dense_to_sparse(T):
    """Convert dense 3D symmetric binary tensor to sparse format.

    Returns array with first row (n, n, n) and remaining rows (i, j, k) triples
    where i < j < k (strictly increasing, exploiting symmetry).
    """
    n = T.shape[0]
    mask = T.astype(bool)
    coords = np.stack(np.nonzero(mask), axis=1)

    if coords.size == 0:
        idx = coords.astype(np.int64, copy=False)
    else:
        sel = (coords[:, 1] > coords[:, 0]) & (coords[:, 2] > coords[:, 1])
        idx = coords[sel].astype(np.int64, copy=False)

    out = np.empty((idx.shape[0] + 1, 3), dtype=np.int64)
    out[0] = (n, n, n)
    if idx.shape[0] > 0:
        out[1:] = idx

    return out


# ---------------------------------------------------------------------------
# Repository management
# ---------------------------------------------------------------------------


def ensure_repo():
    """Clone ATQ repository if not present."""
    if ATQ_DECOMP_DIR.exists():
        npz_files = list(ATQ_DECOMP_DIR.glob("*.npz"))
        print(f"ATQ repo found: {ATQ_REPO}  ({len(npz_files)} .npz files)")
        return

    print(f"Cloning {ATQ_REPO_URL} ...")
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", ATQ_REPO_URL],
        cwd=str(EXTERNAL_DIR),
        check=True,
    )
    npz_files = list(ATQ_DECOMP_DIR.glob("*.npz"))
    print(f"Cloned: {len(npz_files)} .npz files")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ensure_repo()

    TENSORS_DIR.mkdir(parents=True, exist_ok=True)

    # Cache loaded .npz files (one per group)
    npz_cache = {}

    header = f"{'ID':>4}  {'Name':<28}  {'n':>3} {'T':>4} {'nnz':>5}  Tensor"
    print(header)
    print("-" * len(header))

    errors = []

    for tensor_id, name, group in SAVE_INFO:
        # Load decomposition
        if group not in npz_cache:
            npz_path = ATQ_DECOMP_DIR / f"{group}.npz"
            if not npz_path.exists():
                msg = f"missing {npz_path.name}"
                print(f"{tensor_id:04d}  {name:<28}  ERROR: {msg}")
                errors.append((tensor_id, msg))
                continue
            npz_cache[group] = np.load(npz_path)

        npz = npz_cache[group]
        if name not in npz:
            msg = f"'{name}' not in {group}.npz"
            print(f"{tensor_id:04d}  {name:<28}  ERROR: {msg}")
            errors.append((tensor_id, msg))
            continue

        # First decomposition, as int32
        factor = npz[name].astype(np.int32)[0]
        num_factors, n = factor.shape

        # Reconstruct Waring tensor: S_ijk = sum_t f_ti f_tj f_tk (mod 2)
        T_dense = waring_tensor_dense(factor)
        T_sparse = dense_to_sparse(T_dense)
        nnz = T_sparse.shape[0] - 1

        tensor_path = TENSORS_DIR / f"{tensor_id:04d}.npy"
        if tensor_path.exists():
            existing = np.load(tensor_path)
            if np.array_equal(existing, T_sparse):
                tensor_status = "match"
            else:
                tensor_status = "MISMATCH"
                errors.append((tensor_id, "tensor mismatch"))
        else:
            np.save(tensor_path, T_sparse)
            tensor_status = "new"

        print(
            f"{tensor_id:04d}  {name:<28}  "
            f"{n:>3} {num_factors:>4} {nnz:>5}  {tensor_status}"
        )

    print("-" * len(header))
    print(f"Total: {len(SAVE_INFO)} circuits")

    if errors:
        print(f"\n{len(errors)} error(s):")
        for id_, err in errors:
            print(f"  {id_:04d}: {err}")
        sys.exit(1)

    print("Done.")


if __name__ == "__main__":
    main()
