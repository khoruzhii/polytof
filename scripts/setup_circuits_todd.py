"""
Import benchmark .qc circuits and compile them to phase polynomials and tensors.

Clones the quantum-circuit-optimization repository (if needed), copies .qc files
to data/circuits/raw/ with numeric IDs, and optionally runs bin/compile --idx to
extract phase polynomials and signature tensors.

If a tensor already exists in data/tensors/, the newly compiled tensor is compared
against it and a mismatch is reported as an error.

Usage:
    python setup_circuits.py

Prerequisites:
    git (for cloning the circuit repository)
    bin/compile (optional; set RUN_COMPILE = False to skip)
"""

import filecmp
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Set to False to only import .qc files without compiling
RUN_COMPILE = True

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent

# External circuit repository (cloned on demand)
EXTERNAL_DIR = REPO_DIR / "data" / "external"
CIRCUITS_REPO = EXTERNAL_DIR / "quantum-circuit-optimization"
CIRCUITS_REPO_URL = "https://github.com/VivienVandaele/quantum-circuit-optimization"

SRC_DIR = CIRCUITS_REPO / "circuits" / "inputs"
DST_DIR = REPO_DIR / "data" / "circuits" / "raw"
TENSORS_DIR = REPO_DIR / "data" / "tensors"

# ---------------------------------------------------------------------------
# Compile binary (platform-dependent)
# ---------------------------------------------------------------------------

def get_compile_bin() -> Path:
    if sys.platform == "win32":
        return REPO_DIR / "bin" / "compile.exe"
    return REPO_DIR / "bin" / "compile"

# ---------------------------------------------------------------------------
# Circuit ID assignments
# ---------------------------------------------------------------------------

# Table 1 benchmarks (800-824)
FIXED_800 = {
    "adder_8.qc": 800,
    "barenco_tof_3.qc": 801,
    "barenco_tof_4.qc": 802,
    "barenco_tof_5.qc": 803,
    "barenco_tof_10.qc": 804,
    "csla_mux_3.qc": 805,
    "csum_mux_9.qc": 806,
    "grover_5.qc": 807,
    "ham15-low.qc": 808,
    "ham15-med.qc": 809,
    "ham15-high.qc": 810,
    "hwb6.qc": 811,
    "mod5_4.qc": 812,
    "mod_adder_1024.qc": 813,
    "mod_mult_55.qc": 814,
    "mod_red_21.qc": 815,
    "qcla_adder_10.qc": 816,
    "qcla_com_7.qc": 817,
    "qcla_mod_7.qc": 818,
    "rc_adder_6.qc": 819,
    "tof_3.qc": 820,
    "tof_4.qc": 821,
    "tof_5.qc": 822,
    "tof_10.qc": 823,
    "vbe_adder_3.qc": 824,
}

# GF(2^k) multiplication circuits (827-833 for k=4..10)
FIXED_GF = {f"gf2^{k}_mult.qc": 827 + (k - 4) for k in range(4, 11)}

# Other
FIXED_OTHER = {"qft_4.qc": 876}

# All fixed assignments (IDs <= 876)
FIXED = {**FIXED_800, **FIXED_GF, **FIXED_OTHER}

# ---------------------------------------------------------------------------
# Compile output parsing
# ---------------------------------------------------------------------------

@dataclass
class CompileResult:
    qubits: int = 0
    ancillas: int = 0
    t_before: int = 0
    t_after: int = 0
    h_count: int = 0
    nnz: int = 0
    f1: int = 0
    f2: int = 0
    f3: int = 0
    has_tensor: bool = False
    error: str = ""


def parse_compile_output(output: str) -> CompileResult:
    r = CompileResult()

    m = re.search(r"input:\s+(\d+)\s+qubits,.*T=(\d+),\s*H=(\d+)", output)
    if m:
        r.qubits = int(m.group(1))
        r.t_before = int(m.group(2))
        r.h_count = int(m.group(3))

    m = re.search(r"output:\s+(\d+)\s+qubits\s+\((\d+)\s+ancillas\),\s*P=\[(\d+)x", output)
    if m:
        r.qubits = int(m.group(1))
        r.ancillas = int(m.group(2))
        r.t_after = int(m.group(3))

    m = re.search(
        r"tensor:\s+(\d+)\s+nnz\s+\(\|f1\|=(\d+),\s*\|f2\|=(\d+),\s*\|f3\|=(\d+)\)",
        output,
    )
    if m:
        r.has_tensor = True
        r.nnz = int(m.group(1))
        r.f1 = int(m.group(2))
        r.f2 = int(m.group(3))
        r.f3 = int(m.group(4))

    return r


def run_compile(id_: int, compile_bin: Path) -> CompileResult:
    cmd = [str(compile_bin), str(id_), "--idx"]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300, cwd=str(REPO_DIR)
        )
        if result.returncode != 0:
            return CompileResult(
                error=result.stderr.strip() or f"exit {result.returncode}"
            )
        return parse_compile_output(result.stdout)
    except subprocess.TimeoutExpired:
        return CompileResult(error="timeout")
    except Exception as e:
        return CompileResult(error=str(e))


# ---------------------------------------------------------------------------
# Tensor verification
# ---------------------------------------------------------------------------

def verify_tensor(id_: int) -> str | None:
    """Compare newly compiled tensor against a pre-existing one.

    Returns None on success (or if no pre-existing tensor), error string on mismatch.
    """
    import numpy as np

    path = TENSORS_DIR / f"{id_:04d}.npy"
    backup = TENSORS_DIR / f"{id_:04d}.npy.ref"

    if not backup.exists():
        return None  # no reference to compare against

    old = np.load(backup)
    new = np.load(path)
    if np.array_equal(old, new):
        backup.unlink()
        return None

    backup.unlink()
    return "tensor mismatch vs existing"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def ensure_repo():
    """Clone circuit repository if not present."""
    if SRC_DIR.exists():
        n = len(list(SRC_DIR.glob("*.qc")))
        print(f"Circuit repo found: {CIRCUITS_REPO}  ({n} .qc files)")
        return

    print(f"Cloning {CIRCUITS_REPO_URL} ...")
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", CIRCUITS_REPO_URL],
        cwd=str(EXTERNAL_DIR),
        check=True,
    )
    n = len(list(SRC_DIR.glob("*.qc")))
    print(f"Cloned: {n} .qc files")


def main():
    ensure_repo()

    # Collect source files with known IDs
    files = sorted(SRC_DIR.glob("*.qc"))
    if not files:
        print(f"No .qc files in {SRC_DIR}", file=sys.stderr)
        sys.exit(1)

    assignments = [(f, FIXED[f.name]) for f in files if f.name in FIXED]
    assignments.sort(key=lambda x: x[1])

    DST_DIR.mkdir(parents=True, exist_ok=True)

    # Check compile binary
    compile_bin = get_compile_bin()
    do_compile = RUN_COMPILE
    if do_compile and not compile_bin.exists():
        print(f"Warning: {compile_bin} not found, skipping compilation")
        do_compile = False

    # Header
    if do_compile:
        header = (
            f"{'ID':>4}  {'Name':<24}  "
            f"{'n':>3} {'anc':>3} {'T0':>4} {'T1':>4} {'H':>3}  "
            f"{'nnz':>5} {'f1':>4} {'f2':>4} {'f3':>5}  Tensor"
        )
    else:
        header = f"{'ID':>4}  {'Name':<24}  Status"
    print(header)
    print("-" * len(header))

    errors = []

    for src, id_ in assignments:
        dst = DST_DIR / f"{id_:04d}.qc"

        # Copy .qc (skip if identical)
        if dst.exists() and filecmp.cmp(src, dst, shallow=False):
            copied = False
        else:
            shutil.copy2(src, dst)
            copied = True

        if not do_compile:
            status = "OK" if copied else "OK (unchanged)"
            print(f"{id_:04d}  {src.name:<24}  {status}")
            continue

        # Back up existing tensor for comparison
        tensor_path = TENSORS_DIR / f"{id_:04d}.npy"
        tensor_backup = TENSORS_DIR / f"{id_:04d}.npy.ref"
        had_tensor = tensor_path.exists()
        if had_tensor:
            shutil.copy2(tensor_path, tensor_backup)

        r = run_compile(id_, compile_bin)

        if r.error:
            print(f"{id_:04d}  {src.name:<24}  ERROR: {r.error}")
            errors.append((id_, r.error))
            if tensor_backup.exists():
                tensor_backup.unlink()
            continue

        # Verify tensor
        tensor_status = ""
        if had_tensor and r.has_tensor:
            err = verify_tensor(id_)
            tensor_status = "MISMATCH" if err else "match"
            if err:
                errors.append((id_, err))
        elif r.has_tensor:
            tensor_status = "new"
        else:
            tensor_status = "-"
            if tensor_backup.exists():
                tensor_backup.unlink()

        print(
            f"{id_:04d}  {src.name:<24}  "
            f"{r.qubits:>3} {r.ancillas:>3} "
            f"{r.t_before:>4} {r.t_after:>4} {r.h_count:>3}  "
            f"{r.nnz:>5} {r.f1:>4} {r.f2:>4} {r.f3:>5}  {tensor_status}"
        )

    print("-" * len(header))
    print(f"Total: {len(assignments)} circuits")

    if errors:
        print(f"\n{len(errors)} error(s):")
        for id_, err in errors:
            print(f"  {id_:04d}: {err}")
        sys.exit(1)

    print("Done.")


if __name__ == "__main__":
    main()
