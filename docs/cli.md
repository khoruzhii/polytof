# Command-Line Tools

Three C++ programs implement the main optimization pipeline:

1. **bco** -- Basis Change Optimization (BCO): minimize tensor nnz via beam search over transvections
2. **cpd** -- CP decomposition: flip graph search (FGS) and symmetric greedy elimination (SGE)
3. **waring** -- Waring decomposition: FastTODD algorithm for T-count optimization

## Building

All programs require a C++20 compiler with pthreads support.  No build system is needed -- each program is a single translation unit.

```bash
# BCO (basis change optimization)
g++ -Ofast -std=c++20 -march=native -s -pthread -I third_party -I src src/bco.cpp -o bin/bco

# CP decomposition (topp = symmetric cubic tensor)
g++ -D VEC_WORDS=1 -Ofast -std=c++20 -march=native -s -pthread -I third_party -I src src/cpd.cpp -o bin/topp1
g++ -D VEC_WORDS=2 -Ofast -std=c++20 -march=native -s -pthread -I third_party -I src src/cpd.cpp -o bin/topp2
g++ -D VEC_WORDS=6 -Ofast -std=c++20 -march=native -s -pthread -I third_party -I src src/cpd.cpp -o bin/topp6

# CP decomposition (base = general bilinear tensor)
g++ -D VEC_WORDS=1 -D BASE -Ofast -std=c++20 -march=native -s -pthread -I third_party -I src src/cpd.cpp -o bin/base1

# Waring decomposition
g++ -D VEC_WORDS=1 -Ofast -std=c++20 -march=native -s -pthread -I third_party -I src src/waring.cpp -o bin/waring1
g++ -D VEC_WORDS=2 -Ofast -std=c++20 -march=native -s -pthread -I third_party -I src src/waring.cpp -o bin/waring2
g++ -D VEC_WORDS=6 -Ofast -std=c++20 -march=native -s -pthread -I third_party -I src src/waring.cpp -o bin/waring6
```

### Compile-time parameters

| Define | Default | Description |
|--------|---------|-------------|
| `VEC_WORDS` | 1 | Number of 64-bit words per bitvector. Use 1 for n <= 64, 2 for n <= 128, 6 for n <= 384. |
| `BASE` | (unset) | Build `cpd` for general bilinear tensors instead of symmetric cubic (topp). |
| `RUN=N` | (unset) | Append run suffix `-NNN` to output files, for parallel experiments. |

## bco -- Basis Change Optimization

Beam search over GF(2) transvections (x_i <- x_i + x_j) to reduce the number of nonzero entries in a symmetric cubic tensor.

```
bin\bco <tensor_id> [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `tensor_id` | (required) | Input tensor ID (decimal integer) |
| `-o, --output` | input + 1000 | Output tensor ID |
| `-b, --beam` | 1 | Beam width |
| `--patience` | 5 | Stop after N iterations without improvement |
| `-t, --threads` | 8 | Number of threads |
| `--save` | off | Save output tensor and transform matrix |
| `--verify` | off | Verify correctness of the result |
| `--log` | off | Write JSON log to `data/logs/` |
| `-v, --verbose` | off | Print per-iteration progress |

### Output files (with `--save`)

| File | Description |
|------|-------------|
| `data/tensors/{output_id}.npy` | Optimized tensor (sparse triples) |
| `data/transform/{input_id}-{output_id}.npy` | Transform matrix A, shape (n, n), uint8 |

The transform matrix A records the basis change: column A[:,i'] gives the representation of new basis vector i' in the original basis.

### Example

```bash
bin\bco 101 -b 10 -t 8 --save --verify -v
```

Runs BCO on tensor 0101 (Barenco Tof_3) with beam width 10, saves the optimized tensor as 1101 and the transform matrix.

## cpd -- CP Decomposition

Pool-based search for canonical polyadic decomposition of GF(2) tensors.  Three operational modes:

- **`--reduce`** -- Simple reduce loop (greedy rank reduction)
- **`--sge`** -- Symmetric Greedy Elimination beam search (topp variant only)
- **`--fgs`** -- Flip Graph Search (main algorithm, pool-based)

Modes can be combined: `--sge --fgs` runs SGE as preprocessing, then FGS.

```
bin\topp1 <tensor_id> [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `tensor_id` | (required) | Tensor ID (decimal integer) |
| `--reduce` | off | Apply simple reduce loop |
| `--sge` | off | Apply SGE beam search (topp only) |
| `-b, --beam` | 1 | SGE beam width |
| `--fgs` | off | Run flip graph search |
| `-f, --path-limit` | 1000000 | Max flips per walk |
| `-s, --pool-size` | 200 | Max schemes per rank bin |
| `-p, --plus-lim` | 50000 | Flips before plus transition |
| `-r, --reduce-interval` | 0 | Flips between reduce passes (0 = off) |
| `-t, --threads` | 4 | Worker threads |
| `-m, --max-attempts` | 1000 | Max walk attempts per epoch |
| `--plus` | off | Enable plus transitions |
| `--verify` | off | Verify collected schemes |
| `--save` | off | Save pool on rank improvement |
| `--continue` | off | Resume from previously saved pool |
| `--log` | off | Write JSON log |

### Output files (with `--save`)

| File | Description |
|------|-------------|
| `data/cpd/topp/{id}-{rank}.npy` | Pool of CP schemes at best rank (uint64) |

Pool files have shape `(pool_size, 3 * rank * VEC_WORDS)`.  Each row encodes `rank` rank-1 terms as consecutive bitvector triplets `(u, v, w)`.

### Choosing VEC_WORDS

The binary must be compiled with `VEC_WORDS >= ceil(n / 64)`:

| n range | VEC_WORDS | Binary |
|---------|-----------|--------|
| 1 -- 64 | 1 | `topp1` |
| 65 -- 128 | 2 | `topp2` |
| 129 -- 384 | 6 | `topp6` |

### Examples

```bash
# SGE + FGS on a small tensor
bin\topp1 101 --sge --fgs -s 100 -t 8 --save --verify

# FGS with plus transitions on a medium tensor
bin\topp2 807 --fgs --plus -s 200 -t 16 --save --log

# Resume a previous run
bin\topp1 101 --fgs --continue -s 500 --save
```

## waring -- Waring Decomposition (T-count)

Optimizes T-count via Waring decomposition using the FastTODD algorithm.  Three input modes:

- **Default** -- Build trivial decomposition from tensor, then optimize
- **`--cpd`** -- Load CP decomposition, convert to Waring, then optimize
- **`--continue`** -- Load existing Waring decomposition, continue optimizing

```
bin\waring1 <tensor_id> [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `tensor_id` | (required) | Tensor ID (decimal integer) |
| `--cpd [RANK]` | -- | Load CPD from pool; optionally specify rank (default: best available) |
| `--continue` | off | Continue from saved Waring decomposition |
| `-n, --num` | 1 | Number of CPD schemes to try (`--cpd` mode) |
| `-b, --beam` | 1 | Beam width (1 = greedy) |
| `-t, --threads` | 4 | Number of threads |
| `--no-tohpe` | off | Disable TOHPE preprocessing |
| `--verify` | off | Verify result against original tensor |
| `--save` | off | Save best result at the end |
| `--save-all` | off | Save after each improvement |
| `--log` | off | Write JSON log to `data/logs/` |

### Output files (with `--save` or `--save-all`)

| File | Description |
|------|-------------|
| `data/waring/{id}-{tcount}.npy` | Waring decomposition, shape (tcount, n), uint8 |

### Examples

```bash
# From trivial decomposition
bin\waring1 101 -b 3 -t 8 --save --verify

# From CPD pool, try 5 schemes
bin\waring1 101 --cpd -n 5 -b 3 --save-all --verify

# Continue optimizing
bin\waring1 101 --continue -t 16 --save
```

## Typical Pipeline

A full optimization run for a circuit tensor:

```bash
# 1. Setup: generate tensors from circuits
python scripts\setup_circuits_todd.py
python scripts\setup_circuits_atq.py

# 2. BCO: minimize nnz via basis change
bin\bco 101 -b 10 -t 8 --save --verify

# 3. CP decomposition on the BCO-optimized tensor
bin\topp1 1101 --sge --fgs -s 200 -t 8 --save --verify

# 4. Waring decomposition from CPD
bin\waring1 1101 --cpd -n 10 -b 3 -t 8 --save --verify
```

Steps 2-4 use the BCO-shifted tensor ID (input + 1000) by convention.

## Data Directories

| Directory | Contents |
|-----------|----------|
| `data/tensors/` | Input tensors (NumPy `.npy`, sparse triples) |
| `data/transform/` | BCO transform matrices |
| `data/cpd/topp/` | CP decomposition pools (topp variant) |
| `data/cpd/base/` | CP decomposition pools (base variant) |
| `data/waring/` | Waring decompositions |
| `data/logs/` | JSON progress logs |
| `data/circuits/raw/` | Raw circuit files (`.qc`) |
| `data/circuits/opt/` | Optimized circuits |
| `data/paper/` | Collected best results for the paper |
