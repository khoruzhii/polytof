// pauli.h
// Pauli product P = (-1)^s * Z^z ⊗ X^x for n qubits.
// Used in Clifford tableau simulation and T-merge algorithms.

#pragma once

#include "core/bit_vec.h"
#include "unordered_dense.h"

struct Pauli {
    BitVec z, x;
    bool sign = false;  // true = negative phase

    Pauli() = default;
    explicit Pauli(std::size_t n) : z(n), x(n) {}
    Pauli(BitVec z_, BitVec x_, bool s = false) : z(std::move(z_)), x(std::move(x_)), sign(s) {}

    std::size_t n_qubits() const noexcept { return z.capacity_bits(); }

    // Two Paulis commute iff |z1 & x2| + |x1 & z2| is even.
    bool commutes_with(const Pauli& o) const noexcept {
        return and_parity(z, o.x) == and_parity(x, o.z);
    }

    // Multiply this by o: P = P * o, updating phase correctly.
    // Phase rule: i^a * i^b = i^(a+b), and XZ = iY, ZX = -iY.
    // Net phase contribution from anticommuting pairs.
    void operator*=(const Pauli& o) noexcept {
        // Count phase: for each qubit, XZ vs ZX ordering matters.
        // ac = positions where (z1,x1) and (z2,x2) anticommute locally.
        BitVec x1z2 = z; x1z2 &= o.x;
        BitVec z1x2 = x; z1x2 &= o.z;
        BitVec ac = x1z2; ac ^= z1x2;  // anticommuting positions

        // Phase from multiplication: depends on bit patterns.
        // Full formula: phase += popcount(x1&z2) - popcount(z1&x2) + 2*correction (mod 4)
        // Simplified: sign flips if (popcount(ac) + 2*popcount(x1z2 & ~(x^z))) mod 4 >= 2
        x ^= o.x;
        z ^= o.z;
        BitVec corr = x1z2; corr ^= x; corr ^= z; corr &= ac;
        std::size_t phase = ac.count() + 2 * corr.count();
        sign ^= o.sign ^ ((phase % 4) >= 2);
    }

    Pauli operator*(const Pauli& o) const { Pauli r = *this; r *= o; return r; }

    bool operator==(const Pauli& o) const noexcept {
        return sign == o.sign && z == o.z && x == o.x;
    }
    bool operator!=(const Pauli& o) const noexcept { return !(*this == o); }
};

// Key for HashMap: concatenated z|x bitvectors (ignoring sign).
// Used in T-merge to find matching Pauli images.
struct PauliKey {
    BitVec zx;  // z bits followed by x bits

    PauliKey() = default;
    explicit PauliKey(const Pauli& p, std::size_t n) {
        zx.resize_bits(2 * n);
        for (std::size_t i = 0; i < n; ++i) {
            zx.set(i, p.z.test(i));
            zx.set(i + n, p.x.test(i));
        }
    }

    bool operator==(const PauliKey& o) const noexcept { return zx == o.zx; }
};

template <>
struct ankerl::unordered_dense::hash<PauliKey> {
    using is_avalanching = void;
    [[nodiscard]] auto operator()(const PauliKey& k) const noexcept -> U64 {
        return ankerl::unordered_dense::hash<BitVec>{}(k.zx);
    }
};