// random.h
#pragma once

#include "core/types.h"

// splitmix64 step used for seeding and seed derivation.
inline U64 splitmix64(U64& x) {
    U64 z = (x += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// Derive a new seed from a base seed and a stream identifier.
// Intended for deterministic derivation of per-thread or per-attempt seeds
// from one global seed.
inline U64 derive_seed(U64 base_seed, U64 stream_id) {
    U64 x = base_seed ^ (stream_id + 0x9E3779B97F4A7C15ull);
    return splitmix64(x);
}

// 64-bit RNG based on xoroshiro128+ with splitmix64 seeding.
// - Small state (128 bits).
// - Very fast next_u64() for inner loops.
// - No hidden globals: each Rng instance is fully self-contained.
struct Rng {
    U64 s0;
    U64 s1;

    static inline U64 rotl(U64 x, int k) {
        return (x << k) | (x >> (64 - k));
    }

    explicit Rng(U64 seed = 1) {
        reseed(seed);
    }

    // Reseed the generator using splitmix64.
    // The state is never left as all zeros.
    void reseed(U64 seed) {
        U64 x = seed ? seed : 0xdeadbeefcafef00dull;
        s0 = splitmix64(x);
        s1 = splitmix64(x);
        if (s0 == 0 && s1 == 0) {
            s0 = 0x0123456789abcdefull;
            s1 = 0xfedcba9876543210ull;
        }
    }

    inline U64 next_u64() {
        U64 res = s0 + s1;
        s1 ^= s0;
        s0 = rotl(s0, 24) ^ s1 ^ (s1 << 16);
        s1 = rotl(s1, 37);
        return res;
    }

    inline U32 next_u32() {
        return static_cast<U32>(next_u64());
    }
};
