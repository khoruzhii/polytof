// bit_arr.h
#pragma once

#include <array>
#include <bit>
#include <cassert>
#include <cstddef>

#include "unordered_dense.h"
#include "core/types.h"

// Number of 64-bit words used to represent one vector.
// Can be overridden via compiler definitions.
#ifndef VEC_WORDS
#define VEC_WORDS 1
#endif

// Base storage word type.
using Word = U64;

// Fixed-size bit vector: capacity is 64 * VEC_WORDS bits.
struct BitArr {
    std::array<Word, VEC_WORDS> words{};

    // Number of 64-bit words in the internal storage.
    static constexpr std::size_t word_count() noexcept {
        return VEC_WORDS;
    }

    // Maximum number of addressable bits.
    static constexpr std::size_t max_bits() noexcept {
        return 64u * VEC_WORDS;
    }

    // Direct access to underlying words.
    Word& operator[](std::size_t i) noexcept {
        return words[i];
    }

    const Word& operator[](std::size_t i) const noexcept {
        return words[i];
    }

    Word* data() noexcept {
        return words.data();
    }

    const Word* data() const noexcept {
        return words.data();
    }

    // Test a single bit.
    // Precondition: pos < max_bits().
    bool test(std::size_t pos) const noexcept {
        assert(pos < max_bits());
        const std::size_t word_idx = pos / 64;
        const std::size_t bit_idx  = pos % 64;
        return (words[word_idx] & (Word(1) << bit_idx)) != 0;
    }

    // Set bit at position pos to 1.
    // Precondition: pos < max_bits().
    void set(std::size_t pos) noexcept {
        set(pos, true);
    }

    // Set bit at position pos to given boolean value.
    // Precondition: pos < max_bits().
    void set(std::size_t pos, bool value) noexcept {
        assert(pos < max_bits());
        const std::size_t word_idx = pos / 64;
        const std::size_t bit_idx  = pos % 64;
        const Word mask = Word(1) << bit_idx;
        if (value) {
            words[word_idx] |= mask;
        } else {
            words[word_idx] &= ~mask;
        }
    }

    // Reset all bits to zero.
    void reset() noexcept {
        for (auto& w : words) {
            w = 0;
        }
    }

    // Reset bit at position pos to zero.
    // Precondition: pos < max_bits().
    void reset(std::size_t pos) noexcept {
        set(pos, false);
    }

    // Flip a single bit.
    // Precondition: pos < max_bits().
    void flip(std::size_t pos) noexcept {
        assert(pos < max_bits());
        const std::size_t word_idx = pos / 64;
        const std::size_t bit_idx  = pos % 64;
        words[word_idx] ^= (Word(1) << bit_idx);
    }

    // Return true if any bit is set.
    bool any() const noexcept {
        for (Word w : words) {
            if (w != 0) return true;
        }
        return false;
    }

    // Return true if no bits are set.
    bool none() const noexcept {
        return !any();
    }

    // Count set bits.
    std::size_t count() const noexcept {
        std::size_t total = 0;
        for (Word w : words) {
            total += static_cast<std::size_t>(std::popcount(w));
        }
        return total;
    }
};

// Construct vector with a single bit set at position pos.
// Precondition: pos < BitArr::max_bits().
inline BitArr vec_from_bit(std::size_t pos) noexcept {
    assert(pos < BitArr::max_bits());
    BitArr v{};
    const std::size_t word_idx = pos / 64;
    const std::size_t bit_idx  = pos % 64;
    v.words[word_idx] = Word(1) << bit_idx;
    return v;
}

// Equality and inequality.
inline bool operator==(const BitArr& a, const BitArr& b) noexcept {
    for (std::size_t i = 0; i < VEC_WORDS; ++i) {
        if (a.words[i] != b.words[i]) return false;
    }
    return true;
}

inline bool operator!=(const BitArr& a, const BitArr& b) noexcept {
    return !(a == b);
}

// Lexicographical comparison by words (word 0 is most significant).
inline bool operator<(const BitArr& a, const BitArr& b) noexcept {
    for (std::size_t i = 0; i < VEC_WORDS; ++i) {
        if (a.words[i] < b.words[i]) return true;
        if (a.words[i] > b.words[i]) return false;
    }
    return false;
}

// Bitwise XOR (word-wise).
inline BitArr& operator^=(BitArr& a, const BitArr& b) noexcept {
    for (std::size_t i = 0; i < VEC_WORDS; ++i) {
        a.words[i] ^= b.words[i];
    }
    return a;
}

inline BitArr operator^(BitArr a, const BitArr& b) noexcept {
    a ^= b;
    return a;
}

// Bitwise OR (word-wise).
inline BitArr& operator|=(BitArr& a, const BitArr& b) noexcept {
    for (std::size_t i = 0; i < VEC_WORDS; ++i) {
        a.words[i] |= b.words[i];
    }
    return a;
}

inline BitArr operator|(BitArr a, const BitArr& b) noexcept {
    a |= b;
    return a;
}

// Hash support for ankerl::unordered_dense containers.
template <>
struct ankerl::unordered_dense::hash<BitArr> {
    using is_avalanching = void;

    [[nodiscard]] auto operator()(const BitArr& v) const noexcept -> std::uint64_t {
        return detail::wyhash::hash(v.words.data(), sizeof(v.words));
    }
};