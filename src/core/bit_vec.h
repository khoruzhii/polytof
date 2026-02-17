// bit_vec.h
// Dynamic bit vector for quantum circuit computations.

#pragma once

#include <bit>
#include <cassert>
#include <cstddef>
#include <vector>

#include "small_vector.hpp"
#include "unordered_dense.h"
#include "core/types.h"

struct BitVec {
    using Word = U64;
    static constexpr std::size_t kWordBits = 64;
    static constexpr std::size_t kInlineWords = 8;

    itlib::small_vector<Word, kInlineWords> words;

    BitVec() = default;
    explicit BitVec(std::size_t n_bits) { resize_bits(n_bits); }

    std::size_t word_count() const noexcept { return words.size(); }
    std::size_t capacity_bits() const noexcept { return words.size() * kWordBits; }

    void resize_bits(std::size_t n_bits) {
        words.resize((n_bits + kWordBits - 1) / kWordBits, 0);
    }
    void resize_words(std::size_t n_words) { words.resize(n_words, 0); }
    void clear() noexcept { words.clear(); }
    void reset() noexcept { for (auto& w : words) w = 0; }

    void reset(std::size_t pos) noexcept {
        std::size_t wi = pos / kWordBits;
        assert(wi < words.size());
        words[wi] &= ~(Word(1) << (pos % kWordBits));
    }

    bool test(std::size_t pos) const noexcept {
        std::size_t wi = pos / kWordBits;
        if (wi >= words.size()) return false;
        return (words[wi] >> (pos % kWordBits)) & 1;
    }

    void set(std::size_t pos) noexcept {
        std::size_t wi = pos / kWordBits;
        assert(wi < words.size());
        words[wi] |= Word(1) << (pos % kWordBits);
    }

    void set(std::size_t pos, bool value) noexcept {
        if (value) set(pos); else reset(pos);
    }

    void flip(std::size_t pos) noexcept {
        std::size_t wi = pos / kWordBits;
        assert(wi < words.size());
        words[wi] ^= Word(1) << (pos % kWordBits);
    }

    std::size_t count() const noexcept {
        std::size_t cnt = 0;
        for (Word w : words) cnt += static_cast<std::size_t>(std::popcount(w));
        return cnt;
    }

    bool parity() const noexcept {
        Word x = 0;
        for (Word w : words) x ^= w;
        return std::popcount(x) & 1;
    }

    // Index of first set bit, or capacity_bits() if none.
    std::size_t first_one() const noexcept {
        for (std::size_t i = 0; i < words.size(); ++i)
            if (words[i] != 0)
                return i * kWordBits + static_cast<std::size_t>(std::countr_zero(words[i]));
        return capacity_bits();
    }

    // Collect indices of all set bits up to n_bits.
    std::vector<std::size_t> ones(std::size_t n_bits) const {
        std::vector<std::size_t> result;
        for (std::size_t i = 0; i < words.size(); ++i) {
            Word w = words[i];
            std::size_t base = i * kWordBits;
            while (w != 0) {
                std::size_t bit = static_cast<std::size_t>(std::countr_zero(w));
                std::size_t idx = base + bit;
                if (idx >= n_bits) return result;
                result.push_back(idx);
                w &= w - 1;  // clear lowest set bit
            }
        }
        return result;
    }

    bool any() const noexcept {
        for (Word w : words) if (w != 0) return true;
        return false;
    }
    bool none() const noexcept { return !any(); }

    bool all(std::size_t n_bits) const noexcept {
        if (n_bits == 0) return true;
        std::size_t full = n_bits / kWordBits, rem = n_bits % kWordBits;
        for (std::size_t i = 0; i < full && i < words.size(); ++i)
            if (words[i] != ~Word(0)) return false;
        if (rem > 0 && full < words.size()) {
            Word mask = (Word(1) << rem) - 1;
            if ((words[full] & mask) != mask) return false;
        }
        return true;
    }

    BitVec& operator^=(const BitVec& o) noexcept {
        assert(words.size() == o.words.size());
        for (std::size_t i = 0; i < words.size(); ++i) words[i] ^= o.words[i];
        return *this;
    }

    BitVec& operator&=(const BitVec& o) noexcept {
        assert(words.size() == o.words.size());
        for (std::size_t i = 0; i < words.size(); ++i) words[i] &= o.words[i];
        return *this;
    }

    BitVec& operator|=(const BitVec& o) noexcept {
        assert(words.size() == o.words.size());
        for (std::size_t i = 0; i < words.size(); ++i) words[i] |= o.words[i];
        return *this;
    }

    BitVec operator~() const noexcept {
        BitVec r = *this;
        for (auto& w : r.words) w = ~w;
        return r;
    }

    bool operator==(const BitVec& o) const noexcept {
        if (words.size() != o.words.size()) return false;
        for (std::size_t i = 0; i < words.size(); ++i)
            if (words[i] != o.words[i]) return false;
        return true;
    }
    bool operator!=(const BitVec& o) const noexcept { return !(*this == o); }

    bool operator<(const BitVec& o) const noexcept {
        std::size_t n = std::min(words.size(), o.words.size());
        for (std::size_t i = 0; i < n; ++i) {
            if (words[i] < o.words[i]) return true;
            if (words[i] > o.words[i]) return false;
        }
        return words.size() < o.words.size();
    }
};

inline BitVec operator^(BitVec a, const BitVec& b) noexcept { return a ^= b; }
inline BitVec operator&(BitVec a, const BitVec& b) noexcept { return a &= b; }
inline BitVec operator|(BitVec a, const BitVec& b) noexcept { return a |= b; }

// Popcount of (a & b).
inline std::size_t and_count(const BitVec& a, const BitVec& b) noexcept {
    assert(a.words.size() == b.words.size());
    std::size_t cnt = 0;
    for (std::size_t i = 0; i < a.words.size(); ++i)
        cnt += static_cast<std::size_t>(std::popcount(a.words[i] & b.words[i]));
    return cnt;
}

// Parity of popcount(a & b), i.e. popcount(a & b) % 2.
inline bool and_parity(const BitVec& a, const BitVec& b) noexcept {
    assert(a.words.size() == b.words.size());
    BitVec::Word x = 0;
    for (std::size_t i = 0; i < a.words.size(); ++i) x ^= a.words[i] & b.words[i];
    return std::popcount(x) & 1;
}

template <>
struct ankerl::unordered_dense::hash<BitVec> {
    using is_avalanching = void;
    [[nodiscard]] auto operator()(const BitVec& v) const noexcept -> U64 {
        return ankerl::unordered_dense::detail::wyhash::hash(
            v.words.data(), v.words.size() * sizeof(BitVec::Word));
    }
};