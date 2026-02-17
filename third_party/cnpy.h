// Copyright (C) 2011  Carl Rogers
// Released under MIT License
#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <vector>

namespace cnpy {

// ── NpyArray struct ──────────────────────────────────────────────────
struct NpyArray {
    NpyArray(const std::vector<size_t>& _shape, size_t _word_size, bool _fortran_order)
        : shape(_shape), word_size(_word_size), fortran_order(_fortran_order) {
        num_vals = 1;
        for (size_t d : shape) num_vals *= d;
        data_holder = std::make_shared<std::vector<char>>(num_vals * word_size);
    }

    NpyArray() : word_size(0), fortran_order(false), num_vals(0) {}

    template <typename T>       T* data()       { return reinterpret_cast<T*>(&(*data_holder)[0]); }
    template <typename T> const T* data() const { return reinterpret_cast<const T*>(&(*data_holder)[0]); }
    template <typename T> std::vector<T> as_vec() const { return {data<T>(), data<T>() + num_vals}; }
    size_t num_bytes() const { return data_holder->size(); }

    std::shared_ptr<std::vector<char>> data_holder;
    std::vector<size_t> shape;
    size_t word_size{};
    bool fortran_order{};
    size_t num_vals{};
};

// ── inline helpers ───────────────────────────────────────────────────
inline char BigEndianTest() {
    int x = 1;
    return (reinterpret_cast<char*>(&x)[0]) ? '<' : '>';
}

inline char map_type(const std::type_info& t) {
    if (t == typeid(float) || t == typeid(double) || t == typeid(long double))         return 'f';
    if (t == typeid(int) || t == typeid(char) || t == typeid(short) || 
        t == typeid(signed char) || t == typeid(long) || t == typeid(long long))       return 'i';
    if (t == typeid(unsigned char) || t == typeid(unsigned short) ||
        t == typeid(unsigned long) || t == typeid(unsigned long long) ||
        t == typeid(unsigned int))                                                     return 'u';
    if (t == typeid(bool))                                                             return 'b';
    return '?';
}

// operator+= specializations
template <typename T>
inline std::vector<char>& operator+=(std::vector<char>& lhs, const T rhs) {
    for (size_t byte = 0; byte < sizeof(T); ++byte)
        lhs.push_back(*((char*)&rhs + byte));  // little-endian
    return lhs;
}

template <>
inline std::vector<char>& operator+=(std::vector<char>& lhs, const std::string rhs) {
    lhs.insert(lhs.end(), rhs.begin(), rhs.end());
    return lhs;
}

template <>
inline std::vector<char>& operator+=(std::vector<char>& lhs, const char* rhs) {
    lhs.insert(lhs.end(), rhs, rhs + std::strlen(rhs));
    return lhs;
}

// ── header parsing ───────────────────────────────────────────────────
inline void parse_npy_header(unsigned char* buffer, size_t& word_size,
                            std::vector<size_t>& shape, bool& fortran_order) {
    uint16_t header_len = *reinterpret_cast<uint16_t*>(buffer + 8);
    std::string header(reinterpret_cast<char*>(buffer + 9), header_len);

    // fortran_order
    auto loc = header.find("fortran_order") + 16;
    fortran_order = header.substr(loc, 4) == "True";

    // shape
    loc = header.find('(');
    auto loc2 = header.find(')');
    std::regex num_rgx("[0-9]+");
    std::smatch sm;
    shape.clear();
    std::string shp = header.substr(loc + 1, loc2 - loc - 1);
    while (std::regex_search(shp, sm, num_rgx)) {
        shape.push_back(std::stoul(sm[0].str()));
        shp = sm.suffix().str();
    }

    // word size
    loc = header.find("descr") + 9;
    bool little = header[loc] == '<' || header[loc] == '|';
    assert(little && "Big-endian data not supported");
    word_size = std::stoul(header.substr(loc + 2, header.find('\'', loc + 2) - (loc + 2)));
}

inline void parse_npy_header(FILE* fp, size_t& word_size,
                            std::vector<size_t>& shape, bool& fortran_order) {
    char buffer[256];
    if (fread(buffer, 1, 11, fp) != 11)
        throw std::runtime_error("parse_npy_header: failed fread");
    std::string header = fgets(buffer, 256, fp);
    if (header.back() != '\n')
        throw std::runtime_error("parse_npy_header: malformed header");

    // fortran_order
    auto loc = header.find("fortran_order");
    if (loc == std::string::npos)
        throw std::runtime_error("parse_npy_header: 'fortran_order' not found");
    fortran_order = header.substr(loc + 16, 4) == "True";

    // shape
    auto loc1 = header.find('(');
    auto loc2 = header.find(')');
    if (loc1 == std::string::npos || loc2 == std::string::npos)
        throw std::runtime_error("parse_npy_header: shape not found");
    std::regex num_rgx("[0-9]+");
    std::smatch sm;
    shape.clear();
    std::string shp = header.substr(loc1 + 1, loc2 - loc1 - 1);
    while (std::regex_search(shp, sm, num_rgx)) {
        shape.push_back(std::stoul(sm[0].str()));
        shp = sm.suffix().str();
    }

    // word size
    loc = header.find("descr");
    if (loc == std::string::npos)
        throw std::runtime_error("parse_npy_header: 'descr' not found");
    loc += 9;
    bool little = header[loc] == '<' || header[loc] == '|';
    assert(little && "Big-endian data not supported");
    word_size = std::stoul(header.substr(loc + 2, header.find('\'', loc + 2) - (loc + 2)));
}

// ── create header ────────────────────────────────────────────────────
template <typename T>
inline std::vector<char> create_npy_header(const std::vector<size_t>& shape) {
    std::vector<char> dict;
    dict += "{'descr': '";
    dict += BigEndianTest();
    dict += map_type(typeid(T));
    dict += std::to_string(sizeof(T));
    dict += "', 'fortran_order': False, 'shape': (";
    dict += std::to_string(shape[0]);
    for (size_t i = 1; i < shape.size(); ++i) {
        dict += ", ";
        dict += std::to_string(shape[i]);
    }
    if (shape.size() == 1) dict += ",";
    dict += "), }";

    int pad = 16 - (10 + int(dict.size())) % 16;  // preamble: 10 bytes
    dict.insert(dict.end(), pad, ' ');
    dict.back() = '\n';

    std::vector<char> header;
    header += char(0x93);              // magic
    header += "NUMPY";
    header += char(0x01);              // major
    header += char(0x00);              // minor
    header += uint16_t(dict.size());   // header length (little-endian)
    header.insert(header.end(), dict.begin(), dict.end());
    return header;
}

// ── load functions ───────────────────────────────────────────────────
inline NpyArray load_npy_file(FILE* fp) {
    std::vector<size_t> shape;
    size_t word_size;
    bool fortran;
    parse_npy_header(fp, word_size, shape, fortran);

    NpyArray arr(shape, word_size, fortran);
    if (fread(arr.data<char>(), 1, arr.num_bytes(), fp) != arr.num_bytes())
        throw std::runtime_error("npy_load: failed fread (data)");
    return arr;
}

inline NpyArray npy_load(const std::string& fname) {
    FILE* fp = std::fopen(fname.c_str(), "rb");
    if (!fp) throw std::runtime_error("npy_load: unable to open '" + fname + '\'');
    NpyArray arr = load_npy_file(fp);
    std::fclose(fp);
    return arr;
}

// ── save functions ───────────────────────────────────────────────────
template <typename T>
inline void npy_save(const std::string& fname, const T* data,
                    const std::vector<size_t>& shape, std::string mode = "w") {
    FILE* fp = nullptr;
    std::vector<size_t> true_shape;

    if (mode == "a") fp = std::fopen(fname.c_str(), "r+b");

    if (fp) {
        size_t word_size; bool fortran;
        parse_npy_header(fp, word_size, true_shape, fortran);
        assert(!fortran && "Fortran order not supported");
        assert(word_size == sizeof(T));
        assert(true_shape.size() == shape.size());
        for (size_t i = 1; i < shape.size(); ++i)
            assert(shape[i] == true_shape[i]);
        true_shape[0] += shape[0];  // grow first dimension
    } else {
        fp = std::fopen(fname.c_str(), "wb");
        true_shape = shape;
    }

    auto header = create_npy_header<T>(true_shape);
    const size_t nels = std::accumulate(shape.begin(), shape.end(), 
                                       size_t{1}, std::multiplies<size_t>());

    std::fseek(fp, 0, SEEK_SET);
    std::fwrite(header.data(), 1, header.size(), fp);
    std::fseek(fp, 0, SEEK_END);
    std::fwrite(data, sizeof(T), nels, fp);
    std::fclose(fp);
}

template <typename T>
inline void npy_save(const std::string& fname, const std::vector<T>& data,
                    std::string mode = "w") {
    npy_save(fname, data.data(), {data.size()}, std::move(mode));
}

}