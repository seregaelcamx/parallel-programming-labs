#include "matrix.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>

namespace lab {

Matrix::Matrix(std::size_t rows, std::size_t cols, double fill)
    : rows_(rows), cols_(cols) {
    if (cols != 0 && rows > data_.max_size() / cols) {
        throw std::length_error("matrix dimensions are too large");
    }
    data_.assign(rows * cols, fill);
}

Matrix Matrix::load(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open input file: " + path.string());

    std::size_t rows = 0, cols = 0;
    in >> rows >> cols;
    if (!in || rows == 0 || cols == 0)
        throw std::runtime_error("bad header (expected \"rows cols\"): " + path.string());

    Matrix m(rows, cols);
    for (auto& v : m.data_) {
        in >> v;
        if (!in || !std::isfinite(v))
            throw std::runtime_error("missing or invalid matrix value in " + path.string());
    }
    in >> std::ws;
    if (!in.eof()) throw std::runtime_error("extra matrix values in " + path.string());
    return m;
}

void Matrix::save(const std::filesystem::path& path) const {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot open output file: " + path.string());
    out << rows_ << ' ' << cols_ << '\n';
    out << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (std::size_t i = 0; i < rows_; ++i) {
        for (std::size_t j = 0; j < cols_; ++j) {
            out << (*this)(i, j);
            if (j + 1 < cols_) out << ' ';
        }
        out << '\n';
    }
    out.close();
    if (!out) throw std::runtime_error("cannot write output file: " + path.string());
}

} // namespace lab
