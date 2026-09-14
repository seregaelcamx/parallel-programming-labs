#include "matrix.hpp"

#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace lab {

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
        if (!in) throw std::runtime_error("not enough values in " + path.string());
    }
    return m;
}

void Matrix::save(const std::filesystem::path& path) const {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot open output file: " + path.string());
    out << rows_ << ' ' << cols_ << '\n';
    out << std::setprecision(12);
    for (std::size_t i = 0; i < rows_; ++i) {
        for (std::size_t j = 0; j < cols_; ++j) {
            out << (*this)(i, j);
            if (j + 1 < cols_) out << ' ';
        }
        out << '\n';
    }
}

Matrix multiply_sequential(const Matrix& a, const Matrix& b) {
    if (a.cols() != b.rows())
        throw std::invalid_argument("matrix dimensions mismatch in multiply_sequential");

    Matrix c(a.rows(), b.cols(), 0.0);

    for (std::size_t i = 0; i < a.rows(); ++i) {
        for (std::size_t k = 0; k < a.cols(); ++k) {
            const double aik = a(i, k);
            const double* brow = &b(k, 0);
            double* crow = &c(i, 0);
            for (std::size_t j = 0; j < b.cols(); ++j) {
                crow[j] += aik * brow[j];
            }
        }
    }
    return c;
}

}