#pragma once

#include <algorithm>
#include <climits>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

namespace lab {

// Формат и ядро i-k-j перенесены из лабораторной 1.
// Строки хранятся подряд: такой буфер можно передавать в MPI.
struct Matrix {
    int n = 0;
    std::vector<double> values;

    static Matrix load(const std::string& path) {
        std::ifstream in(path.c_str());
        if (!in) throw std::runtime_error("cannot open input: " + path);
        long long rows = 0, cols = 0;
        if (!(in >> rows >> cols) || rows <= 0 || rows != cols ||
            rows > INT_MAX || rows > INT_MAX / rows)
            throw std::runtime_error("expected square matrix, 0 < N*N <= INT_MAX: " + path);
        Matrix m;
        m.n = static_cast<int>(rows);
        m.values.resize(static_cast<std::size_t>(rows * rows));
        for (double& v : m.values)
            if (!(in >> v) || !std::isfinite(v))
                throw std::runtime_error("missing or invalid matrix value: " + path);
        std::string extra;
        if (in >> extra) throw std::runtime_error("extra values after matrix: " + path);
        return m;
    }

    void save(const std::string& path) const {
        std::ofstream out(path.c_str());
        if (!out) throw std::runtime_error("cannot open output: " + path);
        out << n << ' ' << n << '\n' << std::setprecision(17);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (j) out << ' ';
                out << values[static_cast<std::size_t>(i) * n + j];
            }
            out << '\n';
        }
        out.close();
        if (!out) throw std::runtime_error("cannot finish writing: " + path);
    }
};

// Нулевые строки допустимы: это нужно для числа процессов P > N.
inline void multiply_rows(const double* a, const double* b, double* c,
                          int rows, int n) {
    std::fill(c, c + static_cast<std::size_t>(rows) * n, 0.0);
    for (int i = 0; i < rows; ++i) {
        double* crow = c + static_cast<std::size_t>(i) * n;
        for (int k = 0; k < n; ++k) {
            const double aik = a[static_cast<std::size_t>(i) * n + k];
            const double* brow = b + static_cast<std::size_t>(k) * n;
            for (int j = 0; j < n; ++j) crow[j] += aik * brow[j];
        }
    }
}
} // namespace lab
