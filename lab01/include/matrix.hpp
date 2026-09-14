#pragma once

#include <cstddef>
#include <filesystem>
#include <vector>

namespace lab {

class Matrix {
public:
    Matrix() = default;
    Matrix(std::size_t rows, std::size_t cols, double fill = 0.0)
        : rows_(rows), cols_(cols), data_(rows * cols, fill) {}

    [[nodiscard]] auto rows() const noexcept { return rows_; }
    [[nodiscard]] auto cols() const noexcept { return cols_; }
    [[nodiscard]] auto size() const noexcept { return rows_ * cols_; }
    [[nodiscard]] bool is_square() const noexcept { return rows_ == cols_; }

    double& operator()(std::size_t i, std::size_t j) noexcept { return data_[i * cols_ + j]; }
    const double& operator()(std::size_t i, std::size_t j) const noexcept { return data_[i * cols_ + j]; }

    [[nodiscard]] static Matrix load(const std::filesystem::path& path);
    void save(const std::filesystem::path& path) const;

private:
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    std::vector<double> data_;
};

Matrix multiply_sequential(const Matrix& a, const Matrix& b);

} 