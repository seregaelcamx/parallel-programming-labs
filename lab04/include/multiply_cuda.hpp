#pragma once

#include "matrix.hpp"

#include <string>
#include <vector>

namespace lab {

struct CudaBenchmark {
    std::string device_name;
    unsigned grid_x;
    unsigned grid_y;
    std::vector<double> kernel_ms;
    std::vector<double> total_ms;
};

CudaBenchmark benchmark_cuda(const Matrix& a, const Matrix& b, Matrix& c,
                             unsigned block_x, unsigned block_y, int repeats);

} // namespace lab
