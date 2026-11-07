#include "multiply_cuda.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>

namespace lab {
namespace {

__global__ void multiply_kernel(const double* a, const double* b, double* c,
                                std::size_t rows, std::size_t inner, std::size_t cols) {
    const std::size_t row = static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const std::size_t col = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows || col >= cols) {
        return;
    }

    double sum = 0.0;
    for (std::size_t k = 0; k < inner; ++k) {
        sum += a[row * inner + k] * b[k * cols + col];
    }
    c[row * cols + col] = sum;
}

void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

using DeviceBuffer = std::unique_ptr<double, decltype(&cudaFree)>;

DeviceBuffer allocate_device(std::size_t bytes) {
    void* pointer = nullptr;
    cuda_check(cudaMalloc(&pointer, bytes), "cudaMalloc");
    return DeviceBuffer(static_cast<double*>(pointer), cudaFree);
}

class Event {
public:
    Event() { cuda_check(cudaEventCreate(&handle_), "cudaEventCreate"); }
    ~Event() { cudaEventDestroy(handle_); }

    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;

    cudaEvent_t get() const { return handle_; }

private:
    cudaEvent_t handle_ = nullptr;
};

} // namespace

CudaBenchmark benchmark_cuda(const Matrix& a, const Matrix& b, Matrix& c,
                             unsigned block_x, unsigned block_y, int repeats) {
    if (a.size() == 0 || b.size() == 0 || a.cols() != b.rows() ||
        c.rows() != a.rows() || c.cols() != b.cols()) {
        throw std::invalid_argument("incompatible matrix dimensions");
    }
    if (repeats < 1) {
        throw std::invalid_argument("at least one measured run is required");
    }

    int device = 0;
    cudaDeviceProp limits{};
    cuda_check(cudaGetDevice(&device), "cudaGetDevice");
    cuda_check(cudaGetDeviceProperties(&limits, device), "cudaGetDeviceProperties");
    if (block_x == 0 || block_y == 0 ||
        block_x > static_cast<unsigned>(limits.maxThreadsDim[0]) ||
        block_y > static_cast<unsigned>(limits.maxThreadsDim[1]) ||
        block_x > static_cast<unsigned>(limits.maxThreadsPerBlock) / block_y) {
        throw std::invalid_argument("block dimensions exceed the device limits");
    }

    const std::size_t grid_x = 1 + (b.cols() - 1) / block_x;
    const std::size_t grid_y = 1 + (a.rows() - 1) / block_y;
    if (grid_x > static_cast<unsigned>(limits.maxGridSize[0]) ||
        grid_y > static_cast<unsigned>(limits.maxGridSize[1])) {
        throw std::invalid_argument("grid dimensions exceed the device limits");
    }
    const dim3 block(block_x, block_y);
    const dim3 grid(static_cast<unsigned>(grid_x), static_cast<unsigned>(grid_y));

    const auto bytes_a = a.size() * sizeof(double);
    const auto bytes_b = b.size() * sizeof(double);
    const auto bytes_c = c.size() * sizeof(double);
    const auto device_a = allocate_device(bytes_a);
    const auto device_b = allocate_device(bytes_b);
    const auto device_c = allocate_device(bytes_c);
    const Event start, stop;

    CudaBenchmark result{limits.name, grid.x, grid.y, {}, {}};
    result.kernel_ms.reserve(repeats);
    result.total_ms.reserve(repeats);

    // Run -1 warms up the same copies and kernel used in the measured runs.
    for (int run = -1; run < repeats; ++run) {
        const auto wall_start = std::chrono::steady_clock::now();
        cuda_check(cudaMemcpy(device_a.get(), a.data(), bytes_a, cudaMemcpyHostToDevice),
                   "cudaMemcpy A to device");
        cuda_check(cudaMemcpy(device_b.get(), b.data(), bytes_b, cudaMemcpyHostToDevice),
                   "cudaMemcpy B to device");

        cuda_check(cudaEventRecord(start.get()), "cudaEventRecord start");
        multiply_kernel<<<grid, block>>>(device_a.get(), device_b.get(), device_c.get(),
                                         a.rows(), a.cols(), b.cols());
        cuda_check(cudaGetLastError(), "launch multiply_kernel");
        cuda_check(cudaEventRecord(stop.get()), "cudaEventRecord stop");
        cuda_check(cudaEventSynchronize(stop.get()), "cudaEventSynchronize stop");

        cuda_check(cudaMemcpy(c.data(), device_c.get(), bytes_c, cudaMemcpyDeviceToHost),
                   "cudaMemcpy C to host");
        const auto wall_stop = std::chrono::steady_clock::now();

        if (run >= 0) {
            float kernel_ms = 0.0f;
            cuda_check(cudaEventElapsedTime(&kernel_ms, start.get(), stop.get()),
                       "cudaEventElapsedTime");
            result.kernel_ms.push_back(kernel_ms);
            result.total_ms.push_back(
                std::chrono::duration<double, std::milli>(wall_stop - wall_start).count());
        }
    }
    return result;
}

} // namespace lab
