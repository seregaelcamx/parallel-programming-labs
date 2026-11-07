#include "multiply_cuda.hpp"

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string_view>

namespace {

struct Args {
    std::filesystem::path a;
    std::filesystem::path b;
    std::filesystem::path out;
    std::filesystem::path csv;
    int repeats = 5;
    unsigned block_x = 16;
    unsigned block_y = 16;
};

[[noreturn]] void usage(const char* prog, const std::string& error = {}) {
    if (!error.empty()) {
        std::cerr << "error: " << error << '\n';
    }
    auto& out = error.empty() ? std::cout : std::cerr;
    out << "usage: " << prog << " --a A.txt --b B.txt --out C.txt"
        << " [--block XxY] [--repeat N] [--csv bench.csv]\n";
    std::exit(error.empty() ? 0 : 2);
}

int positive_int(std::string_view text, const char* option) {
    if (!text.empty()) {
        int number = 0;
        const auto end = text.data() + text.size();
        const auto parsed = std::from_chars(text.data(), end, number);
        if (parsed.ec == std::errc{} && parsed.ptr == end && number > 0) {
            return number;
        }
    }
    throw std::invalid_argument(std::string(option) + " requires a positive integer");
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string_view key = argv[i];
        if (key == "--help" || key == "-h") {
            usage(argv[0]);
        }
        auto value = [&]() -> const char* {
            if (++i >= argc) {
                throw std::invalid_argument("missing value for " + std::string(key));
            }
            return argv[i];
        };

        if (key == "--a" || key == "-a") {
            args.a = value();
        } else if (key == "--b" || key == "-b") {
            args.b = value();
        } else if (key == "--out" || key == "-o") {
            args.out = value();
        } else if (key == "--csv") {
            args.csv = value();
        } else if (key == "--repeat" || key == "-r") {
            args.repeats = positive_int(value(), "--repeat");
        } else if (key == "--block") {
            const std::string_view block = value();
            const auto separator = block.find('x');
            if (separator == std::string_view::npos) {
                throw std::invalid_argument("--block must have the form XxY");
            }
            args.block_x = static_cast<unsigned>(positive_int(block.substr(0, separator), "block X"));
            args.block_y = static_cast<unsigned>(positive_int(block.substr(separator + 1), "block Y"));
        } else {
            throw std::invalid_argument("unknown option " + std::string(key));
        }
    }
    if (args.a.empty() || args.b.empty() || args.out.empty()) {
        throw std::invalid_argument("--a, --b and --out are required");
    }
    return args;
}

struct Statistics {
    double min_ms;
    double mean_ms;
};

Statistics summarize(const std::vector<double>& samples) {
    return {*std::min_element(samples.begin(), samples.end()),
            std::accumulate(samples.begin(), samples.end(), 0.0) /
                static_cast<double>(samples.size())};
}

void create_parent_directory(const std::filesystem::path& path) {
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
}

void append_csv(const Args& args, const lab::Matrix& a, const lab::Matrix& b,
                const lab::CudaBenchmark& run, Statistics kernel, Statistics total,
                double volume_mib, double flops, double gflops) {
    constexpr std::string_view header =
        "n,rows,inner,cols,block_x,block_y,grid_x,grid_y,runs,volume_mib,flops,"
        "kernel_min_ms,kernel_mean_ms,total_min_ms,total_mean_ms,kernel_gflops";
    const bool need_header = !std::filesystem::exists(args.csv) ||
                             std::filesystem::file_size(args.csv) == 0;
    if (!need_header) {
        std::ifstream existing(args.csv);
        std::string first_line;
        std::getline(existing, first_line);
        if (!first_line.empty() && first_line.back() == '\r') {
            first_line.pop_back();
        }
        if (first_line != header) {
            throw std::runtime_error("unexpected CSV header; use a new file: " + args.csv.string());
        }
    }

    create_parent_directory(args.csv);
    std::ofstream csv(args.csv, std::ios::app);
    if (need_header) {
        csv << header << '\n';
    }
    if (a.rows() == a.cols() && a.cols() == b.cols()) {
        csv << a.rows();
    }
    csv << ',' << a.rows() << ',' << a.cols() << ',' << b.cols()
        << ',' << args.block_x << ',' << args.block_y << ',' << run.grid_x << ',' << run.grid_y
        << ',' << run.kernel_ms.size() << std::setprecision(12)
        << ',' << volume_mib << ',' << flops << ',' << kernel.min_ms << ',' << kernel.mean_ms
        << ',' << total.min_ms << ',' << total.mean_ms << ',' << gflops << '\n';
    csv.close();
    if (!csv) {
        throw std::runtime_error("cannot write CSV: " + args.csv.string());
    }
}

} // namespace

int main(int argc, char** argv) {
    Args args;
    try {
        args = parse_args(argc, argv);
    } catch (const std::invalid_argument& error) {
        usage(argv[0], error.what());
    }

    try {
        const lab::Matrix a = lab::Matrix::load(args.a);
        const lab::Matrix b = lab::Matrix::load(args.b);
        if (a.cols() != b.rows()) {
            throw std::invalid_argument("A columns must match B rows");
        }
        lab::Matrix c(a.rows(), b.cols());

        const auto run = lab::benchmark_cuda(a, b, c, args.block_x, args.block_y, args.repeats);
        const Statistics kernel = summarize(run.kernel_ms);
        const Statistics total = summarize(run.total_ms);
        const double flops = 2.0 * static_cast<double>(a.rows()) *
                             static_cast<double>(a.cols()) * static_cast<double>(b.cols());
        const double elements = static_cast<double>(a.size()) + static_cast<double>(b.size()) +
                                static_cast<double>(c.size());
        const double volume_mib = elements * sizeof(double) / (1024.0 * 1024.0);
        const double gflops = kernel.min_ms > 0.0 ? flops / (kernel.min_ms * 1e6) : 0.0;

        create_parent_directory(args.out);
        c.save(args.out);
        if (!args.csv.empty()) {
            append_csv(args, a, b, run, kernel, total, volume_mib, flops, gflops);
        }

        std::cout << "=== lab04: CUDA matrix multiplication ===\n"
                  << "device: " << run.device_name << '\n'
                  << "A: " << a.rows() << 'x' << a.cols()
                  << ", B: " << b.rows() << 'x' << b.cols()
                  << ", C: " << c.rows() << 'x' << c.cols() << '\n'
                  << "block: " << args.block_x << 'x' << args.block_y
                  << ", grid: " << run.grid_x << 'x' << run.grid_y
                  << ", measured runs: " << run.kernel_ms.size() << '\n'
                  << std::fixed << std::setprecision(6)
                  << "task volume: " << volume_mib << " MiB (3 matrices, double)\n"
                  << "kernel: min " << kernel.min_ms << " ms, mean " << kernel.mean_ms << " ms\n"
                  << "total:  min " << total.min_ms << " ms, mean " << total.mean_ms << " ms\n"
                  << "kernel performance: " << gflops << " GFLOPS (from min)\n"
                  << "result saved: " << args.out.string() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
