#include "matrix.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

struct Args {
    std::filesystem::path a;
    std::filesystem::path b;
    std::filesystem::path out;
    std::filesystem::path csv;
    int repeats = 5;
    std::string mode = "omp";   
    int threads = 0;            
};

[[noreturn]] void usage(const char* prog, const std::string& err = {}) {
    if (!err.empty()) std::cerr << "error: " << err << '\n';
    std::cerr << "usage: " << prog
              << " --a A.txt --b B.txt --out C.txt [--csv bench.csv] [--repeat N]"
                 " [--mode seq|omp] [--threads T]\n";
    std::exit(err.empty() ? 0 : 2);
}

Args parse_args(int argc, char** argv) {
    Args args;
    bool has_a = false, has_b = false, has_out = false;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) usage(argv[0], std::string("missing value for ") + name);
            return argv[++i];
        };
        if (key == "--a" || key == "-a")            { args.a = value("--a"); has_a = true; }
        else if (key == "--b" || key == "-b")       { args.b = value("--b"); has_b = true; }
        else if (key == "--out" || key == "-o")     { args.out = value("--out"); has_out = true; }
        else if (key == "--csv")                    { args.csv = value("--csv"); }
        else if (key == "--repeat" || key == "-r")  { args.repeats = std::atoi(value("--repeat").c_str()); }
        else if (key == "--mode" || key == "-m")    { args.mode = value("--mode"); }
        else if (key == "--threads" || key == "-t") { args.threads = std::atoi(value("--threads").c_str()); }
        else usage(argv[0], "unknown option " + key);
    }
    if (!has_a || !has_b || !has_out) usage(argv[0], "--a, --b and --out are required");
    if (args.repeats < 1) usage(argv[0], "--repeat must be >= 1");
    if (args.mode != "seq" && args.mode != "omp") usage(argv[0], "--mode must be seq or omp");
    if (args.threads < 0) usage(argv[0], "--threads must be >= 0");
    return args;
}

}

int main(int argc, char** argv) {
    const Args args = parse_args(argc, argv);

#ifdef _OPENMP
    if (args.threads > 0) omp_set_num_threads(args.threads);
    const int nthreads = omp_get_max_threads();
#else
    const int nthreads = 1;   // сборка без OpenMP: прагмы проигнорированы
#endif

    try {
        const lab::Matrix a = lab::Matrix::load(args.a);
        const lab::Matrix b = lab::Matrix::load(args.b);
        if (a.cols() != b.rows()) {
            std::cerr << "error: dimensions mismatch ("
                      << a.rows() << 'x' << a.cols() << " * "
                      << b.rows() << 'x' << b.cols() << ")\n";
            return 1;
        }

        const double flops = 2.0 * static_cast<double>(a.rows())
                                   * static_cast<double>(a.cols())
                                   * static_cast<double>(b.cols());

        using MulFn = lab::Matrix (*)(const lab::Matrix&, const lab::Matrix&);
        const MulFn multiply = (args.mode == "seq") ? &lab::multiply_sequential
                                                    : &lab::multiply_parallel;

        lab::Matrix c = multiply(a, b);   // прогревочный прогон (не учитывается)
        std::vector<double> ms;
        ms.reserve(static_cast<std::size_t>(args.repeats));
        for (int run = 0; run < args.repeats; ++run) {
            const auto t0 = std::chrono::steady_clock::now();
            c = multiply(a, b);
            const auto t1 = std::chrono::steady_clock::now();
            ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }

        const double t_min  = *std::min_element(ms.begin(), ms.end());
        const double t_mean = std::accumulate(ms.begin(), ms.end(), 0.0)
                              / static_cast<double>(ms.size());
        const double gflops = flops / (t_min * 1e6);

        c.save(args.out);

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "=== lab02: OpenMP matrix multiplication (C++) ===\n"
                  << "mode: " << args.mode << ", threads: " << nthreads << "\n"
                  << "A: " << a.rows() << 'x' << a.cols()
                  << ", B: " << b.rows() << 'x' << b.cols()
                  << ", C: " << c.rows() << 'x' << c.cols() << '\n'
                  << "task volume: "
                  << (a.size() + b.size() + c.size()) * sizeof(double) / (1024.0 * 1024.0)
                  << " MiB (3 matrices, double), flops = "
                  << std::scientific << flops << std::fixed << '\n'
                  << "runs: " << ms.size()
                  << " | t_min = " << t_min << " ms"
                  << " | t_mean = " << t_mean << " ms\n"
                  << "performance (t_min): " << std::setprecision(2) << gflops << " GFLOPS\n"
                  << "result saved: " << args.out.string() << '\n';

        if (!args.csv.empty()) {
            const bool need_header = !std::filesystem::exists(args.csv) ||
                                     std::filesystem::file_size(args.csv) == 0;
            std::ofstream f(args.csv, std::ios::app);
            if (!f) { std::cerr << "error: cannot open " << args.csv.string() << '\n'; return 1; }
            if (need_header) f << "n,threads,runs,t_min_ms,t_mean_ms,gflops\n";
            f << a.rows() << ',' << nthreads << ',' << ms.size() << ','
              << t_min << ',' << t_mean << ',' << gflops << '\n';
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}