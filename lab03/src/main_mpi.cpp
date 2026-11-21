// Лабораторная 3. Построчное разбиение A и C, репликация B.
#include "matrix.hpp"
#include <mpi.h>

#include <cstdlib>
#include <iostream>
#include <numeric>
#include <set>

namespace {
struct Args {
    std::string a, b, out, csv, data_dir = "data";
    int n = 0, repeats = 5, warmups = 1, expected_procs = 0;
    bool help = false;
};

int integer(const std::string& s, int minimum) {
    std::size_t pos = 0;
    long long x = std::stoll(s, &pos);
    if (pos != s.size() || x < minimum || x > INT_MAX)
        throw std::runtime_error("invalid integer: " + s);
    return static_cast<int>(x);
}

Args parse(int argc, char** argv, int p) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        const auto value = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + key);
            return argv[++i];
        };
        if (key == "--a") a.a = value();
        else if (key == "--b") a.b = value();
        else if (key == "--out") a.out = value();
        else if (key == "--csv") a.csv = value();
        else if (key == "--data-dir") a.data_dir = value();
        else if (key == "--repeat") a.repeats = integer(value(), 1);
        else if (key == "--warmup") a.warmups = integer(value(), 0);
        else if (key == "--expected-procs") a.expected_procs = integer(value(), 1);
        else if (key == "--size") a.n = integer(value(), 1);
        else if (key == "--help" || key == "-h") a.help = true;
        else if (!key.empty() && key[0] != '-' && a.n == 0) a.n = integer(key, 1);
        else throw std::runtime_error("unknown option: " + key);
    }
    if (a.help) return a;
    if (a.n) {
        if (a.a.empty()) a.a = a.data_dir + "/A_" + std::to_string(a.n) + ".txt";
        if (a.b.empty()) a.b = a.data_dir + "/B_" + std::to_string(a.n) + ".txt";
        if (a.out.empty()) a.out = "C_" + std::to_string(a.n) + "_p" + std::to_string(p) + ".txt";
    }
    if (a.a.empty() || a.b.empty() || a.out.empty())
        throw std::runtime_error("use N or --a A.txt --b B.txt --out C.txt");
    if (a.out == a.a || a.out == a.b || (!a.csv.empty() &&
        (a.csv == a.a || a.csv == a.b || a.csv == a.out)))
        throw std::runtime_error("input, result and CSV paths must be different");
    if (a.warmups > INT_MAX - a.repeats) throw std::runtime_error("too many runs");
    return a;
}

void append_csv(const Args& args, int n, int p, int nodes,
                const std::vector<double>& total, const std::vector<double>& compute) {
    if (args.csv.empty()) return;
    const std::string header = "n,p,nodes,repeat,parallel_s,compute_s";
    bool empty = true;
    {
        std::ifstream in(args.csv.c_str());
        if (in && in.peek() != std::char_traits<char>::eof()) {
            std::string first;
            std::getline(in, first);
            if (first != header) throw std::runtime_error("incompatible CSV header: " + args.csv);
            empty = false;
        }
    }
    std::ofstream out(args.csv.c_str(), std::ios::app);
    if (!out) throw std::runtime_error("cannot open CSV: " + args.csv);
    if (empty) out << header << '\n';
    out << std::setprecision(17);
    for (std::size_t i = 0; i < total.size(); ++i)
        out << n << ',' << p << ',' << nodes << ',' << i + 1 << ','
            << total[i] << ',' << compute[i] << '\n';
    out.close();
    if (!out) throw std::runtime_error("cannot finish CSV: " + args.csv);
}
} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL);
    int rank = 0, p = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    try {
        const Args args = parse(argc, argv, p);
        if (args.help) {
            if (!rank) std::cout << "matrix_mul_mpi N [--data-dir data] [--out C.txt]\n"
                "or: --a A.txt --b B.txt --out C.txt\n"
                "options: --repeat 5 --warmup 1 --csv timings.csv [--expected-procs P]\n";
            MPI_Finalize();
            return 0;
        }
        if (args.expected_procs && args.expected_procs != p)
            throw std::runtime_error("MPI process count differs from --expected-procs; check launcher/PMI compatibility");

        lab::Matrix a, b, c;
        int n = 0;
        if (!rank) {
            a = lab::Matrix::load(args.a);
            b = lab::Matrix::load(args.b);
            n = a.n;
            if (b.n != n || (args.n && args.n != n))
                throw std::runtime_error("matrix sizes do not match");
            c.n = n;
            c.values.resize(static_cast<std::size_t>(n) * n);
        }
        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank) {
            b.n = n;
            b.values.resize(static_cast<std::size_t>(n) * n);
        }

        // Первые N % P процессов получают по одной дополнительной строке.
        std::vector<int> counts(p), displs(p);
        int offset = 0;
        for (int r = 0; r < p; ++r) {
            counts[r] = (n / p + (r < n % p ? 1 : 0)) * n;
            displs[r] = offset;
            offset += counts[r];
        }
        const int local_rows = counts[rank] / n;
        // Буфер из одного элемента также действителен при MPI count == 0.
        std::vector<double> local_a(std::max(1, counts[rank]));
        std::vector<double> local_c(std::max(1, counts[rank]));

        char host[MPI_MAX_PROCESSOR_NAME] = {};
        int host_length = 0;
        MPI_Get_processor_name(host, &host_length);
        std::vector<char> hosts(static_cast<std::size_t>(p) * MPI_MAX_PROCESSOR_NAME);
        MPI_Allgather(host, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, hosts.data(),
                      MPI_MAX_PROCESSOR_NAME, MPI_CHAR, MPI_COMM_WORLD);
        std::set<std::string> unique_hosts;
        for (int r = 0; r < p; ++r)
            unique_hosts.insert(std::string(hosts.data() + r * MPI_MAX_PROCESSOR_NAME));

        std::vector<double> times, compute_times;
        for (int run = 0; run < args.warmups + args.repeats; ++run) {
            MPI_Barrier(MPI_COMM_WORLD); // Все процессы готовы; барьер вне замера.
            const double t0 = MPI_Wtime();
            MPI_Scatterv(rank == 0 ? a.values.data() : nullptr, counts.data(),
                         displs.data(), MPI_DOUBLE, local_a.data(), counts[rank],
                         MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(b.values.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            const double tc = MPI_Wtime();
            lab::multiply_rows(local_a.data(), b.values.data(), local_c.data(), local_rows, n);
            const double compute = MPI_Wtime() - tc;

            MPI_Gatherv(local_c.data(), counts[rank], MPI_DOUBLE,
                        rank == 0 ? c.values.data() : nullptr, counts.data(),
                        displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
            const double duration = MPI_Wtime() - t0;

            const double local[2] = {duration, compute};
            double maxima[2] = {};
            // Берём самое большое локальное время; часы узлов не вычитаются.
            MPI_Reduce(local, maxima, 2, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (!rank && run >= args.warmups) {
                times.push_back(maxima[0]);
                compute_times.push_back(maxima[1]);
            }
        }
        if (!rank) {
            c.save(args.out); // Чтение/запись и CSV находятся вне измеряемого участка.
            append_csv(args, n, p, static_cast<int>(unique_hosts.size()), times, compute_times);
            const double minimum = *std::min_element(times.begin(), times.end());
            const double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
            std::cout << std::setprecision(8) << "N=" << n << " P=" << p
                      << " nodes=" << unique_hosts.size() << " repeats=" << args.repeats
                      << " parallel_min_s=" << minimum << " parallel_mean_s=" << mean
                      << " GFLOPS=" << 2.0 * n * n * n / minimum / 1e9
                      << "\nresult=" << args.out << "\nhosts=";
            for (const auto& name : unique_hosts) std::cout << name << ' ';
            std::cout << '\n';
        }
        MPI_Finalize();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "rank " << rank << ": " << e.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1); // Не оставляем другие процессы ждать коллективный вызов.
        return 1;
    }
}
