/*
 *  Standalone CPU microbench for the forward-pass eval() at a pre-grown basis,
 *  isolated from the Python fit() loop: time append() while growing the basis
 *  with random hinges, then eval() on one X column with every basis active --
 *  once linear_only (orthonormalize + linear ΔSSE) and once with the hinge sweep.
 *  Single-threaded; the difference between the two eval() timings is the sweep.
 *
 *  Usage:  bench_sweep [n] [m] [px] [min_span] [repeat] [dump]
 *    n        rows                       (default 500000)
 *    m        basis columns to grow to   (default 100)
 *    px       X columns                  (default 20)
 *    min_span hinge-cut grid spacing     (default 7, ~Friedman's at n=5e5)
 *    repeat   timed eval() repeats       (default 2)
 *    dump     optional path; writes the final linear/hinge/cut arrays as raw
 *             f64 for a bit-exact A/B comparison between builds
 *
 *  The basis is grown the way fit() grows it -- degree <= 3, no input repeated
 *  within a term, alternating linear terms and hinge pairs -- so it is as well
 *  conditioned as a real fit's. (Random hinges on random parents build a nearly
 *  collinear basis whose orthonormalized directions are noise-dominated, which
 *  makes the delta-SSEs hypersensitive and useless for A/B checks.)
 */
#include "../marsalgo.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <algorithm>
#include <string>

namespace {
double now_s()
{
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
}
size_t arg(int argc, char **argv, int i, size_t dflt)
{
    return (argc > i) ? (size_t)std::strtoull(argv[i], nullptr, 10) : dflt;
}
}  // namespace

int main(int argc, char **argv)
{
    const size_t n        = arg(argc, argv, 1, 500000);
    const size_t m        = arg(argc, argv, 2, 100);
    const size_t px       = arg(argc, argv, 3, 20);
    const int    min_span = (int)arg(argc, argv, 4, 7);
    const int    repeat   = (int)arg(argc, argv, 5, 2);
    std::printf("# n=%zu m=%zu px=%zu min_span=%d repeat=%d\n", n, m, px, min_span, repeat);

    std::mt19937 rng(12345);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::vector<float> X(n * px), y(n), w(n, 1.0f);
    for (auto &v : X) v = gauss(rng);
    for (size_t i = 0; i < n; ++i) {
        const float x0 = X[i], x1 = X[n + i], x2 = X[2*n + i], x3 = X[3*n + i], x4 = X[4*n + i];
        y[i] = std::max(x0 - 0.3f, 0.0f) - 0.8f*std::max(0.5f - x1, 0.0f)
               + 0.6f*x2*std::max(x3, 0.0f) + 0.4f*x4 + 0.3f*gauss(rng);
    }

    MarsAlgo algo(X.data(), y.data(), w.data(), n, px, m + 2, n);

    // Grow the basis like fit(): pick (input, parent) pairs the degree / no-self-
    // interaction filters allow, alternating a linear term and a mirrored hinge
    // pair. Each (input, parent) pair is used once: a second term on the same
    // pair (another linear, or a hinge pair once the linear or another pair is
    // in) is exactly collinear. The same product can still arise by two routes
    // (x1 on lin(x2) == x2 on lin(x1)), so a rejected append is simply skipped,
    // as fit() would never select a zero-gain candidate. `inputs[b]` is the set
    // of inputs in basis column b.
    std::vector<std::vector<int>> inputs(1);
    std::vector<char> used(px * (m + 2), 0);
    std::uniform_real_distribution<float> cut(-1.0f, 1.0f);
    double t_app = 0.0, t_app_last = 0.0;
    int n_last = 0, step = 0, n_rejected = 0;
    while ((size_t)algo.nbasis() < m) {
        int xcol, bcol;
        do {
            xcol = (int)(rng() % px);
            bcol = (int)(rng() % algo.nbasis());
        } while (inputs[bcol].size() >= 3 || used[xcol * (m + 2) + bcol] ||
                 std::find(inputs[bcol].begin(), inputs[bcol].end(), xcol) != inputs[bcol].end());
        used[xcol * (m + 2) + bcol] = 1;
        const bool  hinge = (step++ % 2) == 1;
        const float h     = cut(rng);
        for (char type : (hinge ? std::string("+-") : std::string("l"))) {
            if ((size_t)algo.nbasis() >= m) {
                break;
            }
            const double t0  = now_s();
            const double mse = algo.append(type, xcol, bcol, h);
            const double dt  = now_s() - t0;
            t_app += dt;
            if ((size_t)algo.nbasis() + 10 >= m) {
                t_app_last += dt;
                ++n_last;
            }
            if (mse < 0) {
                ++n_rejected;
                break;
            }
            inputs.push_back(inputs[bcol]);
            inputs.back().push_back(xcol);
        }
    }
    std::printf("append: total %.2fs, mean of last %d: %.1f ms  (%d collinear candidates rejected)\n",
                t_app, n_last, 1e3 * t_app_last / n_last, n_rejected);

    std::vector<char> mask(m, 1);
    std::vector<double> lin(m), hin(m), cutp(m);
    const int end_span = (int)(0.02 * n);
    for (int linear = 1; linear >= 0; --linear) {
        algo.eval(lin.data(), hin.data(), cutp.data(), 0, (const bool *)mask.data(),
                  min_span, end_span, linear != 0);  // warm
        const double t0 = now_s();
        for (int r = 0; r < repeat; ++r) {
            algo.eval(lin.data(), hin.data(), cutp.data(), 0, (const bool *)mask.data(),
                      min_span, end_span, linear != 0);
        }
        const double dt = (now_s() - t0) / repeat;
        std::printf("eval(%s): %.3f s  (%.2f ms per basis)\n", linear ? "linear" : "hinge ",
                    dt, 1e3 * dt / m);
    }
    // Keep the outputs observable; optionally dump them for an A/B diff.
    double chk = 0.0;
    for (size_t j = 0; j < m; ++j) {
        chk += lin[j] + hin[j];
    }
    std::printf("# checksum %.9e\n", chk);
    if (argc > 6) {
        FILE *f = std::fopen(argv[6], "wb");
        if (!f) {
            return 1;
        }
        std::fwrite(lin.data(),  sizeof(double), m, f);
        std::fwrite(hin.data(),  sizeof(double), m, f);
        std::fwrite(cutp.data(), sizeof(double), m, f);
        std::fclose(f);
    }
    return 0;
}
