#pragma once
/*
 *  A fit-like linear_only problem for the linear delta-SSE accuracy tests
 *  (tests/unittest.cc, tests/test_cuda.cc), with an f64 oracle.
 *
 *  The inputs are correlated and uncentered (mean/std = `mu`), so most
 *  candidates b_k * x_c are dominated by their in-span part b_k * mean: the
 *  regime where the f32 storage of Bo shows up in the linear delta-SSE (see
 *  MarsAlgo::eval()) and where the DGKS retry fires on most columns.
 *
 *  The oracle projects onto span(Bo) -- the f32 orthonormal basis the
 *  algorithm actually holds, read through MarsAlgo::ortho_basis() -- with an
 *  exact f64 QR of it, so it does not assume BoᵀBo = I. (Projecting onto
 *  span(B) instead differs from span(Bo) by Bo's f32 rounding, ~eps_f32·κ
 *  relative on the residual, which swamps the effect under test.)
 */
#include "../marsalgo.h"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace linear_problem {

struct Problem {
    size_t n = 0, px = 0;
    std::vector<float> X, y, w;   // X col-major (n, px)
    std::vector<float> s;         // MarsAlgo's per-column 1/rms
};

inline Problem make(size_t n, size_t px, float mu, unsigned seed)
{
    Problem P;
    P.n  = n;
    P.px = px;
    P.X.resize(n * px);
    P.y.resize(n);
    P.w.assign(n, 1.0f);
    std::mt19937 rng(seed);
    std::normal_distribution<double> g(0.0, 1.0);
    std::vector<double> z(n * px);
    for (auto &v : z) {
        v = g(rng);
    }
    for (size_t c = 1; c < px; ++c) {          // chain correlation between columns
        for (size_t i = 0; i < n; ++i) {
            z[c * n + i] = 0.6 * z[(c - 1) * n + i] + 0.8 * z[c * n + i];
        }
    }
    for (size_t i = 0; i < n * px; ++i) {
        P.X[i] = (float)(mu + z[i]);
    }
    P.s.resize(px);
    for (size_t c = 0; c < px; ++c) {          // as in the MarsAlgo constructor
        double sq = 0.0;
        for (size_t i = 0; i < n; ++i) {
            sq += (double)P.X[c * n + i] * (double)P.X[c * n + i];
        }
        const double rms = std::sqrt(sq / n);
        P.s[c] = rms > 0.0 ? (float)(1.0 / rms) : 1.0f;
    }
    /*
     *  The target is built from the unit-rms inputs, so it has a mean well
     *  above its spread (most targets do: prices, counts, ...). The first-order
     *  error under test scales with the intercept's share of y (ybo[0]).
     */
    for (size_t i = 0; i < n; ++i) {
        const double x0 = P.X[i] * P.s[0], x1 = P.X[n + i] * P.s[1],
                     x2 = P.X[2 * n + i] * P.s[2], x3 = P.X[3 * n + i] * P.s[3];
        P.y[i] = (float)(std::sin(x0) * x1 + 0.3 * x2 * x3 + 0.1 * g(rng));
    }
    return P;
}

// Grow the basis to m columns with append('l', c, k) on random (c, k).
inline void grow(MarsAlgo &algo, const Problem &P, size_t m, unsigned seed)
{
    std::mt19937 rng(seed);
    while ((size_t)algo.nbasis() < m) {
        const int k = (int)(rng() % algo.nbasis());
        const int c = (int)(rng() % P.px);
        algo.append('l', c, k, 0.0f);
    }
}

/*
 *  f64 oracle of the linear delta-SSE of every (X column c, parent k) candidate,
 *  laid out (px, m) row-major like eval_batch(): the f32 candidate
 *  b_k * (x_c * s_c), formed exactly as eval() forms it, projected onto the
 *  orthogonal complement of span(Bo) and scored as (r'y)^2 / r'r against
 *  MarsAlgo's normalized f32 target. kappa = ||candidate|| / ||residual||.
 */
inline void oracle(const MarsAlgo &algo, const Problem &P,
                   std::vector<double> &dsse, std::vector<double> &kappa)
{
    using namespace Eigen;
    typedef Matrix<float, Dynamic, Dynamic, RowMajor> MatrixXfR;
    const size_t n = P.n, m = (size_t)algo.nbasis();
    size_t ld_bo = 0;
    const float *bo = algo.ortho_basis(&ld_bo);
    Map<const MatrixXfR, 0, OuterStride<>> Bo(bo, n, m, OuterStride<>(ld_bo));
    Map<const MatrixXf> B(algo.basis(), n, m);
    const MatrixXd Q = HouseholderQR<MatrixXd>(Bo.cast<double>()).householderQ() *
                       MatrixXd::Identity(n, m);

    VectorXd y(n);
    double yy = 0.0;
    for (size_t i = 0; i < n; ++i) {
        yy += (double)P.y[i] * (double)P.y[i];
    }
    const double ynorm = std::sqrt(yy);
    for (size_t i = 0; i < n; ++i) {
        y[i] = (double)(float)((double)P.y[i] / ynorm);   // MarsAlgo's stored target
    }

    dsse.assign(P.px * m, 0.0);
    kappa.assign(P.px * m, 0.0);
    MatrixXd V(n, m);
    for (size_t c = 0; c < P.px; ++c) {
        for (size_t k = 0; k < m; ++k) {
            for (size_t i = 0; i < n; ++i) {
                const float x = P.X[c * n + i] * P.s[c];
                V(i, k) = (double)(B(i, k) * x);
            }
        }
        const MatrixXd R = V - Q * (Q.transpose() * V);
        for (size_t k = 0; k < m; ++k) {
            const double rr = R.col(k).squaredNorm();
            const double ry = R.col(k).dot(y);
            dsse [c * m + k] = rr > 0.0 ? ry * ry / rr : 0.0;
            kappa[c * m + k] = rr > 0.0 ? std::sqrt(V.col(k).squaredNorm() / rr) : INFINITY;
        }
    }
}

struct Stats {
    int    count = 0;       // candidates checked
    double max_rel = 0.0;   // max |est - ref| / ref
    double med_rel = 0.0;   // median of the same
};

/*
 *  Relative error of `est` against the oracle over the candidates that carry a
 *  real signal (ref > floor) and are not near-degenerate (kappa < kmax).
 */
inline Stats compare(const std::vector<double> &est, const std::vector<double> &ref,
                     const std::vector<double> &kappa, double floor, double kmax)
{
    std::vector<double> rel;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (ref[i] > floor && kappa[i] < kmax) {
            rel.push_back(std::abs(est[i] - ref[i]) / ref[i]);
        }
    }
    Stats S;
    S.count = (int)rel.size();
    if (!rel.empty()) {
        std::sort(rel.begin(), rel.end());
        S.max_rel = rel.back();
        S.med_rel = rel[rel.size() / 2];
    }
    return S;
}

}  // namespace linear_problem
