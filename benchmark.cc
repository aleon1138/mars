#include "marsalgo.h"

class Timer {
    uint64_t t0;

public:
    void tic()
    {
        unsigned cycles_low, cycles_high;
        asm volatile ("CPUID\n\t"
                      "RDTSC\n\t"
                      "mov %%edx, %0\n\t"
                      "mov %%eax, %1\n\t": "=r" (cycles_high), "=r" (cycles_low)::
                      "%rax", "%rbx", "%rcx", "%rdx");
        this->t0 = (uint64_t(cycles_high) << 32) + cycles_low;
    }

    uint64_t toc()
    {
        unsigned cycles_low, cycles_high;
        asm volatile("RDTSCP\n\t"
                     "mov %%edx, %0\n\t"
                     "mov %%eax, %1\n\t"
                     "CPUID\n\t": "=r" (cycles_high), "=r" (cycles_low)::
                     "%rax", "%rbx", "%rcx", "%rdx");
        uint64_t t1 = (uint64_t(cycles_high) << 32) + cycles_low;
        return t1 - this->t0;
    }
};

void bench_marsalgo()
{
    int n = 100 * 1000;
    int m = 600;
    MatrixXf X = MatrixXfC::Random(n,m);
    ArrayXf  y = ArrayXf::Random(n);
    ArrayXf  w = ArrayXf::Ones(n);

    MarsAlgo algo(X.data(), y.data(), w.data(), n, m, 300, n);

    int p = 20;
    Timer timer;
    for (int i = 0; i < p; ++i) {
        algo.append('l', i, 0, 0);
    }


    ArrayXd linear_dsse(p);
    ArrayXd hinge_dsse(p);
    ArrayXd hinge_cuts(p);
    ArrayXb mask = ArrayXb::Ones(p);

    timer.tic();
    algo.eval(linear_dsse.data(), hinge_dsse.data(), hinge_cuts.data(),
              142, mask.data(), 0, 0);
    printf("%.2f\n",timer.toc()*1e-6);
}

void bench_covariates()
{
    int n = 100 * 1000;
    int m = 500;

    MatrixXfC X  = MatrixXfC::Random(n,m);
    ArrayXd   f0 = ArrayXd::Zero(m+1);
    ArrayXd   g0 = ArrayXd::Zero(m+1);
    ArrayXd   f1 = ArrayXd::Zero(m+1);
    ArrayXd   g1 = ArrayXd::Zero(m+1);
    VectorXd  y  = VectorXd::Random(m);
    MatrixXf  k  = MatrixXf::Random(n,4);

    Timer timer;
    uint64_t dt = 0;

    for (int i = 0; i < X.rows(); ++i) {
        timer.tic();
        covariates(f0, g0, X.row(i).data(), y.data(), k(i,0), k(i,1), k(i,2), k(i,3),m);
        dt += timer.toc();
    }
    printf("covariates: %.2f\n", double(dt)/double(X.rows()));
}

int main()
{
    bench_covariates();
    bench_marsalgo();
}
