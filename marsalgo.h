#include <Eigen/Dense>
#include <numeric> // for std::iota
#include <cfloat>  // for DBL_EPSILON

using namespace Eigen;
typedef Matrix<double,Dynamic,Dynamic,RowMajor> MatrixXdC;
typedef Array<int32_t,Dynamic,1> ArrayXi32;
typedef Array<int64_t,Dynamic,1> ArrayXi64;

///////////////////////////////////////////////////////////////////////////////
///  Return indices in REVERSED order
///////////////////////////////////////////////////////////////////////////////
void argsort(int32_t *idx, const float *v, int n) {
    std::iota(idx, idx+n, 0);
    std::sort(idx, idx+n, [&v](size_t i, size_t j) { return v[i] > v[j]; });
}

///////////////////////////////////////////////////////////////////////////////
///  Ortho-normalize via inverse Cholesky method
///////////////////////////////////////////////////////////////////////////////
void orthonormalize(Ref<MatrixXd> Bx, const Ref<MatrixXf> &B, const Ref<MatrixXdC> &Bo,
                    const ArrayXf &x, const ArrayXi64 &mask, double tol)
{
    assert(B.cols()  == Bo.cols() && Bx.rows() == B.rows());
    assert(Bx.cols() == mask.rows());

    for (int j = 0; j < mask.rows(); ++j) {
        Bx.col(j) = (B.col(mask[j]).array() * x).cast<double>();
    }

    const MatrixXd h = Bo.transpose() * Bx;
    const ArrayXd  s = (Bx.colwise().squaredNorm() - h.colwise().squaredNorm()).array() + tol;

    Bx -= Bo * h;
    Bx *= (s > 0).select(1/s.sqrt(), 0).matrix().asDiagonal();
}

///////////////////////////////////////////////////////////////////////////////
///  Sort columns in-place
///////////////////////////////////////////////////////////////////////////////
void sort_columns(Ref<MatrixXd> X, const ArrayXi32 &k)
{
    VectorXd tmp;
    for (int j = 0; j < X.cols(); ++j) {
        tmp = X.col(j); // make a copy
        Ref<VectorXd> x = X.col(j); // keep a reference
        for (int i = 0; i < x.rows(); ++i) {
            tmp[i] = x[k[i]];
        }
        X.col(j) = tmp;
    }
}

void covariates(double *ff_, double *fy_, ArrayXd &f, ArrayXd &g,
    const Ref<VectorXd> &x, const VectorXd &y, double k0, double k1)
{
    assert(x.cols()==1 && x.rows()>=x.cols());
    assert(x.rows()==y.rows());

    const int m = x.rows();
    double s0 = 0;
    double s1 = 0;

    static_assert(FP_FAST_FMA, "-mfma must be enabled");
    for (int i = 0; i < m; ++i) {
        f[i] = fma(k0,g[i],f[i]);
        g[i] = fma(k1,x[i],g[i]);
        s0   = fma(f[i],f[i],s0);
        s1   = fma(f[i],y[i],s1);
    }
    *ff_ = s0;
    *fy_ = s1;
}

///////////////////////////////////////////////////////////////////////////////

class MarsAlgo {

    // TODO - all large scratch buffers should be 32 bit

    Map<const MatrixXf,Aligned,Stride<Dynamic,1>>  _X;   // read-only view of the regressors
    ArrayXd     _y;         // target vector
    MatrixXf    _B;         // all basis
    MatrixXdC   _Bo;        // all basis, ortho-normalized
    MatrixXdC   _Bok;       // all basis, ortho-normalized and sorted (scratch buffer)
    MatrixXd    _Bx;        // B[k,mask]*x[k,None] (scratch buffer)
    ArrayXf     _s;         // scale of columns of 'X'
    int         _m    = 1;  // number of basis found
    double      _yvar = 0;  // variance of 'y'
    double      _tol  = 0;  // numerical error tolerance

public:
    MarsAlgo(const float *x, const float *y, const float *w, int n, int m, int p, int ldx)
       : _X  (x,n,m,Stride<Dynamic,1>(ldx,1))
       , _B  (MatrixXf ::Zero(n,p))
       , _Bo (MatrixXdC::Zero(n,p))
       , _Bok(MatrixXdC::Zero(n,p))
       , _Bx (MatrixXdC::Zero(n,p))
       , _tol((n*0.02)*DBL_EPSILON) // rough guess
    {
        if (std::isfinite(NAN)) {
            throw std::runtime_error("NAN check is disabled, recompile without --fast-math");
        }
        ArrayXd vv;
        _y = Map<const ArrayXf>(y,n).cast<double>();  // copy and upcast
        vv = Map<const ArrayXf>(w,n).cast<double>();  // copy and upcast

        //---------------------------------------------------------------------
        // Filter out NAN's and apply weight to the target 'y'.
        //---------------------------------------------------------------------
        for (int i = 0; i < n; ++i) {
            if (!std::isfinite(y[i])) { _y[i] = vv[i] = 0; }
        }
        _y *= vv; // apply weight to target

        _y /= _y.matrix().norm();
        vv /= vv.matrix().norm();

        //---------------------------------------------------------------------
        // Initialize the first column of our basis with the intercept
        //---------------------------------------------------------------------
        _B .col(0) = vv.cast<float>();
        _Bo.col(0) = vv;

        //---------------------------------------------------------------------
        // Calculate the sample variance of the target 'y'.
        //---------------------------------------------------------------------
        _yvar = (_y - _y.mean()).square().mean();

        //---------------------------------------------------------------------
        // Calculate the column norm of 'X'.
        //---------------------------------------------------------------------
        _s = (_X.colwise().squaredNorm()/_X.rows()).cwiseSqrt();
        _s = (_s > 0.f).select(1.f/_s, 1.f);
    }

    ///////////////////////////////////////////////////////////////////////////

    void dsse(double *linear_sse, double *hinge_sse, double *hinge_cut,
              int xcol, const int64_t *mask, int p, int endspan, bool linear_only)
    {
        if (p > _m) {
            throw std::runtime_error("invalid mask array");
        }

        const int n = _X.rows();
        const int m = _m;
        ArrayXf   x = _X.col(xcol) * _s[xcol]; // normalize 'x' column
        Ref<MatrixXd>  Bx  = _Bx.leftCols(p);
        Ref<MatrixXdC> Bo  = _Bo.leftCols(m);
        Ref<MatrixXdC> Bok = _Bok.leftCols(m);

        //---------------------------------------------------------------------
        // Ortho-normalize via inverse Cholesky method
        //---------------------------------------------------------------------
        orthonormalize(Bx, _B.leftCols(m), Bo, x, Map<const ArrayXi64>(mask,p), _tol);

        //---------------------------------------------------------------------
        // Calculate the linear delta SSE
        //---------------------------------------------------------------------
        VectorXd yb  = (Bo.transpose() * _y.matrix());
        ArrayXd  yb2 = (Bx.transpose() * _y.matrix()).array();
        Map<ArrayXd>(linear_sse,p) = (yb.transpose()*yb) + yb2.square();

        //---------------------------------------------------------------------
        // Evaluate the delta SSE on all hinge locations
        //---------------------------------------------------------------------
        if (linear_only == false) {
            ArrayXi hinge_idx(p);
            for (int j = 0; j < p; ++j) {
                hinge_idx[j] = -1;
                hinge_sse[j] =  0;
            }

            //-----------------------------------------------------------------
            // Get sort indices
            //-----------------------------------------------------------------
            ArrayXi32 k(n);
            argsort(k.data(), _X.col(xcol).data(), n);

            //-----------------------------------------------------------------
            // Sort all rows of y, Bo, Bx and take the deltas of x
            //-----------------------------------------------------------------
            VectorXd yk(n);
            for (int i = 0; i < n; ++i) {
                yk[i] = _y[k[i]];
                Bok.row(i) = Bo.row(k[i]);
            }
            sort_columns(Bx, k);

            ArrayXd _d(n-1);
            double *d = _d.data()-1; // note minus-one hack
            for (int i = 1; i < n; ++i) {
                d[i] = x[k[i-1]] - x[k[i]];
            }

            const int head = endspan;
            const int tail = n-endspan;
            static_assert(FP_FAST_FMA, "-mfma must be enabled");

            for (int j = 0; j < p; ++j) {
                const Ref<VectorXf> b  = _B.col(mask[j]);
                const Ref<VectorXd> bx = Bx.col(j);

                double b_i = b[k[0]]; // sort and upscale to double
                double k0 = 0;
                double k1 = 0;
                double w  = 0;
                double bd = 0;
                double vb = b_i*yk[0];
                double b2 = b_i*b_i;
                ArrayXd f = ArrayXd::Zero(m+1);
                ArrayXd g = ArrayXd::Zero(m+1);
                double ff = 0;
                double fy = 0;

                covariates(&ff,&fy,f,g,Bok.row(0),yb,0,b_i);
                g[m] += b_i*bx[0];

                for (int i = 1; i < tail; ++i) {
                    b_i = b[k[i]]; // sort and upcast to double
                    covariates(&ff,&fy,f,g,Bok.row(i),yb,d[i],b_i);
                    f[m] = fma(d[i],g[m],f[m]);
                    g[m] = fma(b_i,bx[i],g[m]);
                    ff   = fma(f[m],f[m],ff);
                    fy   = fma(f[m],yb2[j],fy);

                    k0 = fma(d[i]*d[i],b2,k0);
                    k1 = fma(d[i]*2,bd,k1);
                    w  = fma(d[i],vb,w);
                    bd = fma(d[i],b2,bd);
                    b2 = fma(b_i, b_i,b2);
                    vb = fma(yk[i],b_i,vb);

                    const double uw  = fy - w;
                    const double den = (k0+k1) - ff;
                    const double sse = den > _tol? (uw*uw)/(den+_tol) : 0;
                    if ((i > head) and (sse > hinge_sse[j])) {
                        hinge_sse[j] = sse;
                        hinge_idx[j] = i;
                    }
                }
            }

            for (int j = 0; j < p; ++j) {
                hinge_sse[j] += linear_sse[j];
                hinge_cut[j] = hinge_idx[j]>=0? x[k[hinge_idx[j]]]/_s[xcol] : NAN;
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////

    double append(char type, int xcol, int bcol, float h) {
        if (_m >= _B.cols()) {
            throw std::runtime_error("basis matrix is full");
        }

        const float  s = _s[xcol];
        Ref<ArrayXf> b = _B.col(bcol).array();
        Ref<const ArrayXf> x = _X.col(xcol).array();

        switch(type) {
            case 'l':
                _B.col(_m) = (s*b*x);
                break;
            case '+':
                _B.col(_m) = (s*b*(x-h).cwiseMax(0));
                break;
            case '-':
                _B.col(_m) = (s*b*(h-x).cwiseMax(0));
                break;
            default:
                throw std::runtime_error("invalid basis type");
        }

        //---------------------------------------------------------------------
        // Use Gram-Schmidt orthogonalization.
        //---------------------------------------------------------------------
        VectorXd v = _B.col(_m).cast<double>(); // make a copy
        _B.col(_m) /= v.norm();

        for (int j = 0; j < _m; ++j) {
            v -= (_Bo.col(j).transpose()*v) * _Bo.col(j);
        }

        const double w = v.norm();
        if (w*w > _tol) {
            _Bo.col(_m) = v/w;

            // This assumes the norm of '_y' == 1
            VectorXd yb = _Bo.leftCols(_m+1).transpose() * _y.matrix();
            const double mse = (1. - yb.squaredNorm()) / _X.rows();
            if (mse >= 0.) {
                ++_m;
                return mse;
            }
        }
        return -1.;
    }
};
