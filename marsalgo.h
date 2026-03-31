#include <stdexcept>

inline void verify(bool check, const char *msg)
{
    if (!check) {
        throw std::runtime_error(msg);
    }
}

class MarsAlgo {
    struct MarsData *_data = nullptr;
    int     _m    = 1;  // number of basis found
    double  _yvar = 0;  // variance of 'y'
    double  _tol  = 0;  // numerical error tolerance

public:
    MarsAlgo(const float *x, const float *y, const float *w, int n, int m, int p, int ldx);
    ~MarsAlgo();

    int nbasis() const;
    double dsse() const;
    double yvar() const;

    /*
     *  Returns the delta SSE (sum of squared errors) given the existing basis
     *  set and a candidate column of `X` to evaluate.
     *
     *  linear_dsse : [out]
     *
     *  hinge_dsse : [out]
     *
     *  hinge_cuts : [out]
     *
     *  xcol : int
     *      which column of the training `X` data to use.
     *
     *  bmask : bool(m)
     *      a boolean mask to filter out which bases to use.
     *
     *  endspan : int
     *      how many samples to ignore from both the extreme ends of the
     *      training data.
     *
     *  linear_only : bool
     *      do not attempt to find any hinge cuts, only build a linear model.
     *      This will ignore the input values of `hinge_sse` and `hinge_cut`
     */
    void eval(double *linear_dsse, double *hinge_dsse, double *hinge_cuts,
              int xcol, const bool *bmask, int endspan, bool linear_only);

    double append(char type, int xcol, int bcol, float h);
};
