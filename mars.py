#!/usr/bin/env python3
"""
Multivariate Adaptive Regression Splines
"""
import time
import numba
import numpy as np
import marslib

# pylint: disable=consider-using-f-string
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=unnecessary-lambda-assignment

# -----------------------------------------------------------------------------


def _dump_header(logger):
    if logger:
        logger.write("time        #   n    b    x   o    r²    r²_cv\n")
        logger.write("---------- --- --- ---- ---- --  ------ -------\n")


def _dump_row(logger, epoch, nbasis, dt, row, labels):
    logger.write("%02d:%02d:%04.1f " % (dt // 3600, (dt // 60) % 60, dt % 60))
    logger.write(
        "%3d %3d %4d %4d %2d  %6.4f %7.4f"
        % (
            epoch,
            nbasis - 1,
            row["basis"],
            row["input"],
            row["order"],
            row["r2"],
            row["r2_cv"],
        )
    )

    if labels is not None:
        fmt = "%%-%ds" % max(len(_) for _ in labels)
        if row["type"] == b"l":
            logger.write(("  l " + fmt) % labels[row["input"]])
        else:
            logger.write(
                ("  h " + fmt + " %.4g") % (labels[row["input"]], row["hinge"])
            )

    logger.write("\n")


# -----------------------------------------------------------------------------


def fit(X, y, w=None, **kwargs):
    """
    Multivariate Adaptive Regression Splines

    A flexible regression method that automatically searches for interactions
    and non-linear relationships.

    Parameters
    ----------
    max_epochs : int (default=min(p + n/20, 15)), where X is (n x p)
        The maximum number of epochs in the forward pass.

    max_degree : int (default=3)
        The maximum degree of terms generated by the forward pass.

    max_basis : int (default=max_epochs*2)
        For Fast-MARS, the number of bases to keep in cache.

    max_inputs : int (default=p), where X is (n x p)
        For Fast-MARS, the number of input features to keep in cache.

    penalty : float (default=3.0)
        A smoothing parameter used to calculate GCV R².
        Used during the pruning pass and to determine whether to add a hinge
        or linear basis function during the forward pass.
        See the 'd' parameter in equation 32, Friedman, 1991.

    tail_span : float (default=0.05)
        Fraction of samples at the tails of the training data to ignore.
        Prevents hinge formation at the extreme tails of the input range.

    max_runtime : float (default=48*3600)
        Maximum runtime in seconds.

    self_interact : bool (default=False)
        Allow an input to interact with itself. This may cause some numerical
        instability and is best modelled with a linear hinge instead.

    linear_only : bool (default=False)
        Disable formation of any hinges. This will still find interactions
        of inputs.

    n_true : int (default=n), where X is (n x p)
        Allows for an adjustment in the number of truly independent samples
        in the data.

    logger : file-like (default=None)
        A file-like object which can be used to log runtime messages.

    callback : callable (default=None)
        Can be used to get progress updates on the following:
          * current epoch
          * number of basis found
          * total max epochs to run
          * a dict with information about the last basis found

    r2_window : int (default=16)
        Length of the rolling window in which a smoothed delta R² is measured.
        This is used a stopping criteria.

    r2_thresh : float (default=3e-5)
        Stop the fit if the GCV R² increases less than this. Used in conjunction
        with 'r2_window'.

    aging_factor : float (default=1.0)
        Used to increase visibility of features not used in a while. See Eq 27
        in the Fast-MARS paper.

    labels : list-like (default=None)
        A list of column labels. This is only used for console output.

    xfilter : function (default=lambda x,b: True)
        Allows custom filtering of inputs.

    threads : int (default=-1)
        Number of cores to use. A negative number implies usage of all cores.
    """

    X = np.asarray(X)  # do not make a copy of this data!
    y = np.asarray(y, dtype="f")
    w = np.asarray(w, dtype="f") if w is not None else np.ones(len(X), dtype="f")
    assert X.dtype == "f", "X data must be 32-bit float"
    assert X.strides[0] == X.itemsize, "X data must be column-major"
    assert y.strides[0] == y.itemsize, "y data must be column-major"
    assert len(X) == len(y) == len(w)
    assert len(X) > 0, "empty dataset"

    # fmt: off
    max_epochs    = kwargs.pop('max_epochs', min(X.shape[1]+len(X)//20, 15))
    max_degree    = kwargs.pop('max_degree', 3)
    penalty       = kwargs.pop('penalty', 3.0)
    tail_span     = kwargs.pop('tail_span', 0.05)
    max_runtime   = kwargs.pop('max_runtime', 48*3600) # in seconds
    self_interact = kwargs.pop('self_interact', False)
    linear_only   = kwargs.pop('linear_only', False)
    n_true        = kwargs.pop('n_true', len(X))
    logger        = kwargs.pop('logger', None)
    callback      = kwargs.pop("callback", None)
    r2_window     = kwargs.pop('r2_window', 16) # window over which to measure R2
    r2_thresh     = kwargs.pop('r2_thresh', 3e-5)
    labels        = kwargs.pop('labels', None)
    aux_filter    = kwargs.pop('xfilter', lambda x,b: True)
    max_basis     = kwargs.pop('max_basis', max_epochs*2) # for "Fast MARS"
    max_inputs    = kwargs.pop('max_inputs', X.shape[1])  # for "Faster MARS v2"
    aging_factor  = kwargs.pop('aging_factor', 1.0)
    threads       = kwargs.pop('threads', -1)
    # fmt: on

    if kwargs:
        raise TypeError("unknown argument: %s" % kwargs)

    start_t = time.time()

    # Equation numbers refer to the original MARS paper
    get_dof = lambda m: m + penalty * (m - 1)  # Eq. (31,32)
    gcv_adj = lambda mse, m: mse / (1.0 - get_dof(m) / n_true) ** 2  # Eq. (30)
    avg_diff = lambda x, n: np.sort(np.diff(x))[1:-1].mean() if len(x) > n else np.nan

    # Set up a basic filter which caps the polynomial degree of basis and optionally
    # prevents features from interacting with themselves. Here 'i' is the index of the
    # feature to be added and 'b' is a list of features that exist in the parent basis.
    basic_filter = lambda i, b: (len(b) < max_degree) and (
        self_interact or (i not in b)
    )

    # Make sure the DOF's never exceed the number of samples
    max_terms = 1 + 2 * max_epochs
    i = np.arange(1, max_terms + 1)
    max_terms = i[get_dof(i) < n_true].max()

    n = len(X)
    basis = [[]]  # list of lists of used basis
    tail = max(int(n * tail_span), 1)
    algo = marslib.MarsAlgo(X, y, w, max_terms)  # pylint: disable=c-extension-no-member
    var_y = algo.yvar()

    # Set up the SSE caches. Note that for inputs, we initialize to +inf, so that
    # we cover all inputs, even if they are temporarily masked. We then need to
    # adjust the 'max_inputs' dynamically to account for these inf values.
    epoch = 0
    basis_sse = np.array([0.0])
    input_sse = np.full(X.shape[1], np.inf)
    input_age = np.full(X.shape[1], epoch)

    # Define output data structure
    # fmt: off
    model = np.zeros(
        max_terms,
        dtype=[
            ("type",  "S1"),
            ("basis", "i4"),
            ("input", "i4"),
            ("hinge", "f8"),
            ("r2",    "f4"),
            ("r2_cv", "f4"),
            ("order", "i4"),
            ("time",  "f4"),
        ],
    )
    model[0] = ("i", 0, 0, np.nan, 0, 0, 0, 0)  # add the intercept
    # fmt: on

    _dump_header(logger)

    def _ranks(x):
        y = np.empty(len(x), dtype="l")
        y[np.argsort(x)] = np.arange(len(x))
        return y

    while algo.nbasis() < max_terms and epoch < max_epochs:
        # "Fast MARS" - rank the basis and inputs by delta SSE
        # contribution and take only the most promising ones.
        basis_to_use = np.argsort(basis_sse)[::-1][:max_basis]
        input_to_use = np.argsort(
            _ranks(input_sse) + aging_factor * (epoch - input_age) * (input_sse > 0)
        )
        input_to_use = input_to_use[::-1][: (max_inputs + np.isinf(input_sse).sum())]
        input_to_use = np.sort(input_to_use)

        # Build up the mask block here
        bmask = np.zeros((X.shape[1], len(basis)), dtype="bool")
        bmask[np.ix_(input_to_use, basis_to_use)] = True
        for i in input_to_use:
            bmask[i] &= np.array([basic_filter(i, b) for b in basis])
            bmask[i] &= np.array([aux_filter(i, b) for b in basis])

        # Find the delta-SSE for the entire block
        # 'sse1' is the improvement by adding a linear term
        # 'sse2' is the improvement by adding two disjoint hinges
        sse0, sse1, sse2, cut = algo.eval(bmask, tail, linear_only, threads)

        # Update the delta-SSE cache
        # TODO - should we really be using GCV adjusted SSE instead?
        basis_sse[bmask.any(axis=0)] = sse2[:, bmask.any(axis=0)].max(axis=0)
        input_sse[input_to_use] = sse2[input_to_use].max(axis=1)
        input_age[input_to_use] = epoch
        epoch += 1

        j1 = np.unravel_index(np.argmax(sse1), sse1.shape)
        j2 = np.unravel_index(np.argmax(sse2), sse2.shape)
        dt = time.time() - start_t
        if sse1[j1] <= 0 and sse2[j2] <= 0:
            break  # all input data is filtered or zero

        # Estimate the out-sample error with Generalized Cross-Validation
        m0 = m = algo.nbasis()
        mse1 = gcv_adj((1.0 - sse0 - sse1[j1]) / n, m + 1)
        mse2 = gcv_adj((1.0 - sse0 - sse2[j2]) / n, m + 2)
        if mse1 <= mse2:
            xcol, bcol, hcut = j1[0], j1[1], np.nan
            htypes = ["l"]
        else:
            xcol, bcol, hcut = j2[0], j2[1], cut[j2]
            htypes = ["+", "-"]

        for htype in htypes:
            mse = algo.append(htype, xcol, bcol, hcut)

            # If 'y' is noise-free then MSE might be 0.0; handle this edge case.
            if mse >= 0:
                # fmt:off
                basis.append(basis[bcol] + [xcol])
                model[m] = (
                    htype,                              # type
                    bcol,                               # basis
                    xcol,                               # input
                    hcut,                               # hinge
                    1.0 - mse / var_y,                  # r2
                    1.0 - gcv_adj(mse, m + 1) / var_y,  # r2_cv
                    len(basis[-1]),                     # order
                    dt,                                 # time
                )
                basis_sse = np.append(basis_sse, [0.0])
                m += 1
                # fmt:on
        assert algo.nbasis() == len(basis) == m

        if logger:
            _dump_row(logger, epoch, len(basis), dt, model[m - 1], labels)
        if callback:
            callback(epoch, len(basis), max_epochs, model[m - 1])

        # Stopping conditions
        model_tail = model[max(algo.nbasis() - r2_window - 1, 0) : algo.nbasis()]
        if m == m0:
            break  # no progress
        if model_tail[-1]["r2"] > 1 - r2_thresh:
            break  # R2 almost reached 100%
        if dt > max_runtime:
            break  # exceed max runtime
        if avg_diff(model_tail["r2_cv"], r2_window) < r2_thresh:
            break  # no progress in GCV R2

    return model[: algo.nbasis()]


# -----------------------------------------------------------------------------


@numba.njit(parallel=True)
def _expand_linear_basis(y, b, x):
    for i in numba.prange(len(y)):  # pylint: disable=E1133
        y[i] = b[i] * x[i]


def expand(X, model):
    """
    Expand a feature set X into the bases of a MARS model.
    """
    X = np.asarray(X)
    B = np.empty((len(X), len(model)), dtype=X.dtype, order="F")
    decode = lambda s: s.decode("utf8") if isinstance(s, bytes) else s

    for i, node in enumerate(model):
        t = decode(node["type"])
        b = node["basis"]
        x = node["input"]
        h = node["hinge"]

        if t == "i":
            B[:, i] = 1.0
        else:
            assert 0 <= x < X.shape[1]
            assert 0 <= b < i
            if t == "l":
                _expand_linear_basis(B[:, i], B[:, b], X[:, x])
            elif t == "+":
                B[:, i] = B[:, b] * np.maximum(X[:, x] - h, 0)
            elif t == "-":
                B[:, i] = B[:, b] * np.maximum(h - X[:, x], 0)
            else:
                assert False
    return B


# -----------------------------------------------------------------------------


def prune(XX, XY, YY, n_true, penalty=3, ridge=0, mask=None):
    """
    Solve for the linear coefficients, with pruning.
    """

    def _solve(xx, xy, mask):
        # You must use 'lstsq' so we can handle under-determined problems
        beta = np.zeros(len(xy))
        if mask.any():
            xx = xx[np.ix_(mask, mask)]
            xy = xy[mask]
            beta[mask] = np.linalg.lstsq(xx, xy, rcond=None)[0]
        return beta

    def _gcv_sse(xx, xy, yy, mask, dof, n_true):
        sse = yy - np.dot(xy, _solve(xx, xy, mask))
        return sse / (1.0 - dof / n_true) ** 2

    M = len(XX)
    mask = np.array(mask) if mask is not None else np.ones(M, dtype="bool")
    mask = mask & (np.diag(XX) > 0)
    dof = M + penalty * (M - 1)
    min_sse = _gcv_sse(XX, XY, YY, mask, dof, n_true)

    while True:
        sse = np.ones(M) * np.inf
        for i in np.where(mask)[0]:
            k = mask & (np.arange(len(mask)) != i)
            dof = sum(k) + penalty * (M - 1)
            sse[i] = _gcv_sse(XX, XY, YY, k, dof, n_true)
        if sse.min() >= min_sse:
            break
        mask[np.argmin(sse)] = False
        min_sse = sse.min()
        assert np.isfinite(min_sse)

    if ridge > 0:
        assert np.allclose(np.diag(XX)[mask], np.ones(mask.sum()))
        XX = XX + np.eye(len(XX)) * ridge
    return _solve(XX, XY, mask)


# -----------------------------------------------------------------------------


def pprint(model, beta=None, labels=None):
    """
    Pretty-print the model. Useful for debugging.
    """

    def xcol(i):
        if labels is not None:
            return labels[i]
        return "X[%d]" % i

    def get_inputs(i: int):
        row = model[i]
        if row["type"] == b"+":
            node = "MAX(%s-%g,0)" % (xcol(row["input"]), row["hinge"])
        elif row["type"] == b"-":
            node = "MAX(%g-%s,0)" % (row["hinge"], xcol(row["input"]))
        else:
            node = xcol(row["input"])

        if row["basis"] > 0:
            return [node] + get_inputs(row["basis"])
        return [node]

    for i in range(len(model)):  # pylint: disable=consider-using-enumerate
        bstr = "  "
        if beta is not None and hasattr(beta, "__getitem__"):
            bstr += "%+9.4g" % beta[i]
        if model[i]["type"] != b"i":
            bstr += " * " + " * ".join(get_inputs(i))
        print(bstr)
