#!/usr/bin/env python3
"""
Multivariate Adaptive Regression Splines
"""

import time
import numpy as np
import marslib

# -----------------------------------------------------------------------------

def _dump_logger_header(logger):
    if logger:
        logger.write('time       #   n    b    x   o   r2    r2_cv\n')
        logger.write('--------- --- --- ---- ---- -- ------ -------\n')


def _dump_logger_row(logger, epoch, nbasis, dt, row, labels):
    if logger:
        logger.write('%02d:%02d:%04.1f ' % (dt//3600,(dt//60)%60,dt%60))
        logger.write('%3d %3d %4d %4d %2d  %6.4f %7.4f'%
                     (epoch, nbasis-1, row['basis'], row['input'], row['order'], row['r2'], row['r2_cv']))

        if labels is not None:
            fmt = '%%-%ds'%max([len(_) for _ in labels])
            if row['type'] == b'l':
                logger.write(('  l '+fmt) % labels[row['input']])
            else:
                logger.write(('  h '+fmt+' %.4g') % (labels[row['input']], row['hinge']))

        logger.write('\n')

# -----------------------------------------------------------------------------

def fit(X, y, w=None, **kwargs):

    X = np.asarray(X)
    y = np.asarray(y, dtype='f')
    w = np.asarray(w, dtype='f') if w is not None else np.ones(len(X), dtype='f')
    assert X.strides[0] == X.itemsize, "must be column-major"
    assert y.strides[0] == y.itemsize, "must be column-major"
    assert len(X) == len(y) == len(w)
    assert len(X) > 0, "empty dataset"

    max_epochs    = kwargs.pop('max_epochs', min(X.shape[1]+len(X)//20, 15))
    max_degree    = kwargs.pop('max_degree', 3)
    penalty       = kwargs.pop('penalty', 3.0)
    tail_span     = kwargs.pop('tail_span', 0.05)
    max_runtime   = kwargs.pop('max_runtime', 48*3600) # in seconds
    self_interact = kwargs.pop('self_interact', False)
    linear_only   = kwargs.pop('linear_only', False)
    n_true        = kwargs.pop('n_true', len(X))
    logger        = kwargs.pop('logger', None)
    r2_window     = kwargs.pop('r2_window', 16) # window over which to measure R2
    r2_thresh     = kwargs.pop('r2_thresh', 3e-5)
    labels        = kwargs.pop('labels', None)
    aux_filter    = kwargs.pop('xfilter', lambda x,b: True)
    max_basis     = kwargs.pop('max_basis', max_epochs*2)
    max_inputs    = kwargs.pop('max_inputs', X.shape[1])
    aging_factor  = kwargs.pop('aging_factor', 1.0)

    if kwargs:
        raise TypeError('unknown argument: %s' % kwargs)

    start_t  = time.time()
    get_dof  = lambda m: m+penalty*(m-1)  # Eq. (31,32) in original MARS paper
    gcv_adj  = lambda mse,dof: mse/(1.-dof/n_true)**2  # Eq. (30) in original MARS paper
    avg_diff = lambda x,n: np.sort(np.diff(x))[1:-1].mean() if len(x)>n else np.nan

    # Set up a basic filter which caps the polynomial degree of basis and optionally
    # prevents features from interacting with themselves. Here 'i' is the index of the
    # feature to be added and 'b' is a list of features that exist in the parent basis.
    basic_filter = lambda i,b: (len(b) < max_degree) and (self_interact or (i not in b))

    # Make sure the DOF's never exceed the number of samples
    max_terms = 1+2*max_epochs
    i = np.arange(1,max_terms+1)
    max_terms = i[get_dof(i) < n_true].max()

    n     = len(X)
    basis = [list()] # list of lists of used basis
    tail  = max(int(n*tail_span),1)
    algo  = marslib.MarsAlgo(X,y,w,max_terms)
    var_y = algo.yvar()

    # Set up the SSE caches. Note that for inputs, we initialize to +inf, so that
    # we cover all inputs, even if they are temporarily masked. We then need
    # to adjust the 'max_inputs' dynamically to account for these inf values.
    epoch = 0
    basis_sse = [0]
    input_sse = np.full(X.shape[1],np.inf)
    input_age = np.full(X.shape[1],epoch)

    # Define output data structure
    model = np.zeros(max_terms, dtype=[
        ('type','S1'), ('basis','i4'), ('input','i4'), ('hinge','f8'),
        ('r2','f4'), ('r2_csv','f4'), ('order','i4'), ('time','f4'),
    ])
    model[0] = ('i',0,0,np.nan,0,0,0,0) # add the intercept

    _dump_logger_header(logger)

    def _ranks(x):
        y = np.empty(len(x))
        y[np.argsort(x)] = np.arange(len(x))
        return y

    while algo.nbasis() < max_terms and epoch < max_epochs:
        results = [] # delta-SSE results for new potential candidates
        basis_to_use = np.argsort(basis_sse)[::-1][:max_basis] # "Fast MARS" tweak
        input_to_use = np.argsort(_ranks(input_sse) + aging_factor*(epoch-input_age)*(input_sse>0))[::-1]

        for i in basis_to_use:
            basis_sse[i] = -np.inf # reset cache, so that max() works later on

        inputs_used = 0
        for i in input_to_use:
            mask = [(basic_filter(i,b) and aux_filter(i,b)) for b in basis]
            mask = np.intersect1d(np.nonzero(mask)[0], basis_to_use)

            if len(mask) > 0:
                inputs_used += np.isfinite(input_sse[i]) # don't count unseen inputs
                byyb,sse1,sse2,cut = algo.dsse(i,mask,tail,linear_only)

                # TODO - we should use GCV adjusted SSE here
                for j, sse2_j in zip(mask,sse2):
                    basis_sse[j] = max(sse2_j,basis_sse[j])
                input_sse[i] = sse2.max()
                input_age[i] = epoch

                j1 = sse1.argmax() # SSE improvement by adding a linear term
                j2 = sse2.argmax() # SSE improvement by adding two disjoint hinges
                if sse1[j1] > 0:
                    results.append((i,mask[j1],np.nan,1,(1.-byyb-sse1[j1])/n))
                if sse2[j2] > 0:
                    results.append((i,mask[j2],cut[j2],2,(1.-byyb-sse2[j2])/n))

                if inputs_used >= max_inputs:
                    break
        epoch += 1
