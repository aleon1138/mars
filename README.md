# MARS

A C++/CUDA implementation of [Multivariate Adaptive Regression Splines](https://w.wiki/GVPL),
a semi-brute-force search for interactions and non-linearities. It can deliver
regression performance competitive with neural networks, while providing inference
performance competitive with linear models.

References:

* [Write-up describing the method](https://uc-r.github.io/mars)
* [Commercial MARS package](https://www.salford-systems.com/products/mars) by Salford Systems
* [R "earth" package documentation](https://cran.r-project.org/web/packages/earth/earth.pdf)
* [Stephen Milborrow's resource page](http://www.milbo.users.sonic.net/earth)
* A scikit-learn implementation, [py-earth](https://contrib.scikit-learn.org/py-earth)

## Performance

We use [OpenMP](https://www.openmp.org) to parallelize the forward pass across
cores. Each thread carries some memory overhead, which can limit how many cores
one can use in practice. Control the thread count with the `OMP_NUM_THREADS`
environment variable or the `threads` argument.

The timings below were measured on an AMD EPYC 9654 (96 cores, 192 logical
CPUs). Multi-threaded scaling is nearly ideal up to about 30 cores.

![Performance timings of MARS](timings.png)

## Supported Platforms

The build has been verified on the following platforms:

* Ubuntu 18.04 through 24.04
* macOS High Sierra (10.13) through Tahoe (26.4)
* Raspbian 10

## Build Requirements

* [CMake](https://cmake.org/) ≥ 3.18
* A C++17 compiler with OpenMP support
* *optional* — GPU builds only:
  * [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) with `nvcc`
* *optional* - unit tests only:
  * [Eigen](http://eigen.tuxfamily.org/) ≥ 3.3
  * [GoogleTest](https://github.com/google/googletest)

```bash
# Linux — library needs only cmake; eigen + gtest are for the tests:
sudo apt install -y cmake libeigen3-dev libgtest-dev
# macOS — libomp is for the library; eigen + googletest are for the tests:
brew install cmake libomp eigen googletest
```

`pip install .` pulls `pybind11` and `scikit-build-core` automatically as build
dependencies — no need to install them by hand.

On macOS, `libomp` is statically linked into `marslib.so` so the OpenMP
runtime stays private to this extension. This avoids the "OMP: Error #15:
libomp.dylib already initialized" crash that fires whenever the Python
interpreter already loaded a different `libomp.dylib` via another extension
(numpy/scipy/sklearn/...). No `KMP_DUPLICATE_LIB_OK=TRUE` workaround needed.

## Build Instructions

CMake drives the build — you never invoke `cmake` by hand (except for the
sanitizer builds noted at the top of the `Makefile`). Both `pip install .` and
`make` configure *and* build for you. There is no `setup.py` as packaging is
driven by `pyproject.toml` via the
[scikit-build-core](https://scikit-build-core.readthedocs.io) backend.

### CPU build (default)

A plain `pip install .` or `make` produces a CPU-only library.

```bash
cd mars
pip install .
```

For local development, the Makefile is a thin wrapper around CMake:

```bash
cd mars
make            # configures + builds; drops marslib*.so at the source root
make test       # runs the C++ unit tests and pytest
make clean      # removes the build/ directory and built .so
```

### GPU build (opt-in, CUDA)

The GPU orthonormalize kernels are gated behind the `USE_CUDA` option,
which is disabled by default. Turn it on to build them into `marslib.so`:

```bash
# wheel: pass the CMake define through pip
pip install . -C cmake.define.USE_CUDA=ON

# local dev: one-shot GPU build (run `make clean` first to switch a CPU build
# dir over, and again to switch back)
make cuda
```

A GPU build additionally needs the CUDA Toolkit (`nvcc`). It targets compute
capabilities 7.5 and 12.0 by default; pin a single target with
`-DCMAKE_CUDA_ARCHITECTURES=<arch>` (append
`-C cmake.define.CMAKE_CUDA_ARCHITECTURES=<arch>` to the pip command).

Only a GPU build can run the forward pass on the device — pass `cuda=True` to
`mars.fit` to use it (most effective in the `linear_only` regime). The module
exposes no capability flag, so the only way to tell a CPU and a GPU build apart
at runtime is behavioral: `fit(..., cuda=True)` on a CPU-only build raises
*"marslib was built without CUDA support"*.

## An Example

Here we train a linear model with a categorical interaction.

```python
import numpy as np
X      = np.random.randn(10000, 2)
X[:,1] = np.random.binomial(1, .5, size=len(X))
y      = 2*X[:,0] + 3*X[:,1] + X[:,0]*X[:,1] + np.random.randn(len(X))

# convert to column-major float
X = np.array(X, order='F', dtype='f')
y = np.array(y, dtype='f')

# Fit the model
import mars
model = mars.fit(X, y, max_terms=17, tail_span=0, linear_only=True)
B     = mars.expand(X, model)                         # expand the basis
model["beta"] = np.linalg.lstsq(B, y, rcond=None)[0]  # store coefficients on the model
y_hat = B @ model["beta"]

# Pretty-print the model
mars.pprint(model)
```

Depending on the random seed, the result should look similar to this:

```
  -0.003
  +1.972 * X[0]
  +3.001 * X[1]
  +1.048 * X[0] * X[1]
```
