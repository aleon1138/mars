# MARS

A C++ implementation of [Multivariate Adaptive Regression Splines](https://w.wiki/GVPL).
This is a semi-brute force search for interactions and non-linearities. It will
provide competitive regression performance compared to neural network for, but
with much faster model evaluation runtimes.

References:

* [Write-up describing the method](https://uc-r.github.io/mars)
* [Commercial MARS package](https://www.salford-systems.com/products/mars) by Salford Systems
* [R "earth" package documentation](https://cran.r-project.org/web/packages/earth/earth.pdf)
* [Stephen Milborrow's resource page](http://www.milbo.users.sonic.net/earth)
* Additionally, there is a scikit-learn module [here](https://contrib.scikit-learn.org/py-earth)

## Performance

We use [OpenMP](https://www.openmp.org) to achieve good speed-up per core. There
is some memory overhead for each thread launched, which might constrain the total
number of cores available. You can control the number of threads via the
`OMP_NUM_THREADS` environment variable or the `threads` argument.

The following timings were obtained on an AMD EPYC 9654 96-Core Processor with
192 logical CPUs. Note that multi-threaded performance is nearly ideal up to 30
cores or so.

![Performance timings of MARS](timings.png)

## Supported Platforms

These instructions have been verified to work on the following platforms:

* Ubuntu 18.04 through 24.04
* Raspbian 10
* macOS High Sierra (10.13) through Tahoe (26.4)

## Build Requirements

* [CMake](https://cmake.org/) ≥ 3.18
* [Eigen](http://eigen.tuxfamily.org/) ≥ 3.3
* [GoogleTest](https://github.com/google/googletest) (only needed for `make test`)
* A C++17 compiler with OpenMP support

```bash
sudo apt install -y cmake libeigen3-dev libgtest-dev
```
... on macOS:
```bash
brew install cmake eigen googletest libomp
```

`pip install .` pulls `pybind11` and `scikit-build-core` automatically as build
dependencies — no need to install them by hand.

## Build Instructions

Install directly via pip (recommended):

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
model = mars.fit(X, y, max_epochs=8, tail_span=0, linear_only=True)
B     = mars.expand(X, model) # expand the basis
beta  = np.linalg.lstsq(B, y, rcond=None)[0]
y_hat = B @ beta

# Pretty-print the model
mars.pprint(model, beta)
```

Depending on the random seed, the result should look similar to this:

```
  -0.003
  +1.972 * X[0]
  +3.001 * X[1]
  +1.048 * X[0] * X[1]
```
