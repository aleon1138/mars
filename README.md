# MARS

A C++ implementation of [Multivariate Adaptive Regression Splines](https://bit.ly/3cAc7xp). This is
a semi-brute force search for interactions and non-linearities. It will give almost as good
performance as a neural network, but with much faster model evaluation run-times.

Some references:
* There is a nice write-up [here](https://uc-r.github.io/mars) describing the method.
* There is also a commercial package [here](https://www.salford-systems.com/products/mars).
* The documentation for the R "earth" package is [here](https://cran.r-project.org/web/packages/earth/earth.pdf).
* Stephen Milborrow maintains an excellent resource [here](http://www.milbo.users.sonic.net/earth).
* Additionally there is an module for to scikit-learn [here](https://contrib.scikit-learn.org/py-earth).

## Performance

We use [OpenMP](https://www.openmp.org) to achieve nearly linear speed-up per core used. Note that
there is additional memory overhead for each core used, which might constraint the total number
of cores available. As always, one can control the number of threads used via the `OMP_NUM_THREADS`
environment variable.

| Threads | Speed-Up |
|:-------:|:--------:|
|    1    |   1.0 x  |
|    2    |   2.0 x  |
|    3    |   2.8 x  |
|    4    |   3.4 x  |

## Supported Platforms

These instructions have been verified to work on the following platforms:
* Ubuntu 18.04 and 20.04
* Raspbian 10
* macOS 10.13 (WIP)

## Build Requirements

[Eigen](http://eigen.tuxfamily.org/) - The code has been tested with version 3.3.4.

```bash
sudo apt install -y libeigen3-dev
```
... on macOS:
```bash
brew install pkg-config eigen
```

[GoogleTest](https://github.com/google/googletest) - Unfortunately, the library is [no longer
available pre-compiled](https://bit.ly/2vNUBWN) on Ubuntu.

```bash
sudo apt install -y libgtest-dev cmake
cd /usr/src/gtest
sudo cmake CMakeLists.txt && sudo make
sudo cp *.a /usr/lib
```

[pybind11](https://github.com/pybind/pybind11) - Install using your python package manager of choice:

```bash
pip3 install pybind11
```
... or ...
```bash
conda install -y pybind11
```

## Build Instructions
You can either use the Makefile:

```bash
cd mars
make
make test # optional - build and run the unit tests
```

Or the `setup.py` script provided:

```bash
cd mars
pip install .
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

# Fit the earth model
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
