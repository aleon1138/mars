# MARS

A C++ implementation of [Multivariate Adaptive Regression Splines](https://bit.ly/3cAc7xp). This is
a semi-brute force search for interactions and non-linearities. It will give almost as good
performance as a neural network, but with much faster model evaluation runtimes.

Some references:
* There is a nice writeup [here](https://uc-r.github.io/mars) describing the method.
* There is also a commercial package [here](https://www.salford-systems.com/products/mars).
* The documentation for the R "earth" package is [here](https://cran.r-project.org/web/packages/earth/earth.pdf).
* Stephen Milborrow maintains an excellent resource [here](http://www.milbo.users.sonic.net/earth).
* Additionally there is an module for to scikit-learn [here](https://contrib.scikit-learn.org/py-earth).

## Build Requirements
[Eigen](http://eigen.tuxfamily.org/) - The code has been tested with version 3.3.4.

```bash
    sudo apt install -y libeigen3-dev
```

[GoogleTest](https://github.com/google/googletest) - Unfortunately, the library is [no longer
available pre-compiled](https://bit.ly/2vNUBWN) on Ubuntu.

```bash
    sudo apt install -y libgtest-dev cmake
    cd /usr/src/gtest
    sudo cmake CMakeLists.txt && sudo make
    sudo cp *.a /usr/lib
```

[pybind11](https://github.com/pybind/pybind11) - Install using your python package manager

```bash
    conda install -y pybind11
```

## An Example
Here we train a linear model with a categorical interaction.

```python
import numpy as np
X      = np.random.randn(10000,2)
X[:,1] = np.random.binomial(1, .5, size=len(X))
y      = 2*X[:,0] + 3*X[:,1] + X[:,0]*X[:,1] + np.random.normal(size=len(X))

# convert to column-major float
X = np.array(X,order='F',dtype='f')
y = np.array(y,dtype='f')

# Fit the earth model
import mars
model = mars.fit(X,y,max_epochs=8)
B     = mars.expand(X,model) # expand the basis
beta  = np.linalg.lstsq(B,y,rcond=None)[0]
y_hat = B @ beta
```

Depending on the random seed, the result should look similar to this:

```python
mars.pprint(model,beta)
    -0.023
    +1.994 * X[0]
    +3.018 * X[1]
    +0.975 * X[0] * X[1]
```
