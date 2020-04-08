# MARS
A C++ implementation of [Multivariate Adaptive Regression Splines](https://en.wikipedia.org/wiki/Multivariate_adaptive_regression_spline).

## Build Requirements
[Eigen](http://eigen.tuxfamily.org/) - The code has been tested with version 3.3.4.
```bash
    sudo apt install -y libeigen3-dev
```

[GoogleTest](https://github.com/google/googletest) - According to [this](https://bit.ly/2vNUBWN),
the library is no longer available pre-compiled on Ubuntu.
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
