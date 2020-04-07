CFLAGS += -O2 -Wall -DNDEBUG -DEIGEN_DONT_PARALLELIZE -std=c++11
CFLAGS += -mfma -mavx2 -march=native -I/usr/include/eigen3

PYBIND += $(shell python3 -m pybind11 --includes)
PYBIND += $(shell python3-config --includes)

marslib.so: mars.cc marsalgo.h
	c++ $(CFLAGS) -shared -fPIC $(PYBIND) -o $@ mars.cc

test: unittest
	./unittest

unittest: unittest.cc marsalgo.h
	c++ $(CFLAGS) -o $@ $< -lgtest -lpthread

clean:
	rm -f unittest marslib.so
