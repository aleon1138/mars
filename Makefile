CFLAGS += -O2 -Wall -DNDEBUG -DEIGEN_DONT_PARALLELIZE -std=c++11
CFLAGS += -mfma -mavx2 -march=native -fvisibility=hidden
CFLAGS += -I/usr/include/eigen3

PYBIND += $(shell python3 -m pybind11 --includes)
PYBIND += $(shell python3-config --includes)

marslib.so: marslib.cc marsalgo.h
	c++ $(CFLAGS) -shared -fPIC $(PYBIND) -o $@ $<

test: unittest
	./unittest

unittest: unittest.cc marsalgo.h
	c++ $(CFLAGS) -o $@ $< -lgtest -lpthread

clean:
	rm -rf __pycache__ unittest marslib.so
