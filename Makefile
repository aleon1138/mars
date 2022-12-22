CXXFLAGS += -O3 -Wall -std=c++11 -fopenmp
CXXFLAGS += -march=native -fvisibility=hidden
ifeq ($(shell uname), Darwin)
	CXXFLAGS += -mfma # strange, but this is not default under arch=native
	CXXFLAGS += -undefined dynamic_lookup # needed for pybind
	CXXFLAGS += -Wno-unknown-warning-option # needed for eigen
endif

CPPFLAGS += $(shell pkg-config --cflags eigen3)
CPPFLAGS += $(shell python3 -m pybind11 --includes)
CPPFLAGS += $(shell python3-config --includes)
CPPFLAGS += -DNDEBUG -DEIGEN_DONT_PARALLELIZE

marslib.so: marslib.cc marsalgo.h array.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -shared -fPIC -o $@ $<

test: unittest
	./unittest

format:
	astyle -A4 -S -z2 -n -j *.h *.cc

unittest: unittest.cc marsalgo.h array.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $< -lgtest -lpthread

clean:
	rm -rf __pycache__ unittest marslib.so
