CXXFLAGS += -O2 -Wall -std=c++11
CXXFLAGS += -mfma -mavx2 -march=native -fvisibility=hidden

CPPFLAGS += $(shell pkg-config --cflags eigen3)
CPPFLAGS += $(shell python3 -m pybind11 --includes)
CPPFLAGS += $(shell python3-config --includes)
CPPFLAGS += -DNDEBUG -DEIGEN_DONT_PARALLELIZE

marslib.so: marslib.cc marsalgo.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -shared -fPIC -o $@ $<

test: unittest
	./unittest

unittest: unittest.cc marsalgo.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $< -lgtest -lpthread

clean:
	rm -rf __pycache__ unittest marslib.so
