# for memchecks use -O0 -g -fsanitize=address

UNAME_S := $(shell uname)
UNAME_M := $(shell uname -m)

CXXFLAGS += -O3 -fvisibility=hidden
CXXFLAGS += -Wall -march=native -std=c++17

ifeq ($(UNAME_S), Darwin)
	ifeq ($(UNAME_M), x86_64)
		CXXFLAGS += -mfma # strange, but this is not default under arch=native
	endif
	CXXFLAGS += -undefined dynamic_lookup # needed for pybind
	CXXFLAGS += -Wno-unknown-warning-option # needed for eigen
	# Apple Clang does not ship an OpenMP runtime; use Homebrew libomp.
	LIBOMP_PREFIX ?= $(shell brew --prefix libomp 2>/dev/null)
	CXXFLAGS += -Xpreprocessor -fopenmp -I$(LIBOMP_PREFIX)/include
	LDFLAGS  += -L$(LIBOMP_PREFIX)/lib
	LDLIBS   += -lomp
else
	CXXFLAGS += -fopenmp
endif

CPPFLAGS += $(shell pkg-config --cflags eigen3)
CPPFLAGS += $(shell python3 -m pybind11 --includes)
CPPFLAGS += $(shell python3-config --includes)
CPPFLAGS += -DNDEBUG -DEIGEN_DONT_PARALLELIZE

TARGET = marslib$(shell python3-config --extension-suffix)

$(TARGET): marslib.cc marsalgo.cc marsalgo.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -shared -fPIC marslib.cc marsalgo.cc -o $@ $(LDFLAGS) $(LDLIBS)

test: unittest $(TARGET)
	python3 -m pytest tests/ -v
	./unittest

format:
	astyle -A4 -S -z2 -n -j *.h *.cc

unittest: tests/unittest.cc marsalgo.cc marsalgo.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) tests/unittest.cc marsalgo.cc -lgtest -lgtest_main -lpthread -o $@ $(LDFLAGS) $(LDLIBS)

clean:
	rm -rf __pycache__/ build/ mars.egg-info/ unittest $(TARGET) dist
