CFLAGS += -O2 -Wall -DNDEBUG -DEIGEN_DONT_PARALLELIZE -std=c++11
CFLAGS += -mfma -mavx2 -march=native -I/usr/include/eigen3

unittest: unittest.cc marsalgo.cc
	g++ $(CFLAGS) -o $@ $< -lgtest -lpthread

clean:
	rm -f unittest
