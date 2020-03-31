# sudo apt install -y libeigen3-dev

# Installing google test:
# It seems that google test is no longer available as a pre-compiled library on Ubuntu: https://bit.ly/2vNUBWN
#
#     sudo apt-get install -y libgtest-dev cmake
#     cd /usr/src/gtest
#     sudo cmake CMakeLists.txt && sudo make
#     sudo cp *.a /usr/lib


CFLAGS = -Wall $(DEFINES) -O2 -mfma -mavx -march=native -I/usr/include/eigen3

unittest: marsalgo.cpp
	g++ $(CFLAGS) -DUNIT_TEST -o $@ $< -lgtest -lpthread

clean:
	rm -f unittest
