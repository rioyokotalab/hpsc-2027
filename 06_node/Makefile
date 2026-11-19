CXX = g++
CXXFLAGS = -O3 -march=native -fopenmp
PROGS = 00_hello 01_race 02_pi 03_triad

all: $(PROGS)
%: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@
clean:
	rm -f $(PROGS)
