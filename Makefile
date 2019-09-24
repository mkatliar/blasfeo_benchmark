#!
CXX = g++
CC = gcc

BLAS_LIB = openblas
# BLAS_LIB = mkl_rt

# Needed for MKL in archlinux
BLAS_PATH=/opt/intel/compilers_and_libraries_2019.5.281/linux/mkl/lib/intel64_lin/

all: google_benchmark_local blasfeo_local bench

google_benchmark_local:
	git submodule update --init --recursive
	cd benchmark; mkdir -p build; cd build; cmake -DCMAKE_CXX_COMPILER=$(CXX) -DCMAKE_C_COMPILER=$(CC) -DCMAKE_CXX_FLAGS="-fPIC" -DCMAKE_BUILD_TYPE=Release -DBENCHMARK_ENABLE_TESTING=0 ..; make -j4

blasfeo_local:
	git submodule update --init --recursive
	cd blasfeo; $(MAKE) static_library

bench: Gemm.cpp Main.cpp
	$(CXX) -std=c++14 -g -O2 -DNDEBUG \
		-I./benchmark/include \
		-L$(BLAS_PATH) \
		-I./benchmark/include \
		-L./benchmark/build/src \
		-I./blasfeo/include \
		-L./blasfeo/lib \
		Gemm.cpp Main.cpp -lbenchmark -lblasfeo -lpthread -l$(BLAS_LIB) -o bench

run:
	./bench

run_statistics:
	./bench --benchmark_report_aggregates_only=true --benchmark_repetitions=5

clean:
	rm -f bench

deep_clean:
	rm -f bench
	( cd blasfeo; $(MAKE) deep_clean)
	( cd benchmark/build; rm -r CMakeFiles CMakeCache.txt)
