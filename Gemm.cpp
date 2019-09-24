#include <blasfeo_d_aux.h>
#include <blasfeo_d_blas.h>
#include <blasfeo_d_aux_ext_dep.h>
#include <blasfeo_stdlib.h>

#include <cblas.h>

#include <benchmark/benchmark.h>

#include <random>
#include <memory>
#include <map>
#include <iostream>

#include <stdlib.h>


#define FORWARD 0

extern "C"
{
	void dgemm_(
		const char	 *transa,
		const char	 *transb,
		const int *m,
		const int *n,
		const int *k,
		const double *alpha,
		const double *a,
		const int *lda,
		const double *b,
		const int *ldb,
		const double *beta,
		double *c,
		const int *ldc
	);
}


using std::size_t;


static void randomize(size_t m, size_t n, blasfeo_dmat * A)
{
	std::random_device rd; // Will be used to obtain a seed for the random number engine
	std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
	std::uniform_real_distribution<> dis(-1.0, 1.0);

	for (size_t i = 0; i < m; ++i)
		for (size_t j = 0; j < n; ++j)
			BLASFEO_DMATEL(A, i, j) = dis(gen);
}


static void randomize(size_t m, size_t n, double * A)
{
	std::random_device rd; // Will be used to obtain a seed for the random number engine
	std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
	std::uniform_real_distribution<> dis(-1.0, 1.0);

	for (size_t i = 0; i < m; ++i)
		for (size_t j = 0; j < n; ++j)
			A[i + j * m] = dis(gen);
}


static auto alignedAlloc(size_t bytes, size_t alignment)
{
	return std::unique_ptr<char[], decltype(&std::free)>(
		reinterpret_cast<char *>(aligned_alloc(alignment, bytes)), &std::free);
}


// Allocates, holds and and frees memory chunks required by the benchmark
// using aligned_alloc() and blasfeo_memsize_dmat(). 
struct AlignedAllocMem
{
	// Disable copying
	AlignedAllocMem(AlignedAllocMem const&) = delete;


	// Move constructor
	AlignedAllocMem(AlignedAllocMem&& rhs)
	:	A_(rhs.A_)
	,	B_(rhs.B_)
	,	C_(rhs.C_)
	,	D_(rhs.D_)
	{
		rhs.A_ = rhs.B_ = rhs.C_ = rhs.D_ = nullptr;
	}


	AlignedAllocMem(size_t m, size_t n, size_t k, size_t alignment)
	:	A_(aligned_alloc(alignment, blasfeo_memsize_dmat(k, m)))
	,	B_(aligned_alloc(alignment, blasfeo_memsize_dmat(k, n)))
	,	C_(aligned_alloc(alignment, blasfeo_memsize_dmat(m, n)))
	,	D_(aligned_alloc(alignment, blasfeo_memsize_dmat(m, n)))
	{
	}


	~AlignedAllocMem()
	{
		free(D_);
		free(C_);
		free(B_);
		free(A_);
	}


	void * A_ = nullptr;
	void * B_ = nullptr;
	void * C_ = nullptr;
	void * D_ = nullptr;
};


inline std::ostream& operator<<(std::ostream& os, AlignedAllocMem const& mem)
{
	return os << "A: " << mem.A_ << "\tB: " << mem.B_ << "\tC: " << mem.C_ << "\tD: " << mem.D_;
}


// Allocates, holds and and frees memory chunks required by the benchmark
// using blasfeo_malloc_align() and blasfeo_memsize_dmat(). 
struct BlasfeoMallocAlignMem
{
	// Disable copying
	BlasfeoMallocAlignMem(BlasfeoMallocAlignMem const&) = delete;


	// Move constructor
	BlasfeoMallocAlignMem(BlasfeoMallocAlignMem&& rhs)
	:	A_(rhs.A_)
	,	B_(rhs.B_)
	,	C_(rhs.C_)
	,	D_(rhs.D_)
	{
		rhs.A_ = rhs.B_ = rhs.C_ = rhs.D_ = nullptr;
	}


	BlasfeoMallocAlignMem(size_t m, size_t n, size_t k, size_t alignment)
	{
		blasfeo_malloc_align(&A_, blasfeo_memsize_dmat(k, m));
		blasfeo_malloc_align(&B_, blasfeo_memsize_dmat(k, n));
		blasfeo_malloc_align(&C_, blasfeo_memsize_dmat(m, n));
		blasfeo_malloc_align(&D_, blasfeo_memsize_dmat(m, n));
	}


	~BlasfeoMallocAlignMem()
	{
		blasfeo_free_align(D_);
		blasfeo_free_align(C_);
		blasfeo_free_align(B_);
		blasfeo_free_align(A_);
	}


	void * A_ = nullptr;
	void * B_ = nullptr;
	void * C_ = nullptr;
	void * D_ = nullptr;
};


template <typename Alloc>
static void BM_gemm_blasfeo(::benchmark::State& state)
{
	size_t const m = state.range(0);
	size_t const n = state.range(1);
	size_t const k = state.range(2);
	size_t const alignment = state.range(3);

	Alloc mem(m, n, k, alignment);

	// std::cout << "Benchmark size " << m << ", " << n << ", " << k << std::endl;
	// std::cout << "Allocated memory pointers " << mem << std::endl;

	blasfeo_dmat A, B, C, D;
	blasfeo_create_dmat(k, m, &A, mem.A_);
	blasfeo_create_dmat(k, n, &B, mem.B_);
	blasfeo_create_dmat(m, n, &C, mem.C_);
	blasfeo_create_dmat(m, n, &D, mem.D_);

	randomize(k, m, &A);
	randomize(k, n, &B);
	randomize(m, n, &C);

	for (auto _ : state)
		blasfeo_dgemm_tn(m, n, k,
			1.0,
			&A, 0, 0,
			&B, 0, 0,
			1.0,
			&C, 0, 0,
			&D, 0, 0);
}


template <typename Alloc>
static void BM_gemm_blasfeo_reuse_memory(::benchmark::State& state)
{
	size_t const m = state.range(0);
	size_t const n = state.range(1);
	size_t const k = state.range(2);
	size_t const alignment = state.range(3);

	// Data type describing benchmark settings
	using Settings = std::array<size_t, 4>;
	Settings const settings {m, n, k, alignment};

	// Map from settings to memory chunks
	// (persistent between function calls due to static)
	static std::map<Settings, Alloc> mem_map;

	// Check if we have already allocated memory for these settings
	auto mem = mem_map.find(settings);
	if (mem == mem_map.end())
		// Allocate memory chunks if not already allocated
		mem = mem_map.emplace(settings, Alloc(m, n, k, alignment)).first;

	// std::cout << "Benchmark size " << m << ", " << n << ", " << k << std::endl;
	// std::cout << "Allocated memory pointers " << mem << std::endl;

	blasfeo_dmat A, B, C, D;
	blasfeo_create_dmat(k, m, &A, mem->second.A_);
	blasfeo_create_dmat(k, n, &B, mem->second.B_);
	blasfeo_create_dmat(m, n, &C, mem->second.C_);
	blasfeo_create_dmat(m, n, &D, mem->second.D_);

	randomize(k, m, &A);
	randomize(k, n, &B);
	randomize(m, n, &C);

	for (auto _ : state)
		blasfeo_dgemm_tn(m, n, k,
			1.0,
			&A, 0, 0,
			&B, 0, 0,
			1.0,
			&C, 0, 0,
			&D, 0, 0);
}


static void BM_gemm_cblas(::benchmark::State& state)
{
	size_t const m = state.range(0);
	size_t const n = state.range(1);
	size_t const k = state.range(2);

	auto A = std::make_unique<double []>(k * m);
	auto B = std::make_unique<double []>(k * n);
	auto C = std::make_unique<double []>(m * n);

	randomize(k, m, A.get());
	randomize(k, n, B.get());
	randomize(m, n, C.get());

	for (auto _ : state)
		cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, m, n, k,
		1.0, A.get(), k, B.get(), k, 1.0, C.get(), m);

}


static void BM_gemm_blas(::benchmark::State& state)
{
	int const m = state.range(0);
	int const n = state.range(1);
	int const k = state.range(2);

	auto A = std::make_unique<double []>(k * m);
	auto B = std::make_unique<double []>(k * n);
	auto C = std::make_unique<double []>(m * n);

	randomize(k, m, A.get());
	randomize(k, n, B.get());
	randomize(m, n, C.get());

	char const trans_A = 'T';
	char const trans_B = 'N';
	double const alpha = 1.0;
	double const beta = 1.0;

	for (auto _ : state)
		dgemm_(&trans_A, &trans_B, &m, &n, &k, &alpha, A.get(), &k, B.get(), &k, &beta, C.get(), &m);
}


static double computeMin(const std::vector<double>& v)
{
	return *min_element(begin(v), end(v));
}


static double computeMax(const std::vector<double>& v)
{
	return *max_element(begin(v), end(v));
}


#define COMPUTE_CUSTOM_STATISTICS(bm) bm->ComputeStatistics("min", &computeMin)->ComputeStatistics("max", &computeMax)


#if FORWARD

COMPUTE_CUSTOM_STATISTICS(BENCHMARK_TEMPLATE(BM_gemm_blasfeo, AlignedAllocMem))
	->Args({2, 2, 2, 0x40})
	->Args({3, 3, 3, 0x40})
	->Args({5, 5, 5, 0x40})
	->Args({10, 10, 10, 0x40})
	->Args({20, 20, 20, 0x40})
	->Args({30, 30, 30, 0x40})
	->Args({2, 2, 2, 0x1000})
	->Args({3, 3, 3, 0x1000})
	->Args({5, 5, 5, 0x1000})
	->Args({10, 10, 10, 0x1000})
	->Args({20, 20, 20, 0x1000})
	->Args({30, 30, 30, 0x1000});


COMPUTE_CUSTOM_STATISTICS(BENCHMARK_TEMPLATE(BM_gemm_blasfeo, BlasfeoMallocAlignMem))
	->Args({2, 2, 2, 0x40})
	->Args({3, 3, 3, 0x40})
	->Args({5, 5, 5, 0x40})
	->Args({10, 10, 10, 0x40})
	->Args({20, 20, 20, 0x40})
	->Args({30, 30, 30, 0x40})
	->Args({2, 2, 2, 0x1000})
	->Args({3, 3, 3, 0x1000})
	->Args({5, 5, 5, 0x1000})
	->Args({10, 10, 10, 0x1000})
	->Args({20, 20, 20, 0x1000})
	->Args({30, 30, 30, 0x1000});
#else
COMPUTE_CUSTOM_STATISTICS(BENCHMARK_TEMPLATE(BM_gemm_blasfeo, AlignedAllocMem))
	->Args({30, 30, 30, 0x1000})
	->Args({20, 20, 20, 0x1000})
	->Args({10, 10, 10, 0x1000})
	->Args({5, 5, 5, 0x1000})
	->Args({3, 3, 3, 0x1000})
	->Args({2, 2, 2, 0x1000})
	->Args({30, 30, 30, 0x40})
	->Args({20, 20, 20, 0x40})
	->Args({10, 10, 10, 0x40})
	->Args({5, 5, 5, 0x40})
	->Args({3, 3, 3, 0x40})
	->Args({2, 2, 2, 0x40});
	
COMPUTE_CUSTOM_STATISTICS(BENCHMARK_TEMPLATE(BM_gemm_blasfeo, BlasfeoMallocAlignMem))
	->Args({30, 30, 30, 0x1000})
	->Args({20, 20, 20, 0x1000})
	->Args({10, 10, 10, 0x1000})
	->Args({5, 5, 5, 0x1000})
	->Args({3, 3, 3, 0x1000})
	->Args({2, 2, 2, 0x1000})
	->Args({30, 30, 30, 0x40})
	->Args({20, 20, 20, 0x40})
	->Args({10, 10, 10, 0x40})
	->Args({5, 5, 5, 0x40})
	->Args({3, 3, 3, 0x40})
	->Args({2, 2, 2, 0x40});
#endif

#if FORWARD
// Run benchmarks in normal order

COMPUTE_CUSTOM_STATISTICS(BENCHMARK_TEMPLATE(BM_gemm_blasfeo_reuse_memory, AlignedAllocMem))
	->Args({2, 2, 2, 0x40})
	->Args({3, 3, 3, 0x40})
	->Args({5, 5, 5, 0x40})
	->Args({10, 10, 10, 0x40})
	->Args({20, 20, 20, 0x40})
	->Args({30, 30, 30, 0x40});

COMPUTE_CUSTOM_STATISTICS(BENCHMARK_TEMPLATE(BM_gemm_blasfeo_reuse_memory, BlasfeoMallocAlignMem))
	->Args({2, 2, 2, 0x40})
	->Args({3, 3, 3, 0x40})
	->Args({5, 5, 5, 0x40})
	->Args({10, 10, 10, 0x40})
	->Args({20, 20, 20, 0x40})
	->Args({30, 30, 30, 0x40});

#else
// Run benchmarks in reverse order

COMPUTE_CUSTOM_STATISTICS(BENCHMARK_TEMPLATE(BM_gemm_blasfeo_reuse_memory, AlignedAllocMem))
	->Args({30, 30, 30, 0x40})
	->Args({20, 20, 20, 0x40})
	->Args({10, 10, 10, 0x40})
	->Args({5, 5, 5, 0x40})
	->Args({3, 3, 3, 0x40})
	->Args({2, 2, 2, 0x40});

COMPUTE_CUSTOM_STATISTICS(BENCHMARK_TEMPLATE(BM_gemm_blasfeo_reuse_memory, BlasfeoMallocAlignMem))
	->Args({30, 30, 30, 0x40})
	->Args({20, 20, 20, 0x40})
	->Args({10, 10, 10, 0x40})
	->Args({5, 5, 5, 0x40})
	->Args({3, 3, 3, 0x40})
	->Args({2, 2, 2, 0x40});
#endif


COMPUTE_CUSTOM_STATISTICS(BENCHMARK(BM_gemm_cblas))
	->Args({2, 2, 2})
	->Args({3, 3, 3})
	->Args({5, 5, 5})
	->Args({10, 10, 10})
	->Args({20, 20, 20})
	->Args({30, 30, 30});


COMPUTE_CUSTOM_STATISTICS(BENCHMARK(BM_gemm_blas))
	->Args({2, 2, 2})
	->Args({3, 3, 3})
	->Args({5, 5, 5})
	->Args({10, 10, 10})
	->Args({20, 20, 20})
	->Args({30, 30, 30});