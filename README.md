# blasfeo_benchmark
Benchmarks for investigating the [BLASFEO Unpredictable execution time](https://github.com/giaf/blasfeo/issues/103).

## Dependencies

### Google benchmarks
If you want to install it manually clone the [repository](https://github.com/google/benchmark.git).
and follow the instructions.
Otherwise it will be automatically added as submodule by the Makefile.

Don't forget to specify `-DCMAKE_BUILD_TYPE=Release` when building the benchmark library manually.
If you see the following warning when running a benchmark
```
***WARNING*** Library was built as DEBUG. Timings may be affected. 
```
it means that you didn't do it.


### BLASFEO
If you want to install it manually clone the [repository](https://github.com/giaf/blasfeo.git).
and follow the instructions.
Otherwise it will be automatically added as submodule by the Makefile

### MKL
See your distribution documentation
* Archlinux: `yay -S intel-mkl openmp`

### Openblas
See your distribution documentation
* Archlinux: `pacman -S openblas`

## Build and run
To compile the benchmark, just run
```bash
make
```

If you already all dependencies manually you can run just
```bash
make bench
```

Before running the benchmark, disable CPU scaling by the following command:
```bash
sudo cpupower frequency-set -g performance
```
otherwise you will see this message:
```
***WARNING*** CPU scaling is enabled, the benchmark real time measurements may be noisy and will incur extra overhead.
```

To run the google-based benchmark, run
```bash
./bench
```

To see more options, run
```bash
./bench --help
```

or refer to the [documentation](https://github.com/google/benchmark#command-line).
