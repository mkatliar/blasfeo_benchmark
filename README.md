# blasfeo_benchmark
Benchmarks for investigating the [BLASFEO Unpredictable execution time](https://github.com/giaf/blasfeo/issues/103).

## Dependencies

### Google benchmarks
If you want to install it manually clone the [repository](https://github.com/google/benchmark.git).
and follow the instructions.
Otherwise it will be automatically added as submodule by the Makefile

### BLASFEO
If you want to install it manually clone the [repository](https://github.com/giaf/blasfeo.git).
and follow the instructions.
Otherwise it will be automatically added as submodule by the Makefile

### MKL
See your distribution documentation
* Archlinux: `yay -S intel-mkl`

### Openblas
See your distribution documentation
* Archlinux: `pacman -S openblas`

## Build and run
To compile the benchmark, just run
```
make
```

If you already installed Google Benchmarks manually you can run just
```
make bench
```

To run the google-based benchmark, run
```
./bench
```

To see more options, run
```
./bench --help
```

or refer to the [documentation](https://github.com/google/benchmark#command-line).
