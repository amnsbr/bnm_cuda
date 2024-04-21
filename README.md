# Biophysical network modeling of the brain

This repository includes a collection of C++ and CUDA code for the simulation and optimization of biophysical network models of the brain on CPU and GPU. The model is based on reduced Wong-Wang model as described in [Deco 2014 J Neurosci](https://doi.org/10.1523/JNEUROSCI.5068-13.2014) with the option to run Feedback Inhibition Control (FIC). FIC is implemented using a hybrid analytial-numerical approach. The inhibitory weights are initialized based on analytical calculations according to [Demirtaş 2019 Neuron](https://doi.org/10.1016/j.neuron.2019.01.017) and their [Python code](https://github.com/murraylab/hbnm) translated to C++.

Optimization of model free parameters are done using covariance matrix adaptation evolution strategy (CMA-ES).

For GPU/CPU compilation see `./example/run_CMAES.sh`.

For usage, after compiling run `run_CMAES_gpu` or `run_CMAES_cpu` with no arguments.

Please note that this code is now depracated. Use [https://github.com/amnsbr/cuBNM](https://github.com/amnsbr/cuBNM) instead.

### Build dependencies
The program requires the following dependencies:

**GCC**: Required for GPU and CPU compilation (tested with GCC 11 and 12).

**Nvidia GPU and CUDA Toolkit**: Required for running simulations on GPU (tested with CUDA Toolkit 11.7 and 11.8). 

**OpenMP**: Required for running simulations in parallel on multiple CPUs

**GSL 2.7**: Built libraries and include should be in `$GSL_BUILD_DIR`:
```
wget https://mirror.ibcp.fr/pub/gnu/gsl/gsl-2.7.tar.gz -O gsl-2.7.tar.gz",
tar -xf gsl-2.7.tar.gz &&"
cd gsl-2.7 && ./configure --prefix=<path-to-gsl-build> --enable-shared &&"
make && make install
export GSL_BUILD_DIR="<path-to-gsl-build>"
```

**libks**: Built libraries and include should be in `$LIBKS_DIR`:
```
git clone https://github.com/agentlans/libks.git
cd libks && make
export LIBKS_DIR=$(pwd)
```