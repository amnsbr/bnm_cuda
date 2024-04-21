#!/bin/bash
cd "$(dirname "$0")/.."

target=${1:-"GPU"} # options: GPU, CPU

# set the path to the location of
# built GSL libraries (GSL_BUILD_DIR) and
# libks (LIBKS_DIR)
if [[ -f "./set_dependency_paths.sh" ]]; then
    # this file is not included in the repo
    # and the only thing it does is:
    # `export GSL_BUILD_DIR=...`
    # `export LIBKS_DIR=...`
    source ./set_dependency_paths.sh
else
    echo "Make sure GSL2.7 and libks built libraries are available under \$GSL_BUILD_DIR and \$LIBKS_DIR"
fi

if [[ "$target" == "GPU" ]]; then
    cp run_CMAES.cpp run_CMAES.cu
    nvcc run_CMAES.cu \
        -std=c++11 \
        -o run_CMAES_gpu  \
        ${GSL_BUILD_DIR}/lib/libgsl.a \
        ${GSL_BUILD_DIR}/lib/libgslcblas.a \
        ${LIBKS_DIR}/libks.so \
        -lm \
        -I ${GSL_BUILD_DIR}/include \
        -I ${LIBKS_DIR}/include
    rm run_CMAES.cu
    EXE="./run_CMAES_gpu"
else
    g++ run_CMAES.cpp \
        -o run_CMAES_cpu  \
        -O3 -m64 \
        -fopenmp \
        ${GSL_BUILD_DIR}/lib/libgsl.a \
        ${GSL_BUILD_DIR}/lib/libgslcblas.a \
        ${LIBKS_DIR}/libks.so \
        -lm \
        -I ${GSL_BUILD_DIR}/include \
        -I ${LIBKS_DIR}/include
    EXE="./run_CMAES_cpu"
fi

# simulation config and parameters
nodes="100"
G="0.5-3.5"
wEE="0.05-0.5"
wEI="0.05-0.5"
wIE="0"
het_params="wee-wei"
timesteps="60000"
TR="1000"
window_step="2"
window_size="10"
sim_seed="410"
# CMAES config
opt_seed="1"
itMax="4"
lambda="10"


time $EXE \
    $(pwd)/example/input/SC.txt \
    same \
    $(pwd)/example/input/emp_FCtril.txt \
    $(pwd)/example/input/emp_FCDtril.txt \
    $(pwd)/example/input/maps.txt \
    $nodes $G $wEE $wEI $wIE $het_params \
    $timesteps $TR \
    $window_step $window_size \
    $sim_seed \
    $lambda $itMax $opt_seed
