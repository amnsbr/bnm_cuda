/*
Reduced Wong-Wang model (Deco 2014) simulation on GPU

This code includes kernels needed for the simulation of BOLD signal and 
calculation of  FC and FCD, in addition to other GPU-related functions.

Each simulation (for a given set of parameters) is run on a single GPU 
block, and each thread in the block simulates one region of the brain. 
The threads in the block form a "cooperative group" and are synchronized 
after each integration time step.
Calculation of FC and FCD are also similarly parallelized across simulations
and region or
window pairs.

Parts of this code are based on https://github.com/BrainModes/The-Hybrid-Virtual-Brain, 
https://github.com/murraylab/hbnm & https://github.com/decolab/cb-neuromod

Author: Amin Saberi, Feb 2023
*/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <cstdlib>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cooperative_groups.h>
#include <random>
#include <algorithm>
#include <cassert>
#include <vector>
#include "constants.h"
#include "fic.cpp"
#include <gsl/gsl_statistics.h>
extern "C" {
#include "ks.h"
};

namespace cg = cooperative_groups;

#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

#define CUDA_CHECK_LAST_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, const int line)
{
    // from https://leimao.github.io/blog/Proper-CUDA-Error-Checking/
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

__device__ int d_I_SAMPLING_START, d_I_SAMPLING_END, d_I_SAMPLING_DURATION;
__device__ u_real d_init_delta;

__global__ void bnm(
    u_real **BOLD_ex,
    u_real **S_E, u_real **I_E, u_real **r_E, 
    u_real **S_I, u_real **I_I, u_real **r_I, 
    int n_vols_remove,
    u_real *SC, u_real *G_list, u_real *w_EE_list, u_real *w_EI_list, u_real *w_IE_list,
    int N_SIMS, int nodes, int BOLD_TR, u_real dt, int time_steps, 
    u_real *noise, bool do_fic, u_real **w_IE_fic,
    bool adjust_fic, int max_fic_trials, bool *FIC_failed, int *fic_n_trials,
    bool extended_output,
    int corr_len,
    bool regional_params = false
    ) {
    /*
        Runs a group of reduced Wong Wang simulations
        in parallel on GPUs
    */
    // convert block to a cooperative group
    int sim_idx = blockIdx.x;
    if (sim_idx >= N_SIMS) return;
    cg::thread_block b = cg::this_thread_block();

    // allocate shared memory for the S_i_1_E (S_E at time t-1)
    // to be able to calculate global input from other regions 
    // (threads on the same block)
    // the memory is allocated dynamically based on the number of nodes
    // (see https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)
    extern __shared__ u_real S_i_1_E[];
    u_real S_i_E, S_i_I, bw_x, bw_f, bw_nu, bw_q;

    int j = b.thread_rank(); // region index
    if (j >= nodes) return;
    // set initial values and sync all threads
    S_i_1_E[j] = 0.001;
    S_i_E = 0.001;
    S_i_I = 0.001;
    bw_x = 0.0;
    bw_f = 1.0;
    bw_nu = 1.0;
    bw_q = 1.0;
    
    // use __shared__ for parameters that are shared
    // between regions in the same simulation, but
    // not for those that (may) vary, e.g. w_IE, w_EE and w_IE
    __shared__ u_real G;
    u_real w_IE, w_EE, w_EI;
    G = G_list[sim_idx];
    if (regional_params) {
        w_EE = w_EE_list[sim_idx*nodes+j];
        w_EI = w_EI_list[sim_idx*nodes+j];
        if (do_fic) {
            w_IE = w_IE_fic[sim_idx][j];
            if (w_IE_1) {
                w_IE = 1;
            }
        } else {
            w_IE = w_IE_list[sim_idx*nodes+j];
        }
    } else {
        w_EE = w_EE_list[sim_idx]; 
        w_EI = w_EI_list[sim_idx];
        // get w_IE from precalculated FIC output or use fixed w_IE
        if (do_fic) {
            w_IE = w_IE_fic[sim_idx][j];
            if (w_IE_1) {
                w_IE = 1;
            }
        } else {
            w_IE = w_IE_list[sim_idx];
        }
    }

    // initialization of extended output
    if (extended_output) {
        S_E[sim_idx][j] = 0;
        I_E[sim_idx][j] = 0;
        r_E[sim_idx][j] = 0;
        S_I[sim_idx][j] = 0;
        I_I[sim_idx][j] = 0;
        r_I[sim_idx][j] = 0;
    }

    // initializations for FIC adjustment
    u_real mean_I_E, delta, I_E_ba_diff;
    int fic_trial;
    if (adjust_fic) {
        mean_I_E = 0;
        delta = d_init_delta;
        fic_trial = 0;
        I_E_ba_diff;
        FIC_failed[sim_idx] = false;
    }
    // copy adjust_fic to a local shared variable and use it to indicate
    // if adjustment should be stopped after N trials
    __shared__ bool _adjust_fic;
    _adjust_fic = adjust_fic;
    __shared__ bool needs_fic_adjustment;

    // integration loop
    u_real tmp_globalinput, tmp_I_E, tmp_I_I, tmp_r_E, tmp_r_I, dSdt_E, dSdt_I, tmp_f,
        tmp_aIb_E, tmp_aIb_I, tmp_rand_E, tmp_rand_I;
    int ts_bold, int_i, k;
    long noise_idx = 0;
    int BOLD_len_i = 0;
    ts_bold = 0;
    while (ts_bold <= time_steps) {
        for (int_i = 0; int_i < 10; int_i++) {
            // wait for other nodes
            // note that a sync call is needed before using
            // and updating S_i_1_E, otherwise the current
            // thread might access S_E of other nodes at wrong
            // times (t or t-2 instead of t-1)
            cg::sync(b);
            tmp_globalinput = 0;
            for (k=0; k<nodes; k++) {
                // calculate global input
                tmp_globalinput += SC[j*nodes+k] * S_i_1_E[k];
            }
            // equations
            tmp_I_E = w_E__I_0 + w_EE * S_i_E + tmp_globalinput * G * J_NMDA - w_IE*S_i_I;
            tmp_I_I = w_I__I_0 + w_EI * S_i_E - S_i_I;
            tmp_aIb_E = a_E * tmp_I_E - b_E;
            tmp_aIb_I = a_I * tmp_I_I - b_I;
            #ifdef USE_FLOATS
            // to avoid firing rate approaching infinity near I = b/a
            if (abs(tmp_aIb_E) < 1e-4) tmp_aIb_E = 1e-4;
            if (abs(tmp_aIb_I) < 1e-4) tmp_aIb_I = 1e-4;
            #endif
            tmp_r_E = tmp_aIb_E / (1 - EXP(-d_E * tmp_aIb_E));
            tmp_r_I = tmp_aIb_I / (1 - EXP(-d_I * tmp_aIb_I));
            noise_idx = (((ts_bold * 10 + int_i) * nodes * 2) + (j * 2));
            tmp_rand_E = noise[noise_idx];
            noise_idx = (((ts_bold * 10 + int_i) * nodes * 2) + (j * 2) + 1);
            tmp_rand_I = noise[noise_idx];
            dSdt_E = tmp_rand_E * sigma_model * sqrt_dt + dt * ((1 - S_i_E) * gamma_E * tmp_r_E - S_i_E * itau_E);
            dSdt_I = tmp_rand_I * sigma_model * sqrt_dt + dt * (gamma_I * tmp_r_I - S_i_I * itau_I);
            S_i_E += dSdt_E;
            S_i_I += dSdt_I;
            // clip S to 0-1
            S_i_E = CUDA_MAX(0.0, CUDA_MIN(1.0, S_i_E));
            S_i_I = CUDA_MAX(0.0, CUDA_MIN(1.0, S_i_I));
            // wait for other regions
            // see note above on why this is needed before updating S_i_1_E
            cg::sync(b);
            S_i_1_E[j] = S_i_E;
        }
        // Compute BOLD for that time-step (subsampled to 1 ms)
        bw_x  = bw_x  +  model_dt * (S_i_E - kappa * bw_x - y * (bw_f - 1.0));
        tmp_f    = bw_f  +  model_dt * bw_x;
        bw_nu = bw_nu +  model_dt * itau * (bw_f - POW(bw_nu, ialpha));
        bw_q  = bw_q  +  model_dt * itau * (bw_f * (1.0 - POW(oneminrho,(1.0/bw_f))) / rho  - POW(bw_nu,ialpha) * bw_q / bw_nu);
        bw_f  = tmp_f;

        if (ts_bold % BOLD_TR == 0) {
            BOLD_ex[sim_idx][BOLD_len_i*nodes+j] = V_0 * (k1 * (1 - bw_q) + k2 * (1 - bw_q/bw_nu) + k3 * (1 - bw_nu));
            if ((BOLD_len_i>=n_vols_remove)) {
                if (extended_output) {
                    S_E[sim_idx][j] += S_i_E;
                    I_E[sim_idx][j] += tmp_I_E;
                    r_E[sim_idx][j] += tmp_r_E;
                    S_I[sim_idx][j] += S_i_I;
                    I_I[sim_idx][j] += tmp_I_I;
                    r_I[sim_idx][j] += tmp_r_I;
                }
            }
            BOLD_len_i++;
        }
        ts_bold++;

        if (_adjust_fic) {
            if ((ts_bold >= d_I_SAMPLING_START) & (ts_bold <= d_I_SAMPLING_END)) {
                mean_I_E += tmp_I_E;
            }
            if (ts_bold == d_I_SAMPLING_END) {
                needs_fic_adjustment = false;
                cg::sync(b); // all threads must be at the same time point here given needs_fic_adjustment is shared
                mean_I_E /= d_I_SAMPLING_DURATION;
                I_E_ba_diff = mean_I_E - b_a_ratio_E;
                if (abs(I_E_ba_diff + 0.026) > 0.005) {
                    needs_fic_adjustment = true;
                    if (fic_trial < max_fic_trials) { // only do the adjustment if max trials is not exceeded
                        // up- or downregulate inhibition
                        if ((I_E_ba_diff) < -0.026) {
                            w_IE -= delta;
                            // printf("sim %d node %d (trial %d): %f ==> adjusting w_IE by -%f ==> %f\n", sim_idx, j, fic_trial, I_E_ba_diff, delta, w_IE);
                            delta -= 0.001;
                            delta = CUDA_MAX(delta, 0.001);
                        } else {
                            w_IE += delta;
                            // printf("sim %d node %d (trial %d): %f ==> adjusting w_IE by +%f ==> %f\n", sim_idx, j, fic_trial, I_E_ba_diff, delta, w_IE);
                        }
                    }
                }
                cg::sync(b); // wait to see if needs_fic_adjustment in any node
                // if needs_fic_adjustment in any node do another trial or declare fic failure and continue
                // the simulation until the end
                if (needs_fic_adjustment) {
                    if (fic_trial < max_fic_trials) {
                        // reset time and random sequence
                        ts_bold = 0;
                        BOLD_len_i = 0;
                        noise_idx = 0;
                        fic_trial++;
                        // reset states
                        S_i_1_E[j] = 0.001;
                        S_i_E = 0.001;
                        S_i_I = 0.001;
                        bw_x = 0.0;
                        bw_f = 1.0;
                        bw_nu = 1.0;
                        bw_q = 1.0;
                        mean_I_E = 0;
                    } else {
                        // continue the simulation but
                        // declare FIC failed
                        FIC_failed[sim_idx] = true;
                        _adjust_fic = false;
                    }
                } else {
                    // if no node needs fic adjustment don't run
                    // this block of code any more
                    _adjust_fic = false;
                }
                cg::sync(b);
            }
        }
    }
    if (adjust_fic) {
        // save the final w_IE after adjustment
        w_IE_fic[sim_idx][j] = w_IE;
        // as well as the number of adjustment trials needed
        fic_n_trials[sim_idx] = fic_trial;
    }
    if (extended_output) {
        // take average
        int extended_output_time_points = BOLD_len_i - n_vols_remove;
        S_E[sim_idx][j] /= extended_output_time_points;
        I_E[sim_idx][j] /= extended_output_time_points;
        r_E[sim_idx][j] /= extended_output_time_points;
        S_I[sim_idx][j] /= extended_output_time_points;
        I_I[sim_idx][j] /= extended_output_time_points;
        r_I[sim_idx][j] /= extended_output_time_points;
    }
}

__global__ void bold_stats(
    u_real **mean_bold, u_real **ssd_bold,
    u_real **BOLD_ex, int N_SIMS, int nodes,
    int output_ts, int corr_len, int n_vols_remove) {
    /*
        Calculates BOLD mean and ssd needed for FC calculation
    */
    // TODO: consider combining this with window_bold_stats
    // get simulation index
    int sim_idx = blockIdx.x;
    if (sim_idx >= N_SIMS) return;
    // get node index
    int j = threadIdx.x;
    if (j >= nodes) return;

    // mean
    u_real _mean_bold = 0;
    int vol;
    for (vol=n_vols_remove; vol<output_ts; vol++) {
        _mean_bold += BOLD_ex[sim_idx][vol*nodes+j];
    }
    _mean_bold /= corr_len;
    // ssd
    u_real _ssd_bold = 0;
    for (vol=n_vols_remove; vol<output_ts; vol++) {
        _ssd_bold += POW(BOLD_ex[sim_idx][vol*nodes+j] - _mean_bold, 2);
    }
    // save to memory
    mean_bold[sim_idx][j] = _mean_bold;
    ssd_bold[sim_idx][j] = SQRT(_ssd_bold);
}

__global__ void window_bold_stats(
    u_real **BOLD_ex, int N_SIMS, int nodes,
    int n_windows, int window_size_1, int *window_starts, int *window_ends,
    u_real **windows_mean_bold, u_real **windows_ssd_bold) {
    /*
        Calculates window BOLD mean and ssd needed for FC calculation
    */
    // get simulation index
    int sim_idx = blockIdx.x;
    if (sim_idx >= N_SIMS) return;
    // get window index
    int w = blockIdx.y;
    if (w >= n_windows) return;
    // get node index
    int j = threadIdx.x;
    if (j >= nodes) return;
    // calculate mean of window
    u_real _mean_bold = 0;
    int vol;
    for (vol=window_starts[w]; vol<=window_ends[w]; vol++) {
        _mean_bold += BOLD_ex[sim_idx][vol*nodes+j];
    }
    _mean_bold /= window_size_1;
    // calculate sd of window
    u_real _ssd_bold = 0;
    for (vol=window_starts[w]; vol<=window_ends[w]; vol++) {
        _ssd_bold += POW(BOLD_ex[sim_idx][vol*nodes+j] - _mean_bold, 2);
    }
    // save to memory
    windows_mean_bold[sim_idx][w*nodes+j] = _mean_bold;
    windows_ssd_bold[sim_idx][w*nodes+j] = SQRT(_ssd_bold);
}

__global__ void fc(u_real **fc_trils, u_real **windows_fc_trils,
    u_real **BOLD_ex, int N_SIMS, int nodes, int n_pairs, int *pairs_i,
    int *pairs_j, int output_ts, int n_vols_remove, 
    int corr_len, u_real **mean_bold, u_real **ssd_bold, 
    int n_windows, int window_size_1, u_real **windows_mean_bold, u_real **windows_ssd_bold,
    int *window_starts, int *window_ends,
    int maxThreadsPerBlock) {
    /*
        Calculates FC and time window FCs
    */
        // get simulation index
        int sim_idx = blockIdx.x;
        if (sim_idx >= N_SIMS) return;
        // get pair index
        int pair_idx = threadIdx.x + (maxThreadsPerBlock * blockIdx.y);
        if (pair_idx >= n_pairs) return;
        int i = pairs_i[pair_idx];
        int j = pairs_j[pair_idx];
        // get window index
        int w = blockIdx.z - 1; // -1 indicates total FC
        if (w >= n_windows) return;
        int vol_start, vol_end;
        u_real _mean_bold_i, _mean_bold_j, _ssd_bold_i, _ssd_bold_j;
        if (w == -1) {
            vol_start = n_vols_remove;
            vol_end = output_ts;
            _mean_bold_i = mean_bold[sim_idx][i];
            _ssd_bold_i = ssd_bold[sim_idx][i];
            _mean_bold_j = mean_bold[sim_idx][j];
            _ssd_bold_j = ssd_bold[sim_idx][j];
        } else {
            vol_start = window_starts[w];
            vol_end = window_ends[w]+1; // +1 because we want to include end in the window
            _mean_bold_i = windows_mean_bold[sim_idx][w*nodes+i];
            _ssd_bold_i = windows_ssd_bold[sim_idx][w*nodes+i];
            _mean_bold_j = windows_mean_bold[sim_idx][w*nodes+j];
            _ssd_bold_j = windows_ssd_bold[sim_idx][w*nodes+j];
        }
        // calculate sigma(x_i * x_j)
        int vol;
        u_real cov = 0;
        for (vol=vol_start; vol<vol_end; vol++) {
            cov += (BOLD_ex[sim_idx][vol*nodes+i] - _mean_bold_i) * (BOLD_ex[sim_idx][vol*nodes+j] - _mean_bold_j);
        }
        // calculate corr(i, j)
        u_real corr = cov / (_ssd_bold_i * _ssd_bold_j);
        // write to memory
        if (w == -1) {
            fc_trils[sim_idx][pair_idx] = corr;
        } else {
            windows_fc_trils[sim_idx][w*n_pairs+pair_idx] = corr;
        }
    }

__global__ void window_fc_stats(
    u_real **windows_mean_fc, u_real **windows_ssd_fc,
    u_real **L_windows_mean_fc, u_real **L_windows_ssd_fc,
    u_real **R_windows_mean_fc, u_real **R_windows_ssd_fc,
    u_real **windows_fc_trils, int N_SIMS, int n_windows, int n_pairs,
    bool save_hemis, int n_pairs_hemi) {
    /*
        Calculates mean and ssd of dynamic FC in each edge
        needed for FCD calculation
    */

        // get simulation index
        int sim_idx = blockIdx.x;
        if (sim_idx >= N_SIMS) return;
        // get window index
        int w = threadIdx.x;
        if (w >= n_windows) return;
        // get hemi
        int hemi = blockIdx.z;
        if (!save_hemis) {
            if (hemi > 0) return;
        } else {
            if (hemi > 2) return;
        }
        // calculate mean fc of window
        u_real _mean_fc = 0;
        int pair_idx_start = 0;
        int pair_idx_end = n_pairs; // non-inclusive
        int pair_idx;
        int _curr_n_pairs = n_pairs;
        // for left and right specify start and end indices
        // that belong to current hemi. Note that this will work
        // regardless of exc_interhemispheric true or false
        if (hemi == 1) { // left
            pair_idx_end = n_pairs_hemi;
            _curr_n_pairs = n_pairs_hemi;
        } else if (hemi == 2) { // right
            pair_idx_start = n_pairs - n_pairs_hemi;
            _curr_n_pairs = n_pairs_hemi;
        }
        for (pair_idx=pair_idx_start; pair_idx<pair_idx_end; pair_idx++) {
            _mean_fc += windows_fc_trils[sim_idx][w*n_pairs+pair_idx];
        }
        _mean_fc /= _curr_n_pairs;
        // calculate ssd fc of window
        u_real _ssd_fc = 0;
        for (pair_idx=pair_idx_start; pair_idx<pair_idx_end; pair_idx++) {
            _ssd_fc += POW(windows_fc_trils[sim_idx][w*n_pairs+pair_idx] - _mean_fc, 2);
        }
        // save to memory
        if (hemi == 0) {
            windows_mean_fc[sim_idx][w] = _mean_fc;
            windows_ssd_fc[sim_idx][w] = SQRT(_ssd_fc);
        } else if (hemi == 1) {
            L_windows_mean_fc[sim_idx][w] = _mean_fc;
            L_windows_ssd_fc[sim_idx][w] = SQRT(_ssd_fc);
        } else if (hemi == 2) {
            R_windows_mean_fc[sim_idx][w] = _mean_fc;
            R_windows_ssd_fc[sim_idx][w] = SQRT(_ssd_fc);
        }
    }

__global__ void fcd(
    u_real **fcd_trils, u_real **L_fcd_trils, u_real **R_fcd_trils,
    u_real **windows_fc_trils,
    u_real **windows_mean_fc, u_real **windows_ssd_fc,
    u_real **L_windows_mean_fc, u_real **L_windows_ssd_fc,
    u_real **R_windows_mean_fc, u_real **R_windows_ssd_fc,
    int N_SIMS, int n_pairs, int n_windows, int n_window_pairs, 
    int *window_pairs_i, int *window_pairs_j, int maxThreadsPerBlock,
    bool save_hemis, int n_pairs_hemi) {
    /*
        Calculates functional connectivity dynamics
        matrix
    */
        // get simulation index
        int sim_idx = blockIdx.x;
        if (sim_idx >= N_SIMS) return;
        // get window pair index
        int window_pair_idx = threadIdx.x + (maxThreadsPerBlock * blockIdx.y);
        if (window_pair_idx >= n_window_pairs) return;
        int w_i = window_pairs_i[window_pair_idx];
        int w_j = window_pairs_j[window_pair_idx];
        // get hemi
        int hemi = blockIdx.z;
        if (!save_hemis) {
            if (hemi > 0) return;
        } else {
            if (hemi > 2) return;
        }
        // calculate cov
        int pair_idx;
        u_real cov = 0;
        if (hemi == 0) {
            for (pair_idx=0; pair_idx<n_pairs; pair_idx++) {
                cov += 
                    (windows_fc_trils[sim_idx][w_i*n_pairs+pair_idx] - windows_mean_fc[sim_idx][w_i]) 
                    * (windows_fc_trils[sim_idx][w_j*n_pairs+pair_idx] - windows_mean_fc[sim_idx][w_j]);
            }
            fcd_trils[sim_idx][window_pair_idx] = cov / (windows_ssd_fc[sim_idx][w_i] * windows_ssd_fc[sim_idx][w_j]);
        } else if (hemi == 1) {
            for (pair_idx=0; pair_idx<n_pairs_hemi; pair_idx++) {
                cov += 
                    (windows_fc_trils[sim_idx][w_i*n_pairs+pair_idx] - L_windows_mean_fc[sim_idx][w_i]) 
                    * (windows_fc_trils[sim_idx][w_j*n_pairs+pair_idx] - L_windows_mean_fc[sim_idx][w_j]);
            }
            L_fcd_trils[sim_idx][window_pair_idx] = cov / (L_windows_ssd_fc[sim_idx][w_i] * L_windows_ssd_fc[sim_idx][w_j]);
        } else if (hemi == 2) {
            for (pair_idx=n_pairs-n_pairs_hemi; pair_idx<n_pairs; pair_idx++) {
                cov += 
                    (windows_fc_trils[sim_idx][w_i*n_pairs+pair_idx] - R_windows_mean_fc[sim_idx][w_i]) 
                    * (windows_fc_trils[sim_idx][w_j*n_pairs+pair_idx] - R_windows_mean_fc[sim_idx][w_j]);
            }
            R_fcd_trils[sim_idx][window_pair_idx] = cov / (R_windows_ssd_fc[sim_idx][w_i] * R_windows_ssd_fc[sim_idx][w_j]);
        }
    }

__global__ void float2double(double **dst, float **src, size_t rows, size_t cols) {
    int row = blockIdx.x;
    int col = blockIdx.y;
    if ((row > rows) | (col > cols)) return;
    dst[row][col] = (float)(src[row][col]);
}

cudaDeviceProp get_device_prop(bool verbose = true) {
    /*
        Gets GPU device properties and prints them to the console.
        Also exits the program if no GPU is found.
    */
    int count;
    cudaError_t error = cudaGetDeviceCount(&count);
    struct cudaDeviceProp prop;
    if (error == cudaSuccess) {
        int device = 0;
        CUDA_CHECK_RETURN( cudaGetDeviceProperties(&prop, device) );
        if (verbose) {
            std::cout << "\nDevice # " << device << ", PROPERTIES: " << std::endl;
            std::cout << "Name: " << prop.name << std::endl;
            std::cout << "totalGlobalMem: " << prop.totalGlobalMem << ", sharedMemPerBlock: " << prop.sharedMemPerBlock; 
            std::cout << ", regsPerBlock: " << prop.regsPerBlock << ", warpSize: " << prop.warpSize << ", memPitch: " << prop.memPitch << std::endl;
            std::cout << "maxThreadsPerBlock: " << prop.maxThreadsPerBlock << ", maxThreadsDim[0]: " << prop.maxThreadsDim[0] 
                << ", maxThreadsDim[1]: " << prop.maxThreadsDim[1] << ", maxThreadsDim[2]: " << prop.maxThreadsDim[2] << std::endl; 
            std::cout << "maxGridSize[0]: " << prop.maxGridSize[0] << ", maxGridSize[1]: " << prop.maxGridSize[1] << ", maxGridSize[2]: " 
                << prop.maxGridSize[2] << ", totalConstMem: " << prop.totalConstMem << std::endl;
            std::cout << "major: " << prop.major << ", minor: " << prop.minor << ", clockRate: " << prop.clockRate << ", textureAlignment: " 
                << prop.textureAlignment << ", deviceOverlap: " << prop.deviceOverlap << ", multiProcessorCount: " << prop.multiProcessorCount << std::endl; 
            std::cout << "kernelExecTimeoutEnabled: " << prop.kernelExecTimeoutEnabled << ", integrated: " << prop.integrated  
                << ", canMapHostMemory: " << prop.canMapHostMemory << ", computeMode: " << prop.computeMode <<  std::endl; 
            std::cout << "concurrentKernels: " << prop.concurrentKernels << ", ECCEnabled: " << prop.ECCEnabled << ", pciBusID: " << prop.pciBusID
                << ", pciDeviceID: " << prop.pciDeviceID << ", tccDriver: " << prop.tccDriver  << std::endl;
        }
    } else {
        std::cout << "No CUDA devices found" << std::endl;
        exit(1);
    }
    return prop;
}

void run_simulations(
    double * corr, u_real * fic_penalties,
    u_real * G_list, u_real * w_EE_list, u_real * w_EI_list, u_real * w_IE_list, u_real **w_IE_fic,
    u_real * SC, int nodes, bool calculate_fic_penalty,
    int time_steps, int BOLD_TR, int _max_fic_trials, bool sim_only,
    gsl_vector * emp_FC_tril, gsl_vector * emp_FCD_tril, int window_size, 
    bool save_output, bool extended_output, std::string sims_out_prefix,
    int N_SIMS, u_real *noise, int output_ts,
    cudaDeviceProp prop, u_real **BOLD_ex, 
    u_real **S_E, u_real **I_E, u_real **r_E, 
    u_real **S_I, u_real **I_I, u_real **r_I,
    bool do_fic, bool adjust_fic, 
    bool *FIC_failed, int *fic_n_trials,
    int n_vols_remove, int corr_len,
    u_real **mean_bold, u_real **ssd_bold,
    int n_windows, int *window_starts, int *window_ends,
    u_real **windows_mean_bold, u_real **windows_ssd_bold,
    u_real **fc_trils, u_real **windows_fc_trils,
    u_real **windows_mean_fc, u_real **windows_ssd_fc,
    int n_pairs, int *pairs_i, int *pairs_j,
    int n_window_pairs, int *window_pairs_i, int *window_pairs_j,
    u_real **fcd_trils, double **d_fc_trils, double **d_fcd_trils,
    gsl_vector * emp_FCD_tril_copy
)
{
    /* 
        Runs a group of reduced Wong Wang simulations in parallel
        on a GPU
    */
    // run simulations
    dim3 numBlocks(N_SIMS);
    dim3 threadsPerBlock(nodes);
    // (third argument is extern shared memory size for S_i_1_E)
    // provide NULL for extended output variables and FIC variables
    bnm<<<numBlocks,threadsPerBlock,nodes*sizeof(u_real)>>>(
        BOLD_ex, 
        S_E, I_E, r_E, 
        S_I, I_I, r_I,
        n_vols_remove,
        SC, G_list, w_EE_list, w_EI_list, w_IE_list,
        N_SIMS, nodes, BOLD_TR, dt, time_steps, noise,
        do_fic, w_IE_fic, 
        adjust_fic, _max_fic_trials, FIC_failed, fic_n_trials,
        extended_output,
        corr_len, true); // true refers to regional_params
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    // calculate mean and sd bold for FC calculation
    bold_stats<<<numBlocks, threadsPerBlock>>>(
        mean_bold, ssd_bold,
        BOLD_ex, N_SIMS, nodes,
        output_ts, corr_len, n_vols_remove);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    // calculate window mean and sd bold for FCD calculations
    numBlocks.y = n_windows;
    window_bold_stats<<<numBlocks,threadsPerBlock>>>(
        BOLD_ex, N_SIMS, nodes, 
        n_windows, window_size+1, window_starts, window_ends,
        windows_mean_bold, windows_ssd_bold);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    // calculate FC and window FCs
    int maxThreadsPerBlock = prop.maxThreadsPerBlock;
    numBlocks.y = ceil((float)n_pairs / (float)maxThreadsPerBlock);
    numBlocks.z = 1 + n_windows; // total FC + windows FCs
    if (prop.maxThreadsPerBlock!=prop.maxThreadsDim[0]) {
        printf("Error: Code not implemented for GPUs in which maxThreadsPerBlock!=maxThreadsDim[0]\n");
        exit(1);
    }
    threadsPerBlock.x = maxThreadsPerBlock;
    threadsPerBlock.y = 1;
    threadsPerBlock.z = 1;
    fc<<<numBlocks, threadsPerBlock>>>(
        fc_trils, windows_fc_trils, BOLD_ex, N_SIMS, nodes, n_pairs, pairs_i, pairs_j,
        output_ts, n_vols_remove, corr_len, mean_bold, ssd_bold,
        n_windows, window_size+1, windows_mean_bold, windows_ssd_bold,
        window_starts, window_ends,
        maxThreadsPerBlock
    );
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    // calculate window mean and sd fc_tril for FCD calculations
    numBlocks.y = 1;
    numBlocks.z = 1;
    threadsPerBlock.x = n_windows;
    if (n_windows >= prop.maxThreadsPerBlock) {
        printf("Error: Mean/ssd FC tril of %d windows cannot be calculated on this device\n", n_windows);
        exit(1);
    }
    window_fc_stats<<<numBlocks,threadsPerBlock>>>(
        windows_mean_fc, windows_ssd_fc,
        NULL, NULL, NULL, NULL, // no need for L and R stats in CMAES
        windows_fc_trils, N_SIMS, n_windows, n_pairs,
        false, 0);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    // calculate FCD
    numBlocks.y = ceil((float)n_window_pairs / (float)maxThreadsPerBlock);
    numBlocks.z = 1;
    if (prop.maxThreadsPerBlock!=prop.maxThreadsDim[0]) {
        printf("Code not implemented for GPUs in which maxThreadsPerBlock!=maxThreadsDim[0]\n");
        exit(1);
    }
    threadsPerBlock.x = maxThreadsPerBlock;
    threadsPerBlock.y = 1;
    threadsPerBlock.z = 1;
    fcd<<<numBlocks, threadsPerBlock>>>(
        fcd_trils, NULL, NULL, // no need for L and R fcd in CMAES
        windows_fc_trils, 
        windows_mean_fc, windows_ssd_fc,
        NULL, NULL, NULL, NULL,
        N_SIMS, n_pairs, n_windows, n_window_pairs, 
        window_pairs_i, window_pairs_j, maxThreadsPerBlock,
        false, 0);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    #ifdef USE_FLOATS
    // Convert FC and FCD to doubles for GOF calculation
    numBlocks.y = n_pairs;
    numBlocks.z = 1;
    threadsPerBlock.x = 1;
    float2double<<<numBlocks, threadsPerBlock>>>(d_fc_trils, fc_trils, N_SIMS, n_pairs);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    numBlocks.y = n_window_pairs;
    float2double<<<numBlocks, threadsPerBlock>>>(d_fcd_trils, fcd_trils, N_SIMS, n_window_pairs);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    #endif

    // GOF calculation (on CPU)
    if (!(sim_only)) {
        u_real fc_corr, fcd_ks, fc_diff;
        for(int IndPar = 0; IndPar < N_SIMS; IndPar++) {
            fc_corr = gsl_stats_correlation(
                emp_FC_tril->data, 1,
                d_fc_trils[IndPar], 1,
                n_pairs
            );
            gsl_vector_memcpy(emp_FCD_tril_copy, emp_FCD_tril);
            fcd_ks = ks_stat(
                emp_FCD_tril_copy->data, 
                d_fcd_trils[IndPar],
                emp_FCD_tril_copy->size, 
                n_window_pairs, 
                NULL);
            corr[IndPar] = 1 + fcd_ks - fc_corr;
            if (use_fc_diff) {
                // get means of simulated and empirical fc
                fc_diff = abs(
                    gsl_stats_mean(emp_FC_tril->data, 1, n_pairs) -
                    gsl_stats_mean(d_fc_trils[IndPar], 1, n_pairs)
                );
                corr[IndPar] += fc_diff;
            }
            // FIC penalty calculation
            if (calculate_fic_penalty) {
                u_real fic_penalty = 0;
                u_real diff_r_E;
                for (int j=0; j<nodes; j++) {
                    diff_r_E = abs(r_E[IndPar][j] - 3);
                    if (diff_r_E > 1) {
                        fic_penalty += 1 - (EXP(-0.05 * (diff_r_E-1)));
                    }
                }
                fic_penalty *= fic_penalty_scale; // scale by BNM_CMAES_FIC_PENALTY_SCALE 
                fic_penalty /= nodes; // average across nodes
                if ((fic_reject_failed) & (FIC_failed[IndPar])) {
                    fic_penalty += 1;
                }
                fic_penalties[IndPar] = fic_penalty;
            }
        }
    }
    // Saving output
    if (save_output) {
        FILE *BOLDout,
            *S_E_out, *I_E_out, *r_E_out, 
            *S_I_out, *I_I_out, *r_I_out, 
            *w_IE_out_file, *w_EE_out_file, *w_EI_out_file;
        if (N_SIMS == 1) {
            // e.g. best CMAES simulation
            BOLDout = fopen((sims_out_prefix + "bold.txt").c_str(), "w");
            S_E_out = fopen((sims_out_prefix + "S_E.txt").c_str(), "w");
            I_E_out = fopen((sims_out_prefix + "I_E.txt").c_str(), "w");
            r_E_out = fopen((sims_out_prefix + "r_E.txt").c_str(), "w");
            S_I_out = fopen((sims_out_prefix + "S_I.txt").c_str(), "w");
            I_I_out = fopen((sims_out_prefix + "I_I.txt").c_str(), "w");
            r_I_out = fopen((sims_out_prefix + "r_I.txt").c_str(), "w");
            w_EE_out_file = fopen((sims_out_prefix + "w_EE.txt").c_str(), "w");
            w_EI_out_file = fopen((sims_out_prefix + "w_EI.txt").c_str(), "w");
            if (do_fic) {
                w_IE_out_file = fopen((sims_out_prefix + "w_IE.txt").c_str(), "w");
                if (adjust_fic) {
                    FILE * FIC_failed_out = fopen((sims_out_prefix + "fic_failed.txt").c_str(), "w");
                    fprintf(FIC_failed_out, "%d", FIC_failed[0]);
                    fclose(FIC_failed_out);
                    FILE * fic_n_trials_out = fopen((sims_out_prefix + "fic_ntrials.txt").c_str(), "w");
                    fprintf(fic_n_trials_out, "%d of max %d", fic_n_trials[0], _max_fic_trials);
                    fclose(fic_n_trials_out);
                }
            }
            for (int i=0; i<output_ts; i++) {
                for (int j=0; j<nodes; j++) {
                    fprintf(BOLDout, "%f ",BOLD_ex[0][i*nodes+j]);
                }
                fprintf(BOLDout, "\n");
            }
            for (int j=0; j<nodes; j++) {
                fprintf(S_E_out, "%f ",S_E[0][j]);
                fprintf(I_E_out, "%f ",I_E[0][j]);
                fprintf(r_E_out, "%f ",r_E[0][j]);
                fprintf(S_I_out, "%f ",S_I[0][j]);
                fprintf(I_I_out, "%f ",I_I[0][j]);
                fprintf(r_I_out, "%f ",r_I[0][j]);
                fprintf(w_EE_out_file, "%f ",w_EE_list[0*nodes+j]);
                fprintf(w_EI_out_file, "%f ",w_EI_list[0*nodes+j]);
                if (do_fic) {
                    fprintf(w_IE_out_file, "%f ", w_IE_fic[0][j]);
                }
            }   
        } else {
            printf("Saving multiple simulations output not implemented\n");
            exit(1);
        }
        fclose(BOLDout);
        fclose(S_E_out);
        fclose(S_I_out);
        fclose(r_E_out);
        fclose(r_I_out);
        fclose(I_E_out);
        fclose(I_I_out);
        fclose(w_EE_out_file);
        fclose(w_EI_out_file);
        if (do_fic) {
            fclose(w_IE_out_file);
        }
    }
}