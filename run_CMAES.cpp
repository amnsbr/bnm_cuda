/*
CMAES optimization of reduced Wong-Wang model (Deco 2014) with heterogeneous
local parameters using CPU or GPU

Optimizes selected parameters of the reduced Wong-Wang model using CMAES 
algorithm. Local parameters (wEE, wEI and wIE) can vary across regions 
based on a set  of coefficients (which are free parameters) and fixed maps.

The particles in each generation are run in parallel on GPU/CPU, which 
calculates simulated BOLD, FC and FCD of all particles. The goodness of fit
caclulation and sampling of the next generation is always done on CPU. 

In GPU implementation, each simulation (for a given set of parameters) 
is run on a single GPU block, and each thread in the block simulates one 
region of the brain. The threads in the block are synchronized after each
integration time step.

CMAES implementation based on code from Kevin Wischnewski

Compile with:
GPU:
cp run_CMAES.cpp run_CMAES.cu && \
nvcc run_CMAES.cu \
    -o run_CMAES_gpu \
    -lm -lgsl -lgslcblas \
    <path-to-libks>/libks.so \
    -I <path-to-libks>/include
CPU:
g++ run_CMAES.cpp \
    -o run_CMAES_cpu \
    -O3 -fopenmp -lm -lgsl -lgslcblas \
    <path-to-libks>/libks.so \
    -I <path-to-libks>/include
Add `-D USE_FLOATS=1` to use float precision instead of doubles

Author: Amin Saberi, Feb 2023
*/
// determine the target hardware based on the compiler
// and use a common macro for mallocing across GPU/CPU
#ifdef __NVCC__ // nvcc
    #warning "Compiling for GPU"
    #define UMALLOC(var, type, size) CUDA_CHECK_RETURN(cudaMallocManaged(&var, sizeof(type) * size))
#else
    #warning "Compiling for CPU"
    #define UMALLOC(var, type, size) var = (type *)malloc(sizeof(type) * size)
#endif
#include "cmaes.cpp"
#include <vector>
#include <map>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_statistics.h>
#include <chrono>
#include <ctime>
#include "helpers.cpp"
#ifdef __NVCC__
    #include "bnm.cu"
#else
    #include "bnm.cpp"
#endif

void calculate_params(
    u_real * G_list, u_real * w_EE_list, u_real * w_EI_list, u_real * w_IE_list,
    std::map<string, u_real>& curr_params, int IndPar, int nodes, 
    std::vector<std::string>& het_params, std::vector<std::string>& homo_params,
    std::vector<std::vector<u_real>>& maps,
    bool do_fic
#ifdef __NVCC__
    , u_real ** w_IE_fic, gsl_matrix * SC_gsl
#endif
)
{
    /*
        Calculates the parameters of `IndPar` simulation based on `curr_params`
        and writes them to <param>_list arrays. 
        The local parameters may be regionally variable based on input maps 
    */
    u_real curr_scaler, curr_regional_param;
    std::map<std::string, u_real> min_regional_params, max_regional_params;
    G_list[IndPar] = curr_params["G"];
    // calculate local parameters
    for (auto het_param: het_params) {
        min_regional_params[het_param] = 1000;
        max_regional_params[het_param] = -1000;
    }
    for (int j=0; j<nodes; j++) {
        // calculate regional parameters based on maps and their scalers
        // for local heterogeneous parameters
        for (auto het_param: het_params) {
            // for local parameters that are heterogeneous calculate
            // a scaler based on the maps and their coefficients
            curr_scaler = 1;
            for (int map_idx=0; map_idx<maps.size(); map_idx++) {
                curr_scaler += curr_params[het_param+"scale_"+std::to_string(map_idx)] * maps[map_idx][j];
            }
            // based on the scaler calculate local parameter of
            // current node/parameter
            curr_regional_param = curr_params[het_param] * curr_scaler; // base * scaler
            // determine min and max parameters for subsequent
            // rescaling
            if (curr_regional_param < min_regional_params[het_param]) {
                min_regional_params[het_param] = curr_regional_param;
            }
            if (curr_regional_param > max_regional_params[het_param]) {
                max_regional_params[het_param] = curr_regional_param;
            }
            // write to <parm>_list
            if (het_param=="wee") {
                w_EE_list[IndPar*nodes+j] = curr_regional_param;
            } else if (het_param=="wei") {
                w_EI_list[IndPar*nodes+j] = curr_regional_param;
            } else if (het_param=="wie") {
                w_IE_list[IndPar*nodes+j] = curr_regional_param;
            } 
        }
        // set the parameter of all regions to the same value if it
        // is a local homogeneous parameter
        for (auto homo_param: homo_params) {
            if (homo_param=="wee") {
                w_EE_list[IndPar*nodes+j] = curr_params["wee"];
            } else if (homo_param=="wei") {
                w_EI_list[IndPar*nodes+j] = curr_params["wei"];
            } else if ((homo_param=="wie")) {
                // note that in FIC+ this will be all 0s and ignored
                w_IE_list[IndPar*nodes+j] = curr_params["wie"];
            }
        }
    }
    // rescale parameter maps to have a min of 0.001
    for (auto het_param: het_params) {
        if (min_regional_params[het_param] < 0.001) {
            for (int j=0; j<nodes; j++) {
                if (het_param=="wee") {
                    w_EE_list[IndPar*nodes+j] -= min_regional_params[het_param] - 0.001;
                } else if (het_param=="wei") {
                    w_EI_list[IndPar*nodes+j] -= min_regional_params[het_param] - 0.001;
                } else if (het_param=="wie") {
                    w_IE_list[IndPar*nodes+j] -= min_regional_params[het_param] - 0.001;
                } 
            }
            max_regional_params[het_param] -= min_regional_params[het_param] - 0.001;
            min_regional_params[het_param] = 0.001;
        }
    }
    #ifdef __NVCC__
    // do analytical FIC for the current particle
    // (not needed on CPU because there fic is done in bnm function)
    if (do_fic) {
        // make a copy of regional wEE and wEI
        double *curr_w_EE, *curr_w_EI;
        curr_w_EE = (double *)malloc(nodes * sizeof(double));
        curr_w_EI = (double *)malloc(nodes * sizeof(double));
        for (int j=0; j<nodes; j++) {
            curr_w_EE[j] = (double)(w_EE_list[IndPar*nodes+j]);
            curr_w_EI[j] = (double)(w_EI_list[IndPar*nodes+j]);
        }
        // do FIC for the current particle
        bool _unstable = false;
        gsl_vector * curr_w_IE = gsl_vector_alloc(nodes);
        analytical_fic(
            SC_gsl, curr_params["G"], curr_w_EE, curr_w_EI,
            curr_w_IE, &_unstable);
        if (_unstable) {
            printf("In simulation #%d FIC solution is unstable. Setting wIE to 1 in all nodes\n", IndPar);
            for (int j=0; j<nodes; j++) {
                w_IE_fic[IndPar][j] = 1.0;
            }
        } else {
            // copy to w_IE_fic which will be passed on to the device
            for (int j=0; j<nodes; j++) {
                w_IE_fic[IndPar][j] = (u_real)gsl_vector_get(curr_w_IE, j);
            }
        }
    }
    #endif
}

int main(int argc, char* argv[]) {
    #ifdef __NVCC__
    // check if any CUDA devices are available
    cudaDeviceProp prop = get_device_prop();
    #endif
    // get cmd arguments
    std::string sc_path, out_path, fc_tril_path, fc_tril_filename, 
        sims_out_dir, fcd_tril_path, CMAES_out_dir, CMAES_log_path,
        out_prefix, sims_out_prefix, maps_path, het_params_str;
    int nodes, time_steps, BOLD_TR, window_size, window_step, rand_seed;
    bool sim_only = false;
    bool no_fcd = false;
    bool extended_output = false;
    std::map<std::string, u_real> params, param_mins, param_maxs, param_diffs;
    std::map<std::string, std::string> param_strings;
    if (argc == 20) {
        sc_path = argv[1];
        out_path = argv[2]; // can be 'same' which would save it in <sc_path>/sims*  & /cmaes*
        fc_tril_path = argv[3];
        fcd_tril_path = argv[4];
        maps_path = argv[5]; // no_maps or txt matrix of (maps, nodes) size
        nodes = atoi(argv[6]);
        param_strings["G"] = argv[7];
        param_strings["wee"] = argv[8];
        param_strings["wei"] = argv[9];
        param_strings["wie"] = argv[10];
        het_params_str = argv[11]; // none or e.g. wee-wei-wie, wee-wie
        time_steps = atoi(argv[12]);
        BOLD_TR = atoi(argv[13]); // msec
        window_step = atoi(argv[14]);
        window_size = atoi(argv[15]);
        rand_seed = atoi(argv[16]);
        lambda = atoi(argv[17]);
        itMax = atoi(argv[18]);
        SeedMW = atoi(argv[19]);
    } else {
        std::cout << "Usage: ./run_CMAES_<cpu|gpu> <sc_path> <out_prefix|'same'> <fc_tril_path> "
            << "<fcd_tril_path> <maps_path> <n_regions> <G_min[-G_max]> <wee_min[-wee_max]> "
            << "<wei_min[-wei_max]> <wie_min[-wie_max]> <het_params> <time_steps> <BOLD_TR> " 
            << "<fcd_step> <fcd_window> <rand_seed> <lambda> <itMax> <SeedMW>" << std::endl;
        exit(0);
    }

    // time_t start, end;
    std::chrono::time_point<std::chrono::system_clock> start, end;
    std::chrono::duration<double> elapsed_seconds;

    int N_SIMS = lambda; // number of parallel simulations in each generation

    // identify global parameters and their specified ranges
    // any parameter with a "-" (that is not at the first position, 
    // i.e., the negative sign) is considered free. Otherwise it will
    // fixed to the specified value
    std::string param_substr, min_substr, max_substr;
    std::vector<std::string> free_params;
    for (auto var: param_strings) {
        param_substr = var.second.substr(1);
        if (param_substr.find('-') != std::string::npos) {
            params[var.first] = -100; // setting them to -100 indicates free parameter
            min_substr = var.second.substr(0, param_substr.find('-')+1); // +1 because the first char was removed from param_substr
            max_substr = var.second.substr(param_substr.find('-')+1+1); // additional +1 to remove "-"
            param_mins[var.first] = atof(min_substr.c_str());
            param_maxs[var.first] = atof(max_substr.c_str());
            // calculate parameter range
            param_diffs[var.first] = param_maxs[var.first] - param_mins[var.first];
            // add it to the list of free parameters and increment the
            // number of free parmaters
            Dimension++;
            free_params.push_back(var.first);
        }
        else {
            params[var.first] = atof(var.second.c_str());
        }
    }

    // specify heterogeneous and homogeneous parameters
    bool heterogeneous = false;
    std::vector<std::string> local_params = {"wee", "wie", "wei"};
    std::vector<std::string> het_params, homo_params;
    for (auto local_param: local_params) {
        if (het_params_str.find(local_param) != std::string::npos) {
            heterogeneous = true;
            het_params.push_back(local_param);
        } else {
            homo_params.push_back(local_param);
        }
    }

    std::vector<std::vector<u_real>> maps;
    if (heterogeneous) {
        // read maps, add free parameters for each map-param combination, and specify
        // its lower and upper range that leads to non-negative values
        u_real curr_map_value, map_max, map_min, scale_min, scale_max;
        int map_idx = 0;
        std::string map_string;
        std::ifstream maps_file(maps_path);
        if (!maps_file) {
            std::cerr << "Error opening maps file" << std::endl;
            exit(1);
        }
        while (std::getline(maps_file, map_string)) {
            std::vector<u_real> curr_map;
            std::istringstream iss(map_string);
            while (iss >> curr_map_value) {
                curr_map.push_back(curr_map_value);
            }
            map_max = *(std::max_element(curr_map.begin(), curr_map.end()));
            map_min = *(std::min_element(curr_map.begin(), curr_map.end()));
            if ((map_min == 0) & (map_max == 1)) {
                // map is min-max normalized
                scale_min = 0;
                scale_max = scale_max_minmax; // defined in constants
            } else {
                // e.g. z-scored
                scale_min = -1 / map_max;
                scale_min = std::ceil(scale_min / 0.01) * 0.01; // round up
                scale_max = -1 / map_min;
                scale_max = std::floor(scale_max / 0.01) * 0.01; // round down 
            }
            maps.push_back(curr_map);
            // add free parameters for each (local parameter - map) combination
            for (auto het_param: het_params) {
                std::string param_name = het_param+"scale_"+std::to_string(map_idx);
                params[param_name] = -100;
                param_mins[param_name] = scale_min;
                param_maxs[param_name] = scale_max;
                param_diffs[param_name] = param_maxs[param_name] - param_mins[param_name];
                Dimension++;
                free_params.push_back(param_name);
            }
            map_idx ++;
        }
    }
    
    // read emp FC and FCD
    gsl_vector * emp_FC_tril, * emp_FCD_tril;
    std::vector<double> emp_FC_tril_vec, emp_FCD_tril_vec;
    double emp_element;
    if (!sim_only) {
        // read emp FC
        std::ifstream emp_FC_tril_file(fc_tril_path);
        if (!emp_FC_tril_file) {
            std::cerr << "Error opening empirical FC file" << std::endl;
            exit(1);
        }
        while (emp_FC_tril_file >> emp_element) {
            emp_FC_tril_vec.push_back(emp_element);
        }
        emp_FC_tril = gsl_vector_alloc(emp_FC_tril_vec.size());
        for (int i = 0; i < emp_FC_tril_vec.size(); i++) {
            gsl_vector_set(emp_FC_tril, i, emp_FC_tril_vec[i]);
        }
        emp_FC_tril_file.close();
        // read emp FCD
        if (fcd_tril_path!=std::string("no_fcd")) {
            std::ifstream emp_FCD_tril_file(fcd_tril_path);
            if (!emp_FCD_tril_file) {
                std::cerr << "Error opening empirical FCD file" << std::endl;
                exit(1);
            }
            while (emp_FCD_tril_file >> emp_element) {
                emp_FCD_tril_vec.push_back(emp_element);
            }
            emp_FCD_tril = gsl_vector_alloc(emp_FCD_tril_vec.size());
            for (int i = 0; i < emp_FCD_tril_vec.size(); i++) {
                gsl_vector_set(emp_FCD_tril, i, emp_FCD_tril_vec[i]);
            }
            emp_FCD_tril_file.close();
            no_fcd = false;
        }
    }

    bool do_fic = (params["wie"] == 0);

    // Read SC
    u_real *SC;
    UMALLOC(SC, u_real, nodes*nodes);
    gsl_matrix *SC_gsl;
    if (do_fic) {
        SC_gsl = gsl_matrix_alloc(nodes, nodes);
    }
    FILE *fp = fopen(sc_path.c_str(), "r");
    if (fp == NULL) {
        printf("Error opening SC file\n");
        exit(1);
    }
    for (int i = 0; i < nodes; i++) {
        for (int j = 0; j < nodes; j++) {
            #ifdef USE_FLOATS
            fscanf(fp, "%f", &SC[i*nodes + j]);
            #else
            fscanf(fp, "%lf", &SC[i*nodes + j]);
            #endif
            if (do_fic) {
                gsl_matrix_set(SC_gsl, i, j, SC[i*nodes + j]);
            }
        }
    }
    fclose(fp);

    // FIC initialization
    #ifdef __NVCC__
    u_real **w_IE_fic;
    #endif
    gsl_vector * curr_w_IE;
    bool calculate_fic_penalty = false;
    int _max_fic_trials = max_fic_trials_cmaes;
    if (do_fic) {
        if (std::find(het_params.begin(), het_params.end(), "wie") != het_params.end()) {
            std::cerr << "Error: wIE cannot be a heterogeneous parameter in FIC+ model" << std::endl;
            exit(1);
        } else {
            calculate_fic_penalty = true;
            extended_output = true;
            curr_w_IE = gsl_vector_alloc(nodes);
            #ifdef __NVCC__
            CUDA_CHECK_RETURN(cudaMallocManaged((void**)&w_IE_fic, sizeof(u_real*) * N_SIMS));
            #endif
        }
    }

    // Specify output directory
    if (out_path==std::string("same")) {
        out_path = sc_path;
        out_path = out_path.replace(out_path.length()-4, 4, "");
    } 
    mkdir(out_path.c_str(), 0751); 
    sims_out_dir = out_path + "/sims";
    CMAES_out_dir = out_path + "/cmaes";
    if (heterogeneous) {
        sims_out_dir += "_multimaps";
        CMAES_out_dir += "_multimaps";
    }
    #ifdef __NVCC__
    sims_out_dir += "_gpu";
    CMAES_out_dir += "_gpu";
    #endif
    mkdir(sims_out_dir.c_str(), 0751);
    mkdir(CMAES_out_dir.c_str(), 0751); 
    // and output filenames
    std::string::size_type pos = fc_tril_path.find_last_of("/");
    fc_tril_filename = fc_tril_path.substr(pos+1, fc_tril_path.length()-pos-17); // excluding _desc-FCtril.txt
    out_prefix = fc_tril_filename;
    for (auto var: param_strings) {
        out_prefix += "_" + var.first + "_" + param_strings[var.first];
    }
    out_prefix += "_het-" + het_params_str;
    out_prefix += "_SeedMW-" + std::to_string(SeedMW);
    out_prefix += "_SeedSim-" + std::to_string(rand_seed);
    out_prefix += "_n-" + std::to_string(itMax) + "x" + std::to_string(lambda);
    CMAES_log_path = CMAES_out_dir + "/" + out_prefix + ".txt";
    sims_out_prefix = sims_out_dir + "/" + out_prefix + "_";
    FILE * log_file = fopen(CMAES_log_path.c_str(), "w");

    fprintf(log_file, "Running %dD CMAES\nSC: %s\nFC: %s\n", Dimension, sc_path.c_str(), fc_tril_path.c_str());
    printf("Running %dD CMAES\nSC: %s\nFC: %s\n", Dimension, sc_path.c_str(), fc_tril_path.c_str());
    if (heterogeneous) {
        fprintf(log_file, "Maps: %s\n", maps_path.c_str());
        printf("Maps: %s\n", maps_path.c_str());
    }
    fprintf(log_file, "Free parameters: ");
    printf("Free parameters: ");
    for (auto var: param_mins) {
        fprintf(log_file, "%s (%f-%f) ", var.first.c_str(), param_mins[var.first], param_maxs[var.first]);
        printf("%s (%f-%f) ", var.first.c_str(), param_mins[var.first], param_maxs[var.first]);
    }
    fprintf(log_file, "\nFCD step %d, FCD window %d, Simulation random seed %d, CMAES random seed %d, sigma %f, alphacov %f, gamma_scale %f, bound_soft_edge %f, Variante %d, early stop gens %d, early stop tol %f\n", window_step, window_size, rand_seed, SeedMW, sigma, alphacov, gamma_scale, bound_soft_edge, Variante, early_stop_gens, early_stop_tol);
    printf("\nFCD step %d, FCD window %d, Simulation random seed %d, CMAES random seed %d, sigma %f, alphacov %f, gamma_scale %f, bound_soft_edge %f, Variante %d, early stop gens %d, early stop tol %f\n", window_step, window_size, rand_seed, SeedMW, sigma, alphacov, gamma_scale, bound_soft_edge, Variante, early_stop_gens, early_stop_tol);

#ifdef __NVCC__
    // GPU initializations
    // make a copy of constants that will be used in the device
    cudaMemcpyToSymbol(d_I_SAMPLING_START, &I_SAMPLING_START, sizeof(int));
    cudaMemcpyToSymbol(d_I_SAMPLING_END, &I_SAMPLING_END, sizeof(int));
    cudaMemcpyToSymbol(d_I_SAMPLING_DURATION, &I_SAMPLING_DURATION, sizeof(int));
    cudaMemcpyToSymbol(d_init_delta, &init_delta, sizeof(u_real));

    // FIC adjustment init
    // adjust_fic is set to true by default but only for
    // assessing FIC success. With default max_fic_trials_cmaes = 0
    // no adjustment is done but FIC success is assessed
    bool adjust_fic = do_fic & numerical_fic;
    bool *FIC_failed;
    CUDA_CHECK_RETURN(cudaMallocManaged(&FIC_failed, sizeof(bool) * N_SIMS));
    int *fic_n_trials;

    CUDA_CHECK_RETURN(cudaMallocManaged(&fic_n_trials, sizeof(int) * N_SIMS));
    u_real **S_E, **I_E, **r_E, **S_I, **I_I, **r_I;
    if (do_fic) { // we only need r_E but currently bnm writes all or none of the extended output
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&S_E, sizeof(u_real*) * N_SIMS));
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&I_E, sizeof(u_real*) * N_SIMS));
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&r_E, sizeof(u_real*) * N_SIMS));
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&S_I, sizeof(u_real*) * N_SIMS));
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&I_I, sizeof(u_real*) * N_SIMS));
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&r_I, sizeof(u_real*) * N_SIMS));
    }

    // allocated memory for BOLD time-series of all simulations
    // BOLD_ex will be a 2D array of size N_SIMS x (nodes x output_ts)
    u_real TR        = (u_real)BOLD_TR / 1000; // (s) TR of fMRI data
    int   output_ts = (time_steps / (TR / model_dt))+1; // Length of BOLD time-series written to HDD
    size_t bold_size = nodes * output_ts;
    u_real **BOLD_ex;
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&BOLD_ex, sizeof(u_real*) * N_SIMS));

    // specify n_vols_remove (for extended output and FC calculations)
    int n_vols_remove = bold_remove_s * 1000 / BOLD_TR; // 30 seconds

    // preparing FC calculations
    int corr_len = output_ts - n_vols_remove;
    if (corr_len < 2) {
        printf("Number of volumes (after removing initial volumes) is too low for FC calculations\n");
        exit(1);
    }
    u_real **mean_bold, **ssd_bold;
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&mean_bold, sizeof(u_real*) * N_SIMS));
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&ssd_bold, sizeof(u_real*) * N_SIMS));
    int n_pairs = ((nodes) * (nodes - 1)) / 2;
    int rh_idx;
    if (exc_interhemispheric) {
        assert((nodes % 2) == 0);
        rh_idx = nodes / 2; // assumes symmetric number of parcels and L->R order
        n_pairs -= pow(rh_idx, 2); // exc the middle square
    }
    if (n_pairs!=emp_FC_tril->size) {
        printf("Empirical and simulated FC size do not match\n");
        exit(1);
    }
    // create a mapping between pair_idx and i and j
    int *pairs_i, *pairs_j;
    int curr_idx = 0;
    CUDA_CHECK_RETURN(cudaMallocManaged(&pairs_i, sizeof(int) * n_pairs));
    CUDA_CHECK_RETURN(cudaMallocManaged(&pairs_j, sizeof(int) * n_pairs));
    for (int i=0; i < nodes; i++) {
        for (int j=0; j < nodes; j++) {
            if (i > j) {
                if (exc_interhemispheric) {
                    // skip if each node belongs to a different hemisphere
                    if ((i < rh_idx) ^ (j < rh_idx)) {
                        continue;
                    }
                }
                pairs_i[curr_idx] = i;
                pairs_j[curr_idx] = j;
                curr_idx++;
            }
        }
    }
    // allocate memory for fc trils
    u_real **fc_trils;
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&fc_trils, sizeof(u_real*) * N_SIMS));

    // FCD preparation
    // calculate number of windows and window start/end indices
    bool drop_edges = true;
    std::vector<int> _window_starts, _window_ends;
    int first_center, last_center, window_center, window_start, window_end;
    if (drop_edges) {
        first_center = window_size / 2;
        last_center = corr_len - 1 - (window_size / 2);
    } else {
        first_center = 0;
        last_center = corr_len - 1;
    }
    first_center += n_vols_remove;
    last_center += n_vols_remove;
    int n_windows = 0;
    window_center = first_center;
    while (window_center <= last_center) {
        window_start = window_center - (window_size/2);
        if (window_start < 0)
            window_start = 0;
        window_end = window_center + (window_size/2);
        if (window_end >= output_ts)
            window_end = output_ts-1;
        _window_starts.push_back(window_start);
        _window_ends.push_back(window_end);
        window_center += window_step;
        n_windows ++;
    }
    if (n_windows == 0) {
        printf("Error: Number of windows is 0\n");
        exit(1);
    }
    int *window_starts, *window_ends;
    CUDA_CHECK_RETURN(cudaMallocManaged(&window_starts, sizeof(int) * n_windows));
    CUDA_CHECK_RETURN(cudaMallocManaged(&window_ends, sizeof(int) * n_windows));
    for (int i=0; i<n_windows; i++) {
        window_starts[i] = _window_starts[i];
        window_ends[i] = _window_ends[i];
    }
    // allocate memory for mean and ssd BOLD of each window
    // (n_sims x n_windows x nodes)
    u_real **windows_mean_bold, **windows_ssd_bold;
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&windows_mean_bold, sizeof(u_real*) * N_SIMS));
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&windows_ssd_bold, sizeof(u_real*) * N_SIMS));
    u_real **windows_fc_trils;
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&windows_fc_trils, sizeof(u_real*) * N_SIMS));
    // allocate memory for mean and ssd fc_tril of each window
    // (n_sims x n_windows)
    u_real **windows_mean_fc, **windows_ssd_fc;
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&windows_mean_fc, sizeof(u_real*) * N_SIMS));
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&windows_ssd_fc, sizeof(u_real*) * N_SIMS));
    // create a mapping between window_pair_idx and i and j
    int n_window_pairs = (n_windows * (n_windows-1)) / 2;
    int *window_pairs_i, *window_pairs_j;
    curr_idx = 0;
    CUDA_CHECK_RETURN(cudaMallocManaged(&window_pairs_i, sizeof(int) * n_window_pairs));
    CUDA_CHECK_RETURN(cudaMallocManaged(&window_pairs_j, sizeof(int) * n_window_pairs));
    for (int i=0; i < n_windows; i++) {
        for (int j=0; j < n_windows; j++) {
            if (i > j) {
                window_pairs_i[curr_idx] = i;
                window_pairs_j[curr_idx] = j;
                curr_idx++;
            }
        }
    }
    // allocate memory for fcd trils
    u_real **fcd_trils;
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&fcd_trils, sizeof(u_real*) * N_SIMS));

    #ifdef USE_FLOATS
    // allocate memory for double versions of fc and fcd trils on CPU
    double ** d_fc_trils;
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&d_fc_trils, sizeof(double*) * N_SIMS));
    double ** d_fcd_trils;
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&d_fcd_trils, sizeof(double*) * N_SIMS));
    #else
    // use d_fc_trils and d_fcd_trils as aliases for fc_trils and fcd_trils
    // which will later be used for GOF calculations
    #define d_fc_trils fc_trils
    #define d_fcd_trils fcd_trils
    #endif


    // allocate memory per each simulation
    for (int sim_idx=0; sim_idx<N_SIMS; sim_idx++) {
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&BOLD_ex[sim_idx], sizeof(u_real) * bold_size));
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&mean_bold[sim_idx], sizeof(u_real) * nodes));
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&ssd_bold[sim_idx], sizeof(u_real) * nodes));
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&fc_trils[sim_idx], sizeof(u_real) * n_pairs));
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&windows_mean_bold[sim_idx], sizeof(u_real) * n_windows * nodes));
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&windows_ssd_bold[sim_idx], sizeof(u_real) * n_windows * nodes));
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&windows_fc_trils[sim_idx], sizeof(u_real) * n_windows * n_pairs));
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&windows_mean_fc[sim_idx], sizeof(u_real) * n_windows));
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&windows_ssd_fc[sim_idx], sizeof(u_real) * n_windows));
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&fcd_trils[sim_idx], sizeof(u_real) * n_window_pairs));
        #ifdef USE_FLOATS
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&d_fc_trils[sim_idx], sizeof(double) * n_pairs));
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&d_fcd_trils[sim_idx], sizeof(double) * n_window_pairs));
        #endif
        if (do_fic) {
            CUDA_CHECK_RETURN(cudaMallocManaged((void**)&w_IE_fic[sim_idx], sizeof(u_real) * nodes));
            CUDA_CHECK_RETURN(cudaMallocManaged((void**)&S_E[sim_idx], sizeof(u_real) * nodes));
            CUDA_CHECK_RETURN(cudaMallocManaged((void**)&I_E[sim_idx], sizeof(u_real) * nodes));
            CUDA_CHECK_RETURN(cudaMallocManaged((void**)&r_E[sim_idx], sizeof(u_real) * nodes));
            CUDA_CHECK_RETURN(cudaMallocManaged((void**)&S_I[sim_idx], sizeof(u_real) * nodes));
            CUDA_CHECK_RETURN(cudaMallocManaged((void**)&I_I[sim_idx], sizeof(u_real) * nodes));
            CUDA_CHECK_RETURN(cudaMallocManaged((void**)&r_I[sim_idx], sizeof(u_real) * nodes));
        }
    }

    // pre-calculate normally-distributed noise on CPU
    printf("Noise precalculation...\n");
    start = std::chrono::high_resolution_clock::now();
    int noise_size = nodes * (time_steps+1) * 10 * 2; // +1 for inclusive last time point, 2 for E and I
    std::mt19937 rand_gen(rand_seed);
    std::normal_distribution<float> normal_dist(0, 1);
    u_real *noise;
    CUDA_CHECK_RETURN(cudaMallocManaged(&noise, sizeof(u_real) * noise_size));
    for (int i = 0; i < noise_size; i++) {
        #ifdef USE_FLOATS
        noise[i] = normal_dist(rand_gen);
        #else
        noise[i] = (double)normal_dist(rand_gen);
        #endif
    }
    end = std::chrono::high_resolution_clock::now();
    elapsed_seconds = end - start;
    printf("took %lf s\n", elapsed_seconds.count());
#endif // __NVCC__

    // create the parameter lists. parameters will be calculated inside CMAES loop
    u_real *G_list, *w_EE_list, *w_EI_list, *w_IE_list;
    UMALLOC(G_list, u_real, N_SIMS);
    // w_EE, w_EI and w_IE can vary across nodes
    UMALLOC(w_EE_list, u_real, N_SIMS * nodes);
    UMALLOC(w_EI_list, u_real, N_SIMS * nodes);
    UMALLOC(w_IE_list, u_real, N_SIMS * nodes);
    // CMAES initializations
    weights = new double[lambda];
    Gamma = new double[Dimension];
    lb = new double[Dimension];
    ub = new double[Dimension];
    ksi = new double[Dimension];
    psigma = new double[Dimension];
    pc = new double[Dimension];
    m = new double[Dimension];
    //   mrep = new double[Dimension];
    mpref = new double[Dimension];
    malt = new double[Dimension];
    IQAhist = new double[(int)(20+std::ceil((float)(3*Dimension)/(float)lambda))];
    IQAhistsort = new double[(int)(20+std::ceil((float)(3*Dimension)/(float)lambda))];
    CBVfeas = new double[Dimension+1];
    CBGVariables = new double[Dimension+1];
    ww = new double[lambda];
    kk = new double[lambda];
    BDz = new double[Dimension];
    Temporaer = new double[Dimension];
    corr = new double[lambda];
    corrraw = new double[lambda];
    Helper = new double[Dimension];
    C1d = new double[Dimension*Dimension];
    indexx = new int[lambda];
    u_real *fic_penalties = new u_real[lambda];

    C = new double*[Dimension];
    B = new double*[Dimension];
    D = new double*[Dimension];
    Variables = new double*[Dimension];
    Variablesfeas = new double*[Dimension];
    Variablespref = new double*[Dimension];

    for (int i = 0; i < Dimension; i++){
        C[i] = new double[Dimension];
        B[i] = new double[Dimension];
        D[i] = new double[Dimension];
        Variables[i] = new double[lambda];
        Variablesfeas[i] = new double[lambda];
        Variablespref[i] = new double[lambda];
        lb[i] = 0; // lower feasible bound
        ub[i] = 1; // uper feasible bound
    }
    // After defining lambda (e.g. in an input file), mue must be set  
    mue = std::floor(lambda/2); 

    it = CMAES_Points(); // first function call to prepare algorithm and set "it" to 1
    bool save_output = false; // do not save the BOLD etc. output inside the CMAES loop
    std::map<string, u_real> curr_params;
    std::map<std::string, u_real> min_regional_params, max_regional_params;
    #ifdef __NVCC__
    gsl_vector * emp_FCD_tril_copy = gsl_vector_alloc(emp_FCD_tril->size); // for fcd_ks
    #endif

    while (it < itMax){ // optimization procedure
        printf("Iteration %d\n", it);
        fprintf(log_file, "Iteration %d\n", it);
        // Defining the global and local parameters of all particles 
        // into G_list, w_EE_list, w_EI_list and w_IE_list
        for(int IndPar = 0; IndPar < lambda; IndPar++) {
            std::map<string, u_real> curr_params;
            curr_params = params;
            for (int f=0; f<free_params.size(); f++) {
                curr_params[free_params[f]] = param_diffs[free_params[f]] * Variablesfeas[f][IndPar] + param_mins[free_params[f]];
            }
            calculate_params(
                G_list, w_EE_list, w_EI_list, w_IE_list,
                curr_params, IndPar, nodes, 
                het_params, homo_params, maps,
                do_fic
            #ifdef __NVCC__
                , w_IE_fic, SC_gsl
            #endif
            );
        }
        // Running the simulations
        start = std::chrono::high_resolution_clock::now();
        #ifndef __NVCC__
        // on CPU
        run_simulations(
            corr, fic_penalties,
            G_list, w_EE_list, w_EI_list, w_IE_list,
            SC, SC_gsl, nodes, N_SIMS, calculate_fic_penalty,
            time_steps, BOLD_TR, rand_seed, _max_fic_trials,
            emp_FC_tril, emp_FCD_tril, sim_only, 
            no_fcd, window_step, window_size, 
            sims_out_prefix, sim_verbose, save_output, extended_output
        );
        #else
        // on GPU
        run_simulations(
            corr, fic_penalties,
            G_list, w_EE_list, w_EI_list, w_IE_list, w_IE_fic,
            SC, nodes, calculate_fic_penalty,
            time_steps, BOLD_TR, _max_fic_trials, sim_only,
            emp_FC_tril, emp_FCD_tril, window_size, 
            save_output, extended_output, sims_out_prefix,
            N_SIMS, noise, output_ts,
            prop, BOLD_ex,
            S_E, I_E, r_E, S_I, I_I, r_I,
            do_fic, adjust_fic, 
            FIC_failed, fic_n_trials,
            n_vols_remove, corr_len,
            mean_bold, ssd_bold,
            n_windows, window_starts, window_ends,
            windows_mean_bold, windows_ssd_bold,
            fc_trils, windows_fc_trils,
            windows_mean_fc, windows_ssd_fc,
            n_pairs, pairs_i, pairs_j,
            n_window_pairs, window_pairs_i, window_pairs_j,
            fcd_trils, d_fc_trils, d_fcd_trils,
            emp_FCD_tril_copy
        );
        #endif
        end = std::chrono::high_resolution_clock::now();
        elapsed_seconds = end - start;
        printf("took %lf s\n", elapsed_seconds.count());

        // Boundary and FIC penalty calculation and saving the logs
        for(int IndPar = 0; IndPar < lambda; IndPar++) { 
            if (calculate_fic_penalty) {
                corr[IndPar] += fic_penalties[IndPar];
            }
            corrraw[IndPar] = corr[IndPar]; // important for sorting and penalizing
            kk[IndPar] = -corr[IndPar]; // important for sorting and penalizing, no minus here
        }

        penalty = 1;
        it = CMAES_Points(); // "it" will not be increased here, this function call is only needed for the computation of penalty terms

        for (int i = 0; i < lambda; i++){
            printf("\t IndPar %d: ", i);
            fprintf(log_file, "\t IndPar %d: ", i);
            curr_params = params;
            for (int j = 0; j < Dimension; j++){
                curr_params[free_params[j]] = param_diffs[free_params[j]] * Variablesfeas[j][i] + param_mins[free_params[j]];
                corr[i] += Gamma[j]/ksi[j]/Dimension*(Variablespref[j][i]-Variables[j][i])*(Variablespref[j][i]-Variables[j][i]);
                printf("%s=%f,", free_params[j].c_str(), curr_params[free_params[j]]);
                fprintf(log_file, "%s=%f,", free_params[j].c_str(), curr_params[free_params[j]]);
            }
            if (do_fic) {
                printf("gof=%f,fic_penalty=%f,penalized_cost=%f\n", kk[i]+fic_penalties[i], fic_penalties[i], corr[i]);
                fprintf(log_file, "gof=%f,fic_penalty=%f,penalized_cost=%f\n", kk[i]+fic_penalties[i], fic_penalties[i], corr[i]);
            } else {
                printf("gof=%f,penalized_cost=%f\n", kk[i], corr[i]);
                fprintf(log_file, "gof=%f,penalized_cost=%f\n", kk[i], corr[i]);
            }
        }
        it = CMAES_Points(); // Function call to increase "it" and determine sample points for next iteration

    } // end of optimization procedure (while-loop)


    // Save the best feasible parameters ("CBVfeas[0:Dimension-1]") and corresponding best goal function value ("CBVfeas[Dimension")
    curr_params = params;
    printf("Best feasible parameters: ");
    fprintf(log_file, "Best feasible parameters: ");
    for (int f=0; f<free_params.size(); f++) {
        curr_params[free_params[f]] = param_diffs[free_params[f]] * CBVfeas[f] + param_mins[free_params[f]];
        printf("%s=%f,", free_params[f].c_str(), curr_params[free_params[f]]);
        fprintf(log_file, "%s=%f,", free_params[f].c_str(), curr_params[free_params[f]]);
    }
    // Rerun the best simulation and save the output
    N_SIMS=1;
    int IndPar=0; // uses the memory allocated for the first particle
    save_output = true;
    extended_output = true;
    calculate_params(
        G_list, w_EE_list, w_EI_list, w_IE_list,
        curr_params, IndPar, nodes,
        het_params, homo_params, maps,
        do_fic
    #ifdef __NVCC__
        , w_IE_fic, SC_gsl
    #endif
    );
    // Running the simulations
    start = std::chrono::high_resolution_clock::now();
    #ifndef __NVCC__
    // on CPU
    run_simulations(
        corr, fic_penalties,
        G_list, w_EE_list, w_EI_list, w_IE_list,
        SC, SC_gsl, nodes, N_SIMS, calculate_fic_penalty,
        time_steps, BOLD_TR, rand_seed, _max_fic_trials,
        emp_FC_tril, emp_FCD_tril, sim_only, 
        no_fcd, window_step, window_size, 
        sims_out_prefix, sim_verbose, save_output, extended_output
    );
    #else
    // on GPU
    // allocate memory for extended output of the best simulation
    if (!(do_fic)) {
        // In FIC+ memory is already allocated to these variables
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&S_E, sizeof(u_real*)));
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&I_E, sizeof(u_real*)));
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&r_E, sizeof(u_real*)));
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&S_I, sizeof(u_real*)));
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&I_I, sizeof(u_real*)));
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&r_I, sizeof(u_real*)));
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&S_E[0], sizeof(u_real) * nodes));
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&I_E[0], sizeof(u_real) * nodes));
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&r_E[0], sizeof(u_real) * nodes));
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&S_I[0], sizeof(u_real) * nodes));
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&I_I[0], sizeof(u_real) * nodes));
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&r_I[0], sizeof(u_real) * nodes));
    }
    run_simulations(
        corr, fic_penalties,
        G_list, w_EE_list, w_EI_list, w_IE_list, w_IE_fic,
        SC, nodes, calculate_fic_penalty,
        time_steps, BOLD_TR, _max_fic_trials, sim_only,
        emp_FC_tril, emp_FCD_tril, window_size, 
        save_output, extended_output, sims_out_prefix,
        N_SIMS, noise, output_ts,
        prop, BOLD_ex, 
        S_E, I_E, r_E, S_I, I_I, r_I,
        do_fic, adjust_fic, 
        FIC_failed, fic_n_trials,
        n_vols_remove, corr_len,
        mean_bold, ssd_bold,
        n_windows, window_starts, window_ends,
        windows_mean_bold, windows_ssd_bold,
        fc_trils, windows_fc_trils,
        windows_mean_fc, windows_ssd_fc,
        n_pairs, pairs_i, pairs_j,
        n_window_pairs, window_pairs_i, window_pairs_j,
        fcd_trils, d_fc_trils, d_fcd_trils,
        emp_FCD_tril_copy
    );
    #endif
    // write the GOF of the best simulation
    if (do_fic) {
        printf("gof=%f,fic_penalty=%f,gof_fic_penalty=%f\n", -corr[IndPar], fic_penalties[IndPar], CBVfeas[Dimension]);
        fprintf(log_file, "gof=%f,fic_penalty=%f,gof_fic_penalty=%f\n", -corr[IndPar], fic_penalties[IndPar], CBVfeas[Dimension]);
    } else {
        printf("gof=%f\n", CBVfeas[Dimension]);
        fprintf(log_file, "gof=%f\n", CBVfeas[Dimension]);
    }

    delete[] weights; 
    delete[] Gamma;
    delete[] lb;
    delete[] ub;
    delete[] ksi;
    delete[] psigma;
    delete[] pc;
    delete[] m;
    //   delete[] mrep;
    delete[] mpref;
    delete[] malt;
    delete[] IQAhist;
    delete[] IQAhistsort;
    delete[] CBVfeas;
    delete[] CBGVariables;
    delete[] ww;
    delete[] kk;
    delete[] BDz;
    delete[] Temporaer;
    delete[] corr;
    delete[] corrraw;
    delete[] Helper;
    delete[] C1d;
    delete[] indexx;

    for (int i = 0; i < Dimension; i++){
        delete[] C[i];
        delete[] B[i];
        delete[] D[i];
        delete[] Variables[i];
        delete[] Variablesfeas[i];
        delete[] Variablespref[i];
    }


    delete[] C;
    delete[] B;
    delete[] D;
    delete[] Variables;
    delete[] Variablesfeas;
    delete[] Variablespref;

    ::fclose(log_file);
}
