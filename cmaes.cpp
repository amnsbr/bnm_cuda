/* 
Covariance Matrix Adaptation Evolution Strategy (CMAES)

In order to optimize a given goal function (e.g. correlation between sFC and eFC),
the function CMAES_Points is repetitively called.
CMAES_Points initializes the optimization, computes the sample points
for every iteration, penalizes unfeasible solutions and increases the
current number of iterations by 1.

This code contains 2 variants of the CMAES algorithm,
one from 2006 and a newer one from 2016.

N. Hansen. The CMA Evolution Strategy: A Comparing Review (2006). https://doi.org/10.1007/3-540-32494-1_4
N. Hansen. The CMA Evolution Strategy: A Tutorial (2016). https://doi.org/10.48550/arXiv.1604.00772

Boundary handling derived from
N. Hansen et al. A Method for Handling Uncertainty in Evolutionary Optimization With 
an Application to Feedback Control of Combustion (2009) doi: 10.1109/TEVC.2008.924423

Code written by Kevin Wischnewski (k.wischnewski@fz-juelich.de), strongly inspired by
N. Hansen's implementations in C (https://github.com/cma-es/c-cmaes)
and MATLAB (https://cma-es.github.io/cmaes.m).

N. Hansen's homepage (http://www.cmap.polytechnique.fr/~nikolaus.hansen/) contains
further useful information about the algorithm.
*/


#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <ios>
#include <fstream>
#include <iomanip>
#include <string.h>
#include <cmath>
#include <complex>
#include <vector>
#include <random>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>
#include <algorithm>
#include "helpers.cpp"
#include "constants.h"
using namespace std;

#ifndef AND
#define AND &&
#endif

#ifndef OR
#define OR ||
#endif

#ifndef MAX
#define MAX(a,b) (((a) < (b)) ? (b) : (a))
#endif

#ifndef MIN
#define MIN(a,b) (((a) > (b)) ? (b) : (a))
#endif

#ifndef MYSIGN
#define MYSIGN(a) (((a) < 0.0) ? (-1) : (1))
#endif

// Scalar algorithm constants + helping variables 
double cc, ccov, csigma, dsigma, chiN, muecov, mueeff, s1, s2, t1, t2, Summe, normpsigma2;
double c1, cmue, alphamueminus, alphamueeffminus, alphaposdefminus, mueeffminus, normHelper2;
double Q1, Q3, IQA, Diagonale;

// Memory allocation for vectors
double *weights;
double *Gamma;
double *lb;
double *ub;
double *ksi;
double *psigma;
double *pc;
double *m;
// double *mrep;
double *mpref;
double *malt;
double *IQAhist;
double *IQAhistsort;
double *CBVfeas;
double *CBGVariables;
double *ww;
double *kk;
double *BDz;
double *Temporaer;
double *corr;
double *corrraw;
double *Helper;
double *C1d;
int *indexx;

// Memory allocation for matrices
double **C; 
double **B;
double **D;
double **Variables;
double **Variablesfeas;
double **Variablespref; // preferred variables excluding the soft bound




int lambda, mue, Dimension, itMax, SeedMW, Wert, Wert2, hsigma;
// lambda - Constant number of sample points in every iteration (has to be defined by users)
// mue - Automatically computed from lambda (best points of a given iteration)
// Dimension - Number of model parameters to be optimized (has to be defined by users)
// itMax - Max. number of iteration steps (has to be defined by users)
// SeedMW - Random seed for initial distribution mean (has to be defined by users)
// Variante - "6"--2006 version, "16"--2016 version (has to be defined by users)
// Remaining variables - Helping variables and other algorithm-related quantities

int it = 0;

int itn = 0;
int Cancel = 0;
int penalty = 0; // switches between 0 and 1 after new sample points were generated / penalized

std::vector<double> tVektor(early_stop_gens, 0.0);

int CMAES_Points(void){

int i,j,k;

std::default_random_engine generatorMW;
generatorMW.seed(SeedMW);
std::uniform_real_distribution<double> uni_distrMW(0, 1.0); // only for initial distribution mean

std::default_random_engine generatorADr;
generatorADr.seed(11+it); 
std::normal_distribution<double> norm_distrADr(0, 1.0); // for sample point updates



if (penalty == 0){ // 
if (it == 0){ // Initialization

	s1 = 0;
	s2 = 0;
	
	for (i = 0; i < mue; i++){
		
		if (Variante == 6){ // Version 2006
			weights[i] = log(mue+1) - log(i + 1); 
		}
		else if (Variante == 16){ // Version 2016
			weights[i] = log((lambda+1)/2.) - log(i + 1); 
		}
		
		s1 += weights[i];
		s2 += weights[i]*weights[i];
	}
	
	for (i = 0; i < mue; i++){ // normalize positive weights
		weights[i] /= s1;
	}
	
	mueeff = (s1*s1)/s2; 
	muecov = mueeff; 
	
	if (Variante == 16){
		s1 = 0;
		s2 = 0;
		
		for (i = mue; i < lambda; i++){
			weights[i] = log((lambda+1)/2.) - log(i + 1); // Version 2016, negative weights
			s1 += weights[i]; // <= 0
			s2 += weights[i]*weights[i];
		}
		
		mueeffminus = (s1*s1)/s2;
		c1 = alphacov/(((Dimension+1.3)*(Dimension+1.3))+mueeff); // Version 2016
		t1 = alphacov*(mueeff-2+(1/mueeff))/(((Dimension+2)*(Dimension+2))+(alphacov*mueeff/2));
		cmue = ((1-c1) < t1) ? (1-c1):t1; // Minimum, Version 2016
		
		alphamueminus = 1+(c1/cmue);
		alphamueeffminus = 1+((2*mueeffminus)/(mueeff+2));
		alphaposdefminus = (1-c1-cmue)/(Dimension*cmue);
		t2 = MIN(MIN(alphamueminus,alphamueeffminus),alphaposdefminus);
		
		for (i = mue; i < lambda; i++){
			weights[i] /= (-1*s1);
			weights[i] *= t2;
		}
		cc = (4+(mueeff/Dimension))/(Dimension+4+(2*mueeff/Dimension)); // Version 2016
		csigma = (mueeff+2)/(Dimension+mueeff+5); // Version 2016
		
	}
	
	if (Variante == 6){
		cc = 4./(Dimension+4); // Version 2006
		csigma = (mueeff+2)/(Dimension+mueeff+3); // Version 2006, otherwise 5 in denominator
	}
	

	t1 = 2./((Dimension+sqrt(2))*(Dimension+sqrt(2)));
	t2 = ((2*mueeff)-1)/((Dimension+2)*(Dimension+2)+mueeff);
	t2 = (t2 > 1) ? 1:t2; // Minimum
	ccov = (1/muecov)*t1+(1-(1/muecov))*t2; 
	
	dsigma = 1+2*MAX(0.,sqrt((mueeff-1)/(Dimension+1.))-1)+csigma; 
	chiN = sqrt(Dimension)*(1-1./(4*Dimension)+1./(21*Dimension*Dimension)); 
	
	CBVfeas[Dimension] = 0; // current best function value (of feasible points)
	CBGVariables[Dimension] = -(2.1/0); // best function value of all iterations, initially -Inf
	
	for (i = 0; i < Dimension; i++){
		CBVfeas[i] = 0;
		CBGVariables[i] = 0;
		psigma[i] = 0;
		pc[i] = 0;
		malt[i] = 0; // mean from "previous" iteration
		Gamma[i] = 0;
		m[i] = uni_distrMW(generatorMW); // first distribution mean guaranteed to be feasible
		for (j = 0; j < Dimension; j++){
			if (i == j){
				C[i][j] = 1;
				D[i][j] = 1;
				B[i][j] = 1;
			}
			else {
				C[i][j] = 0;
				D[i][j] = 0;
				B[i][j] = 0;
			}
		}
	}
	

	for (i = 0; i < Dimension; i++){
		for (j = 0; j < lambda; j++){
			Variables[i][j] = m[i] + sigma*norm_distrADr(generatorADr); // first generation of sample points
		} // may be unfeasible (will then be penalized)
	}
	
	for (i = 0; i < (int)(20+std::ceil((float)(3*Dimension)/(float)lambda)); i++){
		IQAhist[i] = 0;
	}
	
}


else { // all following iterations
	for (i = 0; i < lambda; i++){
		ww[i] = corr[i];
	}
	
	std::sort(corr, corr + lambda); // sort goal function values and identify best index
	Wert = 0;
	Wert2 = 0;
	for (i = 0; i < lambda; i++){
		if (Wert2 > 1){
			Wert2 -= 1;
			continue;
		}
		Wert2 = 0;
		for (j = 0; j < lambda; j++){
			if (corr[i] == ww[j]){
				indexx[Wert] = j;
				Wert += 1;
				Wert2 += 1;
			}
		}
	}

	
	if (-corr[0] > CBGVariables[Dimension]){ // comparison with penalized value
		CBVfeas[Dimension] = kk[indexx[0]];
		CBGVariables[Dimension] = -corr[0]; // update global best value
		for (i = 0; i < Dimension; i++){
			CBGVariables[i] = Variables[i][indexx[0]]; // parameter values of best value of this iteration
			CBVfeas[i] = Variablesfeas[i][indexx[0]]; // feasible parameter values of current best value
		}
	}
	
	for (i = 0; i < Dimension; i++){
		malt[i] = m[i];
		m[i] = 0;
		for (j = 0; j < mue; j++){
			m[i] += weights[j]*Variables[i][indexx[j]]; //update distribution mean
		}
		BDz[i] = sqrt(mueeff)*(m[i]-malt[i])/sigma; 
	}
	
	for (i = 0; i < Dimension; i++){ 
		Summe = 0;
		for (j = 0; j < Dimension; j++){
			Summe += B[j][i]*BDz[j]; // B^T * "Phi"
		}
		Temporaer[i] = Summe/D[i][i]; // D^(-1) * B^T * "Phi"
	}
	
	for (i = 0; i < Dimension; i++){
		Summe = 0;
		for (j = 0; j < Dimension; j++){
			Summe += B[i][j]*Temporaer[j]; // B * D^(-1) * B^T * "Phi"
		}
		psigma[i] = (1-csigma)*psigma[i]+sqrt(csigma*(2-csigma))*Summe; // update psigma
	}
	
	normpsigma2 = 0;
	for (i = 0; i < Dimension; i++){ 
		normpsigma2 += psigma[i]*psigma[i];
	}
	if (Variante == 16){
		hsigma = ((sqrt(normpsigma2)/sqrt(1-pow((1-csigma),2*it)))/chiN < 1.4+(2./(Dimension+1))) ? 1:0; 
	}
	else if (Variante == 6){
		hsigma = ((sqrt(normpsigma2)/sqrt(1-pow((1-csigma),2*it)))/chiN < 1.5+(1./(Dimension-0.5))) ? 1:0;
	}

	
	for (i = 0; i < Dimension; i++){ // vgl. 895-898
		pc[i] = (1-cc)*pc[i]+hsigma*sqrt(cc*(2-cc))*BDz[i]; // update pc
	}
	
	sigma *= exp(((sqrt(normpsigma2)/chiN)-1)*csigma/dsigma); // update sigma
	
	if (Variante == 6){
		
		for (i = 0; i < Dimension; i++){
			for (j = 0; j <= i; j++){
				C[i][j] = (1-ccov)*C[i][j]+(ccov/muecov)*(pc[i]*pc[j]+(1-hsigma)*cc*(2-cc)*C[i][j]);
				for (k = 0; k < mue; k++){
					C[i][j] += ccov*(1-(1/muecov))*weights[k]*(Variables[i][indexx[k]]-malt[i])*(Variables[j][indexx[k]]-malt[j])/(sigma*sigma);
				} // update C
				C[j][i] = C[i][j]; // due to symmetry
			}
		}
	
	}

	else if(Variante == 16){
		
		t1 = 0;
		for (i = 0; i < lambda; i++){
			t1 += weights[i]; // sum of all weights
		}
		
		for (i = 0; i < Dimension; i++){
			for (j = 0; j <= i; j++){
				C[i][j] = (1-c1-(cmue*t1))*C[i][j]+c1*(pc[i]*pc[j]+(1-hsigma)*cc*(2-cc)*C[i][j]);
				for (k = 0; k < mue; k++){ // here come the positive weights
					C[i][j] += cmue*weights[k]*(Variables[i][indexx[k]]-malt[i])*(Variables[j][indexx[k]]-malt[j])/(sigma*sigma);
				}
				for (k = mue; k < lambda; k++){ // here come the negative weights
					for (int z = 0; z < Dimension; z++){
						BDz[z] = (Variables[z][indexx[k]]-m[z])/sigma; // BDz in new role y_i:lambda
					}
					for (int z = 0; z < Dimension; z++){
						Summe = 0;
						for (int zz = 0; zz < Dimension; zz++){
							Summe += B[zz][z]*BDz[zz]; // B^(-1) * y
						}
						Temporaer[z] = Summe/D[z][z]; // D^(-1) * B^(-1) * y
					}
					for (int z = 0; z < Dimension; z++){
						Summe = 0;
						for (int zz = 0; zz < Dimension; zz++){
							Summe += B[z][zz]*Temporaer[zz]; // B * D^(-1) * B^(-1) * y
						}
						Helper[z] = Summe;
					}
					normHelper2 = 0;
					for (int z = 0; z < Dimension; z++){ // ||C^(-1/2)*y_i:lambda||^2
						normHelper2 += Helper[z]*Helper[z];
					}
					C[i][j] += cmue*weights[k]*(Dimension/normHelper2)*(Variables[i][indexx[k]]-malt[i])*(Variables[j][indexx[k]]-malt[j])/(sigma*sigma);
				} // update C
				C[j][i] = C[i][j]; // due to  symmetry
			}
		}
	}
	
	
	//EIGENVALUE DECOMPOSITION C = B * D^2 * B^T
	
	for (i = 0; i < Dimension; i++){
		for (j = 0; j < Dimension; j++){
			C1d[(i*Dimension)+j] = C[i][j]; // C as a vector
		}
	}
	
	
	gsl_matrix_view A = gsl_matrix_view_array (C1d, Dimension, Dimension); // creates matrix from vector
	gsl_vector *eigenvalues = gsl_vector_alloc (Dimension); // creates vector
	gsl_matrix *eigenvektoren = gsl_matrix_alloc (Dimension, Dimension); // creates matrix
	gsl_eigen_symmv_workspace * ws = gsl_eigen_symmv_alloc (Dimension); // allocates required memory
	
	gsl_eigen_symmv (&A.matrix, eigenvalues, eigenvektoren, ws); // does A = Q*D*Q^T
	
	gsl_eigen_symmv_free (ws); // empties ws
	gsl_eigen_symmv_sort (eigenvalues, eigenvektoren, GSL_EIGEN_SORT_ABS_ASC); // sorts eigenvalues in ascending order and adjusts columns of Q accordingly
	
	for (i = 0; i < Dimension; i++){
		if (gsl_vector_get (eigenvalues, i) <= 0){
			Cancel = 1;
		}
		D[i][i] = sqrt(gsl_vector_get (eigenvalues, i)); // update D (with sqrt)
		for (j = 0; j < Dimension; j++){
			B[i][j] = gsl_matrix_get (eigenvektoren, i, j); // update B
		}
	}
	
	gsl_vector_free (eigenvalues);
	gsl_matrix_free (eigenvektoren);
	
	
	if (Cancel == 1){
		cout << "C no longer positive definit!" << endl;
		itn = -it;
		return itMax;
	}
	
	//END OF EIGENVALUE DECOMPOSITION
	
	for (k = 0; k < lambda; k++){ 
		for (i = 0; i < Dimension; i++){
			Temporaer[i] = D[i][i]*norm_distrADr(generatorADr); // D * z_k
		}
		for (i = 0; i < Dimension; i++){
			Summe = 0;
			for (j = 0; j < Dimension; j++){
				Summe += B[i][j]*Temporaer[j]; // B * D * z_k
			}
			Variables[i][k] = m[i]+sigma*Summe; // update sample points x_k = m + sigma * B * D * z_k
		} // may be unfeasible (will then be penalized)
	}
	
}


// Make sample points feasible
for (i = 0; i < Dimension; i++){
	for (j = 0; j < lambda; j++){
		if (Variables[i][j] < lb[i]){
			Variablesfeas[i][j] = lb[i];
		}
		else if (Variables[i][j] > ub[i]){
			Variablesfeas[i][j] = ub[i];
		}
		else {
			Variablesfeas[i][j] = Variables[i][j];
		}
        
        if (Variables[i][j] < (lb[i]+bound_soft_edge)) {
            Variablespref[i][j] = lb[i]+bound_soft_edge;
        } else if (Variables[i][j] > (ub[i]-bound_soft_edge)) {
            Variablespref[i][j] = ub[i]-bound_soft_edge;
        } else {
            Variablespref[i][j] = Variables[i][j];
        }
	}
	// if (m[i] < lb[i]){
	// 	mrep[i] = lb[i];
	// }
	// else if (m[i] > ub[i]){
	// 	mrep[i] = ub[i];
	// }
	// else {
	// 	mrep[i] = m[i];
	// }
    if (m[i] < (lb[i]+bound_soft_edge)) {
        mpref[i] = lb[i]+bound_soft_edge;
    } else if (m[i] > (ub[i]-bound_soft_edge)) {
        mpref[i] = ub[i]-bound_soft_edge;
    } else {
        mpref[i] = m[i];
    }
}



tVektor[it%early_stop_gens] = CBGVariables[Dimension]; 
double tVektor_range, tVektor_max, tVektor_min;
if (it > early_stop_gens){ // if no clear change in goal function values for {early_stop_gens} iterations
    tVektor_max = gsl_stats_max(tVektor.data(),1,early_stop_gens);
    tVektor_min = gsl_stats_min(tVektor.data(),1,early_stop_gens);
    tVektor_range = tVektor_max - tVektor_min;
    printf("Range of best GOF over the past %d generations %f (%f, %f)\n", early_stop_gens, tVektor_range, tVektor_min, tVektor_max);
    if (tVektor_range < early_stop_tol) {
        itn = it;
        printf("Early stop at %d with range of GOF over the past %d generations = %f < tol %f\n", it, early_stop_gens, tVektor_range, early_stop_tol);
        return itMax;
    }
}

it += 1; // increase it (iteration step), this happens when CMAES_Points was called to generate new sample points
}

else if (penalty == 1){ // when CMAES_Points was called to compute penalty terms, it (iteration step) will not be increased
	penalty = 0;

	std::sort(corrraw, corrraw + lambda);
	Q3 = gsl_stats_quantile_from_sorted_data(corrraw, 1, lambda, 0.75);
	Q1 = gsl_stats_quantile_from_sorted_data(corrraw, 1, lambda, 0.25);
	IQA = Q3 - Q1;

	Diagonale = 0;
	for (i = 0; i < Dimension; i++){
		Diagonale += C[i][i];
	}
	
	IQA /= (sigma*sigma*Diagonale); // see Hansen's MATLAB implementation
	IQAhist[(it-1)%(int)(20+std::ceil((float)(3*Dimension)/(float)lambda))] = IQA;

	for (i = 0; i < (int)(20+std::ceil((float)(3*Dimension)/(float)lambda)); i++){
		IQAhistsort[i] = IQAhist[i]; // IQAhistsort will be sorted later
	} 
	
	Diagonale = 0; // here log(Diagonale)
	for (i = 0; i < Dimension; i++){
		if (C[i][i] <= 0){
			Cancel = 1;
			break;
		}
		Diagonale += log(C[i][i]);
		if (m[i] < (lb[i]+bound_soft_edge) || m[i] > (ub[i]-bound_soft_edge) || it == 1){
			if (it < (int)(20+std::ceil((float)(3*Dimension)/(float)lambda))){ // early iterations
					Gamma[i] = 5.0001*Dimension*gsl_stats_median(IQAhistsort,1,it+1);
					if (abs(m[i]-mpref[i]) > 3*sigma*sqrt(C[i][i])*MAX(1,sqrt(Dimension)/mueeff) && MYSIGN(m[i]-mpref[i]) == MYSIGN(m[i]-malt[i])){
						Gamma[i] *= 1.1; //pow(1050.2,MIN(1,mueeff/(10*Dimension))); 
					}
			}
			else { // later iterations
					Gamma[i] = 5.0001*Dimension*gsl_stats_median(IQAhistsort,1,(int)(20+std::ceil((float)(3*Dimension)/(float)lambda)));
					if (abs(m[i]-mpref[i]) > 3*sigma*sqrt(C[i][i])*MAX(1,sqrt(Dimension)/mueeff) && MYSIGN(m[i]-mpref[i]) == MYSIGN(m[i]-malt[i])){
						Gamma[i] *= 1.1; //pow(1050.2,MIN(1,mueeff/(10*Dimension)));
					}
			}
            // scale Gamma (and in turn boundary penalty) based on user input
            Gamma[i] *= gamma_scale;
		}
	}
	
	if (Cancel == 1){
		cout << "Diagonal of C no longer positive!" << endl;
		itn = -it;
		return itMax;
	}
	
	for (i = 0; i < Dimension; i++){
		ksi[i] = exp(0.9*(log(C[i][i])-(Diagonale/Dimension)));
	}
	
}

return it;

}
