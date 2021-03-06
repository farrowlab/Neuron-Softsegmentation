#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include "mex.h"
#include "../src/TVopt.h"

/* solveTV2D_Kolmogorov.cpp

   Solves the 2 dimensional TV proximity problem by applying Kolmogorov et al splitting algorithm.

   Parameters:
     - 0: bidimensional reference signal y.
     - 1: lambda penalty
     - 2: (optional) maximum number of iterations to run (default: as defined in TVopt.h)
     
   Outputs:
     - 0: solution x.
     - 1: array with optimizer information:
        + [0]: number of iterations run.
        + [1]: stopping tolerance.
*/
void mexFunction(int nlhs, mxArray *plhs[ ],int nrhs, const mxArray *prhs[ ]) {
    const mwSize *sDims;
    double *x=NULL,*y,*info=NULL,*dims;
    double lambda;
    int *ns=NULL;
    int nds,N,M,maxIters,i;
    
    #define FREE \
        if(!nlhs) free(x); \
        if(ns) free(ns);
    #define CANCEL(txt) \
        printf("Error in solveTV2D_Kolmogorov: %s\n",txt); \
        if(x) free(x); \
        if(info) free(info); \
        if(ns) free(ns); \
        return;
        
    /* Check input correctness */
    if(nrhs < 2){CANCEL("not enought inputs");}
    if(!mxIsClass(prhs[0],"double")) {CANCEL("input signal must be in double format")}

    /* Find number of dimensions of the input signal */
    nds=mxGetNumberOfDimensions(prhs[0]);
    /* Must be 2 */
    if ( nds != 2 )
        {CANCEL("input signal must be 2-dimensional")}
    M = mxGetM(prhs[0]);
    N = mxGetN(prhs[0]);
    /* Get dimensions of input signal */
    sDims = mxGetDimensions(prhs[0]);
    /* Convert dimensions to C array */
    ns = (int*)malloc(sizeof(int)*nds);
    if(!ns) {CANCEL("out of memory")}
    for(i=0;i<nds;i++) ns[i] = (int)sDims[i];
    
    /* Get input signal */
    y = mxGetPr(prhs[0]);
    
    /* Get rest of inputs */
    lambda = mxGetScalar(prhs[1]);
    if(nrhs >= 3) maxIters = (int)(mxGetPr(prhs[2]))[0];
    else maxIters = 0;

    /* Create output arrays */
    if(nlhs >= 1){
        plhs[0]=mxCreateNumericArray(nds,sDims,mxDOUBLE_CLASS,mxREAL);
        if(!plhs[0]){CANCEL("out of memory")}
        x=mxGetPr(plhs[0]);
    }
    else x = (double*)malloc(sizeof(double)*mxGetNumberOfElements(prhs[0]));
    if(nlhs >= 2){
        plhs[1] = mxCreateDoubleMatrix(N_INFO,1,mxREAL);
        info = mxGetPr(plhs[1]);
    }
    else info = NULL;
    
    /* Run algorithm */
    Kolmogorov2_TV(M, N, y, lambda, x, maxIters, info);
    
    /* Free resources */
    FREE
    
    return;
    
    #undef FREE
    #undef CANCEL
}


