#ifndef CORE_KERS
#define CORE_KERS

#include "ProjHelperFun.cu.h"
#include "Constants.h"

template <int T>
__global__ void kernelRollback1(
        PrivGlobsCuda* globsList, const unsigned g, const unsigned outer, 
        REAL *u, REAL *uT, REAL *v, REAL *y, REAL *yy,
        REAL *a, REAL *b, REAL *c, REAL *aT, REAL *bT, REAL *cT
){
    int ii = blockIdx.x * blockDim.x;
    int jj = blockIdx.y * blockDim.y;
    int kk = blockIdx.z * blockDim.z;
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int tidz = threadIdx.z;
    int i = tidx+ii,
        j = tidy+jj,
        k = tidz+kk;

    if(k >= outer)
        return;

    PrivGlobsCuda globs = globsList[z];
    REAL dtInv = 1.0/(globs.myTimeline[g+1]-globs.myTimeline[g]);

    unsigned numX = globs.myXsize,
             numY = globs.myYsize;

    unsigned numZ = max(numX,numY);

    if(i < numX && j < numY){
        //idx3d(int row, int col, int z, int lengt, int depth)
        uT[idx3d(i,j,k,numX, numY)] = dtInv*globs.myResult[idx2d(i,j,globs.myResultCols)];
        //uT[idx2d(i,j,numY)] = dtInv*globs.myResult[i][j];

        REAL p = 0.5*0.5*globs.myVarX[idx2d(i,j,globs.myVarXCols)];    //[i][j];
        if(i > 0) {
            uT[idx3d(i,j,k,numX, numY)] += p*globs.myDxx[idx2d(i,0,globs.myDxxCols)]            //[i][0]
                        * globs.myResult[idx2d(i-1, j, globs.myResultCols)];   //[i-1][j];
        }
        uT[idx3d(i,j,k,numX, numY)]  += p*globs.myDxx[idx2d(i,1,globs.myDxxCols)] //[i][1]
                        * globs.myResult[idx2d(i, j, globs.myResultCols)];   //[i][j];
        if(i < numX-1) {
            uT[idx3d(i,j,k,numX, numY)] += p*globs.myDxx[idx2d(i,2,globs.myDxxCols)] //[i][2]
                        * globs.myResult[idx2d(i+1, j, globs.myResultCols)];  //[i+1][j];
        }
    }
    __syncthreads();

    if(i < numX && j < numY){
        v[idx3d(i,j,k,numX, numY)] = 0.0;
        //v[idx2d(k,i,numY)] = 0.0;

        REAL p = 0.5*globs.myVarY[idx2d(i,j,globs.myVarYCols)]; //[i][j];
        if(j > 0) {
          v[idx3d(i,j,k,numX, numY)] +=  p*globs.myDyy[idx2d(j,0,globs.myDyyCols)] //[j][0]
                     * globs.myResult[idx2d(i, j-1, globs.myResultCols)];   //[i][j-1];
        }
        v[idx3d(i,j,k,numX, numY)]  +=  p*globs.myDyy[idx2d(j,1,globs.myDyyCols)]  //[j][1]
                     * globs.myResult[idx2d(i, j, globs.myResultCols)];      //[i][j];
        if(j < numY-1) {
          v[idx3d(i,j,k,numX, numY)] += p*globs.myDyy[idx2d(j,2,globs.myDyyCols)]  //[j][2]
                     * globs.myResult[idx2d(i, j+1, globs.myResultCols)];  //[i][j+1];
        }
        uT[idx3d(i,j,k,numX, numY)] += v[idx3d(i,j,k,numX, numY)];
    }
    __syncthreads();
    transpose3d<T>(uT, u, numY, numX);

    __syncthreads();
    if(i < numX && j < numY){
        REAL p = 0.5*0.5*globs.myVarX[idx2d(i,j,globs.myVarXCols)]; //[i][j]

        aT[idx3d(i,j,k,numZ,numY)] = -p*globs.myDxx[idx2d(i,0,globs.myDxxCols)]; //[i][0];
        bT[idx3d(i,j,k,numZ,numY)] = 
                            dtInv    -p*globs.myDxx[idx2d(i,1,globs.myDxxCols)]; //[i][1];
        cT[idx3d(i,j,k,numZ,numY)] = -p*globs.myDxx[idx2d(i,2,globs.myDxxCols)]; //[i][2];
    }
    __syncthreads();

    transpose3d<T>(aT, &a, numY, numZ);
    transpose3d<T>(bT, &b, numY, numZ);
    transpose3d<T>(cT, &c, numY, numZ);
}


template <int T>
__global__ void kernelRollback2(
        PrivGlobsCuda* globsList, const unsigned g, const unsigned outer, 
        REAL *u, REAL *uT, REAL *v, REAL *y, REAL *yy,
        REAL *a, REAL *b, REAL *c, REAL *aT, REAL *bT, REAL *cT
){
    int ii = blockIdx.x * blockDim.x;
    int jj = blockIdx.y * blockDim.y;
    int kk = blockIdx.z * blockDim.z;
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int tidz = threadIdx.z;
    int i = tidx+ii,
        j = tidy+jj,
        k = tidz+kk;

    if(k >= outer)
        return;

    PrivGlobsCuda globs = globsList[z];
    REAL dtInv = 1.0/(globs.myTimeline[g+1]-globs.myTimeline[g]);

    unsigned numX = globs.myXsize,
             numY = globs.myYsize;

    unsigned numZ = max(numX,numY);

    if(i < numX && j < numY){
        REAL p = 0.5*0.5*globs.myVarX[idx2d(i,j,globs.myVarXCols)]; //[i][j]

        aT[idx3d(i,j,k,numZ,numY)] = -p*globs.myDxx[idx2d(i,0,globs.myDxxCols)]; //[i][0];
        bT[idx3d(i,j,k,numZ,numY)] = 
                            dtInv    -p*globs.myDxx[idx2d(i,1,globs.myDxxCols)]; //[i][1];
        cT[idx3d(i,j,k,numZ,numY)] = -p*globs.myDxx[idx2d(i,2,globs.myDxxCols)]; //[i][2];
    }
    __syncthreads();


    if(i < numX && j < numY){
        REAL p = 0.5*(0.5*globs.myVarY[idx2d(i,j,globs.myVarYCols)];  //[i][j];
        aT[idx3d(i,j,k,numZ,numY)] = -p*globs.myDyy[idx2d(j,0,globs.myDyyCols)]; //[j][0];
        bT[idx3d(i,j,k,numZ,numY)] = 
                            dtInv    -p*globs.myDyy[idx2d(j,1,globs.myDyyCols)]; //[j][1];
        cT[idx3d(i,j,k,numZ,numY)] = -p*globs.myDyy[idx2d(j,2,globs.myDyyCols)]; //[j][2];
    }
    __syncthreads();

    transpose3d<T>(aT, &a, numY, numZ);
    transpose3d<T>(bT, &b, numY, numZ);
    transpose3d<T>(cT, &c, numY, numZ);

    transpose3d<T>(u, &uT, numX, numY);
    __syncthreads();

    if(i < numX && j < numY){
        y[idx3d(i,j,k,numX, numZ)] = dtInv * uT[idx3d(i,j,k,numX, numY)]
                                   - 0.5*v[idx3d(i,j,k,numX, numY)];
    }
}

__global__ void kernelUpdate(
        PrivGlobsCuda* globsList, const unsigned g,
        const unsigned numX, const unsigned numY, const unsigned outer, 
        const REAL alpha, const REAL beta, const REAL nu
){
    //for( unsigned j = 0; j < outer; ++ j ) { //par
    int ii = blockIdx.x * blockDim.x;
    int jj = blockIdx.y * blockDim.y;
    int kk = blockIdx.z * blockDim.z;
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int tidz = threadIdx.z;
    int i = tidx+ii,
        j = tidy+jj,
        k = tidz+kk;

    if(k >= outer)
        return;

    PrivGlobsCuda globs = globsList[z];

    if(i < globs.myXsize && j < globs.myYsize){ //updateParams(g,alpha,beta,nu,globs[j]);
        globs.myVarX[idx2d(i, j, globs.myVarXCols)] = 
                            exp(2.0*(  beta*log(globs.myX[i])
                                      + globs.myY[j]
                                      - 0.5*nu*nu*globs.myTimeline[g] )
                                );
        globs.myVarY[idx2d(i, j, globs.myVarYCols)] = 
                            exp(2.0*(  alpha*log(globs.myX[i])
                                          + globs.myY[j]
                                          - 0.5*nu*nu*globs.myTimeline[g] )
                                );
    }
}



__global__ void kernelGetResults(PrivGlobsCuda* globsList, 
                                 REAL* res_out,
                                 const unsigned outer
){
    const unsigned int gid = threadIdx.x + blockIdx.x*blockDim.x;

    PrivGlobsCuda globs = globsList[gid];
    if(gid < outer)
        res_out[gid] = globs.myResult[idx2d(globs.myXindex,
                                            globs.myYindex,
                                            globs.myResultCols)];

    /*
    for( unsigned j = 0; j < outer; ++ j ) { //par
        res[j] = globs[j].myResult[globs[j].myXindex][globs[j].myYindex];
    }
    */
}


#endif //CORE_KERS