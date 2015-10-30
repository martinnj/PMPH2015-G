#ifndef CORE_KERS
#define CORE_KERS

#include "ProjHelperFun.cu.h"
#include "Constants.h"
#include "TridagKernel.cu.h"



//3d kernel for the 2d-tridag kernel
__global__ void kernelTridag1(const unsigned outer, REAL *u, REAL *yy, 
                                REAL *a, REAL *b, REAL *c, const unsigned numX, 
                                const unsigned numY
){
    int kk = blockIdx.z * blockDim.z;
    int tidz = threadIdx.z;
    int k = tidz+kk;

    const unsigned n = numY*numX; //based on u (output)
    const unsigned sgmSize = numX;
    if(k < outer) {
        TRIDAG_SOLVER  (&a[idx3d(0,0,k,numY,numX)], //[idx2d(i,0,numZ)], 
                        &b[idx3d(0,0,k,numY,numX)], //[idx2d(i,0,numZ)], 
                        &c[idx3d(0,0,k,numY,numX)], //[idx2d(i,0,numZ)],
                        &u[idx3d(0,0,k,numY,numX)], //[idx2d(i,0,numX)],
                        n,
                        sgmSize,
                        &u[idx3d(0,0,k,numY,numX)], //[idx2d(i,0,numX)],
                        //&yy[idx2d(k,0,numX)] //[0]
                        &yy[idx3d(0,0,k,numY,numX)]
                        );
    }
}

//3d kernel for the 2d-tridag kernel
__global__ void kernelTridag2(PrivGlobsCuda* globsList, const unsigned outer, 
                              REAL *y, REAL *yy, REAL *aT, REAL *bT, REAL *cT,
                const unsigned numX, const unsigned numY
){
    int kk = blockIdx.z * blockDim.z;
    int tidz = threadIdx.z;
    int k = tidz+kk;

    const unsigned n = numY*numX; //based on u (output)
    const unsigned sgmSize = numY;
    if(k < outer) {
        PrivGlobsCuda globs = globsList[k];
        TRIDAG_SOLVER(  &aT[idx3d(0,0,k,numX,numY)], //[idx2d(i,0,numY)], 
                        &bT[idx3d(0,0,k,numX,numY)], //[idx2d(i,0,numY)], 
                        &cT[idx3d(0,0,k,numX,numY)], //[idx2d(i,0,numY)],
                        & y[idx3d(0,0,k,numX,numX)], //[idx2d(i,0,numZ)]
                        n,
                        sgmSize,
                        &globs.myResult[0], //[i][0]
                        //&yy[idx2d(k,0,numY)] //[0]
                        &yy[idx3d(0,0,k,numX,numY)]
                     );
    }
}


//Expects thread sizes x=numX, y=numY
__global__ void kernelRollback1(
        PrivGlobsCuda* globsList, const unsigned g, const unsigned outer, 
        REAL *u, REAL *uT, REAL *v, REAL *y,
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

    PrivGlobsCuda globs = globsList[k];
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

    // transpose3dTiled<TVAL>(uT, u, numY, numX);

    // __syncthreads();
    if(i < numX && j < numY){
        REAL p = 0.5*0.5*globs.myVarX[idx2d(i,j,globs.myVarXCols)]; //[i][j]

        aT[idx3d(i,j,k,numX,numY)] = -p*globs.myDxx[idx2d(i,0,globs.myDxxCols)]; //[i][0];
        bT[idx3d(i,j,k,numX,numY)] = 
                            dtInv    -p*globs.myDxx[idx2d(i,1,globs.myDxxCols)]; //[i][1];
        cT[idx3d(i,j,k,numX,numY)] = -p*globs.myDxx[idx2d(i,2,globs.myDxxCols)]; //[i][2];
    }
    __syncthreads();

    // transpose3dTiled<TVAL>(aT, a, numZ, numY);
    // transpose3dTiled<TVAL>(bT, b, numZ, numY);
    // transpose3dTiled<TVAL>(cT, c, numZ, numY);  
      
}

//template <int TVAL>
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

    PrivGlobsCuda globs = globsList[k];
    REAL dtInv = 1.0/(globs.myTimeline[g+1]-globs.myTimeline[g]);

    unsigned numX = globs.myXsize,
             numY = globs.myYsize;

    if(i < numX && j < numY){
        REAL p = 0.5*0.5*globs.myVarX[idx2d(i,j,globs.myVarXCols)]; //[i][j]

        aT[idx3d(i,j,k,numX,numY)] = -p*globs.myDxx[idx2d(i,0,globs.myDxxCols)]; //[i][0];
        bT[idx3d(i,j,k,numX,numY)] = 
                            dtInv    -p*globs.myDxx[idx2d(i,1,globs.myDxxCols)]; //[i][1];
        cT[idx3d(i,j,k,numX,numY)] = -p*globs.myDxx[idx2d(i,2,globs.myDxxCols)]; //[i][2];
    }
    __syncthreads();


    if(i < numX && j < numY){
        REAL p = 0.5*0.5*globs.myVarY[idx2d(i,j,globs.myVarYCols)];  //[i][j];
        aT[idx3d(i,j,k,numX,numY)] = -p*globs.myDyy[idx2d(j,0,globs.myDyyCols)]; //[j][0];
        bT[idx3d(i,j,k,numX,numY)] = 
                            dtInv    -p*globs.myDyy[idx2d(j,1,globs.myDyyCols)]; //[j][1];
        cT[idx3d(i,j,k,numX,numY)] = -p*globs.myDyy[idx2d(j,2,globs.myDyyCols)]; //[j][2];
    }
    // __syncthreads();

    // transpose3dTiled<TVAL>(aT, a, numY, numX);
    // transpose3dTiled<TVAL>(bT, b, numY, numX);
    // transpose3dTiled<TVAL>(cT, c, numY, numX);

    // transpose3dTiled<TVAL>(u, uT, numX, numY);

    // __syncthreads();
    // if(i < numX && j < numY){
    //     y[idx3d(i,j,k,numX, numY)] = dtInv * uT[idx3d(i,j,k,numX, numY)]
    //                                - 0.5*v[idx3d(i,j,k,numX, numY)];
    // }
}

__global__ void kernelRollback3(
        PrivGlobsCuda* globsList, const unsigned g, const unsigned outer, 
        REAL *uT, REAL *v, REAL *y
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

    PrivGlobsCuda globs = globsList[k];
    REAL dtInv = 1.0/(globs.myTimeline[g+1]-globs.myTimeline[g]);

    unsigned numX = globs.myXsize,
             numY = globs.myYsize;

    if(i < numX && j < numY){
        y[idx3d(i,j,k,numX, numY)] = dtInv * uT[idx3d(i,j,k,numX, numY)]
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

    PrivGlobsCuda globs = globsList[k];

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