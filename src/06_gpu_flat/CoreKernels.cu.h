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
__global__ void kernelTridag2(REAL* myResult, const unsigned myResultSize, 
                              const unsigned outer, 
                              REAL *y, REAL *yy, REAL *aT, REAL *bT, REAL *cT,
                              const unsigned numX, const unsigned numY
){
    int kk = blockIdx.z * blockDim.z;
    int tidz = threadIdx.z;
    int k = tidz+kk;

    const unsigned n = numY*numX; //based on u (output)
    const unsigned sgmSize = numY;
    if(k < outer) {
        myResult = &myResult[k*myResultSize];
        TRIDAG_SOLVER(  &aT[idx3d(0,0,k,numX,numY)], //[idx2d(i,0,numY)], 
                        &bT[idx3d(0,0,k,numX,numY)], //[idx2d(i,0,numY)], 
                        &cT[idx3d(0,0,k,numX,numY)], //[idx2d(i,0,numY)],
                        & y[idx3d(0,0,k,numX,numX)], //[idx2d(i,0,numZ)]
                        n,
                        sgmSize,
                        myResult, //[i][0]
                        //&yy[idx2d(k,0,numY)] //[0]
                        &yy[idx3d(0,0,k,numX,numY)]
                     );
    }
}


//Expects thread sizes x=numX, y=numY
__global__ void kernelRollback1( 
                    REAL* myTimeline, REAL* myVarX, REAL* myVarY, 
                    REAL* myResult, REAL* myDxx, REAL* myDyy,
                    const unsigned myTimelineSize, 
                    const unsigned myVarXRows, const unsigned myVarXCols,
                    const unsigned myVarYRows, const unsigned myVarYCols,
                    const unsigned myResultRows, const unsigned myResultCols, 
                    const unsigned myDxxRows, const unsigned myDxxCols,
                    const unsigned myDyyRows, const unsigned myDyyCols,
                    const unsigned g, const unsigned outer, 
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

    //PrivGlobsCuda globs = globsList[k];

    myTimeline = &myTimeline[k*myTimelineSize];
    myVarX = &myVarX[k*myVarXCols*myVarXRows];
    myVarY = &myVarY[k*myVarYCols*myVarYRows];
    myResult = &myResult[k*myResultCols*myResultRows];
    myDxx = &myDxx[k*myDxxCols*myDxxRows];
    myDyy = &myDyy[k*myDyyCols*myDyyRows];

    REAL dtInv = 1.0/(myTimeline[g+1]-myTimeline[g]);

    unsigned numX = myVarXCols,//myXsize,
             numY = myVarXRows;//myYsize;

    unsigned numZ = max(numX,numY);

    if(i < numX && j < numY){
        //idx3d(int row, int col, int z, int lengt, int depth)
        uT[idx3d(i,j,k,numX, numY)] = dtInv*myResult[idx2d(i,j,myResultCols)];
        //uT[idx2d(i,j,numY)] = dtInv*myResult[i][j];

        REAL p = 0.5*0.5*myVarX[idx2d(i,j,myVarXCols)];    //[i][j];
        if(i > 0) {
            uT[idx3d(i,j,k,numX, numY)] += p*myDxx[idx2d(i,0,myDxxCols)]            //[i][0]
                        * myResult[idx2d(i-1, j, myResultCols)];   //[i-1][j];
        }
        uT[idx3d(i,j,k,numX, numY)]  += p*myDxx[idx2d(i,1,myDxxCols)] //[i][1]
                        * myResult[idx2d(i, j, myResultCols)];   //[i][j];
        if(i < numX-1) {
            uT[idx3d(i,j,k,numX, numY)] += p*myDxx[idx2d(i,2,myDxxCols)] //[i][2]
                        * myResult[idx2d(i+1, j, myResultCols)];  //[i+1][j];
        }
    }
    __syncthreads();

    if(i < numX && j < numY){
        v[idx3d(i,j,k,numX, numY)] = 0.0;
        //v[idx2d(k,i,numY)] = 0.0;

        REAL p = 0.5*myVarY[idx2d(i,j,myVarYCols)]; //[i][j];
        if(j > 0) {
          v[idx3d(i,j,k,numX, numY)] +=  p*myDyy[idx2d(j,0,myDyyCols)] //[j][0]
                     * myResult[idx2d(i, j-1, myResultCols)];   //[i][j-1];
        }
        v[idx3d(i,j,k,numX, numY)]  +=  p*myDyy[idx2d(j,1,myDyyCols)]  //[j][1]
                     * myResult[idx2d(i, j, myResultCols)];      //[i][j];
        if(j < numY-1) {
          v[idx3d(i,j,k,numX, numY)] += p*myDyy[idx2d(j,2,myDyyCols)]  //[j][2]
                     * myResult[idx2d(i, j+1, myResultCols)];  //[i][j+1];
        }
        uT[idx3d(i,j,k,numX, numY)] += v[idx3d(i,j,k,numX, numY)];
    }
    __syncthreads();

    // transpose3dTiled<TVAL>(uT, u, numY, numX);

    // __syncthreads();
    if(i < numX && j < numY){
        REAL p = 0.5*0.5*myVarX[idx2d(i,j,myVarXCols)]; //[i][j]

        aT[idx3d(i,j,k,numX,numY)] = -p*myDxx[idx2d(i,0,myDxxCols)]; //[i][0];
        bT[idx3d(i,j,k,numX,numY)] = 
                            dtInv    -p*myDxx[idx2d(i,1,myDxxCols)]; //[i][1];
        cT[idx3d(i,j,k,numX,numY)] = -p*myDxx[idx2d(i,2,myDxxCols)]; //[i][2];
    }
    __syncthreads();

    // transpose3dTiled<TVAL>(aT, a, numZ, numY);
    // transpose3dTiled<TVAL>(bT, b, numZ, numY);
    // transpose3dTiled<TVAL>(cT, c, numZ, numY);  
      
}

//template <int TVAL>
__global__ void kernelRollback2(
        REAL* myTimeline, REAL* myVarX, REAL* myVarY, 
        REAL* myDxx, REAL* myDyy,
        const unsigned myTimelineSize, 
        const unsigned myVarXRows, const unsigned myVarXCols,
        const unsigned myVarYRows, const unsigned myVarYCols,
        const unsigned myDxxRows, const unsigned myDxxCols,
        const unsigned myDyyRows, const unsigned myDyyCols,
        const unsigned g, const unsigned outer, 
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

    //PrivGlobsCuda globs = globsList[k];
    myTimeline = &myTimeline[k*myTimelineSize];
    myVarX = &myVarX[k*myVarXCols*myVarXRows];
    myVarY = &myVarY[k*myVarYCols*myVarYRows];
    myDxx = &myDxx[k*myDxxCols*myDxxRows];
    myDyy = &myDyy[k*myDyyCols*myDyyRows];

    REAL dtInv = 1.0/(myTimeline[g+1]-myTimeline[g]);

    unsigned numX = myVarXCols,//myXsize,
             numY = myVarXRows;//myYsize;

    if(i < numX && j < numY){
        REAL p = 0.5*0.5*myVarX[idx2d(i,j,myVarXCols)]; //[i][j]

        aT[idx3d(i,j,k,numX,numY)] = -p*myDxx[idx2d(i,0,myDxxCols)]; //[i][0];
        bT[idx3d(i,j,k,numX,numY)] = 
                            dtInv    -p*myDxx[idx2d(i,1,myDxxCols)]; //[i][1];
        cT[idx3d(i,j,k,numX,numY)] = -p*myDxx[idx2d(i,2,myDxxCols)]; //[i][2];
    }
    __syncthreads();


    if(i < numX && j < numY){
        REAL p = 0.5*0.5*myVarY[idx2d(i,j,myVarYCols)];  //[i][j];
        aT[idx3d(i,j,k,numX,numY)] = -p*myDyy[idx2d(j,0,myDyyCols)]; //[j][0];
        bT[idx3d(i,j,k,numX,numY)] = 
                            dtInv    -p*myDyy[idx2d(j,1,myDyyCols)]; //[j][1];
        cT[idx3d(i,j,k,numX,numY)] = -p*myDyy[idx2d(j,2,myDyyCols)]; //[j][2];
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

__global__ void kernelRollback3(REAL* myTimeline, unsigned myTimelineSize,
                                const unsigned numX, const unsigned numY,
                                const unsigned g, const unsigned outer, 
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

    myTimeline = &myTimeline[k*myTimelineSize];
    REAL dtInv = 1.0/(myTimeline[g+1]-myTimeline[g]);

    if(i < numX && j < numY){
        y[idx3d(i,j,k,numX, numY)] = dtInv * uT[idx3d(i,j,k,numX, numY)]
                                   - 0.5*v[idx3d(i,j,k,numX, numY)];
    }
}


__global__ void kernelUpdate(
        REAL* myVarX, REAL* myX, REAL* myVarY, REAL* myY,
        REAL* myTimeline, const unsigned myXsize, const unsigned myYsize, 
        const unsigned myVarXCols, const unsigned myVarXRows, 
        const unsigned myVarYCols, const unsigned myVarYRows,
        const unsigned myTimelineSize, const unsigned g,
        const unsigned outer, const REAL alpha, const REAL beta, const REAL nu
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

    myVarX = &myVarX[k*myVarXCols*myVarXRows];
    myX = &myX[k*myXsize];
    myVarY = &myVarY[k*myVarYCols*myVarYRows];
    myY = &myY[k*myYsize];
    myTimeline = &myTimeline[k*myTimelineSize];

    if(i < myXsize && j < myYsize){ //updateParams(g,alpha,beta,nu,globs[j]);
        myVarX[idx2d(i, j, myVarXCols)] = 
                            exp(2.0*(  beta*log(myX[i])
                                      + myY[j]
                                      - 0.5*nu*nu*myTimeline[g] )
                                );
        myVarY[idx2d(i, j, myVarYCols)] = 
                            exp(2.0*(  alpha*log(myX[i])
                                      + myY[j]
                                      - 0.5*nu*nu*myTimeline[g] )
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