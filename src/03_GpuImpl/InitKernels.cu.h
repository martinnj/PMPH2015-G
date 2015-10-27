#ifndef INIT_KERS
#define INIT_KERS

#include <cuda_runtime.h>
#include "ProjHelperFun.h"
#include "Constants.h"

void FKernelInitGrid( int index,
                PrivGlobsCuda& globs, const unsigned numX, const unsigned numY, 
                const unsigned numT, const REAL t, const REAL dx, const REAL dy,
                const REAL logAlpha, const REAL s0) {
    if(index < numT){
        globs.myTimeline[index] = t*index/(numT-1);
    }
    if(index < numX){
        globs.myX[index] = index*dx - globs.myXindex*dx + s0;
    }
    if(index < numY){
        globs.myY[index] = index*dy - globs.myYindex*dy + logAlpha;
    }
}


void FKernelInitOperator(int index, const unsigned n, REAL* x, REAL* Dxx, 
                         unsigned DxxCols){
    if(index == 0){
        Dxx[0] =  0.0;
        Dxx[1] =  0.0;
        Dxx[2] =  0.0;
        Dxx[3] =  0.0;
        Dxx[idx2d(n-1, 0, DxxCols)] = 0.0;
        Dxx[idx2d(n-1, 1, DxxCols)] = 0.0;
        Dxx[idx2d(n-1, 2, DxxCols)] = 0.0;
        Dxx[idx2d(n-1, 3, DxxCols)] = 0.0;
    }
    if(0 < index && index < n-1){
        dxl      = x[index]   - x[index-1];
        dxu      = x[index+1] - x[index];
        
        Dxx[idx2d(index, 0, DxxCols)] =  2.0/dxl/(dxl+dxu);
        Dxx[idx2d(index, 1, DxxCols)] = -2.0*(1.0/dxl + 1.0/dxu)/(dxl+dxu);
        Dxx[idx2d(index, 2, DxxCols)] =  2.0/dxu/(dxl+dxu);
        Dxx[idx2d(index, 3, DxxCols)] =  0.0;
    }
}

void FKernelSetPayoff(int j, int k, const REAL strike, PrivGlobsCuda& globs){
    unsigned xSize = globs.myXsize;
    if(j < globs.myXsize)
        payoff[j] = max(globs.myX[j]-strike, (REAL)0.0);

    if(j < xSize && k < globs.myYsize)
        globs.myResult[idx2d(j, k, globs.myResultCols)] = payoff[j];
}


//3 dim kernel: [x][y][z]
//x = outer
//y = max(myX.size, numT)
//z = max(y, myY.size)
__global__ void kernelInit(
    PrivGlobsCuda* globsList, const unsigned numX, const unsigned numY, 
    const unsigned numT, const REAL t, const REAL dx, const REAL dy, 
    const REAL logAlpha, const REAL s0, REAL* payoff
){
/*
for( unsigned i = 0; i < outer; ++ i ) {
    initGrid(s0,alpha,nu,t, numX, numY, numT, globs[i]);
    initOperator(globs[i].myX, globs[i].myDxx);
    initOperator(globs[i].myY, globs[i].myDyy);
    setPayoff(0.001*i, globs[i]); //2 dim
}
*/
    PrivGlobsCuda globs = globsList[i];
    const unsigned n;

    int ii = blockIdx.x * blockDim.x;
    int jj = blockIdx.y * blockDim.y;
    int kk = blockIdx.z * blockDim.z;
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int tidz = threadIdx.z;
    int i = tidx+ii,
        j = tidy+jj,
        k = tidz+kk;

    if(i >= outer)
        return;

    REAL strike = 0.001*i;
    if(k == 0) {//initGrid
        globs.myXindex = static_cast<unsigned>(s0/dx) % numX;
        globs.myYindex = static_cast<unsigned>(numY/2.0);
        FKernelInitGrid(j, globs, numX, numY, numT, t, dx, dy, logAlpha, s0);
    }
    __syncthreads();
    if(k == 0){//initOperator myX
        REAL* x = globs.myX;     //1d
        REAL* Dxx = globs.myDxx; //2d
        n = globs.myXsize;
        FKernelInitOperator(j, n, x, Dxx);
    }
    if(k == 0){//initOperator myY
        Real* y = globs.myY;     //1d
        REAL* Dyy = globs.myDyy; //2d
        const unsigned n = globs.myYsize;
        FKernelInitOperator(j, n, y, Dyy);
    }
    //__syncthreads(); //Prolly not needed here
    //setPayoff
    FKernelSetPayoff(j, k, strike, globs);
}


__global__ void constructGlobs(PrivGlobsCuda* globs, const unsigned outer, 
                               const unsigned numX, const unsigned numY, 
                               const unsigned numT
){
    const unsigned int gid = threadIdx.x + blockIdx.x*blockDim.x;
    if(gid < outer)
        globs[gid] = PrivGlobsCuda(numX, numY, numT);
}



void init(PrivGlobsCuda& globsList, const unsigned outer, const REAL s0, 
          const REAL alpha, const REAL nu, const REAL t, const unsigned numX, 
          const unsigned numY, const unsigned numT
){
    const REAL stdX = 20.0*alpha*s0*sqrt(t);
    const REAL dx = stdX/numX;
    const REAL stdY = 10.0*nu*sqrt(t);
    const REAL dy = stdY/numY;
    const REAL logAlpha = log(alpha);

    REAL* payoff;
    cudaMalloc((void**)&payoff, numX*sizeof(REAL)); //myXsize = numX
    {
        // Construct the CUDA globs
        const unsigned int block_size   = 512;
        unsigned int num_blocks         = ceil(((float) outer) / block_size);
        constructGlobs<<< num_blocks, block_size>>> (globsList, outer, numX, 
                                                     numY, numT);
        cudaThreadSynchronize();

        // Initialize the globs
        const unsigned int T = 8; //8*8*8 = 512 =< 1024
        const int x = outer;
        const int y = numT;    //max(myXsize, numT), myXsize = numX
        const int z = numT;    //max(y, myYsize), myYsize = numY

        const int dimx = ceil( ((float)x) / T );
        const int dimy = ceil( ((float)y) / T );
        const int dimz = ceil( ((float)z) / T );
        dim3 block(T,T,T), grid(dimx,dimy,dimz); 

        kernelInit<<< grid, block>>>(globsList, numX, numY, numT, t, dx, dy, 
                                     logAlpha, s0, payoff);
        cudaThreadSynchronize();

        cudaFree(payoff);
    }
}

#endif //INIT_KERS