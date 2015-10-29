#ifndef INIT_KERS
#define INIT_KERS

#include <cuda_runtime.h>
#include "ProjHelperFun.cu.h"
#include "Constants.h"

__global__ void kernelInitGridTimeline(
                                PrivGlobsCuda* globsList, const unsigned outer, 
                                const unsigned numT, const REAL t
) {
    int ii = blockIdx.x * blockDim.x;//numT
    int jj = blockIdx.y * blockDim.y;//outer
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int i = tidx+ii;
    int j = tidy+jj;

    if(j >= outer)
        return;

    PrivGlobsCuda globs = globsList[j];
    if(i < numT){
        globs.myTimeline[i] = t*i/(numT-1);
    }
}

__global__ void kernelInitGridMyX(
            PrivGlobsCuda* globsList, const unsigned outer, const unsigned numX, 
            const REAL dx, const REAL s0
) {
    int ii = blockIdx.x * blockDim.x;//numT
    int jj = blockIdx.y * blockDim.y;//outer
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int i = tidx+ii;
    int j = tidy+jj;

    if(j >= outer)
        return;

    PrivGlobsCuda globs = globsList[j];
    if(i < numX){
        globs.myX[i] = i*dx - globs.myXindex*dx + s0;
    }
}

__global__ void kernelInitGridMyY(
            PrivGlobsCuda* globsList, const unsigned outer, const unsigned numY,
            const REAL dy, const REAL logAlpha
) {
    int ii = blockIdx.x * blockDim.x;//numT
    int jj = blockIdx.y * blockDim.y;//outer
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int i = tidx+ii;
    int j = tidy+jj;

    if(j >= outer)
        return;

    PrivGlobsCuda globs = globsList[j];
    if(i < numY){
        globs.myY[i] = i*dy - globs.myYindex*dy + logAlpha;
    }
}

__global__ void kernelStaticCastX(PrivGlobsCuda* globsList, const unsigned outer, 
                                  const unsigned numX, const REAL s0, 
                                  const REAL dx
){
    const unsigned int gid = threadIdx.x + blockIdx.x*blockDim.x;
    if(gid < outer)
        globsList[gid].myXindex = static_cast<unsigned>(s0/dx) % numX;
}
__global__ void kernelStaticCastY(PrivGlobsCuda* globsList, 
                                  const unsigned outer, const unsigned numY
){
    const unsigned int gid = threadIdx.x + blockIdx.x*blockDim.x;
    if(gid < outer)
        globsList[gid].myYindex = static_cast<unsigned>(numY/2.0);
}

void initGridWrapper(PrivGlobsCuda* globsList, const unsigned outer, 
                     const unsigned numX, const unsigned numY, 
                     const unsigned numT, const REAL t, const REAL alpha, 
                     const REAL s0, const REAL nu
){
    int x;
    const int y = outer;
    int dimx;
    int dimy;

    const REAL stdX = 20.0*alpha*s0*sqrt(t);
    const REAL dx = stdX/numX;

    printf("\ndx = %.5f\n", dx);
    printf("t = %d\n", t);
    printf("alpha = %.5f\n", alpha);
    printf("s0 = %d\n", s0);
    printf("nu = %d\n", nu);

    const REAL stdY = 10.0*nu*sqrt(t);
    const REAL dy = stdY/numY;
    const REAL logAlpha = log(alpha);

    { //timeline
        x = numT;
        dimx = ceil( ((float)x) / TVAL );
        dimy = ceil( ((float)y) / TVAL );
        dim3 block(TVAL,TVAL,1), grid(dimx,dimy,1); 

        kernelInitGridTimeline<<< grid, block>>>(globsList, outer, numT, t);
        cudaThreadSynchronize();
    }
    { //static cast
        x = outer;
        dimx = ceil( ((float)x) / TVAL );
        dim3 block(TVAL,1,1), grid(dimx,1,1); 
        kernelStaticCastX<<< grid, block>>>(globsList, outer, numX, s0, dx);
        cudaThreadSynchronize();
    }
    { //myX
        x = numX;
        dimx = ceil( ((float)x) / TVAL );
        dimy = ceil( ((float)y) / TVAL );
        dim3 block(TVAL,TVAL,1), grid(dimx,dimy,1); 

        kernelInitGridMyX<<< grid, block>>>(globsList, outer, numX, dx, s0);
        cudaThreadSynchronize();
    }
    { //static cast
        x = outer;
        dimx = ceil( ((float)x) / TVAL );
        dim3 block(TVAL,1,1), grid(dimx,1,1); 
        kernelStaticCastY<<< grid, block>>>(globsList, outer, numY);
        cudaThreadSynchronize();
    }
    { //myY
        x = numY;
        dimx = ceil( ((float)x) / TVAL );
        dimy = ceil( ((float)y) / TVAL );
        dim3 block(TVAL,TVAL,1), grid(dimx,dimy,1); 

        kernelInitGridMyY<<< grid, block>>>
                                        (globsList, outer, numY, dy, logAlpha);
        cudaThreadSynchronize();
    } 
}

__device__ void FKernelInitOperator(REAL* x, REAL* Dxx, const unsigned n, 
                                    const unsigned DxxCols){
    const unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;

    if(i == 0){
        Dxx[0] =  0.0;
        Dxx[1] =  0.0;
        Dxx[2] =  0.0;
        Dxx[3] =  0.0;
        Dxx[idx2d(n-1, 0, DxxCols)] = 0.0;
        Dxx[idx2d(n-1, 1, DxxCols)] = 0.0;
        Dxx[idx2d(n-1, 2, DxxCols)] = 0.0;
        Dxx[idx2d(n-1, 3, DxxCols)] = 0.0;
    }
    if(0 < i && i < n-1){
        REAL dxl      = x[i]   - x[i-1];
        REAL dxu      = x[i+1] - x[i];
        
        Dxx[idx2d(i, 0, DxxCols)] = 2.0/dxl/(dxl+dxu);
        Dxx[idx2d(i, 1, DxxCols)] = -2.0*(1.0/dxl + 1.0/dxu)/(dxl+dxu);
        Dxx[idx2d(i, 2, DxxCols)] = 2.0/dxu/(dxl+dxu);
        Dxx[idx2d(i, 3, DxxCols)] =  0.0;
    }
}

//choice == 1: Dxx
//choice == 2: Dyy
//dim x must correspond to size of choice array.
__global__ void kernelInitOperator(PrivGlobsCuda* globsList, unsigned outer, 
                                   int choice){
    //int ii = blockIdx.x * blockDim.x;//numX
    int jj = blockIdx.y * blockDim.y;//numY
    //int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    //int i = tidx+ii;
    int j = tidy+jj;

    if(j >= outer)
        return;

    PrivGlobsCuda globs = globsList[j];
    REAL* x;     //1d
    REAL* Dxx; //2d
    unsigned n;

    if(choice == 1){
        x = globs.myX;
        Dxx = globs.myDxx;
        n = globs.myXsize;
    } else if(choice == 2){
        x = globs.myY;
        Dxx = globs.myDyy;
        n = globs.myYsize;
    } else{
        return;
    }
    FKernelInitOperator(x, Dxx, n, globs.myDxxCols);
}

void initOperatorWrapper(PrivGlobsCuda* globsList, const unsigned size, 
                                const unsigned outer, const unsigned choice){
    const int x = size;
    const int y = outer;

    const int dimx = ceil( ((float)x) / TVAL );
    const int dimy = ceil( ((float)y) / TVAL );
    dim3 block(TVAL,TVAL,1), grid(dimx,dimy,1); 

    kernelInitOperator<<< grid, block>>>(globsList, outer, choice);
    cudaThreadSynchronize();
}



//3 dim kernel: [x][y][z]
//x = outer
//y = max(myX.size, numT)
//z = max(y, myY.size)
__global__ void kernelSetPayoff(PrivGlobsCuda* globsList, REAL* payoff, 
                                const unsigned outer
){
    int ii = blockIdx.x * blockDim.x;//numX
    int jj = blockIdx.y * blockDim.y;//numY
    int kk = blockIdx.z * blockDim.z;//outer
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int tidz = threadIdx.z;
    int i = tidx+ii,
        j = tidy+jj,
        k = tidz+kk;

    if(k >= outer)
        return;

    PrivGlobsCuda globs = globsList[k];

    REAL strike = 0.001*k;
    if(i < globs.myXsize){
        unsigned int p = idx2d(i, k, outer);
        payoff[p] = max(globs.myX[i]-strike, (REAL)0.0);
        if(j < globs.myYsize)
            globs.myResult[idx2d(i, j, globs.myResultCols)] = payoff[p];
    }
}

void setPayoffWrapper(PrivGlobsCuda* globsList, const unsigned outer, 
                    const unsigned numX, const unsigned numY
){
    // Initialize the globs
    const int x = numX;
    const int y = numY;
    const int z = outer;

    const int dimx = ceil( ((float)x) / TVAL );
    const int dimy = ceil( ((float)y) / TVAL );
    const int dimz = ceil( ((float)z) / TVAL );
    dim3 block(TVAL,TVAL,TVAL), grid(dimx,dimy,dimz); 

    printf("init: kernelInit begin\n");
    REAL* payoff; //must be 2 dim with outer
    cudaMalloc((void**)&payoff, outer*numX*sizeof(REAL)); //myXsize = numX

    kernelSetPayoff<<< grid, block>>>(globsList, payoff, outer);
    cudaThreadSynchronize();

    cudaFree(payoff);
}


void constructGlobs(PrivGlobsCuda** globsList, const unsigned outer, 
                    const unsigned numX, const unsigned numY, 
                    const unsigned numT
){
    unsigned mem_size = outer*sizeof(struct PrivGlobsCuda);
    PrivGlobsCuda* globsLocal = (PrivGlobsCuda*) malloc(mem_size);

    for(unsigned i=0; i<outer; i++)
        globsLocal[i] = PrivGlobsCuda(numX, numY, numT);

    cudaMalloc((void**)globsList, mem_size);
    cudaMemcpy(*globsList, globsLocal, mem_size, cudaMemcpyHostToDevice);
}


void init(PrivGlobsCuda** globsList, const unsigned outer, const REAL s0, 
          const REAL alpha, const REAL nu, const REAL t, const unsigned numX, 
          const unsigned numY, const unsigned numT
){
    
    {
        // Construct the CUDA globs
        printf("\ninit: constructing globs\n");
        constructGlobs (globsList, outer, numX, numY, numT);
        printf("init: done globs\n");


        //init grid
        initGridWrapper(*globsList, outer, numX, numY, numT, t, alpha, s0, nu);

        //init op Dxx
        initOperatorWrapper(*globsList, numX, outer, 1);
        //init op Dyy
        initOperatorWrapper(*globsList, numY, outer, 2);

        setPayoffWrapper(*globsList, outer, numX, numY);

        printf("init: kernelInit done\n");
    }
}

#endif //INIT_KERS