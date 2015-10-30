#ifndef INIT_KERS
#define INIT_KERS

#include <cuda_runtime.h>
#include "ProjHelperFun.cu.h"
#include "Constants.h"

__global__ void kernelInitGridTimeline(
                                REAL* myTimeline, const unsigned myTimelineSize,
                                       const unsigned outer, 
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

    //PrivGlobsCuda globs = globsList[j];
    if(i < numT){
        myTimeline[idx2d(j,i,myTimelineSize)] = t*i/(numT-1);
    }
}

__global__ void kernelInitGridMyX(
            REAL* myX, unsigned myXsize, unsigned myXindex,
            const unsigned outer, const unsigned numX, 
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

    if(i < numX)
        myX[idx2d(j,i,myXsize)] = i*dx - myXindex*dx + s0;
}

__global__ void kernelInitGridMyY(
            REAL* myY, unsigned myYsize, unsigned myYindex,
            const unsigned outer, const unsigned numY,
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

    if(i < numY)
        myY[idx2d(j,i,myYsize)] = i*dy - myYindex*dy + logAlpha;
}

void initGridWrapper(PrivGlobs* globs, const unsigned outer, 
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
    const REAL stdY = 10.0*nu*sqrt(t);
    const REAL dy = stdY/numY;
    const REAL logAlpha = log(alpha);

    { //timeline
        x = numT;
        dimx = ceil( ((float)x) / TVAL );
        dimy = ceil( ((float)y) / TVAL );
        dim3 block(TVAL,TVAL,1), grid(dimx,dimy,1);

        REAL* d_myTimeline;
        unsigned myTimelineSize = globs[0].myTimelineSize;
        globToDevice(globs, outer, myTimelineSize, &d_myTimeline, 3);

        kernelInitGridTimeline<<< grid, block>>>(d_myTimeline, myTimelineSize, 
                                                 outer, numT, t);
        cudaThreadSynchronize();
        globFromDevice(globs, outer, myTimelineSize, d_myTimeline, 3);
    }
    for(unsigned i=0; i<outer;i++)
        globs[i].myXindex = static_cast<unsigned>(s0/dx) % numX;
    { //myX
        x = numX;
        dimx = ceil( ((float)x) / TVAL );
        dimy = ceil( ((float)y) / TVAL );
        dim3 block(TVAL,TVAL,1), grid(dimx,dimy,1);

        REAL* d_myX;
        unsigned myXsize = globs[0].myXsize;
        unsigned myXindex = globs[0].myXindex;
        globToDevice(globs, outer, myXsize, &d_myX, 1);

        kernelInitGridMyX<<< grid, block>>>(d_myX, myXsize, myXindex, outer, numX, dx, 
                                            s0);
        cudaThreadSynchronize();
        globFromDevice(globs, outer, myXsize, d_myX, 1);
    }
    for(unsigned i=0; i<outer;i++)
        globs[i].myYindex = static_cast<unsigned>(numY/2.0);
    { //myY
        x = numY;
        dimx = ceil( ((float)x) / TVAL );
        dimy = ceil( ((float)y) / TVAL );
        dim3 block(TVAL,TVAL,1), grid(dimx,dimy,1);

        REAL* d_myY;
        unsigned myYsize = globs[0].myYsize;
        unsigned myYindex = globs[0].myYindex;
        globToDevice(globs, outer, myYsize, &d_myY, 2);

        kernelInitGridMyY<<< grid, block>>> (d_myY, myYsize, myYindex, outer, numY, dy, 
                                             logAlpha);
        cudaThreadSynchronize();
        globFromDevice(globs, outer, myYsize, d_myY, 2);
    } 
}

//2d kernel.
__global__ void kernelInitOperator(REAL* x, REAL* Dxx, const unsigned outer,
                                   const unsigned n, const unsigned DxxCols
){
    //const unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
    int ii = blockIdx.x * blockDim.x;//numX
    int jj = blockIdx.y * blockDim.y;//numY
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int i = tidx+ii;
    int j = tidy+jj;

    if(j >= outer)
        return;
    Dxx = &Dxx[j*DxxCols];
    x = &x[j*n];
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
// __global__ void kernelInitOperator(REAL* myDxx, REAL* myX, n, unsigned outer, 
//                                    int choice){
//     //int ii = blockIdx.x * blockDim.x;//numX
//     int jj = blockIdx.y * blockDim.y;//numY
//     //int tidx = threadIdx.x;
//     int tidy = threadIdx.y;
//     //int i = tidx+ii;
//     int j = tidy+jj;

//     if(j >= outer)
//         return;

//     PrivGlobsCuda globs = globsList[j];
//     REAL* x;     //1d
//     REAL* Dxx; //2d
//     unsigned n;

//     // if(choice == 1){
//     //     x = globs.myX;
//     //     Dxx = globs.myDxx;
//     //     //n = globs.myXsize;
//     // } else if(choice == 2){
//     //     // x = globs.myY;
//     //     // Dxx = globs.myDyy;
//     //     //n = globs.myYsize;
//     // } else{
//     //     return;
//     // }
//     FKernelInitOperator(x, Dxx, n, globs.myDxxCols);
// }

void initOperatorWrapper(PrivGlobs* globs, const unsigned outer){
    int x = globs[0].myXsize;
    const int y = outer;

    {
        const int dimx = ceil( ((float)x) / TVAL );
        const int dimy = ceil( ((float)y) / TVAL );
        dim3 block(TVAL,TVAL,1), grid(dimx,dimy,1);
    
        REAL* d_myDxx, *d_myX;
        unsigned myDxxCols = globs[0].myDxxCols;
        unsigned myDxxSize = globs[0].myDxxRows*myDxxCols;
        unsigned myXsize = globs[0].myXsize;
    
        globToDevice(globs, outer, myDxxSize, &d_myDxx, 7);
        globToDevice(globs, outer, myXsize, &d_myX, 1);
    
        //kernelInitOperator<<< grid, block>>>(d_myDxx, myDxxSize, outer, choice);
        kernelInitOperator<<< grid, block>>>(d_myX, d_myDxx, outer, myXsize, 
                                             myDxxCols);
        cudaThreadSynchronize();
        globFromDevice(globs, outer, myDxxSize, d_myDxx, 7);
    }
    {
        x = globs[0].myYsize;
        const int dimx = ceil( ((float)x) / TVAL );
        const int dimy = ceil( ((float)y) / TVAL );
        dim3 block(TVAL,TVAL,1), grid(dimx,dimy,1);

        REAL* d_myDyy, *d_myY;
        unsigned myDyyCols = globs[0].myDyyCols;
        unsigned myDyySize = globs[0].myDyyRows*myDyyCols;
        unsigned myYsize = globs[0].myYsize;


        globToDevice(globs, outer, myDyySize, &d_myDyy, 7);
        globToDevice(globs, outer, myYsize, &d_myY, 1);

        //kernelInitOperator<<< grid, block>>>(d_myDxx, myDxxSize, outer, choice);
        kernelInitOperator<<< grid, block>>>(d_myY, d_myDyy, outer, myYsize, 
                                             myDyyCols);
        cudaThreadSynchronize();
        globFromDevice(globs, outer, myDyySize, d_myDyy, 7);
    }
}



//3 dim kernel: [x][y][z]
//x = outer
//y = max(myX.size, numT)
//z = max(y, myY.size)
__global__ void kernelSetPayoff(REAL* myX, REAL* myResult, unsigned myXsize, 
                                unsigned myYsize, unsigned myResultCols,
                                REAL* payoff, const unsigned outer
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

    REAL strike = 0.001*k;
    if(i < myXsize){
        unsigned int p = idx2d(i, k, outer);
        payoff[p] = max(myX[i]-strike, (REAL)0.0);
        if(j < myYsize)
            myResult[idx2d(i, j, myResultCols)] = payoff[p];
    }

    //PrivGlobsCuda globs = globsList[k];
    // if(i < globs.myXsize){
    //     unsigned int p = idx2d(i, k, outer);
    //     payoff[p] = max(globs.myX[i]-strike, (REAL)0.0);
    //     if(j < globs.myYsize)
    //         globs.myResult[idx2d(i, j, globs.myResultCols)] = payoff[p];
    // }
}

void setPayoffWrapper(PrivGlobs* globs, const unsigned outer, 
                    const unsigned numX, const unsigned numY
){
    // Initialize the globs
    const int x = numX;
    const int y = numY;
    const int z = outer;

    const int dimx = ceil( ((float)x) / TVAL );
    const int dimy = ceil( ((float)y) / TVAL );
    const int dimz = ceil( ((float)z));
    dim3 block(TVAL,TVAL,1), grid(dimx,dimy,dimz); 

    REAL* payoff; //must be 2 dim with outer
    cudaMalloc((void**)&payoff, outer*numX*sizeof(REAL)); //myXsize = numX

    REAL *d_myX, *d_myResult;
    unsigned myXsize = globs[0].myXsize;
    unsigned myYsize = globs[0].myYsize;
    unsigned myResultCols = globs[0].myResultCols;
    unsigned myResultSize = globs[0].myResultRows*myResultCols;
    globToDevice(globs, outer, myXsize, &d_myX, 1);
    globToDevice(globs, outer, myResultSize, &d_myResult, 4);

    kernelSetPayoff<<< grid, block>>>(d_myX, d_myResult, myXsize, myYsize, 
                                      myResultCols, payoff, outer);
    cudaThreadSynchronize();
    globFromDevice(globs, outer, myResultSize, d_myResult, 4);

    cudaFree(payoff);
}


#endif //INIT_KERS