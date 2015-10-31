#ifndef PROJ_HELPER_FUNS
#define PROJ_HELPER_FUNS

#include <vector>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include "Constants.h"

#include <cuda_runtime.h>

using namespace std;

struct PrivGlobsCuda {

    //  grid
    REAL* myX;
    unsigned myXsize;
    REAL* myY;
    unsigned myYsize;
    REAL* myTimeline;
    unsigned myTimelineSize;

    unsigned            myXindex;
    unsigned            myYindex;

    //  variable
    REAL* myResult;
    unsigned myResultRows;
    unsigned myResultCols;

    //  coeffs
    REAL* myVarX;
    unsigned myVarXRows;
    unsigned myVarXCols;
    REAL* myVarY;
    unsigned myVarYRows;
    unsigned myVarYCols;

    //  operators
    REAL* myDxx;
    unsigned myDxxRows;
    unsigned myDxxCols;
    REAL* myDyy;
    unsigned myDyyRows;
    unsigned myDyyCols;

    PrivGlobsCuda( ) {
        printf("Invalid Contructor: need to provide the array sizes! EXITING...!\n");
        exit(0);
    }

    PrivGlobsCuda( const unsigned int& numX,
                   const unsigned int& numY,
                   const unsigned int& numT ) {
        cudaMalloc((void**) &(this->myX) ,sizeof(REAL)*numX);
        this->myXsize = numX;

        cudaMalloc((void**) &(this->myDxx) ,sizeof(REAL)*numX*4);
        this->myDxxRows = numX;
        this->myDxxCols = 4;

        cudaMalloc((void**) &(this->myY) ,sizeof(REAL)*numY);
        this->myYsize = numY;

        cudaMalloc((void**) &(this->myDyy) ,sizeof(REAL)*numY*4);
        this->myDyyRows = numY;
        this->myDyyCols = 4;

        cudaMalloc((void**) &(this->myTimeline) ,sizeof(REAL)*numT);
        this->myTimelineSize = numT;

        cudaMalloc((void**) &(this->myVarX) ,sizeof(REAL)*numX*numY);
        this->myVarXRows = numX;
        this->myVarXCols = numY;

        cudaMalloc((void**) &(this->myVarY) ,sizeof(REAL)*numX*numY);
        this->myVarYRows = numX;
        this->myVarYCols = numY;

        cudaMalloc((void**) &(this->myResult) ,sizeof(REAL)*numX*numY);
        this->myResultRows = numX;
        this->myResultCols = numY;

    }
};


struct PrivGlobs {

    //  grid
    REAL* myX;
    unsigned myXsize;
    REAL* myY;
    unsigned myYsize;
    REAL* myTimeline;
    unsigned myTimelineSize;

    unsigned            myXindex;
    unsigned            myYindex;

    //  variable
    REAL* myResult;
    unsigned myResultRows;
    unsigned myResultCols;

    //  coeffs
    REAL* myVarX;
    unsigned myVarXRows;
    unsigned myVarXCols;
    REAL* myVarY;
    unsigned myVarYRows;
    unsigned myVarYCols;

    //  operators
    REAL* myDxx;
    unsigned myDxxRows;
    unsigned myDxxCols;
    REAL* myDyy;
    unsigned myDyyRows;
    unsigned myDyyCols;

    PrivGlobs( ) {
        printf("Invalid Contructor: need to provide the array sizes! EXITING...!\n");
        exit(0);
    }

    PrivGlobs(  const unsigned int& numX,
                const unsigned int& numY,
                const unsigned int& numT ) {

        this->myX = (REAL*) malloc(sizeof(REAL)*numX);
        this->myXsize = numX;

        this->myDxx = (REAL*) malloc(sizeof(REAL)*numX*4);
        this->myDxxRows = numX;
        this->myDxxCols = 4;

        this->myY = (REAL*) malloc(sizeof(REAL)*numY);
        this->myYsize = numY;

        this->myDyy = (REAL*) malloc(sizeof(REAL)*numY*4);
        this->myDyyRows = numY;
        this->myDyyCols = 4;

        this->myTimeline = (REAL*) malloc(sizeof(REAL)*numT);
        this->myTimelineSize = numT;

        this->myVarX = (REAL*) malloc(sizeof(REAL)*numX*numY);
        this->myVarXRows = numX;
        this->myVarXCols = numY;

        this->myVarY = (REAL*) malloc(sizeof(REAL)*numX*numY);
        this->myVarYRows = numX;
        this->myVarYCols = numY;

        this->myResult = (REAL*) malloc(sizeof(REAL)*numX*numY);
        this->myResultRows = numX;
        this->myResultCols = numY;

    }
}; __attribute__ ((aligned (128)));


void run_GPU(
                const unsigned int&   outer,
                const unsigned int&   numX,
                const unsigned int&   numY,
                const unsigned int&   numT,
                const REAL&           s0,
                const REAL&           t,
                const REAL&           alpha,
                const REAL&           nu,
                const REAL&           beta,
                      REAL*           res   // [outer] RESULT
            );


///////////// GIVEN CODE FROM SLIDES project.pdf p 17. /////////////
///////////// assumes z is the outer dim and all matrices are same dims.
//Note that it does not make a check on the outer dim, so num threads of outer 
//dimz must be precise!
template <int T>
__global__ void transpose3dTiled(REAL* A, REAL* trA, int rowsA, int colsA ) {
    __shared__ REAL tile[T][T+1];

    int gidz=blockIdx.z*blockDim.z*threadIdx.z;
    A+=gidz*rowsA*colsA; 
    trA+=gidz*rowsA*colsA;

    // follows code for matrix transp in x & y
    int tidx = threadIdx.x, 
        tidy = threadIdx.y;
    int j=blockIdx.x*T+tidx,
        i=blockIdx.y*T+tidy;

    if( j < colsA && i < rowsA )
        tile[tidy][tidx] = A[i*colsA+j];
    __syncthreads();

    i=blockIdx.y*T+tidx;
    j=blockIdx.x*T+tidy;
    if( j < colsA && i < rowsA )
        trA[j*rowsA+i] = tile[tidx][tidy];
}
void transpose2d(REAL* A, REAL** B, int M, int N);

// row = row idx
// col = col idx
// width = number of columns in the matrix
// ex: A[row,col] = A[idx2d(row, col, a.cols)]
__device__ __host__ inline 
unsigned int idx2d(int row, int col, int width) {
    return row * width + col;
}

//lenght = rows
//width = cols
__device__ __host__  inline 
unsigned int idx3d(int row, int col, int z, int length, int width) {
    return z*length*width + idx2d(row, col, width);
    //gidz*rowsA*colsA;
}


/////////////////////////// COPY METHODS ///////////////////////////


void globToDevice(PrivGlobs* globs, unsigned outer, unsigned size, 
                        REAL** d_out, int type);

void globFromDevice(PrivGlobs* globs, unsigned outer, unsigned size, REAL* d_in,
                    int type);




#endif // PROJ_HELPER_FUNS
