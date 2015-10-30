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
    //vector<REAL>        myX;        // [numX]
    //vector<REAL>        myY;        // [numY]
    //vector<REAL>        myTimeline; // [numT]
    REAL* myX;
    unsigned myXsize;
    REAL* myY;
    unsigned myYsize;
    REAL* myTimeline;
    unsigned myTimelineSize;

    unsigned            myXindex;
    unsigned            myYindex;

    //  variable
    //vector<vector<REAL> > myResult; // [numX][numY]
    REAL* myResult;
    unsigned myResultRows;
    unsigned myResultCols;

    //  coeffs
    //vector<vector<REAL> >   myVarX; // [numX][numY]
    //vector<vector<REAL> >   myVarY; // [numX][numY]
    //vector<REAL> myVarX;
    REAL* myVarX;
    unsigned myVarXRows;
    unsigned myVarXCols;
    //vector<REAL> myVarY;
    REAL* myVarY;
    unsigned myVarYRows;
    unsigned myVarYCols;

    //  operators
    //vector<vector<REAL> >   myDxx;  // [numX][4]
    //vector<vector<REAL> >   myDyy;  // [numY][4]
    //vector<REAL> myDxx;
    REAL* myDxx;
    unsigned myDxxRows;
    unsigned myDxxCols;
    //vector<REAL> myDyy;
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
        //this->  myX.resize(numX);
        //this->myX = (REAL*) malloc(sizeof(REAL)*numX);
        cudaMalloc((void**) &(this->myX) ,sizeof(REAL)*numX);
        this->myXsize = numX;
        // this->myDxx.resize(numX);
        // for(int k=0; k<numX; k++) {
        //     this->myDxx[k].resize(4);
        // }
        //this->myDxx.resize(numX*4);
        cudaMalloc((void**) &(this->myDxx) ,sizeof(REAL)*numX*4);
        this->myDxxRows = numX;
        this->myDxxCols = 4;

        //this->  myY.resize(numY);
        //this->myY = (REAL*) malloc(sizeof(REAL)*numY);
        cudaMalloc((void**) &(this->myY) ,sizeof(REAL)*numY);
        this->myYsize = numY;
        // this->myDyy.resize(numY);
        // for(int k=0; k<numY; k++) {
        //     this->myDyy[k].resize(4);
        // }
        //this->myDyy.resize(numY*4);
        cudaMalloc((void**) &(this->myDyy) ,sizeof(REAL)*numY*4);
        this->myDyyRows = numY;
        this->myDyyCols = 4;

        //this->myTimeline.resize(numT);
        //this->myTimeline = (REAL*) malloc(sizeof(REAL)*numT);
        cudaMalloc((void**) &(this->myTimeline) ,sizeof(REAL)*numT);
        this->myTimelineSize = numT;

        //this->  myVarX.resize(numX);
        //this->  myVarY.resize(numX);
        //this->myResult.resize(numX);
        //for(unsigned i=0;i<numX;++i) {
            //this->  myVarX[i].resize(numY);
            //this->  myVarY[i].resize(numY);
            //this->myResult[i].resize(numY);
        //}
        //this->myVarX.resize(numX*numY);
        cudaMalloc((void**) &(this->myVarX) ,sizeof(REAL)*numX*numY);
        this->myVarXRows = numX;
        this->myVarXCols = numY;

        //this->myVarY.resize(numX*numY);
        cudaMalloc((void**) &(this->myVarY) ,sizeof(REAL)*numX*numY);
        this->myVarYRows = numX;
        this->myVarYCols = numY;

        //this->myResult.resize(numX*numY);
        cudaMalloc((void**) &(this->myResult) ,sizeof(REAL)*numX*numY);
        this->myResultRows = numX;
        this->myResultCols = numY;

    }
};


struct PrivGlobs {

    //  grid
    //vector<REAL>        myX;        // [numX]
    //vector<REAL>        myY;        // [numY]
    //vector<REAL>        myTimeline; // [numT]
    REAL* myX;
    unsigned myXsize;
    REAL* myY;
    unsigned myYsize;
    REAL* myTimeline;
    unsigned myTimelineSize;

    unsigned            myXindex;
    unsigned            myYindex;

    //  variable
    //vector<vector<REAL> > myResult; // [numX][numY]
    //vector<REAL> myResult;
    REAL* myResult;
    unsigned myResultRows;
    unsigned myResultCols;

    //  coeffs
    //vector<vector<REAL> >   myVarX; // [numX][numY]
    //vector<vector<REAL> >   myVarY; // [numX][numY]
    //vector<REAL> myVarX;
    REAL* myVarX;
    unsigned myVarXRows;
    unsigned myVarXCols;
    //vector<REAL> myVarY;
    REAL* myVarY;
    unsigned myVarYRows;
    unsigned myVarYCols;

    //  operators
    //vector<vector<REAL> >   myDxx;  // [numX][4]
    //vector<vector<REAL> >   myDyy;  // [numY][4]
    //vector<REAL> myDxx;
    REAL* myDxx;
    unsigned myDxxRows;
    unsigned myDxxCols;
    //vector<REAL> myDyy;
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
        //this->  myX.resize(numX);
        this->myX = (REAL*) malloc(sizeof(REAL)*numX);
        this->myXsize = numX;
        // this->myDxx.resize(numX);
        // for(int k=0; k<numX; k++) {
        //     this->myDxx[k].resize(4);
        // }
        //this->myDxx.resize(numX*4);
        this->myDxx = (REAL*) malloc(sizeof(REAL)*numX*4);
        this->myDxxRows = numX;
        this->myDxxCols = 4;

        //this->  myY.resize(numY);
        this->myY = (REAL*) malloc(sizeof(REAL)*numY);
        this->myYsize = numY;
        // this->myDyy.resize(numY);
        // for(int k=0; k<numY; k++) {
        //     this->myDyy[k].resize(4);
        // }
        //this->myDyy.resize(numY*4);
        this->myDyy = (REAL*) malloc(sizeof(REAL)*numY*4);
        this->myDyyRows = numY;
        this->myDyyCols = 4;

        //this->myTimeline.resize(numT);
        this->myTimeline = (REAL*) malloc(sizeof(REAL)*numT);
        this->myTimelineSize = numT;

        //this->  myVarX.resize(numX);
        //this->  myVarY.resize(numX);
        //this->myResult.resize(numX);
        //for(unsigned i=0;i<numX;++i) {
            //this->  myVarX[i].resize(numY);
            //this->  myVarY[i].resize(numY);
            //this->myResult[i].resize(numY);
        //}
        //this->myVarX.resize(numX*numY);
        this->myVarX = (REAL*) malloc(sizeof(REAL)*numX*numY);
        this->myVarXRows = numX;
        this->myVarXCols = numY;

        //this->myVarY.resize(numX*numY);
        this->myVarY = (REAL*) malloc(sizeof(REAL)*numX*numY);
        this->myVarYRows = numX;
        this->myVarYCols = numY;

        //this->myResult.resize(numX*numY);
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

/*
template <int T>
__device__ void transpose3dTiled(REAL* A, REAL* trA, int rowsA, int colsA );
*/

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

void transpose(REAL* A, REAL** B, int M, int N) {
    for(int i = 0 ; i < M ; i++) {
        for(int j = 0 ; j < N ; j++) {
            //(*B)[j*M+i] = A[i*N+j];
            (*B)[i*N+j] = A[j*M+i];
        }
    }
}

/*
__device__ __host__ inline 
unsigned int idx2d(int row, int col, int width);

__device__ __host__ inline 
unsigned int idx3d(int row, int col, int z, int length, int width);
*/
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
                        REAL** d_out, int type){
    unsigned mem_size = outer*size*sizeof(REAL);
    REAL* tmp = (REAL*) malloc(mem_size);

    cudaMalloc((void**)d_out, mem_size);
    
    if(type == 1){ //myX
        for(unsigned j=0; j<outer;j++){
            for(unsigned i=0; i<size; i++){
                tmp[idx2d(j,i,size)] = globs[j].myX[i];
            }
        }
    }
    if(type == 2){ //myY
        for(unsigned j=0; j<outer;j++){
            for(unsigned i=0; i<size; i++){
                tmp[idx2d(j,i,size)] = globs[j].myY[i];
            }
        }
    }
    if(type == 3){ //myTimeline
        for(unsigned j=0; j<outer;j++){
            for(unsigned i=0; i<size; i++){ //2d works even though 3d
                tmp[idx2d(j,i,size)] = globs[j].myTimeline[i];
            }
        }
    }
    if(type == 4){ //myResult
        for(unsigned j=0; j<outer;j++){
            for(unsigned i=0; i<size; i++){
                tmp[idx2d(j,i,size)] = globs[j].myResult[i];
            }
        }
    }
    if(type == 5){ //myVarX
        for(unsigned j=0; j<outer;j++){
            for(unsigned i=0; i<size; i++){
                tmp[idx2d(j,i,size)] = globs[j].myVarX[i];
            }
        }
    }
    if(type == 6){ //myVarY
        for(unsigned j=0; j<outer;j++){
            for(unsigned i=0; i<size; i++){
                tmp[idx2d(j,i,size)] = globs[j].myVarY[i];
            }
        }
    }
    if(type == 7){ //myDxx
        for(unsigned j=0; j<outer;j++){
            for(unsigned i=0; i<size; i++){
                tmp[idx2d(j,i,size)] = globs[j].myDxx[i];
            }
        }
    }
    if(type == 8){ //myDyy
        for(unsigned j=0; j<outer;j++){
            for(unsigned i=0; i<size; i++){
                tmp[idx2d(j,i,size)] = globs[j].myDyy[i];
            }
        }
    }
    
    cudaMemcpy(*d_out, tmp, mem_size, cudaMemcpyHostToDevice);
    free(tmp);
}

void globFromDevice(PrivGlobs* globs, unsigned outer, unsigned size, REAL* d_in,
                    int type){
    unsigned mem_size = outer*size*sizeof(REAL);
    REAL* tmp = (REAL*) malloc(mem_size);

    cudaMemcpy(tmp, d_in, mem_size, cudaMemcpyDeviceToHost);
    
    if(type == 1){ //myX
        for(unsigned j=0; j<outer;j++){
            for(unsigned i=0; i<size; i++){
                globs[j].myX[i] = tmp[idx2d(j,i,size)];
            }
        }
    }
    if(type == 2){ //myY
        for(unsigned j=0; j<outer;j++){
            for(unsigned i=0; i<size; i++){
                globs[j].myY[i] = tmp[idx2d(j,i,size)];
            }
        }
    }
    if(type == 3){ //myTimeline
        for(unsigned j=0; j<outer;j++){
            for(unsigned i=0; i<size; i++){
                globs[j].myTimeline[i] = tmp[idx2d(j,i,size)];
            }
        }
    }
    if(type == 4){ //myResult
        for(unsigned j=0; j<outer;j++){
            for(unsigned i=0; i<size; i++){
                globs[j].myResult[i] = tmp[idx2d(j,i,size)];
            }
        }
    }
    if(type == 5){ //myVarX
        for(unsigned j=0; j<outer;j++){
            for(unsigned i=0; i<size; i++){
                globs[j].myVarX[i] = tmp[idx2d(j,i,size)];
            }
        }
    }
    if(type == 6){ //myVarY
        for(unsigned j=0; j<outer;j++){
            for(unsigned i=0; i<size; i++){
                globs[j].myVarY[i] = tmp[idx2d(j,i,size)];
            }
        }
    }
    if(type == 7){ //myDxx
        for(unsigned j=0; j<outer;j++){
            for(unsigned i=0; i<size; i++){
                globs[j].myDxx[i] = tmp[idx2d(j,i,size)];
            }
        }
    }
    if(type == 8){ //myDyy
        for(unsigned j=0; j<outer;j++){
            for(unsigned i=0; i<size; i++){
                globs[j].myDyy[i] = tmp[idx2d(j,i,size)];
            }
        }
    }
    free(tmp);
    cudaFree(d_in);
}




#endif // PROJ_HELPER_FUNS
