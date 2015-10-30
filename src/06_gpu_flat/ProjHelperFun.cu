#include "ProjHelperFun.cu.h"

/**************************/
/**** HELPER FUNCTIONS ****/
/**************************/
/*
///////////// GIVEN CODE FROM SLIDES project.pdf p 17. /////////////
///////////// assumes z is the outer dim and all matrices are same dims.
template <int T>
__device__ void transpose3dTiled(REAL* A, REAL* trA, int rowsA, int colsA ) {
    __shared__ REAL tile[T][T+1];

    int gidz=blockIdx.z*blockDim.z*threadIdx.z;
    A+=gidz*rowsA*colsA; trA+=gidz*rowsA*colsA;

    // follows code for matrix transp in x & y
    int tidx = threadIdx.x, tidy = threadIdx.y;
    int j=blockIdx.x*T+tidx,i=blockIdx.y*T+tidy;

    if( j < colsA && i < rowsA )
        tile[tidy][tidx] = A[i*colsA+j];
    __syncthreads();

    i=blockIdx.y*T+tidx; j=blockIdx.x*T+tidy;
    if( j < colsA && i < rowsA )
        trA[j*rowsA+i] = tile[tidx][tidy];
}
*/
/* //moved to header//

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
*/
/*
//assumes same size
void flatMatTo2dVect(REAL* flatMat, vector<vector<REAL> > v, int rows, int cols){
    for(unsigned i=0; i < rows; i++){
        for(unsigned j=0; j < cols; j++){
            v[i][j] = flatMat[idx2d(i,j,cols)];
        }
    }
}
*/