#include "ProjHelperFun.cu.h"


///////////// GIVEN CODE FROM SLIDES project.pdf p 17. /////////////
///////////// assumes z is the outer dim and all matrices are same dims.
//Note that it does not make a check on the outer dim, so num threads of outer 
//dimz must be precise!


void transpose2d(REAL* A, REAL** B, int M, int N) {
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

//frees d_in
void globFromDevice(PrivGlobs* globs, unsigned outer, unsigned size, 
                    REAL* d_in, int type){
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