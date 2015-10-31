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