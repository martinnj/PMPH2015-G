#include "ProjHelperFun.cu.h"
#include "Constants.h"
#include "InitKernels.cu.h"
#include "CoreKernels.cu.h"
#include "TridagKernel.cu.h"


/**
 * solves a segmented tridag, i.e., 
 * solves `n/sgm_size` independent tridag problems.
 * Logically, the arrays should be of size [n/sgm_size][sgm_size],
 * and the segmented tridag corresponds to a map on the outer
 * dimension which is applying tridag to its inner dimension.
 * This is the CUDA parallel implementation, which uses
 * block-level segmented scans. This version assumes that
 * `n` is a multiple of `sgm_sz` and also that `block_size` is
 * a multiple of `sgm_size`, i.e., such that segments do not
 * cross block boundaries.
 */
void tridagCUDAWrapper( const unsigned int block_size,
                        REAL*   a,
                        REAL*   b,
                        REAL*   c,
                        REAL*   r,
                        const unsigned int n,
                        const unsigned int sgm_sz,
                        REAL*   u,
                        REAL*   uu 
) {
    unsigned int num_blocks;
    unsigned int sh_mem_size = block_size * 32;

    // assumes sgm_sz divides block_size
    if((block_size % sgm_sz)!=0) {
        printf("Invalid segment or block size. Exiting!\n\n!");
        exit(0);
    }
    if((n % sgm_sz)!=0) {
        printf("Invalid total size (not a multiple of segment size). Exiting!\n\n!");
        exit(0);
    }
    num_blocks = (n + (block_size - 1)) / block_size;
    TRIDAG_SOLVER<<< num_blocks, block_size, sh_mem_size >>>(a, b, c, r, n, sgm_sz, u, uu);
    cudaThreadSynchronize();
}


void tridag1(const unsigned outer, REAL *u, REAL *yy,REAL *a, REAL *b, REAL *c
             const unsigned numX, const unsigned numY, const unsigned numZ){
    /*
    if(i == 0 && j < numY){
        TRIDAG_SOLVER(  &a[idx3d(k,j,0,numY,numZ)], //[idx2d(i,0,numZ)], 
                        &b[idx3d(k,j,0,numY,numZ)], //[idx2d(i,0,numZ)], 
                        &c[idx3d(k,j,0,numY,numZ)], //[idx2d(i,0,numZ)],
                        &u[idx3d(k,j,0,numY,numX)], //[idx2d(i,0,numX)],
                        numX,
                        sgmSize, //TODO ??????
                        &u[idx3d(k,j,0,numY,numX)], //[idx2d(i,0,numX)],
                        &yy[idx2d(k,0,numZ)] //[0]
                     );
    }*/
    const unsigned n = numY*numX; //based on u (output)
    const unsigned sgmSize = numX;
    for(unsigned k=0;k<outer;k++) {
        TRIDAG_SOLVER(  &a[idx3d(0,0,k,numY,numZ)], //[idx2d(i,0,numZ)], 
                        &b[idx3d(0,0,k,numY,numZ)], //[idx2d(i,0,numZ)], 
                        &c[idx3d(0,0,k,numY,numZ)], //[idx2d(i,0,numZ)],
                        &u[idx3d(0,0,k,numY,numX)], //[idx2d(i,0,numX)],
                        n,
                        sgmSize,
                        &u[idx3d(0,0,k,numY,numX)], //[idx2d(i,0,numX)],
                        &yy[idx2d(k,0,numZ)] //[0]
                     );
    }
    /*
    for(i=0;i<numY;i++) {
        tridagPar(&a[idx2d(i,0,numZ)], &b[idx2d(i,0,numZ)], &c[idx2d(i,0,numZ)]
                 ,&u[idx2d(i,0,numX)],numX,&u[idx2d(i,0,numX)],&yy[0]);
    }*/
}

void tridag2(PrivGlobsCuda* globsList, const unsigned outer, 
        REAL *y, REAL *yy, REAL *aT, REAL *bT, REAL *cT,
        const unsigned numX, const unsigned numY, const unsigned numZ
){
    const unsigned n = numY*numX;
    const unsigned sgmSize = numY;
    for(unsigned k=0;k<outer;k++) {
        PrivGlobsCuda glob = globsList[k];
        TRIDAG_SOLVER(  &aT[idx3d(0,0,k,numZ,numY)], //[idx2d(i,0,numY)], 
                        &bT[idx3d(0,0,k,numZ,numY)], //[idx2d(i,0,numY)], 
                        &cT[idx3d(0,0,k,numZ,numY)], //[idx2d(i,0,numY)],
                        & y[idx3d(0,0,k,numX,numZ)], //[idx2d(i,0,numZ)]
                        n,
                        sgmSize,
                        &globs.myResult[idx2d(i,0,globs.myResultCols)], //[i][0]
                        &yy[idx2d(k,0,numZ)] //[0]
                     );
    }
    /*
    for(i=0;i<numX;i++) { // par
        tridagPar(&aT[idx2d(i,0,numY)], &bT[idx2d(i,0,numY)],
                  &cT[idx2d(i,0,numY)], &y[idx2d(i,0,numZ)], numY,
                  &globs.myResult[i][0],&yy[0]);
                  //&globs.myResult[idx2d(i,0, globs.myResultCols)],&yy[0]);
    }*/
}




//wrapper for the kernelUpdate
void updateWrapper( PrivGlobsCuda* globsList, const unsigned g,
        const unsigned numX, const unsigned numY, const unsigned outer, 
        const REAL alpha, const REAL beta, const REAL nu, const unsigned int T
){

    //8*8*8 = 512 =< 1024
    const int x = numX;
    const int y = numY;
    const int z = outer;

    const int dimx = ceil( ((float)x) / T );
    const int dimy = ceil( ((float)y) / T );
    const int dimz = ceil( ((float)z) / T );
    dim3 block(T,T,T), grid(dimx,dimy,dimz);

    kernelUpdate <<< grid, block>>>(globsList, g, x, y, z, aplha, beta, nu);
    cudaThreadSynchronize();
}


void rollbackWrapper(PrivGlobsCuda* globsList, const unsigned g, 
                     const unsigned outer, const unsigned numX, 
                     const unsigned numY, const unsigned numT,
                     const unsigned T
){
    // create all arrays as multidim arrays for rollback()
    REAL *u, *uT, *v, *y, *yy;
    cudaMalloc((void**)&u,  outer*( numY*numX*sizeof(REAL)  ));
    cudaMalloc((void**)&uT, outer*( numX*numY*sizeof(REAL)  ));
    cudaMalloc((void**)&v,  outer*( numX*numY*sizeof(REAL)  ));
    cudaMalloc((void**)&y,  outer*( numX*numZ*sizeof(REAL)  ));
    cudaMalloc((void**)&yy, outer*(      numZ*sizeof(REAL)  ));

    REAL *a, *b, *c, *aT, *bT, *cT;
    cudaMalloc((void**)&a,  outer*( numY*numZ*sizeof(REAL)  ));
    cudaMalloc((void**)&b,  outer*( numY*numZ*sizeof(REAL)  ));
    cudaMalloc((void**)&c,  outer*( numY*numZ*sizeof(REAL)  ));
    cudaMalloc((void**)&aT, outer*( numZ*numY*sizeof(REAL)  ));
    cudaMalloc((void**)&bT, outer*( numZ*numY*sizeof(REAL)  ));
    cudaMalloc((void**)&cT, outer*( numZ*numY*sizeof(REAL)  ));

    const int x = numT;    //max(myXsize, numY), myXsize = numX
    const int y = numT;    //max(y, myYsize), myYsize = numY
    const int z = outer;

    const int dimx = ceil( ((float)x) / T );
    const int dimy = ceil( ((float)y) / T );
    const int dimz = ceil( ((float)z) / T );
    dim3 block(T,T,T), grid(dimx,dimy,dimz);

    kernelRollback1 <T> <<< grid, block>>> (globsList, g, outer, 
                                            u, uT, v, y, yy, 
                                            a, b, c, aT, bT, cT);
    cudaThreadSynchronize();

    //TODO: Tridag 1
    tridag1(outer, u, yy, a, b, c, numX, numY, numZ);


    kernelRollback2 <T> <<< grid, block>>> (globsList, g, outer, 
                                            u, uT, v, y, yy, 
                                            a, b, c, aT, bT, cT);
    cudaThreadSynchronize();

    tridag2(globList, outer, y, yy, aT, bT, cT, numX, numY, numZ);
    //TODO: Tridag 2

    cudaFree(u);
    cudaFree(uT);
    cudaFree(v);
    cudaFree(y);
    cudaFree(yy);
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    cudaFree(aT);
    cudaFree(bT);
    cudaFree(cT);
}

void getResultsWrapper(PrivGlobsCuda* globsList, 
                       const unsigned outer, 
                       REAL** res){
    const unsigned int num_threads  = outer;
    const unsigned int block_size   = 512;
    unsigned int num_blocks         = ceil(((float) num_threads) / block_size);

    unsigned int mem_size           = outer * sizeof(REAL);

    res = (REAL*) malloc(mem_size);
    {
        float* d_out;
        cudaMalloc((void**)&d_out, mem_size);
    
        kernelGetResults<<< num_blocks, block_size>>> (globsList, REAL* d_out, 
                                                       outer);
        cudaThreadSynchronize();
        
        //cuda results to mem
        cudaMemcpy(*res, d_out, mem_size, cudaMemcpyDeviceToHost);
        cudaFree(d_out);
    }
}

void   run_GPU(
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
) {
 
    // sequential loop distributed.
    const unsigned int T = 8; //8*8*8 = 512 =< 1024

    PrivGlobsCuda* globsList;
    cudaMalloc((void**)&globsList, outer*sizeof(struct PrivGlobsCuda));

    for(int g = numT-2;g>=0;--g){ //seq
        //updateParams()
        updateWrapper(globsList, g, numX, numY, outer, aplha, beta, nu, T);

        //rollback()
        rollbackWrapper(globsList, g, outer, numX, numY, numT, T);
    }
    getResultsWrapper(globsList, outer, &res);
}


//#endif // PROJ_CORE_ORIG
