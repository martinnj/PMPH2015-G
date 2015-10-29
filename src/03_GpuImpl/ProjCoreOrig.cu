#include "ProjHelperFun.cu.h"
#include "Constants.h"
#include "InitKernels.cu.h"
#include "CoreKernels.cu.h"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////DEBUGGING//////////////////////
__global__ void getList(PrivGlobsCuda* globsList, 
                                 REAL* res_out,
                                 const unsigned size
                                 // REAL* mat
){
    const unsigned int gid = threadIdx.x + blockIdx.x*blockDim.x;

    PrivGlobsCuda globs = globsList[8];
    if(gid < size){
        //res_out[gid] = globs.myVarX[gid];
        //res_out[gid] = globs.myTimeline[gid];
        // res_out[gid] = globs.myResult[gid];
        //res_out[gid] = mat[gid];
        res_out[0] = (REAL) globs.myXindex;
        res_out[1] = (REAL) globs.myYindex;
        res_out[2] = (REAL) globs.myResult[idx2d(globs.myXindex,
                                            globs.myYindex,
                                            globs.myResultCols)];
        //res_out[gid] = globs.myResult[gid];
    }
    /*
    for( unsigned j = 0; j < outer; ++ j ) { //par
        res[j] = globs[j].myResult[globs[j].myXindex][globs[j].myYindex];
    }
    */
}

////////////////////////////////////////////////////////////////////////////////



//wrapper for the kernelUpdate
void updateWrapper( PrivGlobsCuda* globsList, const unsigned g,
        const unsigned numX, const unsigned numY, const unsigned outer, 
        const REAL alpha, const REAL beta, const REAL nu
){

    //8*8*8 = 512 =< 1024
    const int x = numX;
    const int y = numY;
    const int z = outer;

    const int dimx = ceil( ((float)x) / TVAL );
    const int dimy = ceil( ((float)y) / TVAL );
    const int dimz = ceil( ((float)z) / TVAL );
    dim3 block(TVAL,TVAL,TVAL), grid(dimx,dimy,dimz);

    kernelUpdate <<< grid, block>>>(globsList, g, x, y, z, alpha, beta, nu);
    cudaThreadSynchronize();
}


void rollbackWrapper(PrivGlobsCuda* globsList, const unsigned g, 
                     const unsigned outer, const unsigned numX, 
                     const unsigned numY, const unsigned numZ
){
    // create all arrays as multidim arrays for rollback()
    REAL *u, *uT, *v, *y, *yy;
    //[3.dim][1.dim][2.dim]
    //u = [numY][numX][outer]; numY rows, numX cols
    cudaMalloc((void**)&u,  outer*( numY*numX*sizeof(REAL)  ));
    cudaMalloc((void**)&uT, outer*( numX*numY*sizeof(REAL)  ));
    cudaMalloc((void**)&v,  outer*( numX*numY*sizeof(REAL)  ));
    cudaMalloc((void**)&y,  outer*( numX*numY*sizeof(REAL)  ));
    cudaMalloc((void**)&yy, outer*( numX*numY*sizeof(REAL)  ));
    // cudaMalloc((void**)&yy, outer*( numX*sizeof(REAL)  ));

    REAL *a, *b, *c, *aT, *bT, *cT;
    cudaMalloc((void**)&a,  outer*( numY*numX*sizeof(REAL)  ));
    cudaMalloc((void**)&b,  outer*( numY*numX*sizeof(REAL)  ));
    cudaMalloc((void**)&c,  outer*( numY*numX*sizeof(REAL)  ));
    cudaMalloc((void**)&aT, outer*( numX*numY*sizeof(REAL)  ));
    cudaMalloc((void**)&bT, outer*( numX*numY*sizeof(REAL)  ));
    cudaMalloc((void**)&cT, outer*( numX*numY*sizeof(REAL)  ));

    const int x = numZ;    //max(myXsize, numY), myXsize = numX
    //const int y = numZ = x;    //max(y, myYsize), myYsize = numY

    int dimx = ceil( ((float)x) / TVAL );
    int dimy = ceil( ((float)x) / TVAL );
    int dimz = outer;
    dim3 block(TVAL,TVAL,1), grid(dimx,dimy,dimz);

    const unsigned n = numY*numX;
    unsigned int block_size = 512;
    unsigned int num_blocks = (n + (block_size - 1)) / block_size;
    unsigned int sh_mem_size = block_size * 32;

    kernelRollback1 <<< grid, block >>> (   globsList, g, outer, 
                                            u, uT, v, y, 
                                            a, b, c, aT, bT, cT);
    cudaThreadSynchronize();


    transpose3dTiled<TVAL><<< grid, block >>>(uT, u, numY, numX);
    cudaThreadSynchronize();

    transpose3dTiled<TVAL><<< grid, block >>>(aT, a, numX, numY);
    cudaThreadSynchronize();
    transpose3dTiled<TVAL><<< grid, block >>>(bT, b, numX, numY);
    cudaThreadSynchronize();
    transpose3dTiled<TVAL><<< grid, block >>>(cT, c, numX, numY);  
    cudaThreadSynchronize();

    //Tridag 1
    //tridag1(outer, u, yy, a, b, c, numX, numY, numZ);
    kernelTridag1 <<< num_blocks, block_size, sh_mem_size >>> 
                                    (outer, u, yy, a, b, c, numX, numY);
    cudaThreadSynchronize();

    kernelRollback2 <<< grid, block>>> (    globsList, g, outer, 
                                            u, uT, v, y, yy, 
                                            a, b, c, aT, bT, cT);
    cudaThreadSynchronize();

    transpose3dTiled<TVAL><<< grid, block >>>(aT, a, numY, numX);
    cudaThreadSynchronize();
    transpose3dTiled<TVAL><<< grid, block >>>(bT, b, numY, numX);
    cudaThreadSynchronize();
    transpose3dTiled<TVAL><<< grid, block >>>(cT, c, numY, numX);
    cudaThreadSynchronize();
    transpose3dTiled<TVAL><<< grid, block >>>(u, uT, numX, numY);
    cudaThreadSynchronize();

    kernelRollback3 <<< grid, block>>> (globsList, g, outer, uT, v, y);
    cudaThreadSynchronize();

    //tridag2(globsList, outer, y, yy, aT, bT, cT, numX, numY, numZ);
    kernelTridag2 <<< num_blocks, block_size, sh_mem_size >>> 
                        (globsList, outer, y, yy, aT, bT, cT, numX, numY);
    cudaThreadSynchronize();

    {
        //unsigned s = numX*numY;
        unsigned size = 3;
        unsigned mem_size = size*sizeof(REAL);

        unsigned num_threads = size;
        unsigned block_size = 512;
        unsigned int num_blocks = ceil(((float) num_threads) / block_size);

        REAL *res, *d_res;
        cudaMalloc((void**)&d_res, mem_size);
        res = (REAL*) malloc(mem_size);

        getList<<< num_blocks, block_size>>>(globsList, d_res, size);
        cudaThreadSynchronize();

        cudaMemcpy(res, d_res, mem_size, cudaMemcpyDeviceToHost);

        printf("\nres = [\n");
        for(unsigned i=0; i < size; i++)
            printf("[%d] = %.5f\n", i, res[i]);
        printf("\n]\n");

        //exit(0);
    }

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
                       REAL* res){
    const unsigned int num_threads  = outer;
    const unsigned int block_size   = 512;
    unsigned int num_blocks         = ceil(((float) num_threads) / block_size);

    unsigned int mem_size           = outer * sizeof(REAL);

    //(*res) = (REAL*) malloc(mem_size);
    {
        float* d_out;
        cudaMalloc((void**)&d_out, mem_size);
    
        kernelGetResults<<< num_blocks, block_size>>> (globsList, d_out, outer);
        cudaThreadSynchronize();
        
        //cuda results to mem
        cudaMemcpy(res, d_out, mem_size, cudaMemcpyDeviceToHost);
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
    PrivGlobsCuda* globsList;
    const unsigned numZ = max(numX, numY);
    //cudaMalloc((void**)&globsList, outer*sizeof(struct PrivGlobsCuda));

    printf("init begin\n");
    init(&globsList, outer, s0, alpha, nu, t, numX, numY, numT);
    printf("init done\n");
    



    ///////////////////////////////////////////////////

// {
//         //unsigned s = numX*numY;
//         unsigned size = 4;
//         unsigned mem_size = size*sizeof(REAL);

//         unsigned num_threads = size;
//         unsigned block_size = 512;
//         unsigned int num_blocks = ceil(((float) num_threads) / block_size);

//         REAL *res, *d_res;
//         cudaMalloc((void**)&d_res, mem_size);
//         res = (REAL*) malloc(mem_size);

//         getList<<< num_blocks, block_size>>>(globsList, d_res, size);
//         cudaThreadSynchronize();

//         cudaMemcpy(res, d_res, mem_size, cudaMemcpyDeviceToHost);

//         printf("\nres = [\n");
//         for(unsigned i=0; i < size; i++)
//             printf("[%d] = %.5f\n", i, res[i]);
//         printf("\n]\n");

//         //exit(0);
//     }

    //////////////////////////////////////////////////////

    
    
    for(int g = numT-2;g>=0;--g){ //seq
        //updateParams()
        printf("update begin\n");
        updateWrapper(globsList, g, numX, numY, outer, alpha, beta, nu);
        printf("update done\n");
        //rollback()
        printf("rollback begin\n");
        rollbackWrapper(globsList, g, outer, numX, numY, numZ);
        printf("rollback done\n");
    }
    getResultsWrapper(globsList, outer, res);
}


//#endif // PROJ_CORE_ORIG
