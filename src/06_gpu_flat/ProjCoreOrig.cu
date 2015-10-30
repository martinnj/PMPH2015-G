//#include "ProjHelperFun.cu.h"
#include "Constants.h"
#include "InitKernels.cu.h"
#include "CoreKernels.cu.h"
#include "tridagpar.cu.h"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////DEBUGGING//////////////////////
__global__ void getList(PrivGlobsCuda* globsList, 
                                 REAL* res_out,
                                 const unsigned size,
                                 // const unsigned numX,
                                 // const unsigned numY,
                                 int g,
                                 REAL* mat
){
    const unsigned int gid = threadIdx.x + blockIdx.x*blockDim.x;

    PrivGlobsCuda globs = globsList[8];
    if(gid < size){
        //res_out[gid] = globs.myVarX[idx2d(10,158,globs.myVarXCols)];
        //res_out[gid] = globs.myTimeline[gid];
        res_out[gid] = globs.myResult[gid];
        // res_out[gid] = mat[gid];
        //res_out[gid] = mat[idx2d(10,gid,size)];


        // res_out[0] = (REAL) globs.myXindex;
        // res_out[1] = (REAL) globs.myYindex;
        // res_out[2] = (REAL) globs.myResult[idx2d(globs.myXindex,
        //                                     globs.myYindex,
        //                                     globs.myResultCols)];
        //res_out[gid] = globs.myResult[idx2d(10,gid,size)];
        // REAL dtInv = 1.0/(globs.myTimeline[g+1]-globs.myTimeline[g]);
        // REAL p = dtInv*globs.myResult[idx2d(gid,250,globs.myResultCols)];
        
        // res_out[gid] = p*globs.myDxx[idx2d(gid,1,globs.myDxxCols)] //[i][1]
        //                 * globs.myResult[idx2d(gid, 250, globs.myResultCols)];//globs.myDxx[idx2d(gid,1,globs.myDxxCols)];

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
                     const unsigned numY
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

    const int x = max(numX, numY);  //max(myXsize, numY), myXsize = numX
    //const int y = numZ = x;    //max(y, myYsize), myYsize = numY

    int dimx = ceil( ((float)x) / TVAL );
    int dimy = ceil( ((float)x) / TVAL );
    int dimz = outer;
    dim3 block(TVAL,TVAL,1), grid(dimx,dimy,dimz);

    //const unsigned n = numY*numX;
    // unsigned int block_size = 512;
    // unsigned int num_blocks = (n + (block_size - 1)) / block_size;
    unsigned int sh_mem_size =  TVAL*TVAL;//numY*numX*outer;

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


    {
        int dimx = ceil( ((float)numX) / TVAL );
        int dimy = ceil( ((float)numY) / TVAL );
        dim3 block(TVAL,TVAL,1), grid(dimx,dimy,dimz);
        //Tridag 1
        //tridag1(outer, u, yy, a, b, c, numX, numY, numZ);
        kernelTridag1 <<< block, grid, sh_mem_size >>> 
                                        (outer, u, yy, a, b, c, numX, numY);
        cudaThreadSynchronize();

    }




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

// {
//         unsigned s = numX;
//         unsigned size = s;
//         unsigned mem_size = size*sizeof(REAL);

//         unsigned num_threads = size;
//         unsigned block_size = 512;
//         unsigned int num_blocks = ceil(((float) num_threads) / block_size);

//         REAL *res, *d_res;
//         cudaMalloc((void**)&d_res, mem_size);
//         res = (REAL*) malloc(mem_size);

//         getList<<< num_blocks, block_size>>>(globsList, d_res, size, g, uT);
//         cudaThreadSynchronize();

//         cudaMemcpy(res, d_res, mem_size, cudaMemcpyDeviceToHost);

//         printf("\nBEFORE TRIDAG\n");
//         printf("res = [\n");
//         for(unsigned i=0; i < size; i++)
//             printf("[%d] = %.5f\n", i, res[i]);
//         printf("\n]\n");

//         //exit(0);
//     }
    //tridag2(globsList, outer, y, yy, aT, bT, cT, numX, numY, numZ);
    {
        dimx = ceil( ((float)numY) / TVAL );
        dimy = ceil( ((float)numX) / TVAL );
        dim3 block(TVAL,TVAL,1), grid(dimx,dimy,dimz);
        kernelTridag2 <<< block, grid, sh_mem_size >>> 
                            (globsList, outer, y, yy, aT, bT, cT, numX, numY);
        cudaThreadSynchronize();
    }
    /////////////////////////////////////////////////
// {
//         unsigned s = numX;
//         unsigned size = s;
//         unsigned mem_size = size*sizeof(REAL);

//         unsigned num_threads = size;
//         unsigned block_size = 512;
//         unsigned int num_blocks = ceil(((float) num_threads) / block_size);

//         REAL *res, *d_res;
//         cudaMalloc((void**)&d_res, mem_size);
//         res = (REAL*) malloc(mem_size);

//         getList<<< num_blocks, block_size>>>(globsList, d_res, size, g, uT);
//         cudaThreadSynchronize();

//         cudaMemcpy(res, d_res, mem_size, cudaMemcpyDeviceToHost);

//         printf("AFTER TRIDAG\n");
//         printf("res = [\n");
//         for(unsigned i=0; i < size; i++)
//             printf("[%d] = %.5f\n", i, res[i]);
//         printf("\n]\n");

//         //exit(0);
//     }
////////////////////////////////////////////////////
/*
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
    }*/

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






void init(PrivGlobs* globs, const unsigned outer, const REAL s0, 
          const REAL alpha, const REAL nu, const REAL t, const unsigned numX, 
          const unsigned numY, const unsigned numT
){
    
    {
        //init grid
        initGridWrapper(globs, outer, numX, numY, numT, t, alpha, s0, nu);

        //init op Dxx and Dyy
        initOperatorWrapper(globs, outer);
        //init op Dyy
        //initOperatorWrapper(*globsList, numY, outer, 2);

        setPayoffWrapper(globs, outer, numX, numY);
    }
}


//////////////////////// sequential ////////////////////////

void updateParams(const unsigned g, const REAL alpha, const REAL beta, const REAL nu, PrivGlobs& globs)
{
    // parallelizable directly since all reads and writes are independent.
    // Degree of parallelism: myX.size*myY.size
    // Access to myVarX and myVarY is already coalesced.
    // TODO: Examine how tiling/shared memory can be used.
    for(unsigned i=0;i<globs.myXsize;++i) // par
        for(unsigned j=0;j<globs.myYsize;++j) { // par
            globs.myVarX[idx2d(i,j,globs.myVarXCols)] = exp(2.0*(  beta*log(globs.myX[i])
                                          + globs.myY[j]
                                          - 0.5*nu*nu*globs.myTimeline[g] )
                                    );
            globs.myVarY[idx2d(i,j,globs.myVarYCols)] = exp(2.0*(  alpha*log(globs.myX[i])
                                          + globs.myY[j]
                                          - 0.5*nu*nu*globs.myTimeline[g] )
                                    ); // nu*nu
        }
}

void rollback( const unsigned g, PrivGlobs& globs ) {
    unsigned numX = globs.myXsize,
             numY = globs.myYsize;

    unsigned numZ = max(numX,numY);
    unsigned i, j;

    REAL dtInv = 1.0/(globs.myTimeline[g+1]-globs.myTimeline[g]);

    REAL *u = (REAL*) malloc(numY*numX*sizeof(REAL));           // [numY][numX]
    REAL *uT = (REAL*) malloc(numX*numY*sizeof(REAL));          // [numX][numY]
    REAL *v = (REAL*) malloc(numX*numY*sizeof(REAL));           // [numX][numY]
    REAL *y = (REAL*) malloc(numX*numZ*sizeof(REAL));           // [numX][numZ]
    REAL *yy = (REAL*) malloc(numZ*sizeof(REAL));           // [max(numX,numY)]


    for(i=0;i<numX;i++) { //par
        for(j=0;j<numY;j++) { //par
            //TODO: This can be combined in the tridag kernel, in shared mem.

            uT[idx2d(i,j,numY)] = dtInv*globs.myResult[idx2d(i,j,globs.myResultCols)];
            if(i > 0) {
                uT[idx2d(i,j,numY)] += 0.5*( 0.5*globs.myVarX[idx2d(i,j,globs.myVarXCols)]*globs.myDxx[idx2d(i,0,globs.myDxxCols)] )
                            * globs.myResult[idx2d(i-1,j,globs.myResultCols)];
            }
            uT[idx2d(i,j,numY)]  += 0.5*( 0.5*globs.myVarX[idx2d(i,j,globs.myVarXCols)]*globs.myDxx[idx2d(i,1,globs.myDxxCols)] )
                            * globs.myResult[idx2d(i,j,globs.myResultCols)];
            if(i < numX-1) {
                uT[idx2d(i,j,numY)] += 0.5*( 0.5*globs.myVarX[idx2d(i,j,globs.myVarXCols)]*globs.myDxx[idx2d(i,2,globs.myDxxCols)] )
                            * globs.myResult[idx2d(i+1,j,globs.myResultCols)];
            }
        }
    }

  
    for(i=0;i<numX;i++) { //par
        for(j=0;j<numY;j++) { //par
            //TODO: This can be combined in the tridag kernel too, as parameters.
            v[idx2d(i,j,numY)] = 0.0;
            if(j > 0) {
              v[idx2d(i,j,numY)] +=  ( 0.5*globs.myVarY[idx2d(i,j,globs.myVarYCols)]*globs.myDyy[idx2d(j,0,globs.myDyyCols)])
                         *  globs.myResult[idx2d(i,j-1,globs.myResultCols)];
            }
            v[idx2d(i,j,numY)]  +=  ( 0.5*globs.myVarY[idx2d(i,j,globs.myVarYCols)]*globs.myDyy[idx2d(j,1,globs.myDyyCols)])
                         *  globs.myResult[idx2d(i,j,globs.myResultCols)];
            if(j < numY-1) {
              v[idx2d(i,j,numY)] += ( 0.5*globs.myVarY[idx2d(i,j,globs.myVarYCols)]*globs.myDyy[idx2d(j,2,globs.myDyyCols)])
                         *  globs.myResult[idx2d(i,j+1,globs.myResultCols)];
            }
            uT[idx2d(i,j,numY)] += v[idx2d(i,j,numY)];
        }
    }

    transpose(uT, &u, numY, numX);

    REAL *a = (REAL*) malloc(numY*numZ*sizeof(REAL));           // [numY][numZ]
    REAL *b = (REAL*) malloc(numY*numZ*sizeof(REAL));           // [numY][numZ]
    REAL *c = (REAL*) malloc(numY*numZ*sizeof(REAL));           // [numY][numZ]
    //vector<vector<REAL> > a(numY, vector<REAL>(numZ)), b(numY, vector<REAL>(numZ)), c(numY, vector<REAL>(numZ));     // [max(numX,numY)]
    REAL *aT = (REAL*) malloc(numZ*numY*sizeof(REAL));           // [numZ][numY]
    REAL *bT = (REAL*) malloc(numZ*numY*sizeof(REAL));           // [numZ][numY]
    REAL *cT = (REAL*) malloc(numZ*numY*sizeof(REAL));           // [numZ][numY]
    //vector<vector<REAL> > aT(numZ, vector<REAL>(numY)), bT(numZ, vector<REAL>(numY)), cT(numZ, vector<REAL>(numY));

    for(i=0;i<numX;i++) {  // par // here a, b,c should have size [numX]
        for(j=0;j<numY;j++) { // par
            aT[idx2d(i,j,numY)] =    - 0.5*(0.5*globs.myVarX[idx2d(i,j,globs.myVarXCols)]*globs.myDxx[idx2d(i,0,globs.myDxxCols)]);
            bT[idx2d(i,j,numY)] = dtInv
                            - 0.5*(0.5*globs.myVarX[idx2d(i,j,globs.myVarXCols)]*globs.myDxx[idx2d(i,1,globs.myDxxCols)]);
            cT[idx2d(i,j,numY)] =    - 0.5*(0.5*globs.myVarX[idx2d(i,j,globs.myVarXCols)]*globs.myDxx[idx2d(i,2,globs.myDxxCols)]);
        }
    }
    transpose(aT, &a, numY, numZ);
    transpose(bT, &b, numY, numZ);
    transpose(cT, &c, numY, numZ);

    for(j=0;j<numY;j++) { // par
        // here yy should have size [numX]
        tridagPar(&a[idx2d(j,0,numZ)], &b[idx2d(j,0,numZ)], &c[idx2d(j,0,numZ)]
                 ,&u[idx2d(j,0,numX)],numX,&u[idx2d(j,0,numX)],&yy[0]);
    }


    for(i=0;i<numX;i++) { // par
        // parallelizable via loop distribution / array expansion.
        for(j=0;j<numY;j++) { // par  // here a, b, c should have size [numY]
            aT[idx2d(i,j,numY)] =       - 0.5*(0.5*globs.myVarY[idx2d(i,j,globs.myVarYCols)]*globs.myDyy[idx2d(j,0,globs.myDyyCols)]);
            bT[idx2d(i,j,numY)] = dtInv - 0.5*(0.5*globs.myVarY[idx2d(i,j,globs.myVarYCols)]*globs.myDyy[idx2d(j,1,globs.myDyyCols)]);
            cT[idx2d(i,j,numY)] =       - 0.5*(0.5*globs.myVarY[idx2d(i,j,globs.myVarYCols)]*globs.myDyy[idx2d(j,2,globs.myDyyCols)]);
        }
    }
    transpose(aT, &a, numY, numZ);
    transpose(bT, &b, numY, numZ);
    transpose(cT, &c, numY, numZ);

    transpose(u, &uT, numX, numY); //Must retranspose to uT because prev tridag
                                   // modified u.


    // Coalesced memory acces.
    for(i=0;i<numX;i++) { // par
        for(j=0;j<numY;j++) { // par
            y[idx2d(i,j,numZ)] = dtInv * uT[idx2d(i,j,numY)]
                               - 0.5*v[idx2d(i,j,numY)];
        }
    }


    for(i=0;i<numX;i++) { // par
        // here yy should have size [numX]

        tridagPar(&aT[idx2d(i,0,numY)], &bT[idx2d(i,0,numY)],
                  &cT[idx2d(i,0,numY)], &y[idx2d(i,0,numZ)], numY,
                  &globs.myResult[idx2d(i,0,globs.myResultCols)],&yy[0]);
                  //&globs.myResult[idx2d(i,0, globs.myResultCols)],&yy[0]);
    }

    free(u);
    free(uT);
    free(v);
    free(y);
    free(yy);
}

////////////////////////////////////////////////////////////////////////


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
    /*
    // Outerloop - Technically parallelizable, but restricts further
    // parallization further in.
    // If strike and globs is privatized, the loop can be parallelized.
    // Value is the limiting factor since most of the actual work is deeper in
    // the function.

    // Sequential loop (value) in between parallel loops (this loop).
    // Move seq to outer loop via array expansion (globs) and distribution.
    #pragma omp parallel for default(shared) schedule(static) if(outer>8)
    for( unsigned i = 0; i < outer; ++ i ) {
        REAL strike = 0.001*i;
        PrivGlobs    globs(numX, numY, numT);
        res[i] = value( globs, s0, strike, t,
                        alpha, nu,    beta,
                        numX,  numY,  numT );
    }*/
    // globs array expanded. Init moved to individual parallel loop
    //vector<PrivGlobs> globs(outer, PrivGlobs(numX, numY, numT));
        // globs array expanded. Init moved to individual parallel loop
    
    //PrivGlobs *globs = (PrivGlobs*) malloc(outer*sizeof(struct PrivGlobs));
/*
    #pragma omp parallel for default(shared) schedule(static) if(outer>8)
    for( unsigned i = 0; i < outer; ++ i ) { //par
            initGrid(s0,alpha,nu,t, numX, numY, numT, globs[i]);
            initOperator(globs[i].myX, globs[i].myXsize, globs[i].myDxx, globs[i].myDxxCols);
            initOperator(globs[i].myY, globs[i].myYsize, globs[i].myDyy, globs[i].myDyyCols);
            setPayoff(0.001*i, globs[i]);
    }

    // sequential loop distributed.
    for(int i = numT-2;i>=0;--i){ //seq
        // inner loop parallel on each outer (par) instead of each time step (seq).
        #pragma omp parallel for default(shared) schedule(static) if(outer>8)
        for( unsigned j = 0; j < outer; ++ j ) { //par
            updateParams(i,alpha,beta,nu,globs[j]);
            rollback(i, globs[j]);
        }
    }
    // parallel assignment of results.
    #pragma omp parallel for default(shared) schedule(static) if(outer>8)
    for( unsigned j = 0; j < outer; ++ j ) { //par
        res[j] = globs[j].myResult[idx2d(globs[j].myXindex,globs[j].myYindex,globs[j].myResultCols)];
    }
*/

    PrivGlobs *globs = (PrivGlobs*) malloc(outer*sizeof(struct PrivGlobs));
    #pragma omp parallel for default(shared) schedule(static) if(outer>8)
    for(int i = 0 ; i < outer ; i++) {
        globs[i] = PrivGlobs(numX,numY,numT);
    }



printf("wtf\n");
    init(globs, outer, s0, alpha, nu, t, numX, numY, numT);
printf("wtf2\n");


    for(int g = numT-2;g>=0;--g){ //seq
        updateWrapper(globsList, g, numX, numY, outer, alpha, beta, nu);
        rollback(i, globs[j]);
        //rollbackWrapper(globsList, g, outer, numX, numY);
    }
    for( unsigned j = 0; j < outer; ++ j ) { //par
        res[j] = globs[j].myResult[idx2d(globs[j].myXindex,globs[j].myYindex,globs[j].myResultCols)];
    }



    // for(int i = numT-2;i>=0;--i){ //seq
    //     for( unsigned j = 0; j < outer; ++ j ) { //par
    //         updateParams(i,alpha,beta,nu,globs[j]);
    //         rollback(i, globs[j]);
    //     }
    // }
    // // parallel assignment of results.
    // for( unsigned j = 0; j < outer; ++ j ) { //par
    //     res[j] = globs[j].myResult[idx2d(globs[j].myXindex,globs[j].myYindex,globs[j].myResultCols)];
    // }
}












/*
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
 
    PrivGlobsCuda* globsList;
    init(&globsList, outer, s0, alpha, nu, t, numX, numY, numT);

    for(int g = numT-2;g>=0;--g){ //seq
        updateWrapper(globsList, g, numX, numY, outer, alpha, beta, nu);
        rollbackWrapper(globsList, g, outer, numX, numY);
        ///////////////////////////////////////////////
// {
//         unsigned s = numX*numY;
//         unsigned size = s;
//         unsigned mem_size = size*sizeof(REAL);

//         unsigned num_threads = size;
//         unsigned block_size = 512;
//         unsigned int num_blocks = ceil(((float) num_threads) / block_size);

//         REAL *res, *d_res;
//         cudaMalloc((void**)&d_res, mem_size);
//         res = (REAL*) malloc(mem_size);

//         getList<<< num_blocks, block_size>>>(globsList, d_res, size, g, res);
//         cudaThreadSynchronize();

//         cudaMemcpy(res, d_res, mem_size, cudaMemcpyDeviceToHost);

//         printf("\nres = [\n");
//         for(unsigned i=0; i < size; i++)
//             printf("[%d] = %.5f\n", i, res[i]);
//         printf("\n]\n");

//         //exit(0);
//         if(g == numT-3)
//             break;
//     }

////////////////////////////////////////////////////
    }
    getResultsWrapper(globsList, outer, res);

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
}
*/

//#endif // PROJ_CORE_ORIG