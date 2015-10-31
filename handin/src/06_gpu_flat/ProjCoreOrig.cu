//#include "ProjHelperFun.cu.h"
#include "Constants.h"
#include "InitKernels.cu.h"
#include "CoreKernels.cu.h"
#include "tridagpar.cu.h"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////DEBUGGING//////////////////////
void printMatrix(REAL* matrix, unsigned int rows, unsigned int cols){
    printf("Matrix = \n[\n");
    for(unsigned int i=0; i< rows; ++i){
        printf("[");
        for(unsigned int j=0; j< cols; ++j){
            printf("%.5f, ", matrix[i*cols+j]);
        }
        printf("]\n");
    }
    printf("]\n");
}

__global__ void getList(PrivGlobsCuda* globsList, 
                                 REAL* res_out,
                                 const unsigned size,
                                 int g,
                                 REAL* mat
){
    const unsigned int gid = threadIdx.x + blockIdx.x*blockDim.x;

    PrivGlobsCuda globs = globsList[8];
    if(gid < size){
        res_out[gid] = globs.myResult[gid];
    }
}

void cpListSeq(REAL** h_out, REAL* d_in, const unsigned size, 
               const unsigned outer, unsigned index){
    unsigned mem_size = outer*size*sizeof(REAL);
    printf("cpListSeq1\n");
    REAL* tmp = (REAL*) malloc(mem_size);
    printf("cpListSeq2\n");

    cudaMemcpy(tmp, d_in, mem_size, cudaMemcpyDeviceToHost);
    tmp = &tmp[index*size];
    printf("cpListSeq3\n");
    for(unsigned i=0; i<size; i++){
        (*h_out)[i] = tmp[i];
    }
    free(tmp);
}

////////////////////////////////////////////////////////////////////////////////



//wrapper for the kernelUpdate
void updateWrapper( PrivGlobs* globs, const unsigned g, const unsigned outer, 
                   const REAL alpha, const REAL beta, const REAL nu
){
    PrivGlobs glob = globs[0];
    unsigned myXsize = glob.myXsize;
    unsigned myYsize = glob.myYsize;
    unsigned myVarXCols = glob.myVarXCols;
    unsigned myVarXRows = glob.myVarXRows;
    unsigned myVarYCols = glob.myVarYCols;
    unsigned myVarYRows = glob.myVarYRows;
    unsigned myTimelineSize = glob.myTimelineSize;

    const int x = myXsize;
    const int y = myYsize;
    const int z = outer;

    const int dimx = ceil( ((float)x) / TVAL );
    const int dimy = ceil( ((float)y) / TVAL );
    const int dimz = z;
    dim3 block(TVAL,TVAL,1), grid(dimx,dimy,dimz);

    REAL *d_myVarX, *d_myX, *d_myVarY, *d_myY, *d_myTimeline;

    globToDevice(globs, outer, myXsize, &d_myX, 1);
    globToDevice(globs, outer, myYsize, &d_myY, 2);
    globToDevice(globs, outer, myVarXCols*myVarXRows, &d_myVarX, 5);
    globToDevice(globs, outer, myVarYCols*myVarYRows, &d_myVarY, 6);
    globToDevice(globs, outer, myTimelineSize, &d_myTimeline, 3);

    kernelUpdate <<< grid, block>>>(d_myVarX, d_myX, d_myVarY, d_myY, 
                                    d_myTimeline, myXsize, myYsize, myVarXCols, 
                                    myVarXRows, myVarYCols, myVarYRows, 
                                    myTimelineSize, g, outer, alpha, beta, 
                                    nu);
    cudaThreadSynchronize();
    globFromDevice(globs, outer, myVarXCols*myVarXRows, d_myVarX, 5);
    globFromDevice(globs, outer, myVarYCols*myVarYRows, d_myVarY, 6);
}


void rollbackWrapper(PrivGlobs* globs, const unsigned g, 
                     const unsigned outer, const unsigned numX, 
                     const unsigned numY
){
    // create all arrays as multidim arrays for rollback()
    REAL *u, *uT, *v, *y, *yy;

    cudaMalloc((void**)&u,  outer*( numY*numX*sizeof(REAL)  ));
    cudaMalloc((void**)&uT, outer*( numX*numY*sizeof(REAL)  ));
    cudaMalloc((void**)&v,  outer*( numX*numY*sizeof(REAL)  ));
    cudaMalloc((void**)&y,  outer*( numX*numY*sizeof(REAL)  ));
    cudaMalloc((void**)&yy, outer*( numX*numY*sizeof(REAL)  ));

    REAL *a, *b, *c, *aT, *bT, *cT;
    cudaMalloc((void**)&a,  outer*( numY*numX*sizeof(REAL)  ));
    cudaMalloc((void**)&b,  outer*( numY*numX*sizeof(REAL)  ));
    cudaMalloc((void**)&c,  outer*( numY*numX*sizeof(REAL)  ));
    cudaMalloc((void**)&aT, outer*( numX*numY*sizeof(REAL)  ));
    cudaMalloc((void**)&bT, outer*( numX*numY*sizeof(REAL)  ));
    cudaMalloc((void**)&cT, outer*( numX*numY*sizeof(REAL)  ));

    const int x = max(numX, numY);  //max(myXsize, numY), myXsize = numX

    int dimx = ceil( ((float)x) / TVAL );
    int dimy = ceil( ((float)x) / TVAL );
    int dimz = outer;
    dim3 block(TVAL,TVAL,1), grid(dimx,dimy,dimz);

    unsigned int sh_mem_size =  TVAL*TVAL;//numY*numX*outer;

    REAL *myTimeline, *myVarX, *myVarY, *myResult, *myDxx, *myDyy;
    PrivGlobs glob = globs[0];
    unsigned myTimelineSize = glob.myTimelineSize;
    unsigned myVarXRows = glob.myVarXRows;
    unsigned myVarXCols = glob.myVarXCols;
    unsigned myVarYRows = glob.myVarYRows;
    unsigned myVarYCols = glob.myVarYCols;
    unsigned myResultRows = glob.myResultRows;
    unsigned myResultCols = glob.myResultCols;
    unsigned myDxxRows = glob.myDxxRows;
    unsigned myDxxCols = glob.myDxxCols;
    unsigned myDyyRows = glob.myDyyRows;
    unsigned myDyyCols = glob.myDyyCols;

    globToDevice(globs, outer, myTimelineSize, &myTimeline, 3);
    globToDevice(globs, outer, myVarXRows*myVarXCols, &myVarX, 5);
    globToDevice(globs, outer, myVarYRows*myVarYCols, &myVarY, 6);
    globToDevice(globs, outer, myResultRows*myResultCols, &myResult, 4);
    globToDevice(globs, outer, myDxxRows*myDxxCols, &myDxx, 7);
    globToDevice(globs, outer, myDyyRows*myDyyCols, &myDyy, 8);

    //oh the humanity!
    kernelRollback1 <<< grid, block >>> (myTimeline, myVarX, myVarY, myResult, 
                                         myDxx, myDyy,
                                         myTimelineSize, myVarXRows, myVarXCols, 
                                         myVarYRows, myVarYCols, myResultRows, 
                                         myResultCols, myDxxRows, myDxxCols, 
                                         myDyyRows, myDyyCols, g, outer, 
                                         u, uT, v, y, 
                                         a, b, c, aT, bT, cT
                                        );


////sequential part
    for(unsigned k = 0; k < outer; ++k){
        unsigned i,j;

        REAL *us = (REAL*) malloc(numY*numX*sizeof(REAL));           // [numY][numX]
        REAL *uTs = (REAL*) malloc(numX*numY*sizeof(REAL));          // [numX][numY]
        REAL *vs = (REAL*) malloc(numX*numY*sizeof(REAL));           // [numX][numY]
        REAL *ys = (REAL*) malloc(numX*numY*sizeof(REAL));           // [numX][numY]
        REAL *yys = (REAL*) malloc(numY*sizeof(REAL));           // [max(numX,numY)]
        
        REAL *as = (REAL*) malloc(numY*numX*sizeof(REAL));           // [numY][numY]
        REAL *bs = (REAL*) malloc(numY*numX*sizeof(REAL));           // [numY][numY]
        REAL *cs = (REAL*) malloc(numY*numX*sizeof(REAL));           // [numY][numY]
        REAL *aTs = (REAL*) malloc(numX*numY*sizeof(REAL));           // [numY][numY]
        REAL *bTs = (REAL*) malloc(numX*numY*sizeof(REAL));           // [numY][numY]
        REAL *cTs = (REAL*) malloc(numX*numY*sizeof(REAL));           // [numZ][numY]

        PrivGlobs glob = globs[k];
        REAL dtInv = 1.0/(glob.myTimeline[g+1]-glob.myTimeline[g]);

        //printf("\nbefore cpListSeq\n");
        cpListSeq(&aTs, aT, numY*numX, outer, k);
        cpListSeq(&bTs, bT, numY*numX, outer, k);
        cpListSeq(&cTs, cT, numY*numX, outer, k);

        cpListSeq(&uTs, uT, numY*numX, outer, k);

        printMatrix(as, numY, numX);

        //printf("\nafter cpListSeq\n");

        transpose2d(uTs, &us, numY, numX);
        transpose2d(aTs, &as, numY, numX);
        transpose2d(bTs, &bs, numY, numX);
        transpose2d(cTs, &cs, numY, numX);

        for(j=0;j<numY;j++) { // par
            // h ere yys should have size [numX]
            tridagPar(&as[idx2d(j,0,numX)], &bs[idx2d(j,0,numX)], &cs[idx2d(j,0,numX)]
                     ,&us[idx2d(j,0,numX)],numX,&us[idx2d(j,0,numX)],&yys[0]);
        }


        for(i=0;i<numX;i++) { // par
            // parallelizable via loop distribution / array expansion.
            for(j=0;j<numY;j++) { // par  // here as, bs, cs should have size [numY]
                aTs[idx2d(i,j,numY)] =       - 0.5*(0.5*glob.myVarY[idx2d(i,j,glob.myVarYCols)]*glob.myDyy[idx2d(j,0,glob.myDyyCols)]);
                bTs[idx2d(i,j,numY)] = dtInv - 0.5*(0.5*glob.myVarY[idx2d(i,j,glob.myVarYCols)]*glob.myDyy[idx2d(j,1,glob.myDyyCols)]);
                cTs[idx2d(i,j,numY)] =       - 0.5*(0.5*glob.myVarY[idx2d(i,j,glob.myVarYCols)]*glob.myDyy[idx2d(j,2,glob.myDyyCols)]);
            }
        }
        transpose2d(aTs, &as, numY, numX);
        transpose2d(bTs, &bs, numY, numX);
        transpose2d(cTs, &cs, numY, numX);

        transpose2d(us, &uTs, numX, numY); //Must retranspose to uTs because prev tridag
                                         // modified us.
        // Coalesced memory acces.
        for(i=0;i<numX;i++) { // par
            for(j=0;j<numY;j++) { // par
                ys[idx2d(i,j,numY)] = dtInv * uTs[idx2d(i,j,numY)]
                                   - 0.5*vs[idx2d(i,j,numY)];
            }
        }


        for(i=0;i<numX;i++) { // par
            // here yys should have size [numX]

            tridagPar(&aTs[idx2d(i,0,numY)], &bTs[idx2d(i,0,numY)],
                      &cTs[idx2d(i,0,numY)], &ys[idx2d(i,0,numY)], numY,
                      &glob.myResult[idx2d(i,0,glob.myResultCols)],&yys[0]);
        }

        free(us);
        free(uTs);
        free(vs);
        free(ys);
        free(yys);
        free(as);
        free(bs);
        free(cs);
        free(aTs);
        free(bTs);
        free(cTs);
        
    }
    
////////////////////////
// This code was supposed to be the rest of the CUDA implementation, but it is
// Commented out because it does not work yet.


    /*

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

    kernelRollback2 <<< grid, block>>> ( myTimeline, myVarX, myVarY, 
                                         myDxx, myDyy,
                                         myTimelineSize,
                                         myVarXRows, myVarXCols, 
                                         myVarYRows, myVarYCols, 
                                         myDxxRows, myDxxCols, 
                                         myDyyRows, myDyyCols,
                                         g, outer, 
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

    kernelRollback3 <<< grid, block>>> (myTimeline, myTimelineSize, numX, numY,
                                        g, outer, uT, v, y);
    cudaThreadSynchronize();

    //tridag2(globsList, outer, y, yy, aT, bT, cT, numX, numY, numZ);
    {
        unsigned myResultSize = myResultRows*myResultCols;
        dimx = ceil( ((float)numY) / TVAL );
        dimy = ceil( ((float)numX) / TVAL );
        dim3 block(TVAL,TVAL,1), grid(dimx,dimy,dimz);
        kernelTridag2 <<< block, grid, sh_mem_size >>> 
                            (myResult, myResultSize, outer, y, yy, aT, bT, cT, 
                             numX, numY);
        cudaThreadSynchronize();

        globFromDevice(globs, outer, myResultSize, myResult, 4);
    }
*/
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

        setPayoffWrapper(globs, outer, numX, numY);
    }
}


//////////////////////// sequential ////////////////////////

void updateParams(const unsigned g, const REAL alpha, const REAL beta, const REAL nu, PrivGlobs& globs)
{
    // parallelizable directly since all reads and writes are independent.
    // Degree of parallelism: myX.size*myY.size
    // Access to myVarX and myVarY is already coalesced.
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
    REAL *y = (REAL*) malloc(numX*numY*sizeof(REAL));           // [numX][numZ]
    REAL *yy = (REAL*) malloc(numY*sizeof(REAL));           // [max(numX,numY)]


    for(i=0;i<numX;i++) { //par
        for(j=0;j<numY;j++) { //par

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

    transpose2d(uT, &u, numY, numX);

    REAL *a = (REAL*) malloc(numY*numX*sizeof(REAL));           // [numY][numZ]
    REAL *b = (REAL*) malloc(numY*numX*sizeof(REAL));           // [numY][numZ]
    REAL *c = (REAL*) malloc(numY*numX*sizeof(REAL));           // [numY][numZ]
    REAL *aT = (REAL*) malloc(numX*numY*sizeof(REAL));           // [numZ][numY]
    REAL *bT = (REAL*) malloc(numX*numY*sizeof(REAL));           // [numZ][numY]
    REAL *cT = (REAL*) malloc(numX*numY*sizeof(REAL));           // [numZ][numY]

    for(i=0;i<numX;i++) {  // par // here a, b,c should have size [numX]
        for(j=0;j<numY;j++) { // par
            aT[idx2d(i,j,numY)] =    - 0.5*(0.5*globs.myVarX[idx2d(i,j,globs.myVarXCols)]*globs.myDxx[idx2d(i,0,globs.myDxxCols)]);
            bT[idx2d(i,j,numY)] = dtInv
                            - 0.5*(0.5*globs.myVarX[idx2d(i,j,globs.myVarXCols)]*globs.myDxx[idx2d(i,1,globs.myDxxCols)]);
            cT[idx2d(i,j,numY)] =    - 0.5*(0.5*globs.myVarX[idx2d(i,j,globs.myVarXCols)]*globs.myDxx[idx2d(i,2,globs.myDxxCols)]);
        }
    }
    transpose2d(aT, &a, numY, numX);
    transpose2d(bT, &b, numY, numX);
    transpose2d(cT, &c, numY, numX);

    for(j=0;j<numY;j++) { // par
        // here yy should have size [numX]
        tridagPar(&a[idx2d(j,0,numX)], &b[idx2d(j,0,numX)], &c[idx2d(j,0,numX)]
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
    transpose2d(aT, &a, numY, numX);
    transpose2d(bT, &b, numY, numX);
    transpose2d(cT, &c, numY, numX);

    transpose2d(u, &uT, numX, numY); //Must retranspose to uT because prev tridag
                                     // modified u.
    // Coalesced memory acces.
    for(i=0;i<numX;i++) { // par
        for(j=0;j<numY;j++) { // par
            y[idx2d(i,j,numY)] = dtInv * uT[idx2d(i,j,numY)]
                               - 0.5*v[idx2d(i,j,numY)];
        }
    }


    for(i=0;i<numX;i++) { // par
        // here yy should have size [numX]

        tridagPar(&aT[idx2d(i,0,numY)], &bT[idx2d(i,0,numY)],
                  &cT[idx2d(i,0,numY)], &y[idx2d(i,0,numY)], numY,
                  &globs.myResult[idx2d(i,0,globs.myResultCols)],&yy[0]);
    }

    free(u);
    free(uT);
    free(v);
    free(y);
    free(yy);
    free(a);
    free(b);
    free(c);
    free(aT);
    free(bT);
    free(cT);
}

void initGrid(  const REAL s0, const REAL alpha, const REAL nu,const REAL t,
                const unsigned numX, const unsigned numY, const unsigned numT, PrivGlobs& globs
) {
    // Can be parallelized directly as each iteration writes to independent
    // globs.myTimeline indices
    for(unsigned i=0;i<numT;++i)  // par
        globs.myTimeline[i] = t*i/(numT-1);

    const REAL stdX = 20.0*alpha*s0*sqrt(t);
    const REAL dx = stdX/numX;
    globs.myXindex = static_cast<unsigned>(s0/dx) % numX;

    // Can be parallelized directly as each iteration writes to independent
    // globs.myX indices.
    for(unsigned i=0;i<numX;++i)  // par
        globs.myX[i] = i*dx - globs.myXindex*dx + s0;

    const REAL stdY = 10.0*nu*sqrt(t);
    const REAL dy = stdY/numY;
    const REAL logAlpha = log(alpha);
    globs.myYindex = static_cast<unsigned>(numY/2.0);

    // Can be parallelized directly as each iteration writes to independent
    // globs.myY indices.
    for(unsigned i=0;i<numY;++i)  // par
        globs.myY[i] = i*dy - globs.myYindex*dy + logAlpha;
}

void initOperator(  const REAL *x, unsigned xsize,
                    REAL* &Dxx, unsigned DxxCols
) {
    const unsigned n = xsize;

    REAL dxl, dxu;

    //  lower boundary
    dxl      =  0.0;
    dxu      =  x[1] - x[0];

    Dxx[idx2d(0,0,DxxCols)] =  0.0;
    Dxx[idx2d(0,1,DxxCols)] =  0.0;
    Dxx[idx2d(0,2,DxxCols)] =  0.0;
    Dxx[idx2d(0,3,DxxCols)] =  0.0;

    //  standard case
    // Can be parallelized directly as each iteration writes to independent
    // Dxx indices. x is only read, so each iteration is independent.
    // x could be put in shared memory.
    for(unsigned i=1;i<n-1;i++) // par
    {
        dxl      = x[i]   - x[i-1];
        dxu      = x[i+1] - x[i];

        Dxx[idx2d(i,0,DxxCols)] =  2.0/dxl/(dxl+dxu);
        Dxx[idx2d(i,1,DxxCols)] = -2.0*(1.0/dxl + 1.0/dxu)/(dxl+dxu);
        Dxx[idx2d(i,2,DxxCols)] =  2.0/dxu/(dxl+dxu);
        Dxx[idx2d(i,3,DxxCols)] =  0.0;
    }

    //  upper boundary
    dxl        =  x[n-1] - x[n-2];
    dxu        =  0.0;

    Dxx[idx2d(n-1,0,DxxCols)] = 0.0;
    Dxx[idx2d(n-1,1,DxxCols)] = 0.0;
    Dxx[idx2d(n-1,2,DxxCols)] = 0.0;
    Dxx[idx2d(n-1,3,DxxCols)] = 0.0;
}


void setPayoff(const REAL strike, PrivGlobs& globs )
{
    // Assuming globs is local the loop can be parallelized since
    // - reads independent
    // - writes independent (not same as read array)
    // Problem in payoff. Can be put inline (scalar variable), but this results
    // in myX.size*myY.size mem accesses.
    // If array expansion, only myX.size+myY.size mem accesses.
    // TODO: To be determined based on myX.size and myY.size.
    // Small dataset= NUM_X = 32 ; NUM_Y = 256
    // 8192 vs 288 -- array expansion is preferable.

    // Array expansion **DONE.
    REAL payoff[globs.myXsize];
    for(unsigned i=0;i<globs.myXsize;++i)
        payoff[i] = max(globs.myX[i]-strike, (REAL)0.0);

    // Already coalesced.
    for(unsigned i=0;i<globs.myXsize;++i) { // par
        for(unsigned j=0;j<globs.myYsize;++j) // par
            globs.myResult[idx2d(i,j,globs.myResultCols)] = payoff[i];
    }
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

    PrivGlobs *globs = (PrivGlobs*) malloc(outer*sizeof(struct PrivGlobs));
    for(int i = 0 ; i < outer ; i++) {
        globs[i] = PrivGlobs(numX,numY,numT);
    }



    init(globs, outer, s0, alpha, nu, t, numX, numY, numT);

    for(int i = numT-2;i>=0;--i){ //seq
        updateWrapper(globs, i, outer, alpha, beta, nu);
        //rollbackWrapper(globs, i, outer, numX, numY);
        for( unsigned j = 0; j < outer; ++ j ) { //par
            // updateParams(i,alpha,beta,nu,globs[j]);
            rollback(i, globs[j]);
        }
    }
    // parallel assignment of results.
    for( unsigned j = 0; j < outer; ++ j ) { //par
        res[j] = globs[j].myResult[idx2d(globs[j].myXindex,globs[j].myYindex,globs[j].myResultCols)];
    }
}