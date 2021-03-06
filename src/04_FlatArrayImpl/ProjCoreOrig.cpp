#include "ProjHelperFun.h"
#include "Constants.h"

#include "TridagPar.h"

void printFlatMatrix(REAL* matrix, unsigned int rows, unsigned int cols){
    printf("Matrix[%d,%d] = \n[\n", rows, cols);
    for(unsigned int i=0; i< rows; ++i){
        printf("[");
        for(unsigned int j=0; j< cols; ++j){
            printf("%.5f, ", matrix[i*cols+j]);
        }
        printf("]\n");
    }
    printf("]\n");
}

void printVectMatrix(vector<REAL> matrix, unsigned int rows, unsigned int cols){
    printf("Matrix[%d,%d] = \n", rows, cols);
    for(unsigned int i=0; i< rows; ++i){
        for(unsigned int j=0; j< cols; ++j){
            printf("[%d] = %.5f \n",i*cols+j, matrix[i*cols+j]);
        }
    }
    printf("\n");
}

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
                  //&globs.myResult[idx2d(i,0, globs.myResultCols)],&yy[0]);
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
    //vector<PrivGlobs> globs(outer, PrivGlobs(numX, numY, numT));
    PrivGlobs *globs = (PrivGlobs*) malloc(outer*sizeof(struct PrivGlobs));
    #pragma omp parallel for default(shared) schedule(static) if(outer>8)
    for(int i = 0 ; i < outer ; i++) {
        globs[i] = PrivGlobs(numX,numY,numT);
    }

    #pragma omp parallel for default(shared) schedule(static) if(outer>8)
    for( unsigned i = 0; i < outer; ++ i ) { //par
            initGrid(s0,alpha,nu,t, numX, numY, numT, globs[i]);
            initOperator(globs[i].myX, globs[i].myXsize, globs[i].myDxx, globs[i].myDxxCols);
            initOperator(globs[i].myY, globs[i].myYsize, globs[i].myDyy, globs[i].myDyyCols);
            setPayoff(0.001*i, globs[i]);
    }


    //printFlatMatrix(globs[0].myX, 32, 1);
    //printVectMatrix(globs[0].myDxx, 32, 4);
    //printFlatMatrix(globs[0].myDxx, 32, 4);

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

    //TODO: Free all struct and their pointers.
}

//#endif // PROJ_CORE_ORIG
