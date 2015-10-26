#include "ProjHelperFun.h"
#include "Constants.h"

#include "TridagPar.h"
/*
void updateParams(const unsigned g, const REAL alpha, const REAL beta, 
                  const REAL nu, PrivGlobs& globs)
{
    // parallelizable directly since all reads and writes are independent.
    // Degree of parallelism: myX.size*myY.size
    // Access to myVarX and myVarY is already coalesced.
    // TODO: Examine how tiling/shared memory can be used.
    for(unsigned i=0;i<globs.myXsize;++i) // par
        for(unsigned j=0;j<globs.myYsize;++j) { // par
            globs.myVarX[idx2d(i,j, globs.myVarXCols)] =
                                            exp(2.0*(  beta*log(globs.myX[i])
                                          + globs.myY[j]
                                          - 0.5*nu*nu*globs.myTimeline[g] )
                                    );
            globs.myVarY[idx2d(i,j, globs.myVarXCols)] =
                                            exp(2.0*(  alpha*log(globs.myX[i])
                                          + globs.myY[j]
                                          - 0.5*nu*nu*globs.myTimeline[g] )
                                    ); // nu*nu
        }
}
*/

void updateParams(const unsigned g, const REAL alpha, const REAL beta, const REAL nu, PrivGlobs& globs)
{
    // parallelizable directly since all reads and writes are independent.
    // Degree of parallelism: myX.size*myY.size
    // Access to myVarX and myVarY is already coalesced.
    // TODO: Examine how tiling/shared memory can be used.
    for(unsigned i=0;i<globs.myX.size();++i) // par
        for(unsigned j=0;j<globs.myY.size();++j) { // par
            globs.myVarX[i][j] = exp(2.0*(  beta*log(globs.myX[i])
                                          + globs.myY[j]
                                          - 0.5*nu*nu*globs.myTimeline[g] )
                                    );
            globs.myVarY[i][j] = exp(2.0*(  alpha*log(globs.myX[i])
                                          + globs.myY[j]
                                          - 0.5*nu*nu*globs.myTimeline[g] )
                                    ); // nu*nu
        }
}
/*
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
			globs.myResult[idx2d(i,j, globs.myResultCols)] = payoff[i];
	}
}
*/
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
    REAL payoff[globs.myX.size()];
    for(unsigned i=0;i<globs.myX.size();++i)
        payoff[i] = max(globs.myX[i]-strike, (REAL)0.0);

    // Already coalesced.
    for(unsigned i=0;i<globs.myX.size();++i) { // par
        for(unsigned j=0;j<globs.myY.size();++j) // par
            globs.myResult[i][j] = payoff[i];
    }
}

void
rollback( const unsigned g, PrivGlobs& globs ) {
    /*
    unsigned numX = globs.myXsize,
             numY = globs.myYsize;
    */
    unsigned numX = globs.myX.size(),
             numY = globs.myY.size();

    unsigned numZ = max(numX,numY);

    unsigned i, j;

    REAL dtInv = 1.0/(globs.myTimeline[g+1]-globs.myTimeline[g]);

    REAL *u = (REAL*) malloc(numY*numX*sizeof(REAL));           // [numY][numX]
    //vector<vector<REAL> > u(numY, vector<REAL>(numX));        // [numY][numX]

    REAL *uT = (REAL*) malloc(numX*numY*sizeof(REAL));          // [numX][numY]
    //vector<vector<REAL> > uT(numX, vector<REAL>(numY));       // [numX][numY]

    REAL *v = (REAL*) malloc(numX*numY*sizeof(REAL));           // [numX][numY]
    //vector<vector<REAL> > v(numX, vector<REAL>(numY));        // [numX][numY]
    //vector<REAL> a(numZ), b(numZ), c(numZ), y(numZ);      // [max(numX,numY)]
    //vector<REAL> y(numZ);

    REAL *y = (REAL*) malloc(numX*numZ*sizeof(REAL));           // [numX][numZ]
    //vector<vector<REAL> > y(numX, vector<REAL>(numZ));        // [numX][numZ]

    REAL *yy = (REAL*) malloc(numZ*sizeof(REAL));           // [max(numX,numY)]
    //vector<REAL> yy(numZ);  // temporary used in tridag  // [max(numX,numY)]


    //	explicit x
    // parallelizable directly since all reads and writes are independent.
    // Degree of parallelism: numX*numY.
    // TODO: Examine how tiling/shared memory can be used on globs (.myResult).

    // Reads are coalosced but writes are not.
    // TODO: Coalesced access via matrix transposition of u/uT. **DONE
/*
    for(i=0;i<numX;i++) { //par
        for(j=0;j<numY;j++) { //par
            //TODO: This can be combined in the tridag kernel, in shared mem.

            uT[idx2d(i,j,numY)] = dtInv*globs.myResult[idx2d(i,j,
                                                       globs.myResultCols)];
            REAL x = 0.5*0.5*globs.myVarX[idx2d(i,j, globs.myVarXCols)];
            if(i > 0) {
                uT[idx2d(i,j,numY)] += x*globs.myDxx[idx2d(i,0,globs.myDxxCols)]
                            * globs.myResult[idx2d(i-1,j, globs.myResultCols)];
            }
            uT[idx2d(i,j,numY)]  += x*globs.myDxx[idx2d(i,1,globs.myDxxCols)]
                            * globs.myResult[idx2d(i,j, globs.myResultCols)];
            if(i < numX-1) {
                uT[idx2d(i,j,numY)] += x*globs.myDxx[idx2d(i,2,globs.myDxxCols)]
                            * globs.myResult[idx2d(i+1,j, globs.myResultCols)];
            }
        }
    }
*/
    for(i=0;i<numX;i++) { //par
        for(j=0;j<numY;j++) { //par
            //TODO: This can be combined in the tridag kernel, in shared mem.

            uT[idx2d(i,j,numY)] = dtInv*globs.myResult[i][j];
            if(i > 0) {
                uT[idx2d(i,j,numY)] += 0.5*( 0.5*globs.myVarX[i][j]*globs.myDxx[i][0] )
                            * globs.myResult[i-1][j];
            }
            uT[idx2d(i,j,numY)]  += 0.5*( 0.5*globs.myVarX[i][j]*globs.myDxx[i][1] )
                            * globs.myResult[i][j];
            if(i < numX-1) {
                uT[idx2d(i,j,numY)] += 0.5*( 0.5*globs.myVarX[i][j]*globs.myDxx[i][2] )
                            * globs.myResult[i+1][j];
            }
        }
    }

    //	explicit y
    // parallelizable directly since all reads and writes are independent.
    // Degree of parallelism: numY*numX.
    // TODO: Examine how tiling/shared memory can be used on globs (.myResult).
    // and u.?

    // Reads are coalosced but writes are not.
    // TODO: Coalesced access via matrix transposition.
    // TODO: Interchange loop. **DONE

    // Loop interchanged, u transposed used here also, further utilizing the
    // time used on allocation.
    // Loop interchange chosen over matrix transposition, as then both
    // v, globs.myVarY and globs.myResult would have to be transposed (i.e.
    // mem allocation and computation time on transposition), and we deem this
    // overhead greater than that of globs.myDyy non-coalesced mem access (which
    // can be avoided in transposition).
/*
    for(i=0;i<numX;i++) { //par
        for(j=0;j<numY;j++) { //par
            //TODO: This can be combined in the tridag kernel too, as parameters.
            v[idx2d(i,j,numY)] = 0.0;
            REAL y = 0.5*globs.myVarY[idx2d(i,j, globs.myVarYCols)];

            if(j > 0) {
              v[idx2d(i,j,numY)] +=  (y*globs.myDyy[idx2d(j,0,globs.myDxxCols)])
                         *  globs.myResult[idx2d(i,j-1, globs.myResultCols)];
            }
            v[idx2d(i,j,numY)]  +=   (y*globs.myDyy[idx2d(j,1,globs.myDxxCols)])
                         *  globs.myResult[idx2d(i,j, globs.myResultCols)];
            if(j < numY-1) {
              v[idx2d(i,j,numY)] +=  (y*globs.myDyy[idx2d(j,2,globs.myDxxCols)])
                         *  globs.myResult[idx2d(i,j+j, globs.myResultCols)];
            }
            uT[idx2d(i,j,numY)] += v[idx2d(i,j,numY)];
        }
    }
*/
    for(i=0;i<numX;i++) { //par
        for(j=0;j<numY;j++) { //par
            //TODO: This can be combined in the tridag kernel too, as parameters.
            v[idx2d(i,j,numY)] = 0.0;
            if(j > 0) {
              v[idx2d(i,j,numY)] +=  ( 0.5*globs.myVarY[i][j]*globs.myDyy[j][0] )
                         *  globs.myResult[i][j-1];
            }
            v[idx2d(i,j,numY)]  +=  ( 0.5*globs.myVarY[i][j]*globs.myDyy[j][1] )
                         *  globs.myResult[i][j];
            if(j < numY-1) {
              v[idx2d(i,j,numY)] += ( 0.5*globs.myVarY[i][j]*globs.myDyy[j][2] )
                         *  globs.myResult[i][j+1];
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

    //	implicit x
    // ASSUMING tridag is independent.
    // parallelizable directly since all reads and writes are independent.
    // Degree of parallelism: numY*numX.
    // TODO: MyDxx and myVarX is not coalesced. **DONE
    /*

/*
    // parallelizable via loop distribution / array expansion.
    for(i=0;i<numX;i++) {  // par // here a, b,c should have size [numX]
        for(j=0;j<numY;j++) { // par
            REAL x = 0.5*0.5*globs.myVarX[idx2d(i,j, globs.myVarXCols)];
            aT[idx2d(i,j,numY)] =
                            - x*globs.myDxx[idx2d(i,0,globs.myDxxCols)];
            bT[idx2d(i,j,numY)] = dtInv
                            - x*globs.myDxx[idx2d(i,0,globs.myDxxCols)];
            cT[idx2d(i,j,numY)] =
                            - x*globs.myDxx[idx2d(i,0,globs.myDxxCols)];
        }
    }
*/
    for(i=0;i<numX;i++) {  // par // here a, b,c should have size [numX]
        for(j=0;j<numY;j++) { // par
            aT[idx2d(i,j,numY)] =    - 0.5*(0.5*globs.myVarX[i][j]*globs.myDxx[i][0]);
            bT[idx2d(i,j,numY)] = dtInv
                            - 0.5*(0.5*globs.myVarX[i][j]*globs.myDxx[i][1]);
            cT[idx2d(i,j,numY)] =    - 0.5*(0.5*globs.myVarX[i][j]*globs.myDxx[i][2]);
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

    //	implicit y
    // ASSUMING tridag is independent.
    // parallelizable directly since all reads and writes are independent.
    // Degree of parallelism: numY*numX.
    // TODO: transpose myDyy and u for coalesced access. **DONE
    // **DONE loop distributed (tridag part), uT mem reused (refilled with u),
    // loop interchanged for mem colesced access, a,b,c matrices reused from
    // prev allocation; used via matrix trandposition.
    // myDyy is still not coalesced, same arguments as previus loop.
/*
    for(i=0;i<numX;i++) { // par
        // parallelizable via loop distribution / array expansion.
        for(j=0;j<numY;j++) { // par  // here a, b, c should have size [numY]
            REAL y = 0.5*0.5*globs.myVarY[idx2d(i,j, globs.myVarYCols)];
            aT[idx2d(i,j,numY)] =       - y*globs.myDyy[idx2d(j,0,
                                                        globs.myDyyCols)];
            bT[idx2d(i,j,numY)] = dtInv - y*globs.myDyy[idx2d(j,1,
                                                        globs.myDyyCols)];
            cT[idx2d(i,j,numY)] =		- y*globs.myDyy[idx2d(j,2,
                                                        globs.myDyyCols)];
        }
    }
*/
    for(i=0;i<numX;i++) { // par
        // parallelizable via loop distribution / array expansion.
        for(j=0;j<numY;j++) { // par  // here a, b, c should have size [numY]
            aT[idx2d(i,j,numY)] =       - 0.5*(0.5*globs.myVarY[i][j]*globs.myDyy[j][0]);
            bT[idx2d(i,j,numY)] = dtInv - 0.5*(0.5*globs.myVarY[i][j]*globs.myDyy[j][1]);
            cT[idx2d(i,j,numY)] =       - 0.5*(0.5*globs.myVarY[i][j]*globs.myDyy[j][2]);
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
                  &globs.myResult[i][0],&yy[0]);
                  //&globs.myResult[idx2d(i,0, globs.myResultCols)],&yy[0]);
    }
}

void   run_GPU( const unsigned int&   outer,
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

    vector<PrivGlobs> globs(outer, PrivGlobs(numX, numY, numT));

    #pragma omp parallel for default(shared) schedule(static) if(outer>8)
    for( unsigned i = 0; i < outer; ++ i ) { //par
            initGrid(s0,alpha,nu,t, numX, numY, numT, globs[i]);
            initOperator(globs[i].myX,globs[i].myDxx);
            initOperator(globs[i].myY,globs[i].myDyy);

            REAL strike = 0.001*i;
            setPayoff(strike, globs[i]);
    }
/*
    // globs array expanded. Init moved to individual parallel loop
    //vector<PrivGlobs> globs(outer, PrivGlobs(numX, numY, numT));
    PrivGlobs *globs = (PrivGlobs*) malloc(outer*sizeof(PrivGlobs));

    #pragma omp parallel for default(shared) schedule(static) if(outer>8)
    for( unsigned i = 0; i < outer; ++ i ) { //par
            globs[i] = PrivGlobs(numX, numY, numT);
            initGrid(s0, alpha,nu,t, numX, numY, numT, globs[i]);
            initOperator(globs[i].myX, globs[i].myXsize, globs[i].myDxx, 
                         globs[i].myDxxCols);
            initOperator(globs[i].myY, globs[i].myYsize, globs[i].myDyy, 
                         globs[i].myDyyCols);
            setPayoff(0.001*i, globs[i]);
    }
*/

    // sequential loop distributed.
    for(int i = numT-2;i>=0;--i){ //seq
        // inner loop parallel on each outer (par) instead of each time step (seq).
        #pragma omp parallel for default(shared) schedule(static) if(outer>8)
        for( unsigned j = 0; j < outer; ++ j ) { //par
            updateParams(i,alpha,beta,nu,globs[j]);
            rollback(i, globs[j]);
        }
    }
    /*
    // parallel assignment of results.
    #pragma omp parallel for default(shared) schedule(static) if(outer>8)
    for( unsigned j = 0; j < outer; ++ j ) { //par
        res[j] = globs[j].myResult[idx2d(globs[j].myXindex,globs[j].myYindex,
                                   globs[j].myResultCols)];
    }
    */

    #pragma omp parallel for default(shared) schedule(static) if(outer>8)
    for( unsigned j = 0; j < outer; ++ j ) { //par
        res[j] = globs[j].myResult[globs[j].myXindex][globs[j].myYindex];
    }
}

//#endif // PROJ_CORE_ORIG
