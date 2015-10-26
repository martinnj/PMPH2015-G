#include "ProjHelperFun.h"
#include "Constants.h"

#include "TridagPar.h"

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

// inline void tridag(
//     vector<REAL>&         a,   // size [n]
//     const vector<REAL>&   b,   // size [n]
//     const vector<REAL>&   c,   // size [n]
//     const vector<REAL>&   r,   // size [n]
//     const int             n,
//           vector<REAL>&   u,   // size [n]
//           vector<REAL>&   uu   // size [n] temporary
// ) {
//     int    i, offset;
//     REAL   beta;
//
//     u[0]  = r[0];
//     uu[0] = b[0];
//
//     // Can be parallelized by rewriting the loop into a series of scans on u and uu.
//
//     /*
//     for(i=1; i<n; i++) { // seq
//         beta  = a[i] / uu[i-1];
//         uu[i] = b[i] - beta*c[i-1]; // uu[i] = b[i] - (a[i] / uu[i-1])*c[i-1] = b[i] - a[i]*c[i-1] / uu[i-1]
//         u[i]  = r[i] - beta*u[i-1]; // u[i]  = r[i] - (a[i] / uu[i-1])*u[i-1]
//     }
//     */
//
//     /*
//     uu = - a*c
//     uu' = b*uu''[-1]
//
//     scanInc (/) head(uu)^2 uu
//     */
//     uu = vector<REAL>(b); //wrong
//     for(i=1; i<n; i++) {
//         uu[i] = uu[i] - a[i]*c[i-1];
//     }
//     scanIncDiv(uu[0]*uu[0], &uu); //id, array
//
//     /*
//     u = r
//     uu' = [0] ++ u //do same for uu.....
//     u = u - a / uu'
//     - (a[i] / uu[i-1])*u[i-1]
//
//     scanInc (*) 1 u
//     */
//     for(i=1; i<n; i++) {
//
//     }
//
// #if 0
//     // X) this is a backward recurrence
//     u[n-1] = u[n-1] / uu[n-1];
//
//     // Can be parallelized by rewriting the loop into a series of scans.
//     for(i=n-2; i>=0; i--) { // seq
//         u[i] = (u[i] - c[i]*u[i+1]) / uu[i];
//     }
// #else
//     // Hint: X) can be written smth like (once you make a non-constant)
//     for(i=0; i<n; i++) // TODO: par
//         a[i] = u[n-1-i]; // a = reverse u
//
//     a[0] = a[0] / uu[n-1];
//
//     // tridag can be re-written in terms of scans with 2x2 matrix multiplication and
//     // linear function composition, as we shall discuss in class.
//     for(i=1; i<n; i++) // TODO: Make it a scan bro.
//         a[i] = (a[i] - c[n-1-i]*a[i-1]) / uu[n-1-i];
//         //a[i] = (a[i] - c[n-1-i]*a[i-1]) / uu[n-1-i];
//
//     for(i=0; i<n; i++) // TODO: par
//         u[i] = a[n-1-i]; // u = reverse a
// #endif
// }


void
rollback( const unsigned g, PrivGlobs& globs ) {
    unsigned numX = globs.myX.size(),
             numY = globs.myY.size();

    unsigned numZ = max(numX,numY);

    unsigned i, j;

    REAL dtInv = 1.0/(globs.myTimeline[g+1]-globs.myTimeline[g]);

    vector<vector<REAL> > u(numY, vector<REAL>(numX));   // [numY][numX]
    vector<vector<REAL> > uT(numX, vector<REAL>(numY));   // [numX][numY]
    vector<vector<REAL> > v(numX, vector<REAL>(numY));   // [numX][numY]
    //vector<REAL> a(numZ), b(numZ), c(numZ), y(numZ);     // [max(numX,numY)]
    //vector<REAL> y(numZ);
    vector<vector<REAL> > y(numX, vector<REAL>(numZ));   // [numX][numZ]

    vector<REAL> yy(numZ);  // temporary used in tridag  // [max(numX,numY)]

    //	explicit x
    // parallelizable directly since all reads and writes are independent.
    // Degree of parallelism: numX*numY.
    // TODO: Examine how tiling/shared memory can be used on globs (.myResult).

    // Reads are coalosced but writes are not.
    // TODO: Coalesced access via matrix transposition of u/uT. **DONE
    for(i=0;i<numX;i++) { //par
        for(j=0;j<numY;j++) { //par
            //TODO: This can be combined in the tridag kernel, in shared mem.
            uT[i][j] = dtInv*globs.myResult[i][j];

            if(i > 0) {
              uT[i][j] += 0.5*( 0.5*globs.myVarX[i][j]*globs.myDxx[i][0] )
                            * globs.myResult[i-1][j];
            }
            uT[i][j]  +=  0.5*( 0.5*globs.myVarX[i][j]*globs.myDxx[i][1] )
                            * globs.myResult[i][j];
            if(i < numX-1) {
              uT[i][j] += 0.5*( 0.5*globs.myVarX[i][j]*globs.myDxx[i][2] )
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
    for(i=0;i<numX;i++) { //par
        for(j=0;j<numY;j++) { //par
            //TODO: This can be combined in the tridag kernel too, as parameters.
            v[i][j] = 0.0;

            if(j > 0) {
              v[i][j] +=  ( 0.5*globs.myVarY[i][j]*globs.myDyy[j][0] )
                         *  globs.myResult[i][j-1];
            }
            v[i][j]  +=   ( 0.5*globs.myVarY[i][j]*globs.myDyy[j][1] )
                         *  globs.myResult[i][j];
            if(j < numY-1) {
              v[i][j] +=  ( 0.5*globs.myVarY[i][j]*globs.myDyy[j][2] )
                         *  globs.myResult[i][j+1];
            }
            uT[i][j] += v[i][j];
        }
    }
    transpose(uT, &u, numY, numX);


    vector<vector<REAL> > a(numY, vector<REAL>(numZ)), b(numY, vector<REAL>(numZ)), c(numY, vector<REAL>(numZ));     // [max(numX,numY)]
    vector<vector<REAL> > aT(numZ, vector<REAL>(numY)), bT(numZ, vector<REAL>(numY)), cT(numZ, vector<REAL>(numY));
    //	implicit x
    // ASSUMING tridag is independent.
    // parallelizable directly since all reads and writes are independent.
    // Degree of parallelism: numY*numX.
    // TODO: MyDxx and myVarX is not coalesced. **DONE
    /*
    for(j=0;j<numY;j++) { // par
        // parallelizable via loop distribution / array expansion.
        for(i=0;i<numX;i++) {  // par // here a, b,c should have size [numX]
            a[i] =		 - 0.5*(0.5*globs.myVarX[i][j]*globs.myDxx[i][0]);
            b[i] = dtInv - 0.5*(0.5*globs.myVarX[i][j]*globs.myDxx[i][1]);
            c[i] =		 - 0.5*(0.5*globs.myVarX[i][j]*globs.myDxx[i][2]);
        }
        // here yy should have size [numX]
        tridag(a,b,c,u[j],numX,u[j],yy);
    } */

    // parallelizable via loop distribution / array expansion.
    for(i=0;i<numX;i++) {  // par // here a, b,c should have size [numX]
        for(j=0;j<numY;j++) { // par
            aT[i][j] =		 - 0.5*(0.5*globs.myVarX[i][j]*globs.myDxx[i][0]);
            bT[i][j] = dtInv - 0.5*(0.5*globs.myVarX[i][j]*globs.myDxx[i][1]);
            cT[i][j] =		 - 0.5*(0.5*globs.myVarX[i][j]*globs.myDxx[i][2]);
        }

    }
    transpose(aT, &a, numY, numZ);
    transpose(bT, &b, numY, numZ);
    transpose(cT, &c, numY, numZ);

    for(j=0;j<numY;j++) { // par
        // here yy should have size [numX]
        tridagPar(&a[j][0],&b[j][0],&c[j][0],&u[j][0],numX,&u[j][0],&yy[0]);
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
    for(i=0;i<numX;i++) { // par
        // parallelizable via loop distribution / array expansion.
        for(j=0;j<numY;j++) { // par  // here a, b, c should have size [numY]
            aT[i][j] =		 - 0.5*(0.5*globs.myVarY[i][j]*globs.myDyy[j][0]);
            bT[i][j] = dtInv - 0.5*(0.5*globs.myVarY[i][j]*globs.myDyy[j][1]);
            cT[i][j] =		 - 0.5*(0.5*globs.myVarY[i][j]*globs.myDyy[j][2]);
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
            y[i][j] = dtInv * uT[i][j] - 0.5*v[i][j];
        }
    }


    for(i=0;i<numX;i++) { // par
        // here yy should have size [numX]
        tridagPar(&aT[i][0],&bT[i][0],&cT[i][0],&y[i][0],numY,&globs.myResult[i][0],&yy[0]);
    }
}

REAL   value(   PrivGlobs    globs,
                const REAL s0,
                const REAL strike,
                const REAL t,
                const REAL alpha,
                const REAL nu,
                const REAL beta,
                const unsigned int numX,
                const unsigned int numY,
                const unsigned int numT
) {
    initGrid(s0,alpha,nu,t, numX, numY, numT, globs);
    initOperator(globs.myX,globs.myDxx);
    initOperator(globs.myY,globs.myDyy);

    setPayoff(strike, globs);
    // globs is global and cannot be privatized thus this loop cannot be
    // parallelized yet.
    // If updateParams and rollback is independent on i and globs, loop can be
    // parallelized by privatization of initGrid, initOperator and setPayoff calls.
    // If they write indepedently to globs, privatization is not needed.
    for(int i = globs.myTimeline.size()-2;i>=0;--i) // seq, based on num_T indirectly.
    {
        updateParams(i,alpha,beta,nu,globs); // Updates all values in globs.myVarX and globs.myVarY. So not independent.
        rollback(i, globs);
    }

    return globs.myResult[globs.myXindex][globs.myYindex];
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
    vector<PrivGlobs> globs(outer, PrivGlobs(numX, numY, numT));

    #pragma omp parallel for default(shared) schedule(static) if(outer>8)
    for( unsigned i = 0; i < outer; ++ i ) { //par
            initGrid(s0,alpha,nu,t, numX, numY, numT, globs[i]);
            initOperator(globs[i].myX,globs[i].myDxx);
            initOperator(globs[i].myY,globs[i].myDyy);

            REAL strike = 0.001*i;
            setPayoff(strike, globs[i]);
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
        res[j] = globs[j].myResult[globs[j].myXindex][globs[j].myYindex];
    }
}

//#endif // PROJ_CORE_ORIG
