#ifndef PROJ_HELPER_FUNS
#define PROJ_HELPER_FUNS

#include <vector>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include "Constants.h"

using namespace std;


struct PrivGlobs {

    //	grid
    vector<REAL>        myX;        // [numX]
    vector<REAL>        myY;        // [numY]
    vector<REAL>        myTimeline; // [numT]
    unsigned            myXindex;
    unsigned            myYindex;

    //	variable
    //vector<vector<REAL> > myResult; // [numX][numY]
    vector<REAL> myResult;
    unsigned myResultRows;
    unsigned myResultCols;

    //	coeffs
    //vector<vector<REAL> >   myVarX; // [numX][numY]
    //vector<vector<REAL> >   myVarY; // [numX][numY]
    vector<REAL> myVarX;
    unsigned myVarXRows;
    unsigned myVarXCols;
    vector<REAL> myVarY;
    unsigned myVarYRows;
    unsigned myVarYCols;

    //	operators
    //vector<vector<REAL> >   myDxx;  // [numX][4]
    //vector<vector<REAL> >   myDyy;  // [numY][4]
    vector<REAL> myDxx;
    unsigned myDxxRows;
    unsigned myDxxCols;
    vector<REAL> myDyy;
    unsigned myDyyRows;
    unsigned myDyyCols;

    PrivGlobs( ) {
        printf("Invalid Contructor: need to provide the array sizes! EXITING...!\n");
        exit(0);
    }

    PrivGlobs(  const unsigned int& numX,
                const unsigned int& numY,
                const unsigned int& numT ) {
        this->  myX.resize(numX);
        // this->myDxx.resize(numX);
        // for(int k=0; k<numX; k++) {
        //     this->myDxx[k].resize(4);
        // }
        this->myDxx.resize(numX*4);
        this->myDxxRows = numX;
        this->myDxxCols = 4;

        this->  myY.resize(numY);
        // this->myDyy.resize(numY);
        // for(int k=0; k<numY; k++) {
        //     this->myDyy[k].resize(4);
        // }
        this->myDyy.resize(numY*4);
        this->myDyyRows = numY;
        this->myDyyCols = 4;

        this->myTimeline.resize(numT);

        //this->  myVarX.resize(numX);
        //this->  myVarY.resize(numX);
        //this->myResult.resize(numX);
        //for(unsigned i=0;i<numX;++i) {
            //this->  myVarX[i].resize(numY);
            //this->  myVarY[i].resize(numY);
            //this->myResult[i].resize(numY);
        //}
        this->myVarX.resize(numX*numY);
        this->myVarXRows = numX;
        this->myVarXCols = numY;

        this->myVarY.resize(numX*numY);
        this->myVarYRows = numX;
        this->myVarYCols = numY;

        this->myResult.resize(numX*numY);
        this->myResultRows = numX;
        this->myResultCols = numY;

    }
}; //__attribute__ ((aligned (128)));


void initGrid(  const REAL s0, const REAL alpha, const REAL nu,const REAL t,
                const unsigned numX, const unsigned numY, const unsigned numT, PrivGlobs& globs
            );

//void initOperator(  const vector<REAL>& x,
//                    vector<vector<REAL> >& Dxx
//                 );
void initOperator(  const vector<REAL>& x,
                    vector<REAL> &Dxx, unsigned DxxCols
                 );


void updateParams(const unsigned g, const REAL alpha, const REAL beta, const REAL nu, PrivGlobs& globs);

void setPayoff(const REAL strike, PrivGlobs& globs );

void tridag(
    const vector<REAL>&   a,   // size [n]
    const vector<REAL>&   b,   // size [n]
    const vector<REAL>&   c,   // size [n]
    const vector<REAL>&   r,   // size [n]
    const int             n,
          vector<REAL>&   u,   // size [n]
          vector<REAL>&   uu   // size [n] temporary
);

void rollback( const unsigned g, PrivGlobs& globs );

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
            );

void run_GPU(
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
            );


//void transpose(REAL** MIn, REAL** MOut, int M, int N);
void transposeVect(vector<REAL> MIn,
               vector<REAL> *MOut,
               unsigned int M,
               unsigned int N);

void transpose(REAL* A, REAL** B, int M, int N);

unsigned idx2d(unsigned row, unsigned col, unsigned width);

#endif // PROJ_HELPER_FUNS
