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
    REAL* myX;        // [numX]
    unsigned myXsize; 
    REAL* myY;        // [numY]
    unsigned myYsize;
    REAL* myTimeline; // [numT]
    unsigned myTimelineSize;

    unsigned            myXindex;
    unsigned            myYindex;

    //	variable
    REAL* myResult;
    unsigned myResultRows;
    unsigned myResultCols;

    //	coeffs
    REAL* myVarX;        // [numX][numY]
    unsigned myVarXRows;
    unsigned myVarXCols;
    REAL* myVarY;        // [numX][numY]
    unsigned myVarYRows;
    unsigned myVarYCols;

    //	operators
    REAL* myDxx;        // [numX][4]
    unsigned myDxxRows;
    unsigned myDxxCols;
    REAL* myDyy;        // [numY][4]
    unsigned myDyyRows;
    unsigned myDyyCols;

    PrivGlobs( ) {
        printf("Invalid Contructor: need to provide the array sizes! EXITING...!\n");
        exit(0);
    }

    PrivGlobs(  const unsigned int& numX,
                const unsigned int& numY,
                const unsigned int& numT ) {

        this->myX = (REAL*) malloc(sizeof(REAL)*numX);
        this->myXsize = numX;

        this->myDxx = (REAL*) malloc(sizeof(REAL)*numX*4);
        this->myDxxRows = numX;
        this->myDxxCols = 4;

        this->myY = (REAL*) malloc(sizeof(REAL)*numY);
        this->myYsize = numY;

        this->myDyy = (REAL*) malloc(sizeof(REAL)*numY*4);
        this->myDyyRows = numY;
        this->myDyyCols = 4;

        this->myTimeline = (REAL*) malloc(sizeof(REAL)*numT);
        this->myTimelineSize = numT;

        this->myVarX = (REAL*) malloc(sizeof(REAL)*numX*numY);
        this->myVarXRows = numX;
        this->myVarXCols = numY;

        this->myVarY = (REAL*) malloc(sizeof(REAL)*numX*numY);
        this->myVarYRows = numX;
        this->myVarYCols = numY;

        this->myResult = (REAL*) malloc(sizeof(REAL)*numX*numY);
        this->myResultRows = numX;
        this->myResultCols = numY;

    }
};


void initGrid(  const REAL s0, const REAL alpha, const REAL nu,const REAL t,
                const unsigned numX, const unsigned numY, const unsigned numT, PrivGlobs& globs
            );

//void initOperator(  const vector<REAL>& x,
//                    vector<vector<REAL> >& Dxx
//                 );
void initOperator(  const REAL *x, unsigned xsize,
                    REAL* &Dxx, unsigned DxxCols
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
