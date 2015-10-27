#ifndef PROJ_HELPER_FUNS
#define PROJ_HELPER_FUNS

#include <vector>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include "Constants.h"

using namespace std;

struct PrivGlobs {

<<<<<<< HEAD
    //  grid
    vector<REAL>        myX;        // [numX]
=======
    //	grid
    //vector<REAL>        myX;        // [numX]
    REAL*    myX;
    unsigned myXsize;


>>>>>>> c117f84a56405f135ed90e1d34e02a6bf2beffd6
    vector<REAL>        myY;        // [numY]
    vector<REAL>        myTimeline; // [numT]
    unsigned            myXindex;
    unsigned            myYindex;

    //  variable
    vector<vector<REAL> > myResult; // [numX][numY]

    //  coeffs
    vector<vector<REAL> >   myVarX; // [numX][numY]
    vector<vector<REAL> >   myVarY; // [numX][numY]

    //  operators
    vector<vector<REAL> >   myDxx;  // [numX][4]
    vector<vector<REAL> >   myDyy;  // [numY][4]

    PrivGlobs( ) {
        printf("Invalid Contructor: need to provide the array sizes! EXITING...!\n");
        exit(0);
    }

    PrivGlobs(  const unsigned int& numX,
                const unsigned int& numY,
                const unsigned int& numT ) {
        //this->  myX.resize(numX);
        this->myX = (REAL*) malloc(sizeof(REAL)*numX);
        this->myXsize = numX;

        this->myDxx.resize(numX);
        for(int k=0; k<numX; k++) {
            this->myDxx[k].resize(4);
        }

        this->  myY.resize(numY);
        this->myDyy.resize(numY);
        for(int k=0; k<numY; k++) {
            this->myDyy[k].resize(4);
        }

        this->myTimeline.resize(numT);

        this->  myVarX.resize(numX);
        this->  myVarY.resize(numX);
        this->myResult.resize(numX);
        for(unsigned i=0;i<numX;++i) {
            this->  myVarX[i].resize(numY);
            this->  myVarY[i].resize(numY);
            this->myResult[i].resize(numY);
        }

    }
} __attribute__ ((aligned (128)));


/*
struct PrivGlobs {

    //  grid
    REAL*    myX;        // [numX]
    unsigned myXsize;

    REAL*    myY;        // [numY]
    unsigned myYsize;

    REAL*    myTimeline; // [numT]
    unsigned myTimelineSize;

    unsigned myXindex;
    unsigned myYindex;

    //  variable
    //vector<vector<REAL> > myResult; // [numX][numY]
    REAL* myResult;
    unsigned myResultRows;
    unsigned myResultCols;

    //  coeffs
    //vector<vector<REAL> >   myVarX; // [numX][numY]
    REAL* myVarX;
    unsigned myVarXRows;
    unsigned myVarXCols;

    //vector<vector<REAL> >   myVarY; // [numX][numY]
    REAL* myVarY;
    unsigned myVarYRows;
    unsigned myVarYCols;

    //  operators
    //vector<vector<REAL> >   myDxx;  // [numX][4]
    REAL* myDxx;
    unsigned myDxxRows;
    unsigned myDxxCols;

    //vector<vector<REAL> >   myDyy;  // [numY][4]
    REAL* myDyy;
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

        //this->myDxx.resize(numX);
        this->myDxx = (REAL*) malloc(sizeof(REAL)*(numX*4));
        this->myDxxRows = numX;
        this->myDxxCols = 4;

        this->myY = (REAL*) malloc(sizeof(REAL)*numY);
        this->myYsize = numY;

        //this->myDyy.resize(numY);
        this->myDyy = (REAL*) malloc(sizeof(REAL)*(numY*4));
        this->myDyyRows = numY;
        this->myDyyCols = 4;

        this->myTimeline = (REAL*) malloc(sizeof(REAL)*numT);
        this->myTimelineSize = numT;

        this->myResult = (REAL*) malloc(sizeof(REAL)*(numX*numY));
        this->myResultRows = numX;
        this->myResultCols = numY;

        this->myVarX = (REAL*) malloc(sizeof(REAL)*(numX*numY));
        this->myVarXRows = numX;
        this->myVarXCols = numY;

        this->myVarY = (REAL*) malloc(sizeof(REAL)*(numX*numY));
        this->myVarYRows = numX;
        this->myVarYCols = numY;

    }
} __attribute__ ((aligned (128)));
*/

void initGrid(  const REAL s0, const REAL alpha, const REAL nu,const REAL t,
                const unsigned numX, const unsigned numY, const unsigned numT, PrivGlobs& globs
            );

void initOperator(  const REAL* x, unsigned xsize,
                    vector<vector<REAL> >& Dxx
                 );

void updateParams(const unsigned g, const REAL alpha, const REAL beta, const REAL nu, PrivGlobs& globs);

void setPayoff(const REAL strike, PrivGlobs& globs );

void rollback( const unsigned g, PrivGlobs& globs );

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
void transposeVect(vector<vector<REAL> > MIn,
               vector<vector<REAL> >* MOut,
               unsigned int M,
               unsigned int N);

void transpose(REAL* A, REAL** B, int M, int N);


unsigned int idx2d(int row, int col, int width);



#endif // PROJ_HELPER_FUNS
