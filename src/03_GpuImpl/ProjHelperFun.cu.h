#ifndef PROJ_HELPER_FUNS
#define PROJ_HELPER_FUNS

#include <vector>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include "Constants.h"

#include <cuda_runtime.h>

using namespace std;

struct PrivGlobsCuda {

    //  grid
    //vector<REAL>        myX;        // [numX]
    //vector<REAL>        myY;        // [numY]
    //vector<REAL>        myTimeline; // [numT]
    REAL* myX;
    unsigned myXsize;
    REAL* myY;
    unsigned myYsize;
    REAL* myTimeline;
    unsigned myTimelineSize;

    unsigned            myXindex;
    unsigned            myYindex;

    //  variable
    //vector<vector<REAL> > myResult; // [numX][numY]
    REAL* myResult;
    unsigned myResultRows;
    unsigned myResultCols;

    //  coeffs
    //vector<vector<REAL> >   myVarX; // [numX][numY]
    //vector<vector<REAL> >   myVarY; // [numX][numY]
    //vector<REAL> myVarX;
    REAL* myVarX;
    unsigned myVarXRows;
    unsigned myVarXCols;
    //vector<REAL> myVarY;
    REAL* myVarY;
    unsigned myVarYRows;
    unsigned myVarYCols;

    //  operators
    //vector<vector<REAL> >   myDxx;  // [numX][4]
    //vector<vector<REAL> >   myDyy;  // [numY][4]
    //vector<REAL> myDxx;
    REAL* myDxx;
    unsigned myDxxRows;
    unsigned myDxxCols;
    //vector<REAL> myDyy;
    REAL* myDyy;
    unsigned myDyyRows;
    unsigned myDyyCols;

    PrivGlobsCuda( ) {
        printf("Invalid Contructor: need to provide the array sizes! EXITING...!\n");
        exit(0);
    }

    PrivGlobsCuda( const unsigned int& numX,
                   const unsigned int& numY,
                   const unsigned int& numT ) {
        //this->  myX.resize(numX);
        //this->myX = (REAL*) malloc(sizeof(REAL)*numX);
        cudaMalloc((void**) &(this->myX) ,sizeof(REAL)*numX);
        this->myXsize = numX;
        // this->myDxx.resize(numX);
        // for(int k=0; k<numX; k++) {
        //     this->myDxx[k].resize(4);
        // }
        //this->myDxx.resize(numX*4);
        cudaMalloc((void**) &(this->myDxx) ,sizeof(REAL)*numX*4);
        this->myDxxRows = numX;
        this->myDxxCols = 4;

        //this->  myY.resize(numY);
        //this->myY = (REAL*) malloc(sizeof(REAL)*numY);
        cudaMalloc((void**) &(this->myY) ,sizeof(REAL)*numY);
        this->myYsize = numY;
        // this->myDyy.resize(numY);
        // for(int k=0; k<numY; k++) {
        //     this->myDyy[k].resize(4);
        // }
        //this->myDyy.resize(numY*4);
        cudaMalloc((void**) &(this->myDyy) ,sizeof(REAL)*numY*4);
        this->myDyyRows = numY;
        this->myDyyCols = 4;

        //this->myTimeline.resize(numT);
        //this->myTimeline = (REAL*) malloc(sizeof(REAL)*numT);
        cudaMalloc((void**) &(this->myTimeline) ,sizeof(REAL)*numT);
        this->myTimelineSize = numT;

        //this->  myVarX.resize(numX);
        //this->  myVarY.resize(numX);
        //this->myResult.resize(numX);
        //for(unsigned i=0;i<numX;++i) {
            //this->  myVarX[i].resize(numY);
            //this->  myVarY[i].resize(numY);
            //this->myResult[i].resize(numY);
        //}
        //this->myVarX.resize(numX*numY);
        cudaMalloc((void**) &(this->myVarX) ,sizeof(REAL)*numX*numY);
        this->myVarXRows = numX;
        this->myVarXCols = numY;

        //this->myVarY.resize(numX*numY);
        cudaMalloc((void**) &(this->myVarY) ,sizeof(REAL)*numX*numY);
        this->myVarYRows = numX;
        this->myVarYCols = numY;

        //this->myResult.resize(numX*numY);
        cudaMalloc((void**) &(this->myResult) ,sizeof(REAL)*numX*numY);
        this->myResultRows = numX;
        this->myResultCols = numY;

    }
};

struct PrivGlobs {

    //  grid
    vector<REAL>        myX;        // [numX]
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
        this->  myX.resize(numX);
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


void CopyDevicePrivGlobsToHost(PrivGlobsCuda *d_globs, PrivGlobs *globs) {
    size_t rSize = sizeof(REAL);
    cudaMemcpy(globs->myX, d_globs->myX, rSize*d_globs.myXsize, cudaMemcpyDeviceToHost);
    globs.myXsize = d_globs.myXsize;

    cudaMemcpy(globs.myY, d_globs.myY, rSize*d_globs.myYsize, cudaMemcpyDeviceToHost);
    globs.myYsize = d_globs.myYsize;

    cudaMemcpy(globs.myTimeline, d_globs.myTimeline, rSize*d_globs.myTimelineSize, cudaMemcpyDeviceToHost);
    globs.myTimelineSize = d_globs.myTimelineSize;

    globs.myXindex = d_globs.myXindex;
    globs.myYindex = d_globs.myYindex;

    cudaMemcpy(globs.myResult, d_globs.myResult, rSize*d_globs.myResultRows*d_globs.myResultCols, cudaMemcpyDeviceToHost);
    globs.myResultRows = d_globs.myResultRows;
    globs.myResultCols = d_globs.myResultCols;

    cudaMemcpy(globs.myVarX, d_globs.myVarX, rSize*d_globs.myVarXRows*d_globs.myVarXCols, cudaMemcpyDeviceToHost);
    globs.myVarXRows = d_globs.myVarXRows;
    globs.myVarXCols = d_globs.myVarXCols;

    cudaMemcpy(globs.myVarY, d_globs.myVarY, rSize*d_globs.myVarYRows*d_globs.myVarYCols, cudaMemcpyDeviceToHost);
    globs.myVarYRows = d_globs.myVarYRows;
    globs.myVarYCols = d_globs.myVarYCols;

    cudaMemcpy(globs.myDxx, d_globs.myDxx, rSize*d_globs.myDxxRows*d_globs.myDxxCols, cudaMemcpyDeviceToHost);
    globs.myDxxRows = d_globs.myDxxRows;
    globs.myDxxCols = d_globs.myDxxCols;

    cudaMemcpy(globs.myDyy, d_globs.myDyy, rSize*d_globs.myDyyRows*d_globs.myDyyCols, cudaMemcpyDeviceToHost);
    globs.myDyyRows = d_globs.myDyyRows;
    globs.myDyyCols = d_globs.myDyyCols;
}


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

void initOperator(  const vector<REAL>& x,
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
