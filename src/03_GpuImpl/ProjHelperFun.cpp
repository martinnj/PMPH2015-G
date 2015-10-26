#include "ProjHelperFun.h"

/**************************/
/**** HELPER FUNCTIONS ****/
/**************************/

/**
 * Fills in:
 *   globs.myTimeline  of size [0..numT-1]
 *   globs.myX         of size [0..numX-1]
 *   globs.myY         of size [0..numY-1]
 * and also sets
 *   globs.myXindex and globs.myYindex (both scalars)
 */
void initGrid(  const REAL s0, const REAL alpha, const REAL nu,const REAL t,
                const unsigned numX, const unsigned numY, const unsigned numT, 
                PrivGlobs& globs
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

/**
 * Fills in:
 *    Dx  [0..n-1][0..3] and
 *    Dxx [0..n-1][0..3]
 * Based on the values of x,
 * Where x's size is n.
 */
void initOperator(        const REAL* x,    //vector<REAL>& x,
                      unsigned  xSize,
                          REAL* Dxx,        //vector<vector<REAL> >& Dxx
                      unsigned  DxxCols
) {
	const unsigned n = xSize;

	REAL dxl, dxu;

	//	lower boundary
	dxl		 =  0.0;
	dxu		 =  x[1] - x[0];

	Dxx[idx2d(0,0, DxxCols)] =  0.0;
	Dxx[idx2d(0,1, DxxCols)] =  0.0;
	Dxx[idx2d(0,2, DxxCols)] =  0.0;
    Dxx[idx2d(0,3, DxxCols)] =  0.0;

	//	standard case
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

	//	upper boundary
	dxl		   =  x[n-1] - x[n-2];
	dxu		   =  0.0;

	Dxx[idx2d(n-1,0,DxxCols)] = 0.0;
	Dxx[idx2d(n-1,1,DxxCols)] = 0.0;
	Dxx[idx2d(n-1,2,DxxCols)] = 0.0;
    Dxx[idx2d(n-1,3,DxxCols)] = 0.0;
}

void transpose(REAL* A, REAL** B, int M, int N) {
    for(int i = 0 ; i < M ; i++) {
        for(int j = 0 ; j < N ; j++) {
            (*B)[j*M+i] = A[i*N+j];
        }
    }
}

// row = row idx
// col = col idx
// width = number of columns in the matrix
// ex: A[row,col] = A[idx2d(row, col, a.cols)]
unsigned int idx2d(int row, int col, int width) {
    return row * width + col;
}
