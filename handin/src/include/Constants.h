#ifndef CONSTANTS
#define CONSTANTS

#if (WITH_FLOATS==0)
    typedef double REAL;
#else
    typedef float  REAL;
#endif

#define TVAL 8 //8*8*8 = 512 =< 1024

#endif // CONSTANTS
