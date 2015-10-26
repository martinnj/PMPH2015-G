
class MyReal2 {
  public:
    REAL x; REAL y;

      inline MyReal2() {
        x = 0.0; y = 0.0;
    }
      inline MyReal2(const REAL& a, const REAL& b) {
        x = a; y = b;
    }
      inline MyReal2(const MyReal2& i4) {
        x = i4.x; y = i4.y;
    }
    volatile   inline MyReal2& operator=(const MyReal2& i4) volatile {
        x = i4.x; y = i4.y;
        return *this;
    }
      inline MyReal2& operator=(const MyReal2& i4) {
        x = i4.x; y = i4.y;
        return *this;
    }
};

class MyReal4 {
  public:
    REAL x; REAL y; REAL z; REAL w;

      inline MyReal4() {
        x = 0.0; y = 0.0; z = 0.0; w = 0.0;
    }
      inline MyReal4(const REAL& a, const REAL& b, const REAL& c, const REAL& d) {
        x = a; y = b; z = c; w = d;
    }
      inline MyReal4(const MyReal4& i4) {
        x = i4.x; y = i4.y; z = i4.z; w = i4.w;
    }
    volatile   inline MyReal4& operator=(const MyReal4& i4) volatile {
        x = i4.x; y = i4.y; z = i4.z; w = i4.w;
        return *this;
    }
      inline MyReal4& operator=(const MyReal4& i4) {
        x = i4.x; y = i4.y; z = i4.z; w = i4.w;
        return *this;
    }
};

class LinFunComp {
  public:
    typedef MyReal2 BaseType;

    static   inline
    MyReal2 apply(volatile MyReal2& a, volatile MyReal2& b) {
      return MyReal2( b.x + b.y*a.x, a.y*b.y );
    }

    static   inline
    MyReal2 identity() {
      return MyReal2(0.0, 1.0);
    }
};

class MatMult2b2 {
  public:
    typedef MyReal4 BaseType;

    static   inline
    MyReal4 apply(volatile MyReal4& a, volatile MyReal4& b) {
      REAL val = 1.0/(a.x*b.x);
      return MyReal4( (b.x*a.x + b.y*a.z)*val,
                      (b.x*a.y + b.y*a.w)*val,
                      (b.z*a.x + b.w*a.z)*val,
                      (b.z*a.y + b.w*a.w)*val );
    }

    static   inline
    MyReal4 identity() {
      return MyReal4(1.0,  0.0, 0.0, 1.0);
    }
};


template<class OP>
void inplaceScanInc(const int n, typename OP::BaseType* inpres) {
  typename OP::BaseType acc = OP::identity();//inpres[0];
  for(int i=0; i<n; i++) {
    acc = OP::apply(acc,inpres[i]);
    inpres[i] = acc;
  }
}

inline void tridagPar(
    const REAL*   a,   // size [n]
    const REAL*   b,   // size [n]
    const REAL*   c,   // size [n]
    const REAL*   r,   // size [n]
    const int     n,
          REAL*   u,   // size [n]
          REAL*   uu   // size [n] temporary
) {
    //vector<MyReal4> scanres(n); // supposed to also be in shared memory and to reuse the space of mats
    //--------------------------------------------------
    // Recurrence 1: b[i] = b[i] - a[i]*c[i-1]/b[i-1] --
    //   solved by scan with 2x2 matrix mult operator --
    //--------------------------------------------------
    MyReal4* mats = (MyReal4*)malloc(n*sizeof(MyReal4));    // supposed to be in shared memory!
    REAL b0 = b[0];
    for(int i=0; i<n; i++) { //parallel, map-like semantics
        if (i==0) { mats[i].x = 1.0;  mats[i].y = 0.0;          mats[i].z = 0.0; mats[i].w = 1.0; }
        else      { mats[i].x = b[i]; mats[i].y = -a[i]*c[i-1]; mats[i].z = 1.0; mats[i].w = 0.0; }
    }
    inplaceScanInc<MatMult2b2>(n,mats);
    for(int i=0; i<n; i++) { //parallel, map-like semantics
        uu[i] = (mats[i].x*b0 + mats[i].y) / (mats[i].z*b0 + mats[i].w);
    }

    // b -> uu
    //----------------------------------------------------
    // Recurrence 2: y[i] = y[i] - (a[i]/b[i-1])*y[i-1] --
    //   solved by scan with linear func comp operator  --
    //----------------------------------------------------
    MyReal2* lfuns = (MyReal2*)malloc(n*sizeof(MyReal2));
    REAL y0 = r[0];
    for(int i=0; i<n; i++) { //parallel, map-like semantics
        if (i==0) { lfuns[0].x = 0.0;  lfuns[0].y = 1.0;           }
        else      { lfuns[i].x = r[i]; lfuns[i].y = -a[i]/uu[i-1]; }
    }
    inplaceScanInc<LinFunComp>(n,lfuns);
    for(int i=0; i<n; i++) { //parallel, map-like semantics
        u[i] = lfuns[i].x + y0*lfuns[i].y;
    }
    // y -> u
//#if 1
    //----------------------------------------------------
    // Recurrence 3: backward recurrence solved via     --
    //             scan with linear func comp operator  --
    //----------------------------------------------------
    REAL yn = u[n-1]/uu[n-1];
    for(int i=0; i<n; i++) { //parallel, map-like semantics
        int k = n - i - 1;
        if (i==0) { lfuns[0].x = 0.0;  lfuns[0].y = 1.0;           }
        else      { lfuns[i].x = u[k]/uu[k]; lfuns[i].y = -c[k]/uu[k]; }
    }
    inplaceScanInc<LinFunComp>(n,lfuns);
    for(int i=0; i<n; i++) { //parallel, map-like semantics
        int k = n - i - 1;
        u[k] = lfuns[i].x + yn*lfuns[i].y;
    }
//#endif
    free(lfuns);
    free(mats);
}
