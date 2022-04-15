//#include <cuComplex.h>
 
//system('nvcc -ptx cudaComplexMatrixOperation.cu --gpu-architecture=compute_61 --gpu-code=sm_61');
/* Double precision */
typedef double2 cuDoubleComplex;
#include <math.h>
__device__  double cuCreal (cuDoubleComplex x) 
{ 
    return x.x; 
}
__device__ double cuCimag (cuDoubleComplex x) 
{ 
    return x.y; 
}

__device__ cuDoubleComplex make_cuDoubleComplex(double r, double i)
{
    cuDoubleComplex res;
    res.x = r;
    res.y = i;
    return res;
}
// c = a+b a ve b birer complex number
__device__ cuDoubleComplex cuCadd(cuDoubleComplex x, cuDoubleComplex y)
{
    return make_cuDoubleComplex (cuCreal(x) + cuCreal(y), 
                                 cuCimag(x) + cuCimag(y));
}
// c = conj(a) a bir complex number
__device__ cuDoubleComplex cuConj(cuDoubleComplex x)
{
    return make_cuDoubleComplex (cuCreal(x), -cuCimag(x));
}
// c = a-b a ve b birer complex number
__device__ cuDoubleComplex cuCsub(cuDoubleComplex x, cuDoubleComplex y)
{
    return make_cuDoubleComplex (cuCreal(x) - cuCreal(y), 
                                 cuCimag(x) - cuCimag(y));
}

/* This implementation could suffer from intermediate overflow even though
 * the final result would be in range. However, various implementations do
 * not guard against this (presumably to avoid losing performance), so we 
 * don't do it either to stay competitive.
 */
 // c = a*b a ve b birer complex number
__device__ cuDoubleComplex cuCmul(cuDoubleComplex x,cuDoubleComplex y)
{
    cuDoubleComplex prod;
    prod = make_cuDoubleComplex ((cuCreal(x) * cuCreal(y)) - 
                                 (cuCimag(x) * cuCimag(y)),
                                 (cuCreal(x) * cuCimag(y)) + 
                                 (cuCimag(x) * cuCreal(y)));
    return prod;
}

/* This implementation guards against intermediate underflow and overflow
 * by scaling. Such guarded implementations are usually the default for
 * complex library implementations, with some also offering an unguarded,
 * faster version.
 */
 // c = a/b a ve b birer complex number
__device__ cuDoubleComplex cuCdiv(cuDoubleComplex x, cuDoubleComplex y)
{
    cuDoubleComplex quot;
    double s = (fabs(cuCreal(y))) + (fabs(cuCimag(y)));
    double oos = 1.0 / s;
    double ars = cuCreal(x) * oos;
    double ais = cuCimag(x) * oos;
    double brs = cuCreal(y) * oos;
    double bis = cuCimag(y) * oos;
    s = (brs * brs) + (bis * bis);
    oos = 1.0 / s;
    quot = make_cuDoubleComplex (((ars * brs) + (ais * bis)) * oos,
                                 ((ais * brs) - (ars * bis)) * oos);
    return quot;
}

/* This implementation guards against intermediate underflow and overflow
 * by scaling. Otherwise we would lose half the exponent range. There are
 * various ways of doing guarded computation. For now chose the simplest
 * and fastest solution, however this may suffer from inaccuracies if sqrt
 * and division are not IEEE compliant.
 */
 // c = abs(a) a bir complex number
__device__ double cuCabs (cuDoubleComplex x)
{
    double a = cuCreal(x);
    double b = cuCimag(x);
    double v, w, t;
    a = fabs(a);
    b = fabs(b);
    if (a > b) {
        v = a;
        w = b; 
    } else {
        v = b;
        w = a;
    }
    t = w / v;
    t = 1.0 + t * t;
    t = v * sqrt(t);
    if ((v == 0.0) || 
        (v > 1.79769313486231570e+308) || (w > 1.79769313486231570e+308)) {
        t = v + w;
    }
    return t;
}
// global kernel function call all math operations..
// c = a/b
__global__ void cudaComplexDiv(const double2 *MatA, const double2 *MatB, double2 *MatC,const int nx, const int ny)
{   
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;
    if (ix < nx && iy < ny)
    {
        MatC[idx] = cuCdiv(MatA[idx], MatB[idx]);
    }
	//*MatC = cuCdiv(MatA[0],MatB[0]);    
} 

// c = a+b
__global__ void cudaComplexAdd(const double2 *MatA, const double2 *MatB, double2 *MatC,const int nx, const int ny)
{   
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;
    if (ix < nx && iy < ny)
    {
        MatC[idx] = cuCadd(MatA[idx], MatB[idx]);
    }
	//*MatC = cuCdiv(MatA[0],MatB[0]);    
}

// c = a-b
__global__ void cudaComplexSub(const double2 *MatA, const double2 *MatB, double2 *MatC,const int nx, const int ny)
{   
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;
    if (ix < nx && iy < ny)
    {
        MatC[idx] = cuCsub(MatA[idx], MatB[idx]);
    }
	//*MatC = cuCdiv(MatA[0],MatB[0]);    
}
      
// c = a*b
__global__ void cudaComplexMul(const double2 *MatA, const double2 *MatB, double2 *MatC,const int nx, const int ny)
{   
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;
    if (ix < nx && iy < ny)
    {
        MatC[idx] = cuCmul(MatA[idx], MatB[idx]);
    }
	//*MatC = cuCdiv(MatA[0],MatB[0]);    
}
// c = conj(a)
__global__ void cudaComplexConj(double2 *MatA,const int nx, const int ny)
{   
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;
    if (ix < nx && iy < ny)
    {
        MatA[idx] = cuConj(MatA[idx]);
    }
	//*MatC = cuCdiv(MatA[0],MatB[0]);    
}
// c = abs(a)
__global__ void cudaComplexAbs(const double2 *MatA, double *MatC,const int nx, const int ny)
{   
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;
    if (ix < nx && iy < ny)
    {
        MatC[idx] = cuCabs(MatA[idx]);
    }
	//*MatC = cuCdiv(MatA[0],MatB[0]);    
}

// c = a>const
__global__ void cudaMakeBinaryMatrix(const double *MatA, double *MatC, const double C,const int nx, const int ny)
{   
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;
    if (ix < nx && iy < ny)
    {
       if (MatA[idx] > C)
            MatC[idx] = 1;
        else
            MatC[idx] = 0;
    }
	//*MatC = cuCdiv(MatA[0],MatB[0]);    
}
// c = a > const
__global__ void cudaMakeBinaryMatrix2(double *MatA, const double C,const int nx, const int ny)
{   
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;
    if (ix < nx && iy < ny)
    {
        if (MatA[idx] > C)
            MatA[idx] = 1;
        else
            MatA[idx] = 0;
            
    }
	//*MatC = cuCdiv(MatA[0],MatB[0]);    
}