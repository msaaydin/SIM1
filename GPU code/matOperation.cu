//#include <iostream>

/*

matrix add		            = c = a+b
matrix inner product        = c = a.*b
matrix sub                  = c = a-b
matrix sum                  = s = sum(sum(a)); // reduction shared memory 
matrix multiplication       = c = a*b
matrix inner division       = c = a./b; 
matrix element wise square  = c = a.^2; 
matrix element wise pow     = c = a.^m; 
matrix element wise prod    = c  = a.*m; 
matrix compare and binarise = c = a > const
matrix compare and binarise = c = a < const

*/
// compile and generate ptx file
//nvcc -ptx matOperation.cu --gpu-architecture=compute_61 --gpu-code=sm_61 
        // direk matlan command window dan compile yapabiliriz.
//system('nvcc -ptx matOperation.cu --gpu-architecture=compute_61 --gpu-code=sm_61');
//template <typename T>
        // c = a + b;
#include <math.h>
                // c = a>const G = grater
__global__ void cudaMatrixCompareAndBinarise_G(double *MatA, const double C,const int nx, const int ny)
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
}
 // c = a < const L = lower
__global__ void cudaMatrixCompareAndBinarise_L(double *MatA, const double C,const int nx, const int ny)
{   
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;
    if (ix < nx && iy < ny)
    {
       if (MatA[idx] < C)
            MatA[idx] = 1;
        else
            MatA[idx] = 0;
    }
}
__global__ void sumMatrixGPU(const double *MatA, const double *MatB, double *MatC, const int nx, const int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny)
    {
        MatC[idx] = MatA[idx] + MatB[idx];
    }

}

// c  = a.*b; 
__global__ void pointProductMatrixGPU(const double *MatA, const double *MatB, double *MatC, const int nx, const int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny)
    {
        MatC[idx] = MatA[idx] * MatB[idx];
    }
}
// c  = a.*m; 
__global__ void matrixProductConstGPU(const double *MatA, const double constB, double *MatB, const int nx, const int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny)
    {
        MatB[idx] = MatA[idx] * constB;
    }
}
// c  = a.^m; 
__global__ void matrixCalculatePowGPU(const double *MatA, const double constB, double *MatB, const int nx, const int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny)
    {
        MatB[idx] = pow(MatA[idx],constB);
    }
}
// c  = n*(a.^m);  

__global__ void matrixCalculatePowAndMultGPU(const double *MatA, const double constB,const double constC, double *MatB, const int nx, const int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny)
    {
        MatB[idx] = pow(MatA[idx],constB);
        MatB[idx] = MatB[idx] * constC;
    }
}
// c = a - b;
__global__ void subtractMatrixGPU(const double *MatA, const double *MatB, double *MatC, const int nx, const int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny)
    {
        MatC[idx] = MatA[idx] - MatB[idx];
    }
}
// s = sum(sum(a));
__global__ void reduceSum( double* g_odata,  const double* g_idata,  const double len) {
	extern __shared__  double sdata[];

	// each thread loads one element from global to shared mem
	// Do the first stage of the reduction on the global-to-shared load step
	// This reduces the previous inefficiency of having half of the threads being
	//  inactive on the first for-loop iteration below (previous first step of reduction)
	// Previously, only less than or equal to 512 out of 1024 threads in a block are active.
	// Now, all 512 threads in a block are active from the start
	 unsigned int tid = threadIdx.x;
	 unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

	sdata[tid] = 0;

	if (i < len)
	{
		sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
	}

	__syncthreads();

	// do reduction in shared mem
	// this loop now starts with s = 512 / 2 = 256
	for ( unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

	if (tid < 32)
	{
		sdata[tid] += sdata[tid + 32];
		sdata[tid] += sdata[tid + 16];
		sdata[tid] += sdata[tid + 8];
		sdata[tid] += sdata[tid + 4];
		sdata[tid] += sdata[tid + 2];
		sdata[tid] += sdata[tid + 1];
	}

	// write result for this block to global mem
	if (tid == 0)
		g_odata[blockIdx.x] = sdata[0];
}

// reduceSumV2
__global__ void reduceSmemV2(const double *g_idata, double *g_odata, unsigned int n)
{
    // static shared memory
    __shared__ double smem[128];

    // set thread ID
    unsigned int tid = threadIdx.x;

    // global index, 4 blocks of input data processed at a time
    unsigned int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;

    // unrolling 4 blocks
    double tmpSum = 0;

    // boundary check
    if (idx + 4 * blockDim.x <= n)
    {
        double a1 = g_idata[idx];
        double a2 = g_idata[idx + blockDim.x];
        double a3 = g_idata[idx + 2 * blockDim.x];
        double a4 = g_idata[idx + 3 * blockDim.x];
        tmpSum = a1 + a2 + a3 + a4;
    }

    smem[tid] = tmpSum;
    __syncthreads();

    // in-place reduction in shared memory
    if (blockDim.x >= 1024 && tid < 512) smem[tid] += smem[tid + 512];

    __syncthreads();

    if (blockDim.x >= 512 && tid < 256)  smem[tid] += smem[tid + 256];

    __syncthreads();

    if (blockDim.x >= 256 && tid < 128)  smem[tid] += smem[tid + 128];

    __syncthreads();

    if (blockDim.x >= 128 && tid < 64)   smem[tid] += smem[tid + 64];

    __syncthreads();

    // unrolling warp
    if (tid < 32)
    {
        volatile double *vsmem = smem;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid +  8];
        vsmem[tid] += vsmem[tid +  4];
        vsmem[tid] += vsmem[tid +  2];
        vsmem[tid] += vsmem[tid +  1];
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = smem[0];
}
// c = a*b matrix multiplication 

__global__ void MatrixMulKernel(double* Md, double* Nd, double* Pd, int Width)
{
     __shared__ double Mds[32][32];
     __shared__ double Nds[32][32];
     int bx = blockIdx.x; int by = blockIdx.y;
     int tx = threadIdx.x; int ty = threadIdx.y;
    // Identify the row and column of the Pd element to work on
     int Row = by * 32 + ty;
     int Col = bx * 32 + tx;
     double Pvalue = 0;
    // Loop over the Md and Nd tiles required to compute the Pd element
     for (int m = 0; m < Width/32; ++m) {
    // Collaborative loading of Md and Nd tiles into shared memory
     Mds[ty][tx] = Md[Row*Width + (m*32 + tx)];
     Nds[ty][tx] = Nd[Col + (m*32 + ty)*Width];
     __syncthreads();
     for (int k = 0; k < 32; ++k)
       Pvalue += Mds[ty][k] * Nds[k][tx];
     __syncthreads();
     }
     Pd[Row*Width+Col] = Pvalue;
}

// c  = a./b; 
__global__ void pointDivisionMatrixGPU(const double *MatA, const double *MatB, double *MatC, const int nx, const int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny)
    {
        MatC[idx] = MatA[idx] / MatB[idx];
    }
}
// c  = a.^2  
__global__ void pointSquareMatrixGPU(const double *MatA, double *MatC, const int nx, const int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny)
    {
        MatC[idx] = MatA[idx] * MatA[idx];
    }
}