

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include <thrust/complex.h>

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <Windows.h>

#if __STDC_VERSION__ >= 199901L
//using a C99 compiler
#include <complex.h>
typedef float _Complex float_complex;
#else
typedef struct
{
	float re, im;
} float_complex;
#endif
#if __STDC_VERSION__ >= 199901L
//creal, cimag already defined in complex.h

inline complex_float make_complex_float(float real, float imag)
{
	return real + imag * I;
}
#else
#define creal(z) ((z).re)
#define cimag(z) ((z).im)

extern const float_complex complex_i; //put in a translation unit somewhere
#define I complex_i
inline float_complex make_complex_float(float real, float imag)
{
	float_complex z = { real, imag };
	return z;
}
#endif
#if __STDC_VERSION__ >= 199901L
#define add_complex(a, b) ((a)+(b))
//similarly for other operations
#else //not C99
inline float_complex add_complex(float_complex a, float_complex b)
{
	float_complex z = { a.re + b.re, a.im + b.im };
	return z;
}
inline float_complex subtract_complex(float_complex a, float_complex b)
{
	float_complex z = { a.re - b.re, a.im - b.im };
	return z;
}
inline float_complex multiply_complex(float_complex a, float_complex b)
{
	float_complex z = { a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
	return z;
}
inline float_complex divide_complex(float_complex a, float_complex b)
{
	float_complex z = { (a.re * b.re + a.im * b.im) / (b.re * b.re + b.im * b.im), (a.im * b.re - a.re * b.im) / (b.re * b.re + b.im * b.im) };
	return z;
}
inline float_complex conjugate(float_complex a)
{
	float_complex z = { a.re, a.im * -1 };
	return z;
}

inline float_complex multiply(float_complex a, float b)
{
	float_complex z = { a.re * b, a.im * b };
	return z;
}
#endif

// *********************************************************************
// ALL HELPER FUNCTIONS
// *********************************************************************
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

void buildCorrelationTestData();

void runSingleThreadedCorrelation(float* inputI, float* inputQ, int* oversamplePRN, float* localCarrierPhaseIn, float dopplerFrqTimesSampleRate, float* localCarrierPhaseout, float* outputI, float* outputQ);

void runMultiThreadedCorrelation(float* inputI, float* inputQ, int* oversamplePRN, float* localCarrierPhaseIn, float dopplerFrqTimesSampleRate, float* localCarrierPhaseout, float* outputI, float* outputQ, int numberOfThreads);

void runGPGPUCorrelation();

float randomFloat(float a);

int randomInt(int a);
// *********************************************************************
// VARIABLES
// *********************************************************************

// INPUTS DO NOT MUTATE!
int arrayLen = 10000;

float *g_InputI, *g_InputQ;

int *overSamplePRNCode;

float localCarrierPhaseIn;

float dopplerFrqTimesSampleRate;

// Outputs of the function:
float singleCPUI, singleCPUQ;
float singleCPULocalCarrierPhaseOut;

float multiCPUI, multiCPUQ;
float multiCPULocalCarrierPhaseOut;

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

#define MaxSamples 10
#define MAX_THREADS 8
//#define VERBOSE_MODE
#define TABLE_MODE

int main()
{
#ifdef TABLE_MODE
	printf("[# Of Samples]\t[Single Threaded CPU Time (uS)]\t");
	for (int i = 1; i < MAX_THREADS + 1; i++)
		printf("[Multi-Threaded (%d) CPU Time (us)]\t", i);
	printf("\n");
#endif
	int samples[] = { 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 15000, 20000 };
	for (int s = 0; s < MaxSamples; s++)
	{
#ifdef TABLE_MODE
		printf("%d\t", samples[s]);
#endif
		arrayLen = samples[s];
		LARGE_INTEGER StartingTime, EndingTime, ElapsedMicroseconds;
		LARGE_INTEGER Frequency;

		// This is a low performance Timer.
		int tick = time(NULL);
		buildCorrelationTestData();
#ifdef VERBOSE_MODE
		printf("Time to build sample data[%d](s): %d\n",samples[s], time(NULL) - tick);
#endif

		// Starting High performance Timer
		QueryPerformanceFrequency(&Frequency);
		QueryPerformanceCounter(&StartingTime);
		// Running activity.
		runSingleThreadedCorrelation(g_InputI, g_InputQ, overSamplePRNCode, &localCarrierPhaseIn, dopplerFrqTimesSampleRate, &singleCPULocalCarrierPhaseOut, &singleCPUI, &singleCPUQ);

		// Ending High performance Timer.
		QueryPerformanceCounter(&EndingTime);

		// Calculating Elapsed Time.
		ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;

		//
		// We now have the elapsed number of ticks, along with the
		// number of ticks-per-second. We use these values
		// to convert to the number of elapsed microseconds.
		// To guard against loss-of-precision, we convert
		// to microseconds *before* dividing by ticks-per-second.
		//

		ElapsedMicroseconds.QuadPart *= 1000000;
		ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;
#ifdef VERBOSE_MODE
		printf("Single Threaded CPU Correlation Output: Z = {%+.2f,%+.2f}, Local carrier phase out: %f\nTime taken (uS): %d\n", singleCPUI, singleCPUQ, singleCPULocalCarrierPhaseOut, ElapsedMicroseconds.QuadPart);
#endif
#ifdef TABLE_MODE
		printf("%d\t", ElapsedMicroseconds.QuadPart);
#endif
		// Now we need to do it for multithreaded CPU
		for (int numThreads = 1; numThreads < MAX_THREADS + 1; numThreads++)
		{
			// Starting High performance Timer
			QueryPerformanceFrequency(&Frequency);
			QueryPerformanceCounter(&StartingTime);
			// Running Activity
			runMultiThreadedCorrelation(g_InputI, g_InputQ, overSamplePRNCode, &localCarrierPhaseIn, dopplerFrqTimesSampleRate, &multiCPULocalCarrierPhaseOut, &multiCPUI, &multiCPUQ, numThreads);
			// Ending High performance Timer.
			QueryPerformanceCounter(&EndingTime);

			// Calculating Elapsed Time.
			ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;

			ElapsedMicroseconds.QuadPart *= 1000000;
			ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;
#ifdef VERBOSE_MODE
			printf("Multi-Threaded(%02d) CPU Correlation Output: Z = {%+.2f,%+.2f}, Local carrier phase out: %f\nTime taken (uS): %d\n", numThreads, multiCPUI, multiCPUQ, multiCPULocalCarrierPhaseOut, ElapsedMicroseconds.QuadPart);
#endif
#ifdef TABLE_MODE
			printf("%d\t", ElapsedMicroseconds.QuadPart);
#endif
		}

		// Finally the GPU.

		// Present all the data in some nice way.
		printf("\n");
	}

    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}

// --------------------------------------------------------------------------
// DATA GENERATION
// --------------------------------------------------------------------------

#define INPUT_ARRAY_LENGTH arrayLen
#define DOPPLER_RATE 12.5
#define SAMPLE_RATE 25000000
void buildCorrelationTestData()
{
	// Allocate the Arrays.
	g_InputI = (float*)malloc(sizeof(float) * INPUT_ARRAY_LENGTH);
	g_InputQ = (float*)malloc(sizeof(float) * INPUT_ARRAY_LENGTH);
	overSamplePRNCode = (int*)malloc(sizeof(int) * INPUT_ARRAY_LENGTH);

	srand(time(NULL));

	// Run the Generation Function to build the arrays.
	for (int i = 0; i < INPUT_ARRAY_LENGTH; i++)
	{
		g_InputI[i] = randomFloat(2*3.14) - 3.14;
		g_InputQ[i] = randomFloat(2 * 3.14) - 3.14;
		overSamplePRNCode[i] = randomInt(2) > 0 ? -1:1;
	}
#ifdef VERBOSE_MODE

	// Print the Array for testing purposes
	for (int i = 0; i < INPUT_ARRAY_LENGTH; i++)
	{
		printf("[%07d]{%+.4f, %+.4f}[OSPRN: %+d] \n", i, g_InputI[i], g_InputQ[i], overSamplePRNCode[i]);
	}

	float dpl = randomFloat(DOPPLER_RATE);
	dopplerFrqTimesSampleRate = (1.0 / SAMPLE_RATE) * dpl * 2 * 3.1415;
	printf("Doppler Frq: %.1f\nDoppler * 2pi * 1/Sample Rates: %f\n", dpl, dopplerFrqTimesSampleRate);
#endif
}

float randomFloat(float a)
{
	return (float)rand() / (float)(RAND_MAX / a);
}

int randomInt(int a)
{
	return rand() % a;
}

// --------------------------------------------------------------------------
// SINGLE THREADED EXAMPLE
// --------------------------------------------------------------------------

void runSingleThreadedCorrelation(float* inputI, float* inputQ, int* oversamplePRN, float* localCarrierPhaseIn, float dopplerFrqTimesSampleRate, float* localCarrierPhaseout, float* outputI, float* outputQ)
{	
	*localCarrierPhaseout = *localCarrierPhaseIn + INPUT_ARRAY_LENGTH *  dopplerFrqTimesSampleRate;
	float_complex total = make_complex_float(0.0f, 0.0f);
	float_complex z1;
	float_complex z2;
	for (int i = 0; i < INPUT_ARRAY_LENGTH; i++)
	{
		z1 = make_complex_float(cosf(i * dopplerFrqTimesSampleRate + *localCarrierPhaseIn), sinf(i * dopplerFrqTimesSampleRate + *localCarrierPhaseIn));
		z2 = make_complex_float(inputI[i], inputQ[i]);
		z2 = multiply_complex(conjugate(z1), z2);
		z2 = multiply(z2, oversamplePRN[i]);
		total = add_complex(z2, total);
	}

	*outputI = total.re;
	*outputQ = total.im;
}

// --------------------------------------------------------------------------
// MULTITHREADED EXAMPLE
// --------------------------------------------------------------------------

DWORD WINAPI threadCorrelation(LPVOID lpParam);
void ErrorHandler(LPTSTR lpszFunction);

// Sample custom data structure for threads to use.
// This is passed by void pointer so it can be any data type
// that can be passed using a single void pointer (LPVOID).
typedef struct threadData {
	int start;
	int stop;
	float_complex *result;
	float* inputI;
	float* inputQ;
	int* prn;
	float* localCarrierPhaseIn;
	float* dopplerFrqTimesSampleRate;
} MYDATA, *PMYDATA;

void runMultiThreadedCorrelation(float* inputI, float* inputQ, int* oversamplePRN, float* localCarrierPhaseIn, float dopplerFrqTimesSampleRate, float* localCarrierPhaseout, float* outputI, float* outputQ, int numberOfThreads)
{
	*localCarrierPhaseout = *localCarrierPhaseIn + INPUT_ARRAY_LENGTH *  dopplerFrqTimesSampleRate;
	float_complex *subTotals = (float_complex*)malloc(sizeof(float_complex) * numberOfThreads);

	PMYDATA *pDataArray = (PMYDATA*)malloc(sizeof(MYDATA) * numberOfThreads);
	DWORD   *dwThreadIdArray = (DWORD*)malloc(sizeof(DWORD) * numberOfThreads);
	HANDLE  *hThreadArray = (HANDLE*)malloc(sizeof(HANDLE) * numberOfThreads);

	for (int i = 0; i < numberOfThreads; i++)
	{
		// Allocate memory for thread data.

		pDataArray[i] = (PMYDATA)HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY,
			sizeof(MYDATA));

		if (pDataArray[i] == NULL)
		{
			// If the array allocation fails, the system is out of memory
			// so there is no point in trying to print an error message.
			// Just terminate execution.
			ExitProcess(2);
		}

		pDataArray[i]->inputI = inputI;
		pDataArray[i]->inputQ = inputQ;
		pDataArray[i]->prn = oversamplePRN;
		pDataArray[i]->localCarrierPhaseIn = localCarrierPhaseIn;
		pDataArray[i]->dopplerFrqTimesSampleRate = &dopplerFrqTimesSampleRate;
		pDataArray[i]->result = &subTotals[i];


		// Generate unique data for each thread to work with.
		if (i != numberOfThreads - 1)
		{
			pDataArray[i]->start = i *(INPUT_ARRAY_LENGTH / numberOfThreads);
			pDataArray[i]->stop = pDataArray[i]->start + (INPUT_ARRAY_LENGTH / numberOfThreads);
		}
		else
		{
			pDataArray[i]->start = i *(INPUT_ARRAY_LENGTH / numberOfThreads);
			pDataArray[i]->stop = INPUT_ARRAY_LENGTH;
		}

		// Create the thread to begin execution on its own.

		hThreadArray[i] = CreateThread(
			NULL,                   // default security attributes
			0,                      // use default stack size  
			threadCorrelation,       // thread function name
			pDataArray[i],          // argument to thread function 
			0,                      // use default creation flags 
			&dwThreadIdArray[i]);   // returns the thread identifier 


		// Check the return value for success.
		// If CreateThread fails, terminate execution. 
		// This will automatically clean up threads and memory. 

		if (hThreadArray[i] == NULL)
		{
			ErrorHandler(TEXT("CreateThread"));
			ExitProcess(3);
		}

	}


	// Wait until all threads have terminated.

	WaitForMultipleObjects(numberOfThreads, hThreadArray, TRUE, INFINITE);


	// Sum up all the subUnits
	float_complex total = make_complex_float(0.0f, 0.0f);
	for (int i = 0; i < numberOfThreads; i++)
	{
		total = add_complex(subTotals[i], total);
	}

	*outputI = total.re;
	*outputQ = total.im;

	// Close all thread handles and free memory allocations.

	for (int i = 0; i<numberOfThreads; i++)
	{
		CloseHandle(hThreadArray[i]);
		if (pDataArray[i] != NULL)
		{
			HeapFree(GetProcessHeap(), 0, pDataArray[i]);
			pDataArray[i] = NULL;    // Ensure address is not reused.
		}
	}
}

DWORD WINAPI threadCorrelation(LPVOID lpParam)
{
	PMYDATA pDataArray;

	// Cast the parameter to the correct data type.
	// The pointer is known to be valid because 
	// it was checked for NULL before the thread was created.

	pDataArray = (PMYDATA)lpParam;

	// Run the correlation sequence.
	float_complex total = make_complex_float(0.0f, 0.0f);
	float_complex z1;
	float_complex z2;
	for (int i = pDataArray->start; i < pDataArray->stop; i++)
	{
		z1 = make_complex_float(cosf(i * (*pDataArray->dopplerFrqTimesSampleRate) + (*pDataArray->localCarrierPhaseIn)), sinf(i * (*pDataArray->dopplerFrqTimesSampleRate) + (*pDataArray->localCarrierPhaseIn)));
		z2 = make_complex_float(pDataArray->inputI[i], pDataArray->inputQ[i]);
		z2 = multiply_complex(conjugate(z1), z2);
		z2 = multiply(z2, pDataArray->prn[i]);
		total = add_complex(z2, total);
	}

	*pDataArray->result = make_complex_float(total.re, total.im);

	return 0;
}



void ErrorHandler(LPTSTR lpszFunction)
{
	// Retrieve the system error message for the last-error code.

	LPVOID lpMsgBuf;
	LPVOID lpDisplayBuf;
	DWORD dw = GetLastError();

	FormatMessage(
		FORMAT_MESSAGE_ALLOCATE_BUFFER |
		FORMAT_MESSAGE_FROM_SYSTEM |
		FORMAT_MESSAGE_IGNORE_INSERTS,
		NULL,
		dw,
		MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
		(LPTSTR)&lpMsgBuf,
		0, NULL);

	// Display the error message.

	// Free error-handling buffer allocations.

	LocalFree(lpMsgBuf);
	LocalFree(lpDisplayBuf);
}

// --------------------------------------------------------------------------
// GPGPU CORRELATION FUNCTIONS
//--------------------------------------------------------------------------
__global__ void correlation_MAP_KERNEL(const float* inputI, const float* inputQ, const float* oversamplePRN, const float* localCarrierPhaseIn, const float dopplerFrqTimesSampleRate, float* outputI, float* outputQ)
{
	// Get index so we can determine which element to operate on.
	int i = threadIdx.x;

	// Create the data set from the elements.
	// SLOW Operation, reading from global memory.
	thrust::complex<float> z2 = thrust::complex<float>(inputI[i], inputQ[i]);
	float _dopplerFrqTimesSampleRate = dopplerFrqTimesSampleRate;								// Moving these to local memory for the thread will give better performance.
	float _localCarrierPhaseIn = *localCarrierPhaseIn;

	// Faster creation of this float because it's all local memory operations. 
	thrust::complex<float> z1 = thrust::complex<float>(cosf(_localCarrierPhaseIn + (i * _dopplerFrqTimesSampleRate)), sinf(_localCarrierPhaseIn + (i * _dopplerFrqTimesSampleRate)));

	// Do the math.
	z2 *= thrust::conj(z1);
	z2 *= oversamplePRN[i];
	
	// Write the output
	// SLOW OPERATION, writing to global memory.
	outputI[i] = z2.real();
	outputQ[i] = z2.imag();
}

__global__ void correlation_REDUCE_KERNEL(const float* inputI, const float* inputQ, float* outputI, float* outputQ)
{
	extern __shared__ thrust::complex<float> sdata[];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	sdata[tid] = thrust::complex<float>(inputI[i], inputQ[i]);
	__syncthreads();

	// do reduction in shared memory
	for (unsigned int s = 1; s < blockDim.x; s *= 2)
	{
		if (tid % (2 * s) == 0)
		{
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

	// write back to global memory
	if (tid == 0)
	{
		outputI[blockIdx.x] = sdata[0].real();
		outputI[blockIdx.x] = sdata[0].imag();
	}
}