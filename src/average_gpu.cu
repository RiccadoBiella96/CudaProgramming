#include "commons.h"
#include "average_gpu.h"

// STD includes
#include <assert.h>
#include <stdio.h>

// CUDA runtime
#include <cuda_runtime.h>

// Utilities and system includes
#include <helper_cuda.h>

// Con questo visual studio riconosce blockId etc...
#include "device_launch_parameters.h"

static // Print device properties
void printDevProp(cudaDeviceProp devProp)
{
	printf("Major revision number:         %d\n", devProp.major);
	printf("Minor revision number:         %d\n", devProp.minor);
	printf("Name:                          %s\n", devProp.name);
	printf("Total global memory:           %zu\n", devProp.totalGlobalMem);
	printf("Total shared memory per block: %zu\n", devProp.sharedMemPerBlock);
	printf("Total registers per block:     %d\n", devProp.regsPerBlock);
	printf("Warp size:                     %d\n", devProp.warpSize);
	printf("Maximum memory pitch:          %zu\n", devProp.memPitch);
	printf("Maximum threads per block:     %d\n", devProp.maxThreadsPerBlock);
	for (int i = 0; i < 3; ++i)
		printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
	for (int i = 0; i < 3; ++i)
		printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
	printf("Clock rate:                    %d\n", devProp.clockRate);
	printf("Total constant memory:         %zu\n", devProp.totalConstMem);
	printf("Texture alignment:             %zu\n", devProp.textureAlignment);
	printf("Concurrent copy and execution: %s\n", (devProp.deviceOverlap ? "Yes" : "No"));
	printf("Number of multiprocessors:     %d\n", devProp.multiProcessorCount);
	printf("Kernel execution timeout:      %s\n", (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
	return;
}

void print_gpuInfo() {
	int rtVersion = 0;
	printf("*********************************************************************************************\n");
	checkCudaErrors(cudaRuntimeGetVersion(&rtVersion));
	printf("CUDA Runtime Version = %d\n", rtVersion);
	int driverVersion = 0;
	checkCudaErrors(cudaDriverGetVersion(&driverVersion));
	printf("CUDA Driver Version  = %d\n", rtVersion);

	int numDevices = 0;
	checkCudaErrors(cudaGetDeviceCount(&numDevices));
	printf("Devices found        = %d\n", numDevices);

	for (int i = 0; i < numDevices; i++) {
		cudaDeviceProp properties;
		checkCudaErrors(cudaGetDeviceProperties(&properties, i));
		printDevProp(properties);
	}
	printf("*********************************************************************************************\n");
}


/******************************************************************************
* UTILITY FUNCTIONS
******************************************************************************/

__global__ static void readChannel(
	unsigned char * channelData,
	unsigned char *source,
	int imageW,
	int imageH,
	int channelToExtract,
	int numChannels)
{
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	int i = (y * (imageW * numChannels)) + x*numChannels + channelToExtract;
	int i_mono = (y * (imageW)) + x +channelToExtract;
	
	channelData[i_mono] = source[i];
}

__global__ static void writeChannel(
	unsigned char* destination,
	unsigned char* channelData,
	int imageW,
	int imageH,
	int channelToMerge,
	int numChannels)
{
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	int i = (y * (imageW * numChannels)) + x*numChannels + channelToMerge;
	int i_mono = (y * (imageW)) + x + channelToMerge;

	destination[i] = (channelData[i_mono]);
}

/******************************************************************************
* AVERAGE FILTER
******************************************************************************/

__global__ static void compute_average(
	unsigned char* d_Dst,
	unsigned char* d_Src,
	int imageW,
	int imageH)
{
	__shared__ unsigned char sharedMemory[32*32];

	int y = 
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	//calcolo le posizioni

	int y1 = blockIdx.y * blockDim.y + threadIdx.y - KERNEL_RADIUS;
	int x1 = blockIdx.x * blockDim.x + threadIdx.x - KERNEL_RADIUS;
	
	if (x1 <0 || x1 > imageW || y1 < 0 || y1 > imageH) {
		sharedMemory[y1 * 32 + x1] = 0;
	}
	else {
		sharedMemory[y1 * 32 + x1] = d_Src[y1 * 32 + x1];
	}

	int y2 = blockIdx.y * blockDim.y + threadIdx.y - KERNEL_RADIUS;
	int x2 = blockIdx.x * blockDim.x + threadIdx.x + KERNEL_RADIUS;

	if (x2 <0 || x2 > imageW || y2 < 0 || y2 > imageH) {
		sharedMemory[y2 * 32 + x2] = 0;
	}
	else {
		sharedMemory[y2 * 32 + x2] = d_Src[y2 * 32 + x2];
	}

	int y3 = blockIdx.y * blockDim.y + threadIdx.y + KERNEL_RADIUS;
	int x3 = blockIdx.x * blockDim.x + threadIdx.x - KERNEL_RADIUS;

	if (x3 <0 || x3 > imageW || y3 < 0 || y3 > imageH) {
		sharedMemory[y3 * 32 + x3] = 0;
	}
	else {
		sharedMemory[y3 * 32 + x3] = d_Src[y3 * 32 + x3];
	}

	int y4 = blockIdx.y * blockDim.y + threadIdx.y + KERNEL_RADIUS;
	int x4 = blockIdx.x * blockDim.x + threadIdx.x + KERNEL_RADIUS;

	if (x4 <0 || x4 > imageW || y4 < 0 || y4 > imageH) {
		sharedMemory[y4 * 32 + x4] = 0;
	}
	else {
		sharedMemory[y4 * 32 + x4] = d_Src[y4 * 32 + x4];
	}

	__syncthreads();

	const unsigned int numElements = ((2 * KERNEL_RADIUS) + 1) * ((2 * KERNEL_RADIUS) + 1);

	unsigned int sum = 0;
	for (int kY = -KERNEL_RADIUS; kY <= KERNEL_RADIUS; kY++) {
		const int curY = y + kY;
		if (curY < 0 || curY > imageH) {
			continue;
		}

		for (int kX = -KERNEL_RADIUS; kX <= KERNEL_RADIUS; kX++) {
			const int curX = x + kX;
			if (curX < 0 || curX > imageW) {
				continue;
			}

			const int curPosition = (curY * 32 + curX);
			if (curPosition >= 0 && curPosition < (imageW * imageH)) {
				sum += sharedMemory[curPosition];
			}
		}
	}
	d_Dst[y * imageW + x] = (unsigned char)(sum / numElements);
	//d_Dst[i_mono] = d_Src[i_mono];
}

void average_gpu(
	unsigned char* host_inputImage,
	unsigned char* host_outputImage,
	int imageW,
	int imageH,
	int numChannels
) {

	unsigned char* device_inputImage;
	unsigned char* device_outputImage;

	unsigned char*  in_channel;
	unsigned char*  out_channel;

	cudaMalloc(&in_channel, imageW * imageH * sizeof(unsigned char));
	cudaMalloc(&out_channel, imageW * imageH * sizeof(unsigned char));

	int curChannel;
	dim3 dimBlok(16,16);
	dim3 dimGrid((imageW / dimBlok.x), (imageH / dimBlok.y));

	// FIXME: copy from CPU to GPU (inputImage --> GPU)


	cudaMalloc(&device_inputImage, imageW * imageH * 3 * sizeof(unsigned char));
	cudaMalloc(&device_outputImage, imageW * imageH * 3 * sizeof(unsigned char));

	cudaMemcpy(device_inputImage, host_inputImage, imageW * imageH * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);

	// average GPU

	for (curChannel = 0; curChannel < numChannels; curChannel++) {
		readChannel <<< dimGrid, dimBlok >> > (in_channel, device_inputImage, imageW, imageH, curChannel, numChannels);
		compute_average << < dimGrid, dimBlok >> > (out_channel, in_channel, imageW, imageH);
		writeChannel <<< dimGrid, dimBlok >> > (device_outputImage, out_channel, imageW, imageH, curChannel, numChannels);
	}

	// FIXME: Copy back to CPU (GPU --> outputImage)

	cudaMemcpy(host_outputImage, device_outputImage, imageW * imageH * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
}

