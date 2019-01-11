#include "commons.h"
#include "average_gpu.h"

// STD includes
#include <assert.h>
#include <stdio.h>

// CUDA runtime
#include <cuda_runtime.h>

// Utilities and system includes
#include <helper_cuda.h>

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
	int i = (y * (imageW * numChannels)) + (x * numChannels) + channelToExtract;
	int i_mono = (y * (imageW)+x);
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
	int i = (y * (imageW * numChannels)) + (x * numChannels) + channelToMerge;
	int i_mono = (y * (imageW)+x);
	destination[i] = (channelData[i_mono]);
}

/******************************************************************************
* AVERAGE FILTER
******************************************************************************/

__global__ void compute_average(
	unsigned char* h_Dst,
	unsigned char* h_Src,
	int imageW,
	int imageH)
{
	const unsigned int numElements = ((2 * KERNEL_RADIUS) + 1) * ((2 * KERNEL_RADIUS) + 1);

	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int x = blockIdx.x * blockDim.x + threadIdx.x;

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

			const int curPosition = (curY * imageW + curX);
			if (curPosition >= 0 && curPosition < (imageW * imageH)) {
				sum += h_Src[curPosition];
			}
		}
	}
	h_Dst[y * imageW + x] = (unsigned char)(sum / numElements);
}

__global__ void compute_average_shared(
	unsigned char* h_Dst,
	unsigned char* h_Src,
	int imageW,
	int imageH)
{
	const unsigned int numElements = ((2 * KERNEL_RADIUS) + 1) * ((2 * KERNEL_RADIUS) + 1);

	// dichiaro una shared di 32*32 per contenere la cella di 16*16 e i pixel intorno (8 per ogni lato)
	__shared__ unsigned char sharedMemory[32][32];

	// Array per memorizzare le coordinate del blocco da caricare
	int x_p[4];
	int y_p[4];

	// in alto a sinistra
	y_p[0] = blockIdx.y * blockDim.y + threadIdx.y - KERNEL_RADIUS;
	x_p[0] = blockIdx.x * blockDim.x + threadIdx.x - KERNEL_RADIUS;

	// in alto a destra
	y_p[1] = blockIdx.y * blockDim.y + threadIdx.y - KERNEL_RADIUS;
	x_p[1] = blockIdx.x * blockDim.x + threadIdx.x + KERNEL_RADIUS;

	// in basso a sinistra
	y_p[2] = blockIdx.y * blockDim.y + threadIdx.y + KERNEL_RADIUS;
	x_p[2] = blockIdx.x * blockDim.x + threadIdx.x - KERNEL_RADIUS;

	// in basso a destra
	y_p[3] = blockIdx.y * blockDim.y + threadIdx.y + KERNEL_RADIUS;
	x_p[3] = blockIdx.x * blockDim.x + threadIdx.x + KERNEL_RADIUS;

	//// ciclo le coordinate per correggere quelle che sono vicino a uno o più bordi
	//for (int i = 0; i < 4; i++) {
	//	if (y_p[i] < 0)
	//		y_p[i] = 0;
	//	if (x_p[i] < 0)
	//		x_p[i] = 0;
	//	if (y_p[i] > imageH)
	//		y_p[i] = imageH;
	//	if (x_p[i] > imageW)
	//		x_p[i] = imageW;
	//}

	// in alto a sinistra
	sharedMemory[threadIdx.x][threadIdx.y] = (y_p[0] < 0 || x_p[0] < 0) ? 0 : h_Src[x_p[0] + y_p[0] * imageW];
	// in alto a destra
	sharedMemory[threadIdx.x + KERNEL_RADIUS][threadIdx.y] = (y_p[1] < 0 || x_p[1] > imageW) ? 0 : h_Src[x_p[1] + y_p[1] * imageW];
	// in basso a sinistra
	sharedMemory[threadIdx.x - KERNEL_RADIUS][threadIdx.y + KERNEL_RADIUS] = (y_p[2] > imageH || x_p[2] < 0) ? 0 : h_Src[x_p[2] + y_p[2] * imageW];
	// in basso a destra
	sharedMemory[threadIdx.x + KERNEL_RADIUS][threadIdx.y + KERNEL_RADIUS] = (y_p[3] > imageH || x_p[3] > imageW) ? 0 : h_Src[x_p[3] + y_p[3] * imageW];

	__syncthreads();

	int y = threadIdx.y;
	int x = threadIdx.x;

	unsigned int sum = 0;
	for (int kY = -KERNEL_RADIUS; kY <= KERNEL_RADIUS; kY++) {
		for (int kX = -KERNEL_RADIUS; kX <= KERNEL_RADIUS; kX++) {
			sum += sharedMemory[y][x];
		}
	}

	int yImage = blockIdx.y * blockDim.y + threadIdx.y;
	int xImage = blockIdx.x * blockDim.x + threadIdx.x;
	h_Dst[yImage * imageW + xImage] = (unsigned char)(sum / numElements);
}



void average_gpu(
	unsigned char* inputImage,
	unsigned char* outputImage,
	int imageW,
	int imageH,
	int numChannels
) {
	int size = imageW * imageH * 3 * sizeof(unsigned char);

	unsigned char* d_inputImage;
	unsigned char* d_outputImage;

	cudaMalloc((void **)&d_inputImage, size);
	cudaMalloc((void **)&d_outputImage, size);

	unsigned char* d_in_channel;
	unsigned char* d_out_channel;

	cudaMalloc((void **)&d_in_channel, size / 3);
	cudaMalloc((void **)&d_out_channel, size / 3);

	int curChannel;
	dim3 dimBlock(16, 16);
	dim3 dimGrid(imageW / dimBlock.x, imageH / dimBlock.y);

	cudaMemcpy(d_inputImage, inputImage, size, cudaMemcpyHostToDevice);

	for (curChannel = 0; curChannel < numChannels; curChannel++) {
		readChannel << <dimGrid, dimBlock >> > (d_in_channel, d_inputImage, imageW, imageH, curChannel, numChannels);
		compute_average << <dimGrid, dimBlock >> > (d_out_channel, d_in_channel, imageW, imageH);
		writeChannel << <dimGrid, dimBlock >> > (d_outputImage, d_out_channel, imageW, imageH, curChannel, numChannels);
	}

	cudaMemcpy(outputImage, d_outputImage, size, cudaMemcpyDeviceToHost);
	cudaFree(d_in_channel);
	cudaFree(d_out_channel);
}