#include <cuda_runtime.h>
#include <iostream>
#include <ctime>

#include "EBMP/EasyBMP.h"

//Russian characters aren't displayed.Comments in English, sorry...

// 2D float texture
texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;

using namespace std;

#define checkCudaErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

void saveImage(float* image, int height, int width, bool method) {
	BMP Output;
	Output.SetSize(width, height);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			RGBApixel pixel;
			pixel.Red = image[i * width + j];
			pixel.Green = image[i * width + j];
			pixel.Blue = image[i * width + j];
			Output.SetPixel(j, i, pixel);
		}
	}
	if (method)
		Output.WriteToFile("GPUoutAngelina.bmp");
	else
		Output.WriteToFile("CPUoutAngelina.bmp");

}

//Unit square
//we choose a coordinate system in which the four points where f is known are(0, 0), (1, 0), (0, 1), and (1, 1)
// wiki

void medianFilterCPU(float* image, float* resault, int height, int width)
{
	for (int j = 0; j < height-1; j++) {
		for (int i = 0; i < width-1; i++) {


			float f01 = image[j * width + i];
			float f11 = image[j * width + i + 1];
			float f00 = image[j * width + width + i];
			float f10 = image[j * width + width + i + 1];


			float n11 = f01 * 0.5 + f11 * 0.5;
			float n00 = f00 * 0.5 + f01 * 0.5;
			float n10 = f00 * 0.5 * 0.5 + f10 * 0.5 * 0.5 + f01 * 0.5 * 0.5 + f11 * 0.5 * 0.5;

			resault[j* width * 4 + i * 2] = f01;
			resault[j * width * 4 + i * 2 + 1] = n11;
			resault[j * width * 4 + i * 2 + width * 2] = n00;
			resault[j * width * 4 + i * 2 + width * 2 + 1] = n10;

		}
	}

}


// A good example to demonstrate the difference between a CPU and a GPU is because the algorithms are almost ideal.



//Unit square
//we choose a coordinate system in which the four points where f is known are(0, 0), (1, 0), (0, 1), and (1, 1)
// wiki

__global__ void myFilter(float* output, int imageWidth, int imageHeight) {

	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	float f01 = tex2D(texRef, col, row);
	float f11 = tex2D(texRef, col+1, row);
	float f00 = tex2D(texRef, col, row+1);
	float f10 = tex2D(texRef, col + 1, row+1);

	float n11 = f01*0.5+f11*0.5;
	float n00 = f00*0.5+f01*0.5;
	float n10 = f00 * 0.5 * 0.5 + f10 * 0.5*0.5 + f01 * 0.5 * 0.5 + f11 * 0.5 * 0.5;

	output[row * imageWidth * 4 + col*2] = f01;
	output[row * imageWidth * 4 + col * 2 + 1] = n11;
	output[row * imageWidth * 4 + col * 2 + imageWidth * 2] = n00;
	output[row * imageWidth * 4 + col * 2 + imageWidth * 2 + 1] = n10;

}


int main(void)
{
	int nIter = 100;
	BMP Image;
	Image.ReadFromFile("angelina4.bmp");
	int height = Image.TellHeight();
	int width = Image.TellWidth();

	float* imageArray = (float*)calloc(height * width, sizeof(float));
	float* outputCPU = (float*)calloc(height * width*4, sizeof(float));
	float* outputGPU = (float*)calloc(height * width*4, sizeof(float));
	float* outputDevice;


	for (int j = 0; j < Image.TellHeight(); j++) {
		for (int i = 0; i < Image.TellWidth(); i++) {
			imageArray[j * width + i] = Image(i, j)->Red;
		}
	}

	unsigned int start_time = clock();

	for (int j = 0; j < nIter; j++) {
		medianFilterCPU(imageArray, outputCPU, height, width);
	}

	unsigned int elapsedTime = clock() - start_time;
	float msecPerMatrixMulCpu = elapsedTime / nIter;

	cout << "CPU time: " << msecPerMatrixMulCpu << endl;

	int device_count = 0;
	cudaGetDeviceCount(&device_count);

	if (device_count == 0)
		cout << "Sorry! You dont have CudaDevice" << endl;
	else {
		cout << "CudaDevice found! Device count: " << device_count << endl;

		// Allocate CUDA array in device memory

		//Returns a channel descriptor with format f and number of bits of each component x, y, z, and w
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
		cudaArray* cu_arr;

		checkCudaErrors(cudaMallocArray(&cu_arr, &channelDesc, width, height));
		checkCudaErrors(cudaMemcpyToArray(cu_arr, 0, 0, imageArray, height * width * sizeof(float), cudaMemcpyHostToDevice));		// set texture parameters
		texRef.addressMode[0] = cudaAddressModeClamp;
		texRef.addressMode[1] = cudaAddressModeClamp;
		texRef.filterMode = cudaFilterModePoint;


		// Bind the array to the texture
		cudaBindTextureToArray(texRef, cu_arr, channelDesc);

		checkCudaErrors(cudaMalloc(&outputDevice, height * width * 4* sizeof(float)));

		dim3 threadsPerBlock(16, 16);
		dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
			(height + threadsPerBlock.y - 1) / threadsPerBlock.y);

		cudaEvent_t start;
		cudaEvent_t stop;
		checkCudaErrors(cudaEventCreate(&start));
		checkCudaErrors(cudaEventCreate(&stop));

		// start record
		checkCudaErrors(cudaEventRecord(start, 0));

		for (int j = 0; j < nIter; j++) {
			myFilter << <blocksPerGrid, threadsPerBlock >> > (outputDevice, width, height);
		}

		// stop record
		checkCudaErrors(cudaEventRecord(stop, 0));

		// wait end of event
		checkCudaErrors(cudaEventSynchronize(stop));

		float msecTotal = 0.0f;
		checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

		float msecPerMatrixMul = msecTotal / nIter;

		cout << "GPU time: " << msecPerMatrixMul << endl;

		cudaDeviceSynchronize();

		checkCudaErrors(cudaMemcpy(outputGPU, outputDevice, height * width * 4 * sizeof(float), cudaMemcpyDeviceToHost));

		cudaDeviceSynchronize();

		saveImage(outputGPU, height*2, width*2, true);
		saveImage(outputCPU, height*2, width*2, false);

		checkCudaErrors(cudaFreeArray(cu_arr));
		checkCudaErrors(cudaFree(outputDevice));
	}
	return 0;
}

