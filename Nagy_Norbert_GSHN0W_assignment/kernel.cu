
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "lodepng.h"
#include "lodepng.cpp"
#include <stdio.h>
#include <iostream>
#include <fstream> 
#include <chrono>
using namespace std::chrono;

#define THREAD_COUNT_PER_DIM 30
#define INPUT_IMG "input_img.png"
#define PI 3.14159265358979323846

typedef unsigned char byte;

struct Image {
	Image(byte* pixels = nullptr, unsigned int width = 0, unsigned int height = 0) : pixels(pixels), width(width), height(height) {
	};
	byte* pixels;
	unsigned int width;
	unsigned int height;
};

// Image loading converts to grayscale by default, because colors are not needed in this case.
Image loadPngImage(char* filename) {
	unsigned int width, height;
	byte* rgbImage;
	unsigned error = lodepng_decode_file(&rgbImage, &width, &height, filename, LCT_RGBA, 8);
	if (error) {
		printf("Error loading image: %u: %s\n", error, lodepng_error_text(error));
		exit(2);
	}

	byte* grayscale = new byte[width * height];
	byte* img = rgbImage;
	for (int i = 0; i < width * height; ++i) {
		int r = *img++; // red
		int g = *img++; // green
		int b = *img++; // blue
		int a = *img++; // opacity
		grayscale[i] = 0.3 * r + 0.6 * g + 0.1 * b + 0.5;
	}
	free(rgbImage);

	return Image(grayscale, width, height);
}


void writePngImage(char* filename, std::string appendText, Image outputImage) {
	std::string newName = filename;
	newName = newName.substr(0, newName.rfind("."));
	newName.append("_").append(appendText).append(".png");
	unsigned error = lodepng_encode_file(newName.c_str(), outputImage.pixels, outputImage.width, outputImage.height, LCT_GREY, 8);
	if (error) {
		printf("Error writing image: %u: %s\n", error, lodepng_error_text(error));
		exit(3);
	}
}


// Sobel X
// -1  0  1
// -2  0  2
// -1  0  1

// Sobel Y
// -1 -2 -1
//  0  0  0
//  1  2  1

// arr[x][y] == arr[y*width + x]

void sobelEdgeDetectionCpu(const byte* original, byte* destination, const unsigned int width, const unsigned int height) {
	for (int y = 1; y < height - 1; y++) {
		for (int x = 1; x < width - 1; x++) {
			int dx = (-1 * original[(y - 1) * width + (x - 1)]) + (-2 * original[y * width + (x - 1)]) + (-1 * original[(y + 1) * width + (x - 1)]) +
				(original[(y - 1) * width + (x + 1)]) + (2 * original[y * width + (x + 1)]) + (original[(y + 1) * width + (x + 1)]);
			int dy = (original[(y - 1) * width + (x - 1)]) + (2 * original[(y - 1) * width + x]) + (original[(y - 1) * width + (x + 1)]) +
				(-1 * original[(y + 1) * width + (x - 1)]) + (-2 * original[(y + 1) * width + x]) + (-1 * original[(y + 1) * width + (x + 1)]);
			destination[y * width + x] = sqrt((dx * dx) + (dy * dy));
		}
	}
}

__global__ void sobelEdgeDetectionGpu(const byte* d_sourceImage, byte* d_destinationImage, const unsigned int width, const unsigned int height) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	float sobelX;
	float sobelY;
	if (x > 0 && y > 0 && x < width - 1 && y < height - 1) {
		sobelX = (-1 * d_sourceImage[(y - 1) * width + (x - 1)]) + (-2 * d_sourceImage[y * width + (x - 1)]) + (-1 * d_sourceImage[(y + 1) * width + (x - 1)]) +
			(d_sourceImage[(y - 1) * width + (x + 1)]) + (2 * d_sourceImage[y * width + (x + 1)]) + (d_sourceImage[(y + 1) * width + (x + 1)]);

		sobelY = (-1 * d_sourceImage[(y - 1) * width + (x - 1)]) + (-2 * d_sourceImage[(y - 1) * width + x]) + (-1 * d_sourceImage[(y - 1) * width + (x + 1)]) +
			(d_sourceImage[(y + 1) * width + (x - 1)]) + (2 * d_sourceImage[(y + 1) * width + x]) + (d_sourceImage[(y + 1) * width + (x + 1)]);

		d_destinationImage[y * width + x] = sqrt((sobelX * sobelX) + (sobelY * sobelY));
	}
}


void createGaussKernel(double gaussKernel[5])
{
	// intialising standard deviation to 1.0 
	double sigma = 1.0;
	double s = 2.0 * sigma * sigma;
	double r;

	double sum = 0.0;

	for (int x = -2; x <= 2; x++) {
		for (int y = -2; y <= 2; y++) {
			r = sqrt(x * x + y * y);
			gaussKernel[x + 2 + (y + 2) * 5] = (exp(-(r * r) / s)) / (PI * s);
			sum += gaussKernel[x + 2 + (y + 2) * 5];
		}
	}

	// normalising the Kernel 
	for (int i = 0; i < 5; ++i) {
		for (int j = 0; j < 5; ++j) {
			gaussKernel[i * 5 + j] /= sum;
		}
	}
}

void gaussianBlurCpu(const byte* original, byte* destination, double* gKernel, const unsigned int width, const unsigned int height) {
	for (int y = 0; y < height - 1; y++)
	{
		for (int x = 0; x < width - 1; x++)
		{
			if (x > 0 && y > 0 && x < width - 1 && y < height - 1)
			{
				double sum = 0;
				for (int i = 0; i < 5; i++)
				{
					for (int j = 0; j < 5; j++)
					{
						int num;
						if (y < 4 || x < 4) {
							num = 20;
						}
						else {
							num = original[(y - 2 + i) * width + (x - 2 + j)];
						}
						sum += num * gKernel[i * 5 + j];
					}
				}
				destination[y * width + x] = round(sum);
			}
		}
	}
}

__device__ double d_gKernel[25];

__global__ void gaussianBlurGpu(const byte* d_sourceImage, byte* d_destinationImage, const unsigned int width, const unsigned int height) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x > 0 && y > 0 && x < width - 1 && y < height - 1)
	{
		double sum = 0;
		for (int i = 0; i < 5; i++)
		{
			for (int j = 0; j < 5; j++)
			{
				int num;
				if (y < 4 || x < 4) {
					num = 20;
				}
				else {
					num = d_sourceImage[(y - 2 + i) * width + (x - 2 + j)];
				}
				sum += num * d_gKernel[i * 5 + j];
			}
		}
		d_destinationImage[y * width + x] = round(sum);
	}
}

int main() {
	char* filename = INPUT_IMG;
	Image originalImage = loadPngImage(filename);
	auto width = originalImage.width;
	auto height = originalImage.height;

	dim3 threadsPerBlock(THREAD_COUNT_PER_DIM, THREAD_COUNT_PER_DIM);
	dim3 numberOfBlocks(ceil(width / THREAD_COUNT_PER_DIM), ceil(height / THREAD_COUNT_PER_DIM));

	double gKernel[25];
	createGaussKernel(gKernel);

	std::ofstream metrics("metrics.txt");

	metrics << "gauss-cpu;gauss-gpu;sobel-cpu;sobel-gpu" << std::endl;

	for (int i = 0; i < 100; i++)
	{

#pragma region Gaussian filter[CPU]
		Image gaussDestinationImageCpu(new byte[width * height], width, height);

		auto gaussStartCpu = high_resolution_clock::now();
		gaussianBlurCpu(originalImage.pixels, gaussDestinationImageCpu.pixels, gKernel, width, height);
		auto gaussStopCpu = high_resolution_clock::now();

		auto gaussElapsedTimeCpu = duration_cast<microseconds>(gaussStopCpu - gaussStartCpu);
		printf("Gaussian blur CPU: %ld ms\n", gaussElapsedTimeCpu.count() / 1000);
		writePngImage(filename, "gauss_cpu", gaussDestinationImageCpu);
#pragma endregion

#pragma region Gaussian filter[GPU]
		Image gaussDestinationImageGpu(new byte[width * height], width, height);

		byte *d_gaussSource;
		byte *d_gaussDestination;

		cudaMalloc((void**)&d_gaussSource, (width * height));
		cudaMalloc((void**)&d_gaussDestination, (width * height));
		cudaMemcpy(d_gaussSource, originalImage.pixels, (width * height), cudaMemcpyHostToDevice);
		cudaMemset(d_gaussDestination, 0, (width * height));
		cudaMemcpyToSymbol(d_gKernel, gKernel, sizeof(double) * 25);

		cudaEvent_t gaussStart, gaussEnd;
		cudaEventCreate(&gaussStart);
		cudaEventCreate(&gaussEnd);

		cudaEventRecord(gaussStart, 0);
		gaussianBlurGpu << <numberOfBlocks, threadsPerBlock >> > (d_gaussSource, d_gaussDestination, width, height);
		cudaEventRecord(gaussEnd, 0);

		cudaMemcpy(gaussDestinationImageGpu.pixels, d_gaussDestination, (width * height), cudaMemcpyDeviceToHost);

		cudaEventSynchronize(gaussEnd);
		float gaussElapsedTimeGpu = 0.0f;
		cudaEventElapsedTime(&gaussElapsedTimeGpu, gaussStart, gaussEnd);
		printf("Gaussian blur GPU: %f ms\n", gaussElapsedTimeGpu);

		writePngImage(filename, "gauss_gpu", gaussDestinationImageGpu);

		cudaFree(d_gaussSource);
		cudaFree(d_gaussDestination);
#pragma endregion

#pragma region Sobel edge detection on blurred image[CPU]
		Image sobelDestinationImageCpu(new byte[width * height], width, height);

		auto sobelStartCpu = high_resolution_clock::now();
		sobelEdgeDetectionCpu(gaussDestinationImageCpu.pixels, sobelDestinationImageCpu.pixels, width, height);
		auto sobelEndCpu = high_resolution_clock::now();

		auto sobelElapsedTimeCpu = duration_cast<microseconds>(sobelEndCpu - sobelStartCpu);
		printf("Sobel edge detection CPU: %ld ms\n", sobelElapsedTimeCpu.count() / 1000);
		writePngImage(filename, "sobel_cpu", sobelDestinationImageCpu);
#pragma endregion

#pragma region Sobel edge detection on blurred image[GPU]
		Image sobelDestinationImageGpu(new byte[width * height], width, height);

		byte *d_sobelSource;
		byte *d_sobelDestination;
		cudaMalloc((void**)&d_sobelSource, (width * height));
		cudaMalloc((void**)&d_sobelDestination, (width * height));
		cudaMemcpy(d_sobelSource, gaussDestinationImageGpu.pixels, (width * height), cudaMemcpyHostToDevice);
		cudaMemset(d_sobelDestination, 0, (width * height));

		cudaEvent_t sobelStart, sobelEnd;
		cudaEventCreate(&sobelStart);
		cudaEventCreate(&sobelEnd);

		cudaEventRecord(sobelStart, 0);
		sobelEdgeDetectionGpu << <numberOfBlocks, threadsPerBlock >> > (d_sobelSource, d_sobelDestination, width, height);
		cudaEventRecord(sobelEnd, 0);

		cudaMemcpy(sobelDestinationImageGpu.pixels, d_sobelDestination, (width * height), cudaMemcpyDeviceToHost);

		cudaEventSynchronize(sobelEnd);
		float sobelElapsedTimeGpu = 0.0f;
		cudaEventElapsedTime(&sobelElapsedTimeGpu, sobelStart, sobelEnd);
		printf("Sobel edge detection GPU: %f ms\n", sobelElapsedTimeGpu);

		writePngImage(filename, "sobel_gpu", sobelDestinationImageGpu);

		cudaFree(d_sobelSource);
		cudaFree(d_sobelDestination);
#pragma endregion

		delete[] gaussDestinationImageCpu.pixels;
		delete[] gaussDestinationImageGpu.pixels;
		delete[] sobelDestinationImageCpu.pixels;
		delete[] sobelDestinationImageGpu.pixels;

		std::string line = std::to_string(gaussElapsedTimeCpu.count() / 1000) + ";" + std::to_string(gaussElapsedTimeGpu) + ";"
			+ std::to_string(sobelElapsedTimeCpu.count() / 1000) + ";" + std::to_string(sobelElapsedTimeGpu);

		metrics << line.c_str() << std::endl;
	}

	metrics.close();

	return 0;
}
