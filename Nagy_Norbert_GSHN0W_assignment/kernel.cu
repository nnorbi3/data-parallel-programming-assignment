
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "lodepng.h"
#include "lodepng.cpp"
#include <stdio.h>

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

	// sum is for normalization 
	double sum = 0.0;

	// generating 5x5 kernel 
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

#pragma region Gaussian filter
	double gKernel[25];
	createGaussKernel(gKernel);
	Image gaussDestinationImage(new byte[width * height], width, height);

	byte *d_gaussSource;
	byte *d_gaussDestination;

	cudaMalloc((void**)&d_gaussSource, (width * height));
	cudaMalloc((void**)&d_gaussDestination, (width * height));
	cudaMemcpy(d_gaussSource, originalImage.pixels, (width * height), cudaMemcpyHostToDevice);
	cudaMemset(d_gaussDestination, 0, (width * height));
	cudaMemcpyToSymbol(d_gKernel, gKernel, sizeof(double) * 25);

	gaussianBlurGpu << <numberOfBlocks, threadsPerBlock >> > (d_gaussSource, d_gaussDestination, width, height);

	cudaMemcpy(gaussDestinationImage.pixels, d_gaussDestination, (width * height), cudaMemcpyDeviceToHost);

	writePngImage(filename, "gauss", gaussDestinationImage);

	cudaFree(d_gaussSource);
	cudaFree(d_gaussDestination);
#pragma endregion

#pragma region Sobel edge detection on blurred image
	Image sobelDestinationImage(new byte[width * height], width, height);

	byte *d_sobelSource;
	byte *d_sobelDestination;
	cudaMalloc((void**)&d_sobelSource, (width * height));
	cudaMalloc((void**)&d_sobelDestination, (width * height));
	cudaMemcpy(d_sobelSource, gaussDestinationImage.pixels, (width * height), cudaMemcpyHostToDevice);
	cudaMemset(d_sobelDestination, 0, (width * height));

	sobelEdgeDetectionGpu << <numberOfBlocks, threadsPerBlock >> > (d_sobelSource, d_sobelDestination, width, height);

	cudaMemcpy(sobelDestinationImage.pixels, d_sobelDestination, (width * height), cudaMemcpyDeviceToHost);

	writePngImage(filename, "sobel", sobelDestinationImage);

	cudaFree(d_sobelSource);
	cudaFree(d_sobelDestination);
#pragma endregion

	delete[] gaussDestinationImage.pixels;
	delete[] sobelDestinationImage.pixels;

	return 0;
}
