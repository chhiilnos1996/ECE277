#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#define PI acos(-1.0)

using namespace std;
using namespace cv;

void opencvImageToCpuFloat(Mat & img, float *image);
void printCvDataAndImageFloatArray(Mat & img, float *image);
void cpuComputeKernel(float *kernel, const unsigned int on, const unsigned int kn, const unsigned int ks);
void printKernel(float *kernel, const unsigned int on, const unsigned int kn, const unsigned int ks);
void computeAndPadOctave(float *padded_octave, float *image, const unsigned int nrows, const unsigned int ncols, const unsigned int ps, const float scale);
void printPaddedOctavesSize(const unsigned int nrows, const unsigned int ncols, const unsigned int ps);
void savePaddedOctaves(float *padded_octave1_1, float *padded_octave2_1, float *padded_octave3_1, float *padded_octave4_1, const unsigned int nrows, const unsigned int ncols, const unsigned int ps, const unsigned int img_id);
void blurOctave(float *blurred_octave, float *padded_octave, float *kernel, const unsigned int nrows, const unsigned int ncols, const unsigned int kn, const unsigned int ps, const float scale);
void saveblurredOctaves(float *blurred_octave1, float *blurred_octave2, float *blurred_octave3, float *blurred_octave4, const unsigned int nrows, const unsigned int ncols, const unsigned int kn, const unsigned int img_id);
void computeDogOctave(float *dog_octave, float *blurred_octave, const unsigned int nrows, const unsigned int ncols, const unsigned int kn, const float scale);
void saveDogOctaves(float *dog_octave1, float *dog_octave2, float *dog_octave3, float *dog_octave4, const unsigned int nrows, const unsigned int ncols,const unsigned int kn, const unsigned int img_id);
void findKeypointsOctave(unsigned short *keypoints_octave, float *dog_octave, const unsigned int nrows, const unsigned int ncols, const unsigned int kn, const float scale);

__global__ void gpuUcharToFloat(float *d_a, unsigned char *d_b, const unsigned int nx) {
	const unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int i = iy * nx + ix; // read and write by row
	d_a[i] = (float)d_b[i];
}

__global__ void gpuComputeKernel(float *d_a, const unsigned int on, const unsigned int kn, const unsigned int ks, const unsigned int nx) {
	const unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int i = iy * nx + ix; // read and write by row
	if (i >= 500) return;
	// i/25 is kernel id, i%5 -2 is kernel.x ([-2,-1,0,1,2]), (i%25)/5 -2 is kernel.y ([-2,-1,0,1,2])
	const unsigned int octave_id = i / 125, kernel_id = (i % 125) / 25;
	const unsigned int x = i % 5 - 2, y = (i % 25) / 5 - 2;
	const float sigma_square = powf(2, (kernel_id + 2.0 * octave_id - 1.0) / 2);
	const float power = (x * x + y * y) / (-2 * sigma_square);
	const float denominator = 2 * PI * sigma_square;
	d_a[i] = expf(power) / denominator;
	//printf("i = %d, ix = %d, iy = %d, octave_id = %d, kernel_id = %d, x = %d, y = %d, sigma_square = %f, power = %f, denominator = %f, value = %f\n", i, ix, iy, octave_id, kernel_id, x, y, sigma_square, power, denominator, d_a[i]);	
}

__global__ void gpuComputePaddedOctave(float *d_padded_octave, float *d_image, const int nrows, const int ncols, const int ps, const float scale) {
	const unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int nrows_scale = nrows * scale, ncols_scale = ncols * scale, i = iy * ncols_scale + ix;
	if ((ix >= ncols_scale) || (iy >= nrows_scale)) return;

	// index on gpu padded octave array
	const unsigned int iy_pad = iy + ps, ix_pad = ix + ps, nrows_padded_octave = nrows_scale + 2 * ps, ncols_padded_octave = ncols_scale + 2 * ps;
	const unsigned int i_po = iy_pad * ncols_padded_octave + ix_pad; // read by row

																	 // index on gpu image array
	const unsigned int iy_img = iy / scale, ix_img = ix / scale;
	const unsigned int i_img = iy_img * ncols + ix_img; // write by row

														// copy value
	d_padded_octave[i_po] = d_image[i_img];
	/*
	if (i_po % 10000 == 0) {
	printf("i = %d, ix = %d, ncols_scale = %d, iy = %d, nrows_scale = %d\n", i, ix, ncols_scale, iy, nrows_scale);
	printf("i_po = %d, ix_pad = %d, ncols_padded_octave = %d, iy_pad = %d, nrows_padded_octave = %d\n", i_po, ix_pad, ncols_padded_octave, iy_pad, nrows_padded_octave);
	printf("i_img = %d, ix_img = %d, ncols = %d, iy_img = %d, nrows = %d\n", i_img, ix_img, ncols, iy_img, nrows);
	printf("d_padded_octave[i_po] = %f, d_image[i_img] = %f\n",d_padded_octave[i_po], d_image[i_img]);
	}*/
}

__global__ void gpuBlurOctave(float *d_blurred_octave, float *d_padded_octave, float *d_kernel_octave, const unsigned int nrows, const unsigned int ncols, const unsigned int kn, const int ps, const float scale) {
	// index on gpu padded octave array
	const unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int nrows_scale = nrows * scale, ncols_scale = ncols * scale;
	if ((ix >= ncols_scale) || (iy >= nrows_scale)) return;
	const unsigned int iy_pad = iy + ps, ix_pad = ix + ps;
	const unsigned int ks = 2 * ps + 1;
	const unsigned int nrows_padded_octave = nrows_scale + 2 * ps, ncols_padded_octave = ncols_scale + 2 * ps;
	//printf("ix = %d, blockIdx.x = %d, blockDim.x = %d, threadIdx.x = %d\n", ix, blockIdx.x, blockDim.x, threadIdx.x);
	//printf("iy = %d, blockIdx.y = %d, blockDim.y = %d, threadIdx.y = %d\n", iy, blockIdx.y, blockDim.y, threadIdx.y);

	// copy value 
	for (unsigned int kid = 0; kid < kn; kid++) {
		const unsigned int i = kid * nrows_scale * ncols_scale + iy * ncols_scale + ix;
		float tmp = 0;
		for (int j = -ps; j <= ps; j++) {
			for (int k = -ps; k <= ps; k++) {
				const unsigned int i_ko = kid * ks * ks + (k + ps) * ks + (j + ps);
				const unsigned int i_po = (iy_pad + k) * ncols_padded_octave + (ix_pad + j);
				tmp += d_kernel_octave[i_ko] * d_padded_octave[i_po];
				/*
				if (i % 1000 == 0) {
				printf("i = %d, ix = %d, iy = %d\n", ix, iy);
				printf("i_ko = %d, j+ps = %d, k+ps = %d\n", i_ko, j+ps, k+ps);
				printf("i_po = %d, ix_pad+j = %d, iy_pad+k = %d\n", i_po, ix_pad+j, iy_pad+k);
				printf("d_kernel_octave[i_ko] = %f, d_padded_octave[i_po] = %f, tmp = %f\n", d_kernel_octave[i_ko], d_padded_octave[i_po], tmp);
				}*/
			}
		}
		d_blurred_octave[i] = tmp;
	}
}

__global__ void gpuComputeDogOctave(float *d_dog_octave, float *d_blurred_octave, const unsigned int nrows, const unsigned int ncols, const unsigned int kn, const float scale) {
	// index on gpu dog octave array
	const unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int nrows_scale = nrows * scale, ncols_scale = ncols * scale;
	if ((ix >= ncols_scale) || (iy >= nrows_scale)) return;

	// substract and assign  
	for (unsigned int kid = 0; kid < kn - 1 ; kid++) {
		const unsigned int i = kid * nrows_scale * ncols_scale + iy * ncols_scale + ix;
		d_dog_octave[i] = d_blurred_octave[i] - d_blurred_octave[i + nrows_scale * ncols_scale];
	}
}

__global__ void gpuFindKeypointsOctave(unsigned short *d_keypoints_octave, float *d_dog_octave, const unsigned int nrows, const unsigned int ncols, const unsigned int kn, const float scale){
	// index on gpu dog octave array
	const unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int nrows_scale = nrows * scale, ncols_scale = ncols * scale;
	if ((ix >= ncols_scale) || (iy >= nrows_scale)) return;
	if ((ix == ncols_scale - 1) && (iy == nrows_scale - 1)) printf("*******************************************\n");
	//if (threadIdx.x==0 && threadIdx.y==0) printf("ix = %d, ncols_scale = %d, blockIdx.x = %d, threadIdx.x = %d, iy = %d, nrows_scale = %d, blockIdx.y = %d, threadIdx.y = %d\n", ix, ncols_scale, blockIdx.x, threadIdx.x, iy, nrows_scale, blockIdx.y, threadIdx.y);

	// scan neigbors and determine if it is keypoint  
	for (unsigned int kid = 1; kid < kn - 2; kid++) {
		const unsigned int i = kid * nrows_scale * ncols_scale + iy * ncols_scale + ix;
		bool is_max = true, is_min = true;
		for (int l = -1; l <= 1; l++) { // previous layer to next layer
			for (int j = -1; j <= 1; j++) { //previous column to next column
				for (int k = -1; k <= 1; k++) { // previous row to next row
					// index of neighbor
					//if (threadIdx.x == 0 && threadIdx.y == 0) printf("ix = %d, iy = %d, kid = %d, l = %d, j = %d, k = %d\n", ix, iy, kid, l, j, k);
					const unsigned int i_n = (kid + l) * nrows_scale * ncols_scale + (iy + k) * ncols_scale + (ix + j);
					if (d_dog_octave[i_n]>d_dog_octave[i]) is_max = false;
					if (d_dog_octave[i_n]<d_dog_octave[i]) is_min = false;
				}
			}
		}
		
		if (is_max == true || is_min == true) d_keypoints_octave[i] = 255;
		else d_keypoints_octave[i] = 0;
		if (threadIdx.x == 0 && threadIdx.y == 0) printf("kid = %d, ix = %d, iy = %d, d_keypoints_octave[i] = %d\n", kid, ix, iy, d_keypoints_octave[i]);
	}
}


int main()
{
	//0. opencv read image and put into cpu float array
	// opencv read image
	string img1_path = "C:/Users/c2lo/Desktop/img1.jpg", img2_path = "C:/Users/c2lo/Desktop/img2.jpg";
	Mat img1 = imread(img1_path.c_str(), IMREAD_GRAYSCALE);
	Mat img2 = imread(img2_path.c_str(), IMREAD_GRAYSCALE);
	const unsigned int nrows1 = img1.rows, ncols1 = img1.cols, nrows2 = img2.rows, ncols2 = img2.cols;
	printf("img1 size = %d x %d, img2 size = %d x %d\n",nrows1, ncols1, nrows2, ncols2);
	float *image1 = (float*)malloc(nrows1 * ncols1 * sizeof(float));
	float *image2 = (float*)malloc(nrows2 * ncols2 * sizeof(float));
	opencvImageToCpuFloat(img1, image1); opencvImageToCpuFloat(img2, image2);
	//printCvDataAndImageFloatArray(img1, image1); printCvDataAndImageFloatArray(img2, image2);
	cudaDeviceSynchronize();

	//1. compute kernels
	// compute gpu kernel float array
	const unsigned int  octave_num = 4, kernel_num = 5, kernel_size = 5; //must be odd
	float *kernel = (float*)malloc(octave_num* kernel_num * kernel_size * kernel_size * sizeof(float));
	cpuComputeKernel(kernel, octave_num, kernel_num, kernel_size);
	printKernel(kernel, octave_num, kernel_num, kernel_size);
	

	//2. image to padded octaves
	const unsigned int pad_size = kernel_size / 2;
	const unsigned int nrows_padded_octave1_1 = nrows1 * 2 + 2 * pad_size, ncols_padded_octave1_1 = ncols1 * 2 + 2 * pad_size;
	const unsigned int nrows_padded_octave2_1 = nrows1 + 2 * pad_size, ncols_padded_octave2_1 = ncols1 + 2 * pad_size;
	const unsigned int nrows_padded_octave3_1 = nrows1 / 2 + 2 * pad_size, ncols_padded_octave3_1 = ncols1 / 2 + 2 * pad_size;
	const unsigned int nrows_padded_octave4_1 = nrows1 / 4 + 2 * pad_size, ncols_padded_octave4_1 = ncols1 / 4 + 2 * pad_size;
	float *padded_octave1_1 = (float*)malloc(nrows_padded_octave1_1 * ncols_padded_octave1_1 * sizeof(float));
	float *padded_octave2_1 = (float*)malloc(nrows_padded_octave2_1 * ncols_padded_octave2_1 * sizeof(float));
	float *padded_octave3_1 = (float*)malloc(nrows_padded_octave3_1 * ncols_padded_octave3_1 * sizeof(float));
	float *padded_octave4_1 = (float*)malloc(nrows_padded_octave4_1 * ncols_padded_octave4_1 * sizeof(float));
	//printPaddedOctavesSize(nrows1, ncols1, pad_size);
	computeAndPadOctave(padded_octave1_1, image1, nrows1, ncols1, pad_size, 2.0);
	computeAndPadOctave(padded_octave2_1, image1, nrows1, ncols1, pad_size, 1.0);
	computeAndPadOctave(padded_octave3_1, image1, nrows1, ncols1, pad_size, 0.5);
	computeAndPadOctave(padded_octave4_1, image1, nrows1, ncols1, pad_size, 0.25);
	savePaddedOctaves(padded_octave1_1, padded_octave2_1, padded_octave3_1, padded_octave4_1, nrows1, ncols1, pad_size, 1);
	

	const unsigned int nrows_padded_octave1_2 = nrows2 * 2 + 2 * pad_size, ncols_padded_octave1_2 = ncols2 * 2 + 2 * pad_size;
	const unsigned int nrows_padded_octave2_2 = nrows2 + 2 * pad_size, ncols_padded_octave2_2 = ncols2 + 2 * pad_size;
	const unsigned int nrows_padded_octave3_2 = nrows2 / 2 + 2 * pad_size, ncols_padded_octave3_2 = ncols2 / 2 + 2 * pad_size;
	const unsigned int nrows_padded_octave4_2 = nrows2 / 4 + 2 * pad_size, ncols_padded_octave4_2 = ncols2 / 4 + 2 * pad_size;
	float *padded_octave1_2 = (float*)malloc(nrows_padded_octave1_2 * ncols_padded_octave1_2 * sizeof(float));
	float *padded_octave2_2 = (float*)malloc(nrows_padded_octave2_2 * ncols_padded_octave2_2 * sizeof(float));
	float *padded_octave3_2 = (float*)malloc(nrows_padded_octave3_2 * ncols_padded_octave3_2 * sizeof(float));
	float *padded_octave4_2 = (float*)malloc(nrows_padded_octave4_2 * ncols_padded_octave4_2 * sizeof(float));
	//printPaddedOctavesSize(nrows2, ncols2, pad_size);
	computeAndPadOctave(padded_octave1_2, image2, nrows2, ncols2, pad_size, 2.0);
	computeAndPadOctave(padded_octave2_2, image2, nrows2, ncols2, pad_size, 1.0);
	computeAndPadOctave(padded_octave3_2, image2, nrows2, ncols2, pad_size, 0.5);
	computeAndPadOctave(padded_octave4_2, image2, nrows2, ncols2, pad_size, 0.25);
	savePaddedOctaves(padded_octave1_2, padded_octave2_2, padded_octave3_2, padded_octave4_2, nrows2, ncols2, pad_size, 2);
	

	//3. gaussian blur
	const unsigned int nrows_octave1_1 = nrows1 * 2, ncols_octave1_1 = ncols1 * 2;
	const unsigned int nrows_octave2_1 = nrows1, ncols_octave2_1 = ncols1;
	const unsigned int nrows_octave3_1 = nrows1 / 2, ncols_octave3_1 = ncols1 / 2;
	const unsigned int nrows_octave4_1 = nrows1 / 4, ncols_octave4_1 = ncols1 / 4;
	float *blurred_octave1_1 = (float*)malloc(kernel_num * nrows_octave1_1 * ncols_octave1_1 * sizeof(float));
	float *blurred_octave2_1 = (float*)malloc(kernel_num * nrows_octave2_1 * ncols_octave2_1 * sizeof(float));
	float *blurred_octave3_1 = (float*)malloc(kernel_num * nrows_octave3_1 * ncols_octave3_1 * sizeof(float));
	float *blurred_octave4_1 = (float*)malloc(kernel_num * nrows_octave4_1 * ncols_octave4_1 * sizeof(float));
	blurOctave(blurred_octave1_1, padded_octave1_1, kernel, nrows1, ncols1, kernel_num, pad_size, 2.0);
	blurOctave(blurred_octave2_1, padded_octave2_1, kernel, nrows1, ncols1, kernel_num, pad_size, 1.0);
	blurOctave(blurred_octave3_1, padded_octave3_1, kernel, nrows1, ncols1, kernel_num, pad_size, 0.5);
	blurOctave(blurred_octave4_1, padded_octave4_1, kernel, nrows1, ncols1, kernel_num, pad_size, 0.25);
	saveblurredOctaves(blurred_octave1_1, blurred_octave2_1, blurred_octave3_1, blurred_octave4_1, nrows1, ncols1, kernel_num, 1);
	

	//4. laplacian of gaussian
	float *dog_octave1_1 = (float*)malloc((kernel_num - 1) * nrows_octave1_1 * ncols_octave1_1 * sizeof(float));
	float *dog_octave2_1 = (float*)malloc((kernel_num - 1) * nrows_octave2_1 * ncols_octave2_1 * sizeof(float));
	float *dog_octave3_1 = (float*)malloc((kernel_num - 1)* nrows_octave3_1 * ncols_octave3_1 * sizeof(float));
	float *dog_octave4_1 = (float*)malloc((kernel_num - 1) * nrows_octave4_1 * ncols_octave4_1 * sizeof(float));
	computeDogOctave(dog_octave1_1, blurred_octave1_1, nrows1, ncols1, kernel_num, 2.0);
	computeDogOctave(dog_octave2_1, blurred_octave2_1, nrows1, ncols1, kernel_num, 1.0);
	computeDogOctave(dog_octave3_1, blurred_octave3_1, nrows1, ncols1, kernel_num, 0.5);
	computeDogOctave(dog_octave4_1, blurred_octave4_1, nrows1, ncols1, kernel_num, 0.25);
	saveDogOctaves(dog_octave1_1, dog_octave2_1, dog_octave3_1, dog_octave4_1, nrows1, ncols1, kernel_num, 1);
	

	//5. find key points
	unsigned short *keypoints_octave1_1 = (unsigned short*)malloc((kernel_num - 3) * nrows_octave1_1 * ncols_octave1_1 * sizeof(unsigned short));
	unsigned short *keypoints_octave2_1 = (unsigned short*)malloc((kernel_num - 3) * nrows_octave2_1 * ncols_octave2_1 * sizeof(unsigned short));
	unsigned short *keypoints_octave3_1 = (unsigned short*)malloc((kernel_num - 3)* nrows_octave3_1 * ncols_octave3_1 * sizeof(unsigned short));
	unsigned short *keypoints_octave4_1 = (unsigned short*)malloc((kernel_num - 3) * nrows_octave4_1 * ncols_octave4_1 * sizeof(unsigned short));
	findKeypointsOctave(keypoints_octave1_1, dog_octave1_1, nrows1, ncols1, kernel_num, 2.0);
	findKeypointsOctave(keypoints_octave2_1, dog_octave2_1, nrows1, ncols1, kernel_num, 1.0);
	findKeypointsOctave(keypoints_octave3_1, dog_octave3_1, nrows1, ncols1, kernel_num, 0.5);
	findKeypointsOctave(keypoints_octave4_1, dog_octave4_1, nrows1, ncols1, kernel_num, 0.25);
	//saveKeypointsOctaves(dog_octave1_1, dog_octave2_1, dog_octave3_1, dog_octave4_1, nrows1, ncols1, kernel_num, 1);
	/*
	for (unsigned int i = 0; i < (kernel_num - 3) * nrows_octave1_1 * ncols_octave1_1; i++) printf("%i",keypoints_octave1_1[i]);
	printf("\n");
	for (unsigned int i = 0; i < (kernel_num - 3) * nrows_octave2_1 * ncols_octave2_1; i++) printf("%i", keypoints_octave2_1[i]);
	printf("\n");
	for (unsigned int i = 0; i < (kernel_num - 3) * nrows_octave3_1 * ncols_octave3_1; i++) printf("%i", keypoints_octave3_1[i]);
	printf("\n");*/
	//for (unsigned int i = 0; i < (kernel_num - 3) * nrows_octave4_1 * ncols_octave4_1; i++) printf("%d", keypoints_octave4_1[i]);
	//printf("\n");

	//write image
	imwrite("C:/Users/c2lo/Desktop/store_img1.jpg", img1);
	imwrite("C:/Users/c2lo/Desktop/store_img2.jpg", img2);

	return 0;
}

void opencvImageToCpuFloat(Mat & img, float *image) {
	// read cv Mat cpu uchar array to gpu uchar array
	unsigned char *d_uimage;
	const int nrows = img.rows, ncols = img.cols;
	cudaMalloc((void**)&d_uimage, nrows * ncols * sizeof(unsigned char));
	cudaMemcpyAsync(d_uimage, img.data, nrows * ncols * sizeof(unsigned char), cudaMemcpyHostToDevice);

	// gpu uchar array to gpu float array 
	float *d_image;
	cudaMalloc((void**)&d_image, nrows * ncols * sizeof(float));

	int nx = ncols, ny = nrows, dimx = 32, dimy = 32;
	dim3 block(dimx, dimy);
	dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
	gpuUcharToFloat << <grid, block >> > (d_image, d_uimage, nx);
	cudaDeviceSynchronize();

	// gpu float array to cpu float array
	cudaMemcpyAsync(image, d_image, nrows * ncols * sizeof(float), cudaMemcpyDeviceToHost);

	// free and cuda free
	cudaFree(d_uimage); cudaFree(d_image);
}

void printCvDataAndImageFloatArray(Mat & img, float *image) {
	int nrows = img.rows, ncols = img.cols;
	printf("img.data[:10] and img.data[-10:]\n");
	for (int i = 0; i < 10; i++) {
		printf("%u %u ", img.data[i], img.data[nrows*ncols - 11 + i]);
	}
	printf("\n");
	printf("image[:10] and image[-10:]\n");
	for (int i = 0; i < 10; i++) {
		printf("%f %f ", image[i], image[nrows*ncols - 11 + i]);
	}
	printf("\n================================\n");
}

void cpuComputeKernel(float *kernel, const unsigned int on, const unsigned int kn, const unsigned int ks) {
	// cuda<alloc gpu float array 
	float* d_kernel;
	cudaMalloc((void**)&d_kernel, on * kn * ks * ks * sizeof(float));

	// compute gpu float array
	const unsigned int nx = 32, dimx = 32, dimy = 16;
	dim3 block(dimx, dimy);
	dim3 grid(1, 1);
	gpuComputeKernel << <grid, block >> > (d_kernel, on, kn, ks, nx);
	cudaDeviceSynchronize();

	// cudaMemcpy gpu float array to cpu float array
	cudaMemcpyAsync(kernel, d_kernel, on * kn * ks * ks * sizeof(float), cudaMemcpyDeviceToHost);

	// free and cudaFree
	cudaFree(d_kernel);
}

void printKernel(float *kernel, const unsigned int on, const unsigned int kn, const unsigned int ks) {
	for (unsigned int i = 0; i < on; i++) {
		for (unsigned int j = 0; j < kn; j++) {
			printf("octave_id = %d, kernel_id = %d \n", i, j);
			for (unsigned int k = 0; k < ks; k++) {
				for (unsigned int l = 0; l < ks; l++) {
					printf("%f ", kernel[i * kn * ks * ks + j * ks * ks + k * ks + l]);
				}
				printf("\n");
			}
			printf("---------------------------------\n");
		}
		printf("================================\n");
	}
}

void computeAndPadOctave(float *padded_octave, float *image, const unsigned int nrows, const unsigned int ncols, const unsigned int ps, const float scale) {
	// cudaMalloc gpu image float array and gpu padded octave float array
	const unsigned int nrows_padded_octave = nrows * scale + 2 * ps, ncols_padded_octave = ncols * scale + 2 * ps;
	//printf("nrows = %d, ncols = %d, pad_size = %d, scale = %f, nrows_padded_octave = %d, ncols_padded_octave = %d\n", nrows, ncols, ps, scale, nrows_padded_octave, ncols_padded_octave);
	float *d_image, *d_padded_octave;
	cudaMalloc((void**)&d_image, nrows * ncols * sizeof(float));
	cudaMalloc((void**)&d_padded_octave, nrows_padded_octave * ncols_padded_octave * sizeof(float));

	// memcpy cpu image float array to gpu image float array
	cudaMemcpyAsync(d_image, image, nrows * ncols * sizeof(float), cudaMemcpyHostToDevice);

	// compute gpu padded octave float array
	const unsigned int nx = ncols * scale, ny = nrows * scale, dimx = 32, dimy = 32;
	dim3 block(dimx, dimy);
	dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
	gpuComputePaddedOctave << <grid, block >> > (d_padded_octave, d_image, nrows, ncols, ps, scale);
	cudaDeviceSynchronize();

	// cudaMemcpy gpu padded octave float array to cpu padded octave float array
	cudaMemcpyAsync(padded_octave, d_padded_octave, nrows_padded_octave * ncols_padded_octave * sizeof(float), cudaMemcpyDeviceToHost);

	// free and cudaFree
	cudaFree(d_padded_octave); cudaFree(d_image);
}

void printPaddedOctavesSize(const unsigned int nrows, const unsigned int ncols, const unsigned int ps) {
	const unsigned int nrows_padded_octave1 = nrows * 2 + 2 * ps, ncols_padded_octave1 = ncols * 2 + 2 * ps;
	const unsigned int nrows_padded_octave2 = nrows + 2 * ps, ncols_padded_octave2 = ncols + 2 * ps;
	const unsigned int nrows_padded_octave3 = nrows / 2 + 2 * ps, ncols_padded_octave3 = ncols / 2 + 2 * ps;
	const unsigned int nrows_padded_octave4 = nrows / 4 + 2 * ps, ncols_padded_octave4 = ncols / 4 + 2 * ps;
	printf("nrows = %d, ncols = %d, pad_size = %d\n", nrows, ncols, ps);
	printf("nrows_padded_octave1 = %d, ncols_padded_octave1 = %d\n", nrows_padded_octave1, ncols_padded_octave1);
	printf("nrows_padded_octave2 = %d, ncols_padded_octave2 = %d\n", nrows_padded_octave2, ncols_padded_octave2);
	printf("nrows_padded_octave3 = %d, ncols_padded_octave3 = %d\n", nrows_padded_octave3, ncols_padded_octave3);
	printf("nrows_padded_octave4 = %d, ncols_padded_octave4 = %d\n", nrows_padded_octave4, ncols_padded_octave4);
	printf("================================\n");
}

void savePaddedOctaves(float *padded_octave1, float *padded_octave2, float *padded_octave3, float *padded_octave4, const unsigned int nrows, const unsigned int ncols, const unsigned int ps, const unsigned int img_id) {
	printf("saving padded octaves of image %d\n", img_id);
	const unsigned int nrows_padded_octave1 = nrows * 2 + 2 * ps, ncols_padded_octave1 = ncols * 2 + 2 * ps;
	const unsigned int nrows_padded_octave2 = nrows + 2 * ps, ncols_padded_octave2 = ncols + 2 * ps;
	const unsigned int nrows_padded_octave3 = nrows / 2 + 2 * ps, ncols_padded_octave3 = ncols / 2 + 2 * ps;
	const unsigned int nrows_padded_octave4 = nrows / 4 + 2 * ps, ncols_padded_octave4 = ncols / 4 + 2 * ps;
	Mat padded_octave1_img(nrows_padded_octave1, ncols_padded_octave1, CV_32FC1, padded_octave1);
	Mat padded_octave2_img(nrows_padded_octave2, ncols_padded_octave2, CV_32FC1, padded_octave2);
	Mat padded_octave3_img(nrows_padded_octave3, ncols_padded_octave3, CV_32FC1, padded_octave3);
	Mat padded_octave4_img(nrows_padded_octave4, ncols_padded_octave4, CV_32FC1, padded_octave4);

	char path1[100] = "", path2[100] = "", path3[100] = "", path4[100] = "";
	sprintf(path1, "C:/Users/c2lo/Desktop/padded_octave1_%d_img.jpg", img_id);
	sprintf(path2, "C:/Users/c2lo/Desktop/padded_octave2_%d_img.jpg", img_id);
	sprintf(path3, "C:/Users/c2lo/Desktop/padded_octave3_%d_img.jpg", img_id);
	sprintf(path4, "C:/Users/c2lo/Desktop/padded_octave4_%d_img.jpg", img_id);
	imwrite(path1, padded_octave1_img); imwrite(path2, padded_octave2_img);
	imwrite(path3, padded_octave3_img); imwrite(path4, padded_octave4_img);
}

void blurOctave(float *blurred_octave, float *padded_octave, float* kernel, const unsigned int nrows, const unsigned int ncols, const unsigned int kn, const unsigned int ps, const float scale) {
	// cudaMalloc gpu blurred octave float array, gpu padded octave float array, gpu kernerl float array
	float *d_blurred_octave, *d_padded_octave, *d_kernel_octave;
	const unsigned int ks = 2 * ps + 1;
	const unsigned int nrows_blurred_octave = nrows * scale, ncols_blurred_octave = ncols * scale;
	const unsigned int nrows_padded_octave = nrows * scale + 2 * ps, ncols_padded_octave = ncols * scale + 2 * ps;
	cudaMalloc((void**)&d_blurred_octave, kn * nrows_blurred_octave * ncols_blurred_octave * sizeof(float));
	cudaMalloc((void**)&d_padded_octave, nrows_padded_octave * ncols_padded_octave * sizeof(float));
	cudaMalloc((void**)&d_kernel_octave, kn * ks * ks * sizeof(float));

	// memcpy cpu padded octave and cpu kernel to gpu padded octave and gpu kernel_octave
	const unsigned int octave_id = -log2f(scale) + 1;
	cudaMemcpyAsync(d_padded_octave, padded_octave, nrows_padded_octave * ncols_padded_octave * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_kernel_octave, &kernel[octave_id * kn * ks * ks], kn * ks * ks * sizeof(float), cudaMemcpyHostToDevice);
	/*
	float *kernel_octave = (float*)(malloc(kn * ks * ks * sizeof(float)));
	cudaMemcpy(kernel_octave, d_kernel_octave, kn * ks * ks * sizeof(float), cudaMemcpyDeviceToHost);
	for (int j = 0; j < kn; j++) {
	printf("octave_id = %d, kernel_id = %d \n", octave_id, j);
	for (int k = 0; k < ks; k++) {
	for (int l = 0; l < ks; l++) {
	printf("%f ", kernel_octave[j * ks * ks + k * ks + l]);
	}
	printf("\n");
	}
	printf("---------------------------------\n");
	}
	printf("================================\n");
	*/

	// compute gaussian blur on gpu
	const unsigned int nx = ncols * scale, ny = nrows * scale, dimx = 32, dimy = 32;
	dim3 block(dimx, dimy);
	dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
	gpuBlurOctave << <grid, block >> > (d_blurred_octave, d_padded_octave, d_kernel_octave, nrows, ncols, kn, ps, scale);
	cudaDeviceSynchronize();

	// cudaMemcpy gpu padded octave float array to cpu padded octave float array
	cudaMemcpy(blurred_octave, d_blurred_octave, kn * nrows_blurred_octave * ncols_blurred_octave * sizeof(float), cudaMemcpyDeviceToHost);

	// free and cudaFree
	cudaFree(d_blurred_octave); cudaFree(d_padded_octave); cudaFree(d_kernel_octave);
}

void saveblurredOctaves(float *blurred_octave1, float *blurred_octave2, float *blurred_octave3, float *blurred_octave4, const unsigned int nrows, const unsigned int ncols, const unsigned int kn, const unsigned int img_id) {
	printf("saving blurredd octaves of image %d\n", img_id);
	for (unsigned int kid = 0; kid < kn; kid++) {
		const unsigned int nrows_blurred_octave1 = nrows * 2, ncols_blurred_octave1 = ncols * 2;
		const unsigned int nrows_blurred_octave2 = nrows, ncols_blurred_octave2 = ncols;
		const unsigned int nrows_blurred_octave3 = nrows / 2, ncols_blurred_octave3 = ncols / 2;
		const unsigned int nrows_blurred_octave4 = nrows / 4, ncols_blurred_octave4 = ncols / 4;

		float *blurred_octave1_k = (float*)malloc(nrows_blurred_octave1 * ncols_blurred_octave1 * sizeof(float));
		float *blurred_octave2_k = (float*)malloc(nrows_blurred_octave2 * ncols_blurred_octave2 * sizeof(float));
		float *blurred_octave3_k = (float*)malloc(nrows_blurred_octave3 * ncols_blurred_octave3 * sizeof(float));
		float *blurred_octave4_k = (float*)malloc(nrows_blurred_octave4 * ncols_blurred_octave4 * sizeof(float));
		memcpy(blurred_octave1_k, &blurred_octave1[kid * nrows_blurred_octave1 * ncols_blurred_octave1], nrows_blurred_octave1 * ncols_blurred_octave1 * sizeof(float));
		memcpy(blurred_octave2_k, &blurred_octave2[kid * nrows_blurred_octave2 * ncols_blurred_octave2], nrows_blurred_octave2 * ncols_blurred_octave2 * sizeof(float));
		memcpy(blurred_octave3_k, &blurred_octave3[kid * nrows_blurred_octave3 * ncols_blurred_octave3], nrows_blurred_octave3 * ncols_blurred_octave3 * sizeof(float));
		memcpy(blurred_octave4_k, &blurred_octave4[kid * nrows_blurred_octave4 * ncols_blurred_octave4], nrows_blurred_octave4 * ncols_blurred_octave4 * sizeof(float));
		Mat blurred_octave1_img(nrows_blurred_octave1, ncols_blurred_octave1, CV_32FC1, blurred_octave1_k);
		Mat blurred_octave2_img(nrows_blurred_octave2, ncols_blurred_octave2, CV_32FC1, blurred_octave2_k);
		Mat blurred_octave3_img(nrows_blurred_octave3, ncols_blurred_octave3, CV_32FC1, blurred_octave3_k);
		Mat blurred_octave4_img(nrows_blurred_octave4, ncols_blurred_octave4, CV_32FC1, blurred_octave4_k);

		char path1[100] = "", path2[100] = "", path3[100] = "", path4[100] = "";
		sprintf(path1, "C:/Users/c2lo/Desktop/blurred_octave1_%d_%d_img.jpg", kid, img_id);
		sprintf(path2, "C:/Users/c2lo/Desktop/blurred_octave2_%d_%d_img.jpg", kid, img_id);
		sprintf(path3, "C:/Users/c2lo/Desktop/blurred_octave3_%d_%d_img.jpg", kid, img_id);
		sprintf(path4, "C:/Users/c2lo/Desktop/blurred_octave4_%d_%d_img.jpg", kid, img_id);
		imwrite(path1, blurred_octave1_img); imwrite(path2, blurred_octave2_img);
		imwrite(path3, blurred_octave3_img); imwrite(path4, blurred_octave4_img);
	}
}

void computeDogOctave(float *dog_octave, float *blurred_octave, const unsigned int nrows, const unsigned int ncols, const unsigned int kn, const float scale) {
	// cudaMalloc gpu blurred octave and gpu dog octave
	float *d_dog_octave, *d_blurred_octave;
	const unsigned int nrows_blurred_octave = nrows * scale, ncols_blurred_octave = ncols * scale;
	cudaMalloc((void**) &d_dog_octave, (kn - 1) * nrows_blurred_octave * ncols_blurred_octave * sizeof(float));
	cudaMalloc((void**) &d_blurred_octave, kn * nrows_blurred_octave * ncols_blurred_octave * sizeof(float));

	// memcpy cpu blurredd octave to device
	cudaMemcpy(d_blurred_octave, blurred_octave, kn * nrows_blurred_octave * ncols_blurred_octave * sizeof(float), cudaMemcpyHostToDevice);
	
	// compute difference of gaussian
	const unsigned int nx = ncols * scale, ny = nrows * scale, dimx = 32, dimy = 32;
	dim3 block(dimx, dimy);
	dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
	gpuComputeDogOctave << <grid, block >> > (d_dog_octave, d_blurred_octave, nrows, ncols, kn, scale);
	cudaDeviceSynchronize();

	// cudaMemcpy gpu dog octave host
	cudaMemcpy(dog_octave, d_dog_octave, (kn-1) * nrows_blurred_octave * ncols_blurred_octave * sizeof(float), cudaMemcpyDeviceToHost);

	// cudaFree
	cudaFree(d_dog_octave); cudaFree(d_blurred_octave); 
}

void saveDogOctaves(float *dog_octave1, float *dog_octave2, float *dog_octave3, float *dog_octave4, const unsigned int nrows, const unsigned int ncols, const unsigned int kn, const unsigned int img_id) {
	printf("saving dog octaves of image %d\n", img_id);
	for (unsigned int kid = 0; kid < kn-1; kid++) {
		const unsigned int nrows_dog_octave1 = nrows * 2, ncols_dog_octave1 = ncols * 2;
		const unsigned int nrows_dog_octave2 = nrows, ncols_dog_octave2 = ncols;
		const unsigned int nrows_dog_octave3 = nrows / 2, ncols_dog_octave3 = ncols / 2;
		const unsigned int nrows_dog_octave4 = nrows / 4, ncols_dog_octave4 = ncols / 4;

		float *dog_octave1_k = (float*)malloc(nrows_dog_octave1 * ncols_dog_octave1 * sizeof(float));
		float *dog_octave2_k = (float*)malloc(nrows_dog_octave2 * ncols_dog_octave2 * sizeof(float));
		float *dog_octave3_k = (float*)malloc(nrows_dog_octave3 * ncols_dog_octave3 * sizeof(float));
		float *dog_octave4_k = (float*)malloc(nrows_dog_octave4 * ncols_dog_octave4 * sizeof(float));
		memcpy(dog_octave1_k, &dog_octave1[kid * nrows_dog_octave1 * ncols_dog_octave1], nrows_dog_octave1 * ncols_dog_octave1 * sizeof(float));
		memcpy(dog_octave2_k, &dog_octave2[kid * nrows_dog_octave2 * ncols_dog_octave2], nrows_dog_octave2 * ncols_dog_octave2 * sizeof(float));
		memcpy(dog_octave3_k, &dog_octave3[kid * nrows_dog_octave3 * ncols_dog_octave3], nrows_dog_octave3 * ncols_dog_octave3 * sizeof(float));
		memcpy(dog_octave4_k, &dog_octave4[kid * nrows_dog_octave4 * ncols_dog_octave4], nrows_dog_octave4 * ncols_dog_octave4 * sizeof(float));
		Mat dog_octave1_img(nrows_dog_octave1, ncols_dog_octave1, CV_32FC1, dog_octave1_k);
		Mat dog_octave2_img(nrows_dog_octave2, ncols_dog_octave2, CV_32FC1, dog_octave2_k);
		Mat dog_octave3_img(nrows_dog_octave3, ncols_dog_octave3, CV_32FC1, dog_octave3_k);
		Mat dog_octave4_img(nrows_dog_octave4, ncols_dog_octave4, CV_32FC1, dog_octave4_k);

		char path1[100] = "", path2[100] = "", path3[100] = "", path4[100] = "";
		sprintf(path1, "C:/Users/c2lo/Desktop/dog_octave1_%d_%d_img.jpg", kid, img_id);
		sprintf(path2, "C:/Users/c2lo/Desktop/dog_octave2_%d_%d_img.jpg", kid, img_id);
		sprintf(path3, "C:/Users/c2lo/Desktop/dog_octave3_%d_%d_img.jpg", kid, img_id);
		sprintf(path4, "C:/Users/c2lo/Desktop/dog_octave4_%d_%d_img.jpg", kid, img_id);
		imwrite(path1, dog_octave1_img); imwrite(path2, dog_octave2_img);
		imwrite(path3, dog_octave3_img); imwrite(path4, dog_octave4_img);
	}
}

void findKeypointsOctave(unsigned short *keypoints_octave, float *dog_octave, const unsigned int nrows, const unsigned int ncols, const unsigned int kn, const float scale) {
	// cudaMalloc gpu keypoints octave and gpu dog octave
	unsigned short *d_keypoints_octave;
	float *d_dog_octave;
	const unsigned int nrows_dog_octave = nrows * scale, ncols_dog_octave = ncols * scale;
	cudaMalloc((void**) &d_keypoints_octave, (kn - 3) * nrows_dog_octave * ncols_dog_octave * sizeof(unsigned short));
	cudaMalloc((void**) &d_dog_octave, (kn - 1) * nrows_dog_octave * ncols_dog_octave * sizeof(float));

	// memcpy cpu dog octave to device
	cudaMemcpyAsync(d_dog_octave, dog_octave, (kn - 1) * nrows_dog_octave * ncols_dog_octave * sizeof(float), cudaMemcpyHostToDevice);

	// find keypoints
	const unsigned int nx = ncols * scale, ny = nrows * scale, dimx = 32, dimy = 32;
	dim3 block(dimx, dimy);
	dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
	printf("nx = %d, ny = %d\n", nx, ny);
	gpuFindKeypointsOctave << <grid, block >> > (d_keypoints_octave, d_dog_octave, nrows, ncols, kn, scale);
	cudaDeviceSynchronize();

	// cudaMemcpy gpu keypoints octave to host
	cudaMemcpyAsync(keypoints_octave, d_keypoints_octave, (kn - 3) *  nrows_dog_octave * ncols_dog_octave * sizeof(unsigned short), cudaMemcpyDeviceToHost);

	// cudaFree
	cudaFree(d_keypoints_octave); cudaFree(d_dog_octave);
}