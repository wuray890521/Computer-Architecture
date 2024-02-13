// cuda1.cu

#include "cuda_runtime.h"

__global__ void grayscaleConversion(unsigned char *inputImage, unsigned char *outputImage, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;


    //allocate each pixel to threads 
    if (x < width && y < height) {
        int tid = y * width + x;
        unsigned char r = inputImage[3 * tid];
        unsigned char g = inputImage[3 * tid + 1];
        unsigned char b = inputImage[3 * tid + 2];
        outputImage[tid] = 0.299f * r + 0.587f * g + 0.114f * b;
    }
}




float GPU_kernel(unsigned char *outputImage_GPU,unsigned char *inputImage){

	

        int width = 32;
        int height = 32;
        size_t imageSize = width * height * 3 * sizeof(unsigned char);

 

	// Creat Timing Event
  	cudaEvent_t start, stop;
	cudaEventCreate (&start);
	cudaEventCreate (&stop); 	

   
        //allocate GPU space
        unsigned char *d_inputImage, *d_outputImage_GPU;
        cudaMalloc((void **)&d_inputImage, imageSize);
        cudaMalloc((void **)&d_outputImage_GPU, width * height * sizeof(unsigned char));


        // CPU to GPU
        cudaMemcpy(d_inputImage, inputImage, imageSize, cudaMemcpyHostToDevice);

       // Start Timer
       cudaEventRecord(start, 0);

        //define CUDA block and grid dimension
        dim3 blockDim(4, 4);
        dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

        //Use GPU to count
        grayscaleConversion<<<gridDim, blockDim>>>(d_inputImage, d_outputImage_GPU, width, height);

	// Stop Timer
	cudaEventRecord(stop, 0);
  	cudaEventSynchronize(stop); 


        //GPU to CPU
        cudaMemcpy(outputImage_GPU, d_outputImage_GPU, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);


	// Release Memory Space on Device
	cudaFree(d_inputImage);
	cudaFree(d_outputImage_GPU);
	//cudaFree(dInd);

	// Calculate Elapsed Time
  	float elapsedTime; 
  	cudaEventElapsedTime(&elapsedTime, start, stop);  

	return elapsedTime;//retuen GPU time
}
