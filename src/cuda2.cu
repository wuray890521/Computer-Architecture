// cuda2.cu

#include "cuda_runtime.h"

__global__ void grayscaleConversion(unsigned char *inputImage, unsigned char *outputImage, int width, int height)
{
    int tid = (blockIdx.x * blockDim.x + threadIdx.x) * 3; //done 4 pixel per time
    int stride = blockDim.x * gridDim.x * 3; //move to next
    while (tid < width * height * 3)
    {
        int pixelIndex = tid / 3;
        int channel = tid % 3;

        int i = pixelIndex / width;
        int j = pixelIndex % width;

        unsigned char r = inputImage[3 * pixelIndex];
        unsigned char g = inputImage[3 * pixelIndex + 1];
        unsigned char b = inputImage[3 * pixelIndex + 2];

        //execute from (i, j) 
        outputImage[pixelIndex] = 0.299f * r + 0.587f * g + 0.114f * b;

        int halfHeight = height / 2;
        int halfWidth = width / 2;

        if (i < halfHeight && j < halfWidth)
        {
            int tid2 = ((halfHeight + i) * width + (halfWidth + j)) * 3;
            unsigned char r2 = inputImage[tid2];
            unsigned char g2 = inputImage[tid2 + 1];
            unsigned char b2 = inputImage[tid2 + 2];

            // execute from ((height/2)+i, (width/2)+j) 
            outputImage[tid2 / 3] = 0.299f * r2 + 0.587f * g2 + 0.114f * b2;

            int tid3 = ((halfHeight + i) * width + j) * 3;
            unsigned char r3 = inputImage[tid3];
            unsigned char g3 = inputImage[tid3 + 1];
            unsigned char b3 = inputImage[tid3 + 2];

            //execute from  ((height/2)+i, j) 
            outputImage[tid3 / 3] = 0.299f * r3 + 0.587f * g3 + 0.114f * b3;

            int tid4 = (i * width + (halfWidth + j)) * 3;
            unsigned char r4 = inputImage[tid4];
            unsigned char g4 = inputImage[tid4 + 1];
            unsigned char b4 = inputImage[tid4 + 2];

            // execute from  (i, (width/2)+j) 
            outputImage[tid4 / 3] = 0.299f * r4 + 0.587f * g4 + 0.114f * b4;
        }

        tid += stride; // reset next coordinate directory
    }
}


float GPU_kernel(unsigned char *outputImage_GPU,unsigned char *inputImage){

	

        int width = 32;
        int height = 32;
        size_t imageSize = width * height * 3 * sizeof(unsigned char);

        //int *dA,*dB;
	//IndexSave* dInd;

	// Creat Timing Event
  	cudaEvent_t start, stop;
	cudaEventCreate (&start);
	cudaEventCreate (&stop); 	

   
        //allocate GPU space
        unsigned char *d_inputImage, *d_outputImage_GPU;
        cudaMalloc((void **)&d_inputImage, imageSize);
        cudaMalloc((void **)&d_outputImage_GPU, width * height * sizeof(unsigned char));


	// Copy Data to be Calculated

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

	return elapsedTime; //retuen GPU time
}
