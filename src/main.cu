// main.c

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
//#include "cuda1.cu"
#include "cuda2.cu"


//CPU compute
void CPU(unsigned char *inputImage,unsigned char *outputImage_CPU, int width, int height)
{
    int i;
    for (int i = 0; i < width * height; i++) {
		*(outputImage_CPU+i)=inputImage[3 * i]*0.299+inputImage[3 * i + 1]*0.587+inputImage[3 * i + 2]*0.114;
    }
}




void generateRandomImage(unsigned char *inputImage, int width, int height)
{
    srand(time(NULL));

    //initialize RGB for each pixel
    for (int i = 0; i < width * height; i++) {
        inputImage[3 * i] = rand() % 256;    // red
        inputImage[3 * i + 1] = rand() % 256; // green
        inputImage[3 * i + 2] = rand() % 256; // blue
    }
}

void printImage(unsigned char *outputImage, int width, int height)
{

    //print all graph
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            printf("%d ", outputImage[i * width + j]);
        }
        printf("\n");
    }
}



int main()
{
    int width = 32;
    int height = 32;
    size_t imageSize = width * height * 3 * sizeof(unsigned char);

    // Creat Timing Event
    cudaEvent_t start, stop;
    cudaEventCreate (&start);
    cudaEventCreate (&stop); 




    // allocate CPU space and we use single pointer to do double matrix
    unsigned char *inputImage = (unsigned char *)malloc(imageSize);
    unsigned char *outputImage_GPU = (unsigned char *)malloc(width * height * sizeof(unsigned char));
    unsigned char *outputImage_CPU = (unsigned char *)malloc(width * height * sizeof(unsigned char));

    //take random seed
    srand(time(NULL));

    //generate graph 
    generateRandomImage(inputImage, width, height);

    //print input graph
    printf("Input Input Input Input Input Input Input Input Input Input Input Input Input Input Input Input Input \n");
    printf("===============================================================================================================\n");
    printImage(inputImage, width, height);
    printf("===============================================================================================================\n");



    printf(" ");

    //CPU compute
    CPU(inputImage,outputImage_CPU, width, height);


    //GPU side
    float elapsedTime = GPU_kernel(outputImage_GPU,inputImage);





    //print GPU output graph
    printf("GPU GPU GPU GPU GPU GPU GPU GPU GPU GPU GPU GPU GPU GPU GPU GPU GPU GPU GPU GPU GPU GPU GPU GPU GPU GPU GPU \n");
    printf("===============================================================================================================\n");
    printImage(outputImage_GPU, width, height);
    printf("===============================================================================================================\n");


    //print CPU output graph
    printf("CPU CPU CPU CPU CPU CPU CPU CPU CPU CPU CPU CPU CPU CPU CPU CPU CPU CPU CPU CPU CPU CPU CPU CPU CPU CPU CPU \n");
    printf("===============================================================================================================\n");
    printImage(outputImage_CPU, width, height);
    printf("===============================================================================================================\n");


 
    printf("GPU time = %5.2f ms\n", elapsedTime);



    // relax CPU
    free(inputImage);
    free(outputImage_GPU);

    return 0;
}
