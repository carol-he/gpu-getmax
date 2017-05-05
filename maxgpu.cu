#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define THREADS_PER_BLOCK 512

//function declarations
unsigned int getmax(unsigned int *, unsigned int);
__global__ void getmaxcu(unsigned int *num, unsigned int size);

int main(int argc, char *argv[])
{
    unsigned int size = 0;  // The size of the array
    unsigned int i;  // loop index
    unsigned int * numbers; //pointer to the array
    //size is number of threads total
    size = atol(argv[1]);

    //calculates number of blocks
    unsigned int NUM_BLOCKS = (size + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;

    numbers = (unsigned int *)malloc(size * sizeof(unsigned int));
    if( !numbers )
    {
       printf("Unable to allocate mem for an array of size %u\n", size);
       exit(1);
    }
    srand(time(NULL)); // setting a seed for the random number generator
    // Fill-up the array with random numbers from 0 to size-1
    for( i = 0; i < size; i++){
      numbers[i] = rand()  % size;
    }
    //create device pointers
    unsigned int *d_numbers;
    //transfer array to device memory
    cudaMalloc((void**) &d_numbers, size * sizeof(unsigned int));
    cudaMemcpy(d_numbers, numbers, size * sizeof(unsigned int), cudaMemcpyHostToDevice);
    //sequential
    //printf(" The maximum number in the array is: %u\n", getmax(numbers, size));
    //parallel kernel call
    unsigned int sizea = size;
    while(sizea > 1){
      getmaxcu<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_numbers, sizea);
      sizea = (sizea) / 10;
    }
    cudaMemcpy(numbers, d_numbers, size * sizeof(unsigned int), cudaMemcpyDeviceToHost);
     printf("The max integer in the array is: %d\n", numbers[0]);
    //free device matrices
    cudaFree(d_numbers);
    free(numbers);
    exit(0);
}

__global__ void getmaxcu(unsigned int* num, unsigned int size){
  unsigned int temp;
  unsigned int index = threadIdx.x + (blockDim.x * blockIdx.x);
  unsigned int nTotalThreads = size;
  unsigned int i;
    unsigned int tenPoint = nTotalThreads / 10;	// divide by ten
    if(index < tenPoint){
      for(i = 1; i < 10; i++){
        temp = num[index + tenPoint*i];
        //compare to "0" index
        if(temp > num[index]){
          num[index] = temp;
        }
      }
    }
}

unsigned int getmax(unsigned int num[], unsigned int size)
{
  unsigned int i;
  unsigned int max = num[0];

  for(i = 1; i < size; i++)
	if(num[i] > max)
	   max = num[i];

  return( max );

}
