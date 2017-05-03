#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define THREADS_PER_BLOCK 512

//function declarations
unsigned int getmax(unsigned int *, unsigned int);
__global__ void get_max(unsigned int *num);

int main(int argc, char *argv[])
{
    unsigned int size = 0;  // The size of the array
    unsigned int i;  // loop index
    unsigned int * numbers; //pointer to the array

    if(argc !=2)
    {
       printf("usage: maxseq num\n");
       printf("num = size of the array\n");
       exit(1);
    }
    //size is number of threads total
    size = atol(argv[1]);

    //calculates number of blocks
    unsigned int NUM_BLOCKS = size/THREADS_PER_BLOCK;

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
    printf(" The maximum number in the array is: %u\n", getmax(numbers, size));
    //parallel
    get_max<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_numbers);
    cudaMemcpy(numbers, d_numbers, size * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    for( i = 0; i < size; i++){
      printf("element in %d: %u\n", i, d_numbers[i]);
    }
     printf("The max integer in the array is: %d\n", d_numbers[0]);
    //free device matrices
    cudaFree(d_numbers);
    free(numbers);
    exit(0);
}

__global__ void get_max(unsigned int* num){
  unsigned int temp;
  unsigned int index = threadIdx.x + (blockDim.x * blockIdx.x);
  unsigned int nTotalThreads = 10;

  while(nTotalThreads > 1){
    unsigned int halfPoint = nTotalThreads / 2;	// divide by two
    // only the first half of the threads will be active.
    if (index < halfPoint){
      temp = num[ index + halfPoint ];
      if (temp > num[ index ]) {
        num[index] = temp;
      }
    }
    __syncthreads();


    nTotalThreads = nTotalThreads / 2;	// divide by two.
  }
}

/*
   input: pointer to an array of long int
          number of elements in the array
   output: the maximum number of the array
*/
unsigned int getmax(unsigned int num[], unsigned int size)
{
  unsigned int i;
  unsigned int max = num[0];

  for(i = 1; i < size; i++)
	if(num[i] > max)
	   max = num[i];

  return( max );

}
