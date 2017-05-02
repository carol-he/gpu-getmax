#include <stdio.h>
#include <stdlib.h>
#include <time.h>

unsigned int getmax(unsigned int *, unsigned int);

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

    size = atol(argv[1]);

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
    (unsigned int *) d_numbers;
    //transfer array to device memory
    cudaMalloc((void**) &d_numbers, size * sizeof(unsigned int));
    cudaMemcpy(d_numbers, numbers, size * sizeof(unsigned int)), cudaMemcpyHostToDevice);
    printf(" The maximum number in the array is: %u\n", getmax(numbers, size));
    get_max<<<1, 512>>>(d_numbers, size * sizeof(unsigned int));
    cudaMemcpy(numbers, d_numbers, size * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    for( i = 0; i < size; i++){
      printf("element in %d: %u", );
    }
    //free device matrices
    cudaFree(d_numbers);
    free(numbers);
    exit(0);
}

__global__ void get_max(unsigned int* d_numbers, int size){
  for
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
