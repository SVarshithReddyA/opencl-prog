// Kernel for the register overflow example. The question asks you to change this value.
#define N 32

__kernel
void doSomethingComplex( __global float *array, int L )
{
	int
		i = get_global_id(0),
		j = get_global_id(1),
		k;
	
	// Create a table of sine values in private memory.
	float temp[N];
	for( k=0; k<N; k++ ) temp[k] = sin(0.1f*k);
	
	// Assign array value to somewhere in this table that depends on the work item id
	// (note this means that the compiler cannot 'short cut' these calculations).
	array[i*L+j] = temp[ (i*L+j)%N ];
}
