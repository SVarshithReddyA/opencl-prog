// OpenCL kernel for vector addition.
__kernel
void vectorAdd( __global float *a, __global float *b, __global float *c )
{
	// The global id tells us the index of the vector for this thread.
	int gid = get_global_id(0);
	
	// Perform the addition.
	c[gid] = a[gid] + b[gid];
}
