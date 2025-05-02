//
// Vector addition in OpenCL.
//
// Compile as per the instructions given in lectures.
//
// Execute as normal:
// ./vectorAddition
//


//
// Includes.
//

// Need OpenCL; how to include depends on the OS.
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#define CL_TARGET_OPENCL_VERSION 200
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#endif

// Other libraries needed here.
#include <stdio.h>
#include <stdlib.h>

// This file contains routines that help initialising OpenCL, and loading kernels from file.
#include "helper.h"


//
// Parameters.
//

// The size of the vectors.
#define N 1024


//
// Main.
//
int main( int argc, char **argv )
{
	//
	// Initialisation.
	//

	// Set up OpenCL using the routine in helper.h (which returns the first OpenCL-compliant GPU found).
	cl_device_id device;
	cl_context context = simpleOpenContext_GPU(&device);

	// Open up a single command queue, with the profiling option on.
	cl_int status;
	cl_command_queue queue = clCreateCommandQueue( context, device, CL_QUEUE_PROFILING_ENABLE, &status );

	// Allocate memory for the vector arrays on the host (i.e. the CPU); this is standard C.
	float
		*host_a = (float*) malloc( N*sizeof(float) ),
		*host_b = (float*) malloc( N*sizeof(float) ),
		*host_c = (float*) malloc( N*sizeof(float) );

	// Initialise a and b on the host. Anything will do here for this demonstration.
	int i;
	for( i=0; i<N; i++ ) { host_a[i] = i+1; host_b[i] = 2*i; }

	// Allocate memory for the arrays on the device, using OpenCL routines. Note that the 'CL_MEM_COPY_HOST_PTR' option
	// automatically copies host_a to device_a, and host_b to device_b.
	// If successful, status==CL_SUCCESS after each call.
	cl_mem device_a = clCreateBuffer( context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N*sizeof(float), host_a, &status );
	cl_mem device_b = clCreateBuffer( context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N*sizeof(float), host_b, &status );
	cl_mem device_c = clCreateBuffer( context, CL_MEM_WRITE_ONLY                      , N*sizeof(float), NULL  , &status );


	//
	// Perform the calculation on the GPU.
	//

	// Build the kernel code 'vectorAdd' contained in the file 'vectorAddition.cl'. This uses a routine in helper.h.
	cl_kernel kernel = compileKernelFromFile( "vectorAddition.cl", "vectorAdd", context, device );

	// Specify the arguments to the kernel.
	status = clSetKernelArg( kernel, 0, sizeof(cl_mem), &device_a );
	status = clSetKernelArg( kernel, 1, sizeof(cl_mem), &device_b );
	status = clSetKernelArg( kernel, 2, sizeof(cl_mem), &device_c );

	// Set up the global problem size, and the work group size.
	size_t indexSpaceSize[1], workGroupSize[1];
	indexSpaceSize[0] = N;
	workGroupSize [0] = 128;				// Should match to hardware; can be too large!

	// Put the kernel onto the command queue.
	status = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, indexSpaceSize, workGroupSize, 0, NULL, NULL );
	if( status != CL_SUCCESS )
	{
		printf( "Failure enqueuing kernel: Error %d.\n", status );
		return EXIT_FAILURE;
	}


	//
	// Get the result back from the device to the host, and check.
	//
	status = clEnqueueReadBuffer( queue, device_c, CL_TRUE, 0, N*sizeof(float), host_c, 0, NULL, NULL );
	if( status != CL_SUCCESS )
	{
		printf( "Could not copy device data to host: Error %d.\n", status );
		return EXIT_FAILURE;
	}

	printf( "Checking (will only display first few elements):\n" );
	for( i=0; i<N; i++ )
	{
		// Only print a few values.
		if( i<10 ) printf( "%g + %g = %g.\n", host_a[i], host_b[i], host_c[i] );
		if( host_a[i] + host_b[i] != host_c[i] )
		{
			printf( "Vector addition FAILED.\n" );
			break;
		}
	}


	//
	// Clear up.
	//
	clReleaseMemObject( device_a );
	clReleaseMemObject( device_b );
	clReleaseMemObject( device_c );

	free( host_a );
	free( host_b );
	free( host_c );

	clReleaseKernel      ( kernel  );
	clReleaseCommandQueue( queue   );
	clReleaseContext     ( context );
}


