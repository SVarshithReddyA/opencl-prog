//
// Example of how a register overflow can affect performance.
//
// Compile as usual, as per the instructions given in lectures.
//



//
// Includes.
//
#include <stdio.h>
#include <stdlib.h>
#include "helper.h"


//
// Problem size is L*L. Try to make quite large so times are measurable.
//
#define L 1024


//
// Main.
//
int main()
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

	// Create a static 2D array on the host, and initialise.
	float *host_array = (float*) malloc( L*L*sizeof(float) );

	int i;
	for( i=0; i<L*L; i++ ) host_array[i] = i+1;

	// Create a corresponding array on the device global memory, and copy from the host.
	cl_mem device_array = clCreateBuffer( context, CL_MEM_COPY_HOST_PTR, L*L*sizeof(float), host_array, &status );
	if( status != CL_SUCCESS )
	{
		printf( "Could not create device array: Error %d.\n", status );
		return EXIT_FAILURE;
	}

	// Compile the kernel from file. compileKernelFromFile() already includes some error handling.
	cl_kernel kernel = compileKernelFromFile( "registerOverflow.cl", "doSomethingComplex", context, device );


	//
	// Execute the kernel.
	//

	// Set the kernel arguments.
	int size = L;
	clSetKernelArg( kernel, 0, sizeof(cl_mem), &device_array );
	clSetKernelArg( kernel, 1, sizeof(int   ), &size         );

	// Set up the global problem size, and the work group size. Each workGroupSize[i] must divide indexSpaceSize[i] exactly.
	size_t indexSpaceSize[2] = {L,L}, workGroupSize[2] = {16,16};

	// The timer.
	cl_event timer;

	// Put the kernel onto the command queue.
	status = clEnqueueNDRangeKernel( queue, kernel, 2, NULL, indexSpaceSize, workGroupSize, 0, NULL, &timer );
	if( status != CL_SUCCESS )
	{
		printf( "Failure enqueuing kernel: Error %d.\n", status );
		return EXIT_FAILURE;
	}

	//
	// Copy the data back to the host, and display a few items for checking purposes.
	//
	status = clEnqueueReadBuffer( queue, device_array, CL_TRUE, 0, L*L*sizeof(float), host_array, 0, NULL, NULL );
	if( status != CL_SUCCESS )
	{
		printf( "Could not copy device data to host: Error %d.\n", status );
		return EXIT_FAILURE;
	}

	printf( "First few items are:\n" );
	for( i=0; i<10; i++ )
		printf( "i=%d\tarray[i]=%g\n", i, host_array[i] );

	// Display the time taken for just the kernel. Do not do this until after the kernel has definitely finished.
	cl_ulong start, end;
	clGetEventProfilingInfo( timer, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL );
	clGetEventProfilingInfo( timer, CL_PROFILING_COMMAND_END  , sizeof(cl_ulong), &end  , NULL );
	printf( "Time taken for kernel: %g ms\n", 1e-6*(cl_double)( end - start ) );

	//
	// Clear up.
	//
	free( host_array );

	clReleaseMemObject( device_array );

	clReleaseEvent       ( timer   );
	clReleaseKernel      ( kernel  );
	clReleaseCommandQueue( queue   );
	clReleaseContext     ( context );
}


