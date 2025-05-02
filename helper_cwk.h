//
// This is a modified version of helper.h that has additional routines specific to the coursework.
// As with previous courseworks, this will be replaced with a different version of helper_cwk.h for
// assessment; therefore:
// - Do NOT change this file (any such changes will be lost during assessment).
// - Do NOT copy any of this file into you code.
// - Do NOT reproduce any of this file in your own code.
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
#include <time.h>



//
// Gets the command line argument, and performs some basic error checking.
//
void getCmdLineArg( int argc, char **argv, int *N )
{
    if( argc != 2 )
    {
        printf( "Requires exactly 1 command line argument: The size of the square grid.\n" );
        exit( EXIT_FAILURE );
    }

	*N = atoi(argv[1]);

    if( N<=0 )
    {
        printf( "Grid size must be positive.\n" );
        exit( EXIT_FAILURE );
    }
}


//
// Fills the square grid with some initial random values, but zero on the boundary.
//
void fillGrid( float *grid, int N )
{
	int row, col;

	srand( time(0) );

	for( row=0; row<N; row++ )
		for( col=0; col<N; col++ )
			if( row==0 || row==N-1 || col==0 || col==N-1 )
				grid[row*N+col] = 0.0;
			else
				grid[row*N+col] = 1.0 * rand() / RAND_MAX;
}

//
// Displays the grid, or part of it if the matrix is too large for a typical shell window.
//
void displayGrid( float *grid, int N )
{
    int i, j;

    // Display the grid. Only display the top-left corner if large, with ellipses '...' to suggest continuation.
    for( i=0; i<(N<16?N:16); i++ )
    {
        for( j=0; j<(N<16?N:16); j++ ) printf( "%.3f ", grid[i*N+j] );
        if( N>16 ) printf( ". . ." );
        printf( "\n" );
    }

    // Ellipses at the bottom of the grid.
    if( N>16 )
        for( i=0; i<3; i ++ )
        {
            for( j=0; j<(N<16?N:16); j++ ) printf( "  .   " );
            printf( "\n" );
        }

    printf( "\n" );
}


//
//	Tries to open up a single GPU on the first OpenCL framework, returning the context	
//	and filling the passed device i.d.
//	
//	Fails with a brief error message and calls exit(EXIT_FAILURE) if there was some problem.
// 
cl_context simpleOpenContext_GPU( cl_device_id *device )
{
	// Status; returned/modified after each API call; zero if successful.
	cl_int status;

	// Get the first platform; display status (zero if successful).
	cl_platform_id platform;
	status = clGetPlatformIDs( 1, &platform, NULL );
	if( status != CL_SUCCESS )
	{
		printf( "Could not open an OpenCL platform.\n" );				// Simple error message.
		exit( EXIT_FAILURE );											// Defined in stdlib.h to be something non-zero.
	}

	// Get the first GPU for this platform.
	cl_uint numGPUs;
	status = clGetDeviceIDs( platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numGPUs );
	if( numGPUs==0 )
	{
		printf( "No OpenCL-compatible GPUs on this platform; cannot continue.\n" );
		exit( EXIT_FAILURE );
	}

	cl_device_id *GPUIDs = (cl_device_id*) malloc( numGPUs*sizeof(cl_device_id) );
	status = clGetDeviceIDs( platform, CL_DEVICE_TYPE_GPU, numGPUs, GPUIDs, NULL );
	if( status != CL_SUCCESS )
	{
		printf( "Failed to get a viable GPU ID.\n" );
		exit( EXIT_FAILURE );
	}
	*device = GPUIDs[0];			// Use the first one.
	free( GPUIDs );
	
	// Create a context and associate it with this device.
	cl_context context = clCreateContext( NULL, 1, device, NULL, NULL, &status );
	
	return context;
}


//
//	Attempts to load and compile an OpenCL kernel with the given filename; also need a name,
//	the context and the device.
//
//  Prints a brief error message and calls exit( EXIT_FAILURE ) if there was some problem.
//
cl_kernel compileKernelFromFile( const char *filename, const char *kernelName, cl_context context, cl_device_id device )
{
	FILE *fp;
	char *fileData;
	long fileSize;
	
	// Open the file.
	fp = fopen( filename, "r" );
	if( !fp )
	{
		printf( "Could not open the file '%s'\n", filename );
		exit( EXIT_FAILURE );
	}
	
	// Get the file size; also check it is positive.
	if( fseek(fp,0,SEEK_END) )
	{
		printf( "Could not extract the file size from '%s'.\n", filename );
		exit( EXIT_FAILURE );
	}
	fileSize = ftell(fp);
	if( fileSize<1 )
	{
		printf( "Could not read file (or zero size) for file '%s'.\n", filename );
		exit( EXIT_FAILURE );
	}
	
	// Move to the start of the file.
	if( fseek(fp,0,SEEK_SET) )
	{
		printf( "Error reading the file '%s' (could not move to start).\n", filename );
		exit( EXIT_FAILURE );
	}
	
	// Read the contents; also need a termination character.
	fileData = (char*) malloc( fileSize+1 );
	if( !fileData )
	{
		printf( "Could not allocate memory for the character buffer.\n" );
		exit( EXIT_FAILURE );
	}
	if( fread(fileData,fileSize,1,fp) != 1 )
	{
		printf( "Error reading the file '%s'.\n", filename );
		exit( EXIT_FAILURE );
	}
	
	// Terminate the string.
	fileData[fileSize] = '\0';
	
	// Close the file.
	if( fclose(fp) )
	{
		printf( "Error closing the file '%s'.\n", filename );
		exit( EXIT_FAILURE );
	}
	
	// Now for the OpenCL-specific stuff. Create the program from the character string.
	cl_int status;
	cl_program program = clCreateProgramWithSource( context, 1, (const char**)&fileData, NULL, &status );
	if( status != CL_SUCCESS )
	{
		printf( "Failed to create program from the source '%s'.n", filename );
		exit( EXIT_FAILURE );
	}

	// Build the program.
	if( (status=clBuildProgram(program,1,&device,NULL,NULL,NULL)) != CL_SUCCESS )
	{
		printf( "Failed to build the kernel '%s' from the file '%s'; error code %i.\n", kernelName, filename, status );
		
		// Provide more information about the nature of the fail.
		size_t logSize;
		clGetProgramBuildInfo( program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize );
		char *log = (char*) malloc( (logSize+1)*sizeof(char) );
		
		clGetProgramBuildInfo( program, device, CL_PROGRAM_BUILD_LOG, logSize+1, log, NULL );
		printf( "Build log:\n%s\n", log );
		
		// Clear up and quit.
		free( log );
		exit( EXIT_FAILURE );
	}
	
	// Now compile it.
	cl_kernel kernel = clCreateKernel( program, kernelName, &status );
	if( status != CL_SUCCESS )
	{
		printf( "Failed to create the OpenCL kernel with error code %i.\n", status );
		
		// Common mistake(s).
		if( status==-46 ) printf( "Ensure the kernel name '%s' is also the name of the function.\n", kernelName );

		exit( EXIT_FAILURE );
	}
	
	// Clear up (not the kernel, which will have to be released by the caller).
	clReleaseProgram( program );
	free( fileData );
	
	return kernel;
}

