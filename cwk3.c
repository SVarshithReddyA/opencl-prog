//
// Starting point for the OpenCL coursework for COMP/XJCO3221 Parallel Computation.
//

#include <stdio.h>
#include <stdlib.h>
#include "helper_cwk.h"

int main(int argc, char **argv) {
    int N;
    getCmdLineArg(argc, argv, &N);

    // Initialize OpenCL context and queue.
    cl_device_id device;
    cl_context context = simpleOpenContext_GPU(&device);
    cl_int status;
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &status);

    // Allocate and initialize the host grid.
    float *hostGrid = (float *)malloc(N * N * sizeof(float));
    fillGrid(hostGrid, N);
    printf("Original grid (only top-left shown if too large):\n");
    displayGrid(hostGrid, N);

    // Allocate device memory for input and output grids.
    cl_mem inputGrid = clCreateBuffer(context, CL_MEM_READ_ONLY, N * N * sizeof(float), NULL, &status);
    cl_mem outputGrid = clCreateBuffer(context, CL_MEM_WRITE_ONLY, N * N * sizeof(float), NULL, &status);

    // Write the host grid to the device memory.
    status = clEnqueueWriteBuffer(queue, inputGrid, CL_TRUE, 0, N * N * sizeof(float), hostGrid, 0, NULL, NULL);
    if (status != CL_SUCCESS) {
        printf("Failed to write to input buffer: Error %d\n", status);
        return EXIT_FAILURE;
    }

    // Compile the kernel.
    cl_kernel kernel = compileKernelFromFile("cwk3.cl", "heatEquation", context, device);

    // Set kernel arguments.
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputGrid);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputGrid);
    clSetKernelArg(kernel, 2, sizeof(int), &N);

    // Query the device's maximum work-group size.
    size_t maxWorkGroupSize;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxWorkGroupSize, NULL);

    // Dynamically calculate the local and global work sizes.
    size_t localWorkSize[2] = {N, N}; 
    size_t globalWorkSize[2] = {N,N};
   
    // Enqueue the kernel for execution.
    status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    if (status != CL_SUCCESS) {
        printf("Failed to enqueue kernel: Error %d\n", status);
        return EXIT_FAILURE;
    }

    // Copy the result back to the host.
    status = clEnqueueReadBuffer(queue, outputGrid, CL_TRUE, 0, N * N * sizeof(float), hostGrid, 0, NULL, NULL);
    if (status != CL_SUCCESS) {
        printf("Failed to read buffer: Error %d\n", status);
        return EXIT_FAILURE;
    }


    // Display the final result. This assumes that the iterated grid was copied back to the hostGrid array.
    printf("Final grid (only top-left shown if too large):\n");
    displayGrid(hostGrid, N);

    // Release all resources.
    free(cpuGrid);
    free(hostGrid);
    clReleaseMemObject(inputGrid);
    clReleaseMemObject(outputGrid);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return EXIT_SUCCESS;
}

