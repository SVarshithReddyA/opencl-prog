// Kernel for the heat equation.

__kernel void heatEquation(
    __global const float *inputGrid, // Input grid
    __global float *outputGrid,      // Output grid
    const int N                      // Grid size
) {
    // Get the row and column indices for this work item.
    int row = get_global_id(0);
    int col = get_global_id(1);

    // Compute the 1D index for the current cell.
    int idx = row * N + col;

    // Boundary cells remain zero.
    if (row == 0 || col == 0 || row == N - 1 || col == N - 1) {
        outputGrid[idx] = 0.0f;
    } else {
        // Compute the average of the 4 neighbors.
        float left = inputGrid[idx - 1];
        float right = inputGrid[idx + 1];
        float top = inputGrid[idx - N];
        float bottom = inputGrid[idx + N];
        outputGrid[idx] = 0.25f * (left + right + top + bottom);

        // Debugging output
        if (row == 1 && col == 1) {
            printf("row: %d, col: %d, left: %f, right: %f, top: %f, bottom: %f, result: %f\n",
                   row, col, left, right, top, bottom, outputGrid[idx]);
        }
    }
}
