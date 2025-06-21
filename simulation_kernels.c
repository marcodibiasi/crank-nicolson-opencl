__kernel void fill_vector(__global float* buf){
    int i = get_global_id(0);
    buf[i] = 0.0f;
}

__kernel void populate_b(
    __global const float* u,
    __global float* b,

    const float r,
    const int width,
    const int height,

    __local float* local_u
)
{
    int local_width  = get_local_size(0);
    int local_height = get_local_size(1);

    int li = get_local_id(0); 
    int lj = get_local_id(1);
    int gi = get_global_id(0);  
    int gj = get_global_id(1);

    // linear index
    int global_index = gj * width + gi;
    int local_index  = (lj + 1) * (local_width + 2) + (li + 1);

    // central value
    local_u[local_index] = u[global_index];

    if (li == 0 && gi > 0)
        local_u[local_index - 1] = u[gj * width + (gi - 1)];

    if (li == local_width - 1 && gi < width - 1)
        local_u[local_index + 1] = u[gj * width + (gi + 1)];

    if (lj == 0 && gj > 0)
        local_u[local_index - (local_width + 2)] = u[(gj - 1) * width + gi];

    if (lj == local_height - 1 && gj < height - 1)
        local_u[local_index + (local_width + 2)] = u[(gj + 1) * width + gi];

    barrier(CLK_LOCAL_MEM_FENCE);

    if (gi >= width || gj >= height) return;

    float center = local_u[local_index];
    float up     = local_u[local_index - (local_width + 2)];
    float down   = local_u[local_index + (local_width + 2)];
    float left   = local_u[local_index - 1];
    float right  = local_u[local_index + 1];

    // Compute the value of b using the finite difference method
    b[global_index] = (1 - 2*r) * center + 0.5 * r * (up + down + left + right);
}

/*
Instead of having a kernel for the Conjugate Gradient method,
it is better to have separate kernels for each operation.
 */

__kernel void dot_product(
    __global const float* a,
    __global const float* b,
    __global float* result,
    const unsigned int size
)
{}

__kernel void mat_vec_multiply(

    //CSR Matrix format
    __global const float* values,
    __global const int* col_ind,
    __global const int* row_ptr,

    __global const float* vec_in,
    __global float* vec_out,

    __local float* local_memory,

    const int rows
)
{
    int row = get_group_id(0);
    int local_id = get_local_id(0);
    int local_size = get_local_size(0);

    if(row > rows) return;

    // Get the non-zero values for each row
    int row_start = row_ptr[row];
    int row_end = row_ptr[row + 1];
    int row_lenght = row_end - row_start;

    float sum = 0.0f;

    for (int idx = local_id; idx < row_lenght; idx += local_size) {
        int val_idx = row_start + idx;  // Get the CSR Matrix value of non-zero element
        sum += values[val_idx] * vec_in[col_ind[val_idx]];
    }

    local_memory[local_id] = sum;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = local_size / 2; stride > 0; stride /= 2) {
        if (local_id < stride)
            local_memory[local_id] += local_memory[local_id + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0)
        vec_out[row] = local_memory[0];
}