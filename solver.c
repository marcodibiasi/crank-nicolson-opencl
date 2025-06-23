#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "solver.h"
#include "ocl_boiler.h"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#endif

#define RESET   "\033[0m"
#define TITLE   "\033[92m" 
#define LABEL   "\033[32m"  

Solver *setup_solver(int width, int height, float dx, float dy, float dt, float alpha, float *u_curr){
    Solver *solver = malloc(sizeof(Solver));
    if(!solver) return NULL;

    solver->width = width;
    solver->height = height;
    solver->dx = dx;
    solver->dy = dy;
    solver->dt = dt;
    solver->alpha = alpha;
    solver->u_current = u_curr;

    solver->u_next = malloc(width * height * sizeof(float));
    if (!solver->u_next) {
        fprintf(stderr, "Error: Failed to allocate u_next.\n");
        free(solver);
        return NULL;
    }

    //For simplicity, we assume square heatmaps
    solver->rx = solver->ry = (alpha * dt) / (dx * dx);
    if (solver->rx + solver->ry > 1.0) {
        fprintf(stderr, 
            "Warning: rx + ry = %.2f > 1.0 — numerical oscillations may arise.\n",
            solver->rx + solver->ry);
    }

    solver->time_step = 0;

    solver->A = allocate_CSR_matrix(solver->width, solver->height);
    if (!solver->A.row_ptr || !solver->A.col_ind || !solver->A.values) {
        fprintf(stderr, "Error: Failed to allocate CSR matrix.\n");
        free_solver(solver);
        return NULL;
    }

    solver->b = malloc(solver->width * solver->height * sizeof(float));
    if (!solver->b) {
        fprintf(stderr, "Error: Failed to allocate b vector.\n");
        free_solver(solver);
        return NULL;
    }

    setup_coefficients_matrix(solver->rx, &solver->A, width, height);
    //debug_print_CSR(&solver->A, 20);

    setup_opencl_context(solver);

    printf(TITLE "\nSolver settings:\n" RESET
       LABEL "\tWidth: " RESET "%d\n"
       LABEL "\tHeight: " RESET "%d\n"
       LABEL "\tdx: " RESET "%.2f\n"
       LABEL "\tdy: " RESET "%.2f\n"
       LABEL "\trx: " RESET "%.2f\n"
       LABEL "\try: " RESET "%.2f\n"
       LABEL "\tdt: " RESET "%.2f\n"
       LABEL "\talpha: " RESET "%.2f\n\n",
       width, height, dx, dy, solver->rx, solver->ry, dt, alpha);
  
       
    // CHECHING IF THE B VECTOR IS CORRECTLY POPULATED
    // DEBUGGING START
    
    // cl_int err; 
    // err = clEnqueueReadBuffer(solver->cl.q, solver->cl.b_buffer, CL_TRUE, 0,
    //                       width * height * sizeof(float), solver->b,
    //                       0, NULL, NULL);
    // ocl_check(err, "clEnqueueReadBuffer failed");

    // for (int i = 0; i < 20; i++){
    //     printf(" %.2f ", solver->b[i]);
    // }
    
    // DEBUGGING END
    

    return solver;
}

float **run_simulation(Solver *solver, int steps) {
    if (!solver) {
        fprintf(stderr, "Error: Solver is NULL.\n");
        return NULL;
    }


    // Handling frames vector to store all the frames from the simulation
    float **frames = malloc(steps * sizeof(float*));

    frames[0] = malloc(solver->width * solver->height * sizeof(float));
    memcpy(frames[0], solver->u_current, solver->width * solver->height * sizeof(float));

    // Filling the x0 vector to start the resolution of Ax = b
    cl_event zero_fill_evt = zero_fill(solver);
    clWaitForEvents(1, &zero_fill_evt);
    clReleaseEvent(zero_fill_evt);
    

    // Main loop
    while(solver->time_step < steps) {   
        printf("Step %d\n", solver->time_step);

        update_system(solver);

        solver->time_step++;

        frames[solver->time_step] = malloc(solver->width * solver->height * sizeof(float));
        memcpy(frames[solver->time_step], solver->u_current, solver->width * solver->height * sizeof(float));
    }

    // TESTING MODULES 
    // size_t lenght = solver->width * solver->height;
    // cl_mem try = clCreateBuffer(solver->cl.ctx, CL_MEM_READ_WRITE, lenght * sizeof(float), NULL, NULL);
    // cl_event mat_vec_multiply_evt = mat_vec_multiply(solver, solver->cl. b_buffer, try);
    // printf(" %f ", dot_product_handler(solver, &solver->cl.b_buffer, &try, lenght));
    //printf(" %f ", alpha_calculate(solver, &solver->cl.x_buffer, &solver->cl.b_buffer));
    conjugate_gradient(solver);

    //DEBUGGING
    // Checking if the CSR matrix is positive definite (apparently it is)
    // for(int i = 0; i < 50; i++) {
    //     float val = rand() % 1000 - 500; 
    //     is_positive_definite_csr(solver->A.values, solver->A.col_ind, solver->A.row_ptr, 
    //     solver->width * solver->height, val);
    // }

    clFinish(solver->cl.q);
    return frames;
}

void update_system(Solver *solver){
    if (!solver->u_current) {
        fprintf(stderr, "Error: initial state (u_curr) is NULL.\n");
        free_solver(solver);
        return;
    }

    cl_event populate_b_evt = populate_b(solver);
    clWaitForEvents(1, &populate_b_evt);
    clReleaseEvent(populate_b_evt);

    // Saving the result to u_current (host memory)
    cl_int err;
    float *frame = (float*)clEnqueueMapBuffer(solver->cl.q, solver->cl.u_current_buffer, CL_TRUE, 
        CL_MAP_READ, 0, solver->width * solver->height * sizeof(float), 0, NULL, NULL, &err);
    ocl_check(err, "clEnqueueMapBuffer failed");

    memcpy(solver->u_current, frame, solver->width * solver->height * sizeof(float));

    clEnqueueUnmapMemObject(solver->cl.q, solver->cl.u_current_buffer, frame, 0, NULL, NULL);
    clFinish(solver->cl.q);
}

//The type of the function may change in the future
void conjugate_gradient(Solver *solver) {
    // TODO: Implement the conjugate gradient method for solving the linear system
    cl_int err;
    OpenCLContext *cl = &solver->cl;

    /*
    First step: Calculate initial residue r = b - Ax
                since initial x is 0, r = b
    */
    size_t lenght = solver->width * solver->height;
    cl_mem r_buffer = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, lenght * sizeof(float), NULL, &err);
    err = clEnqueueCopyBuffer(cl->q, cl->b_buffer, r_buffer, 0, 0, 
        lenght * sizeof(float), 0, NULL, NULL);
	ocl_check(err, "clCreateBuffer failed");

    /*
    Second step: set first search direction p = r 
                since r = b, then p = b 
                (we can skip few initial steps for ease)
    */
    cl_mem direction_buffer = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, lenght * sizeof(float), NULL, &err);
	ocl_check(err, "clCreateBuffer failed");

    cl_event copy_buffer_evt;
    err = clEnqueueCopyBuffer(cl->q, cl->b_buffer, direction_buffer, 0, 0, 
        lenght * sizeof(float), 0, NULL, &copy_buffer_evt);
    ocl_check(err, "clEnqueueCopyBuffer failed");

    clWaitForEvents(1, &copy_buffer_evt);
    clReleaseEvent(copy_buffer_evt);

    // DEBUGGING: checking the buffers 
    // Buffers are ok, whats the problem ;((
    
    // float* temp = malloc(lenght * sizeof(float));   
    // clEnqueueReadBuffer(cl->q, r_buffer, CL_TRUE, 0, 
    //     lenght * sizeof(float), temp, 0, NULL, NULL);

    // for(int i = 0; i < 20; i++) {
    //     printf(" %.2f \n", temp[i]);
    // }

    // clEnqueueReadBuffer(cl->q, direction_buffer, CL_TRUE, 0, 
    //     lenght * sizeof(float), temp, 0, NULL, NULL);
    // for(int i = 0; i < 20; i++) {
    //     printf(" %.2f \n", temp[i]);
    // }

    /*
    Third step: the main loop leads the result vector to an optimal solution with an error epsilon
                we set epsilon to 10^(-5), but it may vary in the future
    */
    float epsilon = 1e-5f;
    float alpha;    // Gets the lenght of the step along the search direction p
    float r_norm;
    int max_iter = (int)sqrt(lenght);
    int k = 0;

    cl_mem r_next_buffer = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, 
            lenght * sizeof(float), NULL, &err);
        ocl_check(err, "clCreateBuffer failed for r_next_buffer");

    do{
        // alpha = dot(r, r) / dot(p, mat_vec(A, p))
        alpha = alpha_calculate(solver, &r_buffer, &direction_buffer);
        printf("Iteration %d: alpha = %.6f\n", k, alpha);

        // Update the solution vector x = x + alpha * p
        update_x(solver, &direction_buffer, alpha, lenght);

        // Calculate the new residue r_(k+1) = r - alpha * mat_vec(A, p)
        update_r(solver, &r_buffer, &direction_buffer, &r_next_buffer, alpha, lenght);
 
        // Calculate the norm of the new residue ||r_(k+1)||
        r_norm = dot_product_handler(solver, &r_next_buffer, &r_next_buffer, lenght);
        printf("Iteration %d: Residue norm = %.6f\n", k, sqrt(r_norm));

        // beta = dot(r_(k+1), r_(k+1)) / dot(r, r)
        float r_dot_r = dot_product_handler(solver, &r_buffer, &r_buffer, lenght);
        float beta = r_norm / r_dot_r;
        printf("Iteration %d: beta = (%.6f)/(%.6f) = %.6f\n", k, r_norm, r_dot_r, beta);

        // Update the search direction p = r_(k+1) + beta * p
        update_p(solver, &r_next_buffer, &direction_buffer, beta, lenght);
        printf("Iteration %d: Updated search direction norm = %6f\n", k, 
            sqrt(dot_product_handler(solver, &direction_buffer, &direction_buffer, lenght)));

        // Update the residue for the next iteration
        err = clEnqueueCopyBuffer(cl->q, r_next_buffer, r_buffer, 0, 0, 
            lenght * sizeof(float), 0, NULL, NULL);
        ocl_check(err, "clEnqueueCopyBuffer failed for r_buffer");

        printf(" \n ");

        k++;
    } while(sqrt(r_norm) > epsilon && k < max_iter);

    printf("\nConjugate Gradient converged after %d iterations with norm %.6f\n", k, r_norm);
}

float alpha_calculate(Solver* solver, cl_mem *r, cl_mem *p) {
    cl_int err;
    OpenCLContext *cl = &solver->cl;
    int lenght = solver->width * solver->height;

    // r * r
    float numerator = dot_product_handler(solver, r, r, lenght);

    // p * A * p 
    cl_mem partial_denominator = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, lenght * sizeof(float), NULL, &err);
    ocl_check(err, "clCreateBuffer failed for partial_denominator");

    cl_event mat_vec_multiply_evt = mat_vec_multiply(solver, *p, partial_denominator);
    clWaitForEvents(1, &mat_vec_multiply_evt);
    clReleaseEvent(mat_vec_multiply_evt);

    float denominator = dot_product_handler(solver, p, &partial_denominator, lenght);

    printf("Alpha -> Numerator: %f, Denominator: %f\n", numerator, denominator);
    return numerator / denominator;
}

void update_x(Solver *solver, cl_mem* p, float alpha, int lenght) {
    OpenCLContext *cl = &solver->cl;
    cl_int err;

    // Create a buffer for the updated x
    cl_mem x_next_buffer = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, 
        lenght * sizeof(float), NULL, &err);
    ocl_check(err, "clCreateBuffer failed for x_next_buffer");

    // alpha * p
    cl_event scale_vector_evt = scale_vector(solver, p, alpha, &x_next_buffer, lenght);
    clWaitForEvents(1, &scale_vector_evt);
    clReleaseEvent(scale_vector_evt);

    // Add the scaled p to x
    cl_event sum_vectors_evt = sum_vectors(solver, &cl->x_buffer, &x_next_buffer, &cl->x_buffer, lenght);
    clWaitForEvents(1, &sum_vectors_evt);
    clReleaseEvent(sum_vectors_evt);

    // Release the temporary buffer
    clReleaseMemObject(x_next_buffer);
}   

void update_r(Solver *solver, cl_mem* r, cl_mem* p, cl_mem* r_next, float alpha, int lenght) {
    OpenCLContext *cl = &solver->cl;
    cl_int err;

    // A * p
    cl_event mat_vec_multiply_evt = mat_vec_multiply(solver, *p, *r_next);
    clWaitForEvents(1, &mat_vec_multiply_evt);
    clReleaseEvent(mat_vec_multiply_evt);

    // Scale the result by -alpha
    cl_event scale_vector_evt = scale_vector(solver, r_next, -alpha, r_next, lenght);
    clWaitForEvents(1, &scale_vector_evt);
    clReleaseEvent(scale_vector_evt);

    // Add the scaled result to the original residue
    cl_event sum_vectors_evt = sum_vectors(solver, r, r_next, r_next, lenght);
    clWaitForEvents(1, &sum_vectors_evt);
    clReleaseEvent(sum_vectors_evt);
}

void update_p(Solver *solver, cl_mem* r, cl_mem* p, float beta, int lenght) {
    OpenCLContext *cl = &solver->cl;
    cl_int err;

    cl_mem p_temp = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, 
        lenght * sizeof(float), NULL, &err);
    ocl_check(err, "clCreateBuffer failed for p_temp");
    
    // Scale the previous search direction p by beta
    cl_event scale_vector_evt = scale_vector(solver, p, beta, &p_temp, lenght);
    clWaitForEvents(1, &scale_vector_evt);
    clReleaseEvent(scale_vector_evt);

    // Add the new residue r to the scaled search direction
    cl_event sum_vectors_evt = sum_vectors(solver, r, &p_temp, p, lenght);
    clWaitForEvents(1, &sum_vectors_evt);
    clReleaseEvent(sum_vectors_evt);

    // Release the temporary buffer
    clReleaseMemObject(p_temp);
}

float dot_product_handler(Solver *solver, cl_mem *vec1, cl_mem *vec2, int lenght) {
    OpenCLContext *cl = &solver->cl;
    cl_int err;

    //Partial dot product
    cl_mem partial_dot_product = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, sizeof(float), NULL, &err);
    cl_event r_dot_r = dot_product(solver, vec1, vec1, &partial_dot_product, lenght);
    clWaitForEvents(1, &r_dot_r);
    clReleaseEvent(r_dot_r);

    size_t num_groups = round_div_up(solver->width * solver->height, cl->lws);

    cl_mem temp_buffer = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, 
        num_groups * sizeof(float), NULL, NULL);

    cl_mem *in_buf = &partial_dot_product;
    cl_mem *out_buf = &temp_buffer;

    while(num_groups > 1) {
        // Perform partial sum reduction
        cl_event partial_sum_evt = partial_sum_reduction(solver, in_buf, out_buf, num_groups);
        clWaitForEvents(1, &partial_sum_evt);
        clReleaseEvent(partial_sum_evt);

        // Swap buffers
        cl_mem temp = *in_buf;
        *in_buf = *out_buf;
        *out_buf = temp;

        num_groups = round_div_up(num_groups, cl->lws);
    }

    float final_result;
    clEnqueueReadBuffer(cl->q, *in_buf, CL_TRUE, 0, sizeof(float), &final_result, 0, NULL, NULL);
    
    return final_result;
}

CSRMatrix allocate_CSR_matrix(int width, int height) {
    CSRMatrix matrix;

    matrix.row_ptr = malloc((width * height + 1) * sizeof(int));
    if (!matrix.row_ptr) {
        fprintf(stderr, "Error: Failed to allocate row_ptr.\n");
        matrix.row_ptr = NULL;
        return matrix;
    }

    matrix.col_ind = malloc(5 * width * height * sizeof(int));
    if (!matrix.col_ind) {
        fprintf(stderr, "Error: Failed to allocate col_ind.\n");
        free(matrix.row_ptr);
        matrix.row_ptr = NULL;
        return matrix;
    }

    //The size is estimated for a 5-point stencil
    matrix.values = malloc(5 * width * height * sizeof(float));
    if (!matrix.values) {
        fprintf(stderr, "Error: Failed to allocate values.\n");
        free(matrix.row_ptr);
        free(matrix.col_ind);
        matrix.row_ptr = NULL;
        matrix.col_ind = NULL;
        return matrix;
    }

    return matrix;
}

void free_CSR_matrix(CSRMatrix *matrix) {
    if (matrix) {
        free(matrix->row_ptr);
        free(matrix->col_ind);
        free(matrix->values);
    }
}

void free_solver(Solver *solver) {
    if (solver) {
        free(solver->u_next);
        free(solver->b);
        free_CSR_matrix(&solver->A);
        free(solver);
    }
}

void setup_coefficients_matrix(float rx, CSRMatrix *A, int width, int height) {
    if (!A || !A->row_ptr || !A->col_ind || !A->values) {
        fprintf(stderr, "Error: CSR matrix is not properly allocated.\n");
        return;
    }

    //Initialize the row pointer
    A->row_ptr[0] = 0;
    int index = 0; //Current index in the sparse matrix

    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            int k = i + j * width; //Current position

            A->col_ind[index] = k;
            A->values[index++] = 1.0 + 2.0 * rx;

            if (i > 0) {
                A->col_ind[index] = k - 1;
                A->values[index++] = -rx / 2.0;
            }

            if (i < width - 1) {
                A->col_ind[index] = k + 1;
                A->values[index++] = -rx / 2.0;
            }

            if (j > 0) {
                A->col_ind[index] = k - width;
                A->values[index++] = -rx / 2.0;
            }

            if (j < height - 1) {
                A->col_ind[index] = k + width;
                A->values[index++] = -rx / 2.0;
            }

            A->row_ptr[k + 1] = index;
        }
    }

    A->row_ptr[width * height] = index; 
}

void debug_print_CSR(const CSRMatrix *A, int n) {
    printf("row_ptr: ");
    for (int i = 0; i < n; i++) {  // row_ptr ha length height+1
        printf("%d ", A->row_ptr[i]);
    }
    printf("\n");

    printf("col_ind: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", A->col_ind[i]);
    }
    printf("\n");

    printf("values: ");
    for (int i = 0; i < n; i++) {
        printf("%.4f ", A->values[i]);
    }
    printf("\n");
}

void setup_opencl_context(Solver *solver) {
    OpenCLContext *cl = &solver->cl;
    OpenCLKernels *kernels = &solver->cl.kernels;

    cl->p = select_platform();
	cl->d = select_device(cl->p);
	cl->ctx = create_context(cl->p, cl->d);
	cl->q = create_queue(cl->ctx, cl->d);
	cl->prog = create_program("simulation_kernels.c", cl->ctx, cl->d);

    //Allocate memory
    size_t lenght = solver->width * solver->height;

	cl_int err;

    // Allocate OpenCL buffers
	cl->b_buffer = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, lenght * sizeof(float), NULL, &err);
	ocl_check(err, "clCreateBuffer failed");

    cl->u_current_buffer = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                        lenght * sizeof(float), solver->u_current, &err);
    ocl_check(err, "clCreateBuffer failed for u_buffer");

    cl->x_buffer = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, lenght * sizeof(float), NULL, &err);
    ocl_check(err, "clCreateBuffer failed for u_buffer");

    //CSR Matrix buffers
    cl->csr_row_ptr_buffer = clCreateBuffer(cl->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        (lenght + 1) * sizeof(int), solver->A.row_ptr, &err);
    ocl_check(err, "clCreateBuffer failed for csr_row_ptr_buffer");

    cl->csr_col_ind_buffer = clCreateBuffer(cl->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        5 * lenght * sizeof(int), solver->A.col_ind, &err);
    ocl_check(err, "clCreateBuffer failed for csr_col_ind_buffer");

    cl->csr_values_buffer = clCreateBuffer(cl->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        5 * lenght * sizeof(float), solver->A.values, &err);
    ocl_check(err, "clCreateBuffer failed for csr_values_buffer");
    
    // Create kernels
    kernels->zero_fill = clCreateKernel(cl->prog, "fill_vector", &err);
    ocl_check(err, "clCreateKernel failed");

    kernels->populate_b = clCreateKernel(cl->prog, "populate_b", &err);  
    ocl_check(err, "clCreateKernel failed");

    kernels->parallel_sum_reduction = clCreateKernel(cl->prog, "partial_sum_reduction", &err);  
    ocl_check(err, "clCreateKernel failed");

    kernels->dot_product = clCreateKernel(cl->prog, "dot_product", &err);  
    ocl_check(err, "clCreateKernel failed");

    kernels->mat_vec_multiply = clCreateKernel(cl->prog, "mat_vec_multiply", &err);  
    ocl_check(err, "clCreateKernel failed");

    kernels->sum_vectors = clCreateKernel(cl->prog, "sum_vectors", &err);  
    ocl_check(err, "clCreateKernel failed");

    kernels->scale_vector = clCreateKernel(cl->prog, "scale_vector", &err);  
    ocl_check(err, "clCreateKernel failed");
    
    // New local work size logic
    cl->lws = 16;

    // DEPRECATED: Going for a version with an unified local work size for ease

    // size_t populate_b_lws, populate_b_max_wg_size;
    // size_t dot_product_lws, dot_product_max_wg_size;

    // Local work sizes
    // Get preferred local work size for populate_b kernel and resize as needed

    // clGetKernelWorkGroupInfo(cl->populate_b, cl->d, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), 
    //     &populate_b_max_wg_size, NULL);
    // clGetKernelWorkGroupInfo(cl->populate_b, cl->d, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 
    //     sizeof(size_t), &populate_b_lws, NULL);

    // while((populate_b_lws * populate_b_lws) > populate_b_max_wg_size)
    //     populate_b_lws = populate_b_lws/2;

    // solver->cl.populate_b_preferred_lws[0] = populate_b_lws;
    // solver->cl.populate_b_preferred_lws[1] = populate_b_lws;

    // // Get preferred local work size for dot_product kernel and resize as needed
    // clGetKernelWorkGroupInfo(cl->dot_product, cl->d, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), 
    //     &dot_product_max_wg_size, NULL);
    // clGetKernelWorkGroupInfo(cl->dot_product, cl->d, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 
    //     sizeof(size_t), &dot_product_lws, NULL); 

    // while((dot_product_lws * dot_product_lws) > dot_product_max_wg_size)
    //     dot_product_lws = dot_product_lws/2; 

    // solver->cl.dot_product_preferred_lws[0] = dot_product_lws;
    // solver->cl.dot_product_preferred_lws[1] = dot_product_lws;

    // printf("\nPreferred Local Work Size for populate_b kernel = [%zu, %zu]\n", populate_b_lws, populate_b_lws);
    // printf("Preferred Local Work Size for dot_product kernel = [%zu, %zu]\n", 
    //        dot_product_lws, dot_product_lws);
}

cl_event zero_fill(Solver* solver) {
    OpenCLContext *cl = &solver->cl;
    cl_int err;
    cl_int arg = 0;

    // Set kernel arguments 
    err = clSetKernelArg(cl->kernels.zero_fill, arg, sizeof(cl_mem), &cl->x_buffer);
    ocl_check(err, "clSetKernelArg failed for x_buffer");
    arg++;

    // Launch the kernel 
    cl_event event;
    size_t gws = round_mul_up((solver->width * solver->height), cl->lws);
    err = clEnqueueNDRangeKernel(cl->q, cl->kernels.zero_fill, 1, NULL,
            &gws, &cl->lws, 0, NULL, &event);
    ocl_check(err, "clEnqueueNDRangeKernel failed");

    return event;
}

cl_event populate_b(Solver *solver) {
    OpenCLContext *cl = &solver->cl;
    cl_int err;
    cl_int arg = 0;

    // Set kernel arguments
    err = clSetKernelArg(cl->kernels.populate_b, arg, sizeof(cl_mem), &cl->u_current_buffer);
    ocl_check(err, "clSetKernelArg failed for u_current_buffer");
    arg++;

    err = clSetKernelArg(cl->kernels.populate_b, arg, sizeof(cl_mem), &cl->b_buffer);
    ocl_check(err, "clSetKernelArg failed for b_buffer");
    arg++;

    err = clSetKernelArg(cl->kernels.populate_b, arg, sizeof(float), &solver->rx);
    ocl_check(err, "clSetKernelArg failed for rx");
    arg++;

    err = clSetKernelArg(cl->kernels.populate_b, arg, sizeof(int), &solver->width);
    ocl_check(err, "clSetKernelArg failed for width");
    arg++;

    err = clSetKernelArg(cl->kernels.populate_b, arg, sizeof(int), &solver->height);
    ocl_check(err, "clSetKernelArg failed for height");
    arg++;

    // size_t local_mem_size = sizeof(float) * (cl->populate_b_preferred_lws[0] + 2) * 
    //                         (cl->populate_b_preferred_lws[1] + 2);

    size_t local_mem_size = sizeof(float) * (cl->lws + 2) * (cl->lws + 2);
    err = clSetKernelArg(cl->kernels.populate_b, arg, local_mem_size, NULL);
    ocl_check(err, "clSetKernelArg failed for height");
    arg++;

    // Launch the kernel
    // size_t gws[2] = {round_mul_up(solver->width, cl->populate_b_preferred_lws[0]), 
    //         round_mul_up(solver->height, cl->populate_b_preferred_lws[1])};
    size_t gws[2] = {round_mul_up(solver->width, cl->lws), round_mul_up(solver->height, cl->lws)};
    size_t lws[2] = {cl->lws, cl->lws};

    cl_event event;
    err = clEnqueueNDRangeKernel(cl->q, cl->kernels.populate_b, 2, NULL, gws, lws, 0, NULL, &event);
    ocl_check(err, "clEnqueueNDRangeKernel failed");

    return event;
}

cl_event partial_sum_reduction(Solver *solver, cl_mem* vec_in, cl_mem* vec_out, int lenght) {
    OpenCLContext *cl = &solver->cl;
    cl_int err;
    cl_int arg = 0;

    // Set kernel arguments
    err = clSetKernelArg(cl->kernels.parallel_sum_reduction, arg, sizeof(cl_mem), vec_in);
    ocl_check(err, "clSetKernelArg failed for vec_in");
    arg++;

    err = clSetKernelArg(cl->kernels.parallel_sum_reduction, arg, sizeof(cl_mem), vec_out);
    ocl_check(err, "clSetKernelArg failed for vec_out");
    arg++;

    err = clSetKernelArg(cl->kernels.parallel_sum_reduction, arg, cl->lws * sizeof(float), NULL);
    ocl_check(err, "clSetKernelArg failed for local_memory");
    arg++;

    err = clSetKernelArg(cl->kernels.parallel_sum_reduction, arg, sizeof(int), &lenght);
    ocl_check(err, "clSetKernelArg failed for lenght");
    arg++;

    //Launch the kernel
    cl_event event;
    size_t gws = round_mul_up(lenght, cl->lws);
    err = clEnqueueNDRangeKernel(cl->q, cl->kernels.parallel_sum_reduction, 1, NULL,
            &gws, &cl->lws, 0, NULL, &event);
    ocl_check(err, "clEnqueueNDRangeKernel failed");

    return event;
}

cl_event dot_product(Solver *solver, cl_mem* vec1, cl_mem* vec2, cl_mem* result, int lenght) {
    OpenCLContext *cl = &solver->cl;
    cl_int err;
    cl_int arg = 0;

    // Set kernel arguments
    err = clSetKernelArg(cl->kernels.dot_product, arg, sizeof(cl_mem), vec1);
    ocl_check(err, "clSetKernelArg failed for vec1");
    arg++;

    err = clSetKernelArg(cl->kernels.dot_product, arg, sizeof(cl_mem), vec2);
    ocl_check(err, "clSetKernelArg failed for vec2");
    arg++;  

    err = clSetKernelArg(cl->kernels.dot_product, arg, sizeof(cl_mem), result);
    ocl_check(err, "clSetKernelArg failed for result");
    arg++;

    err = clSetKernelArg(cl->kernels.dot_product, arg, cl->lws * sizeof(float), NULL);
    ocl_check(err, "clSetKernelArg failed for local_memory");
    arg++;

    err = clSetKernelArg(cl->kernels.dot_product, arg, sizeof(int), &lenght);
    ocl_check(err, "clSetKernelArg failed for lenght");
    arg++;

    // Launch the kernel
    cl_event event;
    size_t gws = round_mul_up(lenght, cl->lws);
    err = clEnqueueNDRangeKernel(cl->q, cl->kernels.dot_product, 1, NULL,
            &gws, &cl->lws, 0, NULL, &event);
    ocl_check(err, "clEnqueueNDRangeKernel failed");    

    return event;
}

// While the matrix to multiply is always the same, the vector changes
cl_event mat_vec_multiply(Solver *solver, cl_mem vec, cl_mem result) {
    OpenCLContext *cl = &solver->cl;
    cl_int err;
    cl_int arg = 0;

    // Set kernel arguments
    err = clSetKernelArg(cl->kernels.mat_vec_multiply, arg, sizeof(cl_mem), &cl->csr_values_buffer);
    ocl_check(err, "clSetKernelArg failed for csr_values_buffer");
    arg++;

    err = clSetKernelArg(cl->kernels.mat_vec_multiply, arg, sizeof(cl_mem), &cl->csr_col_ind_buffer);
    ocl_check(err, "clSetKernelArg failed for csr_col_ind_buffer");
    arg++;

    err = clSetKernelArg(cl->kernels.mat_vec_multiply, arg, sizeof(cl_mem), &cl->csr_row_ptr_buffer);
    ocl_check(err, "clSetKernelArg failed for csr_row_ptr_buffer");
    arg++;

    err = clSetKernelArg(cl->kernels.mat_vec_multiply, arg, sizeof(cl_mem), &vec);
    ocl_check(err, "clSetKernelArg failed for vec");
    arg++;

    err = clSetKernelArg(cl->kernels.mat_vec_multiply, arg, sizeof(cl_mem), &result);
    ocl_check(err, "clSetKernelArg failed for result");
    arg++;

    //TODO: Unify local work size for every kernel
    size_t local_mem_size = sizeof(float) * cl->lws;
    err = clSetKernelArg(cl->kernels.mat_vec_multiply, arg, local_mem_size, NULL);
    ocl_check(err, "clSetKernelArg failed for local memory");
    arg++;

    err = clSetKernelArg(cl->kernels.mat_vec_multiply, arg, sizeof(int), &solver->height);
    ocl_check(err, "clSetKernelArg failed for height");
    arg++;

    //Launch the kernel
    cl_event event;
    size_t gws = round_mul_up((solver->width * solver->height), cl->lws);
    err = clEnqueueNDRangeKernel(cl->q, cl->kernels.mat_vec_multiply, 1, NULL,
            &gws, &cl->lws, 0, NULL, &event);
    ocl_check(err, "clEnqueueNDRangeKernel failed");

    return event;
}

cl_event sum_vectors(Solver* solver, cl_mem* vec1, cl_mem* vec2, cl_mem* result, int lenght) {
    OpenCLContext *cl = &solver->cl;
    cl_int err;
    cl_int arg = 0;

    // Set kernel arguments
    err = clSetKernelArg(cl->kernels.sum_vectors, arg, sizeof(cl_mem), vec1);
    ocl_check(err, "clSetKernelArg failed for vec1");
    arg++;      

    err = clSetKernelArg(cl->kernels.sum_vectors, arg, sizeof(cl_mem), vec2);
    ocl_check(err, "clSetKernelArg failed for vec2");
    arg++;  

    err = clSetKernelArg(cl->kernels.sum_vectors, arg, sizeof(cl_mem), result);
    ocl_check(err, "clSetKernelArg failed for result");
    arg++;

    err = clSetKernelArg(cl->kernels.sum_vectors, arg, sizeof(int), &lenght);
    ocl_check(err, "clSetKernelArg failed for lenght");
    arg++;

    // Launch the kernel 
    cl_event event;
    size_t gws = round_mul_up(lenght, cl->lws);
    err = clEnqueueNDRangeKernel(cl->q, cl->kernels.sum_vectors, 1, NULL,
            &gws, &cl->lws, 0, NULL, &event);
    ocl_check(err, "clEnqueueNDRangeKernel failed");    

    return event;
}

cl_event scale_vector(Solver* solver, cl_mem* vec, float scale, cl_mem* result, int lenght) {
    OpenCLContext *cl = &solver->cl;
    cl_int err;
    cl_int arg = 0;

    // Set kernel arguments
    err = clSetKernelArg(cl->kernels.scale_vector, arg, sizeof(cl_mem), vec);
    ocl_check(err, "clSetKernelArg failed for vec");
    arg++;

    err = clSetKernelArg(cl->kernels.scale_vector, arg, sizeof(cl_mem), result);
    ocl_check(err, "clSetKernelArg failed for result");
    arg++;

    err = clSetKernelArg(cl->kernels.scale_vector, arg, sizeof(float), &scale);
    ocl_check(err, "clSetKernelArg failed for scale");
    arg++;

    err = clSetKernelArg(cl->kernels.scale_vector, arg, sizeof(int), &lenght);
    ocl_check(err, "clSetKernelArg failed for lenght");
    arg++;

    // Launch the kernel 
    cl_event event;
    size_t gws = round_mul_up(lenght, cl->lws);
    err = clEnqueueNDRangeKernel(cl->q, cl->kernels.scale_vector, 1, NULL,
            &gws, &cl->lws, 0, NULL, &event);
    ocl_check(err, "clEnqueueNDRangeKernel failed");    

    return event;
}

int is_positive_definite_csr(const float* values, const int* col_ind, const int* row_ptr, int n, float value) {
    float* x = malloc(n * sizeof(float));
    float* Ax = calloc(n, sizeof(float));
    for (int i = 0; i < n; ++i) x[i] = value;

    // Matrice * vettore
    for (int row = 0; row < n; ++row) {
        for (int idx = row_ptr[row]; idx < row_ptr[row + 1]; ++idx) {
            Ax[row] += values[idx] * x[col_ind[idx]];
        }
    }

    // Calcola x^T A x
    float xtAx = 0.0f;
    for (int i = 0; i < n; ++i) {
        xtAx += x[i] * Ax[i];
    }

    free(x);
    free(Ax);

    printf("-----> %d\n", xtAx > 0.0f);

    return xtAx > 0.0f;
}
