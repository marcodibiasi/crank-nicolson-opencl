#include <stdio.h>
#include <stdlib.h>
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

void update_system(Solver *solver){
    if (!solver->u_current) {
        fprintf(stderr, "Error: initial state (u_curr) is NULL.\n");
        free_solver(solver);
        return;
    }

    cl_event populate_b_evt = populate_b(solver);
    clWaitForEvents(1, &populate_b_evt);
    clReleaseEvent(populate_b_evt);
}

float **run_simulation(Solver *solver, int steps) {
    if (!solver) {
        fprintf(stderr, "Error: Solver is NULL.\n");
        return NULL;
    }

    float **frames = malloc(steps * sizeof(float*));
    for (size_t i = 0; i < steps; i++)
        frames[i] = malloc(solver->width * solver->height * sizeof(float));

    while(solver->time_step < steps) {   
        printf("Step %d\n", solver->time_step);
        update_system(solver);
        
        solver->time_step++;
    }

    clFinish(solver->cl.q);
    return frames;
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
    cl->p = select_platform();
	cl->d = select_device(cl->p);
	cl->ctx = create_context(cl->p, cl->d);
	cl->q = create_queue(cl->ctx, cl->d);
	cl->prog = create_program("populate_b.ocl", cl->ctx, cl->d);

    //Allocate memory
    size_t lenght = solver->width * solver->height;

	cl_int err;
	cl->b_buffer = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, lenght * sizeof(float), NULL, &err);
	ocl_check(err, "clCreateBuffer failed");

    cl->u_current_buffer = clCreateBuffer(cl->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          lenght * sizeof(float),
                                          solver->u_current, &err);
    ocl_check(err, "clCreateBuffer failed for u_current_buffer");

    cl->populate_b = clCreateKernel(cl->prog, "populate_b", &err);  
    ocl_check(err, "clCreateKernel failed");
    
    size_t lws, max_wg_size;
    clGetKernelWorkGroupInfo(cl->populate_b, cl->d, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &max_wg_size, NULL);

    clGetKernelWorkGroupInfo(cl->populate_b, cl->d, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 
        sizeof(size_t), &lws, NULL);

    while((lws * lws) > max_wg_size)
        lws = lws/2;
    
    cl->preferred_lws[0] = lws;
    cl->preferred_lws[1] = lws; 
    printf(" \n Preferred Local Work Size = [%zu, %zu]\n", cl->preferred_lws[0], cl->preferred_lws[1]);
}

cl_event populate_b(Solver *solver) {
    OpenCLContext *cl = &solver->cl;
    cl_int err;
    cl_int arg = 0;

    //Set kernel arguments
    err = clSetKernelArg(cl->populate_b, arg, sizeof(cl_mem), &cl->u_current_buffer);
    ocl_check(err, "clSetKernelArg failed for u_current_buffer");
    arg++;

    err = clSetKernelArg(cl->populate_b, arg, sizeof(cl_mem), &cl->b_buffer);
    ocl_check(err, "clSetKernelArg failed for b_buffer");
    arg++;

    err = clSetKernelArg(cl->populate_b, arg, sizeof(float), &solver->rx);
    ocl_check(err, "clSetKernelArg failed for rx");
    arg++;

    err = clSetKernelArg(cl->populate_b, arg, sizeof(int), &solver->width);
    ocl_check(err, "clSetKernelArg failed for width");
    arg++;

    err = clSetKernelArg(cl->populate_b, arg, sizeof(int), &solver->height);
    ocl_check(err, "clSetKernelArg failed for height");
    arg++;

    err = clSetKernelArg(cl->populate_b, arg, 
                                 sizeof(float) * (cl->preferred_lws[0] + 2) * (cl->preferred_lws[1] + 2), NULL);
    ocl_check(err, "clSetKernelArg failed for height");
    arg++;

    //Launch the kernel
    size_t gws[2] = {round_mul_up(solver->width, cl->preferred_lws[0]), 
                                 round_mul_up(solver->height, cl->preferred_lws[1])};

    cl_event event;
    err = clEnqueueNDRangeKernel(cl->q, cl->populate_b, 2, NULL,
                                 gws, cl->preferred_lws, 0, NULL, &event);
    ocl_check(err, "clEnqueueNDRangeKernel failed");

    return event;
}