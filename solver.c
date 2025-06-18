#include <stdio.h>
#include <stdlib.h>
#include "solver.h"
#include "ocl_boiler.h"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define RESET   "\033[0m"
#define TITLE   "\033[92m" 
#define LABEL   "\033[32m"  

Solver *setup_solver(int width, int height, double dx, double dy, double dt, double alpha, float *u_curr){
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

    solver->b = malloc(solver->width * solver->height * sizeof(double));
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
       LABEL "\tdt: " RESET "%.2f\n"
       LABEL "\talpha: " RESET "%.2f\n\n",
       width, height, dx, dy, dt, alpha);
    
    return solver;
}

void update_system(Solver *solver){
    if (!solver->u_current) {
        fprintf(stderr, "Error: initial state (u_curr) is NULL.\n");
        free_solver(solver);
        return;
    }

    //TODO: calculate the right-hand side vector b
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
    matrix.values = malloc(5 * width * height * sizeof(double));
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
        free(solver->u_current);
        free(solver->u_next);
        free(solver->b);
        free_CSR_matrix(&solver->A);
        free(solver);
    }
}

void setup_coefficients_matrix(double rx, CSRMatrix *A, int width, int height) {
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
    size_t memsize = solver->width * solver->height * sizeof(double);

	cl_int err;
	cl_mem d_in = clCreateBuffer(cl->ctx, CL_MEM_WRITE_ONLY, memsize, NULL, &err);
	ocl_check(err, "clCreateBuffer failed");
}