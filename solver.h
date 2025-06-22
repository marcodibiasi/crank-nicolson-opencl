#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#endif

typedef struct{
    int *row_ptr;
    int *col_ind;
    float *values;
} CSRMatrix;

typedef struct{
    cl_kernel zero_fill;
    cl_kernel populate_b;
    cl_kernel parallel_sum_reduction;
    cl_kernel dot_product;
    cl_kernel mat_vec_multiply;
    cl_kernel sum_vectors;
    cl_kernel subtract_vectors;
    cl_kernel scale_vector;
} OpenCLKernels;

typedef struct{
    cl_platform_id p;
	cl_device_id d;
	cl_context ctx;
	cl_command_queue q;
	cl_program prog;

    cl_mem u_current_buffer;
    cl_mem x_buffer;    // Unknown vector to store the Ax = b result, equivalent to u_next
    cl_mem b_buffer;

    // CSRMatrix buffers
    cl_mem csr_row_ptr_buffer;
    cl_mem csr_col_ind_buffer;
    cl_mem csr_values_buffer;

    //size_t populate_b_preferred_lws[2];
    //size_t dot_product_preferred_lws[2];
    
    /*
    Unified local work size for every kernel for ease
    for now the local work size is a fixed value

    TODO: Custom local work size (scalability, performances)
    */
    size_t lws;

    OpenCLKernels kernels;
} OpenCLContext;

typedef struct {
    int width, height;
    int time_step;  //Curent simulation index (starts from 0)
    float dx, dy;  //Physical distance 
    float dt;      //Physical time between steps
    float rx, ry;  //Numerical diffusion coefficients 
    float alpha;   //Diffusion coefficient

    float *u_current;  //Current state of the system
    float *u_next;     //Next state of the system
    CSRMatrix A;  //Sparse matrix in CSR format for the system of equations
    float *b;    
    /*
    Right-hand side vector for the system of equations, 
    just for the sake of visualization and debugging. 
    Having an OpenCL buffer is already enough, since b is used only for 
    the calculation of the linear system during the intemediate steps
    and not as an input or output of the OpenCL kernel.
    */

    OpenCLContext cl;
} Solver;

Solver *setup_solver(int width, int height, float dx, float dy, float dt, float alpha, float *u_curr);
void update_system(Solver *solver);
float** run_simulation(Solver *solver, int steps);
void conjugate_gradient(Solver *solver);
CSRMatrix allocate_CSR_matrix(int width, int height);
void free_solver(Solver *solver);
void free_CSR_matrix(CSRMatrix *matrix);
void setup_coefficients_matrix(float rx, CSRMatrix *A, int width, int height);
void debug_print_CSR(const CSRMatrix *A, int n);
void setup_opencl_context(Solver* solver);
float alpha_calculate(Solver* solver, cl_mem* r, cl_mem* p);
void update_x(Solver *solver, cl_mem* p, float alpha, int lenght);
void update_r(Solver *solver, cl_mem* r, cl_mem* p, cl_mem* r_next, float alpha, int lenght);
void update_p(Solver *solver, cl_mem* r, cl_mem* p, float beta, int lenght);
float dot_product_handler(Solver *solver, cl_mem *vec1, cl_mem *vec2, int lenght);
cl_event zero_fill(Solver* solver);
cl_event partial_sum_reduction(Solver *solver, cl_mem* vec_in, cl_mem* vec_out, int lenght);
cl_event dot_product(Solver *solver, cl_mem* vec1, cl_mem* vec2, cl_mem* result, int lenght);
cl_event populate_b(Solver *solver);
cl_event mat_vec_multiply(Solver *solver, cl_mem vec, cl_mem result);
cl_event sum_vectors(Solver* solver, cl_mem* vec1, cl_mem* vec2, cl_mem* result, int lenght);
cl_event scale_vector(Solver* solver, cl_mem* vec, float scalar, cl_mem* result, int lenght);

//DEBUG FUNCTIONS (tnx chatgpt)
int is_positive_definite_csr(const float* values, const int* col_ind,const int* row_ptr,int n,float val);