typedef struct{
    int *row_ptr;
    int *col_ind;
    double *values;
} CSRMatrix;

typedef struct {
    int width, height;
    int time_step;  //Curent simulation index (starts from 0)
    double dx, dy;  //Physical distance 
    double dt;      //Physical time between steps
    double rx, ry;  //Numerical diffusion coefficients 
    double alpha;   //Diffusion coefficient
    float *u_current;  //Current state of the system
    float *u_next;     //Next state of the system
    CSRMatrix A;  //Sparse matrix in CSR format for the system of equations
    double *b;    //Right-hand side vector for the system of equations
} Solver;

Solver *setup_solver(int width, int height, double dx, double dy, double dt, double alpha, float *u_curr);
void update_system(Solver *solver);
CSRMatrix allocate_CSR_matrix(int width, int height);
void free_solver(Solver *solver);
void free_CSR_matrix(CSRMatrix *matrix);
void setup_coefficients_matrix(double rx, CSRMatrix *A, int width, int height);
void debug_print_CSR(const CSRMatrix *A, int n);