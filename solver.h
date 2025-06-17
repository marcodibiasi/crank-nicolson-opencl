typedef struct {
    int width, height;
    int time_step;  //Curent simulation index (starts from 0)
    double dx, dy;  //Physical distance 
    double dt;      //Physical time between steps
    double rx, ry;  //Numerical diffusion coefficients 
    double alpha;   //Diffusion coefficient
    float *u_current;  //Current state of the system
    float *u_next;     //Next state of the system
} Solver;

Solver *setup_solver(int width, int height, double dx, double dy, double dt, double alpha, float *u_curr);
void update_system(Solver *solver);