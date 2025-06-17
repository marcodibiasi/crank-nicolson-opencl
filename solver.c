#include <stdio.h>
#include <stdlib.h>
#include "solver.h"

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
        free(solver);
        return;
    }
}
