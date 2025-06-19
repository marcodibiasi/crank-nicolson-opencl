#include <stdio.h>
#include <stdlib.h>
#include "pgm_utils.h"
#include "solver.h"

void fast_debug(float* img, int n);

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Main arguments error");
        return EXIT_FAILURE;
    }

    /*
    FIRST STEP: IMAGE PREPARAION
    */

    /*
    Currently, only square heatmaps are supported
    x and y are kept separate for future extensions

    For the simulation to run correctly, the spatial dimensions must be treated equally
    that is, height = width, dx = dy
    */

    //Getting image information from the file
    int width, height;
    int *heatmap = pgm_loader(argv[1], &width, &height);
    if (!heatmap) 
        return EXIT_FAILURE;

    printf("\nLoaded %s -> %d x %d image\n", argv[1], width, height);

    float *norm_heatmap = pgm_normalisation(heatmap, width * height);
    //fast_debug(norm_img, 20);

    /*
    SECOND STEP: SETUP SOLVER (it does setup the OpenCL context as well)
    */

    Solver *solver = setup_solver(width, height, 1, 1, 0.2, 1.0, norm_heatmap);

    /*
    THIRST STEP: RUN SIMULATION
    */

    //Keeping the image conversion logic outside the simulation
    int n_steps = 10;
    float **frames = malloc(n_steps * sizeof(float*));
    for (size_t i = 0; i < n_steps; i++)
        frames[i] = malloc(width * height * sizeof(float));
    
    /*
    The simulation returns n frames needed for visualization and analysis
    In such way it is possible to visualize the full heatmap evolution or just a specific step
    Additionally, it is better to not save the image inside the main loop, to avoid overhead
    */
    frames = run_simulation(solver, n_steps);

    /*
    FORTH STEP: FREE RESOURCES
    */

    free(heatmap);
    free(norm_heatmap);
    free_solver(solver);
    return 0;
}

//n = number of pixel to print
void fast_debug(float* img, int n){
    for(int i = 0; i < n; ++i){
        printf(" %f ", img[i]);
    }
}