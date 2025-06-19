#include "pgm_utils.h"
#include <stdio.h>
#include <stdlib.h>

int *pgm_loader(const char *filename, int *width, int *height) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        perror("Error opening file");
        return NULL;
    }

    if (fscanf(fp, "%d %d", width, height) != 2) {
        fprintf(stderr, "Missing image size\n");
        fclose(fp);
        return NULL;
    }

    //Allocating memory for the image
    int *image = malloc((*width) * (*height) * sizeof(int));
    if (!image) {
        fprintf(stderr, "Error allocating memory\n");
        fclose(fp);
        return NULL;
    }

    //Copying image
    for (int i = 0; i < (*width) * (*height); ++i) {
        if (fscanf(fp, "%d", &image[i]) != 1) {
            fprintf(stderr, "Error reading pixel\n");
            free(image);
            fclose(fp);
            return NULL;
        }
    }

    fclose(fp);
    return image;
}

float *pgm_normalisation(int* matrix, int n) {
    float* normalized = malloc(n * sizeof(float));

    if (!normalized) {
        fprintf(stderr, "Error allocating memory\n");
        return NULL;
    }

    for (int i = 0; i < n; i++) {
        normalized[i] = matrix[i] / 255.0f;
    }

    return normalized;
}

/*
TODO: Conversion from the simulation frames to the image format
    - Convert the values from [0, 1] range to [0, 255] range
    - Convert all the frames from 1D array to 2D matrix
    - Create (if needed) a directory for the output images
    - Save the frames as PNG files into the directory (additional: PGM format support)
*/