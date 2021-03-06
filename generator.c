#include "opencl_lab2.h"

int generate(int n) {
    float* src = malloc(n * sizeof(float));
    if (src == NULL) {
        printf("[generator] Not enough memory");
        return -1;
    }

    for (int i = 0; i < n; i++) {
        src[i] = (float) (rand() % 2) / 2;
    }

    FILE* F = fopen("generated.in", "wb");
    if (F == NULL) {
        printf("[generator] Can't open input file");
        free(src);
        return -1;
    }

    fprintf(F, "%i\n", n);
    for (int i = 0; i < n; i++) {
        fprintf(F, "%f ", src[i]);
    }
    fclose(F);

    float* dst = malloc(n * sizeof(float));
    if (dst == NULL) {
        printf("[generator] Not enough memory");
        return -1;
    }

    printf("Generating sample size %i...\n", n);

    float sum = 0;
    for (int i = 0; i < n; i++) {
        sum += src[i];
        dst[i] = sum;
    }
    free(src);

    F = fopen("generated_res.out", "wb");
    if (F == NULL) {
        printf("[generator] Can't open output file");
        free(dst);
        return -1;
    }

    fprintf(F, "%i\n", n);
    for (int i = 0; i < n; i++) {
        fprintf(F, "%f ", dst[i]);
    }
    fclose(F);

    printf("Generated\n");

    return 0;
}
