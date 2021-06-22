#include <stdio.h>

void foo () {
    int i, j;
#pragma omp parallel num_threads(6)
#pragma omp for collapse(2)
    for (i = 0; i < 2; i++) {
        for (j = 0; j < 3; j++) {
            printf("This is a test.\n");
        }
    }
}
