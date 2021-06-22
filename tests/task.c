#include <stdio.h>

void foo () {
    int i;
#pragma omp parallel num_threads(6)
    {
#pragma omp task
    for (i = 0; i < 2; i++) {
        printf("This is task 1.\n");
    }
#pragma omp task
    for (i = 0; i < 3; i++) {
        printf("This is task 2.\n");
    }
    }
}
