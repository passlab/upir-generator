#include <stdio.h>

void foo () {
    int i;
#pragma omp parallel num_threads(6)
    for (i = 0; i < 10; i++) {
        printf("This is a test.\n");
    }
}
