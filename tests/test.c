#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char** argv) {

    int i = 0;
    int x = 99;

#pragma omp parallel num_threads(6)
{
    int j = 0;
    printf("j = %d\n", j);

#pragma omp for
    for (i = 0; i < 12; i++)
        printf("Thread ID: %d, i = %d\n", omp_get_thread_num(), i);

    j = 66;
#pragma omp barrier
    printf("j = %d\n", j);
}

    printf("x = %d\n", x);

    return 0;
}
