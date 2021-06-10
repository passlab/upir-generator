
void foo () {
#pragma omp parallel num_threads(6)
    for (i = 0; i < 10; i++) {
        printf("This is a test.\n");
    }
}
