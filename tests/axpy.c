
void foo (float* x, float* y, int a, int n) {
#pragma omp parallel for num_threads(6)
    for (int i = 0; i < n; i++) {
        y[i] = y[i] + a * x[i];
    }
}
