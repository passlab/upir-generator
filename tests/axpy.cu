
__global__ void axpy_kernel(float* x, float* y, int a, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        y[i] = y[i] + a * x[i];
    }
}

void axpy(float* d_x, float* d_y, int a, int n) {

    // TODO: dim: (n+255)/256, 256
    axpy_kernel<<<1024, 256>>>(d_x, d_y, a, n);

}
