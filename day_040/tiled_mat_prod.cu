#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define TILE_SIZE 16

// GPU kernel for matrix multiplication
__global__ void tiled_mat_prod(float *out, float *in1, float *in2, int m) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    __shared__ float tile1[TILE_SIZE][TILE_SIZE];
    __shared__ float tile2[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    if (row >= m || col >= m) return;

    for (int i = 0; i < (m + TILE_SIZE - 1) / TILE_SIZE; i++) {
        int idx1 = row * m + (i * TILE_SIZE + tx);
        int idx2 = (i * TILE_SIZE + ty) * m + col;
        tile1[ty][tx] = (idx1 < m * m) ? in1[idx1] : 0.0f;
        tile2[ty][tx] = (idx2 < m * m) ? in2[idx2] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE && (i * TILE_SIZE + k) < m; k++) {
            sum += tile1[ty][k] * tile2[k][tx];
        }

        __syncthreads();
    }

    out[row * m + col] = sum;
}

// CPU matrix multiplication for verification
void cpu_mat_prod(float *out, float *in1, float *in2, int m) {
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < m; col++) {
            float sum = 0.0f;
            for (int k = 0; k < m; k++) {
                sum += in1[row * m + k] * in2[k * m + col];
            }
            out[row * m + col] = sum;
        }
    }
}

int main() {
    int m = 32;  // Matrix size (multiple of TILE_SIZE for simplicity)
    size_t size = m * m * sizeof(float);

    // Host arrays
    float *h_in1 = (float*)malloc(size);
    float *h_in2 = (float*)malloc(size);
    float *h_out_gpu = (float*)malloc(size);
    float *h_out_cpu = (float*)malloc(size);

    // Initialize input matrices
    for (int i = 0; i < m * m; i++) {
        h_in1[i] = (float)(i % 5);  // Simple pattern
        h_in2[i] = (float)((i + 1) % 3);
    }

    // Device arrays
    float *d_in1, *d_in2, *d_out;
    cudaMalloc(&d_in1, size);
    cudaMalloc(&d_in2, size);
    cudaMalloc(&d_out, size);

    // Copy to device
    cudaMemcpy(d_in1, h_in1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_in2, h_in2, size, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((m + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE);
    tiled_mat_prod<<<blocksPerGrid, threadsPerBlock>>>(d_out, d_in1, d_in2, m);
    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(h_out_gpu, d_out, size, cudaMemcpyDeviceToHost);

    // CPU computation
    cpu_mat_prod(h_out_cpu, h_in1, h_in2, m);

    // Verify results
    float max_error = 0.0f;
    for (int i = 0; i < m * m; i++) {
        float diff = fabs(h_out_gpu[i] - h_out_cpu[i]);
        max_error = fmax(max_error, diff);
        if (diff > 1e-5) {
            printf("Mismatch at index %d: GPU = %f, CPU = %f\n", i, h_out_gpu[i], h_out_cpu[i]);
        }
    }
    printf("Max error: %e\n", max_error);
    assert(max_error < 1e-5 && "GPU and CPU results differ!");

    // Cleanup
    cudaFree(d_in1); cudaFree(d_in2); cudaFree(d_out);
    free(h_in1); free(h_in2); free(h_out_gpu); free(h_out_cpu);

    printf("Test passed!\n");
    return 0;
}
