#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdio.h>

namespace cg = cooperative_groups;

const int BLOCK_SIZE = 256;
const int MAX_MATRIX_SIZE = 32; // Max 32x32 matrix (bit-packed in 32-bit int)

// Bit-packed row representation for GF(2)
typedef unsigned int gf2_row;

// Kernel: Parallel Gaussian Elimination for GF(2)
__global__ void gf2_nullspace_kernel(
    gf2_row* matrix,     // Input matrix (row-major, bit-packed)
    gf2_row* nullspace,  // Output basis vectors
    int n,               // Matrix dimension (n x n)
    int* rank            // Output matrix rank
) {
    cg::thread_block cta = cg::this_thread_block();
    __shared__ gf2_row smem[MAX_MATRIX_SIZE][MAX_MATRIX_SIZE/32 + 1];
    __shared__ int pivot_col[MAX_MATRIX_SIZE];
    
    // Load matrix into shared memory
    for(int i = threadIdx.x; i < n; i += blockDim.x) {
        smem[i][0] = matrix[i];
    }
    cta.sync();

    int current_rank = 0;
    
    // Gaussian elimination
    for(int col = 0; col < n; ++col) {
        // Find pivot row
        int pivot = -1;
        for(int row = current_rank; row < n; ++row) {
            if((smem[row][col/32] >> (col%32)) & 1) {
                pivot = row;
                break;
            }
        }
        
        if(pivot == -1) continue; // Free variable
        
        // Swap rows
        if(pivot != current_rank) {
            gf2_row temp = smem[current_rank][0];
            smem[current_rank][0] = smem[pivot][0];
            smem[pivot][0] = temp;
        }
        
        // Eliminate column
        for(int row = 0; row < n; ++row) {
            if(row != current_rank && ((smem[row][col/32] >> (col%32)) & 1) {
                smem[row][0] ^= smem[current_rank][0];
            }
        }
        
        pivot_col[current_rank] = col;
        current_rank++;
        cta.sync();
    }
    
    // Find nullspace basis
    if(threadIdx.x == 0) *rank = current_rank;
    const int nullity = n - current_rank;
    
    for(int var = threadIdx.x; var < nullity; var += blockDim.x) {
        gf2_row basis = 0;
        int free_var = 0;
        
        for(int col = 0; col < n; ++col) {
            if(free_var < nullity && col != pivot_col[free_var]) {
                basis |= (1 << col);
                free_var++;
            }
        }
        
        // Back substitution
        for(int i = current_rank-1; i >= 0; --i) {
            int pc = pivot_col[i];
            if((basis >> pc) & 1) {
                basis ^= smem[i][0];
            }
        }
        
        nullspace[var] = basis;
    }
}

// Host wrapper
void find_nullspace(gf2_row* matrix, gf2_row* nullspace, int n) {
    gf2_row* d_matrix, *d_nullspace;
    int* d_rank, h_rank;
    
    cudaMalloc(&d_matrix, n * sizeof(gf2_row));
    cudaMalloc(&d_nullspace, n * sizeof(gf2_row));
    cudaMalloc(&d_rank, sizeof(int));
    
    cudaMemcpy(d_matrix, matrix, n * sizeof(gf2_row), cudaMemcpyHostToDevice);
    
    gf2_nullspace_kernel<<<1, BLOCK_SIZE>>>(d_matrix, d_nullspace, n, d_rank);
    
    cudaMemcpy(&h_rank, d_rank, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(nullspace, d_nullspace, (n - h_rank) * sizeof(gf2_row), 
              cudaMemcpyDeviceToHost);
    
    cudaFree(d_matrix);
    cudaFree(d_nullspace);
    cudaFree(d_rank);
}

int main() {
    // Example: 4x4 matrix
    const int n = 4;
    gf2_row matrix[n] = {
        0b1101, // Row 0
        0b1010, // Row 1
        0b0110, // Row 2
        0b0001  // Row 3
    };
    
    gf2_row nullspace[n];
    
    find_nullspace(matrix, nullspace, n);
    
    printf("Nullspace basis vectors:\n");
    for(int i = 0; i < n; i++) {
        if(nullspace[i] == 0) break;
        printf("0x%04x\n", nullspace[i]);
    }
    
    return 0;
}