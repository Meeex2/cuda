#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <chrono>
#include <random>
#include <cassert>

#define BLOCK_SIZE 16
#define TOL 1e-4

void matrix_inverse_cpu(float* A, int n) {
    std::vector<float> aug(n * 2 * n, 0);
    
    
    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < n; ++j)
            aug[i * 2 * n + j] = A[i * n + j];
        aug[i * 2 * n + n + i] = 1.0f;
    }
    for(int col = 0; col < n; ++col) {
        
        int max_row = col;
        for(int i = col+1; i < n; ++i)
            if(fabs(aug[i * 2 * n + col]) > fabs(aug[max_row * 2 * n + col]))
                max_row = i;
        
        if(max_row != col)
            for(int j = col; j < 2 * n; ++j)
                std::swap(aug[col * 2 * n + j], aug[max_row * 2 * n + j]);
        
        float pivot = aug[col * 2 * n + col];
        for(int j = col; j < 2 * n; ++j)
            aug[col * 2 * n + j] /= pivot;
        
        for(int i = 0; i < n; ++i) {
            if(i != col) {
                float factor = aug[i * 2 * n + col];
                for(int j = col; j < 2 * n; ++j)
                    aug[i * 2 * n + j] -= factor * aug[col * 2 * n + j];
            }
        }
    }
    
    for(int i = 0; i < n; ++i)
        for(int j = 0; j < n; ++j)
            A[i * n + j] = aug[i * 2 * n + n + j];
}

