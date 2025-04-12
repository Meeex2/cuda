#ifndef KURTOSIS_H
#define KURTOSIS_H

#include <stddef.h>

// CPU implementation
float kurtosis_cpu(const float* data, size_t n);
void moments_cpu(const float* data, size_t n, float* mean, float* variance, float* skewness, float* kurtosis);

// GPU implementation
float kurtosis_gpu(const float* data, size_t n);
void moments_gpu(const float* data, size_t n, float* mean, float* variance, float* skewness, float* kurtosis);

// Utility functions
float* generate_random_data(size_t n);
int validate_results(float cpu_result, float gpu_result, float tolerance);
void print_statistics(const float* data, size_t n, const char* label);

#endif // KURTOSIS_H