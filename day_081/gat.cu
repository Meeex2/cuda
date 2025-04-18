#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define CHECK_CUDA_ERROR(val) check_cuda((val), #val, __FILE__, __LINE__)
#define CHECK_CUBLAS_ERROR(val) check_cublas((val), #val, __FILE__, __LINE__)
#define CHECK_CURAND_ERROR(val) check_curand((val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_t result, const char* func, const char* file, int line) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n",
                file, line, (int)result, cudaGetErrorString(result), func);
        exit(EXIT_FAILURE);
    }
}

void check_cublas(cublasStatus_t result, const char* func, const char* file, int line) {
    if (result != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "CUBLAS error at %s:%d code=%d \"%s\"\n",
                file, line, (int)result, func);
        exit(EXIT_FAILURE);
    }
}

void check_curand(curandStatus_t result, const char* func, const char* file, int line) {
    if (result != CURAND_STATUS_SUCCESS) {
        fprintf(stderr, "CURAND error at %s:%d code=%d \"%s\"\n",
                file, line, (int)result, func);
        exit(EXIT_FAILURE);
    }
}

// Graph structure (CSR format)
struct Graph {
    int num_nodes;
    int num_edges;
    int *edge_ptr;  // Node pointer array
    int *edge_idx;  // Edge indices
};

// GAT layer parameters
struct GATLayer {
    int input_dim;
    int output_dim;
    int num_heads;
    float negative_slope;  // LeakyReLU parameter
    float *weights;        // [num_heads, input_dim, output_dim]
    float *attn_weights;   // [num_heads, 2*output_dim]
    float *bias;          // [num_heads, output_dim]
};

// CPU implementation of GAT forward pass
void gat_cpu(const Graph &graph, const float *features, const GATLayer &layer, float *output) {
    int head_size = layer.output_dim;
    int total_out_dim = layer.num_heads * head_size;
    
    // Temporary storage
    float *transformed = new float[graph.num_nodes * layer.num_heads * head_size]();
    float *attn_scores = new float[layer.num_heads * graph.num_edges]();
    
    // Step 1: Linear transformation for each head
    for (int node = 0; node < graph.num_nodes; node++) {
        for (int h = 0; h < layer.num_heads; h++) {
            for (int out_d = 0; out_d < head_size; out_d++) {
                float sum = layer.bias[h * head_size + out_d];
                for (int in_d = 0; in_d < layer.input_dim; in_d++) {
                    sum += features[node * layer.input_dim + in_d] * 
                           layer.weights[(h * layer.input_dim + in_d) * head_size + out_d];
                }
                transformed[(node * layer.num_heads + h) * head_size + out_d] = sum;
            }
        }
    }
    
    // Step 2: Compute attention scores
    for (int h = 0; h < layer.num_heads; h++) {
        for (int node = 0; node < graph.num_nodes; node++) {
            int start = graph.edge_ptr[node];
            int end = graph.edge_ptr[node+1];
            
            for (int e = start; e < end; e++) {
                int neighbor = graph.edge_idx[e];
                float score = 0.0f;
                
                // Compute attention score (a^T[Wh_i || Wh_j])
                for (int d = 0; d < head_size; d++) {
                    float hi = transformed[(node * layer.num_heads + h) * head_size + d];
                    float hj = transformed[(neighbor * layer.num_heads + h) * head_size + d];
                    score += layer.attn_weights[h * 2 * head_size + d] * hi;
                    score += layer.attn_weights[h * 2 * head_size + head_size + d] * hj;
                }
                
                // LeakyReLU
                score = (score > 0) ? score : score * layer.negative_slope;
                attn_scores[h * graph.num_edges + e] = score;
            }
        }
    }
    
    // Step 3: Softmax attention scores per node
    for (int h = 0; h < layer.num_heads; h++) {
        for (int node = 0; node < graph.num_nodes; node++) {
            int start = graph.edge_ptr[node];
            int end = graph.edge_ptr[node+1];
            
            // Find max for numerical stability
            float max_score = -INFINITY;
            for (int e = start; e < end; e++) {
                float score = attn_scores[h * graph.num_edges + e];
                if (score > max_score) max_score = score;
            }
            
            // Compute exp and sum
            float sum_exp = 0.0f;
            for (int e = start; e < end; e++) {
                float score = attn_scores[h * graph.num_edges + e] - max_score;
                score = expf(score);
                attn_scores[h * graph.num_edges + e] = score;
                sum_exp += score;
            }
            
            // Normalize
            for (int e = start; e < end; e++) {
                attn_scores[h * graph.num_edges + e] /= sum_exp;
            }
        }
    }
    
    // Step 4: Weighted aggregation
    for (int node = 0; node < graph.num_nodes; node++) {
        for (int h = 0; h < layer.num_heads; h++) {
            for (int d = 0; d < head_size; d++) {
                float sum = 0.0f;
                int start = graph.edge_ptr[node];
                int end = graph.edge_ptr[node+1];
                
                for (int e = start; e < end; e++) {
                    int neighbor = graph.edge_idx[e];
                    float alpha = attn_scores[h * graph.num_edges + e];
                    sum += alpha * transformed[(neighbor * layer.num_heads + h) * head_size + d];
                }
                
                output[node * total_out_dim + h * head_size + d] = sum;
            }
        }
    }
    
    delete[] transformed;
    delete[] attn_scores;
}

// GPU kernel for linear transformation
__global__ void linear_transform_kernel(const float *features, const float *weights, 
                                       const float *bias, int num_nodes, int input_dim,
                                       int output_dim, int num_heads, float *transformed) {
    int node = blockIdx.y;
    int h = blockIdx.z;
    int out_d = threadIdx.x;
    
    if (node >= num_nodes || h >= num_heads || out_d >= output_dim) return;
    
    float sum = bias[h * output_dim + out_d];
    for (int in_d = 0; in_d < input_dim; in_d++) {
        sum += features[node * input_dim + in_d] * 
               weights[(h * input_dim + in_d) * output_dim + out_d];
    }
    
    transformed[(node * num_heads + h) * output_dim + out_d] = sum;
}

// GPU kernel for attention score computation
__global__ void compute_attention_scores_kernel(const int *edge_ptr, const int *edge_idx,
                                              const float *transformed, const float *attn_weights,
                                              int num_nodes, int num_edges, int head_size,
                                              int num_heads, float negative_slope,
                                              float *attn_scores) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y;
    
    if (e >= num_edges || h >= num_heads) return;
    
    // Find node for this edge
    int node = 0;
    while (e >= edge_ptr[node+1]) node++;
    
    int neighbor = edge_idx[e];
    float score = 0.0f;
    
    // Compute attention score (a^T[Wh_i || Wh_j])
    for (int d = 0; d < head_size; d++) {
        float hi = transformed[(node * num_heads + h) * head_size + d];
        float hj = transformed[(neighbor * num_heads + h) * head_size + d];
        score += attn_weights[h * 2 * head_size + d] * hi;
        score += attn_weights[h * 2 * head_size + head_size + d] * hj;
    }
    
    // LeakyReLU
    attn_scores[h * num_edges + e] = (score > 0) ? score : score * negative_slope;
}

// GPU kernel for softmax normalization
__global__ void softmax_kernel(const int *edge_ptr, float *attn_scores,
                              int num_nodes, int num_edges, int num_heads) {
    int node = blockIdx.x;
    int h = blockIdx.y;
    
    if (node >= num_nodes || h >= num_heads) return;
    
    int start = edge_ptr[node];
    int end = edge_ptr[node+1];
    
    // Find max for numerical stability
    float max_score = -INFINITY;
    for (int e = start; e < end; e++) {
        float score = attn_scores[h * num_edges + e];
        if (score > max_score) max_score = score;
    }
    
    // Compute exp and sum
    float sum_exp = 0.0f;
    for (int e = start; e < end; e++) {
        float score = attn_scores[h * num_edges + e] - max_score;
        score = expf(score);
        attn_scores[h * num_edges + e] = score;
        sum_exp += score;
    }
    
    // Normalize
    for (int e = start; e < end; e++) {
        attn_scores[h * num_edges + e] /= sum_exp;
    }
}

// GPU kernel for weighted aggregation
__global__ void aggregate_kernel(const int *edge_ptr, const int *edge_idx,
                                const float *transformed, const float *attn_scores,
                                int num_nodes, int head_size, int num_heads,
                                float *output) {
    int node = blockIdx.x;
    int h = blockIdx.y;
    int d = threadIdx.x;
    
    if (node >= num_nodes || h >= num_heads || d >= head_size) return;
    
    int start = edge_ptr[node];
    int end = edge_ptr[node+1];
    float sum = 0.0f;
    
    for (int e = start; e < end; e++) {
        int neighbor = edge_idx[e];
        float alpha = attn_scores[h * (gridDim.x + 1) + e]; // gridDim.x = num_edges
        sum += alpha * transformed[(neighbor * num_heads + h) * head_size + d];
    }
    
    output[node * num_heads * head_size + h * head_size + d] = sum;
}

// GPU implementation of GAT
void gat_gpu(const Graph &graph, const float *h_features, const GATLayer &layer,
             float *h_output, cublasHandle_t cublas_handle) {
    // Device memory pointers
    float *d_features, *d_transformed, *d_attn_scores, *d_output;
    int *d_edge_ptr, *d_edge_idx;
    float *d_weights, *d_attn_weights, *d_bias;
    
    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc(&d_edge_ptr, (graph.num_nodes + 1) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_edge_idx, graph.num_edges * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_features, graph.num_nodes * layer.input_dim * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_transformed, graph.num_nodes * layer.num_heads * layer.output_dim * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_attn_scores, layer.num_heads * graph.num_edges * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, graph.num_nodes * layer.num_heads * layer.output_dim * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_weights, layer.num_heads * layer.input_dim * layer.output_dim * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_attn_weights, layer.num_heads * 2 * layer.output_dim * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_bias, layer.num_heads * layer.output_dim * sizeof(float)));
    
    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_edge_ptr, graph.edge_ptr, (graph.num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_edge_idx, graph.edge_idx, graph.num_edges * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_features, h_features, graph.num_nodes * layer.input_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_weights, layer.weights, layer.num_heads * layer.input_dim * layer.output_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_attn_weights, layer.attn_weights, layer.num_heads * 2 * layer.output_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_bias, layer.bias, layer.num_heads * layer.output_dim * sizeof(float), cudaMemcpyHostToDevice));
    
    // Step 1: Linear transformation for each head
    dim3 block_dim(layer.output_dim);
    dim3 grid_dim(1, graph.num_nodes, layer.num_heads);
    linear_transform_kernel<<<grid_dim, block_dim>>>(d_features, d_weights, d_bias,
                                                   graph.num_nodes, layer.input_dim,
                                                   layer.output_dim, layer.num_heads,
                                                   d_transformed);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Step 2: Compute attention scores
    block_dim = dim3(256);
    grid_dim = dim3((graph.num_edges + block_dim.x - 1) / block_dim.x, layer.num_heads);
    compute_attention_scores_kernel<<<grid_dim, block_dim>>>(d_edge_ptr, d_edge_idx,
                                                           d_transformed, d_attn_weights,
                                                           graph.num_nodes, graph.num_edges,
                                                           layer.output_dim, layer.num_heads,
                                                           layer.negative_slope,
                                                           d_attn_scores);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Step 3: Softmax attention scores per node
    grid_dim = dim3(graph.num_nodes, layer.num_heads);
    softmax_kernel<<<grid_dim, 1>>>(d_edge_ptr, d_attn_scores,
                                   graph.num_nodes, graph.num_edges, layer.num_heads);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Step 4: Weighted aggregation
    block_dim = dim3(layer.output_dim);
    grid_dim = dim3(graph.num_nodes, layer.num_heads);
    aggregate_kernel<<<grid_dim, block_dim>>>(d_edge_ptr, d_edge_idx,
                                            d_transformed, d_attn_scores,
                                            graph.num_nodes, layer.output_dim,
                                            layer.num_heads, d_output);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Copy result back
    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, 
                               graph.num_nodes * layer.num_heads * layer.output_dim * sizeof(float),
                               cudaMemcpyDeviceToHost));
    
    // Cleanup
    CHECK_CUDA_ERROR(cudaFree(d_edge_ptr));
    CHECK_CUDA_ERROR(cudaFree(d_edge_idx));
    CHECK_CUDA_ERROR(cudaFree(d_features));
    CHECK_CUDA_ERROR(cudaFree(d_transformed));
    CHECK_CUDA_ERROR(cudaFree(d_attn_scores));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    CHECK_CUDA_ERROR(cudaFree(d_weights));
    CHECK_CUDA_ERROR(cudaFree(d_attn_weights));
    CHECK_CUDA_ERROR(cudaFree(d_bias));
}

// ... (Graph generation, layer initialization, validation, and test functions similar to previous examples)

int main() {
    // Initialize cuBLAS handle
    cublasHandle_t cublas_handle;
    CHECK_CUBLAS_ERROR(cublasCreate(&cublas_handle));
    
    // Test configurations
    run_gat_test(1000, 32, 64, 4, 0.2f);  // nodes, in_dim, out_dim, heads, negative_slope
    run_gat_test(10000, 64, 128, 8, 0.2f);
    run_gat_test(50000, 128, 256, 8, 0.2f);
    
    // Cleanup
    CHECK_CUBLAS_ERROR(cublasDestroy(cublas_handle));
    return 0;
}