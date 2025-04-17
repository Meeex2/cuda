#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(val) check_cuda((val), #val, __FILE__, __LINE__)
#define CHECK_CUBLAS_ERROR(val) check_cublas((val), #val, __FILE__, __LINE__)

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

// Graph structure with normalization coefficients (CSR format)
struct Graph {
    int num_nodes;
    int num_edges;
    int *edge_ptr;       // Node pointer array
    int *edge_idx;       // Edge indices
    float *edge_norm;    // Normalization coefficients
};

// GCN layer parameters
struct GCNLayer {
    int input_dim;
    int output_dim;
    float *weights;
    float *bias;
};

// CPU implementation of GCN forward pass
void gcn_cpu(const Graph &graph, const float *features, const GCNLayer &layer, float *output) {
    // Compute normalization coefficients
    float *degrees = new float[graph.num_nodes]();
    for (int node = 0; node < graph.num_nodes; node++) {
        degrees[node] = sqrtf(graph.edge_ptr[node+1] - graph.edge_ptr[node]);
    }

    // Aggregate features with normalization
    float *aggregated = new float[graph.num_nodes * layer.input_dim]();
    for (int node = 0; node < graph.num_nodes; node++) {
        int start = graph.edge_ptr[node];
        int end = graph.edge_ptr[node+1];
        
        for (int e = start; e < end; e++) {
            int neighbor = graph.edge_idx[e];
            float norm = graph.edge_norm[e];
            
            for (int d = 0; d < layer.input_dim; d++) {
                aggregated[node * layer.input_dim + d] += 
                    features[neighbor * layer.input_dim + d] * norm;
            }
        }
    }

    // Apply linear transformation and ReLU
    for (int node = 0; node < graph.num_nodes; node++) {
        for (int out_d = 0; out_d < layer.output_dim; out_d++) {
            float sum = layer.bias[out_d];
            for (int in_d = 0; in_d < layer.input_dim; in_d++) {
                sum += aggregated[node * layer.input_dim + in_d] * 
                       layer.weights[in_d * layer.output_dim + out_d];
            }
            output[node * layer.output_dim + out_d] = fmaxf(sum, 0.0f);
        }
    }

    delete[] degrees;
    delete[] aggregated;
}

// GPU kernel to compute normalization coefficients
__global__ void compute_norm_kernel(const int *edge_ptr, const int *edge_idx,
                                   float *edge_norm, int num_nodes) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= num_nodes) return;

    int start = edge_ptr[node];
    int end = edge_ptr[node+1];
    float degree = sqrtf(end - start);

    for (int e = start; e < end; e++) {
        int neighbor = edge_idx[e];
        int nbr_degree = edge_ptr[neighbor+1] - edge_ptr[neighbor];
        edge_norm[e] = 1.0f / (degree * sqrtf(nbr_degree));
    }
}

// GPU kernel for feature aggregation
__global__ void gcn_aggregate_kernel(const int *edge_ptr, const int *edge_idx,
                                    const float *edge_norm, const float *features,
                                    int num_nodes, int feature_dim, float *aggregated) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= num_nodes) return;

    int start = edge_ptr[node];
    int end = edge_ptr[node+1];

    for (int e = start; e < end; e++) {
        int neighbor = edge_idx[e];
        float norm = edge_norm[e];
        
        for (int d = 0; d < feature_dim; d++) {
            atomicAdd(&aggregated[node * feature_dim + d], 
                     features[neighbor * feature_dim + d] * norm);
        }
    }
}

// GPU implementation using cuBLAS
void gcn_gpu(const Graph &graph, const float *h_features, const GCNLayer &layer,
             float *h_output, cublasHandle_t cublas_handle) {
    // Device memory pointers
    float *d_features, *d_aggregated, *d_weights, *d_bias, *d_output;
    int *d_edge_ptr, *d_edge_idx;
    float *d_edge_norm;

    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc(&d_edge_ptr, (graph.num_nodes + 1) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_edge_idx, graph.num_edges * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_edge_norm, graph.num_edges * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_features, graph.num_nodes * layer.input_dim * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_aggregated, graph.num_nodes * layer.input_dim * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_weights, layer.input_dim * layer.output_dim * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_bias, layer.output_dim * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, graph.num_nodes * layer.output_dim * sizeof(float)));

    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_edge_ptr, graph.edge_ptr, (graph.num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_edge_idx, graph.edge_idx, graph.num_edges * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_features, h_features, graph.num_nodes * layer.input_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_weights, layer.weights, layer.input_dim * layer.output_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_bias, layer.bias, layer.output_dim * sizeof(float), cudaMemcpyHostToDevice));

    // Compute normalization coefficients
    dim3 block_dim(256);
    dim3 grid_dim((graph.num_nodes + block_dim.x - 1) / block_dim.x);
    compute_norm_kernel<<<grid_dim, block_dim>>>(d_edge_ptr, d_edge_idx, d_edge_norm, graph.num_nodes);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Aggregate features
    gcn_aggregate_kernel<<<grid_dim, block_dim>>>(d_edge_ptr, d_edge_idx, d_edge_norm,
                                                d_features, graph.num_nodes, 
                                                layer.input_dim, d_aggregated);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Linear transformation: aggregated * weights^T + bias
    const float alpha = 1.0f, beta = 0.0f;
    CHECK_CUBLAS_ERROR(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                 layer.output_dim, graph.num_nodes, layer.input_dim,
                                 &alpha, d_weights, layer.input_dim,
                                 d_aggregated, layer.input_dim,
                                 &beta, d_output, layer.output_dim));

    // Add bias
    CHECK_CUBLAS_ERROR(cublasSaxpy(cublas_handle, graph.num_nodes * layer.output_dim,
                                 &alpha, d_bias, 1, d_output, 1));

    // Apply ReLU
    int output_size = graph.num_nodes * layer.output_dim;
    grid_dim = (output_size + block_dim.x - 1) / block_dim.x;
    relu_kernel<<<grid_dim, block_dim>>>(d_output, output_size);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Copy result back
    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost));

    // Cleanup
    CHECK_CUDA_ERROR(cudaFree(d_edge_ptr));
    CHECK_CUDA_ERROR(cudaFree(d_edge_idx));
    CHECK_CUDA_ERROR(cudaFree(d_edge_norm));
    CHECK_CUDA_ERROR(cudaFree(d_features));
    CHECK_CUDA_ERROR(cudaFree(d_aggregated));
    CHECK_CUDA_ERROR(cudaFree(d_weights));
    CHECK_CUDA_ERROR(cudaFree(d_bias));
    CHECK_CUDA_ERROR(cudaFree(d_output));
}

// Generate graph with self-loops and normalization coefficients
Graph generate_graph(int num_nodes, int avg_degree) {
    Graph graph;
    graph.num_nodes = num_nodes;
    graph.edge_ptr = new int[num_nodes + 1];
    graph.edge_idx = new int[num_nodes * (avg_degree + 1)]; // +1 for self-loop
    graph.edge_norm = new float[num_nodes * (avg_degree + 1)];
    
    int edge_count = 0;
    srand(time(NULL));
    
    for (int node = 0; node < num_nodes; node++) {
        graph.edge_ptr[node] = edge_count;
        
        // Add self-loop
        graph.edge_idx[edge_count] = node;
        edge_count++;
        
        // Add random edges
        for (int j = 0; j < avg_degree; j++) {
            graph.edge_idx[edge_count++] = rand() % num_nodes;
        }
    }
    graph.edge_ptr[num_nodes] = edge_count;
    graph.num_edges = edge_count;
    
    // Precompute normalization coefficients (CPU side)
    float *degrees = new float[num_nodes]();
    for (int node = 0; node < num_nodes; node++) {
        degrees[node] = sqrtf(graph.edge_ptr[node+1] - graph.edge_ptr[node]);
    }
    
    for (int node = 0; node < num_nodes; node++) {
        int start = graph.edge_ptr[node];
        int end = graph.edge_ptr[node+1];
        
        for (int e = start; e < end; e++) {
            int neighbor = graph.edge_idx[e];
            graph.edge_norm[e] = 1.0f / (degrees[node] * degrees[neighbor]);
        }
    }
    
    delete[] degrees;
    return graph;
}

// ... (Remaining functions same as previous GNN example with GCNLayer instead of GNNLayer)

int main() {
    // Test configurations
    run_gcn_test(1000, 32, 64, 5);
    run_gcn_test(10000, 64, 128, 10);
    run_gcn_test(50000, 128, 256, 15);
    return 0;
}