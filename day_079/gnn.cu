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

// Graph structure (CSR format)
struct Graph {
    int num_nodes;
    int num_edges;
    int *edge_ptr;
    int *edge_idx;
};

// GNN layer parameters
struct GNNLayer {
    int input_dim;
    int output_dim;
    float *weights;
    float *bias;
};

// CPU implementation of GNN forward pass
void gnn_cpu(const Graph &graph, const float *features, const GNNLayer &layer,
             float *output) {
    // Step 1: Aggregate neighbor features
    float *aggregated = new float[graph.num_nodes * layer.input_dim]();
    
    for (int node = 0; node < graph.num_nodes; node++) {
        int start = graph.edge_ptr[node];
        int end = graph.edge_ptr[node+1];
        int degree = end - start;
        
        // Sum neighbor features (including self-loop)
        for (int d = 0; d < layer.input_dim; d++) {
            aggregated[node * layer.input_dim + d] = features[node * layer.input_dim + d];
        }
        
        for (int nbr = start; nbr < end; nbr++) {
            int neighbor = graph.edge_idx[nbr];
            for (int d = 0; d < layer.input_dim; d++) {
                aggregated[node * layer.input_dim + d] += features[neighbor * layer.input_dim + d];
            }
        }
        
        // Normalize by degree + 1 (self-loop)
        if (degree > 0) {
            for (int d = 0; d < layer.input_dim; d++) {
                aggregated[node * layer.input_dim + d] /= (degree + 1);
            }
        }
    }

    // Step 2: Apply linear transformation
    for (int node = 0; node < graph.num_nodes; node++) {
        for (int out_d = 0; out_d < layer.output_dim; out_d++) {
            float sum = layer.bias[out_d];
            for (int in_d = 0; in_d < layer.input_dim; in_d++) {
                sum += aggregated[node * layer.input_dim + in_d] * 
                       layer.weights[in_d * layer.output_dim + out_d];
            }
            output[node * layer.output_dim + out_d] = fmaxf(sum, 0);  // ReLU
        }
    }

    delete[] aggregated;
}

// GPU kernel for neighbor aggregation
__global__ void aggregate_kernel(const int *edge_ptr, const int *edge_idx,
                                 const float *features, int num_nodes,
                                 int feature_dim, float *aggregated) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= num_nodes) return;

    int start = edge_ptr[node];
    int end = edge_ptr[node+1];
    int degree = end - start;

    // Start with self features
    for (int d = 0; d < feature_dim; d++) {
        aggregated[node * feature_dim + d] = features[node * feature_dim + d];
    }

    // Sum neighbor features
    for (int nbr = start; nbr < end; nbr++) {
        int neighbor = edge_idx[nbr];
        for (int d = 0; d < feature_dim; d++) {
            atomicAdd(&aggregated[node * feature_dim + d], 
                     features[neighbor * feature_dim + d]);
        }
    }

    // Normalize
    if (degree > 0) {
        for (int d = 0; d < feature_dim; d++) {
            aggregated[node * feature_dim + d] /= (degree + 1);
        }
    }
}

// GPU kernel for ReLU activation
__global__ void relu_kernel(float *output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    output[idx] = fmaxf(output[idx], 0);
}

// GPU implementation using cuBLAS for matrix multiplication
void gnn_gpu(const Graph &graph, const float *h_features, const GNNLayer &layer,
             float *h_output, cublasHandle_t cublas_handle) {
    // Device memory pointers
    float *d_features, *d_aggregated, *d_weights, *d_bias, *d_output;
    int *d_edge_ptr, *d_edge_idx;

    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc(&d_edge_ptr, (graph.num_nodes + 1) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_edge_idx, graph.num_edges * sizeof(int)));
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

    // Step 1: Aggregate features
    dim3 block_dim(256);
    dim3 grid_dim((graph.num_nodes + block_dim.x - 1) / block_dim.x);
    aggregate_kernel<<<grid_dim, block_dim>>>(d_edge_ptr, d_edge_idx, d_features,
                                            graph.num_nodes, layer.input_dim,
                                            d_aggregated);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Step 2: Linear transformation (aggregated * weights^T + bias)
    const float alpha = 1.0f;
    const float beta = 1.0f;
    CHECK_CUBLAS_ERROR(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                 layer.output_dim, graph.num_nodes, layer.input_dim,
                                 &alpha, d_weights, layer.output_dim,
                                 d_aggregated, layer.input_dim,
                                 &beta, d_output, layer.output_dim));

    // Add bias
    CHECK_CUBLAS_ERROR(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 layer.output_dim, graph.num_nodes, 1,
                                 &alpha, d_bias, layer.output_dim,
                                 d_ones, 1,
                                 &alpha, d_output, layer.output_dim));

    // Step 3: Apply ReLU
    int output_size = graph.num_nodes * layer.output_dim;
    grid_dim = (output_size + block_dim.x - 1) / block_dim.x;
    relu_kernel<<<grid_dim, block_dim>>>(d_output, output_size);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Copy result back
    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, graph.num_nodes * layer.output_dim * sizeof(float), cudaMemcpyDeviceToHost));

    // Cleanup
    CHECK_CUDA_ERROR(cudaFree(d_edge_ptr));
    CHECK_CUDA_ERROR(cudaFree(d_edge_idx));
    CHECK_CUDA_ERROR(cudaFree(d_features));
    CHECK_CUDA_ERROR(cudaFree(d_aggregated));
    CHECK_CUDA_ERROR(cudaFree(d_weights));
    CHECK_CUDA_ERROR(cudaFree(d_bias));
    CHECK_CUDA_ERROR(cudaFree(d_output));
}

// Generate random graph
Graph generate_graph(int num_nodes, int avg_degree) {
    Graph graph;
    graph.num_nodes = num_nodes;
    graph.edge_ptr = new int[num_nodes + 1];
    graph.edge_idx = new int[num_nodes * avg_degree];
    
    int edge_count = 0;
    srand(time(NULL));
    
    for (int i = 0; i < num_nodes; i++) {
        graph.edge_ptr[i] = edge_count;
        
        // Add self-loop
        graph.edge_idx[edge_count++] = i;
        
        // Add random edges
        for (int j = 0; j < avg_degree; j++) {
            int random_node = rand() % num_nodes;
            graph.edge_idx[edge_count++] = random_node;
        }
    }
    graph.edge_ptr[num_nodes] = edge_count;
    graph.num_edges = edge_count;
    
    return graph;
}

// Generate random layer parameters
GNNLayer init_layer(int input_dim, int output_dim) {
    GNNLayer layer;
    layer.input_dim = input_dim;
    layer.output_dim = output_dim;
    layer.weights = new float[input_dim * output_dim];
    layer.bias = new float[output_dim];
    
    srand(time(NULL));
    for (int i = 0; i < input_dim * output_dim; i++) {
        layer.weights[i] = (float)rand() / RAND_MAX * 2 - 1;  // [-1, 1]
    }
    for (int i = 0; i < output_dim; i++) {
        layer.bias[i] = (float)rand() / RAND_MAX * 0.1f;  // Small bias
    }
    
    return layer;
}

// Compare outputs with tolerance
int compare_outputs(const float *a, const float *b, int size, float tolerance) {
    for (int i = 0; i < size; i++) {
        if (fabs(a[i] - b[i]) > tolerance) {
            printf("Mismatch at %d: %.4f vs %.4f\n", i, a[i], b[i]);
            return 0;
        }
    }
    return 1;
}

void run_gnn_test(int num_nodes, int feature_dim, int hidden_dim, int avg_degree) {
    printf("\nRunning GNN test with %d nodes, %d features, hidden dim %d\n",
           num_nodes, feature_dim, hidden_dim);
    
    // Initialize data
    Graph graph = generate_graph(num_nodes, avg_degree);
    GNNLayer layer = init_layer(feature_dim, hidden_dim);
    float *features = new float[num_nodes * feature_dim];
    float *cpu_output = new float[num_nodes * hidden_dim];
    float *gpu_output = new float[num_nodes * hidden_dim];
    
    // Generate random features
    srand(time(NULL));
    for (int i = 0; i < num_nodes * feature_dim; i++) {
        features[i] = (float)rand() / RAND_MAX;
    }

    // Create cuBLAS handle
    cublasHandle_t cublas_handle;
    CHECK_CUBLAS_ERROR(cublasCreate(&cublas_handle));

    // CPU implementation
    clock_t cpu_start = clock();
    gnn_cpu(graph, features, layer, cpu_output);
    double cpu_time = (double)(clock() - cpu_start) / CLOCKS_PER_SEC;

    // GPU implementation
    clock_t gpu_start = clock();
    gnn_gpu(graph, features, layer, gpu_output, cublas_handle);
    double gpu_time = (double)(clock() - gpu_start) / CLOCKS_PER_SEC;

    // Validate
    int valid = compare_outputs(cpu_output, gpu_output, num_nodes * hidden_dim, 1e-3f);

    printf("\nResults:\n");
    printf("CPU Time: %.4f sec\n", cpu_time);
    printf("GPU Time: %.4f sec\n", gpu_time);
    printf("Speedup: %.2fx\n", cpu_time / gpu_time);
    printf("Validation: %s\n", valid ? "PASSED" : "FAILED");

    // Cleanup
    delete[] graph.edge_ptr;
    delete[] graph.edge_idx;
    delete[] layer.weights;
    delete[] layer.bias;
    delete[] features;
    delete[] cpu_output;
    delete[] gpu_output;
    CHECK_CUBLAS_ERROR(cublasDestroy(cublas_handle)));
}

int main() {
    // Test configurations
    run_gnn_test(1000, 32, 64, 5);
    run_gnn_test(10000, 64, 128, 10);
    run_gnn_test(50000, 128, 256, 15);

    return 0;
}