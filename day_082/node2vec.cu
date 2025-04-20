#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>

#define CHECK_CUDA_ERROR(val) check_cuda((val), #val, __FILE__, __LINE__)
#define CHECK_CURAND_ERROR(val) check_curand((val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_t result, const char* func, const char* file, int line) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n",
                file, line, (int)result, cudaGetErrorString(result), func);
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
    int *edge_ptr;
    int *edge_idx;
    float *edge_weight;
};

// Node2Vec parameters
struct Node2VecParams {
    int walk_length;
    int walks_per_node;
    int p;  // Return parameter
    int q;  // In-out parameter
    int window_size;
    int embedding_dim;
};

// Alias sampling tables
struct AliasTables {
    float *prob;
    int *alias;
    int *edge_offset;
    int table_size;
};

// CPU implementation of random walks
void node2vec_walks_cpu(const Graph &graph, const Node2VecParams &params, int *walks) {
    srand(time(NULL));
    
    // Precompute transition probabilities
    AliasTables tables;
    tables.table_size = graph.num_edges * 2;  // For both forward and backward transitions
    tables.prob = new float[tables.table_size];
    tables.alias = new int[tables.table_size];
    tables.edge_offset = new int[graph.num_nodes + 1];
    
    // ... (Alias table construction omitted for brevity)
    
    for (int node = 0; node < graph.num_nodes; node++) {
        for (int w = 0; w < params.walks_per_node; w++) {
            int current = node;
            walks[(node * params.walks_per_node + w) * params.walk_length] = current;
            
            int prev = -1;  // Previous node
            
            for (int step = 1; step < params.walk_length; step++) {
                // Get neighbors
                int start = graph.edge_ptr[current];
                int end = graph.edge_ptr[current + 1];
                int degree = end - start;
                
                if (degree == 0) break;
                
                int next;
                if (prev == -1) {
                    // Unbiased random walk
                    next = graph.edge_idx[start + (rand() % degree)];
                } else {
                    // Node2Vec biased walk
                    // ... (Biased sampling implementation omitted)
                }
                
                walks[(node * params.walks_per_node + w) * params.walk_length + step] = next;
                prev = current;
                current = next;
            }
        }
    }
    
    delete[] tables.prob;
    delete[] tables.alias;
    delete[] tables.edge_offset;
}

// GPU kernel for alias sampling
__device__ int alias_sample(curandState *state, float *prob, int *alias, int start, int size) {
    float r1 = curand_uniform(state);
    float r2 = curand_uniform(state);
    
    int idx = start + (int)(r1 * size);
    return (r2 < prob[idx]) ? idx : alias[idx];
}

// GPU kernel for Node2Vec walks
__global__ void node2vec_walks_kernel(const int *edge_ptr, const int *edge_idx, 
                                     const float *edge_weight, const AliasTables tables,
                                     int num_nodes, int walk_length, int walks_per_node,
                                     int p, int q, int *walks, curandState *states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes * walks_per_node) return;
    
    int node = idx / walks_per_node;
    int walk_id = idx % walks_per_node;
    curandState local_state = states[idx];
    
    int current = node;
    walks[idx * walk_length] = current;
    int prev = -1;
    
    for (int step = 1; step < walk_length; step++) {
        int start = edge_ptr[current];
        int end = edge_ptr[current + 1];
        int degree = end - start;
        
        if (degree == 0) break;
        
        int next;
        if (prev == -1) {
            // Unbiased random walk
            next = edge_idx[start + (curand(&local_state) % degree)];
        } else {
            // Node2Vec biased walk
            int prev_start = edge_ptr[prev];
            int prev_end = edge_ptr[prev + 1];
            
            // Find common neighbors
            // ... (Biased sampling implementation omitted)
            
            // Use alias sampling for efficient biased sampling
            int table_start = tables.edge_offset[current];
            int table_size = tables.edge_offset[current + 1] - table_start;
            int sampled_idx = alias_sample(&local_state, tables.prob, tables.alias, table_start, table_size);
            next = edge_idx[sampled_idx];
        }
        
        walks[idx * walk_length + step] = next;
        prev = current;
        current = next;
    }
    
    states[idx] = local_state;
}

// GPU implementation of Node2Vec walks
void node2vec_walks_gpu(const Graph &graph, const Node2VecParams &params, int *h_walks) {
    // Device memory pointers
    int *d_edge_ptr, *d_edge_idx, *d_walks;
    float *d_edge_weight;
    curandState *d_states;
    
    // Alias tables on device
    AliasTables d_tables;
    
    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc(&d_edge_ptr, (graph.num_nodes + 1) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_edge_idx, graph.num_edges * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_edge_weight, graph.num_edges * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_walks, graph.num_nodes * params.walks_per_node * params.walk_length * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_states, graph.num_nodes * params.walks_per_node * sizeof(curandState)));
    
    // Copy graph data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_edge_ptr, graph.edge_ptr, (graph.num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_edge_idx, graph.edge_idx, graph.num_edges * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_edge_weight, graph.edge_weight, graph.num_edges * sizeof(float), cudaMemcpyHostToDevice));
    
    // Build alias tables on device
    // ... (Alias table construction on GPU omitted)
    
    // Initialize random states
    int threads_per_block = 256;
    int blocks = (graph.num_nodes * params.walks_per_node + threads_per_block - 1) / threads_per_block;
    
    // Initialize CURAND states
    setup_curand_kernel<<<blocks, threads_per_block>>>(d_states, time(NULL));
    
    // Launch Node2Vec walks kernel
    node2vec_walks_kernel<<<blocks, threads_per_block>>>(
        d_edge_ptr, d_edge_idx, d_edge_weight, d_tables,
        graph.num_nodes, params.walk_length, params.walks_per_node,
        params.p, params.q, d_walks, d_states
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Copy walks back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_walks, d_walks, 
                               graph.num_nodes * params.walks_per_node * params.walk_length * sizeof(int),
                               cudaMemcpyDeviceToHost));
    
    // Cleanup
    CHECK_CUDA_ERROR(cudaFree(d_edge_ptr));
    CHECK_CUDA_ERROR(cudaFree(d_edge_idx));
    CHECK_CUDA_ERROR(cudaFree(d_edge_weight));
    CHECK_CUDA_ERROR(cudaFree(d_walks));
    CHECK_CUDA_ERROR(cudaFree(d_states));
    // ... (Free alias tables)
}

// Generate a random graph
Graph generate_graph(int num_nodes, int avg_degree) {
    Graph graph;
    graph.num_nodes = num_nodes;
    graph.edge_ptr = new int[num_nodes + 1];
    graph.edge_idx = new int[num_nodes * avg_degree];
    graph.edge_weight = new float[num_nodes * avg_degree];
    
    int edge_count = 0;
    srand(time(NULL));
    
    for (int i = 0; i < num_nodes; i++) {
        graph.edge_ptr[i] = edge_count;
        
        // Add random edges
        for (int j = 0; j < avg_degree; j++) {
            graph.edge_idx[edge_count] = rand() % num_nodes;
            graph.edge_weight[edge_count] = (float)rand() / RAND_MAX;  // Random weight
            edge_count++;
        }
    }
    graph.edge_ptr[num_nodes] = edge_count;
    graph.num_edges = edge_count;
    
    return graph;
}

// Validate walks
int validate_walks(const Graph &graph, const int *walks, int num_walks, int walk_length) {
    for (int i = 0; i < num_walks; i++) {
        for (int j = 1; j < walk_length; j++) {
            int current = walks[i * walk_length + j - 1];
            int next = walks[i * walk_length + j];
            
            // Check if next is a neighbor of current
            int valid = 0;
            for (int e = graph.edge_ptr[current]; e < graph.edge_ptr[current + 1]; e++) {
                if (graph.edge_idx[e] == next) {
                    valid = 1;
                    break;
                }
            }
            
            if (!valid && next != -1) {
                printf("Invalid walk transition: %d -> %d\n", current, next);
                return 0;
            }
        }
    }
    return 1;
}

void run_node2vec_test(int num_nodes, int avg_degree, int walk_length, int walks_per_node, int p, int q) {
    printf("\nRunning Node2Vec test with %d nodes, degree %d, walk length %d, %d walks/node\n",
           num_nodes, avg_degree, walk_length, walks_per_node);
    
    // Generate graph
    Graph graph = generate_graph(num_nodes, avg_degree);
    
    // Initialize parameters
    Node2VecParams params;
    params.walk_length = walk_length;
    params.walks_per_node = walks_per_node;
    params.p = p;
    params.q = q;
    params.window_size = 5;
    params.embedding_dim = 128;
    
    // Allocate walks
    int *cpu_walks = new int[num_nodes * walks_per_node * walk_length];
    int *gpu_walks = new int[num_nodes * walks_per_node * walk_length];
    
    // CPU implementation
    clock_t cpu_start = clock();
    node2vec_walks_cpu(graph, params, cpu_walks);
    double cpu_time = (double)(clock() - cpu_start) / CLOCKS_PER_SEC;
    
    // GPU implementation
    clock_t gpu_start = clock();
    node2vec_walks_gpu(graph, params, gpu_walks);
    double gpu_time = (double)(clock() - gpu_start) / CLOCKS_PER_SEC;
    
    // Validate
    int cpu_valid = validate_walks(graph, cpu_walks, num_nodes * walks_per_node, walk_length);
    int gpu_valid = validate_walks(graph, gpu_walks, num_nodes * walks_per_node, walk_length);
    
    printf("\nResults:\n");
    printf("CPU Time: %.4f sec\n", cpu_time);
    printf("GPU Time: %.4f sec\n", gpu_time);
    printf("Speedup: %.2fx\n", cpu_time / gpu_time);
    printf("CPU Validation: %s\n", cpu_valid ? "PASSED" : "FAILED");
    printf("GPU Validation: %s\n", gpu_valid ? "PASSED" : "FAILED");
    
    // Cleanup
    delete[] graph.edge_ptr;
    delete[] graph.edge_idx;
    delete[] graph.edge_weight;
    delete[] cpu_walks;
    delete[] gpu_walks;
}

int main() {
    // Test configurations
    run_node2vec_test(1000, 10, 20, 5, 1, 1);  // nodes, degree, walk_length, walks_per_node, p, q
    run_node2vec_test(10000, 15, 30, 10, 2, 1);
    run_node2vec_test(50000, 20, 40, 10, 1, 2);
    
    return 0;
}