#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>

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
};

// CPU implementation of random walk
void random_walk_cpu(const Graph &graph, int *walks, int walk_length, int walks_per_node) {
    srand(time(NULL));
    
    for (int node = 0; node < graph.num_nodes; node++) {
        for (int w = 0; w < walks_per_node; w++) {
            int current = node;
            walks[(node * walks_per_node + w) * walk_length] = current;
            
            for (int step = 1; step < walk_length; step++) {
                int start = graph.edge_ptr[current];
                int end = graph.edge_ptr[current + 1];
                int degree = end - start;
                
                if (degree == 0) break; // No outgoing edges
                
                int next = graph.edge_idx[start + (rand() % degree)];
                walks[(node * walks_per_node + w) * walk_length + step] = next;
                current = next;
            }
        }
    }
}

// GPU kernel initialization
__global__ void setup_curand_kernel(curandState *state, unsigned long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, idx, 0, &state[idx]);
}

// GPU random walk kernel
__global__ void random_walk_kernel(const int *edge_ptr, const int *edge_idx,
                                  int num_nodes, int walk_length,
                                  int walks_per_node, int *walks,
                                  curandState *states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes * walks_per_node) return;

    int node = idx / walks_per_node;
    int walk_id = idx % walks_per_node;
    int current = node;
    curandState local_state = states[idx];
    
    walks[idx * walk_length] = current;
    
    for (int step = 1; step < walk_length; step++) {
        int start = edge_ptr[current];
        int end = edge_ptr[current + 1];
        int degree = end - start;
        
        if (degree == 0) break;
        
        int choice = curand(&local_state) % degree;
        int next = edge_idx[start + choice];
        walks[idx * walk_length + step] = next;
        current = next;
    }
    
    states[idx] = local_state;
}

// GPU implementation
void random_walk_gpu(const Graph &graph, int *h_walks, int walk_length, int walks_per_node) {
    int *d_edge_ptr, *d_edge_idx, *d_walks;
    curandState *d_states;
    
    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc(&d_edge_ptr, (graph.num_nodes + 1) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_edge_idx, graph.num_edges * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_walks, graph.num_nodes * walks_per_node * walk_length * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_states, graph.num_nodes * walks_per_node * sizeof(curandState)));

    // Copy graph data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_edge_ptr, graph.edge_ptr, (graph.num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_edge_idx, graph.edge_idx, graph.num_edges * sizeof(int), cudaMemcpyHostToDevice));

    // Setup CURAND states
    int threads_per_block = 256;
    int blocks = (graph.num_nodes * walks_per_node + threads_per_block - 1) / threads_per_block;
    setup_curand_kernel<<<blocks, threads_per_block>>>(d_states, time(NULL));

    // Launch random walk kernel
    random_walk_kernel<<<blocks, threads_per_block>>>(
        d_edge_ptr, d_edge_idx,
        graph.num_nodes, walk_length,
        walks_per_node, d_walks,
        d_states
    );

    // Copy results back
    CHECK_CUDA_ERROR(cudaMemcpy(h_walks, d_walks, graph.num_nodes * walks_per_node * walk_length * sizeof(int), cudaMemcpyDeviceToHost));

    // Cleanup
    CHECK_CUDA_ERROR(cudaFree(d_edge_ptr));
    CHECK_CUDA_ERROR(cudaFree(d_edge_idx));
    CHECK_CUDA_ERROR(cudaFree(d_walks));
    CHECK_CUDA_ERROR(cudaFree(d_states));
}

// Generate a simple ring graph with random edges
Graph generate_graph(int num_nodes, int avg_degree) {
    Graph graph;
    graph.num_nodes = num_nodes;
    graph.edge_ptr = new int[num_nodes + 1];
    graph.edge_idx = new int[num_nodes * (avg_degree + 1)]; // +1 for ring edges
    
    int edge_count = 0;
    srand(time(NULL));
    
    for (int i = 0; i < num_nodes; i++) {
        graph.edge_ptr[i] = edge_count;
        
        // Add ring edges
        graph.edge_idx[edge_count++] = (i + 1) % num_nodes;
        
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

// Validate walks (check if each step is a valid edge)
int validate_walks(const Graph &graph, const int *walks, int walk_length, int walks_per_node) {
    for (int node = 0; node < graph.num_nodes; node++) {
        for (int w = 0; w < walks_per_node; w++) {
            const int *walk = &walks[(node * walks_per_node + w) * walk_length];
            
            for (int step = 1; step < walk_length; step++) {
                int current = walk[step - 1];
                int next = walk[step];
                
                int valid = 0;
                for (int e = graph.edge_ptr[current]; e < graph.edge_ptr[current + 1]; e++) {
                    if (graph.edge_idx[e] == next) {
                        valid = 1;
                        break;
                    }
                }
                
                if (!valid && walk[step] != -1) {
                    printf("Invalid step %d -> %d in walk %d from node %d\n",
                           current, next, w, node);
                    return 0;
                }
            }
        }
    }
    return 1;
}

void print_walk_stats(const int *walks, int num_walks, int walk_length) {
    printf("Walk statistics:\n");
    printf("First walk: ");
    for (int i = 0; i < walk_length; i++) {
        printf("%d ", walks[i]);
    }
    printf("\n");
}

void run_test(int num_nodes, int avg_degree, int walk_length, int walks_per_node) {
    printf("\nRunning test with %d nodes, avg degree %d, walk length %d, %d walks/node\n",
           num_nodes, avg_degree, walk_length, walks_per_node);
    
    // Generate graph
    Graph graph = generate_graph(num_nodes, avg_degree);
    
    // Allocate walks
    int *cpu_walks = new int[num_nodes * walks_per_node * walk_length];
    int *gpu_walks = new int[num_nodes * walks_per_node * walk_length];

    // CPU implementation
    clock_t cpu_start = clock();
    random_walk_cpu(graph, cpu_walks, walk_length, walks_per_node);
    double cpu_time = (double)(clock() - cpu_start) / CLOCKS_PER_SEC;
    
    // GPU implementation
    clock_t gpu_start = clock();
    random_walk_gpu(graph, gpu_walks, walk_length, walks_per_node);
    double gpu_time = (double)(clock() - gpu_start) / CLOCKS_PER_SEC;
    
    // Validate
    int cpu_valid = validate_walks(graph, cpu_walks, walk_length, walks_per_node);
    int gpu_valid = validate_walks(graph, gpu_walks, walk_length, walks_per_node);
    
    // Print results
    print_walk_stats(cpu_walks, num_nodes * walks_per_node, walk_length);
    print_walk_stats(gpu_walks, num_nodes * walks_per_node, walk_length);
    
    printf("\nResults:\n");
    printf("CPU Time: %.4f sec\n", cpu_time);
    printf("GPU Time: %.4f sec\n", gpu_time);
    printf("Speedup: %.2fx\n", cpu_time / gpu_time);
    printf("CPU Validation: %s\n", cpu_valid ? "PASSED" : "FAILED");
    printf("GPU Validation: %s\n", gpu_valid ? "PASSED" : "FAILED");

    // Cleanup
    delete[] graph.edge_ptr;
    delete[] graph.edge_idx;
    delete[] cpu_walks;
    delete[] gpu_walks;
}

int main() {
    // Small test for validation
    run_test(100, 3, 10, 5);
    
    // Medium test
    run_test(10000, 5, 20, 10);
    
    // Large test
    run_test(100000, 8, 30, 10);

    return 0;
}