#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>

const int NUM_POINTS = 1000000;  
const int DIM = 2;               
const int K = 3;                 
const int MAX_ITERS = 100;
const float THRESHOLD = 1e-3f;

#define CHECK_CUDA(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " in " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    }

void kmeans_cpu(const float* points, float* centroids, int num_points) {
    
    float cpu_centroids[K][DIM];
    for(int k=0; k<K; k++) {
        for(int d=0; d<DIM; d++) {
            cpu_centroids[k][d] = points[k*DIM + d];
        }
    }
    int* assignments = new int[num_points];
    int* counts = new int[K];
    for(int iter=0; iter<MAX_ITERS; iter++) {
        
        for(int i=0; i<num_points; i++) {
            float min_dist = INFINITY;
            int cluster = -1;
            
            for(int k=0; k<K; k++) {
                float dist = 0;
                for(int d=0; d<DIM; d++) {
                    float diff = points[i*DIM + d] - cpu_centroids[k][d];
                    dist += diff * diff;
                }
                if(dist < min_dist) {
                    min_dist = dist;
                    cluster = k;
                }
            }
            assignments[i] = cluster;
        }
        
        memset(cpu_centroids, 0, sizeof(float)*K*DIM);
        memset(counts, 0, sizeof(int)*K);
        
        for(int i=0; i<num_points; i++) {
            int cluster = assignments[i];
            counts[cluster]++;
            for(int d=0; d<DIM; d++) {
                cpu_centroids[cluster][d] += points[i*DIM + d];
            }
        }
        for(int k=0; k<K; k++) {
            if(counts[k] > 0) {
                for(int d=0; d<DIM; d++) {
                    cpu_centroids[k][d] /= counts[k];
                }
            }
        }
    }
    
    memcpy(centroids, cpu_centroids, sizeof(float)*K*DIM);
    delete[] assignments;
    delete[] counts;
}

__global__ void assign_clusters(const float* points, const float* centroids, 
    int* assignments, int num_points) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx >= num_points) return;
float min_dist = INFINITY;
int cluster = -1;

for(int k=0; k<K; k++) {
float dist = 0;
for(int d=0; d<DIM; d++) {
float diff = points[idx*DIM + d] - centroids[k*DIM + d];
dist += diff * diff;
}
if(dist < min_dist) {
min_dist = dist;
cluster = k;
}
}
assignments[idx] = cluster;
}

__global__ void update_centroids(const float* points, float* centroids, 
     int* counts, const int* assignments, 
     int num_points) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx >= num_points) return;
int cluster = assignments[idx];
for(int d=0; d<DIM; d++) {
atomicAdd(&centroids[cluster*DIM + d], points[idx*DIM + d]);
}
atomicAdd(&counts[cluster], 1);
}
int main() {

std::vector<float> h_points(NUM_POINTS * DIM);
std::default_random_engine gen(1234);


const float true_centroids[K][DIM] = {{1.0f, 1.0f}, {4.0f, 4.0f}, {8.0f, 8.0f}};

std::normal_distribution<float> dist[K] = {
std::normal_distribution<float>(0.0f, 0.2f),  
std::normal_distribution<float>(0.0f, 0.2f),  
std::normal_distribution<float>(0.0f, 0.2f)   
};
for(int i=0; i<NUM_POINTS; i++) {
int cluster = i % K;
h_points[i*DIM] = true_centroids[cluster][0] + dist[cluster](gen);
h_points[i*DIM+1] = true_centroids[cluster][1] + dist[cluster](gen);
}

std::vector<float> h_centroids_cpu(K*DIM);
auto cpu_start = std::chrono::high_resolution_clock::now();
kmeans_cpu(h_points.data(), h_centroids_cpu.data(), NUM_POINTS);
auto cpu_end = std::chrono::high_resolution_clock::now();
float cpu_time = std::chrono::duration<float>(cpu_end - cpu_start).count();

float *d_points, *d_centroids;
int *d_assignments, *d_counts;


CHECK_CUDA(cudaMalloc(&d_points, NUM_POINTS*DIM*sizeof(float)));
CHECK_CUDA(cudaMalloc(&d_centroids, K*DIM*sizeof(float)));
CHECK_CUDA(cudaMalloc(&d_assignments, NUM_POINTS*sizeof(int)));
CHECK_CUDA(cudaMalloc(&d_counts, K*sizeof(int)));

CHECK_CUDA(cudaMemcpy(d_points, h_points.data(), 
NUM_POINTS*DIM*sizeof(float), cudaMemcpyHostToDevice));

CHECK_CUDA(cudaMemcpy(d_centroids, h_points.data(), 
K*DIM*sizeof(float), cudaMemcpyHostToDevice));

cudaEvent_t start, stop;
CHECK_CUDA(cudaEventCreate(&start));
CHECK_CUDA(cudaEventCreate(&stop));

CHECK_CUDA(cudaEventRecord(start));

const int block_size = 256;
const int grid_size = (NUM_POINTS + block_size - 1) / block_size;

for(int iter=0; iter<MAX_ITERS; iter++) {

assign_clusters<<<grid_size, block_size>>>(d_points, d_centroids, 
                      d_assignments, NUM_POINTS);


CHECK_CUDA(cudaMemset(d_centroids, 0, K*DIM*sizeof(float)));
CHECK_CUDA(cudaMemset(d_counts, 0, K*sizeof(int)));


update_centroids<<<grid_size, block_size>>>(d_points, d_centroids,
                       d_counts, d_assignments,
                       NUM_POINTS);


std::vector<int> h_counts(K);
std::vector<float> h_centroids_gpu(K*DIM);
CHECK_CUDA(cudaMemcpy(h_counts.data(), d_counts, K*sizeof(int),
 cudaMemcpyDeviceToHost));
CHECK_CUDA(cudaMemcpy(h_centroids_gpu.data(), d_centroids,
 K*DIM*sizeof(float), cudaMemcpyDeviceToHost));

for(int k=0; k<K; k++) {
if(h_counts[k] > 0) {
for(int d=0; d<DIM; d++) {
h_centroids_gpu[k*DIM + d] /= h_counts[k];
}
}
}
CHECK_CUDA(cudaMemcpy(d_centroids, h_centroids_gpu.data(),
 K*DIM*sizeof(float), cudaMemcpyHostToDevice));
}

CHECK_CUDA(cudaEventRecord(stop));
CHECK_CUDA(cudaEventSynchronize(stop));


float gpu_time;
CHECK_CUDA(cudaEventElapsedTime(&gpu_time, start, stop));
gpu_time /= 1000.0f;  
std::vector<float> h_centroids_gpu(K*DIM);
CHECK_CUDA(cudaMemcpy(h_centroids_gpu.data(), d_centroids,
K*DIM*sizeof(float), cudaMemcpyDeviceToHost));

bool valid = true;
for(int k=0; k<K; k++) {
float dist = 0;
for(int d=0; d<DIM; d++) {
float diff = h_centroids_gpu[k*DIM+d] - true_centroids[k][d];
dist += diff * diff;
}
if(std::sqrt(dist) > THRESHOLD) valid = false;
}

std::cout << "K-Means Clustering Results\n";
std::cout << "==========================\n";
std::cout << "Data Points: " << NUM_POINTS << "\n";
std::cout << "Dimensions: " << DIM << "\n";
std::cout << "Clusters: " << K << "\n";
std::cout << "Validation: " << (valid ? "PASSED" : "FAILED") << "\n";
std::cout << "CPU Time: " << cpu_time << "s\n";
std::cout << "GPU Time: " << gpu_time << "s\n";
std::cout << "Speedup: " << cpu_time / gpu_time << "x\n";

CHECK_CUDA(cudaFree(d_points));
CHECK_CUDA(cudaFree(d_centroids));
CHECK_CUDA(cudaFree(d_assignments));
CHECK_CUDA(cudaFree(d_counts));
CHECK_CUDA(cudaEventDestroy(start));
CHECK_CUDA(cudaEventDestroy(stop));
return 0;
}