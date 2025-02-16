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

