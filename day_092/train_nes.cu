#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

// Constants
const int POPULATION_SIZE = 1024;
const int PARAM_DIM = 3;
const float LEARNING_RATE = 0.1f;
const float NOISE_STD = 0.2f;
const int NUM_GENERATIONS = 100;

// Kernel for parameter perturbation and evaluation
__global__ void nes_evaluate_population(
    float* current_params,
    float* rewards,
    float* perturbations,
    curandState* states,
    float noise_std,
    float env_target
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < POPULATION_SIZE) {
        curandState localState = states[tid];
        float cumulative_reward = 0.0f;

        // Generate perturbation and compute candidate parameters
        for(int i = 0; i < PARAM_DIM; i++) {
            float perturbation = curand_normal(&localState) * noise_std;
            perturbations[tid * PARAM_DIM + i] = perturbation;
            float param_value = current_params[i] + perturbation;
            
            // Simulated environment interaction (quadratic reward)
            cumulative_reward += -powf(param_value - env_target, 2.0f);
        }

        rewards[tid] = cumulative_reward;
        states[tid] = localState;
    }
}

// Kernel for parameter update
__global__ void nes_update_parameters(
    float* current_params,
    float* perturbations,
    float* rewards,
    float learning_rate,
    float noise_std
) {
    __shared__ float shared_rewards[POPULATION_SIZE];
    int tid = threadIdx.x;
    
    // Load rewards into shared memory
    if (tid < POPULATION_SIZE) {
        shared_rewards[tid] = rewards[tid];
    }
    __syncthreads();

    // Normalize rewards
    float mean_reward = 0.0f;
    float max_reward = -INFINITY;
    float min_reward = INFINITY;
    
    for(int i = 0; i < POPULATION_SIZE; i++) {
        mean_reward += shared_rewards[i];
        max_reward = fmaxf(max_reward, shared_rewards[i]);
        min_reward = fminf(min_reward, shared_rewards[i]);
    }
    mean_reward /= POPULATION_SIZE;

    // Compute fitness shaping
    for(int param_idx = 0; param_idx < PARAM_DIM; param_idx++) {
        float gradient = 0.0f;
        
        for(int i = 0; i < POPULATION_SIZE; i++) {
            float normalized_reward = (shared_rewards[i] - mean_reward) / 
                                    (max_reward - min_reward + 1e-8f);
            gradient += normalized_reward * perturbations[i * PARAM_DIM + param_idx];
        }

        // Update parameters using Adam-like scaling
        if (tid == 0) {
            current_params[param_idx] += learning_rate * gradient / 
                                      (POPULATION_SIZE * noise_std);
        }
    }
}

// Helper function for CUDA error checking
void cuda_check(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

int main() {
    // Initialize parameters
    std::vector<float> params(PARAM_DIM, 0.0f);
    const float TARGET_VALUE = 5.0f;

    // Device allocations
    float *d_params, *d_rewards, *d_perturbations;
    curandState* d_states;

    cuda_check(cudaMalloc(&d_params, PARAM_DIM * sizeof(float)), "Alloc params");
    cuda_check(cudaMalloc(&d_rewards, POPULATION_SIZE * sizeof(float)), "Alloc rewards");
    cuda_check(cudaMalloc(&d_perturbations, POPULATION_SIZE * PARAM_DIM * sizeof(float)), 
              "Alloc perturbations");
    cuda_check(cudaMalloc(&d_states, POPULATION_SIZE * sizeof(curandState)), "Alloc states");

    // Initialize parameters and random states
    cuda_check(cudaMemcpy(d_params, params.data(), PARAM_DIM * sizeof(float), 
              cudaMemcpyHostToDevice), "Init params");
    
    curandGenerator_t gen;
    curand_check(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT), "Create generator");
    curand_check(curandSetPseudoRandomGeneratorSeed(gen, time(0)), "Set seed");
    curand_check(curandGenerateNormal(gen, d_perturbations, 
              POPULATION_SIZE * PARAM_DIM, 0.0f, NOISE_STD), "Init perturbations");

    // Main training loop
    for(int gen = 0; gen < NUM_GENERATIONS; gen++) {
        // Evaluate population
        dim3 block(256);
        dim3 grid((POPULATION_SIZE + block.x - 1) / block.x);
        nes_evaluate_population<<<grid, block>>>(
            d_params, d_rewards, d_perturbations, d_states, 
            NOISE_STD, TARGET_VALUE
        );
        cuda_check(cudaDeviceSynchronize(), "Evaluate sync");

        // Update parameters
        nes_update_parameters<<<1, 256>>>(d_params, d_perturbations, 
                                        d_rewards, LEARNING_RATE, NOISE_STD);
        cuda_check(cudaDeviceSynchronize(), "Update sync");

        // Print progress
        cuda_check(cudaMemcpy(params.data(), d_params, PARAM_DIM * sizeof(float), 
                  "Copy params");
        std::cout << "Generation " << gen+1 << ": Params = [";
        for(float val : params) std::cout << val << " ";
        std::cout << "]" << std::endl;
    }

    // Cleanup
    cudaFree(d_params);
    cudaFree(d_rewards);
    cudaFree(d_perturbations);
    cudaFree(d_states);

    return 0;
}