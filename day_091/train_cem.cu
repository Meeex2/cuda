#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

// Kernel for initializing random states
__global__ void init_curand_states(curandState* states, unsigned long seed, int population_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < population_size) {
        curand_init(seed, tid, 0, &states[tid]);
    }
}

// CEM evaluation kernel
__global__ void cem_evaluate_policies(
    float* current_mean,
    float* current_std,
    float* rewards,
    float* candidate_params,
    curandState* states,
    int param_dim,
    int population_size,
    float env_target
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < population_size) {
        curandState localState = states[tid];
        float cumulative_reward = 0.0f;

        // Generate candidate parameters
        for(int i = 0; i < param_dim; i++) {
            float noise = curand_normal(&localState) * current_std[i];
            candidate_params[tid * param_dim + i] = current_mean[i] + noise;
        }

        // Simulated environment interaction
        for(int i = 0; i < param_dim; i++) {
            float param_value = candidate_params[tid * param_dim + i];
            cumulative_reward += -fabs(param_value - env_target);
        }

        rewards[tid] = cumulative_reward;
        states[tid] = localState;
    }
}

// Host helper functions
std::vector<int> select_elites(const std::vector<float>& rewards, int num_elites) {
    std::vector<int> indices(rewards.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), 
        [&rewards](int a, int b) { return rewards[a] > rewards[b]; });
    return std::vector<int>(indices.begin(), indices.begin() + num_elites);
}

void update_distribution(
    std::vector<float>& new_mean,
    std::vector<float>& new_std,
    const std::vector<float>& candidates,
    const std::vector<int>& elite_indices,
    int param_dim
) {
    int num_elites = elite_indices.size();
    if(num_elites == 0) return;

    // Calculate new mean
    std::fill(new_mean.begin(), new_mean.end(), 0.0f);
    for(int idx : elite_indices) {
        for(int j = 0; j < param_dim; j++) {
            new_mean[j] += candidates[idx * param_dim + j];
        }
    }
    for(float& val : new_mean) val /= num_elites;

    // Calculate new standard deviation
    std::fill(new_std.begin(), new_std.end(), 0.0f);
    for(int idx : elite_indices) {
        for(int j = 0; j < param_dim; j++) {
            float diff = candidates[idx * param_dim + j] - new_mean[j];
            new_std[j] += diff * diff;
        }
    }
    for(int j = 0; j < param_dim; j++) {
        new_std[j] = sqrt(new_std[j] / num_elites) + 1e-6f;
    }
}

int main() {
    // Configuration
    const int param_dim = 3;
    const int population_size = 1024;
    const int num_elites = population_size / 5;
    const float env_target = 5.0f;
    const int iterations = 50;

    // Host allocations
    std::vector<float> current_mean(param_dim, 0.0f);
    std::vector<float> current_std(param_dim, 1.0f);
    std::vector<float> host_rewards(population_size);
    std::vector<float> host_candidates(population_size * param_dim);

    // Device allocations
    float *d_mean, *d_std, *d_rewards, *d_candidates;
    curandState* d_states;

    cudaMalloc(&d_mean, param_dim * sizeof(float));
    cudaMalloc(&d_std, param_dim * sizeof(float));
    cudaMalloc(&d_rewards, population_size * sizeof(float));
    cudaMalloc(&d_candidates, population_size * param_dim * sizeof(float));
    cudaMalloc(&d_states, population_size * sizeof(curandState));

    // Initialize device memory
    cudaMemcpy(d_mean, current_mean.data(), param_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_std, current_std.data(), param_dim * sizeof(float), cudaMemcpyHostToDevice);
    init_curand_states<<<(population_size + 255)/256, 256>>>(d_states, time(0), population_size);

    // Main CEM loop
    for(int iter = 0; iter < iterations; iter++) {
        // Evaluate population
        dim3 block(256);
        dim3 grid((population_size + block.x - 1) / block.x);
        cem_evaluate_policies<<<grid, block>>>(
            d_mean, d_std, d_rewards, d_candidates, d_states,
            param_dim, population_size, env_target
        );
        cudaDeviceSynchronize();

        // Copy results to host
        cudaMemcpy(host_rewards.data(), d_rewards, population_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(host_candidates.data(), d_candidates, 
                  population_size * param_dim * sizeof(float), cudaMemcpyDeviceToHost);

        // Update distribution
        std::vector<int> elite_indices = select_elites(host_rewards, num_elites);
        update_distribution(current_mean, current_std, host_candidates, elite_indices, param_dim);

        // Update device parameters
        cudaMemcpy(d_mean, current_mean.data(), param_dim * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_std, current_std.data(), param_dim * sizeof(float), cudaMemcpyHostToDevice);

        printf("Iteration %d: Mean = [%.3f, %.3f, %.3f]\n", 
              iter+1, current_mean[0], current_mean[1], current_mean[2]);
    }

    // Cleanup
    cudaFree(d_mean);
    cudaFree(d_std);
    cudaFree(d_rewards);
    cudaFree(d_candidates);
    cudaFree(d_states);

    return 0;
}