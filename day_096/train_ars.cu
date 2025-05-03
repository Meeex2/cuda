#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <vector>
#include <cmath>

// Configuration
const int NUM_DIRECTIONS = 1024;     // Parallel perturbations
const int NUM_TOP_DIRECTIONS = 256;  // Elite directions
const int PARAM_DIM = 4;             // Policy parameters
const float STEP_SIZE = 0.02f;
const float NOISE_STD = 0.03f;
const int ITERATIONS = 100;

// Policy structure
struct Policy {
    float params[PARAM_DIM];
};

// CUDA kernel for parallel perturbation evaluation
__global__ void ars_evaluate_directions(
    Policy* policy,
    float* rewards_plus,
    float* rewards_minus,
    float* perturbations,
    curandState* states,
    float env_target
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < NUM_DIRECTIONS) {
        curandState local_state = states[tid];
        Policy local_policy = *policy;
        
        // Generate perturbation
        float perturbation[PARAM_DIM];
        for(int i = 0; i < PARAM_DIM; i++) {
            perturbation[i] = curand_normal(&local_state) * NOISE_STD;
            perturbations[tid * PARAM_DIM + i] = perturbation[i];
        }
        
        // Evaluate positive perturbation
        float reward_plus = 0.0f;
        for(int i = 0; i < PARAM_DIM; i++) {
            float perturbed_param = local_policy.params[i] + perturbation[i];
            reward_plus += -fabs(perturbed_param - env_target);
        }
        rewards_plus[tid] = reward_plus;
        
        // Evaluate negative perturbation
        float reward_minus = 0.0f;
        for(int i = 0; i < PARAM_DIM; i++) {
            float perturbed_param = local_policy.params[i] - perturbation[i];
            reward_minus += -fabs(perturbed_param - env_target);
        }
        rewards_minus[tid] = reward_minus;
        
        states[tid] = local_state;
    }
}

// Kernel for parameter update
__global__ void ars_update_policy(
    Policy* policy,
    const float* perturbations,
    const float* rewards_plus,
    const float* rewards_minus,
    const int* top_indices
) {
    __shared__ float shared_grad[PARAM_DIM];
    const int tid = threadIdx.x % PARAM_DIM;
    
    if (threadIdx.x < PARAM_DIM) {
        shared_grad[threadIdx.x] = 0.0f;
    }
    __syncthreads();
    
    // Accumulate gradients from top directions
    for(int i = 0; i < NUM_TOP_DIRECTIONS; i++) {
        int idx = top_indices[i];
        float reward_diff = rewards_plus[idx] - rewards_minus[idx];
        
        if (tid < PARAM_DIM) {
            atomicAdd(&shared_grad[tid], 
                reward_diff * perturbations[idx * PARAM_DIM + tid]);
        }
    }
    __syncthreads();
    
    // Update policy parameters
    if (tid < PARAM_DIM) {
        policy->params[tid] += STEP_SIZE * shared_grad[tid] / 
                             (NUM_TOP_DIRECTIONS * NOISE_STD);
    }
}

// Host function to select top directions
void select_top_directions(
    const float* rewards_plus,
    const float* rewards_minus,
    std::vector<int>& top_indices
) {
    std::vector<std::pair<float, int>> scores(NUM_DIRECTIONS);
    
    for(int i = 0; i < NUM_DIRECTIONS; i++) {
        scores[i] = {std::max(rewards_plus[i], rewards_minus[i]), i};
    }
    
    std::sort(scores.begin(), scores.end(), 
        [](auto& a, auto& b) { return a.first > b.first; });
    
    for(int i = 0; i < NUM_TOP_DIRECTIONS; i++) {
        top_indices[i] = scores[i].second;
    }
}

int main() {
    // Initialize policy
    Policy h_policy = {0};
    const float TARGET = 1.5f;
    
    // Device allocations
    Policy* d_policy;
    float *d_rewards_plus, *d_rewards_minus, *d_perturbations;
    curandState* d_states;
    int* d_top_indices;
    
    cudaMalloc(&d_policy, sizeof(Policy));
    cudaMalloc(&d_rewards_plus, NUM_DIRECTIONS * sizeof(float));
    cudaMalloc(&d_rewards_minus, NUM_DIRECTIONS * sizeof(float));
    cudaMalloc(&d_perturbations, NUM_DIRECTIONS * PARAM_DIM * sizeof(float));
    cudaMalloc(&d_states, NUM_DIRECTIONS * sizeof(curandState));
    cudaMalloc(&d_top_indices, NUM_TOP_DIRECTIONS * sizeof(int));
    
    // Initialize CUDA random states
    dim3 block(256);
    dim3 grid((NUM_DIRECTIONS + block.x - 1) / block.x);
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, time(0));
    curandGenerateNormal(gen, d_perturbations, 
                        NUM_DIRECTIONS * PARAM_DIM, 0.0f, NOISE_STD);
    
    // Training loop
    for(int iter = 0; iter < ITERATIONS; iter++) {
        // Evaluate all directions
        ars_evaluate_directions<<<grid, block>>>(
            d_policy, d_rewards_plus, d_rewards_minus,
            d_perturbations, d_states, TARGET
        );
        cudaDeviceSynchronize();
        
        // Select top directions
        std::vector<float> h_rewards_plus(NUM_DIRECTIONS);
        std::vector<float> h_rewards_minus(NUM_DIRECTIONS);
        std::vector<int> h_top_indices(NUM_TOP_DIRECTIONS);
        
        cudaMemcpy(h_rewards_plus.data(), d_rewards_plus,
                   NUM_DIRECTIONS * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_rewards_minus.data(), d_rewards_minus,
                   NUM_DIRECTIONS * sizeof(float), cudaMemcpyDeviceToHost);
        
        select_top_directions(h_rewards_plus.data(), h_rewards_minus.data(),
                            h_top_indices);
        
        cudaMemcpy(d_top_indices, h_top_indices.data(),
                  NUM_TOP_DIRECTIONS * sizeof(int), cudaMemcpyHostToDevice);
        
        // Update policy
        ars_update_policy<<<1, 256>>>(d_policy, d_perturbations,
                                     d_rewards_plus, d_rewards_minus,
                                     d_top_indices);
        
        // Monitor progress
        cudaMemcpy(&h_policy, d_policy, sizeof(Policy), cudaMemcpyDeviceToHost);
        std::cout << "Iter " << iter << ": Params [";
        for(float p : h_policy.params) std::cout << p << " ";
        std::cout << "]" << std::endl;
    }
    
    // Cleanup
    cudaFree(d_policy);
    cudaFree(d_rewards_plus);
    cudaFree(d_rewards_minus);
    cudaFree(d_perturbations);
    cudaFree(d_states);
    cudaFree(d_top_indices);
    
    return 0;
}