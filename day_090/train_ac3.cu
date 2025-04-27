#include <cuda.h>
#include <curand_kernel.h>
#include <cmath>

#define CHECK_CUDA(func) { cudaError_t error = (func); if(error != cudaSuccess) printf("CUDA Error: %s\n", cudaGetErrorString(error)); }

// Shared Neural Network for Actor-Critic
struct SharedNetwork {
    float* conv_weights;
    float* fc_weights;
    float* actor_weights;
    float* critic_weights;
    int state_channels;
    int action_dim;
    
    __device__ void forward(const float* state, float* policy, float& value) {
        // Simplified network with shared features
        float features[256];
        // Convolutional feature extraction
        // ... (omitted for brevity)
        
        // Actor head
        for(int a = 0; a < action_dim; a++) {
            policy[a] = 0.0f;
            for(int f = 0; f < 256; f++) {
                policy[a] += features[f] * actor_weights[f * action_dim + a];
            }
        }
        softmax(policy, action_dim);
        
        // Critic head
        value = 0.0f;
        for(int f = 0; f < 256; f++) {
            value += features[f] * critic_weights[f];
        }
    }

    __device__ void softmax(float* logits, int size) {
        float max_val = -INFINITY;
        float sum = 0.0f;
        for(int i = 0; i < size; i++) max_val = fmaxf(max_val, logits[i]);
        for(int i = 0; i < size; i++) {
            logits[i] = expf(logits[i] - max_val);
            sum += logits[i];
        }
        for(int i = 0; i < size; i++) logits[i] /= sum;
    }
};

// Parallel Environment States
struct ParallelEnvs {
    float* states;
    float* rewards;
    bool* dones;
    int num_envs;
    int state_size;
    
    ParallelEnvs(int num_envs, int state_size) :
        num_envs(num_envs), state_size(state_size) {
        CHECK_CUDA(cudaMalloc(&states, num_envs * state_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&rewards, num_envs * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&dones, num_envs * sizeof(bool)));
    }
};

// A3C Worker Kernel
__global__ void a3c_worker_kernel(
    SharedNetwork global_net,
    ParallelEnvs envs,
    float* gradients,
    float gamma,
    float beta,
    int t_max
) {
    int env_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(env_id >= envs.num_envs) return;

    // Thread-local network copy
    SharedNetwork local_net = global_net;
    float local_gradients[NET_PARAM_SIZE] = {0};
    
    float* state = &envs.states[env_id * envs.state_size];
    float total_reward = 0.0f;
    float entropy = 0.0f;

    for(int t = 0; t < t_max && !envs.dones[env_id]; t++) {
        float policy[local_net.action_dim];
        float value;
        local_net.forward(state, policy, value);
        
        // Sample action
        curandState rand_state;
        curand_init(clock64(), env_id, 0, &rand_state);
        float rnd = curand_uniform(&rand_state);
        int action = 0;
        float cum_prob = 0.0f;
        while(cum_prob < rnd && action < local_net.action_dim-1) {
            cum_prob += policy[action++];
        }
        
        // Environment step (simulated)
        float next_state[envs.state_size];
        float reward = simulate_env_step(state, action);
        bool done = check_done_state(next_state);
        
        // Calculate advantage
        float next_value;
        local_net.forward(next_state, policy, next_value);
        float advantage = reward + gamma * next_value * (1 - done) - value;
        
        // Accumulate gradients
        for(int f = 0; f < 256; f++) {
            // Value loss gradient
            local_gradients[CRITIC_OFFSET + f] += advantage * (value - next_value);
            
            // Policy gradient
            float policy_grad = advantage * (1 - policy[action]);
            for(int a = 0; a < local_net.action_dim; a++) {
                local_gradients[ACTOR_OFFSET + f*local_net.action_dim + a] += 
                    policy_grad * (a == action ? 1 : 0);
            }
        }
        
        // Entropy regularization
        for(int a = 0; a < local_net.action_dim; a++) {
            entropy += policy[a] * logf(policy[a] + 1e-10);
        }
        
        total_reward += reward;
        state = next_state;
    }
    
    // Add entropy gradient
    for(int a = 0; a < local_net.action_dim; a++) {
        local_gradients[ENTROPY_OFFSET + a] += beta * entropy;
    }
    
    // Atomic update of global gradients
    for(int i = 0; i < NET_PARAM_SIZE; i++) {
        atomicAdd(&gradients[i], local_gradients[i]);
    }
}

// Global Network Update Kernel
__global__ void global_update_kernel(
    SharedNetwork global_net,
    float* gradients,
    float lr,
    int num_envs
) {
    int param_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(param_idx >= NET_PARAM_SIZE) return;
    
    // Apply RMSProp update
    float g = gradients[param_idx] / num_envs;
    float rms = sqrtf(g * g + 1e-5f);
    global_net.weights[param_idx] -= lr * g / rms;
}

// Training Execution
void train_a3c() {
    const int num_envs = 1024;
    const int state_size = 84*84*4;  // Atari-like input
    const int action_dim = 18;
    const float gamma = 0.99f;
    const float beta = 0.01f;
    const int t_max = 5;
    
    SharedNetwork global_net;
    ParallelEnvs envs(num_envs, state_size);
    float* d_gradients;
    CHECK_CUDA(cudaMalloc(&d_gradients, NET_PARAM_SIZE * sizeof(float)));
    
    // Parallel workers
    dim3 block(256);
    dim3 grid((num_envs + 255)/256);
    a3c_worker_kernel<<<grid, block>>>(global_net, envs, d_gradients, gamma, beta, t_max);
    
    // Global network update
    dim3 update_block(256);
    dim3 update_grid((NET_PARAM_SIZE + 255)/256);
    global_update_kernel<<<update_grid, update_block>>>(global_net, d_gradients, 0.00025f, num_envs);
    
    // Reset gradients
    CHECK_CUDA(cudaMemset(d_gradients, 0, NET_PARAM_SIZE * sizeof(float)));
}

int main() {
    train_a3c();
    return 0;
}