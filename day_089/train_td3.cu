#include <cuda.h>
#include <curand_kernel.h>
#include <cmath>

#define CHECK_CUDA(func) { cudaError_t error = (func); if(error != cudaSuccess) printf("CUDA Error: %s\n", cudaGetErrorString(error)); }

// TD3 Experience Buffer
struct TD3Buffer {
    float* states;
    float* actions;
    float* rewards;
    float* next_states;
    bool* dones;
    int capacity;
    int position;
    int state_dim;
    int action_dim;
    
    TD3Buffer(int capacity, int state_dim, int action_dim) : 
        capacity(capacity), state_dim(state_dim), action_dim(action_dim), position(0) {
        CHECK_CUDA(cudaMalloc(&states, capacity * state_dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&actions, capacity * action_dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&rewards, capacity * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&next_states, capacity * state_dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&dones, capacity * sizeof(bool)));
    }
};

// Deterministic Policy Network
struct DeterministicPolicy {
    float* weights;
    int state_dim;
    int action_dim;
    int hidden_dim;
    
    DeterministicPolicy(int s_dim, int a_dim, int h_dim) :
        state_dim(s_dim), action_dim(a_dim), hidden_dim(h_dim) {
        CHECK_CUDA(cudaMalloc(&weights, (s_dim*h_dim + h_dim*a_dim) * sizeof(float)));
    }

    __device__ void get_action(const float* state, float* action) {
        float hidden[256];
        // Feature extraction
        for(int i = 0; i < hidden_dim; i++) {
            hidden[i] = 0.0f;
            for(int j = 0; j < state_dim; j++) {
                hidden[i] += state[j] * weights[j * hidden_dim + i];
            }
            hidden[i] = tanhf(hidden[i]);
        }
        
        // Action output
        for(int a = 0; a < action_dim; a++) {
            action[a] = 0.0f;
            for(int j = 0; j < hidden_dim; j++) {
                action[a] += hidden[j] * weights[state_dim*hidden_dim + j*action_dim + a];
            }
            action[a] = tanhf(action[a]);  // Assuming action space is [-1,1]
        }
    }
};

// Twin Q-Networks with Target Networks
struct TwinQNetwork {
    float* q1_weights;
    float* q2_weights;
    float* target_q1_weights;
    float* target_q2_weights;
    int state_dim;
    int action_dim;
    int hidden_dim;
    
    TwinQNetwork(int s_dim, int a_dim, int h_dim) :
        state_dim(s_dim), action_dim(a_dim), hidden_dim(h_dim) {
        CHECK_CUDA(cudaMalloc(&q1_weights, (s_dim + a_dim)*h_dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&q2_weights, (s_dim + a_dim)*h_dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&target_q1_weights, (s_dim + a_dim)*h_dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&target_q2_weights, (s_dim + a_dim)*h_dim * sizeof(float)));
    }

    __device__ float q1_value(const float* state, const float* action) {
        float hidden[256];
        for(int i = 0; i < hidden_dim; i++) {
            hidden[i] = 0.0f;
            for(int j = 0; j < state_dim; j++)
                hidden[i] += state[j] * q1_weights[j * hidden_dim + i];
            for(int j = 0; j < action_dim; j++)
                hidden[i] += action[j] * q1_weights[(state_dim + j) * hidden_dim + i];
            hidden[i] = relu(hidden[i]);
        }
        
        float q_value = 0.0f;
        for(int i = 0; i < hidden_dim; i++) {
            q_value += hidden[i] * q1_weights[(state_dim + action_dim)*hidden_dim + i];
        }
        return q_value;
    }

    __device__ float target_q1_value(const float* state, const float* action) {
        // Similar to q1_value but using target weights
    }

    __device__ float relu(float x) { return fmaxf(x, 0.0f); }
};

// TD3 Core Kernels
__global__ void td3_critic_update_kernel(
    TD3Buffer buffer,
    TwinQNetwork q_net,
    DeterministicPolicy policy,
    DeterministicPolicy target_policy,
    float* q_losses,
    float gamma,
    float tau,
    float noise_clip,
    float policy_noise
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= buffer.capacity) return;

    float state[buffer.state_dim];
    float action[buffer.action_dim];
    float next_state[buffer.state_dim];
    
    // Load experience
    for(int i = 0; i < buffer.state_dim; i++) {
        state[i] = buffer.states[idx * buffer.state_dim + i];
        next_state[i] = buffer.next_states[idx * buffer.state_dim + i];
    }
    for(int i = 0; i < buffer.action_dim; i++)
        action[i] = buffer.actions[idx * buffer.action_dim + i];

    // Target action with noise clipping
    float target_action[buffer.action_dim];
    target_policy.get_action(next_state, target_action);
    
    curandState local_state;
    curand_init(clock64(), idx, 0, &local_state);
    for(int i = 0; i < buffer.action_dim; i++) {
        float noise = curand_normal(&local_state) * policy_noise;
        noise = fmaxf(fminf(noise, noise_clip), -noise_clip);
        target_action[i] = fmaxf(fminf(target_action[i] + noise, 1.0f), -1.0f);
    }

    // Compute target Q-values
    float target_q1 = q_net.target_q1_value(next_state, target_action);
    float target_q2 = q_net.target_q2_value(next_state, target_action);
    float target_q = buffer.rewards[idx] + gamma * fminf(target_q1, target_q2) * (1 - buffer.dones[idx]);

    // Current Q-values
    float current_q1 = q_net.q1_value(state, action);
    float current_q2 = q_net.q2_value(state, action);
    
    // Q-losses
    q_losses[idx*2] = 0.5f * powf(current_q1 - target_q, 2);
    q_losses[idx*2+1] = 0.5f * powf(current_q2 - target_q, 2);
}

__global__ void td3_policy_update_kernel(
    TD3Buffer buffer,
    DeterministicPolicy policy,
    TwinQNetwork q_net,
    float* policy_losses
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= buffer.capacity) return;

    float state[buffer.state_dim];
    for(int i = 0; i < buffer.state_dim; i++)
        state[i] = buffer.states[idx * buffer.state_dim + i];
    
    float action[buffer.action_dim];
    policy.get_action(state, action);
    
    float q_value = q_net.q1_value(state, action);
    policy_losses[idx] = -q_value;
}

// Training execution
void train_td3() {
    const int state_dim = 28;
    const int action_dim = 4;
    const int buffer_size = 1000000;
    const float gamma = 0.99f;
    const float tau = 0.005f;
    const float policy_noise = 0.2f;
    const float noise_clip = 0.5f;
    
    TD3Buffer buffer(buffer_size, state_dim, action_dim);
    DeterministicPolicy policy(state_dim, action_dim, 256);
    DeterministicPolicy target_policy(state_dim, action_dim, 256);
    TwinQNetwork q_net(state_dim, action_dim, 256);
    
    float* d_q_losses;
    float* d_policy_losses;
    CHECK_CUDA(cudaMalloc(&d_q_losses, buffer_size * 2 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_policy_losses, buffer_size * sizeof(float)));
    
    // Critic update
    dim3 block(256);
    dim3 grid((buffer_size + 255)/256);
    td3_critic_update_kernel<<<grid, block>>>(
        buffer, q_net, policy, target_policy,
        d_q_losses, gamma, tau, noise_clip, policy_noise
    );
    
    // Delayed policy update (every 2 critic updates)
    if(iteration % 2 == 0) {
        td3_policy_update_kernel<<<grid, block>>>(
            buffer, policy, q_net, d_policy_losses
        );
    }

}

int main() {
    train_td3();
    return 0;
}