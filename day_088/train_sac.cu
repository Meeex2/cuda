#include <cuda.h>
#include <curand_kernel.h>
#include <cmath>

#define CHECK_CUDA(func) { cudaError_t error = (func); if(error != cudaSuccess) printf("CUDA Error: %s\n", cudaGetErrorString(error)); }

// SAC Experience Buffer
struct SACBuffer {
    float* states;
    float* actions;
    float* rewards;
    float* next_states;
    bool* dones;
    int capacity;
    int position;
    int state_dim;
    int action_dim;
    
    SACBuffer(int capacity, int state_dim, int action_dim) : 
        capacity(capacity), state_dim(state_dim), action_dim(action_dim), position(0) {
        CHECK_CUDA(cudaMalloc(&states, capacity * state_dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&actions, capacity * action_dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&rewards, capacity * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&next_states, capacity * state_dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&dones, capacity * sizeof(bool)));
    }
};

// Gaussian Policy Network
struct GaussianPolicy {
    float* weights;
    int state_dim;
    int action_dim;
    int hidden_dim;
    
    GaussianPolicy(int s_dim, int a_dim, int h_dim) :
        state_dim(s_dim), action_dim(a_dim), hidden_dim(h_dim) {
        CHECK_CUDA(cudaMalloc(&weights, (s_dim*h_dim + h_dim*2*a_dim) * sizeof(float)));
    }

    __device__ void sample_action(const float* state, float* action, float& log_prob) {
        float hidden[256];
        // Feature extraction
        for(int i = 0; i < hidden_dim; i++) {
            hidden[i] = 0.0f;
            for(int j = 0; j < state_dim; j++) {
                hidden[i] += state[j] * weights[j * hidden_dim + i];
            }
            hidden[i] = tanhf(hidden[i]);
        }
        
        // Mean and log_std
        float mean[32], log_std[32];
        for(int a = 0; a < action_dim; a++) {
            mean[a] = 0.0f;
            log_std[a] = 0.0f;
            for(int j = 0; j < hidden_dim; j++) {
                mean[a] += hidden[j] * weights[state_dim*hidden_dim + j*2*action_dim + a];
                log_std[a] += hidden[j] * weights[state_dim*hidden_dim + j*2*action_dim + action_dim + a];
            }
        }
        
        // Sample action
        curandState local_state;
        curand_init(clock64(), threadIdx.x, 0, &local_state);
        log_prob = 0.0f;
        for(int a = 0; a < action_dim; a++) {
            float z = curand_normal(&local_state);
            action[a] = mean[a] + exp(log_std[a]) * z;
            log_prob += -(z*z)/2 - log_std[a] - 0.9189385332f; // -0.5*log(2π)
        }
    }
};

// Twin Q-Networks
struct TwinQNetwork {
    float* q1_weights;
    float* q2_weights;
    int state_dim;
    int action_dim;
    int hidden_dim;
    
    TwinQNetwork(int s_dim, int a_dim, int h_dim) :
        state_dim(s_dim), action_dim(a_dim), hidden_dim(h_dim) {
        CHECK_CUDA(cudaMalloc(&q1_weights, (s_dim + a_dim)*h_dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&q2_weights, (s_dim + a_dim)*h_dim * sizeof(float)));
    }

    __device__ void q_values(const float* state, const float* action, float& q1, float& q2) {
        float hidden1[256], hidden2[256];
        
        // Q1 network
        for(int i = 0; i < hidden_dim; i++) {
            hidden1[i] = 0.0f;
            for(int j = 0; j < state_dim; j++)
                hidden1[i] += state[j] * q1_weights[j * hidden_dim + i];
            for(int j = 0; j < action_dim; j++)
                hidden1[i] += action[j] * q1_weights[(state_dim + j) * hidden_dim + i];
            hidden1[i] = relu(hidden1[i]);
        }
        
        // Q2 network
        for(int i = 0; i < hidden_dim; i++) {
            hidden2[i] = 0.0f;
            for(int j = 0; j < state_dim; j++)
                hidden2[i] += state[j] * q2_weights[j * hidden_dim + i];
            for(int j = 0; j < action_dim; j++)
                hidden2[i] += action[j] * q2_weights[(state_dim + j) * hidden_dim + i];
            hidden2[i] = relu(hidden2[i]);
        }
        
        // Output layers
        q1 = 0.0f; q2 = 0.0f;
        for(int i = 0; i < hidden_dim; i++) {
            q1 += hidden1[i] * q1_weights[(state_dim + action_dim)*hidden_dim + i];
            q2 += hidden2[i] * q2_weights[(state_dim + action_dim)*hidden_dim + i];
        }
    }

    __device__ float relu(float x) { return fmaxf(x, 0.0f); }
};

// SAC Core Algorithm Kernels
__global__ void sac_q_update_kernel(
    SACBuffer buffer,
    TwinQNetwork q_net,
    GaussianPolicy policy,
    GaussianPolicy target_policy,
    float* q_losses,
    float gamma,
    float tau
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
    
    // Target Q-value calculation
    float next_action[buffer.action_dim];
    float log_prob;
    target_policy.sample_action(next_state, next_action, log_prob);
    
    float target_q1, target_q2;
    q_net.q_values(next_state, next_action, target_q1, target_q2);
    float target_q = buffer.rewards[idx] + gamma * (fminf(target_q1, target_q2) - 0.2f * log_prob) * (1 - buffer.dones[idx]);
    
    // Current Q-values
    float current_q1, current_q2;
    q_net.q_values(state, action, current_q1, current_q2);
    
    // Q-losses
    q_losses[idx*2] = 0.5f * powf(current_q1 - target_q, 2);
    q_losses[idx*2+1] = 0.5f * powf(current_q2 - target_q, 2);
}

__global__ void sac_policy_update_kernel(
    SACBuffer buffer,
    GaussianPolicy policy,
    TwinQNetwork q_net,
    float* policy_losses,
    float alpha
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= buffer.capacity) return;

    float state[buffer.state_dim];
    for(int i = 0; i < buffer.state_dim; i++)
        state[i] = buffer.states[idx * buffer.state_dim + i];
    
    float action[buffer.action_dim];
    float log_prob;
    policy.sample_action(state, action, log_prob);
    
    float q1, q2;
    q_net.q_values(state, action, q1, q2);
    policy_losses[idx] = (alpha * log_prob) - fminf(q1, q2);
}

// Training execution
void train_sac() {
    const int state_dim = 28;
    const int action_dim = 4;
    const int buffer_size = 1000000;
    const float gamma = 0.99f;
    const float tau = 0.005f;
    const float alpha = 0.2f;
    
    SACBuffer buffer(buffer_size, state_dim, action_dim);
    GaussianPolicy policy(state_dim, action_dim, 256);
    GaussianPolicy target_policy(state_dim, action_dim, 256);
    TwinQNetwork q_net(state_dim, action_dim, 256);
    
    float* d_q_losses;
    float* d_policy_losses;
    CHECK_CUDA(cudaMalloc(&d_q_losses, buffer_size * 2 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_policy_losses, buffer_size * sizeof(float)));
    
    // Q-update
    dim3 block(256);
    dim3 grid((buffer_size + 255)/256);
    sac_q_update_kernel<<<grid, block>>>(buffer, q_net, policy, target_policy, d_q_losses, gamma, tau);
    
    // Policy update
    sac_policy_update_kernel<<<grid, block>>>(buffer, policy, q_net, d_policy_losses, alpha);

}

int main() {
    train_sac();
    return 0;
}