#include <iostream>
#include <cuda_runtime.h>

#define N 16  // Size of the matrices

// CUDA kernel for matrix multiplication
__global__ void matrixMulCUDA(float *C, const float *A, const float *B, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    if(row < width && col < width) {
        for(int k = 0; k < width; ++k) {
            sum += A[row * width + k] * B[k * width + col];
        }
        C[row * width + col] = sum;
    }
}

int main() {
    int size = N * N * sizeof(float);
    float h_A[N*N], h_B[N*N], h_C[N*N];

    // Initialize matrices
    for(int i = 0; i < N*N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i % N);
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the kernel
    matrixMulCUDA<<<blocksPerGrid, threadsPerBlock>>>(d_C, d_A, d_B, N);

    // Copy the result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Display a portion of the result matrix
    std::cout << "Result matrix C:" << std::endl;
    for(int i = 0; i < N; ++i) {
        for(int j = 0; j < N; ++j) {
            std::cout << h_C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

