#include <iostream>

__global__ void T_gpu(int *A, int *At, int n, int m) {
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (x < m && y < n)
        At[y * m + x] = A[x * n + y];
}

// Função para visualizar as matrizes
void printMatrix(int *matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        std::cout << "[";
        for (int j = 0; j < cols; ++j) {
            std::cout << matrix[i * cols + j];
            if (j < cols - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]";
        if (i < rows - 1) {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
}

int main() {
    int m = 3;
    int n = 4;
    
    int* A = new int[m * n];
    int* At = new int[n * m];

    int *d_A, *d_At;

    // preenchendo matriz
    for (int i= 0; i < n*m; i++)
        A[i] = i+1;

    std::cout << "input matrix" << std::endl;
    printMatrix(A, m, n);

    // alocando memória na GPU
    cudaMalloc(&d_A, m*n*sizeof(int));
    cudaMalloc(&d_At, n*m*sizeof(int));

    // copia para GPU
    cudaMemcpy(d_A, A, m*n*sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockSize(4, 4); // número de threads por bloco
    dim3 gridSize(ceil((n + blockSize.x - 1) / blockSize.x),
                    ceil((m + blockSize.y - 1) / blockSize.y)); // numero de blocos

    // executa
    T_gpu<<<gridSize, blockSize>>>(d_A, d_At, n, m);

    // copia de volta para CPU
    cudaMemcpy(At, d_At, n*m*sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "transposed matrix" << std::endl;
    printMatrix(At, n, m);

    // Liberar memória alocada na gpu e cpu
    delete [] A;
    delete [] At;
    cudaFree(d_A);
    cudaFree(d_At);
}