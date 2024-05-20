#include <iostream>

#define IN_TILE_DIM 32
#define OUT_TILE_DIM 32

__global__ void T_opt(int *A, int *At, int n, int m) {
    __shared__ int tile[IN_TILE_DIM][OUT_TILE_DIM];

    int x = blockIdx.x * OUT_TILE_DIM + threadIdx.x; // j é o x
    int y = blockIdx.y * OUT_TILE_DIM + threadIdx.y;

    if (y < m && x < n)
        tile[threadIdx.y][threadIdx.x] = A[y * n + x];

    __syncthreads();

    // tranpõe a matriz
    y = blockIdx.x * OUT_TILE_DIM + threadIdx.y; // i é o y
    x = blockIdx.y * OUT_TILE_DIM + threadIdx.x;

    // troca n por m
    if (y < n && x < m)
        At[y * m + x] = tile[threadIdx.x][threadIdx.y];
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

    dim3 blockSize(32, 32); // n threads por bloco
    dim3 gridSize(ceil((n + blockSize.x - 1) / blockSize.x),
                    ceil((m + blockSize.y - 1) / blockSize.y)); // numero de blcos

    // executa
    T_opt<<<gridSize, blockSize>>>(d_A, d_At, n, m);

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