#include <iostream>
#define IN_TILE_DIM 1024

__global__ void bubbleSortMultiBlock(int *A, int n) {
    __shared__ int tile[IN_TILE_DIM];

    // Cálculo de índices para segmentação do vetor
    int start = blockIdx.x * blockDim.x;
    int end = min(start + blockDim.x, n);
    int x = threadIdx.x;
    int localIdx = start + x;

    if (localIdx < end) {
        tile[x] = A[localIdx];  // Carrega parte do vetor na memória compartilhada
    }
    __syncthreads();

    for (int step = 0; step < (end - start); ++step) {
        for (int j = 0; j < (end - start - step - 1); ++j) {
            if (tile[j] > tile[j + 1]) {
                int temp = tile[j];
                tile[j] = tile[j + 1];
                tile[j + 1] = temp;
            }
            __syncthreads();
        }
    }

    if (localIdx < end) {
        A[localIdx] = tile[x];  // Escreve de volta para a memória global
    }
}

// Função para imprimir um vetor
void printArray(const int* arr, int N) {
    std::cout << "[";
    for (int i = 0; i < N; i++) {
        if (i < N - 1)
            std::cout << arr[i] << ", ";
        else
            std::cout << arr[i];
    }
    std::cout << "]" << std::endl;
}


int main() {
    int n = 100;
    int array_h[n];
    int *array_d;

    // preenchendo array com valores decrescentes
    for (int i = 0; i < n; i++) {
        array_h[i] = n - i;
    }
    std::cout << "Array antes da ordenação: ";
    printArray(array_h, n);

    // alocando memória na GPU
    cudaMalloc((void**)&array_d, n * sizeof(int));

    // copiando dados para a GPU
    cudaMemcpy(array_d, array_h, n * sizeof(int), cudaMemcpyHostToDevice);

    // executa
    int numBlocks = 1024;
    int blockSize = (n + numBlocks - 1) / numBlocks;
    bubbleSortMultiBlock<<<blockSize, numBlocks>>>(array_d, n);

    // copia de volta para CPU
    cudaMemcpy(array_h, array_d, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(array_d);

    std::cout << "Array depois da ordenação: ";
    printArray(array_h, n);

}
