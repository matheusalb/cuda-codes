#include <iostream>
#define IN_TILE_DIM 1024

__global__ void bubbleSortSingleBlock(int *A, int n) {
    // Como é um único bloco, podemos usar a shared memory para fazer a ordenação
    __shared__ int tile[IN_TILE_DIM];

    int x = threadIdx.x; // como é um único bloco...

    if (x < n)
        tile[x] = A[x];  // Cada thread adiciona um elemento na shared memory
    __syncthreads();

    for (int i = 0; i < n; ++i) {
        // alterna indíce para as threads conseguirem contribuir
        int idx = 2 * (x / 2) + (i % 2);
        if (idx < n - 1) {
            if (tile[idx] > tile[idx + 1]) {
                // faz a troca
                int temp = tile[idx];
                tile[idx] = tile[idx + 1];
                tile[idx + 1] = temp;
            }
        }
        // sincroniza a cada iteração
        __syncthreads();  
    }

    if (x < n) {
        A[x] = tile[x];
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
    int blockSize = 1;
    int numBlocks = 1024;
    bubbleSortSingleBlock<<<blockSize, numBlocks>>>(array_d, n);

    // copia de volta para CPU
    cudaMemcpy(array_h, array_d, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(array_d);

    std::cout << "Array depois da ordenação: ";
    printArray(array_h, n);

}