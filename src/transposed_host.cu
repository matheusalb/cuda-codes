#include <iostream>

__host__ void t_seq(int *A, int *At, int n, int m) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            At[j * m + i] = A[i * n + j];
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

    // preenchendo matriz
    for (int i= 0; i < n*m; i++)
        A[i] = i+1;

    std::cout << "input matrix" << std::endl;
    printMatrix(A, m, n);

    t_seq(A, At, n, m);

    std::cout << "transposed matrix" << std::endl;
    printMatrix(At, n, m);

    delete[] A;
    delete[] At;

    return 0;
}