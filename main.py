import numpy as np
import time
from scipy.linalg.blas import zgemm

print("Илья Ляшенко Динисович\n")
print("090304-РПИа-o23")
#1
n = 1024
matrix1 = np.random.rand(n, n) + 1j * np.random.rand(n, n)
matrix2 = np.random.rand(n, n) + 1j * np.random.rand(n, n)
result_matrix = np.dot(matrix1, matrix2)

print(f"Ответ 1: {result_matrix}")


#2
matrix1 = np.random.rand(n, n) + 1j * np.random.rand(n, n)
matrix2 = np.random.rand(n, n) + 1j * np.random.rand(n, n)


matrix1_c = np.ascontiguousarray(matrix1.T)
matrix2_c = np.ascontiguousarray(matrix2.T)


result_matrix_blas = zgemm(alpha=1.0, a=matrix1_c, b=matrix2_c)

print(f"Ответ 2: {result_matrix_blas}")




#3


def strassen_multiply(A, B):
    n = len(A)

    if n == 1:
        return A * B


    mid = n // 2
    A11 = A[:mid, :mid]
    A12 = A[:mid, mid:]
    A21 = A[mid:, :mid]
    A22 = A[mid:, mid:]

    B11 = B[:mid, :mid]
    B12 = B[:mid, mid:]
    B21 = B[mid:, :mid]
    B22 = B[mid:, mid:]

    P1 = strassen_multiply(A11 + A22, B11 + B22)
    P2 = strassen_multiply(A21 + A22, B11)
    P3 = strassen_multiply(A11, B12 - B22)
    P4 = strassen_multiply(A22, B21 - B11)
    P5 = strassen_multiply(A11 + A12, B22)
    P6 = strassen_multiply(A21 - A11, B11 + B12)
    P7 = strassen_multiply(A12 - A22, B21 + B22)

    C11 = P1 + P4 - P5 + P7
    C12 = P3 + P5
    C21 = P2 + P4
    C22 = P1 - P2 + P3 + P6

    C = np.zeros((n, n), dtype=np.complex128)
    C[:mid, :mid] = C11
    C[:mid, mid:] = C12
    C[mid:, :mid] = C21
    C[mid:, mid:] = C22

    return C


start = time.time()


matrix1 = np.random.rand(1024, 1024) + 1j * np.random.rand(1024, 1024)
matrix2 = np.random.rand(1024, 1024) + 1j * np.random.rand(1024, 1024)

result_matrix_strassen = strassen_multiply(matrix1, matrix2)


finish = time.time()
t = (finish-start)*1000
c = 2*1024**3
print(f"Cложность алгоритма (c) = {c}")
print(f"Производительность MFlops (p) = {c/t*10**(-6)}")
print(f"Ответ 3: {result_matrix_strassen}")
