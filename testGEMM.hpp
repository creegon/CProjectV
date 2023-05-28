#include <iostream>
#include <string>
#include <random>
#include <chrono>
#include <cstdlib>  
#include <stdexcept>
#include <mmintrin.h>
#include <xmmintrin.h>  // SSE
#include <pmmintrin.h>  // SSE2
#include <emmintrin.h>  // SSE3

#include "cblas.h"
using namespace std;

// #define A(i, j) A[(i) * lda + (j)]
// #define B(i, j) B[(i) * ldb + (j)]
// #define C(i, j) C[(i) * ldc + (j)]
#define A(i, j) A[(j) * lda + (i)]
#define B(i, j) B[(j) * ldb + (i)]
#define C(i, j) C[(j) * ldc + (i)]
#define X(i)    X[(i) * ldx]
#define mc 4000
#define kc 4000

typedef union
{
  __m128d v;
  double d[2];
} v2df_t;






double* create_random_matrix(int M, int N);
void test_own_dgemm(blasint M, blasint N, blasint K, double* A, double* B, double* C);
void test_final_dgemm(blasint M, blasint N, blasint K, double* A, double* B, double* C);   
void own_dgemm(const CBLAS_ORDER Order, const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
                 blasint M, blasint N, blasint K, double alpha,  const double* A, blasint lda,
                 const double* B, blasint ldb, double beta, double* C, blasint ldc);
void AddDot(int K, const double *X, int ldx, const double *B, double *C, double alpha, double beta);
void AddDot4x1(int K, const double *A, blasint lda, const double *B, blasint ldb, double *C, blasint ldc, double alpha, double beta);    
void regAddDot4x1(int K, const double *A, blasint lda, const double *B, blasint ldb, double *C, blasint ldc, double alpha, double beta); 
void ptrRegAddDot4x1(int K, const double *A, blasint lda, const double *B, blasint ldb, double *C, blasint ldc, double alpha, double beta);     
void AddDot4x4(int K, const double *A, blasint lda, const double *B, blasint ldb, double *C, blasint ldc, double alpha, double beta);
void vecAddDot4x4(int K, const double *A, blasint lda, const double *B, blasint ldb, double *C, blasint ldc, double alpha, double beta);
void pacVecAddDot4x4(int K, const double *A, blasint lda, const double *B, blasint ldb, double *C, blasint ldc, double alpha, double beta);
void doublePacVecAddDot4x4(int K, const double *A, blasint lda, const double *B, blasint ldb, double *C, blasint ldc, double alpha, double beta);
void smallBlock(int M, int N, int K, const double *A, blasint lda, const double *B, blasint ldb, double *C, blasint ldc, double alpha, double beta);
void smallBlockWithPackage(int M, int N, int K, const double *A, blasint lda, const double *B, blasint ldb, double *C, blasint ldc, double alpha, double beta);
void smallBlockWithDoublePackage(int M, int N, int K, const double *A, blasint lda, const double *B, blasint ldb, double *C, blasint ldc, double alpha, double beta);
void PackMatrixB(int K, const double *B, int ldb, double *packageB);       
void PackMatrixA(int K, const double *A, int lda, double *packageA);

void smallBlockWithDoublePackage_new(const CBLAS_ORDER Order, const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, int M, int N, int K, const double *A, blasint lda, const double *B, blasint ldb, double *C, blasint ldc, double alpha, double beta);
void PackMatrixB_new(int K, const double *B, int ldb, double *packageB, const CBLAS_ORDER Order);
void PackMatrixA_new(int K, const double *A, int lda, double *packageA, const CBLAS_ORDER Order);
void doublePacVecAddDot4x4_new(int K, const double *A, blasint lda, const double *B, blasint ldb, double *C, blasint ldc, double alpha, double beta, const CBLAS_ORDER Order);
void check_nullptr(const double* ptr, const char* name);
void check_size(blasint size, const char* name);
void transMatrix(const CBLAS_ORDER Order, const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
                 blasint M, blasint N, blasint K, const double*& A, blasint lda, const double*& B, blasint ldb);
void expandMatrixA(blasint M, blasint K, blasint lda, const double*& A);
void expandMatrixB(blasint K, blasint N, blasint ldb, const double*& B);
void expandMatrixC(blasint M, blasint N, blasint ldc, const double*& C);           