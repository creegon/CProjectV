#include "testGEMM.hpp"


static double own_time = 0.0;
static double real_time = 0.0;
//创建一个双精度浮点数的随机矩阵
double* create_random_matrix(blasint M, blasint N) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(2.0, 2.0);

    // Create an aligned memory space
    double* matrix = nullptr;
    if (posix_memalign((void**)&matrix, 32, M * N * sizeof(double)) != 0) {
        throw std::runtime_error("Failed to allocate aligned memory");
    }
    

    for (int i = 0; i < M * N; ++i) {
        matrix[i] = dis(gen);
    }

    return matrix;
}

//基本的矩阵乘法函数
void test_own_dgemm(blasint M, blasint N, blasint K, double* A, double* B, double* C) {
    double alpha = 1.0;
    double beta = 0.0;

    auto start = std::chrono::high_resolution_clock::now();

    own_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, M, B, K, beta, C, N);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    //同时打印M,N,K地信息
    std::cout << "own: " << elapsed.count() << " seconds. M = " << M << " , N = " << N << ", K =" << K << " \n";
    own_time = elapsed.count();

    //验证下结果：打印C的前10个元素
    cout << "own result: \n";
    for (int i = 0; i < 10; ++i) {
        std::cout << C[i] << " ";
    }
}

void test_final_dgemm(blasint M, blasint N, blasint K, double* A, double* B, double* C) {
    double alpha = 1.0;
    double beta = 0.0;

    auto start = std::chrono::high_resolution_clock::now();

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, M, B, K, beta, C, M);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "real: " << elapsed.count() << " seconds. M = " << M << " , N = " << N << ", K =" << K << " \n";
    real_time = elapsed.count();

    //验证下结果：打印C的前10个元素
    cout << "real result: \n";
    for (int i = 0; i < 10; ++i) {
        std::cout << C[i] << " ";
    }
}

void own_dgemm(const CBLAS_ORDER Order, const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
                 blasint M, blasint N, blasint K, double alpha,  const double* A, blasint lda,
                 const double* B, blasint ldb, double beta, double* C, blasint ldc)
{
    //计算前，先把beta那份的C给加了（其实就是让C矩阵每个元素都乘beta）
    double* C_ptr = &C(0,0);
    for (int i = 0; i < M * N; ++i) {
         //让C_ptr指针指向C矩阵的每个元素，然后让C_ptr指向的元素乘beta
        *C_ptr *= beta;
        C_ptr++;
    }
    /*
        version 1.0
    */

    // for (int i = 0; i < M; i++){        
    //     for (int j = 0; j < N; j++){        
    //         for (int p = 0; p < K; p++){        
	//             C( i,j ) = beta * C( i,j ) +  alpha * A( i,p ) * B( p,j );
    //         }
    //     }
    // }

    /*
        version 2.0
    */
    
    // for(int j = 0; j < N; j++){
    //     for(int i = 0; i < M; i++){
    //         AddDot(K, &A(i,0), lda, &B(0,j), &C(i,j), alpha, beta);
    //     }
    // }

    /*
        version 3.0
    */

    // for(int j = 0; j < N; j += 4){
    //     for(int i = 0; i < M; i++){
    //         AddDot(K, &A(i,0), lda, &B(0,j), &C(i,j), alpha, beta);
    //         AddDot(K, &A(i,0), lda, &B(0,j + 1), &C(i,j + 1), alpha, beta);
    //         AddDot(K, &A(i,0), lda, &B(0,j + 2), &C(i,j + 2), alpha, beta);
    //         AddDot(K, &A(i,0), lda, &B(0,j + 3), &C(i,j + 3), alpha, beta);
    //     }
    // }

    /*
        version 4.0
    */

    // for(int j = 0; j < N; j += 4){
    //     for(int i = 0; i < M; i++){
    //         AddDot4x1(K, &A(i,0), lda, &B(0,j), ldb, &C(i,j), ldc, alpha, beta);
    //     }
    // }

    /*
        version 5.0
    */
   
    // for(int j = 0; j < N; j += 4){
    //     for(int i = 0; i < M; i++){
    //         regAddDot4x1(K, &A(i,0), lda, &B(0,j), ldb, &C(i,j), ldc, alpha, beta);
    //     }
    // }

    /*
        version 6.0
    */

    // for(int j = 0; j < N; j += 4){
    //     for(int i = 0; i < M; i++){
    //         ptrRegAddDot4x1(K, &A(i,0), lda, &B(0,j), ldb, &C(i,j), ldc, alpha, beta);
    //     }
    // }

    /*
        version 7.0
    */

    // for(int j = 0; j < N; j += 4){
    //     for(int i = 0; i < M; i += 4){
    //         AddDot4x4(K, &A(i,0), lda, &B(0,j), ldb, &C(i,j), ldc, alpha, beta);
    //     }
    // }

    /*
        version 8.0
    */

//    for(int j = 0; j < N; j += 4){
//         for(int i = 0; i < M; i += 4){
//             vecAddDot4x4(K, &A(i,0), lda, &B(0,j), ldb, &C(i,j), ldc, alpha, beta);
//         }
//     }

    /*
        version 9.0
    */

    // for(int p = 0; p < K; p += kc){
    //     int pb = min(K - p, kc);
    //     for (int i = 0; i < M; i += mc){
    //         int ib = min(M - i, mc);
    //         smallBlock(ib, N, pb, &A(i, p), lda, &B(p, 0), ldb, &C(i, 0), ldc, alpha, beta);
    //     }
    // }
    
    /*
        version 10.0
    */

    // for(int p = 0; p < K; p += kc){
    //     int pb = min(K - p, kc);
    //     for (int i = 0; i < M; i += mc){
    //         int ib = min(M - i, mc);
    //         smallBlockWithPackage(ib, N, pb, &A(i, p), lda, &B(p, 0), ldb, &C(i, 0), ldc, alpha, beta);
    //     }
    // }

    /*
        version 11.0
    */

    for(int p = 0; p < K; p += kc){
        int pb = min(K - p, kc);
        for (int i = 0; i < M; i += mc){
            int ib = min(M - i, mc);
            smallBlockWithDoublePackage(ib, N, pb, &A(i, p), lda, &B(p, 0), ldb, &C(i, 0), ldc, alpha, beta);
        }
    }

}      

void AddDot(int K, const double *X, int ldx, const double *B, double *C, double alpha, double beta)
{
    for(int p = 0; p < K; p++){
        *C = alpha * X(p) * B[p] + beta * *C;
    }
}

void AddDot4x1(int K, const double *A, blasint lda, const double *B, blasint ldb, double *C, blasint ldc, double alpha, double beta)
{
    for(int p = 0; p < K; p++){
        C( 0, 0 ) = alpha * B( p, 0 ) * A( 0, p ) + beta * C( 0, 0 );     
        C( 1, 0 ) = alpha * B( p, 0 ) * A( 1, p ) + beta * C( 1, 0 );    
        C( 2, 0 ) = alpha * B( p, 0 ) * A( 2, p ) + beta * C( 2, 0 );
        C( 3, 0 ) = alpha * B( p, 0 ) * A( 3, p ) + beta * C( 3, 0 ); 
    }
}

void regAddDot4x1(int K, const double *A, blasint lda, const double *B, blasint ldb, double *C, blasint ldc, double alpha, double beta)
{
    double c_00_reg, c_10_reg, c_20_reg, c_30_reg, b_p0_reg = 0.0;
    for(int p = 0; p < K; p++){
        b_p0_reg = B( p, 0 );
        c_00_reg += b_p0_reg * A( 0, p );     
        c_10_reg += b_p0_reg * A( 1, p );     
        c_20_reg += b_p0_reg * A( 2, p );     
        c_30_reg += b_p0_reg * A( 3, p );    
    }
    C( 0, 0 ) = alpha * c_00_reg + beta * C( 0, 0 ); 
    C( 1, 0 ) = alpha * c_10_reg + beta * C( 1, 0 );
    C( 2, 0 ) = alpha * c_20_reg + beta * C( 2, 0 );
    C( 3, 0 ) = alpha * c_30_reg + beta * C( 3, 0 );
}

void ptrRegAddDot4x1(int K, const double *A, blasint lda, const double *B, blasint ldb, double *C, blasint ldc, double alpha, double beta)
{
    double c_00_reg, c_10_reg, c_20_reg, c_30_reg, b_p0_reg = 0.0;
    const double *ap0_ptr = &A( 0, 0 );
    const double *ap1_ptr = &A( 1, 0 );
    const double *ap2_ptr = &A( 2, 0 );
    const double *ap3_ptr = &A( 3, 0 );

    for(int p = 0; p < K; p += 4){
        b_p0_reg = B( p, 0 );
        c_00_reg += b_p0_reg * *ap0_ptr;     
        c_10_reg += b_p0_reg * *ap1_ptr;     
        c_20_reg += b_p0_reg * *ap2_ptr;     
        c_30_reg += b_p0_reg * *ap3_ptr;   

        b_p0_reg = B( p + 1, 0 );
        c_00_reg += b_p0_reg * *(ap0_ptr + 1);     
        c_10_reg += b_p0_reg * *(ap1_ptr + 1);     
        c_20_reg += b_p0_reg * *(ap2_ptr + 1);  
        c_30_reg += b_p0_reg * *(ap3_ptr + 1); 

        b_p0_reg = B( p + 2, 0 );
        c_00_reg += b_p0_reg * *(ap0_ptr + 2); 
        c_10_reg += b_p0_reg * *(ap1_ptr + 2); 
        c_20_reg += b_p0_reg * *(ap2_ptr + 2); 
        c_30_reg += b_p0_reg * *(ap3_ptr + 2);

        b_p0_reg = B( p + 3, 0 );
        c_00_reg += b_p0_reg * *(ap0_ptr + 3); 
        c_10_reg += b_p0_reg * *(ap1_ptr + 3); 
        c_20_reg += b_p0_reg * *(ap2_ptr + 3); 
        c_30_reg += b_p0_reg * *(ap3_ptr + 3);

        ap0_ptr += 4;
        ap1_ptr += 4;
        ap2_ptr += 4;
        ap3_ptr += 4;
    }
    C( 0, 0 ) = alpha * c_00_reg + beta * C( 0, 0 ); 
    C( 1, 0 ) = alpha * c_10_reg + beta * C( 1, 0 ); 
    C( 2, 0 ) = alpha * c_20_reg + beta * C( 2, 0 );
    C( 3, 0 ) = alpha * c_30_reg + beta * C( 3, 0 );
}

void AddDot4x4(int K, const double *A, blasint lda, const double *B, blasint ldb, double *C, blasint ldc, double alpha, double beta)
{

  //定义寄存器变量
  double 
  c_00_reg,   c_01_reg,   c_02_reg,   c_03_reg,  
  c_10_reg,   c_11_reg,   c_12_reg,   c_13_reg,  
  c_20_reg,   c_21_reg,   c_22_reg,   c_23_reg,  
  c_30_reg,   c_31_reg,   c_32_reg,   c_33_reg, 

  b_p0_reg,   b_p1_reg,   b_p2_reg,   b_p3_reg,

  a_0p_reg,   a_1p_reg,   a_2p_reg,   a_3p_reg = 0.0;

  //定义指针变量
    
  const double* a_0p_ptr = &B( 0, 0 );
  const double* a_1p_ptr = &B( 1, 0 );
  const double* a_2p_ptr = &B( 2, 0 );
  const double* a_3p_ptr = &B( 3, 0 );

  for (int p = 0; p < K; p++){
    b_p0_reg = B( p, 0 );
    b_p1_reg = A( p, 1 );
    b_p2_reg = A( p, 2 );
    b_p3_reg = A( p, 3 );

    a_0p_reg = *a_0p_ptr++;
    a_1p_reg = *a_1p_ptr++;
    a_2p_reg = *a_2p_ptr++;
    a_3p_reg = *a_3p_ptr++;

    c_00_reg += b_p0_reg * a_0p_reg;
    c_01_reg += b_p1_reg * a_0p_reg;

    c_10_reg += b_p0_reg * a_1p_reg;
    c_11_reg += b_p1_reg * a_1p_reg;

    c_20_reg += b_p0_reg * a_2p_reg;
    c_21_reg += b_p1_reg * a_2p_reg;

    c_30_reg += b_p0_reg * a_3p_reg;
    c_31_reg += b_p1_reg * a_3p_reg;


    c_02_reg += b_p2_reg * a_0p_reg;
    c_03_reg += b_p3_reg * a_0p_reg;

    c_12_reg += b_p2_reg * a_1p_reg;
    c_13_reg += b_p3_reg * a_1p_reg;

    c_22_reg += b_p2_reg * a_2p_reg;
    c_23_reg += b_p3_reg * a_2p_reg;

    c_32_reg += b_p2_reg * a_3p_reg;
    c_33_reg += b_p3_reg * a_3p_reg;
  }

  C( 0, 0 ) += alpha * c_00_reg;
  C( 1, 0 ) += alpha * c_10_reg; 
  C( 2, 0 ) += alpha * c_20_reg;
  C( 3, 0 ) += alpha * c_30_reg; 

  C( 0, 1 ) += alpha * c_01_reg;  
  C( 1, 1 ) += alpha * c_11_reg;
  C( 2, 1 ) += alpha * c_21_reg;
  C( 3, 1 ) += alpha * c_31_reg;


  C( 0, 2 ) += alpha * c_02_reg;  
  C( 1, 2 ) += alpha * c_12_reg;
  C( 2, 2 ) += alpha * c_22_reg;
  C( 3, 2 ) += alpha * c_32_reg;


  C( 0, 3 ) += alpha * c_03_reg;
  C( 1, 3 ) += alpha * c_13_reg; 
  C( 2, 3 ) += alpha * c_23_reg;
  C( 3, 3 ) += alpha * c_33_reg; 
  
}

void vecAddDot4x4(int K, const double *A, blasint lda, const double *B, blasint ldb, double *C, blasint ldc, double alpha, double beta)
{
    

  //定义寄存器变量
  v2df_t 
  c_00_c_01_vreg,    c_10_c_11_vreg,    c_20_c_21_vreg,    c_30_c_31_vreg,
  c_02_c_03_vreg,    c_12_c_13_vreg,    c_22_c_23_vreg,    c_32_c_33_vreg,
  b_p0_b_p1_vreg,
  b_p2_b_p3_vreg,
  a_0p_vreg, a_1p_vreg, a_2p_vreg, a_3p_vreg; 

  //定义指针变量
    
  const double* a_0p_ptr = &A( 0, 0 );
  const double* a_1p_ptr = &A( 1, 0 );
  const double* a_2p_ptr = &A( 2, 0 );
  const double* a_3p_ptr = &A( 3, 0 );

  c_00_c_01_vreg.v = _mm_setzero_pd();   
  c_10_c_11_vreg.v = _mm_setzero_pd();
  c_20_c_21_vreg.v = _mm_setzero_pd(); 
  c_30_c_31_vreg.v = _mm_setzero_pd(); 
  c_02_c_03_vreg.v = _mm_setzero_pd();   
  c_12_c_13_vreg.v = _mm_setzero_pd();  
  c_22_c_23_vreg.v = _mm_setzero_pd();   
  c_32_c_33_vreg.v = _mm_setzero_pd(); 

  for (int p = 0; p < K; p++){
    b_p0_b_p1_vreg.v = _mm_load_pd( (double *) &B( p, 0 ) );
    b_p2_b_p3_vreg.v = _mm_load_pd( (double *) &B( p, 2 ) );

    a_0p_vreg.v = _mm_loaddup_pd( (double *) a_0p_ptr++ );  
    a_1p_vreg.v = _mm_loaddup_pd( (double *) a_1p_ptr++ );   
    a_2p_vreg.v = _mm_loaddup_pd( (double *) a_2p_ptr++ );   
    a_3p_vreg.v = _mm_loaddup_pd( (double *) a_3p_ptr++ );  

    c_00_c_01_vreg.v += b_p0_b_p1_vreg.v * a_0p_vreg.v;
    c_10_c_11_vreg.v += b_p0_b_p1_vreg.v * a_1p_vreg.v;
    c_20_c_21_vreg.v += b_p0_b_p1_vreg.v * a_2p_vreg.v;
    c_30_c_31_vreg.v += b_p0_b_p1_vreg.v * a_3p_vreg.v;

    c_02_c_03_vreg.v += b_p2_b_p3_vreg.v * a_0p_vreg.v;
    c_12_c_13_vreg.v += b_p2_b_p3_vreg.v * a_1p_vreg.v;
    c_22_c_23_vreg.v += b_p2_b_p3_vreg.v * a_2p_vreg.v;
    c_32_c_33_vreg.v += b_p2_b_p3_vreg.v * a_3p_vreg.v;
  }

  C( 0, 0 ) += alpha * c_00_c_01_vreg.d[0];  
  C( 1, 0 ) += alpha * c_10_c_11_vreg.d[0];
  C( 2, 0 ) += alpha * c_20_c_21_vreg.d[0];
  C( 3, 0 ) += alpha * c_30_c_31_vreg.d[0]; 

  C( 0, 1 ) += alpha * c_00_c_01_vreg.d[1];  
  C( 1, 1 ) += alpha * c_10_c_11_vreg.d[1];
  C( 2, 1 ) += alpha * c_20_c_21_vreg.d[1];
  C( 3, 1 ) += alpha * c_30_c_31_vreg.d[1];


  C( 0, 2 ) += alpha * c_02_c_03_vreg.d[0];  
  C( 1, 2 ) += alpha * c_12_c_13_vreg.d[0];
  C( 2, 2 ) += alpha * c_22_c_23_vreg.d[0];
  C( 3, 2 ) += alpha * c_32_c_33_vreg.d[0];


  C( 0, 3 ) += alpha * c_02_c_03_vreg.d[1];
  C( 1, 3 ) += alpha * c_12_c_13_vreg.d[1]; 
  C( 2, 3 ) += alpha * c_22_c_23_vreg.d[1];
  C( 3, 3 ) += alpha * c_32_c_33_vreg.d[1]; 
  
}

void pacVecAddDot4x4(int K, const double *A, blasint lda, const double *B, blasint ldb, double *C, blasint ldc, double alpha, double beta)
{
    

  //定义寄存器变量
  v2df_t 
  c_00_c_01_vreg,    c_10_c_11_vreg,    c_20_c_21_vreg,    c_30_c_31_vreg,
  c_02_c_03_vreg,    c_12_c_13_vreg,    c_22_c_23_vreg,    c_32_c_33_vreg,
  b_p0_b_p1_vreg,
  b_p2_b_p3_vreg,
  a_0p_vreg, a_1p_vreg, a_2p_vreg, a_3p_vreg; 

  //定义指针变量
    
  const double* a_0p_ptr = &A( 0, 0 );
  const double* a_1p_ptr = &A( 1, 0 );
  const double* a_2p_ptr = &A( 2, 0 );
  const double* a_3p_ptr = &A( 3, 0 );

  const double* b_p0_b_p1_ptr = &B( 0, 0 );

  c_00_c_01_vreg.v = _mm_setzero_pd();   
  c_10_c_11_vreg.v = _mm_setzero_pd();
  c_20_c_21_vreg.v = _mm_setzero_pd(); 
  c_30_c_31_vreg.v = _mm_setzero_pd(); 
  c_02_c_03_vreg.v = _mm_setzero_pd();   
  c_12_c_13_vreg.v = _mm_setzero_pd();  
  c_22_c_23_vreg.v = _mm_setzero_pd();   
  c_32_c_33_vreg.v = _mm_setzero_pd(); 

  for (int p = 0; p < K; p++){
    b_p0_b_p1_vreg.v = _mm_load_pd( (double *) b_p0_b_p1_ptr );
    b_p2_b_p3_vreg.v = _mm_load_pd( (double *) b_p0_b_p1_ptr + 2 );
    b_p0_b_p1_ptr += 4;


    a_0p_vreg.v = _mm_loaddup_pd( (double *) a_0p_ptr++ );  
    a_1p_vreg.v = _mm_loaddup_pd( (double *) a_1p_ptr++ );   
    a_2p_vreg.v = _mm_loaddup_pd( (double *) a_2p_ptr++ );   
    a_3p_vreg.v = _mm_loaddup_pd( (double *) a_3p_ptr++ );  

    c_00_c_01_vreg.v += b_p0_b_p1_vreg.v * a_0p_vreg.v;
    c_10_c_11_vreg.v += b_p0_b_p1_vreg.v * a_1p_vreg.v;
    c_20_c_21_vreg.v += b_p0_b_p1_vreg.v * a_2p_vreg.v;
    c_30_c_31_vreg.v += b_p0_b_p1_vreg.v * a_3p_vreg.v;

    c_02_c_03_vreg.v += b_p2_b_p3_vreg.v * a_0p_vreg.v;
    c_12_c_13_vreg.v += b_p2_b_p3_vreg.v * a_1p_vreg.v;
    c_22_c_23_vreg.v += b_p2_b_p3_vreg.v * a_2p_vreg.v;
    c_32_c_33_vreg.v += b_p2_b_p3_vreg.v * a_3p_vreg.v;
  }

  C( 0, 0 ) += alpha * c_00_c_01_vreg.d[0];  
  C( 1, 0 ) += alpha * c_10_c_11_vreg.d[0];
  C( 2, 0 ) += alpha * c_20_c_21_vreg.d[0];
  C( 3, 0 ) += alpha * c_30_c_31_vreg.d[0]; 

  C( 0, 1 ) += alpha * c_00_c_01_vreg.d[1];  
  C( 1, 1 ) += alpha * c_10_c_11_vreg.d[1];
  C( 2, 1 ) += alpha * c_20_c_21_vreg.d[1];
  C( 3, 1 ) += alpha * c_30_c_31_vreg.d[1];


  C( 0, 2 ) += alpha * c_02_c_03_vreg.d[0];  
  C( 1, 2 ) += alpha * c_12_c_13_vreg.d[0];
  C( 2, 2 ) += alpha * c_22_c_23_vreg.d[0];
  C( 3, 2 ) += alpha * c_32_c_33_vreg.d[0];


  C( 0, 3 ) += alpha * c_02_c_03_vreg.d[1];
  C( 1, 3 ) += alpha * c_12_c_13_vreg.d[1]; 
  C( 2, 3 ) += alpha * c_22_c_23_vreg.d[1];
  C( 3, 3 ) += alpha * c_32_c_33_vreg.d[1]; 
  
}

void doublePacVecAddDot4x4(int K, const double *A, blasint lda, const double *B, blasint ldb, double *C, blasint ldc, double alpha, double beta)
{
    

  //定义寄存器变量
  v2df_t 
  c_00_c_01_vreg,    c_10_c_11_vreg,    c_20_c_21_vreg,    c_30_c_31_vreg,
  c_02_c_03_vreg,    c_12_c_13_vreg,    c_22_c_23_vreg,    c_32_c_33_vreg,
  b_p0_b_p1_vreg,
  b_p2_b_p3_vreg,
  a_0p_vreg, a_1p_vreg, a_2p_vreg, a_3p_vreg; 

  //定义指针变量
    
  const double* a_0p_ptr = &A( 0, 0 );

  const double* b_p0_b_p1_ptr = &B( 0, 0 );

  c_00_c_01_vreg.v = _mm_setzero_pd();   
  c_10_c_11_vreg.v = _mm_setzero_pd();
  c_20_c_21_vreg.v = _mm_setzero_pd(); 
  c_30_c_31_vreg.v = _mm_setzero_pd(); 
  c_02_c_03_vreg.v = _mm_setzero_pd();   
  c_12_c_13_vreg.v = _mm_setzero_pd();  
  c_22_c_23_vreg.v = _mm_setzero_pd();   
  c_32_c_33_vreg.v = _mm_setzero_pd(); 

  

    
  for (int p = 0; p < K; p++){
    b_p0_b_p1_vreg.v = _mm_load_pd( (double *) b_p0_b_p1_ptr );
    b_p2_b_p3_vreg.v = _mm_load_pd( (double *) b_p0_b_p1_ptr + 2 );
    b_p0_b_p1_ptr += 4;


    a_0p_vreg.v = _mm_loaddup_pd( (double *) a_0p_ptr );  
    a_1p_vreg.v = _mm_loaddup_pd( (double *) a_0p_ptr + 1 );   
    a_2p_vreg.v = _mm_loaddup_pd( (double *) a_0p_ptr + 2 );   
    a_3p_vreg.v = _mm_loaddup_pd( (double *) a_0p_ptr + 3 );
    a_0p_ptr += 4;  

    c_00_c_01_vreg.v += b_p0_b_p1_vreg.v * a_0p_vreg.v;
    c_10_c_11_vreg.v += b_p0_b_p1_vreg.v * a_1p_vreg.v;
    c_20_c_21_vreg.v += b_p0_b_p1_vreg.v * a_2p_vreg.v;
    c_30_c_31_vreg.v += b_p0_b_p1_vreg.v * a_3p_vreg.v;

    c_02_c_03_vreg.v += b_p2_b_p3_vreg.v * a_0p_vreg.v;
    c_12_c_13_vreg.v += b_p2_b_p3_vreg.v * a_1p_vreg.v;
    c_22_c_23_vreg.v += b_p2_b_p3_vreg.v * a_2p_vreg.v;
    c_32_c_33_vreg.v += b_p2_b_p3_vreg.v * a_3p_vreg.v;
  }

  C( 0, 0 ) += alpha * c_00_c_01_vreg.d[0];  
  C( 1, 0 ) += alpha * c_10_c_11_vreg.d[0];
  C( 2, 0 ) += alpha * c_20_c_21_vreg.d[0];
  C( 3, 0 ) += alpha * c_30_c_31_vreg.d[0]; 

  C( 0, 1 ) += alpha * c_00_c_01_vreg.d[1];  
  C( 1, 1 ) += alpha * c_10_c_11_vreg.d[1];
  C( 2, 1 ) += alpha * c_20_c_21_vreg.d[1];
  C( 3, 1 ) += alpha * c_30_c_31_vreg.d[1];


  C( 0, 2 ) += alpha * c_02_c_03_vreg.d[0];  
  C( 1, 2 ) += alpha * c_12_c_13_vreg.d[0];
  C( 2, 2 ) += alpha * c_22_c_23_vreg.d[0];
  C( 3, 2 ) += alpha * c_32_c_33_vreg.d[0];


  C( 0, 3 ) += alpha * c_02_c_03_vreg.d[1];
  C( 1, 3 ) += alpha * c_12_c_13_vreg.d[1]; 
  C( 2, 3 ) += alpha * c_22_c_23_vreg.d[1];
  C( 3, 3 ) += alpha * c_32_c_33_vreg.d[1]; 
  
}

void smallBlock(int M, int N, int K, const double *A, blasint lda, const double *B, blasint ldb, double *C, blasint ldc, double alpha, double beta)
{   
    
    for(int j = 0; j < N; j += 4){
        for(int i = 0; i < M; i += 4){
            vecAddDot4x4(K, &A(i,0), lda, &B(0,j), ldb, &C(i,j), ldc, alpha, beta);
        }
    }
}

void smallBlockWithPackage(int M, int N, int K, const double *A, blasint lda, const double *B, blasint ldb, double *C, blasint ldc, double alpha, double beta)
{   
    // allocate aligned memory
    double* packageB = nullptr;
    if (posix_memalign((void**)&packageB, 32, K * N * sizeof(double)) != 0) {
        throw std::runtime_error("Failed to allocate aligned memory");
    }

    for(int j = 0; j < M; j += 4){
        for(int i = 0; i < N; i += 4){
            if(j == 0) PackMatrixB(K, &B(0,i), ldb, &packageB[i * K]);
            pacVecAddDot4x4(K, &A(i,0), lda, &packageB[i * K], ldb, &C(i,j), ldc, alpha, beta);
        }
    }
}

void smallBlockWithDoublePackage(int M, int N, int K, const double *A, blasint lda, const double *B, blasint ldb, double *C, blasint ldc, double alpha, double beta)
{   
    // allocate aligned memory
    double* packageB = nullptr;
    if (posix_memalign((void**)&packageB, 32, K * N * sizeof(double)) != 0) {
        throw std::runtime_error("Failed to allocate aligned memory");
    }

    double* packageA = nullptr;
    if (posix_memalign((void**)&packageA, 32, M * K * sizeof(double)) != 0) {
        throw std::runtime_error("Failed to allocate aligned memory");
    }


    for(int j = 0; j < M; j += 4){
        PackMatrixA(K, &A(j, 0), lda, &packageA[j * K]);
        for(int i = 0; i < N; i += 4){
            if(j == 0) PackMatrixB(K, &B(0,i), ldb, &packageB[i * K]);
            doublePacVecAddDot4x4(K, &packageA[j * K], lda, &packageB[i * K], ldb, &C(i,j), ldc, alpha, beta);
        }
    }
}


void PackMatrixB(int K, const double *B, int ldb, double *packageB)
{
  
  for(int j = 0; j < K; j++){  /* loop over columns of A */
    const double *b_ij_ptr = &B(j, 0);

    *packageB = *b_ij_ptr;
    *(packageB + 1) = *(b_ij_ptr + 1);
    *(packageB + 2) = *(b_ij_ptr + 2);
    *(packageB + 3) = *(b_ij_ptr + 3);
    packageB += 4;
  }
}

void PackMatrixA(int K, const double *A, int lda, double *packageA)
{

    const double *a_0i_ptr = &A( 0, 0 );
    const double *a_1i_ptr = &A( 1, 0 );
    const double *a_2i_ptr = &A( 2, 0 );
    const double *a_3i_ptr = &A( 3, 0 );
    for(int i = 0; i < K; i++){
        *packageA++ = *a_0i_ptr++;
        *packageA++ = *a_1i_ptr++;
        *packageA++ = *a_2i_ptr++;
        *packageA++ = *a_3i_ptr++;
    }
}


int main() {

    double* a = (double*)malloc(sizeof(double) * 201 * 201);

    double alpha = 1.0;
    double beta = 0.0;

    double* A = create_random_matrix(200, 200);
    double* B = create_random_matrix(200, 200);
    double* C = create_random_matrix(200, 200);
    double* C2 = create_random_matrix(200, 200);

    cout << "Start running the progress" << endl;

    // test_own_dgemm(200, 200, 200, A, B, C);
    // test_final_dgemm(200, 200, 200, A, B, C2);
    // cout << "own_time / real_time = " << own_time / real_time << endl;
    // cout << endl;

    // delete[] A;
    // delete[] B;
    // delete[] C;
    // delete[] C2;

    // A = create_random_matrix(500, 500);
    // B = create_random_matrix(500, 500);
    // C = create_random_matrix(500, 500);
    // C2 = create_random_matrix(500, 500);



    // test_own_dgemm(500, 500, 500, A, B, C);
    // test_final_dgemm(500, 500, 500, A, B, C2);
    // cout << "own_time / real_time = " << own_time / real_time << endl;
    // cout << endl;

    delete[] A;
    delete[] B;
    delete[] C;
    delete[] C2;

    blasint Int = 900;

    A = create_random_matrix(Int, Int);
    B = create_random_matrix(Int, Int);
    C = create_random_matrix(Int, Int);
    C2 = create_random_matrix(Int, Int);


    test_own_dgemm(Int, Int, Int, A, B, C);
    test_final_dgemm(Int, Int, Int, A, B, C2);

        int count = 0;
        for(int i = 0; i < 900 * 900 / 2; i++){
        if(C[i] - C2[i] > 10){
            cout << "the wrong result of i:" << i << endl;
            cout << "the wrong result is C[i]: " << C[i] << endl;
            cout << "the wrong result is C2[i]: " << C2[i] << endl;
            std::cout << "error happen!" << std::endl;
            count++;
            if(count > 5){
                break;
            }
        }
    }

    cout << "own_time / real_time = " << own_time / real_time << endl;
    cout << endl;

    // delete[] A;
    // delete[] B;
    // delete[] C;
    // delete[] C2;

    // test_own_dgemm(1024, 1024, 1024, &A, &B, &C);
    // test_final_dgemm(1024, 1024, 1024, &A, &B, &C);
    // cout << "own_time / real_time = " << own_time / real_time << endl;
    // cout << endl;

    // blasint TEST = 2048;

    // A = create_random_matrix(TEST, TEST);
    // B = create_random_matrix(TEST, TEST);
    // C = create_random_matrix(TEST, TEST);
    // C2 = create_random_matrix(TEST, TEST);


    // test_own_dgemm(TEST, TEST, TEST, A, B, C);
    // test_final_dgemm(TEST, TEST, TEST, A, B, C2);
    // cout << "own_time / real_time = " << own_time / real_time << endl;
    // cout << endl;


    

    cout << "End running the progress" << endl;
}