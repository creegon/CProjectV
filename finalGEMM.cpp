#include "testGEMM.hpp"

double* create_random_matrix(blasint M, blasint N) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(1.0, 100.0);

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

void own_dgemm(const CBLAS_ORDER Order, const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
                 blasint M, blasint N, blasint K, double alpha,  const double* A, blasint lda,
                 const double* B, blasint ldb, double beta, double* C, blasint ldc)
{
    //空指针检查
    check_nullptr(A, "A");
    check_nullptr(B, "B");
    check_nullptr(C, "C");

    //检查数据合法性
    check_size(M, "M");
    check_size(N, "N");
    check_size(K, "K");
    check_size(lda, "lda");
    check_size(ldb, "ldb");
    check_size(ldc, "ldc");

    //计算前，先把beta那份的C给加了（其实就是让C矩阵每个元素都乘beta）
    double* C_ptr = &C(0,0);
    for (int i = 0; i < M * N; ++i) {
        //让C_ptr指针指向C矩阵的每个元素，然后让C_ptr指向的元素乘beta
        *C_ptr *= beta;
        C_ptr++;
    }

    //提前把矩阵转置给弄了。。好麻烦。。
    transMatrix(Order, TransA, TransB, M, N, K, A, lda, B, ldb);

    //把尺寸啥的换下。。
    if(TransA == CblasTrans){
        blasint temp = M;
        M = K;
        K = temp;
    }

    if(TransB == CblasTrans){
        blasint temp = N;
        N = K;
        K = temp;
    }

    //然后如果尺寸不是4的倍数，则扩充至4的倍数
    expandMatrixA(M, K, A);
    expandMatrixB(K, N, B);
    expandMatrixC(M, N, C);

    //接下来将M,K,N全都更新了
    if(M % 4 != 0){
        M = (M / 4 + 1) * 4;
    }
    if(K % 4 != 0){
        K = (K / 4 + 1) * 4;
    }
    if(N % 4 != 0){
        N = (N / 4 + 1) * 4;
    }

    //更新lda和ldb和ldc
    lda = (Order == CblasRowMajor) ? K : M;
    ldb = (Order == CblasRowMajor) ? N : K;
    ldc = (Order == CblasRowMajor) ? N : M;


        for(int p = 0; p < K; p += kc){
            int pb = min(K - p, kc);
            for (int i = 0; i < M; i += mc){
                int ib = min(M - i, mc);
                smallBlockWithDoublePackage_new(Order, TransA, TransB, ib, N, pb, &A(i, p), lda, &B(p, 0), ldb, &C(i, 0), ldc, alpha, beta);
            }
        }
    
}

void smallBlockWithDoublePackage_new(const CBLAS_ORDER Order, const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, int M, int N, int K, const double *A, blasint lda, const double *B, blasint ldb, double *C, blasint ldc, double alpha, double beta)
{   
    // allocate aligned memory
    double* packageB = nullptr;
    if (posix_memalign((void**)&packageB, 32, K * N * sizeof(double)) != 0) {
        throw std::runtime_error("Failed to allocate aligned memory to B");
    }

    double* packageA = nullptr;
    if (posix_memalign((void**)&packageA, 32, M * K * sizeof(double)) != 0) {
        throw std::runtime_error("Failed to allocate aligned memory to A");
    }


    if(Order == CblasRowMajor){
        for(int j = 0; j < M; j += 4){
            PackMatrixA_new(K, &A(j, 0), lda, &packageA[j * K], Order);
            for(int i = 0; i < N; i += 4){
                if(j == 0) PackMatrixB_new(K, &B(0, i), ldb, &packageB[i * K], Order);
                doublePacVecAddDot4x4_new(K, &packageA[j * K], K, &packageB[i * K], 4, &C( i, j ), ldc, alpha, beta, Order);
            }
        }
    }else{
        for (int j = 0; j < N; j += 4 ){  
            PackMatrixB_new(K, &B( 0, j ), ldb, &packageB[ j * K ], Order);
            for (int i = 0; i < M; i += 4 ){  
                if ( j == 0 ) PackMatrixA_new( K, &A( i, 0 ), lda, &packageA[ i * K ], Order);
                doublePacVecAddDot4x4_new( K, &packageA[ i * K ], 4, &packageB[ j * K ], K, &C( i, j ), ldc, alpha, beta, Order);
            }
        }
    }
    
}

void PackMatrixB_new(int K, const double *B, int ldb, double *packageB, const CBLAS_ORDER Order)
{
    if(Order == CblasRowMajor){
        for(int j = 0; j < K; j++){ 
            const double *b_ij_ptr = &B(j, 0);

            *packageB = *b_ij_ptr;
            *(packageB + 1) = *(b_ij_ptr + 1);
            *(packageB + 2) = *(b_ij_ptr + 2);
            *(packageB + 3) = *(b_ij_ptr + 3);
            packageB += 4;
        }
    }else{
        const double *b_i0_ptr = &B( 0, 0 );
        const double *b_i1_ptr = &B( 0, 1 );
        const double *b_i2_ptr = &B( 0, 2 );
        const double *b_i3_ptr = &B( 0, 3 );
        for(int i = 0; i < K; i++){
            *packageB++ = *b_i0_ptr++;
            *packageB++ = *b_i1_ptr++;
            *packageB++ = *b_i2_ptr++;
            *packageB++ = *b_i3_ptr++;
        }
    }
  

  
}

void PackMatrixA_new(int K, const double *A, int lda, double *packageA, const CBLAS_ORDER Order)
{
    if(Order == CblasRowMajor){
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
    }else{
        for(int j = 0; j < K; j++){ 
            const double *a_ij_ptr = &A(0, j);

            *packageA = *a_ij_ptr;
            *(packageA + 1) = *(a_ij_ptr + 1);
            *(packageA + 2) = *(a_ij_ptr + 2);
            *(packageA + 3) = *(a_ij_ptr + 3);
            packageA += 4;
        }
    }
    
}

void doublePacVecAddDot4x4_new(int K, const double *A, blasint lda, const double *B, blasint ldb, double *C, blasint ldc, double alpha, double beta, const CBLAS_ORDER Order)
{
    
    if(Order == CblasRowMajor){
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
    }else if (Order == CblasColMajor) {
        // pointer variables
        const double* a_p0_ptr = &A(0, 0);
        const double* b_0p_b_1p_ptr = &B(0, 0);

        // Register variables
        v2df_t 
        c_00_c_10_vreg,    c_01_c_11_vreg,    c_02_c_12_vreg,    c_03_c_13_vreg,
        c_20_c_30_vreg,    c_21_c_31_vreg,    c_22_c_32_vreg,    c_23_c_33_vreg,
        a_0p_a_1p_vreg,
        a_2p_a_3p_vreg,
        b_p0_vreg, b_p1_vreg, b_p2_vreg, b_p3_vreg;

        // Initialize vectors with zero
        c_00_c_10_vreg.v = _mm_setzero_pd();   
        c_01_c_11_vreg.v = _mm_setzero_pd();
        c_02_c_12_vreg.v = _mm_setzero_pd(); 
        c_03_c_13_vreg.v = _mm_setzero_pd(); 
        c_20_c_30_vreg.v = _mm_setzero_pd();   
        c_21_c_31_vreg.v = _mm_setzero_pd();  
        c_22_c_32_vreg.v = _mm_setzero_pd();   
        c_23_c_33_vreg.v = _mm_setzero_pd(); 

        // Compute matrix multiplication in vectorized manner
        for (int p = 0; p < K; p++){
            a_0p_a_1p_vreg.v = _mm_load_pd( (double *) a_p0_ptr );
            a_2p_a_3p_vreg.v = _mm_load_pd( (double *) (a_p0_ptr + 2) );
            a_p0_ptr += 4;

            b_p0_vreg.v = _mm_loaddup_pd( (double *) b_0p_b_1p_ptr );  
            b_p1_vreg.v = _mm_loaddup_pd( (double *) (b_0p_b_1p_ptr + 1) );   
            b_p2_vreg.v = _mm_loaddup_pd( (double *) (b_0p_b_1p_ptr + 2) );   
            b_p3_vreg.v = _mm_loaddup_pd( (double *) (b_0p_b_1p_ptr + 3) );
            b_0p_b_1p_ptr += 4;

            c_00_c_10_vreg.v += a_0p_a_1p_vreg.v * b_p0_vreg.v;
            c_01_c_11_vreg.v += a_0p_a_1p_vreg.v * b_p1_vreg.v;
            c_02_c_12_vreg.v += a_0p_a_1p_vreg.v * b_p2_vreg.v;
            c_03_c_13_vreg.v += a_0p_a_1p_vreg.v * b_p3_vreg.v;

            c_20_c_30_vreg.v += a_2p_a_3p_vreg.v * b_p0_vreg.v;
            c_21_c_31_vreg.v += a_2p_a_3p_vreg.v * b_p1_vreg.v;
            c_22_c_32_vreg.v += a_2p_a_3p_vreg.v * b_p2_vreg.v;
            c_23_c_33_vreg.v += a_2p_a_3p_vreg.v * b_p3_vreg.v;
        }

        // Store the results in matrix C
        C( 0, 0 ) += alpha * c_00_c_10_vreg.d[0];
        C( 0, 1 ) += alpha * c_01_c_11_vreg.d[0];  
        C( 0, 2 ) += alpha * c_02_c_12_vreg.d[0];
        C( 0, 3 ) += alpha * c_03_c_13_vreg.d[0]; 

        C( 1, 0 ) += alpha * c_00_c_10_vreg.d[1];
        C( 1, 1 ) += alpha * c_01_c_11_vreg.d[1];  
        C( 1, 2 ) += alpha * c_02_c_12_vreg.d[1];
        C( 1, 3 ) += alpha * c_03_c_13_vreg.d[1]; 

        C( 2, 0 ) += alpha * c_20_c_30_vreg.d[0];
        C( 2, 1 ) += alpha * c_21_c_31_vreg.d[0];  
        C( 2, 2 ) += alpha * c_22_c_32_vreg.d[0];
        C( 2, 3 ) += alpha * c_23_c_33_vreg.d[0]; 

        C( 3, 0 ) += alpha * c_20_c_30_vreg.d[1];
        C( 3, 1 ) += alpha * c_21_c_31_vreg.d[1];  
        C( 3, 2 ) += alpha * c_22_c_32_vreg.d[1];
        C( 3, 3 ) += alpha * c_23_c_33_vreg.d[1]; 
}
  
  
}


void check_nullptr(const double* ptr, const char* name) 
{
    if(ptr == nullptr) {
        throw std::runtime_error(std::string(name) + " is a nullptr");
    }
}

void check_size(blasint size, const char* name) 
{
    if(size <= 0) {
        throw std::runtime_error(std::string(name) + " is less than or equal to 0");
    }
}



void transMatrix(const CBLAS_ORDER Order, const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
                 blasint M, blasint N, blasint K, const double*& A, blasint lda, const double*& B, blasint ldb)
{
    //检查转置后是否还能匹配
    blasint A_M = M;
    blasint A_N = K;
    blasint B_M = K;
    blasint B_N = N;
    if(TransA == CblasTrans) {
        A_M = K;
        A_N = M;
    }
    if(TransA == CblasTrans){
        B_M = N;
        B_N = K;
    }

    //当A矩阵的列数不等于B矩阵的行数时，无法相乘直接返回
    if(A_N != B_M){
         throw std::runtime_error("columns of A is not equal to rows of B, can not multiply");
    }

    //当A矩阵的行数不等于M矩阵的行数，或者B矩阵的列数不等于N矩阵的列数时，无法相加直接返回
    if(A_M != M || B_N != N){
         throw std::runtime_error("rows of A is not equal to rows of C, or columns of B is not equal to columns of C, can not add");
    }


        if(TransA == CblasTrans) {
            double* A_ptr = const_cast<double*>(A);
            if(posix_memalign((void**)&A, 32, M * K * sizeof(double)) != 0){
                throw std::runtime_error("Failed to allocate aligned memory");
            }
            double* temp_ptr = const_cast<double*>(A); //创建一个临时指针来填充数据
            for(int i = 0; i < M; i++) {
                for(int j = 0; j < K; j++) {
                    *temp_ptr++ = A_ptr[j * lda + i];
                }
            }
            
            //最后释放掉原本的A矩阵
            free(A_ptr);
        }
        if(TransB == CblasTrans){
            double* B_ptr = const_cast<double*>(B);
            if(posix_memalign((void**)&B, 32, K * N * sizeof(double)) != 0){
                throw std::runtime_error("Failed to allocate aligned memory");
            }
            
            double* temp_ptr = const_cast<double*>(B); //创建一个临时指针来填充数据
            for(int i = 0; i < K; i++) {
                for(int j = 0; j < N; j++) {
                    *temp_ptr++ = B_ptr[j * ldb + i];
                }
            }

            //最后释放掉原本的B矩阵
            free(B_ptr);
        }
}

void expandMatrixA(blasint M, blasint K, blasint lda, const double*& A)
{
    if(M % 4 == 0 && K % 4 == 0){
        return;
    }
    //原来的指针
    double* old_ptr = const_cast<double*>(A);

    int new_col = (K % 4 == 0) ? K : (K / 4 + 1) * 4;
    int new_row = (M % 4 == 0) ? M : (M / 4 + 1) * 4;

    //创建新的指针
    double* new_ptr;
    if(posix_memalign((void**)&new_ptr, 32, new_col * new_row * sizeof(double)) != 0){
        throw std::runtime_error("Failed to allocate aligned memory");
    }

    //遍历并复制数据
    for(int i = 0; i < new_row; i++) {
        for(int j = 0; j < new_col; j++) {
            if(i < M && j < K){
                new_ptr(i, j) = old_ptr(i, j);
            }else{
                new_ptr(i, j) = 0;
            }
            //打印temp_ptr2的值
            // std::cout << new_ptr[i * new_col + j];
        }
        // cout << endl;
    }

    //释放原来的内存
    free(old_ptr);

    //更新ptr的值
    A = new_ptr;
}

void expandMatrixB(blasint K, blasint N, blasint ldb, const double*& B)
{
    if(K % 4 == 0 && N % 4 == 0){
        return;
    }
    //原来的指针
    double* old_ptr = const_cast<double*>(B);

    int new_col = (N % 4 == 0) ? N : (N / 4 + 1) * 4;
    int new_row = (K % 4 == 0) ? K : (K / 4 + 1) * 4;

    //创建新的指针
    double* new_ptr;
    if(posix_memalign((void**)&new_ptr, 32, new_col * new_row * sizeof(double)) != 0){
        throw std::runtime_error("Failed to allocate aligned memory");
    }

    //遍历并复制数据
    for(int i = 0; i < new_row; i++) {
        for(int j = 0; j < new_col; j++) {
            if(i < K && j < N){
                new_ptr(i, j) = old_ptr(i, j);
            }else{
                new_ptr(i, j) = 0;
            }
            //打印temp_ptr2的值
            // std::cout << new_ptr[i * new_col + j];
        }
        // cout << endl;
    }

    //释放原来的内存
    free(old_ptr);

    //更新ptr的值
    B = new_ptr;
}

void expandMatrixC(blasint M, blasint N, blasint ldc, const double*& C)
{
    if(M % 4 == 0 && N % 4 == 0){
        return;
    }
    //原来的指针
    double* old_ptr = const_cast<double*>(C);

    int new_col = (N % 4 == 0) ? N : (N / 4 + 1) * 4;
    int new_row = (M % 4 == 0) ? M : (M / 4 + 1) * 4;

    if(posix_memalign((void**)&C, 32, new_col * new_row * sizeof(double)) != 0){
        throw std::runtime_error("Failed to allocate aligned memory");
    }

    //遍历并复制数据
    for(int i = 0; i < new_row; i++) {
        for(int j = 0; j < new_col; j++) {
            if(i < M && j < N){
                new_ptr(i, j) = old_ptr(i, j);
            }else{
                new_ptr(i, j) = 0;
            }
        }
    }

    //释放原来的内存
    free(old_ptr);
}


int main()
{
    double alpha = 1.0;
    double beta = 0.0;
    const double MAX_INTERVE = 1e-6;

    blasint M = 4;
    blasint N = 4;
    blasint K = 4;

    CBLAS_ORDER order = CblasColMajor;
    CBLAS_TRANSPOSE transA = CblasNoTrans;
    CBLAS_TRANSPOSE transB = CblasNoTrans;


    double* A = create_random_matrix(M, K);
    double* B = create_random_matrix(K, N);
    double* C = create_random_matrix(M, N);
    double* C2 = create_random_matrix(M, N);

    blasint lda = (order == CblasColMajor) ? M : K;
    blasint ldb = (order == CblasColMajor) ? K : N;
    blasint ldc = (order == CblasColMajor) ? M : N;

    cout << "Start running the progress" << endl;

    //打印矩阵信息（完整的）
    cout << "row of A is : " << M << endl;
    cout << "col of A is : " << K << endl;
    cout << "row of B is : " << K << endl;
    cout << "col of B is : " << N << endl;
    cout << "order is : " << ((order == 101) ? "RowMajor" : "ColMajor") << ". transA is : " << ((transA == 111) ? "noTrans" : "haveTrans") << ". transB is : " << ((transB == 111)  ? "noTrans" : "haveTrans") << endl;

    own_dgemm(order, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);

    cblas_dgemm(order, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C2, ldc);

    //检验计算结果的准确性
    bool noError = true;
    for(int i = 0; i < M * N; i++){
        if(C[i] - C2[i] > MAX_INTERVE || C[i] - C2[i] < -MAX_INTERVE){
            cout << "the wrong result is C[i]: " << C[i] << endl;
            cout << "the wrong result is C2[i]: " << C2[i] << endl;
            std::cout << "error happen!" << std::endl;
            noError = false;
            break;
        }
    }

    if(noError){
        cout << "no error!" << endl;
    }else{
        //打印C的前十个元素
        cout << "first ten elements of C is: " << endl;
        for(int i = 0; i < 16; i++){
            cout << C[i] << endl;
        }
        cout << endl;

        //打印C2的前十个元素
        cout << "first ten elements of C2 is: " << endl;
        for(int i = 0; i < 16; i++){
            cout << C2[i] << endl;
        }
        cout << endl;
    }

    cout << "End running the progress" << endl;
}