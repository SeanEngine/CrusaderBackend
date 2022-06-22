//
// Created by Dylan on 6/3/2022.
//

#ifndef CRUSADER_CUGEMM_CUH
#define CRUSADER_CUGEMM_CUH

#include "../tensor/Tensor.cuh"

namespace seblas {
    // normal implementation of FP32 GEMM (optimized to 92% cublas)
    Tensor* sgemm(Tensor *A, Tensor *B, Tensor *C);
    
    // gemm with first matrix transposed
    Tensor* sgemmTN(Tensor *A, Tensor *B, Tensor *C);
    
    // gemm with second matrix transposed
    Tensor* sgemmNT(Tensor *A, Tensor *B, Tensor *C);
    
    // gemm with second matrices transposed and addition
    Tensor* sgemmNTA(Tensor *A, Tensor *B, Tensor *C);
} // seblas

#endif //CRUSADER_CUGEMM_CUH
