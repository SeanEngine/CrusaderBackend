//
// Created by Dylan on 6/14/2022.
//

#ifndef CRUSADER_CUCONV_CUH
#define CRUSADER_CUCONV_CUH

#include "../tensor/Tensor.cuh"


namespace seblas {
    /**
     * Convolution operator by GEMM (will implement FFT and vinograd in future)
     * @param A filters OC * IC * FH * FW
     * @param B feature inputs N * IC * IH * IW
     * @param C feature outputs N ＊ OC * OH * OW
     * @param biases set this to be nullpointer if you don't want biases (OC * 1)
     * @return
     */
    Tensor *conv(Tensor *A, Tensor *B, Tensor *C, int strideH, int strideW, int padH, int padW, Tensor *biases);
    
    /**
     * The back propagation of conv layers with respect to input features
     * @param A filters OC * IC * FH * FW
     * @param B feature outputs N * OC * OH * OW
     * @param C feature inputs N * IC * IH * IW
     * @return
     */
    Tensor *convDerive(Tensor *A, Tensor *B, Tensor *C, int strideH, int strideW, int padH, int padW);
    
    /**
     * Calculate gradients of filters (weights) based on errors of the output features
     * This method will loop over all elements on the "ON" dimension and add the errors up
     * the final deltas will be the sum of errors on ON dimension divide by ON
     * @param A errors in : ON * OC * OH * OW
     * @param B input features : ON * IC * IH * IW
     * @param C filters : OC * IC * FH * FW
     * @return
     */
    Tensor* convError(Tensor *A, Tensor *B, Tensor *C, int strideH, int strideW, int padH, int padW);
    
    Tensor* convBias(Tensor* Y, Tensor* biases);
} // seblas

#endif //CRUSADER_CUCONV_CUH
