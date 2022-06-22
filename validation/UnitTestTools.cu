//
// Created by Dylan on 6/2/2022.
//

#include "UnitTestTools.cuh"
#include "../seblas/operations/cuOperations.cuh"
#include "cudnn.h"
#include <fstream>
#define BASE_PATH string("D:\\Projects\\PyCharm\\UnitTestTool\\TestContents\\")

#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

__global__ void simpleGemmD(Tensor* A, Tensor* B, Tensor* C){
    uint32 x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32 y = blockIdx.y * blockDim.y + threadIdx.y;
    
    const uint32 M = A->dims.h;
    const uint32 N = B->dims.w;
    const uint32 K = A->dims.w;
    
    if(y < M && x < N){
        float sum = 0;
        #pragma unroll
        for(uint32 k = 0; k < K; k++){
            sum += A->elements[y * K + k] * B->elements[k * N + x];
        }
        C->elements[y * N + x] = sum;
    }
}

Tensor* simpleGemm(Tensor* A, Tensor* B, Tensor* C){
    dim3 block = CUDA_BLOCK_SIZE;
    dim3 grid = {
            (B->dims.w + block.x - 1) / block.x,
            (A->dims.h + block.y - 1) / block.y
    };
    
    simpleGemmD<<<grid, block>>>(A, B, C);
    assertCuda(__FILE__, __LINE__);
    return C;
}

void appendTensorToString(string* input, Tensor* target) {
    Tensor* proc;
    if(target->deviceId == -1){
        proc = target;
    }else{
        proc = target->ripOffDevice();
    }

    for(int n = 0; n < proc->dims.n; n++){
        for(int c = 0; c < proc->dims.c; c++){
            for(int h = 0; h < proc->dims.h; h++){
                for(int w = 0; w < proc->dims.w; w++){
                    input->append(to_string(proc->elements[
                            n * target->dims.c * target->dims.h * target->dims.w +
                            c * target->dims.h * target->dims.w +
                            h * target->dims.w +
                            w
                    ]));
                    input->append(", ");
                }
            }
        }
    }
    input->append("\n||\n");

    if(target->deviceId != -1){
        proc->eliminateHost();
    }
    assertCuda(__FILE__, __LINE__);
}

void stringToFile(string* input, const string& fileName) {
    ofstream file;
    file.open(fileName);
    file << *input;
    file.close();
}

void normalUnitTest(){
    //2 multiples
    string XBuf;
    string YBuf;
    string ResultBuf;

    //2 multiples
    for(int i = 2; i < 1025; i+=2){
        Tensor* X = Tensor::declare(i, 1)->instantiate()->randNormal(0, 100);
        Tensor* Y = Tensor::declare(i, 1)->instantiate()->randNormal(0, 100);
        appendTensorToString(&XBuf, X);
        appendTensorToString(&YBuf, Y);
        *X * Y;
        appendTensorToString(&ResultBuf, X);
    }

    //3 multiples
    for(int i = 3; i < 1025; i+=3){
        Tensor* X = Tensor::declare(i, 1)->instantiate()->randNormal(0, 100);
        Tensor* Y = Tensor::declare(i, 1)->instantiate()->randNormal(0, 100);
        appendTensorToString(&XBuf, X);
        appendTensorToString(&YBuf, Y);
        *X * Y;
        appendTensorToString(&ResultBuf, X);
    }

    //5 multiples
    for(int i = 5; i < 1025; i+=5){
        Tensor* X = Tensor::declare(i, 1)->instantiate()->randNormal(0, 100);
        Tensor* Y = Tensor::declare(i, 1)->instantiate()->randNormal(0, 100);
        appendTensorToString(&XBuf, X);
        appendTensorToString(&YBuf, Y);
        *X * Y;
        appendTensorToString(&ResultBuf, X);
    }

    stringToFile(&XBuf,BASE_PATH + "A.txt");
    stringToFile(&YBuf, BASE_PATH + "Y.txt");
    stringToFile(&ResultBuf, BASE_PATH + "Result.txt");
}

void linearUnitTest(){
    string XBuf;
    string YBuf;
    string ResultBuf;
    for(uint32 inSize = 2; inSize < 2048; inSize *=2){
        for(uint32 outSize = 2; outSize < 2048; outSize *= 2){
            for(uint32 nDim = 1; nDim < 256; nDim *=2){
                Tensor* dW = Tensor::declare(outSize, inSize)->instantiate();
                Tensor* dB = Tensor::declare(outSize, 1)->instantiate();
                Tensor* X = Tensor::declare(nDim, 1, inSize, 1)->instantiate()->randNormal(0, 1);
                Tensor* dY = Tensor::declare(nDim, 1, outSize, 1)->instantiate()->randNormal(0, 1);
                Tensor* dWprime = Tensor::declare(outSize, inSize)->instantiate();
    
                linearParamGrad(dY, X, dW, dB);
                sgemmNTA(dY, X, dWprime);
                
                Tensor* tmp1 = dW->ripOffDevice();
                Tensor* tmp2 = dWprime->ripOffDevice();
    
                cout<<"Stage: "<< inSize << " " << outSize << " " << nDim << endl;
                
                for (uint32 i = 0; i < tmp1->dims.size; i++){
                    if(abs(tmp1->elements[i] - tmp2->elements[i]) > 0.001){
                        cout << "Error: " << tmp1->elements[i] << " != " << tmp2->elements[i] <<" "<<i<< endl;
                        inspect(tmp1);
                        inspect(tmp2);
                        exit(1);
                    }
                }
                
                tmp1->eliminateHost();
                tmp2->eliminateHost();
                dW->eliminate();
                X->eliminate();
                dY->eliminate();
                dB->eliminate();
                dWprime->eliminate();
            }
        }
    }
}

void convUnitTest(){
    string FBuf;
    string IBuf;
    string ResultBuf;
    int stride = 1;
    
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);
    
    for(uint32 featureDim = 4; featureDim < 65; featureDim *=2){
        for(uint32 filterDim = 1; filterDim < featureDim; filterDim *=2){
            for(uint32 c1Dim = 1; c1Dim < 129; c1Dim *=2){
                for(uint32 c2Dim = 1; c2Dim < 129; c2Dim *=2){
                    for (uint32 nDim = 1; nDim < 256; nDim *=2){
                        cout<<"Stage began: "<< featureDim << " " << filterDim << " " << c1Dim << " " << c2Dim << " " <<nDim<<endl;
    
                        shape4 inDim = {nDim, c1Dim, featureDim, featureDim};
                        shape4 filterDims = {c2Dim, c1Dim, filterDim, filterDim};
                        
                        shape4 outDims = {nDim, c2Dim,
                                          (featureDim - filterDim)/stride + 1,
                                          (featureDim - filterDim)/stride + 1};
                        Tensor* dX = Tensor::declare(inDim)->instantiate()->randNormal(1, 20);
                        Tensor* dW = Tensor::declare(filterDims)->instantiate();
                        Tensor* dY = Tensor::declare(outDims)->instantiate()->randNormal(1, 20);
                        Tensor* dWPrime = Tensor::declare(filterDims)->instantiate();
                        
                        //cudnn validation:
                        cudnnTensorDescriptor_t input_descriptor;
                        checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor))
                        checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                                              CUDNN_TENSOR_NCHW,
                                                              CUDNN_DATA_FLOAT,
                                                              nDim,
                                                              c1Dim,
                                                              featureDim,
                                                              featureDim))
    
                        cudnnTensorDescriptor_t output_descriptor;
                        checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor))
                        checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                                /*format=*/CUDNN_TENSOR_NCHW,
                                /*dataType=*/CUDNN_DATA_FLOAT,
                                /*batch_size=*/nDim,
                                /*channels=*/c2Dim,
                                /*image_height=*/dY->dims.h,
                                /*image_width=*/dY->dims.w))
    
                        cudnnFilterDescriptor_t kernel_descriptor;
                        checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
                        checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
                                /*dataType=*/CUDNN_DATA_FLOAT,
                                /*format=*/CUDNN_TENSOR_NCHW,
                                /*out_channels=*/c2Dim,
                                /*in_channels=*/c1Dim,
                                /*kernel_height=*/filterDim,
                                /*kernel_width=*/filterDim));

                        cudnnConvolutionDescriptor_t convolution_descriptor;

                        checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
                        checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                /*pad_height=*/0,
                                /*pad_width=*/0,
                                /*vertical_stride=*/1,
                                /*horizontal_stride=*/1,
                                /*dilation_height=*/1,
                                /*dilation_width=*/1,
                                /*mode=*/CUDNN_CROSS_CORRELATION,
                                /*computeType=*/CUDNN_DATA_FLOAT));

                        const float alpha = 1, beta = 0;
                        cudnnConvolutionBackwardFilter(
                                cudnn,
                                &alpha,
                                input_descriptor,
                                dX->elements,
                                output_descriptor,
                                dY->elements,
                                convolution_descriptor,
                                CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
                                nullptr,
                                0,
                                &beta,
                                kernel_descriptor,
                                dWPrime->elements
                                );
                        
                        convError(dY, dX, dW, stride, stride, 0, 0);
                        
                        Tensor* tmp1 = dW->ripOffDevice();
                        Tensor* tmp2 = dWPrime->ripOffDevice();
    
                        for (uint32 i = 0; i < tmp1->dims.size; i++) {
                            if (abs(tmp1->elements[i] - tmp2->elements[i])/abs(tmp1->elements[i]) > 0.01
                            && abs(tmp1->elements[i] - tmp2->elements[i]) > 1) {
                                cout << "Error: " << tmp1->elements[i] << " != " << tmp2->elements[i] << " " << i
                                     << endl;
                                cout<<"=============="<<endl;
                                //inspect(tmp1);
                                cout<<"=============="<<endl;
                                //inspect(tmp2);
                                exit(1);
                            }
                        }
                        
                        dX->eliminate();
                        dW->eliminate();
                        dY->eliminate();
                        dWPrime->eliminate();
                        
                        tmp1->eliminateHost();
                        tmp2->eliminateHost();
                        
                        cudnnDestroyTensorDescriptor(input_descriptor);
                        cudnnDestroyTensorDescriptor(output_descriptor);
                        cudnnDestroyFilterDescriptor(kernel_descriptor);
                        cudnnDestroyConvolutionDescriptor(convolution_descriptor);
                        cout<<"Stage: "<< featureDim << " " << filterDim << " " << c1Dim << " " << c2Dim << " " <<nDim<<endl;
                    }
                }
            }
        }
    }
    
    stringToFile(&FBuf,BASE_PATH + "A.txt");
    stringToFile(&IBuf, BASE_PATH + "Y.txt");
    stringToFile(&ResultBuf, BASE_PATH + "Result.txt");
}

void optimUnitTest(){
    string XBuf;
    string XGradBuf;
    string ResultBuf;
    float LEARNING_RATE = 0.01f;
    float BETA1 = 0.9;
    float BETA2 = 0.999;
    for(int i = 1; i < 512; i++){
        Tensor* X = Tensor::declare(i, 1)->instantiate()->randNormal(1, 20);
        Tensor* M = Tensor::declare(i, 1)->instantiate();
        Tensor* V = Tensor::declare(i, 1)->instantiate();
        for(int seq = 0; seq < 21; seq++){
            Tensor* Xgrad = Tensor::declare(i, 1)->instantiate()->randNormal(1, 20);
            appendTensorToString(&XBuf, X);
            appendTensorToString(&XGradBuf, Xgrad);
            
            adamApply(X, Xgrad, M, V, LEARNING_RATE, BETA1, BETA2, 1e-10, seq);
    
            appendTensorToString(&ResultBuf, X);
            Xgrad->eliminate();
        }
        cout<<"Stage: "<<i<<endl;
        X->eliminate();
        M->eliminate();
    }
    stringToFile(&XBuf, BASE_PATH + "A.txt");
    stringToFile(&XGradBuf, BASE_PATH + "XGrad.txt");
    stringToFile(&ResultBuf, BASE_PATH + "Result.txt");
}

void batchnormUnitTest(){
    
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);
    
    for(int i = 2; i < 2048; i++){
        for(int n = 2; n < 96; n++){
            cout<<"Stage: "<<i<<" "<<n<<endl;
            Tensor* X = Tensor::create(n, 1, i, 1)->randNormal(1, 20);
            Tensor* DY = Tensor::create(n, 1, i, 1)->randNormal(1, 50);
            Tensor* DX = Tensor::create(n, 1, i, 1);
            Tensor* DXPrime = Tensor::create(n, 1, i, 1);
            
            Tensor* mean = Tensor::create(1, 1, i, 1);
            Tensor* var = Tensor::create(1, 1, i, 1);
            
            Tensor* dBeta = Tensor::create(1, 1, i, 1);
            Tensor* dGamma = Tensor::create(1, 1, i, 1);
            Tensor* dGammaPrime = Tensor::create(1, 1, i, 1);
            Tensor* dBetaPrime = Tensor::create(1, 1, i, 1);
            
            Tensor* gamma = Tensor::create(1, 1, i, 1)->randNormal(1, 20);
            Tensor* beta = Tensor::create(1, 1, i, 1)->randNormal(1, 20);
            
            //instantiate description
            cudnnTensorDescriptor_t input_descriptor;
            checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor))
            checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                                  CUDNN_TENSOR_NCHW,
                                                  CUDNN_DATA_FLOAT,
                                                  n,
                                                  1,
                                                  i,
                                                  1))
    
            cudnnTensorDescriptor_t param_descriptor;
            checkCUDNN(cudnnCreateTensorDescriptor(&param_descriptor))
            checkCUDNN(cudnnSetTensor4dDescriptor(param_descriptor,
                                                  CUDNN_TENSOR_NCHW,
                                                  CUDNN_DATA_FLOAT,
                                                  1,
                                                  1,
                                                  i,
                                                  1))
    
            const float a = 1, b = 0;
            checkCUDNN(cudnnBatchNormalizationBackward(cudnn,
                                            CUDNN_BATCHNORM_PER_ACTIVATION,
                                            &a,
                                            &b,
                                            &a,
                                            &b,
                                            input_descriptor,
                                            X->elements,
                                            input_descriptor,
                                            DY->elements,
                                            input_descriptor,
                                            DX->elements,
                                            param_descriptor,
                                            gamma->elements,
                                            dGamma->elements,
                                            dBeta->elements,
                                            1e-10,
                                            nullptr,
                                            nullptr
                                            ))
                                            
            batchNormParamGrads(DY, dGammaPrime, dBetaPrime, X);
            
            Tensor* tmp1 = dBeta->ripOffDevice();
            Tensor* tmp2 = dBetaPrime->ripOffDevice();
    
            //inspect(tmp1);
            //inspect(tmp2);
    
            for (uint32 operate = 0; operate < tmp1->dims.size; operate++) {
                if (abs(tmp1->elements[operate] - tmp2->elements[operate]) > 0.1) {
                    cout << "Error: " << tmp1->elements[operate] <<" "<< tmp2->elements[operate] << " " << operate
                         << endl;
                    if(n > 2) cout<<"========"<<endl;
                    //cout<<"=============="<<endl;
                    //inspect(tmp1);
                    //cout<<"=============="<<endl;
                    //inspect(tmp2);
                    //exit(1);
                }
            }
            
            tmp1->eliminateHost();
            tmp2->eliminateHost();
            
            X->eliminate();
            DY->eliminate();
            DX->eliminate();
            mean->eliminate();
            var->eliminate();
            dGamma->eliminate();
            dBeta->eliminate();
            gamma->eliminate();
            beta->eliminate();
        }
    }
}

void softmaxUnitTest(){
    
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);
    
    for(int i = 2; i < 513; i++){
        for(int n = 1; n < 96; n++){
            Tensor* input = Tensor::create(n, 1, i, 1)->randNormal(1, 20);
            Tensor* output = Tensor::create(n, 1, i, 1);
            Tensor* outputPrime = Tensor::create(n, 1, i, 1);
    
            //instantiate description
            cudnnTensorDescriptor_t input_descriptor;
            checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor))
            checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                                  CUDNN_TENSOR_NCHW,
                                                  CUDNN_DATA_FLOAT,
                                                  n,
                                                  1,
                                                  i,
                                                  1))
    
            float a = 1, b = 0;
            cudnnSoftmaxForward(cudnn,
                                CUDNN_SOFTMAX_ACCURATE,
                                CUDNN_SOFTMAX_MODE_INSTANCE,
                                &a,
                                input_descriptor,
                                input->elements,
                                &b,
                                input_descriptor,
                                output->elements);
    
            softmax(input, outputPrime, nullptr, i);
    
            Tensor* tmp1 = output->ripOffDevice();
            Tensor* tmp2 = outputPrime->ripOffDevice();
    
            //inspect(tmp1);
            //inspect(tmp2);
    
            for (uint32 operate = 0; operate < tmp1->dims.size; operate++) {
                if (abs(tmp1->elements[operate] - tmp2->elements[operate]) > 0.01) {
                    cout << "Error: " << tmp1->elements[operate] <<" "<< tmp2->elements[operate] << " " << operate
                         << endl;
                    if(n > 2) cout<<"========"<<endl;
                    //cout<<"=============="<<endl;
                    //inspect(tmp1);
                    //cout<<"=============="<<endl;
                    //inspect(tmp2);
                    //exit(1);
                }
            }
            
            tmp1->eliminateHost();
            tmp2->eliminateHost();
            output->eliminate();
            outputPrime->eliminate();
        }
        cout<<"Stage: "<<i << endl;
    }
};