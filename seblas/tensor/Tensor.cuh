//
// Created by Dylan on 6/2/2022.
//

#ifndef CRUSADER_TENSOR_CUH
#define CRUSADER_TENSOR_CUH

#include "cuda.h"
#include "cuda_runtime.h"
#include "mma.h"
#include "curand.h"
#include <curand_kernel.h>

#include <string>
#include <cassert>
#include <random>

#include "../../seutil/exec/ThreadController.cuh"
#include "../assist/CudaAssert.cuh"
#include "cudnn.h"

#define CPU_THREADS 20

using namespace std;
using namespace nvcuda;
using namespace seutil;

namespace seblas{

    const dim3 CUDA_BLOCK_SIZE = {16, 16};
    const dim3 CUDA_BLOCK_SIZE_3D = {16, 16, 4};
    typedef unsigned int uint32;
    typedef unsigned long long uint64;
    const uint32 WARP_SIZE = 32;

    /**
     * @Brief index4 is used to navigate within A 4D tensor
     * Tensors are stored in row major order (NCHW)
     *
     * //NHWC is not supported because its A cursed arrangement
     * that should be burned to death in the flame of hell
     *
     * n : the 4th dimension
     * c : channels
     * h : rows (height)
     * w : cols (width)
     */
    struct index4{
        uint32 n=0, c=0, h=0, w=0;
        __device__ __host__ index4(uint32 n, uint32 c, uint32 h, uint32 w) : n(n), c(c), h(h), w(w){}
        __device__ __host__ index4(uint32 c, uint32 h, uint32 w) : c(c), h(h), w(w){}
        __device__ __host__ index4(uint32 h, uint32 w) : h(h), w(w){}

        __device__ __host__ index4 operator+(index4 other) const;
        __device__ __host__ index4 operator-(index4 other) const;
        __device__ __host__ bool operator==(index4 other) const;
        __device__ __host__ bool operator>(index4 other) const;
        __device__ __host__ bool operator<(index4 other) const;

        [[nodiscard]] __device__ __host__ uint32 getOffset() const;
        [[nodiscard]] string toString() const;
    };

    /**
     * shape of tensors
     */
    struct shape4 : public index4{
        uint32 size;
        __device__ __host__ shape4(uint32 n, uint32 c, uint32 h, uint32 w)
                : index4(n, c, h, w){ size = n*c*h*w; }

        __device__ __host__ shape4(uint32 c, uint32 h, uint32 w)
                : index4(1, c, h, w){ size = c*h*w; }

        __device__ __host__ shape4(uint32 h, uint32 w)
                : index4(1, 1, h, w){ size = h*w; }

        string toString() const{
            return "(" + to_string(n) + "," + to_string(c) +
                   "," + to_string(h) + "," + to_string(w) + ")";
        }
    };

    /**
     * Tensor is the base of everything,
     * it records the dimensions and A pointer to data elements
     * Tensor supports FP32 and TF32 as data types
     * device id : -1 for CPU, others for GPU
     */
    class Tensor {
    public:
        shape4 dims;
        float* elements = nullptr;
        int deviceId = 0;
        cudnnTensorDescriptor_t cudnnDesc;

        static Tensor* declare(shape4 dims) {
            Tensor *construct;
            cudaMallocHost(&construct, sizeof(Tensor));
            construct->dims = dims;
    
            cudnnCreateTensorDescriptor(&construct->cudnnDesc);
            construct->setCudnnDesc();
            
            return construct;
        }
    
        template<typename... Args>
        static Tensor* declare(Args &&... args) {
            return declare(shape4(std::forward<Args>(args)...));
        }
        
        static Tensor* create(shape4 dims) {
            return declare(dims)->instantiate();
        }
    
        template<typename... Args>
        static Tensor* create(Args &&... args) {
            return create(shape4(std::forward<Args>(args)...));
        }
        
        static Tensor* createHost(shape4 dims){
            Tensor *construct = declare(dims);
            return construct->instantiateHost();
        }
    
        template<typename... Args>
        static Tensor* createHost(Args... args){
            return createHost(shape4(args...));
        }

        Tensor* reshape(shape4 dim) {
            assert(dim.size == this->dims.size);
            this->dims = dim;
            setCudnnDesc();
            return this;
        }

        template<typename... Args>
        Tensor* reshape(Args &&... args) {
            return reshape(shape4(std::forward<Args>(args)...));
        }

        //instantiate and destruct tensor elements
        Tensor* instantiate();
        Tensor* destroy();
        void eliminate();
        Tensor* instantiateHost();
        Tensor* destroyHost();
        void eliminateHost();

        //transLocate
        //toDevice() and toHost() will migrate the elements
        //the original tensor would be unregistered
        //ripoff() creates an identical tensor on host as it is on device
        Tensor* toDevice();
        Tensor* toHost();

        [[nodiscard]] Tensor* ripOffDevice() const;
        [[nodiscard]] Tensor* ripOffHost() const;
        
        Tensor* ripOffHost(float* src);

        Tensor* copyToH2D(Tensor* onDevice) const;
        Tensor* copyToD2H(Tensor* onHost) const;
        Tensor* copyToD2D(Tensor* onDevice) const;
        Tensor* copyToH2H(Tensor* onHost) const;

        //attaching (Tensors sharing same elements)
        Tensor* attach(Tensor* other);
        Tensor* attach(float* element, int deviceID);

        //common operators
        //[Speed Test Passed][Accuracy Test Passed]
        Tensor* operator+(Tensor* other);
        Tensor* operator+(float other);

        //[Speed Test Passed][Accuracy Test Passed]
        Tensor* operator-(Tensor* other);
        Tensor* operator-(float other);

        //[Speed Test Passed][Accuracy Test Passed]
        Tensor* operator*(Tensor* other);  //hadamard product
        Tensor* operator*(float other);

        //[Speed Test Passed][Accuracy Test Passed]
        Tensor* operator/(Tensor* other);
        Tensor* operator/(float other);

        //initialization
        Tensor* constFill(float val);
        Tensor* randUniform(float min, float max);
        Tensor* randNormal(float mean, float stddev);
        
        //setup cudnn connections
        void setCudnnDesc(){
            cudnnSetTensor4dDescriptor(cudnnDesc,
                                CUDNN_TENSOR_NCHW,
                                CUDNN_DATA_FLOAT,
                                       (int)dims.n,
                                       (int)dims.c,
                                       (int)dims.h,
                                       (int)dims.w);
        }

        //access
        __host__ float get(uint32 offset) const;
    };
}


#endif //CRUSADER_TENSOR_CUH
