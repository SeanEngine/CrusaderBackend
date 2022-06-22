//
// Created by Dylan on 6/2/2022.
//

#include "Tensor.cuh"
#define toFloat4R(ptr) (reinterpret_cast<float4*>(&(ptr))[0])
#define CUDA_BLOCK_SIZE_1D CUDA_BLOCK_SIZE.x * CUDA_BLOCK_SIZE.y
#define topOff(a,b) ((a) + (b) - 1)/(b)


namespace seblas {

    //CPU operators
    void addTensorCPU(const int tid, const int ts, Tensor* A, Tensor* B){
        int start = tid * (int)(A->dims.size/ts);
        int end = tid == ts-1 ? (int)A->dims.size : start + (int)A->dims.size/ts;
        for(int i = start; i < end; i++){
            A->elements[i] += B->elements[i];
        }
    }

    void minusTensorCPU(const int tid, const int ts, Tensor* A, Tensor* B){
        int start = tid * (int)(A->dims.size/ts);
        int end = tid == ts-1 ? (int)A->dims.size : start + (int)A->dims.size/ts;
        for(int i = start; i < end; i++){
            A->elements[i] -= B->elements[i];
        }
    }

    void hadamardTensorCPU(const int tid, const int ts, Tensor* A, Tensor* B){
        int start = tid * (int)(A->dims.size/ts);
        int end = tid == ts-1 ? (int)A->dims.size : start + (int)A->dims.size/ts;
        for(int i = start; i < end; i++){
            A->elements[i] *= B->elements[i];
        }
    }

    void divideTensorCPU(const int tid, const int ts, Tensor* A, Tensor* B){
        int start = tid * (int)(A->dims.size/ts);
        int end = tid == ts-1 ? (int)A->dims.size : start + (int)A->dims.size/ts;
        for(int i = start; i < end; i++){
            A->elements[i] /= B->elements[i];
        }
    }

    void addScalarCPU(const int tid, const int ts, Tensor* A, float b){
        int start = tid * (int)(A->dims.size/ts);
        int end = tid == ts-1 ? (int)A->dims.size : start + (int)A->dims.size/ts;
        for(int i = start; i < end; i++){
            A->elements[i] += b;
        }
    }

    void minusScalarCPU(const int tid, const int ts, Tensor* A, float b){
        int start = tid * (int)(A->dims.size/ts);
        int end = tid == ts-1 ? (int)A->dims.size : start + (int)A->dims.size/ts;
        for(int i = start; i < end; i++){
            A->elements[i] -= b;
        }
    }

    void hadamardScalarCPU(const int tid, const int ts, Tensor* A, float b){
        int start = tid * (int)(A->dims.size/ts);
        int end = tid == ts-1 ? (int)A->dims.size : start + (int)A->dims.size/ts;
        for(int i = start; i < end; i++){
            A->elements[i] *= b;
        }
    }

    void divideScalarCPU(const int tid, const int ts, Tensor* A, float b){
        int start = tid * (int)(A->dims.size/ts);
        int end = tid == ts-1 ? (int)A->dims.size : start + (int)A->dims.size/ts;
        for(int i = start; i < end; i++){
            A->elements[i] /= b;
        }
    }

    void constFillCPU(const int tid, const int ts, Tensor* A, float value){
        int start = tid * (int)(A->dims.size/ts);
        int end = tid == ts-1 ? (int)A->dims.size : start + (int)A->dims.size/ts;
        for(int i = start; i < end; i++){
            A->elements[i] = value;
        }
    }

    void randUniformCPU(const int tid, const int ts, Tensor* A, float min, float max){
        int start = tid * (int)(A->dims.size/ts);
        int end = tid == ts-1 ? (int)A->dims.size : start + (int)A->dims.size/ts;
        uint32 seed = chrono::steady_clock::now().time_since_epoch().count();
        default_random_engine e(seed);
        uniform_real_distribution<float> distr(min, max);
        for(int i = start; i < end; i++){
            srand(time(nullptr) * i);
            A->elements[i] = distr(e);
        }
    }

    void randNormalCPU(const int tid, const int ts, Tensor* A, float mean, float stddev){
        int start = tid * (int)(A->dims.size/ts);
        int end = tid == ts-1 ? (int)A->dims.size : start + (int)A->dims.size/ts;
        uint32 seed = chrono::steady_clock::now().time_since_epoch().count();
        default_random_engine e(seed);
        normal_distribution<float> distr(mean, stddev);
        for(int i = start; i < end; i++){
            srand(time(nullptr) * i);
            A->elements[i] = distr(e);
        }
    }

    //GPU operators
    __global__ void addD(Tensor* in, Tensor* other){
        uint32 index = threadIdx.x + blockIdx.x * blockDim.x;
        if(index < in->dims.size){
            in->elements[index] += other->elements[index];
        }
    }

    __global__ void add4D(Tensor* in, Tensor* other){
        uint32 index = (threadIdx.x + blockIdx.x * blockDim.x )* 4;
        float regisA[4];
        float regisB[4];
        float regisC[4] = {0};
        if(index < in->dims.size){
            toFloat4R(regisA[0]) = toFloat4R(in->elements[index]);
            toFloat4R(regisB[0]) = toFloat4R(other->elements[index]);
            #pragma unroll
            for (int i = 0; i < 4; i++){
                regisC[i] = regisA[i] + regisB[i];
            }
            toFloat4R(in->elements[index]) = toFloat4R(regisC[0]);
        }
    }

    __global__ void subtractD(Tensor* A, Tensor* B){
        uint32 index = threadIdx.x + blockIdx.x * blockDim.x;
        if(index < A->dims.size){
            A->elements[index] -= B->elements[index];
        }
    }

    __global__ void subtract4D(Tensor* A, Tensor* B){
        uint32 index = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
        float regisA[4];
        float regisB[4];
        float regisC[4] = {0};
        if(index < A->dims.size){
            toFloat4R(regisA[0]) = toFloat4R(A->elements[index]);
            toFloat4R(regisB[0]) = toFloat4R(B->elements[index]);
            regisC[0] = regisA[0] - regisB[0];
            regisC[1] = regisA[1] - regisB[1];
            regisC[2] = regisA[2] - regisB[2];
            regisC[3] = regisA[3] - regisB[3];
            toFloat4R(A->elements[index]) = toFloat4R(regisC[0]);
        }
    }

    __global__ void hadamardProductD(Tensor* in, Tensor* other){
        uint32 index = threadIdx.x + blockIdx.x * blockDim.x;
        if(index < in->dims.size){
            in->elements[index] *= other->elements[index];
        }
    }

    __global__ void hadamardProduct4D(Tensor* in, Tensor* other){
        uint32 index = (threadIdx.x + blockIdx.x * blockDim.x )* 4;
        float regisA[4];
        float regisB[4];
        float regisC[4] = {0};
        if(index < in->dims.size){
            toFloat4R(regisA[0]) = toFloat4R(in->elements[index]);
            toFloat4R(regisB[0]) = toFloat4R(other->elements[index]);
            regisC[0] = regisA[0] * regisB[0];
            regisC[1] = regisA[1] * regisB[1];
            regisC[2] = regisA[2] * regisB[2];
            regisC[3] = regisA[3] * regisB[3];
            toFloat4R(in->elements[index]) = toFloat4R(regisC[0]);
        }
    }

    __global__ void divideD(Tensor* in, Tensor* other){
        uint32 index = threadIdx.x + blockIdx.x * blockDim.x;
        if(index < in->dims.size){
            in->elements[index] /= other->elements[index];
        }
    }

    __global__ void divide4D(Tensor* in, Tensor* other){
        uint32 index = (threadIdx.x + blockIdx.x * blockDim.x )* 4;
        float regisA[4];
        float regisB[4];
        float regisC[4] = {0};
        if(index < in->dims.size){
            toFloat4R(regisA[0]) = toFloat4R(in->elements[index]);
            toFloat4R(regisB[0]) = toFloat4R(other->elements[index]);
            regisC[0] = regisA[0] / regisB[0];
            regisC[1] = regisA[1] / regisB[1];
            regisC[2] = regisA[2] / regisB[2];
            regisC[3] = regisA[3] / regisB[3];
            toFloat4R(in->elements[index]) = toFloat4R(regisC[0]);
        }
    }

    __global__ void addScalarD(Tensor* in, float scalar){
        uint32 index = threadIdx.x + blockIdx.x * blockDim.x;
        if(index < in->dims.size){
            in->elements[index] += scalar;
        }
    }

    __global__ void addScalar4D(Tensor* in, float scalar){
        uint32 index = (threadIdx.x + blockIdx.x * blockDim.x )* 4;
        float regisA[4];
        float regisC[4] = {0};
        if(index < in->dims.size){
            toFloat4R(regisA[0]) = toFloat4R(in->elements[index]);
            regisC[0] = regisA[0] + scalar;
            regisC[1] = regisA[1] + scalar;
            regisC[2] = regisA[2] + scalar;
            regisC[3] = regisA[3] + scalar;
            toFloat4R(in->elements[index]) = toFloat4R(regisC[0]);
        }
    }

    __global__ void minusScalarD(Tensor* in, float scalar){
        uint32 index = threadIdx.x + blockIdx.x * blockDim.x;
        if(index < in->dims.size){
            in->elements[index] -= scalar;
        }
    }

    __global__ void minusScalar4D(Tensor* in, float scalar){
        uint32 index = (threadIdx.x + blockIdx.x * blockDim.x )* 4;
        float regisA[4];
        float regisC[4] = {0};
        if(index < in->dims.size){
            toFloat4R(regisA[0]) = toFloat4R(in->elements[index]);
            regisC[0] = regisA[0] - scalar;
            regisC[1] = regisA[1] - scalar;
            regisC[2] = regisA[2] - scalar;
            regisC[3] = regisA[3] - scalar;
            toFloat4R(in->elements[index]) = toFloat4R(regisC[0]);
        }
    }

    __global__ void hadamardScalarD(Tensor* in, float scalar){
        uint32 index = threadIdx.x + blockIdx.x * blockDim.x;
        if(index < in->dims.size){
            in->elements[index] *= scalar;
        }
    }

    __global__ void hadamardScalar4D(Tensor* in, float scalar){
        uint32 index = (threadIdx.x + blockIdx.x * blockDim.x )* 4;
        float regisA[4];
        float regisC[4] = {0};
        if(index < in->dims.size){
            toFloat4R(regisA[0]) = toFloat4R(in->elements[index]);
            regisC[0] = regisA[0] * scalar;
            regisC[1] = regisA[1] * scalar;
            regisC[2] = regisA[2] * scalar;
            regisC[3] = regisA[3] * scalar;
            toFloat4R(in->elements[index]) = toFloat4R(regisC[0]);
        }
    }

    __global__ void divideScalarD(Tensor* in, float scalar){
        uint32 index = threadIdx.x + blockIdx.x * blockDim.x;
        if(index < in->dims.size){
            in->elements[index] /= scalar;
        }
    }

    __global__ void divideScalar4D(Tensor* in, float scalar){
        uint32 index = (threadIdx.x + blockIdx.x * blockDim.x )* 4;
        float regisA[4];
        float regisC[4] = {0};
        if(index < in->dims.size){
            toFloat4R(regisA[0]) = toFloat4R(in->elements[index]);
            regisC[0] = regisA[0] / scalar;
            regisC[1] = regisA[1] / scalar;
            regisC[2] = regisA[2] / scalar;
            regisC[3] = regisA[3] / scalar;
            toFloat4R(in->elements[index]) = toFloat4R(regisC[0]);
        }
    }

    __global__ void constFillD(Tensor* A, float value){
        uint32 index = threadIdx.x + blockIdx.x * blockDim.x;
        if(index < A->dims.size){
            A->elements[index] = value;
        }
    }

    __global__ void constFill4D(Tensor* A, float value){
        uint32 index = (threadIdx.x + blockIdx.x * blockDim.x )* 4;
        float regisA[4];
        if(index < A->dims.size){
            #pragma unroll
            for (float & i : regisA){
                i = value;
            }
            toFloat4R(A->elements[index]) = toFloat4R(regisA[0]);
        }
    }

    __global__ void randUniformD(Tensor* A, long seed, float min, float max){
        uint32 id = (threadIdx.x + blockIdx.x * blockDim.x);
        if(id >= A->dims.size) return;
        curandStateXORWOW_t state;
        curand_init(id * seed, 0, 0, &state);
        float val = curand_uniform(&state);
        A->elements[id] = val * (max - min) + min;
    }

    __global__ void randNormalD(Tensor* A, long seed, float mean, float stddev){
        uint32 id = (threadIdx.x + blockIdx.x * blockDim.x);
        if(id >= A->dims.size) return;
        curandStateXORWOW_t state;
        curand_init(id * seed, 0, 0, &state);
        float val = curand_normal(&state);
        A->elements[id] = val * stddev + mean;
    }

    //index operators
    __device__ __host__ seblas::index4 seblas::index4::operator+(seblas::index4 other) const {
        return {n + other.n, c + other.c, h + other.h, w + other.w};
    }

    __device__ __host__ index4 index4::operator-(index4 other) const {
        return {n - other.n, c - other.c, h - other.h, w - other.w};
    }

    __device__ __host__ bool index4::operator==(index4 other) const {
        return n == other.n && c == other.c && h == other.h && w == other.w;
    }

    __device__ __host__ bool index4::operator>(index4 other) const {
        return n > other.n && c > other.c && h > other.h && w > other.w;
    }

    __device__ __host__ bool index4::operator<(index4 other) const {
        return n < other.n && c < other.c && h < other.h && w < other.w;
    }

    __device__ __host__ uint32 index4::getOffset() const {
        return n == 0 ? 1 : n * c == 0 ? 1 : c * h == 0 ? 1 : h * w;
    }

    string index4::toString() const {
        return "(" + to_string(n) + "," + to_string(c) + "," + to_string(h) + "," + to_string(w) + ")";
    }

    Tensor *Tensor::instantiate() {
        cudaMalloc(&elements, sizeof(float) * dims.size);
        assertCuda(__FILE__, __LINE__);
        cudaMemset(elements, 0, sizeof(float) * dims.size);
        assertCuda(__FILE__, __LINE__);
        deviceId = 0;
        return this;
    }

    Tensor *Tensor::destroy() {
        cudaFree(elements);
        assertCuda(__FILE__, __LINE__);
        return this;
    }

    Tensor *Tensor::instantiateHost() {
        cudaMallocHost(&elements, sizeof(float) * dims.size);
        assertCuda(__FILE__, __LINE__);
        deviceId = -1;
        return this;
    }

    Tensor *Tensor::destroyHost() {
        cudaFreeHost(elements);
        assertCuda(__FILE__, __LINE__);
        return this;
    }

    void Tensor::eliminate() {
        destroy();
        cudaFreeHost(this);
    }

    void Tensor::eliminateHost() {
        destroyHost();
        cudaFreeHost(this);
    }

    Tensor *Tensor::toDevice() {
        auto* output = Tensor::declare(dims)->instantiate();
        copyToH2D(output);
        destroyHost();
        cudaFreeHost(this);
        return output;
    }

    Tensor *Tensor::ripOffDevice() const {
        auto* output = Tensor::declare(dims)->instantiateHost();
        copyToD2H(output);
        assertCuda(__FILE__, __LINE__);
        return output;
    }

    Tensor *Tensor::ripOffHost() const {
        auto* output = Tensor::declare(dims)->instantiate();
        copyToH2D(output);
        assertCuda(__FILE__, __LINE__);
        return output;
    }

    Tensor *Tensor::toHost() {
        auto* output = Tensor::declare(dims)->instantiateHost();
        copyToD2H(output);
        destroy();
        cudaFreeHost(this);
        return output;
    }

    Tensor *Tensor::copyToH2D(Tensor *onDevice) const {
        assert(dims.size == onDevice->dims.size);
        cudaMemcpy(onDevice->elements, elements, sizeof(float) * dims.size, cudaMemcpyHostToDevice);
        assertCuda(__FILE__, __LINE__);
        return onDevice;
    }

    Tensor *Tensor::copyToD2H(Tensor *onHost) const {
        assert(dims.size == onHost->dims.size);
        cudaMemcpy(onHost->elements, elements, sizeof(float) * dims.size, cudaMemcpyDeviceToHost);
        assertCuda(__FILE__, __LINE__);
        return onHost;
    }

    Tensor *Tensor::copyToD2D(Tensor *onDevice) const {
        assert(dims.size == onDevice->dims.size);
        cudaMemcpy(onDevice->elements, elements, sizeof(float) * dims.size, cudaMemcpyDeviceToDevice);
        assertCuda(__FILE__, __LINE__);
        return onDevice;
    }

    Tensor *Tensor::copyToH2H(Tensor *onHost) const {
        assert(dims.size == onHost->dims.size);
        cudaMemcpy(onHost->elements, elements, sizeof(float) * dims.size, cudaMemcpyHostToHost);
        assertCuda(__FILE__, __LINE__);
        return onHost;
    }

    Tensor *Tensor::attach(Tensor *other) {
        assert(dims.size <= other->dims.size);
        elements = other->elements;
        deviceId = other->deviceId;
        return this;
    }

    Tensor *Tensor::attach(float *other, int deviceID) {
        elements = other;
        deviceId = deviceID;
        return this;
    }

    Tensor *Tensor::operator+(Tensor *other) {
        assert(dims.size == other->dims.size);
        assert(deviceId == other->deviceId);
        if(deviceId == -1){
            _alloc<CPU_THREADS>(addTensorCPU, this, other);
            return this;
        }

        if(dims.size % 4 == 0){
            uint32 grid = topOff(dims.size/4, CUDA_BLOCK_SIZE_1D);
            uint32 block = CUDA_BLOCK_SIZE_1D;
            add4D<<<grid, block>>>(this, other);
            assertCuda(__FILE__, __LINE__);
            return this;
        }

        uint32 grid = topOff(dims.size, CUDA_BLOCK_SIZE_1D);
        uint32 block = CUDA_BLOCK_SIZE_1D;
        addD<<<grid, block>>>(this, other);
        assertCuda(__FILE__, __LINE__);
        return this;
    }

    Tensor *Tensor::operator-(Tensor *other) {
        assert(dims.size == other->dims.size);
        assert(deviceId == other->deviceId);
        if(deviceId == -1){
            _alloc<CPU_THREADS>(minusTensorCPU, this, other);
            return this;
        }

        if(dims.size % 4 == 0){
            uint32 grid = topOff(dims.size/4, CUDA_BLOCK_SIZE_1D);
            uint32 block = CUDA_BLOCK_SIZE_1D;
            subtract4D<<<grid, block>>>(this, other);
            assertCuda(__FILE__, __LINE__);
            return this;
        }

        uint32 grid = topOff(dims.size, CUDA_BLOCK_SIZE_1D);
        uint32 block = CUDA_BLOCK_SIZE_1D;
        subtractD<<<grid, block>>>(this, other);
        assertCuda(__FILE__, __LINE__);
        return this;
    }

    Tensor *Tensor::operator*(Tensor *other) {
        assert(dims.size == other->dims.size);
        assert(deviceId == other->deviceId);
        if(deviceId == -1){
            _alloc<CPU_THREADS>(hadamardTensorCPU, this, other);
            return this;
        }

        if(dims.size % 4 == 0){
            uint32 grid = topOff(dims.size/4, CUDA_BLOCK_SIZE_1D);
            uint32 block = CUDA_BLOCK_SIZE_1D;
            hadamardProduct4D<<<grid, block>>>(this, other);
            assertCuda(__FILE__, __LINE__);
            return this;
        }

        uint32 grid = topOff(dims.size, CUDA_BLOCK_SIZE_1D);
        uint32 block = CUDA_BLOCK_SIZE_1D;
        hadamardProductD<<<grid, block>>>(this, other);
        assertCuda(__FILE__, __LINE__);
        return this;
    }

    Tensor *Tensor::operator/(Tensor *other) {
        assert(dims.size == other->dims.size);
        assert(deviceId == other->deviceId);
        if(deviceId == -1){
            _alloc<CPU_THREADS>(divideTensorCPU, this, other);
            return this;
        }

        if(dims.size % 4 == 0){
            uint32 grid = topOff(dims.size/4, CUDA_BLOCK_SIZE_1D);
            uint32 block = CUDA_BLOCK_SIZE_1D;
            divide4D<<<grid, block>>>(this, other);
            assertCuda(__FILE__, __LINE__);
            return this;
        }

        uint32 grid = topOff(dims.size, CUDA_BLOCK_SIZE_1D);
        uint32 block = CUDA_BLOCK_SIZE_1D;
        divideD<<<grid, block>>>(this, other);
        assertCuda(__FILE__, __LINE__);
        return this;
    }

    Tensor *Tensor::operator+(float other){
        if(deviceId == -1){
            _alloc<CPU_THREADS>(addScalarCPU, this, other);
            return this;
        }

        if(dims.size % 4 == 0){
            uint32 grid = topOff(dims.size/4, CUDA_BLOCK_SIZE_1D);
            uint32 block = CUDA_BLOCK_SIZE_1D;
            addScalar4D<<<grid, block>>>(this, other);
            assertCuda(__FILE__, __LINE__);
            return this;
        }

        uint32 grid = topOff(dims.size, CUDA_BLOCK_SIZE_1D);
        uint32 block = CUDA_BLOCK_SIZE_1D;
        addScalarD<<<grid, block>>>(this, other);
        assertCuda(__FILE__, __LINE__);
        return this;
    }

    Tensor *Tensor::operator-(float other){
        if(deviceId == -1){
            _alloc<CPU_THREADS>(minusScalarCPU, this, other);
            return this;
        }

        if(dims.size % 4 == 0){
            uint32 grid = topOff(dims.size/4, CUDA_BLOCK_SIZE_1D);
            uint32 block = CUDA_BLOCK_SIZE_1D;
            minusScalar4D<<<grid, block>>>(this, other);
            assertCuda(__FILE__, __LINE__);
            return this;
        }

        uint32 grid = topOff(dims.size, CUDA_BLOCK_SIZE_1D);
        uint32 block = CUDA_BLOCK_SIZE_1D;
        minusScalarD<<<grid, block>>>(this, other);
        assertCuda(__FILE__, __LINE__);
        return this;
    }

    Tensor *Tensor::operator*(float other){
        if(deviceId == -1){
            _alloc<CPU_THREADS>(hadamardScalarCPU, this, other);
            return this;
        }

        if(dims.size % 4 == 0){
            uint32 grid = topOff(dims.size/4, CUDA_BLOCK_SIZE_1D);
            uint32 block = CUDA_BLOCK_SIZE_1D;
            hadamardScalar4D<<<grid, block>>>(this, other);
            assertCuda(__FILE__, __LINE__);
            return this;
        }

        uint32 grid = topOff(dims.size, CUDA_BLOCK_SIZE_1D);
        uint32 block = CUDA_BLOCK_SIZE_1D;
        hadamardScalarD<<<grid, block>>>(this, other);
        assertCuda(__FILE__, __LINE__);
        return this;
    }

    Tensor *Tensor::operator/(float other){
        if(deviceId == -1){
            _alloc<CPU_THREADS>(divideScalarCPU, this, other);
            return this;
        }

        if(dims.size % 4 == 0){
            uint32 grid = topOff(dims.size/4, CUDA_BLOCK_SIZE_1D);
            uint32 block = CUDA_BLOCK_SIZE_1D;
            divideScalar4D<<<grid, block>>>(this, other);
            assertCuda(__FILE__, __LINE__);
            return this;
        }

        uint32 grid = topOff(dims.size, CUDA_BLOCK_SIZE_1D);
        uint32 block = CUDA_BLOCK_SIZE_1D;
        divideScalarD<<<grid, block>>>(this, other);
        assertCuda(__FILE__, __LINE__);
        return this;
    }

    Tensor* Tensor::constFill(float val){
        if(deviceId == -1){
            _alloc<CPU_THREADS>(constFillCPU, this, val);
            return this;
        }

        if(dims.size % 4 == 0){
            uint32 grid = topOff(dims.size/4, CUDA_BLOCK_SIZE_1D);
            uint32 block = CUDA_BLOCK_SIZE_1D;
            constFill4D<<<grid, block>>>(this, val);
            assertCuda(__FILE__, __LINE__);
            return this;
        }

        uint32 grid = topOff(dims.size, CUDA_BLOCK_SIZE_1D);
        uint32 block = CUDA_BLOCK_SIZE_1D;
        constFillD<<<grid, block>>>(this, val);
        assertCuda(__FILE__, __LINE__);
        return this;
    }

    Tensor* Tensor::randUniform(float min, float max) {
        if(deviceId == -1){
            _alloc<CPU_THREADS>(randUniformCPU, this, min, max);
            return this;
        }

        uint32 grid = topOff(dims.size, CUDA_BLOCK_SIZE_1D);
        uint32 block = CUDA_BLOCK_SIZE_1D;
        long seed = (long)chrono::system_clock::now().time_since_epoch().count();
        randUniformD<<<grid, block>>>(this, seed, min, max);
        assertCuda(__FILE__, __LINE__);
        return this;
    }

    Tensor* Tensor::randNormal(float mean, float stddev) {
        if(deviceId == -1){
            _alloc<CPU_THREADS>(randNormalCPU, this, mean, stddev);
            return this;
        }

        uint32 grid = topOff(dims.size, CUDA_BLOCK_SIZE_1D);
        uint32 block = CUDA_BLOCK_SIZE_1D;
        long seed = (long)chrono::duration_cast<std::chrono::microseconds>(
                chrono::system_clock::now().time_since_epoch()).count();
        randNormalD<<<grid, block>>>(this, seed, mean, stddev);
        assertCuda(__FILE__, __LINE__);
        return this;
    }

    __host__ float Tensor::get(uint32 offset) const {
        if(deviceId == -1) return elements[offset];
        float output;
        cudaMemcpy(&output, elements + offset, sizeof(float), cudaMemcpyDeviceToHost);
        assertCuda(__FILE__, __LINE__);
        return output;
    }
    
    Tensor *Tensor::ripOffHost(float* src) {
        cudaMemcpy(elements, src, sizeof(float) * dims.size, cudaMemcpyHostToDevice);
        assertCuda(__FILE__, __LINE__);
        return this;
    }
}