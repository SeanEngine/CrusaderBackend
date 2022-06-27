//
// Created by Dylan on 6/27/2022.
//

#include "Data.cuh"

namespace seio {
    Data *Data::declare(shape4 dataShape, shape4 labelShape) {
        Data* out;
        cudaMallocHost(&out, sizeof(Data));
        out->X = Tensor::declare(dataShape);
        out->label = Tensor::declare(labelShape);
        return out;
    }
    
    Data *Data::instantiate() {
        X->instantiate();
        label->instantiate();
        return this;
    }
    
    Data *Data::instantiateHost() {
        X->instantiateHost();
        label->instantiateHost();
        return this;
    }
    
    Data *Data::inherit(Tensor *X0, Tensor *label0) {
        X->attach(X0);
        label->attach(label0);
        return this;
    }
    
    Data *Data::copyOffD2D(Data *onDevice) {
        onDevice->X->copyToD2D(X);
        onDevice->label->copyToD2D(label);
        return this;
    }
    
    Data *Data::copyOffH2D(Data *onHost) {
        onHost->X->copyToH2D(X);
        onHost->label->copyToH2D(label);
        return this;
    }
    
    Data *Data::copyOffD2H(Data *onDevice) {
        onDevice->X->copyToD2H(X);
        onDevice->label->copyToD2H(label);
        return this;
    }
    
    Data *Data::copyOffH2H(Data *onHost) {
        onHost->X->copyToH2H(X);
        onHost->label->copyToH2H(label);
        return this;
    }
    
    Data *Data::copyOffD2D(Tensor *X0, Tensor *label0) {
        X0->copyToD2D(X);
        label0->copyToD2D(label);
        return this;
    }
    
    Data *Data::copyOffH2D(Tensor *X0, Tensor *label0) {
        X0->copyToH2D(X);
        label0->copyToH2D(label);
        return this;
    }
    
    Data *Data::copyOffD2H(Tensor *X0, Tensor *label0) {
        X0->copyToD2H(X);
        label0->copyToD2H(label);
        return this;
    }
    
    Data *Data::copyOffH2H(Tensor *X0, Tensor *label0) {
        X0->copyToH2H(X);
        label0->copyToH2H(label);
        return this;
    }
    
    void Data::destroy() {
        X->destroy();
        label->destroy();
        cudaFree(this);
    }
    
    void Data::destroyHost() {
        X->destroyHost();
        label->destroyHost();
        cudaFreeHost(this);
    }
} // seio