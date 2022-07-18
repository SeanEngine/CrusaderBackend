//
// Created by Dylan on 6/20/2022.
//

#ifndef CRUSADER_NETPARAM_CUH
#define CRUSADER_NETPARAM_CUH

#include "../optimizers/Optimizers.cuh"

#include <iostream>
#include <fstream>

using namespace std;
namespace seann {
    /**
     * Netparams are parameters that are actually upgrading
     * such as weights and biases.
     * each net param will contain a optimizer
     */
    class NetParam {
    public:
        Parameter* A;
        Optimizer* opt;
        
        NetParam(Parameter* A, Optimizer* opt) : A(A), opt(opt) {}
        
        NetParam(Parameter* A, OptimizerInfo* info) : A(A) {
            opt = info->create(A);
        }
        
        template<typename... Args>
        explicit NetParam(OptimizerInfo* info, Args&&... args) {
            A = Parameter::create(std::forward<Args>(args)...);
            opt = info->create(A);
        }
        
        void setWeightDecay(float weightDecay) const {
            opt->L2 = weightDecay;
        }
        
        void update() const {
            opt->apply();
        }
        
        uint32 encodeNetParamInfo(fstream* fout, uint64 offset) const {
            fout->seekp((long long)offset);
            fout->write(reinterpret_cast<const char *>(&A->A->dims.n), sizeof(uint32));
            fout->write(reinterpret_cast<const char *>(&A->A->dims.c), sizeof(uint32));
            fout->write(reinterpret_cast<const char *>(&A->A->dims.h), sizeof(uint32));
            fout->write(reinterpret_cast<const char *>(&A->A->dims.w), sizeof(uint32));
            return 16;
        }
        
        uint32 encodeNetParamData(fstream* fout, uint64 offset) const {
            //to prevent memory overuse, I copied elements one by one form device mem
            float* curVal;
            cudaMallocHost(&curVal, 400 * sizeof(float));
            uint32 runningOffset = offset;
    
            fout->seekp((long long) runningOffset);
            //store values
            for(uint32 i = 0; i < A->A->dims.size; i+=400){
                cudaMemcpy(curVal, A->A->elements + i, sizeof(float) * (i + 400 > A->
                           A->dims.size ? A->A->dims.size % 400 : 400), cudaMemcpyDeviceToHost);
                fout->write(reinterpret_cast<const char *>(curVal), (long long)(sizeof(float)
                           * (i + 400 > A->A->dims.size ? A->A->dims.size % 400 : 400)));
            }
            assertCuda(__FILE__, __LINE__);
            runningOffset += A->A->dims.size * sizeof(float);
            
            cudaFreeHost(curVal);
            return runningOffset - offset;
        }
        
        static NetParam* decodeNetParamInfo(fstream* fin, uint64& offset, OptimizerInfo* info) {
            fin->seekg((long long)offset);
            uint32 n, c, h, w;
            fin->read(reinterpret_cast<char *>(&n), sizeof(uint32));
            fin->read(reinterpret_cast<char *>(&c), sizeof(uint32));
            fin->read(reinterpret_cast<char *>(&h), sizeof(uint32));
            fin->read(reinterpret_cast<char *>(&w), sizeof(uint32));
            offset += 16;
            return new NetParam(info, n, c, h, w);
        }
        
        static NetParam* decodeNetParamData(fstream* fin, uint64& offset, NetParam* param) {
            float* curVal;
            cudaMallocHost(&curVal, 400 * sizeof(float));
            uint32 runningOffset = offset;
            //load values
            for(uint32 i = 0; i < param->A->A->dims.size; i+=400){
                fin->seekg((long long)runningOffset);
                fin->read(reinterpret_cast<char *>(curVal), (long long)(sizeof(float) *
                    (i + 400 > param->A->A->dims.size ? param->A->A->dims.size % 400 : 400)));
                cudaMemcpy(param->A->A->elements + i, curVal, sizeof(float) *
                    (i + 400 > param->A->A->dims.size ? param->A->A->dims.size % 400 : 400), cudaMemcpyHostToDevice);
                runningOffset += (i + 400 > param->A->A->dims.size ? param->A->A->dims.size % 400 : 400) * sizeof(float);
            }
            cudaFreeHost(curVal);
            offset = runningOffset;
            return param;
        }
        
        NetParam* setWeight() {
            opt->isWeight = true;
            return this;
        }
        
        Tensor* data() const{
            return A->A;
        }
        
        Tensor* grad() const{
            return A->dA;
        }
    };
    
} // seann

#endif //CRUSADER_NETPARAM_CUH
