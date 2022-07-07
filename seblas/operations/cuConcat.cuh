//
// Created by Dylan on 7/5/2022.
//

#ifndef CRUSADER_CUCONCAT_CUH
#define CRUSADER_CUCONCAT_CUH

#include "../tensor/Tensor.cuh"
#include "../tensor/Parameter.cuh"

namespace seblas {
    Parameter* concat(Parameter** Xs, uint32 paramCount, Parameter* Y);
    
    void concatGrads(Parameter* Y, Parameter** Xs, uint32 paramCount);
} // seblas

#endif //CRUSADER_CUCONCAT_CUH
