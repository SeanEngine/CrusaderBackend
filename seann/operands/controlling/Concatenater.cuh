//
// Created by Dylan on 7/4/2022.
//

#ifndef CRUSADER_CONCATENATER_CUH
#define CRUSADER_CONCATENATER_CUH

#include "../OperandBase.cuh"

namespace seann {
    class Concatenater : public OperandBase {
    public:
        uint32 paramCount;
        Parameter** params;
    };
} // seann

#endif //CRUSADER_CONCATENATER_CUH
