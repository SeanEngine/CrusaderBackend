//
// Created by Dylan on 6/3/2022.
//

#include "Inspections.cuh"

namespace seblas{
    void inspect(Tensor* target){
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
                        printf("%f, ", proc->elements[
                                n * target->dims.c * target->dims.h * target->dims.w +
                                c * target->dims.h * target->dims.w +
                                h * target->dims.w +
                                w
                        ]);
                    }
                    printf("\n");
                }
                printf("\n");
            }
            printf("\n");
        }

        if(target->deviceId != -1){
            proc->eliminateHost();
        }
        assertCuda(__FILE__, __LINE__);
    }
}