//
// Created by Dylan on 6/2/2022.
//

#ifndef CRUSADER_UNITTESTTOOLS_CUH
#define CRUSADER_UNITTESTTOOLS_CUH

#include <string>
#include "seblas/tensor/Tensor.cuh"
#include "seblas/operations/cuGEMM.cuh"
#include "seblas/assist/Inspections.cuh"

using namespace std;
using namespace seblas;

void appendTensorToString(string* input, Tensor* victim);

void stringToFile(string* input, const string& fName);

//generate lots of data to test (not dimensional sensitive)
void normalUnitTest();

void linearUnitTest();

void convUnitTest();

void optimUnitTest();

void batchnormUnitTest();

void softmaxUnitTest();

Tensor* simpleGemm(Tensor* A, Tensor* B, Tensor* C);
#endif //CRUSADER_UNITTESTTOOLS_CUH
