cmake_minimum_required(VERSION 3.22)
project(Crusader CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -lcudnn -lcublas)

set(CMAKE_CUDA_COMPILER D:/Program\ Files/NVIDIA/CUDA_115_Dev/bin/nvcc.exe)
set(CUDNN_PATH "D:/Program\ Files/NVIDIA/CUDNN_115")
find_package(CUDA 11.5 REQUIRED)

# find_package(CUDNN REQUIRED)
add_executable(Crusader main.cu

        seblas/assist/CudaAssert.cu
        seblas/assist/CudaAssert.cuh
        seblas/tensor/Tensor.cu
        seblas/tensor/Tensor.cuh
        seblas/tensor/TensorUtils.cu
        seblas/tensor/TensorUtils.cuh
        seblas/tensor/Parameter.cu
        seblas/tensor/Parameter.cuh
        seblas/assist/Inspections.cu
        seblas/assist/Inspections.cuh
        seblas/operations/cuGEMM.cu
        seblas/operations/cuGEMM.cuh
        seblas/operations/cuLinear.cu
        seblas/operations/cuLinear.cuh
        seblas/operations/cuConv.cu
        seblas/operations/cuConv.cuh
        seblas/operations/cuOptim.cu
        seblas/operations/cuOptim.cuh
        seblas/operations/cuLoss.cu
        seblas/operations/cuLoss.cuh
        seblas/operations/cuReduce.cu
        seblas/operations/cuReduce.cuh
        seblas/operations/cuOperations.cuh
        seblas/operations/cuPool.cu
        seblas/operations/cuPool.cuh
        seblas/operations/cuActivation.cu
        seblas/operations/cuActivation.cuh
        seblas/operations/cuBatchNorm.cu
        seblas/operations/cuBatchNorm.cuh

        seutil/exec/ThreadController.cu
        seutil/exec/ThreadController.cuh

        seio/logging/Color.cu
        seio/logging/Color.cuh
        seio/logging/LogUtils.cu
        seio/logging/LogUtils.cuh

        validation/UnitTestTools.cu
        validation/UnitTestTools.cuh

        seann/optimizers/Optimizers.cu
        seann/optimizers/Optimizers.cuh
        seann/containers/Sequential.cu
        seann/containers/Sequential.cuh
        seann/containers/NetParam.cu
        seann/containers/NetParam.cuh
        seann/operands/OperandBase.cu
        seann/operands/OperandBase.cuh
        seann/operands/dylan/Linear.cu
        seann/operands/dylan/Linear.cuh
        seann/operands/dylan/Conv2D.cu
        seann/operands/dylan/Conv2D.cuh
        seann/operands/dylan/BatchNorm.cu
        seann/operands/dylan/BatchNorm.cuh
        seann/operands/dylan/MaxPool2D.cu
        seann/operands/dylan/MaxPool2D.cuh
        seann/operands/dylan/ReLU.cu
        seann/operands/dylan/ReLU.cuh
        seann/operands/dylan/Softmax.cu
        seann/operands/dylan/Softmax.cuh
        seio/data/Dataset.cu
        seio/data/Dataset.cuh
        seio/data/DataLoader.cu
        seio/data/DataLoader.cuh
        seann/seann.cuh
        seann/containers/Block.cu seann/containers/Block.cuh seblas/operations/cuDropout.cu seblas/operations/cuDropout.cuh seann/operands/dylan/Dropout.cu seann/operands/dylan/Dropout.cuh seann/operands/cudnn/cuConv2D.cu seann/operands/cudnn/cuConv2D.cuh)

include_directories(${CUDNN_PATH}/include)
link_directories(${CUDNN_PATH} ${CUDNN_PATH}/bin ${CUDNN_PATH}/lib/x64 ${CUDNN_PATH}/dec/dll_x64)
target_link_libraries(Crusader ${CUDA_LIBRARIES} cudnn ${CUDNN_PATH}/dec/dll_x64/zlibwapi.lib)

set_target_properties(Crusader PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES 86
)
