cmake_minimum_required(VERSION 3.22)
project(Crusader CUDA CXX)

set(CMAKE_CUDA_STANDARD 23)
set(CMAKE_CXX_STANDARD 23)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -lcudnn -lcublas)

set(CMAKE_CUDA_COMPILER D:/Program\ Files/NVIDIA/CUDA_115_Dev/bin/nvcc.exe)
set(CUDNN_PATH "D:/Program\ Files/NVIDIA/CUDNN_115")
set(OpenCV_DIR "D:\\Dependencies\\OpenCV_CUDA\\build")
include_directories(${OpenCV_DIR}/include)
find_package(OpenCV REQUIRED core highgui imgproc)
find_package(CUDA 11.5 REQUIRED)

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
        seblas/operations/cuDropout.cu
        seblas/operations/cuDropout.cuh
        seblas/operations/cuConcat.cu
        seblas/operations/cuConcat.cuh

        seutil/exec/ThreadController.cu
        seutil/exec/ThreadController.cuh

        seio/logging/Color.cu
        seio/logging/Color.cuh
        seio/logging/LogUtils.cu
        seio/logging/LogUtils.cuh
        seio/data/Dataset.cu
        seio/data/Dataset.cuh
        seio/data/DataLoader.cu
        seio/data/DataLoader.cuh
        seio/data/DataTransform.cu
        seio/data/DataTransform.cuh
        seio/data/DataAugmentor.cu
        seio/data/DataAugmentor.cuh
        seio/data/Data.cu
        seio/data/Data.cuh

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
        seann/operands/seblas/Linear.cu
        seann/operands/seblas/Linear.cuh
        seann/operands/seblas/Conv2D.cu
        seann/operands/seblas/Conv2D.cuh
        seann/operands/seblas/BatchNorm.cu
        seann/operands/seblas/BatchNorm.cuh
        seann/operands/seblas/pooling/MaxPool2D.cu
        seann/operands/seblas/pooling/MaxPool2D.cuh
        seann/operands/seblas/activate/ReLU.cu
        seann/operands/seblas/activate/ReLU.cuh
        seann/operands/seblas/Softmax.cu
        seann/operands/seblas/Softmax.cuh
        seann/operands/seblas/Dropout.cu
        seann/operands/seblas/Dropout.cuh
        seann/operands/cudnn/cuConv2D.cu
        seann/operands/cudnn/cuConv2D.cuh
        seann/operands/controlling/ShortcutEndpoint.cu
        seann/operands/controlling/ShortcutEndpoint.cuh
        seann/operands/prefabs/DenseBlock.cu
        seann/operands/prefabs/DenseBlock.cuh
        seann/operands/controlling/ChannelConcatenater.cu
        seann/operands/controlling/ChannelConcatenater.cuh
        seann/seann.cuh
        seann/operands/seblas/pooling/AvgPool2D.cu
        seann/operands/seblas/pooling/AvgPool2D.cuh
        seann/operands/seblas/pooling/GlobalAvgPool.cu
        seann/operands/seblas/pooling/GlobalAvgPool.cuh
        seann/operands/seblas/activate/LReLU.cu
        seann/operands/seblas/activate/LReLU.cuh

        seio/modelcache/codec.cu
        seio/modelcache/codec.cuh
        seio/modelcache/OperandCodecs.cu
        seio/modelcache/OperandCodecs.cuh
        seio/watchDog/TrainProcDisplay.cu
        seio/watchDog/TrainProcDisplay.cuh
        seio/fileParsers/pugiconfig.cuh
        seio/fileParsers/pugixml.cu
        seio/fileParsers/pugixml.cuh
        seio/data/DataFetcher.cu seio/data/DataFetcher.cuh)
include_directories(${CUDNN_PATH}/include)
link_directories(${CUDNN_PATH} ${CUDNN_PATH}/bin ${CUDNN_PATH}/lib/x64 ${CUDNN_PATH}/dec/dll_x64)
target_link_libraries(Crusader ${CUDA_LIBRARIES} cudnn ${OpenCV_LIBS})

set_target_properties(Crusader PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES 86
)
