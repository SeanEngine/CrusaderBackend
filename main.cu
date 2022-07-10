#include "validation/UnitTestTools.cuh"
#include "seblas/operations/cuOperations.cuh"
#include "cudnn.h"
#include "seann/seann.cuh"
#include "seio/data/Dataset.cuh"
#include "seio/data/DataLoader.cuh"


using namespace std::this_thread;
using namespace chrono;
using namespace seblas;
using namespace seann;

int main(int argc, char** argv) {

    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);
    auto *model = new Sequential(shape4(64, 3, 32, 32), {
            new cuConv2D(cudnn, shape4(24, 3, 3, 3), 1, 1, 1, 1, false),
            new BatchNorm(),
            new ReLU(),

            new DenseBlock(cudnn, 12, 6),

            new BatchNorm(),
            new ReLU(),
            new cuConv2D(cudnn, shape4(48,96,1,1), 1, 1, 0, 0, false),
            new AvgPool2D(2,2,2,2),

            new DenseBlock(cudnn, 12, 6),

            new BatchNorm(),
            new ReLU(),
            new cuConv2D(cudnn, shape4(60,120,1,1), 1, 1, 0, 0, false),
            new AvgPool2D(2,2,2,2),

            new DenseBlock(cudnn, 12, 6),
            new AvgPool2D(2,2,2,2),
            
            new Linear(512),
            new BatchNorm(),
            new ReLU(),
            new Linear(10),
            new Softmax()
    });

    OptimizerInfo *info = new OPTIMIZER_MOMENTUM(0.01/64);

    model->construct(info);
    model->randInit();
    model->setLoss(crossEntropyLoss, crossEntropyCalc);

    auto *dataset = Dataset::construct(6400, 64, 50000, 60000, 150,
                                       shape4(3, 32, 32), shape4(10, 1));
    const char *BASE_PATH = R"(D:\Resources\Datasets\cifar-10-bin\data_batch_)";
    for (int i = 0; i < 5; i++) {
        string binPath = BASE_PATH + to_string(i + 1) + ".bin";
        fetchCIFAR(dataset, binPath.c_str(), i);
    }
    const char* TEST_BASE_PATH = R"(D:\Resources\Datasets\cifar-10-bin\test_batch.bin)";
    fetchCIFAR(dataset, TEST_BASE_PATH, 5);

    dataset->allocTestSet(3200);
    dataset->setProcSteps({
        new UniformNorm()
    });

    dataset->runPreProc();

    dataset->setAugmentSteps({
        new RandFlipW(dataset->MINI_BATCH_SIZE, dataset->dataShape, dataset->labelShape),
        new RandCorp(dataset->MINI_BATCH_SIZE, dataset->dataShape, dataset->labelShape,4,4)
    });

    model->train(dataset, true, 1);
}