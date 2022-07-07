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

            new cuConv2D(cudnn,shape4(16, 3, 3, 3), 1, 1, 1, 1, false),
            new BatchNorm(),
            new ReLU(),

            new ShortcutEndpoint(false, 0x01, {}),
            new cuConv2D(cudnn,shape4(16, 16, 3, 3), 1, 1, 1, 1, false),
            new BatchNorm(),
            new ReLU(),
            new cuConv2D(cudnn,shape4(16, 16, 3, 3), 1, 1, 1, 1, false),
            new BatchNorm(),
            new ShortcutEndpoint(true, 0x01, {}),
            new ReLU(),

            new ShortcutEndpoint(false, 0x01, {}),
            new cuConv2D(cudnn,shape4(16, 16, 3, 3), 1, 1, 1, 1, false),
            new BatchNorm(),
            new ReLU(),
            new cuConv2D(cudnn,shape4(16, 16, 3, 3), 1, 1, 1, 1, false),
            new BatchNorm(),
            new ShortcutEndpoint(true, 0x01, {}),
            new ReLU(),

            new ShortcutEndpoint(false, 0x02, {}),
            new cuConv2D(cudnn,shape4(16, 16, 3, 3), 1, 1, 1, 1, false),
            new BatchNorm(),
            new ReLU(),
            new cuConv2D(cudnn,shape4(16, 16, 3, 3), 1, 1, 1, 1, false),
            new BatchNorm(),
            new ShortcutEndpoint(true, 0x02, {}),
            new ReLU(),

            new ShortcutEndpoint(false, 0x03, {}),
            new cuConv2D(cudnn,shape4(16, 16, 3, 3), 1, 1, 1, 1, false),
            new BatchNorm(),
            new ReLU(),
            new cuConv2D(cudnn,shape4(16, 16, 3, 3), 1, 1, 1, 1, false),
            new BatchNorm(),
            new ShortcutEndpoint(true, 0x03, {}),
            new ReLU(),

            new ShortcutEndpoint(false, 0x04, {}),
            new cuConv2D(cudnn,shape4(16, 16, 3, 3), 1, 1, 1, 1, false),
            new BatchNorm(),
            new ReLU(),
            new cuConv2D(cudnn,shape4(16, 16, 3, 3), 1, 1, 1, 1, false),
            new BatchNorm(),
            new ShortcutEndpoint(true, 0x04, {}),
            new ReLU(),

            new ShortcutEndpoint(false, 0x05, {}),
            new cuConv2D(cudnn,shape4(16, 16, 3, 3), 1, 1, 1, 1, false),
            new BatchNorm(),
            new ReLU(),
            new cuConv2D(cudnn,shape4(16, 16, 3, 3), 1, 1, 1, 1, false),
            new BatchNorm(),
            new ShortcutEndpoint(true, 0x05, {}),
            new ReLU(),

            new ShortcutEndpoint(false, 0x06, {}),
            new cuConv2D(cudnn,shape4(16, 16, 3, 3), 1, 1, 1, 1, false),
            new BatchNorm(),
            new ReLU(),
            new cuConv2D(cudnn,shape4(16, 16, 3, 3), 1, 1, 1, 1, false),
            new BatchNorm(),
            new ShortcutEndpoint(true, 0x06, {}),
            new ReLU(),

            new ShortcutEndpoint(false, 0x07, {}),
            new cuConv2D(cudnn,shape4(16, 16, 3, 3), 1, 1, 1, 1, false),
            new BatchNorm(),
            new ReLU(),
            new cuConv2D(cudnn,shape4(16, 16, 3, 3), 1, 1, 1, 1, false),
            new BatchNorm(),
            new ShortcutEndpoint(true, 0x07, {}),
            new ReLU(),

            new ShortcutEndpoint(false, 0x08, {}),
            new cuConv2D(cudnn,shape4(16, 16, 3, 3), 1, 1, 1, 1, false),
            new BatchNorm(),
            new ReLU(),
            new cuConv2D(cudnn,shape4(16, 16, 3, 3), 1, 1, 1, 1, false),
            new BatchNorm(),
            new ShortcutEndpoint(true, 0x08, {}),
            new ReLU(),
            new ShortcutEndpoint(false, 0x09, {}),
            new cuConv2D(cudnn,shape4(16, 16, 3, 3), 1, 1, 1, 1, false),
            new BatchNorm(),
            new ReLU(),
            new cuConv2D(cudnn,shape4(32, 16, 3, 3), 2, 2, 1, 1, false),
            new BatchNorm(),
            new ShortcutEndpoint(true, 0x09, {new cuConv2D(cudnn,shape4(32, 16, 1, 1), 2, 2, 0, 0, false),}),
            new ReLU(),

            new ShortcutEndpoint(false, 0x10, {}),
            new cuConv2D(cudnn,shape4(32, 32, 3, 3), 1, 1, 1, 1, false),
            new BatchNorm(),
            new ReLU(),
            new cuConv2D(cudnn,shape4(32, 32, 3, 3), 1, 1, 1, 1, false),
            new BatchNorm(),
            new ShortcutEndpoint(true, 0x10, {}),
            new ReLU(),

            new ShortcutEndpoint(false, 0x11, {}),
            new cuConv2D(cudnn,shape4(32, 32, 3, 3), 1, 1, 1, 1, false),
            new BatchNorm(),
            new ReLU(),
            new cuConv2D(cudnn,shape4(32, 32, 3, 3), 1, 1, 1, 1, false),
            new BatchNorm(),
            new ShortcutEndpoint(true, 0x11, {}),
            new ReLU(),

            new ShortcutEndpoint(false, 0x12, {}),
            new cuConv2D(cudnn,shape4(32, 32, 3, 3), 1, 1, 1, 1, false),
            new BatchNorm(),
            new ReLU(),
            new cuConv2D(cudnn,shape4(32, 32, 3, 3), 1, 1, 1, 1, false),
            new BatchNorm(),
            new ShortcutEndpoint(true, 0x12, {}),
            new ReLU(),

            new ShortcutEndpoint(false, 0x13, {}),
            new cuConv2D(cudnn,shape4(32, 32, 3, 3), 1, 1, 1, 1, false),
            new BatchNorm(),
            new ReLU(),
            new cuConv2D(cudnn,shape4(32, 32, 3, 3), 1, 1, 1, 1, false),
            new BatchNorm(),
            new ShortcutEndpoint(true, 0x13, {}),
            new ReLU(),

            new ShortcutEndpoint(false, 0x14, {}),
            new cuConv2D(cudnn,shape4(32, 32, 3, 3), 1, 1, 1, 1, false),
            new BatchNorm(),
            new ReLU(),
            new cuConv2D(cudnn,shape4(32, 32, 3, 3), 1, 1, 1, 1, false),
            new BatchNorm(),
            new ShortcutEndpoint(true, 0x14, {}),
            new ReLU(),

            new ShortcutEndpoint(false, 0x15, {}),
            new cuConv2D(cudnn,shape4(32, 32, 3, 3), 1, 1, 1, 1, false),
            new BatchNorm(),
            new ReLU(),
            new cuConv2D(cudnn,shape4(32, 32, 3, 3), 1, 1, 1, 1, false),
            new BatchNorm(),
            new ShortcutEndpoint(true, 0x15, {}),
            new ReLU(),

            new ShortcutEndpoint(false, 0x16, {}),
            new cuConv2D(cudnn,shape4(32, 32, 3, 3), 1, 1, 1, 1, false),
            new BatchNorm(),
            new ReLU(),
            new cuConv2D(cudnn,shape4(32, 32, 3, 3), 1, 1, 1, 1, false),
            new BatchNorm(),
            new ShortcutEndpoint(true, 0x16, {}),
            new ReLU(),

            new ShortcutEndpoint(false, 0x17, {}),
            new cuConv2D(cudnn,shape4(32, 32, 3, 3), 1, 1, 1, 1, false),
            new BatchNorm(),
            new ReLU(),
            new cuConv2D(cudnn,shape4(32, 32, 3, 3), 1, 1, 1, 1, false),
            new BatchNorm(),
            new ShortcutEndpoint(true, 0x17, {}),
            new ReLU(),

            new ShortcutEndpoint(false, 0x18, {}),
            new cuConv2D(cudnn,shape4(32, 32, 3, 3), 1, 1, 1, 1, false),
            new BatchNorm(),
            new ReLU(),
            new cuConv2D(cudnn,shape4(64, 32, 3, 3), 2, 2, 1, 1, false),
            new BatchNorm(),
            new ShortcutEndpoint(true, 0x18, {new cuConv2D(cudnn, shape4(64, 32, 1, 1), 2, 2, 0, 0, false),}),
            new ReLU(),

            new ShortcutEndpoint(false, 0x19, {}),
            new cuConv2D(cudnn,shape4(64, 64, 3, 3), 1, 1, 1, 1, false),
            new BatchNorm(),
            new ReLU(),
            new cuConv2D(cudnn,shape4(64, 64, 3, 3), 1, 1, 1, 1, false),
            new BatchNorm(),
            new ShortcutEndpoint(true, 0x19, {}),
            new ReLU(),

            new ShortcutEndpoint(false, 0x20, {}),
            new cuConv2D(cudnn,shape4(64, 64, 3, 3), 1, 1, 1, 1, false),
            new BatchNorm(),
            new ReLU(),
            new cuConv2D(cudnn,shape4(64, 64, 3, 3), 1, 1, 1, 1, false),
            new BatchNorm(),
            new ShortcutEndpoint(true, 0x20, {}),
            new ReLU(),

            new ShortcutEndpoint(false, 0x21, {}),
            new cuConv2D(cudnn,shape4(64, 64, 3, 3), 1, 1, 1, 1, false),
            new BatchNorm(),
            new ReLU(),
            new cuConv2D(cudnn,shape4(64, 64, 3, 3), 1, 1, 1, 1, false),
            new BatchNorm(),
            new ShortcutEndpoint(true, 0x21, {}),
            new ReLU(),

            new ShortcutEndpoint(false, 0x22, {}),
            new cuConv2D(cudnn,shape4(64, 64, 3, 3), 1, 1, 1, 1, false),
            new BatchNorm(),
            new ReLU(),
            new cuConv2D(cudnn,shape4(64, 64, 3, 3), 1, 1, 1, 1, false),
            new BatchNorm(),
            new ShortcutEndpoint(true, 0x22, {}),
            new ReLU(),

            new ShortcutEndpoint(false, 0x23, {}),
            new cuConv2D(cudnn,shape4(64, 64, 3, 3), 1, 1, 1, 1, false),
            new BatchNorm(),
            new ReLU(),
            new cuConv2D(cudnn,shape4(64, 64, 3, 3), 1, 1, 1, 1, false),
            new BatchNorm(),
            new ShortcutEndpoint(true, 0x23, {}),
            new ReLU(),

            new ShortcutEndpoint(false, 0x24, {}),
            new cuConv2D(cudnn,shape4(64, 64, 3, 3), 1, 1, 1, 1, false),
            new BatchNorm(),
            new ReLU(),
            new cuConv2D(cudnn,shape4(64, 64, 3, 3), 1, 1, 1, 1, false),
            new BatchNorm(),
            new ShortcutEndpoint(true, 0x24, {}),
            new ReLU(),

            new ShortcutEndpoint(false, 0x25, {}),
            new cuConv2D(cudnn,shape4(64, 64, 3, 3), 1, 1, 1, 1, false),
            new BatchNorm(),
            new ReLU(),
            new cuConv2D(cudnn,shape4(64, 64, 3, 3), 1, 1, 1, 1, false),
            new BatchNorm(),
            new ShortcutEndpoint(true, 0x25, {}),
            new ReLU(),


            new ShortcutEndpoint(false, 0x26, {}),
            new cuConv2D(cudnn,shape4(64, 64, 3, 3), 1, 1, 1, 1, false),
            new BatchNorm(),
            new ReLU(),
            new cuConv2D(cudnn,shape4(64, 64, 3, 3), 1, 1, 1, 1, false),
            new BatchNorm(),
            new ShortcutEndpoint(true, 0x26, {}),
            new ReLU(),

            new ShortcutEndpoint(false, 0x27, {}),
            new cuConv2D(cudnn,shape4(64, 64, 3, 3), 1, 1, 1, 1, false),
            new BatchNorm(),
            new ReLU(),
            new cuConv2D(cudnn,shape4(64, 64, 3, 3), 1, 1, 1, 1, false),
            new BatchNorm(),
            new ShortcutEndpoint(true, 0x27, {}),
            new ReLU(),
            
            new Linear(1024),
            new ReLU(),
            new Linear(10),
            new Softmax()
    });

    OptimizerInfo *info = new OPTIMIZER_MOMENTUM(0.01/64);

    model->construct(info);
    model->randInit();
    model->setLoss(crossEntropyLoss, crossEntropyCalc);

    auto *dataset = Dataset::construct(6400, 64, 50000, 60000, 1000,
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