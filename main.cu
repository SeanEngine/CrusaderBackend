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

    auto *model = new Sequential(shape4(64, 3, 32, 32), {
    
            new Conv2D(shape4(32, 3, 3, 3), 1, 1, 1, 1, false),
            //new BatchNorm(),
            new ReLU(),
            new Conv2D(shape4(32, 32, 3, 3), 1, 1, 1, 1, false),
            //new BatchNorm(),
            new ReLU(),
            new Conv2D(shape4(32, 32, 3, 3), 1, 1, 1, 1, false),
            //new BatchNorm(),
            new ReLU(),
            new Conv2D(shape4(32, 32, 3, 3), 1, 1, 1, 1, false),
            //new BatchNorm(),
            new ReLU(),

            /*
            new MaxPool2D(2, 2),

            new Conv2D(shape4(64, 32, 3, 3), 1, 1, 1, 1, false),
            new ReLU(),
            new Conv2D(shape4(64, 64, 3, 3), 1, 1, 1, 1, false),
            new ReLU(),
            new MaxPool2D(2, 2),

            new Conv2D(shape4(128, 64, 3, 3), 1, 1, 1, 1, false),
            new ReLU(),
            new Conv2D(shape4(128, 128, 3, 3), 1, 1, 1, 1, false),
            new ReLU(),
             */
            //new MaxPool2D(2, 2),
            
            new Linear(120),
            //new BatchNorm(),
            new ReLU(),
            new Linear(10),
            new Softmax()
    });

    OptimizerInfo *info = new OPTIMIZER_MOMENTUM(0.0003);

    model->construct(info);
    model->randInit();
    model->setLoss(crossEntropyLoss, crossEntropyCalc);

    auto *dataset = Dataset::construct(3200, 64, 50000, 50,
                                       shape4(3, 32, 32), shape4(10, 1));
    const char *BASE_PATH = R"(D:\Resources\Datasets\cifar-10-bin\data_batch_)";
    for (int i = 0; i < 5; i++) {
        string binPath = BASE_PATH + to_string(i + 1) + ".bin";
        fetchCIFAR(dataset, binPath.c_str(), i);
    }
    model->train(dataset);
}