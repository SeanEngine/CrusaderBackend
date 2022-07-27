
#include "validation/UnitTestTools.cuh"
#include "seblas/operations/cuOperations.cuh"
#include "cudnn.h"
#include "seann/seann.cuh"
#include "seio/data/Dataset.cuh"
#include "seio/data/DataLoader.cuh"
#include "seio/modelcache/codec.cuh"


using namespace std::this_thread;
using namespace chrono;
using namespace seblas;
using namespace seio;
using namespace seann;


int main(int argc, char** argv) {
    auto* model = new Sequential(shape4(4,3,448,448),{
        new cuConv2D(shape4(64,3,7,7),2,2,3,3, true),
        new BatchNorm(),
        new LReLU(0.1),
        new MaxPool2D(2,2,2,2),
        
        new cuConv2D(shape4(192,64,3,3),1,1,1,1, true),
        new BatchNorm(),
        new LReLU(0.1),
        new MaxPool2D(2,2,2,2),
        
        new cuConv2D(shape4(128,192,1,1),1,1,0,0, true),
        new BatchNorm(),
        new LReLU(0.1),

        new cuConv2D(shape4(256,128,3,3),1,1,1,1, true),
        new BatchNorm(),
        new LReLU(0.1),
        
        new cuConv2D(shape4(256,256,1,1),1,1,0,0, true),
        new BatchNorm(),
        new LReLU(0.1),
        
        new cuConv2D(shape4(512,256,3,3),1,1,1,1, true),
        new BatchNorm(),
        new LReLU(0.1),
        
        new MaxPool2D(2,2,2,2),

        new cuConv2D(shape4(256,512,1,1),1,1,0,0, true),
        new BatchNorm(),
        new LReLU(0.1),

        new cuConv2D(shape4(512,256,3,3),1,1,1,1, true),
        new BatchNorm(),
        new LReLU(0.1),

        new cuConv2D(shape4(256,512,1,1),1,1,0,0, true),
        new BatchNorm(),
        new LReLU(0.1),

        new cuConv2D(shape4(512,256,3,3),1,1,1,1, true),
        new BatchNorm(),
        new LReLU(0.1),

        new cuConv2D(shape4(256,512,1,1),1,1,0,0, true),
        new BatchNorm(),
        new LReLU(0.1),

        new cuConv2D(shape4(512,256,3,3),1,1,1,1, true),
        new BatchNorm(),
        new LReLU(0.1),

        new cuConv2D(shape4(256,512,1,1),1,1,0,0, true),
        new BatchNorm(),
        new LReLU(0.1),

        new cuConv2D(shape4(512,256,3,3),1,1,1,1, true),
        new BatchNorm(),
        new LReLU(0.1),

        new cuConv2D(shape4(512,512,1,1),1,1,0,0, true),
        new BatchNorm(),
        new LReLU(0.1),

        new cuConv2D(shape4(1024,512,3,3),1,1,1,1, true),
        new BatchNorm(),
        new LReLU(0.1),
        
        new MaxPool2D(2,2,2,2),

        new cuConv2D(shape4(512,1024,1,1),1,1,0,0, true),
        new BatchNorm(),
        new LReLU(0.1),

        new cuConv2D(shape4(1024,512,3,3),1,1,1,1, true),
        new BatchNorm(),
        new LReLU(0.1),

        new cuConv2D(shape4(512,1024,1,1),1,1,0,0, true),
        new BatchNorm(),
        new LReLU(0.1),

        new cuConv2D(shape4(1024,512,3,3),1,1,1,1, true),
        new BatchNorm(),
        new LReLU(0.1),

        new cuConv2D(shape4(1024,1024,3,3),1,1,1,1, true),
        new BatchNorm(),
        new LReLU(0.1),

        new cuConv2D(shape4(1024,1024,3,3),2,2,1,1, true),
        new BatchNorm(),
        new LReLU(0.1),

        new cuConv2D(shape4(1024,1024,3,3),1,1,1,1, true),
        new BatchNorm(),
        new LReLU(0.1),

        new cuConv2D(shape4(1024,1024,3,3),1,1,1,1, true),
        new BatchNorm(),
        new LReLU(0.1),
        
        new Linear(4096),
        new LReLU(0.1),
        new Dropout(0.5),
        
        new Linear(7*7*30),
        new ReLU(),
    });
    
    OptimizerInfo* info = new OPTIMIZER_MOMENTUM(0.01);
    
    model->construct(info);
    model->randInit();
    model->setLoss(Yolo1CompositeLoss, Yolo1CompositeCalc);
    
    Dataset* set = Dataset::construct(640, 4, 16000, 500,
                                      shape4(3,448,448), shape4(7,7,30));
    set->setDataFetcher(new DataFetcher(R"(D:\Resources\Datasets\VOCBin)"
                                         , "Yolo1_VOC_Data", 16000, fetchCrdat));
    set->setProcSteps({new UniformNorm(0,1)});
    set->allocTestSet(160);
    set->procDisplay = new ProcDisplay("Yolo1Cps");
    
    model->train(set, true, 1);
    
}

