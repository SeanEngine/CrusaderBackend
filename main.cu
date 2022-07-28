
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
    auto* model = new Sequential(shape4(8,3,224,224),{
        new cuConv2D(shape4(64,3,7,7),2,2,3,3,true),
        new BatchNorm(),
        new ReLU(),
        new MaxPool2D(2,2,2,2),
        
        //first stage blocks
        new ShortcutEndpoint(false, 0, {}),
        new cuConv2D(shape4(64,64,1,1),1,1,0,0,true),
        new BatchNorm(),
        new ReLU(),
        new cuConv2D(shape4(64,64,3,3),1,1,1,1,true),
        new BatchNorm(),
        new ReLU(),
        new cuConv2D(shape4(256,64,1,1),1,1,0,0,true),
        new BatchNorm(),
        new ReLU(),
        new ShortcutEndpoint(true, 0, {
            new cuConv2D(shape4(256,64,1,1),1,1,0,0,true),
            new BatchNorm(),
            new ReLU(),
        }),
        
        new ShortcutEndpoint(false, 1, {}),
        new cuConv2D(shape4(64,256,1,1),1,1,0,0,true),
        new BatchNorm(),
        new ReLU(),
        new cuConv2D(shape4(64,64,3,3),1,1,1,1,true),
        new BatchNorm(),
        new ReLU(),
        new cuConv2D(shape4(256,64,1,1),1,1,0,0,true),
        new BatchNorm(),
        new ReLU(),
        new ShortcutEndpoint(true, 1, {}),
        
        new ShortcutEndpoint(false, 2, {}),
        new cuConv2D(shape4(64,256,1,1),1,1,0,0,true),
        new BatchNorm(),
        new ReLU(),
        new cuConv2D(shape4(64,64,3,3),1,1,1,1,true),
        new BatchNorm(),
        new ReLU(),
        new cuConv2D(shape4(256,64,1,1),1,1,0,0,true),
        new BatchNorm(),
        new ReLU(),
        new ShortcutEndpoint(true, 2, {}),
        
        //second stage blocks
        new ShortcutEndpoint(false, 3, {}),
        new cuConv2D(shape4(128,256,1,1),2,2,0,0,true),
        new BatchNorm(),
        new ReLU(),
        new cuConv2D(shape4(128,128,3,3),1,1,1,1,true),
        new BatchNorm(),
        new ReLU(),
        new cuConv2D(shape4(512,128,1,1),1,1,0,0,true),
        new BatchNorm(),
        new ReLU(),
        new ShortcutEndpoint(true, 3, {
            new cuConv2D(shape4(512,256,1,1),2,2,0,0,true),
            new BatchNorm(),
            new ReLU(),
        }),
        
        new ShortcutEndpoint(false, 4, {}),
        new cuConv2D(shape4(128,512,1,1),1,1,0,0,true),
        new BatchNorm(),
        new ReLU(),
        new cuConv2D(shape4(128,128,3,3),1,1,1,1,true),
        new BatchNorm(),
        new ReLU(),
        new cuConv2D(shape4(512,128,1,1),1,1,0,0,true),
        new BatchNorm(),
        new ReLU(),
        new ShortcutEndpoint(true, 4, {}),
        
        new ShortcutEndpoint(false, 5, {}),
        new cuConv2D(shape4(128,512,1,1),1,1,0,0,true),
        new BatchNorm(),
        new ReLU(),
        new cuConv2D(shape4(128,128,3,3),1,1,1,1,true),
        new BatchNorm(),
        new ReLU(),
        new cuConv2D(shape4(512,128,1,1),1,1,0,0,true),
        new BatchNorm(),
        new ReLU(),
        new ShortcutEndpoint(true, 5, {}),
        
        new ShortcutEndpoint(false, 6, {}),
        new cuConv2D(shape4(128,512,1,1),1,1,0,0,true),
        new BatchNorm(),
        new ReLU(),
        new cuConv2D(shape4(128,128,3,3),1,1,1,1,true),
        new BatchNorm(),
        new ReLU(),
        new cuConv2D(shape4(512,128,1,1),1,1,0,0,true),
        new BatchNorm(),
        new ReLU(),
        new ShortcutEndpoint(true, 6, {}),
        
        new ShortcutEndpoint(false, 7, {}),
        new cuConv2D(shape4(128,512,1,1),1,1,0,0,true),
        new BatchNorm(),
        new ReLU(),
        new cuConv2D(shape4(128,128,3,3),1,1,1,1,true),
        new BatchNorm(),
        new ReLU(),
        new cuConv2D(shape4(512,128,1,1),1,1,0,0,true),
        new BatchNorm(),
        new ReLU(),
        new ShortcutEndpoint(true, 7, {}),
        
        //third stage blocks
        new ShortcutEndpoint(false, 8, {}),
        new cuConv2D(shape4(256,512,1,1),2,2,0,0,true),
        new BatchNorm(),
        new ReLU(),
        new cuConv2D(shape4(256,256,3,3),1,1,1,1,true),
        new BatchNorm(),
        new ReLU(),
        new cuConv2D(shape4(1024,256,1,1),1,1,0,0,true),
        new BatchNorm(),
        new ReLU(),
        new ShortcutEndpoint(true, 8, {
            new cuConv2D(shape4(1024,512,1,1),2,2,0,0,true),
            new BatchNorm(),
            new ReLU(),
        }),
        
        new ShortcutEndpoint(false, 9, {}),
        new cuConv2D(shape4(256,1024,1,1),1,1,0,0,true),
        new BatchNorm(),
        new ReLU(),
        new cuConv2D(shape4(256,256,3,3),1,1,1,1,true),
        new BatchNorm(),
        new ReLU(),
        new cuConv2D(shape4(1024,256,1,1),1,1,0,0,true),
        new BatchNorm(),
        new ReLU(),
        new ShortcutEndpoint(true, 9, {}),

        new ShortcutEndpoint(false, 10, {}),
        new cuConv2D(shape4(256,1024,1,1),1,1,0,0,true),
        new BatchNorm(),
        new ReLU(),
        new cuConv2D(shape4(256,256,3,3),1,1,1,1,true),
        new BatchNorm(),
        new ReLU(),
        new cuConv2D(shape4(1024,256,1,1),1,1,0,0,true),
        new BatchNorm(),
        new ReLU(),
        new ShortcutEndpoint(true, 10, {}),

        new ShortcutEndpoint(false, 11, {}),
        new cuConv2D(shape4(256,1024,1,1),1,1,0,0,true),
        new BatchNorm(),
        new ReLU(),
        new cuConv2D(shape4(256,256,3,3),1,1,1,1,true),
        new BatchNorm(),
        new ReLU(),
        new cuConv2D(shape4(1024,256,1,1),1,1,0,0,true),
        new BatchNorm(),
        new ReLU(),
        new ShortcutEndpoint(true, 11, {}),

        new ShortcutEndpoint(false, 12, {}),
        new cuConv2D(shape4(256,1024,1,1),1,1,0,0,true),
        new BatchNorm(),
        new ReLU(),
        new cuConv2D(shape4(256,256,3,3),1,1,1,1,true),
        new BatchNorm(),
        new ReLU(),
        new cuConv2D(shape4(1024,256,1,1),1,1,0,0,true),
        new BatchNorm(),
        new ReLU(),
        new ShortcutEndpoint(true, 12, {}),
        
        new ShortcutEndpoint(false, 13, {}),
        new cuConv2D(shape4(256,1024,1,1),1,1,0,0,true),
        new BatchNorm(),
        new ReLU(),
        new cuConv2D(shape4(256,256,3,3),1,1,1,1,true),
        new BatchNorm(),
        new ReLU(),
        new cuConv2D(shape4(1024,256,1,1),1,1,0,0,true),
        new BatchNorm(),
        new ReLU(),
        new ShortcutEndpoint(true, 13, {}),
        
        //forth stage blocks
        new ShortcutEndpoint(false, 14, {}),
        new cuConv2D(shape4(512,1024,1,1),2,2,0,0,true),
        new BatchNorm(),
        new ReLU(),
        new cuConv2D(shape4(512,512,3,3),1,1,1,1,true),
        new BatchNorm(),
        new ReLU(),
        new cuConv2D(shape4(2048,512,1,1),1,1,0,0,true),
        new BatchNorm(),
        new ReLU(),
        new ShortcutEndpoint(true, 14, {
            new cuConv2D(shape4(2048,1024,1,1),2,2,0,0,true),
            new BatchNorm(),
            new ReLU(),
        }),
        
        new ShortcutEndpoint(false, 15, {}),
        new cuConv2D(shape4(512,2048,1,1),1,1,0,0,true),
        new BatchNorm(),
        new ReLU(),
        new cuConv2D(shape4(512,512,3,3),1,1,1,1,true),
        new BatchNorm(),
        new ReLU(),
        new cuConv2D(shape4(2048,512,1,1),1,1,0,0,true),
        new BatchNorm(),
        new ReLU(),
        new ShortcutEndpoint(true, 15, {}),
        
        new ShortcutEndpoint(false, 16, {}),
        new cuConv2D(shape4(512,2048,1,1),1,1,0,0,true),
        new BatchNorm(),
        new ReLU(),
        new cuConv2D(shape4(512,512,3,3),1,1,1,1,true),
        new BatchNorm(),
        new ReLU(),
        new cuConv2D(shape4(2048,512,1,1),1,1,0,0,true),
        new BatchNorm(),
        new ReLU(),
        new ShortcutEndpoint(true, 16, {}),
        
        new AvgPool2D(2,2,2,2),
        new Linear(1000),
        new Softmax()
    });

    OptimizerInfo* info = new OPTIMIZER_MOMENTUM(0.01);

    model->construct(info);
    model->randInit();
    model->setLoss(crossEntropyLoss, crossEntropyCalc);
    
    Dataset* set = Dataset::construct(1600, 8, 1280900, 100, shape4(3,224,224), shape4(1000,1));
    set->setDataFetcher(new DataFetcher("E:\\ImageNet\\ILSVRC2012train", 1280900, 256, new ImgLoader(
            shape4(3,224,224)
            )));
    set->allocTestSet();
    set->setProcSteps({
         new UniformNorm()
    });
    
    set->setAugmentSteps({
        new RandFlipW(set->MINI_BATCH_SIZE, set->dataShape, set->labelShape),
        new RandCorp(set->MINI_BATCH_SIZE, set->dataShape, set->labelShape,30,30)
    });
    
    set->procDisplay = new ProcDisplay("CELoss");
    
    model->train(set, true, 1);
    
}

