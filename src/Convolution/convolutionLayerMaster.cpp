#include "convolutionLayerMaster.h"

ConvolutionLayerMaster::ConvolutionLayerMaster( std::vector<std::unique_ptr<Layer>> layers)
:layers(std::move(layers)) 
{
    mainBuffer = new Buffer(960, 540, 3);
    tempBuffer = new Buffer(960, 540, 3);
}

ConvolutionLayerMaster::~ConvolutionLayerMaster()
{
    delete mainBuffer;
    delete tempBuffer;
}

void ConvolutionLayerMaster::initializeBuffer(Buffer initialBuffer)
{
    delete mainBuffer;
    delete tempBuffer;
    mainBuffer = new Buffer(initialBuffer);
    tempBuffer = new Buffer(initialBuffer);
}

void ConvolutionLayerMaster::start()
{
    Buffer* temp;
    for (int i=0; i<layers.size(); i++)
    {
        temp = mainBuffer;
        mainBuffer = tempBuffer;
        tempBuffer = temp;

        layers[i]->forward(tempBuffer, mainBuffer);
    }
}