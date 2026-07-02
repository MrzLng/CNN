#include "convolutionLayerMaster.h"

ConvolutionLayerMaster::ConvolutionLayerMaster(Buffer initialBuffer, std::vector<std::unique_ptr<Layer>> layers)
:layers(std::move(layers)) 
{
    mainBuffer = new Buffer(initialBuffer);
    tempBuffer = new Buffer(initialBuffer);
}

ConvolutionLayerMaster::~ConvolutionLayerMaster()
{
    delete mainBuffer;
    delete tempBuffer;
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