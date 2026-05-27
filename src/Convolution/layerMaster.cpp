#include "layerMaster.h"

LayerMaster::LayerMaster(Buffer initialBuffer, std::vector<std::unique_ptr<Layer>> layers)
:layers(std::move(layers)) 
{
    mainBuffer = new Buffer(initialBuffer);
    tempBuffer = new Buffer(initialBuffer);
}

LayerMaster::~LayerMaster()
{
    delete mainBuffer;
    delete tempBuffer;
}

void LayerMaster::start()
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