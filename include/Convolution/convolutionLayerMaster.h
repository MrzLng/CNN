#pragma once

#include <memory>
#include <vector>

#include "layer.h"
#include "buffer.h"

class ConvolutionLayerMaster
{
    public:
    ConvolutionLayerMaster(Buffer initialBuffer, std::vector<std::unique_ptr<Layer>> layers);
    ~ConvolutionLayerMaster();

    inline Buffer* getBufferPtr() {return mainBuffer;}

    void start();

    private:
    std::vector<std::unique_ptr<Layer>> layers;
    
    Buffer* mainBuffer;
    Buffer* tempBuffer;
};