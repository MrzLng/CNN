#pragma once

#include <memory>
#include <vector>

#include "layer.h"
#include "buffer.h"

class LayerMaster
{
    public:
    LayerMaster(Buffer initialBuffer, std::vector<std::unique_ptr<Layer>> layers);
    ~LayerMaster();

    inline Buffer* getBufferPtr() {return mainBuffer;}

    void start();

    private:
    std::vector<std::unique_ptr<Layer>> layers;
    
    Buffer* mainBuffer;
    Buffer* tempBuffer;
};