#pragma once

#include <functional>

#include "buffer.h"
#include "layer.h"\

class PoolingLayer: public Layer
{
    public:
    PoolingLayer(std::function<float(float,float)> func, int poolSize, int stride): func(func), poolSize(poolSize), stride(stride){}

    void forward(const Buffer& input, Buffer& output);
    void forward(const Buffer* input, Buffer* output);

    private:
    std::function<float(float,float)> func;
    int poolSize;
    int stride;
};