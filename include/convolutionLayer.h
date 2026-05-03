#pragma once

#include <vector>

#include "buffer.h"
#include "kernel.h"
#include "layer.h"

using namespace std;

class ConvolutionLayer: public virtual Layer
{
    public:
    ConvolutionLayer(const Kernel& kernel):kernel(kernel){}

    Buffer forward(const Buffer& input, Buffer& output);

    static inline float ReLU(float x) { return max(0.0f,x);}

    private:
    const Kernel& kernel;
};