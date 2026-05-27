#pragma once

#include <vector>

#include "buffer.h"
#include "kernel.h"
#include "layer.h"
#include "kernelBank.h"

using namespace std;

class ConvolutionLayer: public virtual Layer
{
    public:
    ConvolutionLayer(KernelBank kernels):kernels(kernels){} //vector kernel dimenstions must be same

    void forward(const Buffer& input, Buffer& output);
    void forward(const Buffer* input, Buffer* output);

    private:
    KernelBank kernels;
};