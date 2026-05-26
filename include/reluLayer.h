#pragma once

#include "layer.h"

class ReluLayer : public Layer
{
    public:
    void forward(const Buffer& input, Buffer& output);
    void forward(const Buffer* input, Buffer* output);
};