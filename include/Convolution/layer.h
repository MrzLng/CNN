#pragma once

#include "buffer.h"

class Layer
{
    public:
    virtual ~Layer() = default;

    virtual void forward(const Buffer& input, Buffer& output) = 0;
    virtual void forward(const Buffer* input, Buffer* output) = 0;
};