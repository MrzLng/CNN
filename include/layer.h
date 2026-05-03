#pragma once

#include "buffer.h"

class Layer
{
    public:
    virtual ~Layer() = default;

    virtual Buffer forward(const Buffer& input, Buffer& output)=0;
};