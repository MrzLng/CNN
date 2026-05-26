#pragma once

#include "layer.h"
#include "buffer.h"

class PaddingLayer : public Layer
{
    public:
    enum class Padding_Type
    {
        EXTEND_EDGE,
        ZERO_PADDING,
        REFLECTION,
    };

    PaddingLayer(Padding_Type paddingType, int pad);

    void forward(const Buffer& input, Buffer& output);
    void forward(const Buffer* input, Buffer* output);

    private:
    Padding_Type paddingType;
    int pad;
};