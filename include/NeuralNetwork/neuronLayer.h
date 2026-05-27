#pragma once

#include "buffer.h"
#include "matrix.h"

class NeuronLayer
{
    public:
    NeuronLayer(Buffer buffer);

    private:
    Matrix matrix;
};