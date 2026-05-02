#pragma once

#include <vector>

#include "buffer.h"

using namespace std;

class Convolution
{
    public:
    Convolution();

    static Buffer detectEdges(const Buffer& buffer);

    static inline float ReLU(float x) { return max(0.0f,x);}
};