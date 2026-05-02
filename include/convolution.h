#pragma once

#include <vector>

#include "buffer.h"

using namespace std;

class Convolution
{
    public:
    Convolution();

    static Buffer detectEdges(const Buffer& buffer);
};