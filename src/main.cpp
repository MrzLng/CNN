#include "includes.h"

#include <omp.h>
#include <iostream>

int main()
{
    Image image("C:\\Users\\Moritz\\Documents\\GitHub\\CNN\\resources\\MJ_tiles_circles_bamboo_480x480.png");
    Buffer buffer(image);
    Buffer padded = buffer.padding(Buffer::Padding_Type::ZERO_PADDING, 1);
    Buffer newBuffer = Convolution::detectEdges(padded);
    newBuffer.toImage("edge.png");
    buffer.toImage("normal.png");
    return 0;
}