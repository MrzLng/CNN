#include "includes.h"

#include <omp.h>
#include <iostream>

Kernel vertical({-1.0f,0.0f,1.0f,-2.0f,0.0f,2.0f,-1.0f,0.0f,1.0f}, 3, 3, 0);
Kernel horizontal({-1.0f,-2.0f,-1.0f,0.0f,0.0f,0.0f,1.0f,2.0f,1.0f}, 3, 3, 0);

float fun(float x, float y)
{
    return max(x,y);
}

int main()
{
    Timer timer("main");
    Image image("resources\\MJ_tiles_circles_bamboo_480x480.png");
    Buffer buffer(image);
    Buffer pad = buffer.padding(Buffer::Padding_Type::EXTEND_EDGE, 1);
    Buffer nb(buffer.getWidth(), buffer.getHeight(), buffer.getChannels());
    ConvolutionLayer c(horizontal);
    c.forward(pad,nb);
    nb.toImage("outputs\\new.png");
    Buffer ds(buffer.getWidth(), buffer.getHeight(), buffer.getChannels());
    PoolingLayer pool(fun, 10, 10);
    pool.forward(nb,ds);
    ds.toImage("outputs\\idk.png");
    return 0;
}