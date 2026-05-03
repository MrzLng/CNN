#include "poolingLayer.h"
#include "timer.h"

#include <omp.h>

Buffer PoolingLayer::forward(const Buffer& input, Buffer& output)
{
    Timer timer("pooling");
    Buffer result(input.getWidth()/stride, input.getHeight()/stride, input.getChannels());
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int y=0; y<result.getHeight();y++)
    {
        for (int x=0; x<result.getWidth();x++)
        {
            float poolR = 0.0f;
            float poolG = 0.0f;
            float poolB = 0.0f;
            for (int poolY=0;poolY<poolSize;poolY++)
            {
                for (int poolX=0;poolX<poolSize;poolX++)
                {
                    poolR = func(poolR, input.getData(x*stride+poolX,y*stride+poolY,0));
                    poolG = func(poolR, input.getData(x*stride+poolX,y*stride+poolY,1));
                    poolB = func(poolR, input.getData(x*stride+poolX,y*stride+poolY,2));
                }
            }
            output.setData(poolR,x,y,0);
            output.setData(poolG,x,y,1);
            output.setData(poolB,x,y,2);
        }
    }
    return result;
}