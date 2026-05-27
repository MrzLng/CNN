#include "poolingLayer.h"
#include "timer.h"

#include <omp.h>

void PoolingLayer::forward(const Buffer &input, Buffer &output)
{
    Timer timer("Pooling");
    output.resize((input.getWidth() - poolSize) / stride + 1, (input.getHeight() - poolSize) / stride + 1, input.getChannels());
#pragma omp parallel for collapse(3) schedule(dynamic)
    for (int y = 0; y < output.getHeight(); y++)
    {
        for (int x = 0; x < output.getWidth(); x++)
        {
            for (int c = 0; c < output.getChannels(); c++)
            {
                float value = 0;
                for (int poolY = 0; poolY < poolSize; poolY++)
                {
                    for (int poolX = 0; poolX < poolSize; poolX++)
                    {
                        value = func(value, input.getData(x * stride + poolX, y * stride + poolY, c));
                    }
                }
                output.setData(value, x, y, c);
            }
        }
    }
}

void PoolingLayer::forward(const Buffer* input, Buffer* output)
{
    Timer timer("Pooling");
    output->resize((input->getWidth() - poolSize) / stride + 1, (input->getHeight() - poolSize) / stride + 1, input->getChannels());
#pragma omp parallel for collapse(3) schedule(dynamic)
    for (int y = 0; y < output->getHeight(); y++)
    {
        for (int x = 0; x < output->getWidth(); x++)
        {
            for (int c = 0; c < output->getChannels(); c++)
            {
                float value = 0;
                for (int poolY = 0; poolY < poolSize; poolY++)
                {
                    for (int poolX = 0; poolX < poolSize; poolX++)
                    {
                        value = func(value, input->getData(x * stride + poolX, y * stride + poolY, c));
                    }
                }
                output->setData(value, x, y, c);
            }
        }
    }
}
