#include <omp.h>

#include "reluLayer.h"
#include "buffer.h"
#include "timer.h"

void ReluLayer::forward(const Buffer& input, Buffer& output)
{
    Timer timer("Relu");
    output.resize(input.getWidth(), input.getHeight(), input.getChannels());
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int y=0;y<input.getHeight();y++)
    {
        for (int x=0;x<input.getWidth();x++)
        {
            output.setData(max(input.getData(x,y,0),0.0f),x,y,0);
            output.setData(max(input.getData(x,y,1),0.0f),x,y,1);
            output.setData(max(input.getData(x,y,2),0.0f),x,y,2);
        }
    }
}

void ReluLayer::forward(const Buffer* input, Buffer* output)
{
    Timer timer("Relu");
    output->resize(input->getWidth(), input->getHeight(), input->getChannels());
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int y=0;y<input->getHeight();y++)
    {
        for (int x=0;x<input->getWidth();x++)
        {
            output->setData(max(input->getData(x,y,0),0.0f),x,y,0);
            output->setData(max(input->getData(x,y,1),0.0f),x,y,1);
            output->setData(max(input->getData(x,y,2),0.0f),x,y,2);
        }
    }
}