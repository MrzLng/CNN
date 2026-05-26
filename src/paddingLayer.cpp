#include <omp.h>

#include "paddingLayer.h"
#include "buffer.h"
#include "timer.h"

PaddingLayer::PaddingLayer(PaddingLayer::Padding_Type paddingType, int pad):paddingType(paddingType), pad(pad){}

void PaddingLayer::forward(const Buffer& input, Buffer& output)
{
    Timer timer("Padding");
    output.resize(input.getWidth()+2*pad, input.getHeight()+2*pad, input.getChannels());
    switch(paddingType)
    {
        case Padding_Type::ZERO_PADDING:
        fill(output.getVector(), output.getVector()+output.getSize(), 0);
        #pragma omp parallel for collapse(2) schedule(dynamic)
        for (int y=pad; y<input.getHeight()+pad; y++)
        {
            for (int x=pad; x<input.getWidth()+pad; x++)
            {
                output.setData(input.getData(x-pad,y-pad,0),x,y,0);
                output.setData(input.getData(x-pad,y-pad,1),x,y,1);
                output.setData(input.getData(x-pad,y-pad,2),x,y,2);
            }
        }
        break;
        case Padding_Type::EXTEND_EDGE:
        #pragma omp parallel for schedule(dynamic)
        for (int y=0;y<pad;y++)
        { 
            #pragma omp parallel for schedule(dynamic)
            for (int x=0;x<pad;x++)
            {
                output.setData(input.getData(0, 0, 0), x, y, 0);
                output.setData(input.getData(0, 0, 1), x, y, 1);
                output.setData(input.getData(0, 0, 2), x, y, 2);
            }
            #pragma omp parallel for schedule(dynamic)
            for (int x=pad;x<input.getWidth()+pad;x++)
            {
                output.setData(input.getData(x-pad, 0, 0), x, y, 0);
                output.setData(input.getData(x-pad, 0, 1), x, y, 1);
                output.setData(input.getData(x-pad, 0, 2), x, y, 2);
            }
            #pragma omp parallel for schedule(dynamic)
            for (int x=input.getWidth()+pad;x<input.getWidth()+2*pad;x++)
            {
                output.setData(input.getData(input.getWidth()-1, 0, 0), x, y, 0);
                output.setData(input.getData(input.getWidth()-1, 0, 1), x, y, 1);
                output.setData(input.getData(input.getWidth()-1, 0, 2), x, y, 2);
            }
        }
        #pragma omp parallel for schedule(dynamic)
        for (int y=pad;y<input.getHeight()+pad;y++)
        {
            int x=0;
            #pragma omp parallel for schedule(dynamic)
            for (int x=0;x<pad;x++)
            {
                output.setData(input.getData(0, y-pad, 0), x, y, 0);
                output.setData(input.getData(0, y-pad, 1), x, y, 1);
                output.setData(input.getData(0, y-pad, 2), x, y, 2);
            }
            #pragma omp parallel for schedule(dynamic)
            for (int x=pad;x<input.getWidth()+pad;x++)
            {
                output.setData(input.getData(x-pad, y-pad, 0), x, y, 0);
                output.setData(input.getData(x-pad, y-pad, 1), x, y, 1);
                output.setData(input.getData(x-pad, y-pad, 2), x, y, 2);
            }
            #pragma omp parallel for schedule(dynamic)
            for (int x=input.getWidth()+pad;x<input.getWidth()+2*pad;x++)
            {
                output.setData(input.getData(input.getWidth()-1, y-pad, 0), x, y, 0);
                output.setData(input.getData(input.getWidth()-1, y-pad, 1), x, y, 1);
                output.setData(input.getData(input.getWidth()-1, y-pad, 2), x, y, 2);
            }
        }
        #pragma omp parallel for schedule(dynamic)
        for (int y=input.getHeight()+pad;y<input.getHeight()+2*pad;y++)
        {
            #pragma omp parallel for schedule(dynamic)
            for (int x=0;x<pad;x++)
            {
                output.setData(input.getData(0, input.getHeight()-1, 0), x, y, 0);
                output.setData(input.getData(0, input.getHeight()-1, 1), x, y, 1);
                output.setData(input.getData(0, input.getHeight()-1, 2), x, y, 2);
            }
            #pragma omp parallel for schedule(dynamic)
            for (int x=pad;x<input.getWidth()+pad;x++)
            {
                output.setData(input.getData(x-pad, input.getHeight()-1, 0), x, y, 0);
                output.setData(input.getData(x-pad, input.getHeight()-1, 1), x, y, 1);
                output.setData(input.getData(x-pad, input.getHeight()-1, 2), x, y, 2);
            }
            #pragma omp parallel for schedule(dynamic)
            for (int x=input.getWidth()+pad;x<input.getWidth()+2*pad;x++)
            {
                output.setData(input.getData(input.getWidth()-1, input.getHeight()-1, 0), x, y, 0);
                output.setData(input.getData(input.getWidth()-1, input.getHeight()-1, 1), x, y, 1);
                output.setData(input.getData(input.getWidth()-1, input.getHeight()-1, 2), x, y, 2);
            }
        }
        break;
        case Padding_Type::REFLECTION:
        #pragma omp parallel for schedule(dynamic)
        for (int y=0;y<pad;y++)
        { 
            #pragma omp parallel for schedule(dynamic)
            for (int x=0;x<pad;x++)
            {
                output.setData(input.getData(pad-x-1,pad-y-1, 0), x, y, 0);
                output.setData(input.getData(pad-x-1,pad-y-1, 1), x, y, 1);
                output.setData(input.getData(pad-x-1,pad-y-1, 2), x, y, 2);
            }
            #pragma omp parallel for schedule(dynamic)
            for (int x=pad;x<input.getWidth()+pad;x++)
            {
                output.setData(input.getData(x-pad, pad-y-1, 0), x, y, 0);
                output.setData(input.getData(x-pad, pad-y-1, 1), x, y, 1);
                output.setData(input.getData(x-pad, pad-y-1, 2), x, y, 2);
            }
            #pragma omp parallel for schedule(dynamic)
            for (int x=input.getWidth()+pad;x<input.getWidth()+2*pad;x++)
            {
                output.setData(input.getData(input.getWidth()*2-x+pad-1, pad-y-1, 0), x, y, 0);
                output.setData(input.getData(input.getWidth()*2-x+pad-1, pad-y-1, 1), x, y, 1);
                output.setData(input.getData(input.getWidth()*2-x+pad-1, pad-y-1, 2), x, y, 2);
            }
        }
        #pragma omp parallel for schedule(dynamic)
        for (int y=pad;y<input.getHeight()+pad;y++)
        {
            #pragma omp parallel for schedule(dynamic)
            for (int x=0;x<pad;x++)
            {
                output.setData(input.getData(pad-x-1, y-pad, 0), x, y, 0);
                output.setData(input.getData(pad-x-1, y-pad, 1), x, y, 1);
                output.setData(input.getData(pad-x-1, y-pad, 2), x, y, 2);
            }
            #pragma omp parallel for schedule(dynamic)
            for (int x=pad;x<input.getWidth()+pad;x++)
            {
                output.setData(input.getData(x-pad, y-pad, 0), x, y, 0);
                output.setData(input.getData(x-pad, y-pad, 1), x, y, 1);
                output.setData(input.getData(x-pad, y-pad, 2), x, y, 2);
            }
            #pragma omp parallel for schedule(dynamic)
            for (int x=input.getWidth()+pad;x<input.getWidth()+2*pad;x++)
            {
                output.setData(input.getData(input.getWidth()*2-x+pad-1, y-pad, 0), x, y, 0);
                output.setData(input.getData(input.getWidth()*2-x+pad-1, y-pad, 1), x, y, 1);
                output.setData(input.getData(input.getWidth()*2-x+pad-1, y-pad, 2), x, y, 2);
            }
        }
        #pragma omp parallel for schedule(dynamic)
        for (int y=input.getHeight()+pad;y<input.getHeight()+2*pad;y++)
        {
            #pragma omp parallel for schedule(dynamic)
            for (int x=0;x<pad;x++)
            {
                output.setData(input.getData(pad-x-1, input.getHeight()*2-y+pad-1, 0), x, y, 0);
                output.setData(input.getData(pad-x-1, input.getHeight()*2-y+pad-1, 1), x, y, 1);
                output.setData(input.getData(pad-x-1, input.getHeight()*2-y+pad-1, 2), x, y, 2);
            }
            #pragma omp parallel for schedule(dynamic)
            for (int x=pad;x<input.getWidth()+pad;x++)
            {
                output.setData(input.getData(x-pad, input.getHeight()*2-y+pad-1, 0), x, y, 0);
                output.setData(input.getData(x-pad, input.getHeight()*2-y+pad-1, 1), x, y, 1);
                output.setData(input.getData(x-pad, input.getHeight()*2-y+pad-1, 2), x, y, 2);
            }
            #pragma omp parallel for schedule(dynamic)
            for (int x=input.getWidth()+pad;x<input.getWidth()+2*pad;x++)
            {
                output.setData(input.getData(input.getWidth()*2-x+pad-1, input.getHeight()*2-y+pad-1, 0), x, y, 0);
                output.setData(input.getData(input.getWidth()*2-x+pad-1, input.getHeight()*2-y+pad-1, 0), x, y, 1);
                output.setData(input.getData(input.getWidth()*2-x+pad-1, input.getHeight()*2-y+pad-1, 0), x, y, 2);
            }
        }
        break;
    }
}

void PaddingLayer::forward(const Buffer* input, Buffer* output)
{
    Timer timer("Padding");
    output->resize(input->getWidth()+2*pad, input->getHeight()+2*pad, input->getChannels());
    switch(paddingType)
    {
        case Padding_Type::ZERO_PADDING:
        fill(output->getVector(), output->getVector()+output->getSize(), 0);
        #pragma omp parallel for collapse(2) schedule(dynamic)
        for (int y=pad; y<input->getHeight()+pad; y++)
        {
            for (int x=pad; x<input->getWidth()+pad; x++)
            {
                output->setData(input->getData(x-pad,y-pad,0),x,y,0);
                output->setData(input->getData(x-pad,y-pad,1),x,y,1);
                output->setData(input->getData(x-pad,y-pad,2),x,y,2);
            }
        }
        break;
        case Padding_Type::EXTEND_EDGE:
        #pragma omp parallel for schedule(dynamic)
        for (int y=0;y<pad;y++)
        { 
            #pragma omp parallel for schedule(dynamic)
            for (int x=0;x<pad;x++)
            {
                output->setData(input->getData(0, 0, 0), x, y, 0);
                output->setData(input->getData(0, 0, 1), x, y, 1);
                output->setData(input->getData(0, 0, 2), x, y, 2);
            }
            #pragma omp parallel for schedule(dynamic)
            for (int x=pad;x<input->getWidth()+pad;x++)
            {
                output->setData(input->getData(x-pad, 0, 0), x, y, 0);
                output->setData(input->getData(x-pad, 0, 1), x, y, 1);
                output->setData(input->getData(x-pad, 0, 2), x, y, 2);
            }
            #pragma omp parallel for schedule(dynamic)
            for (int x=input->getWidth()+pad;x<input->getWidth()+2*pad;x++)
            {
                output->setData(input->getData(input->getWidth()-1, 0, 0), x, y, 0);
                output->setData(input->getData(input->getWidth()-1, 0, 1), x, y, 1);
                output->setData(input->getData(input->getWidth()-1, 0, 2), x, y, 2);
            }
        }
        #pragma omp parallel for schedule(dynamic)
        for (int y=pad;y<input->getHeight()+pad;y++)
        {
            int x=0;
            #pragma omp parallel for schedule(dynamic)
            for (int x=0;x<pad;x++)
            {
                output->setData(input->getData(0, y-pad, 0), x, y, 0);
                output->setData(input->getData(0, y-pad, 1), x, y, 1);
                output->setData(input->getData(0, y-pad, 2), x, y, 2);
            }
            #pragma omp parallel for schedule(dynamic)
            for (int x=pad;x<input->getWidth()+pad;x++)
            {
                output->setData(input->getData(x-pad, y-pad, 0), x, y, 0);
                output->setData(input->getData(x-pad, y-pad, 1), x, y, 1);
                output->setData(input->getData(x-pad, y-pad, 2), x, y, 2);
            }
            #pragma omp parallel for schedule(dynamic)
            for (int x=input->getWidth()+pad;x<input->getWidth()+2*pad;x++)
            {
                output->setData(input->getData(input->getWidth()-1, y-pad, 0), x, y, 0);
                output->setData(input->getData(input->getWidth()-1, y-pad, 1), x, y, 1);
                output->setData(input->getData(input->getWidth()-1, y-pad, 2), x, y, 2);
            }
        }
        #pragma omp parallel for schedule(dynamic)
        for (int y=input->getHeight()+pad;y<input->getHeight()+2*pad;y++)
        {
            #pragma omp parallel for schedule(dynamic)
            for (int x=0;x<pad;x++)
            {
                output->setData(input->getData(0, input->getHeight()-1, 0), x, y, 0);
                output->setData(input->getData(0, input->getHeight()-1, 1), x, y, 1);
                output->setData(input->getData(0, input->getHeight()-1, 2), x, y, 2);
            }
            #pragma omp parallel for schedule(dynamic)
            for (int x=pad;x<input->getWidth()+pad;x++)
            {
                output->setData(input->getData(x-pad, input->getHeight()-1, 0), x, y, 0);
                output->setData(input->getData(x-pad, input->getHeight()-1, 1), x, y, 1);
                output->setData(input->getData(x-pad, input->getHeight()-1, 2), x, y, 2);
            }
            #pragma omp parallel for schedule(dynamic)
            for (int x=input->getWidth()+pad;x<input->getWidth()+2*pad;x++)
            {
                output->setData(input->getData(input->getWidth()-1, input->getHeight()-1, 0), x, y, 0);
                output->setData(input->getData(input->getWidth()-1, input->getHeight()-1, 1), x, y, 1);
                output->setData(input->getData(input->getWidth()-1, input->getHeight()-1, 2), x, y, 2);
            }
        }
        break;
        case Padding_Type::REFLECTION:
        #pragma omp parallel for schedule(dynamic)
        for (int y=0;y<pad;y++)
        { 
            #pragma omp parallel for schedule(dynamic)
            for (int x=0;x<pad;x++)
            {
                output->setData(input->getData(pad-x-1,pad-y-1, 0), x, y, 0);
                output->setData(input->getData(pad-x-1,pad-y-1, 1), x, y, 1);
                output->setData(input->getData(pad-x-1,pad-y-1, 2), x, y, 2);
            }
            #pragma omp parallel for schedule(dynamic)
            for (int x=pad;x<input->getWidth()+pad;x++)
            {
                output->setData(input->getData(x-pad, pad-y-1, 0), x, y, 0);
                output->setData(input->getData(x-pad, pad-y-1, 1), x, y, 1);
                output->setData(input->getData(x-pad, pad-y-1, 2), x, y, 2);
            }
            #pragma omp parallel for schedule(dynamic)
            for (int x=input->getWidth()+pad;x<input->getWidth()+2*pad;x++)
            {
                output->setData(input->getData(input->getWidth()*2-x+pad-1, pad-y-1, 0), x, y, 0);
                output->setData(input->getData(input->getWidth()*2-x+pad-1, pad-y-1, 1), x, y, 1);
                output->setData(input->getData(input->getWidth()*2-x+pad-1, pad-y-1, 2), x, y, 2);
            }
        }
        #pragma omp parallel for schedule(dynamic)
        for (int y=pad;y<input->getHeight()+pad;y++)
        {
            #pragma omp parallel for schedule(dynamic)
            for (int x=0;x<pad;x++)
            {
                output->setData(input->getData(pad-x-1, y-pad, 0), x, y, 0);
                output->setData(input->getData(pad-x-1, y-pad, 1), x, y, 1);
                output->setData(input->getData(pad-x-1, y-pad, 2), x, y, 2);
            }
            #pragma omp parallel for schedule(dynamic)
            for (int x=pad;x<input->getWidth()+pad;x++)
            {
                output->setData(input->getData(x-pad, y-pad, 0), x, y, 0);
                output->setData(input->getData(x-pad, y-pad, 1), x, y, 1);
                output->setData(input->getData(x-pad, y-pad, 2), x, y, 2);
            }
            #pragma omp parallel for schedule(dynamic)
            for (int x=input->getWidth()+pad;x<input->getWidth()+2*pad;x++)
            {
                output->setData(input->getData(input->getWidth()*2-x+pad-1, y-pad, 0), x, y, 0);
                output->setData(input->getData(input->getWidth()*2-x+pad-1, y-pad, 1), x, y, 1);
                output->setData(input->getData(input->getWidth()*2-x+pad-1, y-pad, 2), x, y, 2);
            }
        }
        #pragma omp parallel for schedule(dynamic)
        for (int y=input->getHeight()+pad;y<input->getHeight()+2*pad;y++)
        {
            #pragma omp parallel for schedule(dynamic)
            for (int x=0;x<pad;x++)
            {
                output->setData(input->getData(pad-x-1, input->getHeight()*2-y+pad-1, 0), x, y, 0);
                output->setData(input->getData(pad-x-1, input->getHeight()*2-y+pad-1, 1), x, y, 1);
                output->setData(input->getData(pad-x-1, input->getHeight()*2-y+pad-1, 2), x, y, 2);
            }
            #pragma omp parallel for schedule(dynamic)
            for (int x=pad;x<input->getWidth()+pad;x++)
            {
                output->setData(input->getData(x-pad, input->getHeight()*2-y+pad-1, 0), x, y, 0);
                output->setData(input->getData(x-pad, input->getHeight()*2-y+pad-1, 1), x, y, 1);
                output->setData(input->getData(x-pad, input->getHeight()*2-y+pad-1, 2), x, y, 2);
            }
            #pragma omp parallel for schedule(dynamic)
            for (int x=input->getWidth()+pad;x<input->getWidth()+2*pad;x++)
            {
                output->setData(input->getData(input->getWidth()*2-x+pad-1, input->getHeight()*2-y+pad-1, 0), x, y, 0);
                output->setData(input->getData(input->getWidth()*2-x+pad-1, input->getHeight()*2-y+pad-1, 0), x, y, 1);
                output->setData(input->getData(input->getWidth()*2-x+pad-1, input->getHeight()*2-y+pad-1, 0), x, y, 2);
            }
        }
        break;
    }
}
