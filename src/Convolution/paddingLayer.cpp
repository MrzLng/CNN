#include <omp.h>

#include "paddingLayer.h"
#include "buffer.h"
#include "timer.h"

PaddingLayer::PaddingLayer(PaddingLayer::Padding_Type paddingType, int pad) : paddingType(paddingType), pad(pad) {}

void PaddingLayer::forward(const Buffer &input, Buffer &output)
{
    Timer timer("Padding");
    output.resize(input.getWidth() + 2 * pad, input.getHeight() + 2 * pad, input.getChannels());
    switch (paddingType)
    {
    case Padding_Type::ZERO_PADDING:
        fill(output.getVectorPtr(), output.getVectorPtr() + output.getSize(), 0);
#pragma omp parallel for collapse(3) schedule(dynamic)
        for (int c = 0; c < input.getChannels(); c++)
        {
            for (int y = pad; y < input.getHeight() + pad; y++)
            {
                for (int x = pad; x < input.getWidth() + pad; x++)
                {
                    output.setData(input.getData(x - pad, y - pad, c), x, y, c);
                }
            }
        }
        break;
    case Padding_Type::EXTEND_EDGE:
#pragma omp parallel for schedule(dynamic)
        for (int c = 0; c < input.getChannels(); c++)
        {
#pragma omp parallel for schedule(dynamic)
            for (int y = 0; y < pad; y++)
            {
#pragma omp parallel for schedule(dynamic)
                for (int x = 0; x < pad; x++)
                {
                    output.setData(input.getData(0, 0, c), x, y, c);
                }
#pragma omp parallel for schedule(dynamic)
                for (int x = pad; x < input.getWidth() + pad; x++)
                {
                    output.setData(input.getData(x - pad, 0, c), x, y, c);
                }
#pragma omp parallel for schedule(dynamic)
                for (int x = input.getWidth() + pad; x < input.getWidth() + 2 * pad; x++)
                {
                    output.setData(input.getData(input.getWidth() - 1, 0, c), x, y, c);
                }
            }
#pragma omp parallel for schedule(dynamic)
            for (int y = pad; y < input.getHeight() + pad; y++)
            {
                int x = 0;
#pragma omp parallel for schedule(dynamic)
                for (int x = 0; x < pad; x++)
                {
                    output.setData(input.getData(0, y - pad, c), x, y, c);
                }
#pragma omp parallel for schedule(dynamic)
                for (int x = pad; x < input.getWidth() + pad; x++)
                {
                    output.setData(input.getData(x - pad, y - pad, c), x, y, c);
                }
#pragma omp parallel for schedule(dynamic)
                for (int x = input.getWidth() + pad; x < input.getWidth() + 2 * pad; x++)
                {
                    output.setData(input.getData(input.getWidth() - 1, y - pad, c), x, y, c);
                }
            }
#pragma omp parallel for schedule(dynamic)
            for (int y = input.getHeight() + pad; y < input.getHeight() + 2 * pad; y++)
            {
#pragma omp parallel for schedule(dynamic)
                for (int x = 0; x < pad; x++)
                {
                    output.setData(input.getData(0, input.getHeight() - 1, c), x, y, c);
                }
#pragma omp parallel for schedule(dynamic)
                for (int x = pad; x < input.getWidth() + pad; x++)
                {
                    output.setData(input.getData(x - pad, input.getHeight() - 1, c), x, y, c);
                }
#pragma omp parallel for schedule(dynamic)
                for (int x = input.getWidth() + pad; x < input.getWidth() + 2 * pad; x++)
                {
                    output.setData(input.getData(input.getWidth() - 1, input.getHeight() - 1, c), x, y, c);
                }
            }
        }
        break;
    case Padding_Type::REFLECTION:
#pragma omp parallel for schedule(dynamic)
        for (int c = 0; c < input.getChannels(); c++)
        {
#pragma omp parallel for schedule(dynamic)
            for (int y = 0; y < pad; y++)
            {
#pragma omp parallel for schedule(dynamic)
                for (int x = 0; x < pad; x++)
                {
                    output.setData(input.getData(pad - x - 1, pad - y - 1, c), x, y, c);
                }
#pragma omp parallel for schedule(dynamic)
                for (int x = pad; x < input.getWidth() + pad; x++)
                {
                    output.setData(input.getData(x - pad, pad - y - 1, c), x, y, c);
                }
#pragma omp parallel for schedule(dynamic)
                for (int x = input.getWidth() + pad; x < input.getWidth() + 2 * pad; x++)
                {
                    output.setData(input.getData(input.getWidth() * 2 - x + pad - 1, pad - y - 1, c), x, y, c);
                }
            }
#pragma omp parallel for schedule(dynamic)
            for (int y = pad; y < input.getHeight() + pad; y++)
            {
#pragma omp parallel for schedule(dynamic)
                for (int x = c; x < pad; x++)
                {
                    output.setData(input.getData(pad - x - 1, y - pad, c), x, y, c);
                }
#pragma omp parallel for schedule(dynamic)
                for (int x = pad; x < input.getWidth() + pad; x++)
                {
                    output.setData(input.getData(x - pad, y - pad, c), x, y, c);
                }
#pragma omp parallel for schedule(dynamic)
                for (int x = input.getWidth() + pad; x < input.getWidth() + 2 * pad; x++)
                {
                    output.setData(input.getData(input.getWidth() * 2 - x + pad - 1, y - pad, c), x, y, c);
                }
            }
#pragma omp parallel for schedule(dynamic)
            for (int y = input.getHeight() + pad; y < input.getHeight() + 2 * pad; y++)
            {
#pragma omp parallel for schedule(dynamic)
                for (int x = c; x < pad; x++)
                {
                    output.setData(input.getData(pad - x - 1, input.getHeight() * 2 - y + pad - 1, c), x, y, c);
                }
#pragma omp parallel for schedule(dynamic)
                for (int x = pad; x < input.getWidth() + pad; x++)
                {
                    output.setData(input.getData(x - pad, input.getHeight() * 2 - y + pad - 1, c), x, y, c);
                }
#pragma omp parallel for schedule(dynamic)
                for (int x = input.getWidth() + pad; x < input.getWidth() + 2 * pad; x++)
                {
                    output.setData(input.getData(input.getWidth() * 2 - x + pad - 1, input.getHeight() * 2 - y + pad - 1, c), x, y, c);
                }
            }
        }
        break;
    }
}

void PaddingLayer::forward(const Buffer *input, Buffer *output)
{
    Timer timer("Padding");
    output->resize(input->getWidth() + 2 * pad, input->getHeight() + 2 * pad, input->getChannels());
    switch (paddingType)
    {
    case Padding_Type::ZERO_PADDING:
        fill(output->getVectorPtr(), output->getVectorPtr() + output->getSize(), 0);
#pragma omp parallel for collapse(3) schedule(dynamic)
        for (int c = 0; c < input->getChannels(); c++)
        {
            for (int y = pad; y < input->getHeight() + pad; y++)
            {
                for (int x = pad; x < input->getWidth() + pad; x++)
                {
                    output->setData(input->getData(x - pad, y - pad, c), x, y, c);
                }
            }
        }
        break;
    case Padding_Type::EXTEND_EDGE:
#pragma omp parallel for schedule(dynamic)
        for (int c = 0; c < input->getChannels(); c++)
        {
#pragma omp parallel for schedule(dynamic)
            for (int y = 0; y < pad; y++)
            {
#pragma omp parallel for schedule(dynamic)
                for (int x = 0; x < pad; x++)
                {
                    output->setData(input->getData(0, 0, c), x, y, c);
                }
#pragma omp parallel for schedule(dynamic)
                for (int x = pad; x < input->getWidth() + pad; x++)
                {
                    output->setData(input->getData(x - pad, 0, c), x, y, c);
                }
#pragma omp parallel for schedule(dynamic)
                for (int x = input->getWidth() + pad; x < input->getWidth() + 2 * pad; x++)
                {
                    output->setData(input->getData(input->getWidth() - 1, 0, c), x, y, c);
                }
            }
#pragma omp parallel for schedule(dynamic)
            for (int y = pad; y < input->getHeight() + pad; y++)
            {
                int x = 0;
#pragma omp parallel for schedule(dynamic)
                for (int x = 0; x < pad; x++)
                {
                    output->setData(input->getData(0, y - pad, c), x, y, c);
                }
#pragma omp parallel for schedule(dynamic)
                for (int x = pad; x < input->getWidth() + pad; x++)
                {
                    output->setData(input->getData(x - pad, y - pad, c), x, y, c);
                }
#pragma omp parallel for schedule(dynamic)
                for (int x = input->getWidth() + pad; x < input->getWidth() + 2 * pad; x++)
                {
                    output->setData(input->getData(input->getWidth() - 1, y - pad, c), x, y, c);
                }
            }
#pragma omp parallel for schedule(dynamic)
            for (int y = input->getHeight() + pad; y < input->getHeight() + 2 * pad; y++)
            {
#pragma omp parallel for schedule(dynamic)
                for (int x = 0; x < pad; x++)
                {
                    output->setData(input->getData(0, input->getHeight() - 1, c), x, y, c);
                }
#pragma omp parallel for schedule(dynamic)
                for (int x = pad; x < input->getWidth() + pad; x++)
                {
                    output->setData(input->getData(x - pad, input->getHeight() - 1, c), x, y, c);
                }
#pragma omp parallel for schedule(dynamic)
                for (int x = input->getWidth() + pad; x < input->getWidth() + 2 * pad; x++)
                {
                    output->setData(input->getData(input->getWidth() - 1, input->getHeight() - 1, c), x, y, c);
                }
            }
        }
        break;
    case Padding_Type::REFLECTION:
#pragma omp parallel for schedule(dynamic)
        for (int c = 0; c < input->getChannels(); c++)
        {
#pragma omp parallel for schedule(dynamic)
            for (int y = 0; y < pad; y++)
            {
#pragma omp parallel for schedule(dynamic)
                for (int x = 0; x < pad; x++)
                {
                    output->setData(input->getData(pad - x - 1, pad - y - 1, c), x, y, c);
                }
#pragma omp parallel for schedule(dynamic)
                for (int x = pad; x < input->getWidth() + pad; x++)
                {
                    output->setData(input->getData(x - pad, pad - y - 1, c), x, y, c);
                }
#pragma omp parallel for schedule(dynamic)
                for (int x = input->getWidth() + pad; x < input->getWidth() + 2 * pad; x++)
                {
                    output->setData(input->getData(input->getWidth() * 2 - x + pad - 1, pad - y - 1, c), x, y, c);
                }
            }
#pragma omp parallel for schedule(dynamic)
            for (int y = pad; y < input->getHeight() + pad; y++)
            {
#pragma omp parallel for schedule(dynamic)
                for (int x = c; x < pad; x++)
                {
                    output->setData(input->getData(pad - x - 1, y - pad, c), x, y, c);
                }
#pragma omp parallel for schedule(dynamic)
                for (int x = pad; x < input->getWidth() + pad; x++)
                {
                    output->setData(input->getData(x - pad, y - pad, c), x, y, c);
                }
#pragma omp parallel for schedule(dynamic)
                for (int x = input->getWidth() + pad; x < input->getWidth() + 2 * pad; x++)
                {
                    output->setData(input->getData(input->getWidth() * 2 - x + pad - 1, y - pad, c), x, y, c);
                }
            }
#pragma omp parallel for schedule(dynamic)
            for (int y = input->getHeight() + pad; y < input->getHeight() + 2 * pad; y++)
            {
#pragma omp parallel for schedule(dynamic)
                for (int x = c; x < pad; x++)
                {
                    output->setData(input->getData(pad - x - 1, input->getHeight() * 2 - y + pad - 1, c), x, y, c);
                }
#pragma omp parallel for schedule(dynamic)
                for (int x = pad; x < input->getWidth() + pad; x++)
                {
                    output->setData(input->getData(x - pad, input->getHeight() * 2 - y + pad - 1, c), x, y, c);
                }
#pragma omp parallel for schedule(dynamic)
                for (int x = input->getWidth() + pad; x < input->getWidth() + 2 * pad; x++)
                {
                    output->setData(input->getData(input->getWidth() * 2 - x + pad - 1, input->getHeight() * 2 - y + pad - 1, c), x, y, c);
                }
            }
        }
        break;
    }
}
