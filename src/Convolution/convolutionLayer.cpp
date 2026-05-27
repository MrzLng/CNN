#include <memory>
#include <cmath>
#include <omp.h>
#include <iostream>

#include "convolutionLayer.h"
#include "timer.h"

void ConvolutionLayer::forward(const Buffer& input, Buffer& output) // vector kernel dimensions must be same
{
    if (kernels.getChannels() != 0 && kernels.getChannels() != input.getChannels())
    {
        std::cerr << "Kernel size not match" << endl;
        return;
    };

    Timer timer("Convolution");
    output.resize(input.getWidth() - kernels.getWidth() + 1, input.getHeight() - kernels.getHeight() + 1, kernels.getCount());
#pragma omp parallel for collapse(3) schedule(dynamic)
    for (int outputc = 0; outputc < kernels.getCount(); outputc++)
        for (int y = 0; y < output.getHeight(); y++)
        {
            for (int x = 0; x < output.getWidth(); x++)
            {
                float sum = 0;
                for (int c = 0; c < input.getChannels(); c++)
                {
                    for (int kernelY = 0; kernelY < kernels.getHeight(); kernelY++)
                    {
                        for (int kernelX = 0; kernelX < kernels.getWidth(); kernelX++)
                        {
                            sum += kernels.getKernel(outputc).getData(kernelX, kernelY, c) * input.getData(x + kernelX, y + kernelY, c);
                        }
                    }
                }
                output.setData(sum, x, y, outputc);
            }
        }
}

void ConvolutionLayer::forward(const Buffer *input, Buffer *output) // vector kernel dimensions must be same
{
    if (kernels.getChannels() != 0 && kernels.getChannels() != input->getChannels())
    {
        std::cerr << "Kernel size not match" << endl;
        return;
    };

    Timer timer("Convolution");
    output->resize(input->getWidth() - kernels.getWidth() + 1, input->getHeight() - kernels.getHeight() + 1, kernels.getCount());
#pragma omp parallel for collapse(3) schedule(dynamic)
    for (int outputc = 0; outputc < kernels.getCount(); outputc++)
        for (int y = 0; y < output->getHeight(); y++)
        {
            for (int x = 0; x < output->getWidth(); x++)
            {
                float sum = 0;
                for (int c = 0; c < input->getChannels(); c++)
                {
                    for (int kernelY = 0; kernelY < kernels.getHeight(); kernelY++)
                    {
                        for (int kernelX = 0; kernelX < kernels.getWidth(); kernelX++)
                        {
                            sum += kernels.getKernel(outputc).getData(kernelX, kernelY, c) * input->getData(x + kernelX, y + kernelY, c);
                        }
                    }
                }
                output->setData(sum, x, y, outputc);
            }
        }
}