#include "includes.h"

#include <omp.h>
#include <iostream>
#include <vector>

Kernel vertical({-1.0f, 0.0f, 1.0f, -2.0f, 0.0f, 2.0f, -1.0f, 0.0f, 1.0f}, 3, 3, 0);
Kernel horizontal({-1.0f, -2.0f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 1.0f}, 3, 3, 0);

float larger(float a, float b)
{
    return a > b ? a : b;
}

int main()
{
    Timer timer("main");
    Image image("resources/gaston-abascal-8F-6pXyQyLU-unsplash.jpg");
    Buffer buffer(image, 960, 540, 3);

    buffer.toImage("outputs/test");

    KernelBank kBank("bin/kBank.bin");
    // KernelBank kBank(3, 3, 3, 16);

    std::vector<std::unique_ptr<Layer>> layers;
    layers.push_back(std::make_unique<PaddingLayer>(PaddingLayer::Padding_Type::ZERO_PADDING, 1));
    layers.push_back(std::make_unique<ConvolutionLayer>(kBank));
    layers.push_back(std::make_unique<PoolingLayer>(larger, 2, 2));
    layers.push_back(std::make_unique<ReluLayer>());

    ConvolutionLayerMaster convolutionMaster(std::move(layers));
    convolutionMaster.initializeBuffer(buffer);
    convolutionMaster.start();

    kBank.write("bin/kBank.bin");
    convolutionMaster.getBufferPtr()->toImage("outputs");
}