#pragma once

#include <vector>

#include "transformationLayer.h"
#include "neuronLayer.h"

class NeuralLayerMaster
{
    public:
    NeuralLayerMaster(NeuronLayer initialNeuronLayer, std::vector<TransformationLayer> transformationLayers);
    ~NeuralLayerMaster();

    void start();

    private:
    std::vector<TransformationLayer> transformationLayers;
    std::vector<NeuronLayer> neuronLayers;
};