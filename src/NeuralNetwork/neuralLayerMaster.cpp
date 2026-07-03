#include "neuralLayerMaster.h"

#include <iostream>
#include <utility>

NeuralLayerMaster::NeuralLayerMaster(std::vector<TransformationLayer> transformationLayers)
    : transformationLayers(std::move(transformationLayers))
{
    neuronLayers.push_back(NeuronLayer(transformationLayers[0].getInputN()));
    for (int i = 0; i < transformationLayers.size(); i++)
    {
        neuronLayers.push_back(NeuronLayer(transformationLayers[i].getOutputN()));
    }
}
NeuralLayerMaster::~NeuralLayerMaster() {}

void NeuralLayerMaster::initializeNeuronLayer(NeuronLayer initialNeuronLayer) 
{
    neuronLayers[0] = initialNeuronLayer;
}

void NeuralLayerMaster::start()
{
    int i = 0;
    for (; i < transformationLayers.size() - 1; i++)
    {
        transformationLayers[i].forward(&neuronLayers[i], &neuronLayers[i + 1]);
        neuronLayers[i + 1].map(reLU);
    }
    transformationLayers[i].forward(&neuronLayers[i], &neuronLayers[i + 1]);
    neuronLayers[i + 1].map(sigmoid);
}

NeuronLayer NeuralLayerMaster::getResult()
{
    return neuronLayers[neuronLayers.size()-1];
}