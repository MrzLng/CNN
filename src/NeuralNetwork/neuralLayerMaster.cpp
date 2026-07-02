#include "neuralLayerMaster.h"

#include <iostream>
#include <utility>

NeuralLayerMaster::NeuralLayerMaster(NeuronLayer initialNeuronLayer, std::vector<TransformationLayer> transformationLayers)
    : transformationLayers(std::move(transformationLayers))
{
    if (transformationLayers.empty())
    {
        std::cerr << "Transformation layers empty" << std::endl;
    }
    if (transformationLayers[0].getInputN() != initialNeuronLayer.getNueronCount())
    {
        std::cerr << "First transformation layer input number doesn't match inital neuron layer" << std::endl;
        return;
    }
    neuronLayers.push_back(std::move(initialNeuronLayer));
    for (int i = 0; i < transformationLayers.size(); i++)
    {
        neuronLayers.push_back(NeuronLayer(transformationLayers[i].getOutputN()));
    }
}
NeuralLayerMaster::~NeuralLayerMaster() {}

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