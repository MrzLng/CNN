#pragma once

#include <vector>
#include <cmath>

#include "transformationLayer.h"
#include "neuronLayer.h"

class NeuralLayerMaster
{
public:
    NeuralLayerMaster(std::vector<TransformationLayer> transformationLayers);
    ~NeuralLayerMaster();

    void initializeNeuronLayer(NeuronLayer initialNeuronLayer);
    void start();
    NeuronLayer getResult();

private:
    std::vector<TransformationLayer> transformationLayers;
    std::vector<NeuronLayer> neuronLayers;

    static inline float reLU(float x) { return x ? x : 0.0f; };
    static inline float sigmoid(float x) {return 1/(1+std::exp(-x));}
};