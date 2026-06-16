#include "neuronLayer.h"

NeuronLayer::NeuronLayer(Buffer buffer)
    : matrix(buffer.getVector(), buffer.getSize(), 1), neuronCount(buffer.getSize()) {}

NeuronLayer::NeuronLayer(int neuronCount)
: matrix(0, neuronCount, 1), neuronCount(neuronCount) {}