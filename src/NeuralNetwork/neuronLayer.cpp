#include "neuronLayer.h"

NeuronLayer::NeuronLayer(Buffer buffer)
    : matrix(buffer.getVector(), buffer.getSize(), 1) {}