#pragma once

#include "buffer.h"
#include "matrix.h"

#include <functional>

class NeuronLayer
{
    public:
    NeuronLayer(Buffer buffer);
    NeuronLayer(int neuronCount);

    inline int getNueronCount() const {return neuronCount;}
    inline Matrix getMatrix() const {return matrix;}
    inline void setMatrix(Matrix other) {matrix = other;}
    inline void map(std::function<float(float)> function) {matrix.map(function);}

    private:
    Matrix matrix;
    int neuronCount;
};