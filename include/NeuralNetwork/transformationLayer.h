#include "matrix.h"
#include "neuronLayer.h"

#include <string>

class TransformationLayer
{
public:
    TransformationLayer(int inputN, int outputN);
    TransformationLayer(string path);

    void forward(NeuronLayer *source, NeuronLayer *destination);
    void write(string path);

    inline int getInputN() { return inputN; }
    inline int getOutputN() { return outputN; }

private:
    Matrix weights;
    Matrix biases;
    int inputN;
    int outputN;
};