#include "transformationLayer.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

TransformationLayer::TransformationLayer(int inputN, int outputN)
    : weights(Matrix::RandomType::normal, 0, sqrt(2.0f / static_cast<float>(inputN)), outputN, inputN), biases(0, outputN, 1), inputN(inputN), outputN(outputN) {}

TransformationLayer::TransformationLayer(string path)
{
    std::ifstream file(path);

    if (file.is_open())
    {
        std::cerr << "unable to open file" << std::endl;
        return;
    }

    file.read(reinterpret_cast<char *>(&this->outputN), sizeof(int));
    file.read(reinterpret_cast<char *>(&this->inputN), sizeof(int));

    std::vector<float> w(outputN * inputN);
    file.read(reinterpret_cast<char *>(w.data()), sizeof(float) * this->outputN * this->inputN);
    weights = Matrix(w, outputN, inputN);

    std::vector<float> b(outputN, 1);
    file.read(reinterpret_cast<char *>(b.data()), sizeof(float) * this->outputN * this->inputN);
    biases = Matrix(b, outputN, inputN);

    file.close();
}

void TransformationLayer::forward(NeuronLayer* source, NeuronLayer* destination)
{
    if (inputN != source->getNueronCount() || outputN != destination->getNueronCount())
    {
        std::cerr << "invalid source or destination number" << endl;
        return;
    }
    destination->setMatrix(weights * source->getMatrix() + biases);
}

void TransformationLayer::write(string path)
{
    std::ofstream file(path);

    if(!file)
    {
        std::cerr << "unable to write to destination" << endl;
    }

    file.write(reinterpret_cast<char *>(&this->outputN), sizeof(int));
    file.write(reinterpret_cast<char *>(&this->inputN), sizeof(int));
    file.write(reinterpret_cast<char *>(this->weights.getVectorPtr()), sizeof(float) * this->outputN * this->inputN);
    file.write(reinterpret_cast<char *>(this->biases.getVectorPtr()), sizeof(float) * this->outputN * this->inputN);

    file.close();
}