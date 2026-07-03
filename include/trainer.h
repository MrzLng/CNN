#pragma once

#include <string>

#include "convolutionLayerMaster.h"
#include "neuralLayerMaster.h"

using namespace std;

class Trainer
{
public:
Trainer(ConvolutionLayerMaster convolutionLayerMaster, NeuralLayerMaster neuralLayerMaster);
Trainer(string layerWeightsPath);
~Trainer();

void train(string folderPath, int epochs);
void validate(string folderPath);
void storeWeights(string layerWeightsPath);

private:
ConvolutionLayerMaster convolutionLayerMaster;
NeuralLayerMaster neuralLayerMaster;
};