#pragma once

#include <string>

#include "convolutionLayerMaster.h"
#include "neuralLayerMaster.h"

using namespace std;

class NeuralTrainer
{
public:
NeuralTrainer(ConvolutionLayerMaster convolutionLayerMaster, NeuralLayerMaster neuralLayerMaster, string folderPath);
NeuralTrainer(string layerWeightsPath,string folderPath);
~NeuralTrainer();

void storeWeights();

private:
ConvolutionLayerMaster convolutionLayerMaster;
NeuralLayerMaster neuralLayerMaster;
string folderPath;
};