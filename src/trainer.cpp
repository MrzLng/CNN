#include "trainer.h"

#include <vector>

using namespace std;

Trainer::Trainer(ConvolutionLayerMaster convolutionLayerMaster, NeuralLayerMaster neuralLayerMaster)
    : convolutionLayerMaster(convolutionLayerMaster), neuralLayerMaster(neuralLayerMaster) {}
Trainer::Trainer(string layerWeightsPath)
{
}
Trainer::~Trainer()
{
}

void Trainer::train(string folderPath, int epochs)
{
    vector<Buffer> bufferStorage;
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        for (int i = 0; i < bufferStorage.size(); i++)
        {
            convolutionLayerMaster.initializeBuffer(bufferStorage[i]);
            convolutionLayerMaster.start();
            NeuronLayer neuronLayer(*convolutionLayerMaster.getBufferPtr());
            neuralLayerMaster.initializeNeuronLayer(neuronLayer);
            neuralLayerMaster.start();

            neuralLayerMaster.getResult().getMatrix();
        }
    }
}
void Trainer::validate(string folderPath)
{
}
void Trainer::storeWeights(string layerWeightsPath)
{
}