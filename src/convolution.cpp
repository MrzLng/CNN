#include <memory>
#include <cmath>
#include <omp.h>

#include "convolutionLayer.h"
#include "timer.h"

Buffer ConvolutionLayer::forward(const Buffer& input, Buffer& output)
{
    Timer timer("kernelApply");
    Buffer result(input.getWidth()-kernel.getWidth()/2, input.getHeight()-kernel.getHeight()/2, input.getChannels());
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int y=0;y<result.getHeight();y++)
    {
        for (int x=0;x<result.getWidth();x++)
        {
            float sumR=0.0f;
            float sumG=0.0f;
            float sumB=0.0f;
            for (int kernelY=0;kernelY<kernel.getHeight();kernelY++)
            {
                for (int kernelX=0;kernelX<kernel.getWidth();kernelX++)
                {
                    sumR += kernel.getData(kernelX, kernelY, 0) * input.getData(x+kernelX, y+kernelY, 0);
                    sumG += kernel.getData(kernelX, kernelY, 1) * input.getData(x+kernelX, y+kernelY, 1);
                    sumB += kernel.getData(kernelX, kernelY, 2) * input.getData(x+kernelX, y+kernelY, 2);
                }
            }
            output.setData(ReLU(sumR), x, y, 0);
            output.setData(ReLU(sumG), x, y, 1);
            output.setData(ReLU(sumB), x, y, 2);
        }
    }

    return result;
}