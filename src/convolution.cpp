#include <memory>
#include <cmath>
#include <omp.h>

#include "convolution.h"
#include "timer.h"

Convolution::Convolution() {};

Buffer Convolution::detectEdges(const Buffer& buffer)
{
    Timer timer("detectEdge");
    Buffer result(buffer.getWidth()-2, buffer.getHeight()-2, buffer.getChannels());
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int y=0;y<result.getHeight();y++)
    {
        for (int x=0;x<result.getWidth();x++)
        {
            float rVertical = 
            buffer.getData(x,y  ,0)*(-1) + buffer.getData(x+2,y  ,0)*1 +
            buffer.getData(x,y+1,0)*(-2) + buffer.getData(x+2,y+1,0)*2 + 
            buffer.getData(x,y+2,0)*(-1) + buffer.getData(x+2,y+2,0)*1 ;

            float rHorizontal = 
            buffer.getData(x,y  ,0)*(-1) + buffer.getData(x+1,y  ,0)*(-2) + buffer.getData(x+2,y,  0)*(-1) +
            buffer.getData(x,y+2,0)*1    + buffer.getData(x+1,y+2,0)*2    + buffer.getData(x+2,y+2,0)*1    ;

            float rMagnitude = sqrt(pow(rVertical,2) + pow(rHorizontal, 2));

            result.setData(ReLU(rMagnitude), x, y, 0);

            float gVertical = 
            buffer.getData(x,y  ,1)*(-1) + buffer.getData(x+2,y  ,1)*1 +
            buffer.getData(x,y+1,1)*(-2) + buffer.getData(x+2,y+1,1)*2 + 
            buffer.getData(x,y+2,1)*(-1) + buffer.getData(x+2,y+2,1)*1 ;

            float gHorizontal = 
            buffer.getData(x,y  ,1)*(-1) + buffer.getData(x+1,y  ,1)*(-2) + buffer.getData(x+2,y,  1)*(-1) +
            buffer.getData(x,y+2,1)*1    + buffer.getData(x+1,y+2,1)*2    + buffer.getData(x+2,y+2,1)*1    ;

            float gMagnitude = sqrt(pow(gVertical,2) + pow(gHorizontal, 2));

            result.setData(ReLU(gMagnitude), x, y, 1);

            float bVertical = 
            buffer.getData(x,y  ,2)*(-1) + buffer.getData(x+2,y  ,2)*1 +
            buffer.getData(x,y+1,2)*(-2) + buffer.getData(x+2,y+1,2)*2 + 
            buffer.getData(x,y+2,2)*(-1) + buffer.getData(x+2,y+2,2)*1 ;

            float bHorizontal = 
            buffer.getData(x,y  ,2)*(-1) + buffer.getData(x+1,y  ,2)*(-2) + buffer.getData(x+2,y,  2)*(-1) +
            buffer.getData(x,y+2,2)*1    + buffer.getData(x+1,y+2,2)*2    + buffer.getData(x+2,y+2,2)*1    ;

            float bMagnitude = sqrt(pow(bVertical,2) + pow(bHorizontal, 2));

            result.setData(ReLU(bMagnitude), x, y, 2);
        }
    }

    return result;
}