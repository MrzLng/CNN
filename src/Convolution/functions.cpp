#include "functions.h"
#include "timer.h"

#include <iostream>
#include <cmath>

using namespace std;

Buffer Functions::mergeVH(Buffer* bufferV, Buffer* bufferH)
{
    Timer timer("mergeVH");
    if (bufferV->getWidth() != bufferH->getWidth()) {cerr << "Buffer width does not match" << endl;}
    if (bufferV->getHeight() != bufferH->getHeight()) {cerr << "Buffer height does not match" << endl;}
    if (bufferV->getChannels() != bufferH->getChannels()) {cerr << "Buffer channels does not match" << endl;}

    Buffer result(*bufferV);

    for (int y=0; y<bufferV->getHeight(); y++)
    {
        for (int x=0; x<bufferV->getWidth(); x++)
        {
            for (int c=0; c<bufferV->getChannels(); c++)
            {
                result.setData(sqrt(pow(bufferV->getData(x,y,c),2)+pow(bufferH->getData(x,y,c),2)), x, y, c);
            }
        }
    }
    return result;
}