#pragma once

#include <vector>

using namespace std;

class Kernel
{
    public:
    Kernel(vector<float> data, int width, int height, int channels):data(data), width(width), height(height), channels(channels) {} // put channels=0 if one kernel is applied to all channels
    inline float getData(int x, int y, int channel) const {return data[min(channel,channels)*height*width+y*width+x]; }
    inline int getWidth() const {return width;}
    inline int getHeight() const {return height;}
    inline int getChannels() const {return channels;}

    private:
    vector<float> data;
    int width;
    int height;
    int channels;
};