#pragma once

#include <vector>

#include "image.h"

using namespace std;

class Buffer
{
    public:
    enum Channel
    {
        RED=0, GREEN=1, BLUE=2
    };

    public:
    Buffer(int width, int height, int channels);
    Buffer(vector<float> data, int width, int height, int channels);
    Buffer(int value, int width, int height, int channels);
    Buffer(const Image& image);

    inline void setData(float value, int index) {
        #ifdef DEBUG
        if (index >= width*height or index<0) {cerr << "[Buffer Indexing Error]: Index out of range";}
        #endif
        data[index] = value;
    }
    inline void setData(float value, int x, int y, int channel) {
        #ifdef DEBUG
        if (x >= width or x<0 or y>= height or y<0) {cerr << "[Buffer Indexing Error]: Index out of range";}
        #endif
        data[channel*width*height+y*width+x] = value; 
    }
    inline float getData(int index) const {
        #ifdef DEBUG
        if (index >= width*height or index<0) {cerr << "[Buffer Indexing Error]: Index out of range";}
        #endif
        return data[index];
    }
    inline float getData(int x, int y, int channel) const 
    {
        #ifdef DEBUG
        if (x >= width or x<0 or y>= height or y<0) {cerr << "[Buffer Indexing Error]: Index out of range";}
        #endif
        return data[channel*width*height+y*width+x]; 
    }

    inline int getWidth() const {return width;}
    inline int getHeight() const {return height;}
    inline int getChannels() const {return channels;}
    inline int getSize() const {return size;}
    inline vector<float> getVector() {return data;}
    inline float* getVectorPtr() {return data.data();}

    void toImage(string folderPath);
    void resize(int width, int height, int channels);

    void print();

    private:
    vector<float> data;
    int width;
    int height;
    int channels;
    int size;
};