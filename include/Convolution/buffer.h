#pragma once

#include <vector>

#include "image.h"

using namespace std;

class Buffer
{
public:
    enum Channel
    {
        RED = 0,
        GREEN = 1,
        BLUE = 2
    };

public:
    Buffer(int width, int height, int channels);
    Buffer(vector<float> data, int width, int height, int channels);
    Buffer(int value, int width, int height, int channels);
    Buffer(const Image &image);

    Buffer(const Image &image, int width, int height, int channels);

    inline void setData(float value, int index)
    {
        data[index] = value;
    }
    inline void setData(float value, int x, int y, int channel)
    {
        data[channel * width * height + y * width + x] = value;
    }
    inline float getData(int index) const
    {
        return index >= 0 && index < size ? data[index] : 0.0f;
    }
    inline float getData(int x, int y, int channel) const
    {
        return x >= 0 && x < width && y >= 0 && y < height && channel >= 0 && channel < channels ? data[channel * width * height + y * width + x] : 0.0f;
    }

    inline int getWidth() const { return width; }
    inline int getHeight() const { return height; }
    inline int getChannels() const { return channels; }
    inline int getSize() const { return size; }
    inline vector<float> getVector() { return data; }
    inline float *getVectorPtr() { return data.data(); }

    void toImage(string folderPath);
    void resize(int width, int height, int channels);
    Buffer slice(int leftX, int upY, int sliceWidth, int sliceHeight);

    void print();

private:
    vector<float> data;
    int width;
    int height;
    int channels;
    int size;
};