#include <algorithm>
#include <iostream>
#include <omp.h>

#include "buffer.h"
#include "timer.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

Buffer::Buffer(int width, int height, int channels) : width(width), height(height), channels(channels), data(width * height * channels), size(width * height * channels) {}
Buffer::Buffer(vector<float> data, int width, int height, int channels) : width(width), height(height), channels(channels), data(data), size(width * height * channels) {}
Buffer::Buffer(int value, int width, int height, int channels) : width(width), height(height), channels(channels), data(width * height * channels, value), size(width * height * channels) {}

Buffer::Buffer(const Image &image) : width(image.getWidth()), height(image.getHeight()), channels(3), data(image.getWidth() * image.getHeight() * 3), size(image.getWidth() * image.getHeight() * 3)
{
#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            setData(static_cast<float>(image.getData(x, y, 0)) / 255.0f, x, y, 0);
            setData(static_cast<float>(image.getData(x, y, 1)) / 255.0f, x, y, 1);
            setData(static_cast<float>(image.getData(x, y, 2)) / 255.0f, x, y, 2);
        }
    }
}

void Buffer::toImage(string path)
{
    Timer timer("toImage");
    unsigned char *finalImage = new unsigned char[width * height * 3];
#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            finalImage[y * width * 3 + x * 3 + 0] = static_cast<unsigned char>(clamp(getData(x, y, 0) * 255.0f, 0.0f, 255.0f));
            finalImage[y * width * 3 + x * 3 + 1] = static_cast<unsigned char>(clamp(getData(x, y, 1) * 255.0f, 0.0f, 255.0f));
            finalImage[y * width * 3 + x * 3 + 2] = static_cast<unsigned char>(clamp(getData(x, y, 2) * 255.0f, 0.0f, 255.0f));
        }
    }

    stbi_write_png(path.c_str(), width, height, 3, finalImage, width * 3);
    delete[] finalImage;
}

void Buffer::resize(int width, int height, int channels)
{
    this->width = width;
    this->height = height;
    this->channels = channels;
    this->size = width * height * channels;
    if (this->size > data.size())
    {
        data.resize(this->size);
    }
}

void Buffer::print()
{
    for (int c = 0; c < channels; c++)
    {
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                cout << this->getData(x, y, c) << ' ';
            }
            cout << endl;
        }
        cout << endl;
    }
}