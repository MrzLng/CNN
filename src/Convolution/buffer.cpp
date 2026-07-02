#include <algorithm>
#include <iostream>
#include <omp.h>
#include <random>

#include "buffer.h"
#include "timer.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

Buffer::Buffer(int width, int height, int channels) : width(width), height(height), channels(channels), data(width * height * channels), size(width * height * channels)
{
    Timer timer("Buffer");
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    for (int i = 0; i < size; i++)
    {
        data[i] = dis(gen);
    }
}
Buffer::Buffer(vector<float> data, int width, int height, int channels) : width(width), height(height), channels(channels), data(data), size(width * height * channels) {}
Buffer::Buffer(int value, int width, int height, int channels) : width(width), height(height), channels(channels), data(width * height * channels, value), size(width * height * channels) {}

Buffer::Buffer(const Image &image) : width(image.getWidth()), height(image.getHeight()), channels(3), data(image.getWidth() * image.getHeight() * 3), size(image.getWidth() * image.getHeight() * 3)
{
    Timer timer("Buffer");
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

Buffer::Buffer(const Image &image, int width, int height, int channels)
    : width(width), height(height), channels(channels), data(width * height * channels), size(width * height * channels)
{
    float scaling_factor = min(min(static_cast<float>(width) / static_cast<float>(image.getWidth()), static_cast<float>(height) / static_cast<float>(image.getHeight())), 1.0f);
    // factor <= 1
    for (int c = 0; c < channels; c++)
    {
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                float realXOnImage = (static_cast<float>(x) - static_cast<float>(width) / 2.0f) / scaling_factor + static_cast<float>(image.getWidth()) / 2.0f;
                float realYOnImage = (static_cast<float>(y) - static_cast<float>(height) / 2.0f) / scaling_factor + static_cast<float>(image.getHeight()) / 2.0f;
                int roundDownX = static_cast<int>(realXOnImage);
                int roundDownY = static_cast<int>(realYOnImage);
                float Xweights = realXOnImage - roundDownX;
                float Yweights = realYOnImage - roundDownY;

                data[c * height * width + y * width + x] = static_cast<float>(image.getData(roundDownX, roundDownY, c)) * (1 - Xweights) * (1 - Yweights) +
                                                           static_cast<float>(image.getData(roundDownX + 1, roundDownY, c)) * (Xweights) * (1 - Yweights) +
                                                           static_cast<float>(image.getData(roundDownX, roundDownY + 1, c)) * (1 - Xweights) * (Yweights) +
                                                           static_cast<float>(image.getData(roundDownX + 1, roundDownY + 1, c)) * (Xweights) * (Yweights);
            }
        }
    }
}

void Buffer::toImage(string folderPath)
{
    Timer timer("toImage");

#pragma omp parallel for schedule(dynamic)
    for (int c = 0; c < channels; c++)
    {
        unsigned char *finalImage = new unsigned char[width * height];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                finalImage[y * width + x] = static_cast<unsigned char>(clamp(getData(x, y, c) * 255.0f, 0.0f, 255.0f));
            }
        }
        stbi_write_png((folderPath + "/" + to_string(c) + ".png").c_str(), width, height, 1, finalImage, width);
        delete[] finalImage;
    }
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