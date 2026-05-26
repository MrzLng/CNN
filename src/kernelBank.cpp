#include "kernelBank.h"

#include <random>
#include <cmath>
#include <fstream>
#include <iostream>

KernelBank::KernelBank(int width, int height, int channels, int count)
    : width(width), height(height), channels(channels), count(count)
{
    std::vector<float> data(width*height*(this->channels != 0 ? this->channels : 1));
    std::random_device seed;
    std::mt19937 gen(seed());
    int n = width*height*channels;
    normal_distribution<float> distribution(0.0f, sqrt(2.0f/static_cast<float>(n)));
    for (int c=0;c<count;c++)
    {
        for (int i=0;i<data.size();i++)
        {
            data[i] = distribution(gen);
        }
        kernels.push_back({data, width, height, channels});
    }
}

KernelBank::KernelBank(string path)
    : count(0)
{
    std::ifstream file(path, std::ios::binary);

    if (!file.is_open())
    {
        std::cerr << "could not open the file" << std::endl;
        return;
    }

    file.read(reinterpret_cast<char *>(&this->width), sizeof(int));
    file.read(reinterpret_cast<char *>(&this->height), sizeof(int));
    file.read(reinterpret_cast<char *>(&this->channels), sizeof(int));
    file.read(reinterpret_cast<char *>(&this->count), sizeof(int));
    std::vector<float> data(width*height*(this->channels != 0 ? this->channels : 1));
    for (int i = 0; i < count; i++)
    {
        file.read(reinterpret_cast<char *>(data.data()), sizeof(float) * this->width * this->height * (this->channels != 0 ? this->channels : 1));
        kernels.push_back({data, width, height, channels});
    }
}

void KernelBank::write(string path)
{
    std::ofstream file(path, std::ios::binary);

    if (!file)
    {
        std::cerr << "could not open the file" << std::endl;
        return;
    }

    file.write(reinterpret_cast<char *>(&this->width), sizeof(int));
    file.write(reinterpret_cast<char *>(&this->height), sizeof(int));
    file.write(reinterpret_cast<char *>(&this->channels), sizeof(int));
    file.write(reinterpret_cast<char *>(&this->count), sizeof(int));
    for (int i = 0; i < count; i++)
    {
        file.write(reinterpret_cast<char *>(kernels[i].getVector()), sizeof(float) * this->width * this->height * (this->channels != 0 ? this->channels : 1));
    }
}

void KernelBank::print()
{
    for (int c=0;c<count;c++)
    {
        for (int y=0; y<height;y++)
        {
            for (int x=0;x<width;x++)
            {
                for (int ch = 0;ch<channels;ch++)
                {
                    cout << kernels[c].getData(x,y,ch) <<" ";
                }
            }
            cout << endl;
        }
        cout << endl;
    }
}